from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk, ImageChops, ImageDraw, ImageOps

from . import config as C
from .pipeline_core import run_inference


# ---------------- Utils ----------------
def load_pil_with_exif(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im = ImageOps.exif_transpose(im)
    return im


def pil_to_fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    im2 = im.copy()
    im2.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return im2


def apply_mask_rgb_pil(img_rgb: Image.Image, mask_L: Image.Image) -> Image.Image:
    """skin-only: mask外を黒に"""
    img = np.array(img_rgb.convert("RGB"), dtype=np.uint8)
    m = np.array(mask_L.convert("L"), dtype=np.uint8)
    keep = (m > 0)
    out = img.copy()
    out[~keep] = 0
    return Image.fromarray(out, mode="RGB")


# ---------- silhouette weights / cover% ----------
def load_weights(weights_json: Path) -> Tuple[Dict[int, float], Dict[int, float]]:
    obj = json.loads(weights_json.read_text(encoding="utf-8"))
    front = {int(k): float(v) for k, v in obj["front"].items()}
    back = {int(k): float(v) for k, v in obj["back"].items()}
    return front, back


def compute_cover_percent(region_map_L: Image.Image, paint_mask_L: Image.Image, weights: Dict[int, float]) -> float:
    region_px = region_map_L.load()
    paint_px = paint_mask_L.load()
    w, h = region_map_L.size

    denom: Dict[int, int] = {}
    numer: Dict[int, int] = {}

    for y in range(h):
        for x in range(w):
            rid = int(region_px[x, y])
            if rid == 0:
                continue
            denom[rid] = denom.get(rid, 0) + 1
            if paint_px[x, y] > 0:
                numer[rid] = numer.get(rid, 0) + 1

    total = 0.0
    for rid, weight in weights.items():
        d = denom.get(rid, 0)
        if d == 0:
            continue
        n = numer.get(rid, 0)
        total += weight * (n / d)
    return float(total)


# ---------- UI: Silhouette painter (coverage input) ----------
DISPLAY_SCALE_DEFAULT = 0.6
RESAMPLE_VIEW = Image.Resampling.LANCZOS

MAX_PHOTOS = 5

# 写真ごとの色（RGBA）。alphaはここでは“表示用”として使う（実maskはLで0/255）
PHOTO_COLORS_RGBA: List[Tuple[int, int, int, int]] = [
    (0, 200, 255, 120),   # 1: water/cyan
    (255, 90, 180, 120),  # 2: pink
    (255, 220, 0, 120),   # 3: yellow
    (255, 255, 255, 110), # 4: white
    (0, 255, 180, 120),   # 5: mint
]

def rgba_to_hex(rgb_or_rgba: Tuple[int, int, int, int] | Tuple[int, int, int]) -> str:
    r, g, b = rgb_or_rgba[0], rgb_or_rgba[1], rgb_or_rgba[2]
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class PhotoItem:
    path: str
    color_rgba: Tuple[int, int, int, int]
    paint_front: Image.Image  # L 0/255
    paint_back: Image.Image   # L 0/255
    thumb_tk: Optional[ImageTk.PhotoImage] = None


class CoveragePainter(ttk.Frame):
    """
    Right panel:
      FRONT and BACK silhouettes side-by-side.
      複数写真の「撮影範囲」を色分けで合成表示し、編集はアクティブ写真にのみ反映。
    """
    def __init__(self, master):
        super().__init__(master)

        # load assets
        self.front_sil = Image.open(C.SIL_FRONT).convert("RGBA")
        self.back_sil = Image.open(C.SIL_BACK).convert("RGBA")
        self.front_region = Image.open(C.REGION_FRONT).convert("L")
        self.back_region = Image.open(C.REGION_BACK).convert("L")

        if self.front_sil.size != self.front_region.size:
            raise ValueError("Front silhouette and region_map size mismatch.")
        if self.back_sil.size != self.back_region.size:
            raise ValueError("Back silhouette and region_map size mismatch.")

        self.weights_front, self.weights_back = load_weights(Path(C.WEIGHTS_JSON))

        self.Wf, self.Hf = self.front_sil.size
        self.Wb, self.Hb = self.back_sil.size

        self.scale = self._compute_initial_scale()

        # body masks (alpha > 0)
        self.body_front = self._alpha_to_body_mask(self.front_sil)
        self.body_back = self._alpha_to_body_mask(self.back_sil)

        # items (複数写真)
        self.items: List[PhotoItem] = []
        self.active_index: int = 0

        # state
        self.mode = tk.StringVar(value="paint")   # paint/erase
        self.tool = tk.StringVar(value="brush")   # brush/freehand
        self.brush_size = tk.IntVar(value=6)      # 1..20 in image coords

        self.cover_var = tk.StringVar(value="cover%: 0.00")
        self.active_var = tk.StringVar(value="active: (none)")

        self.active_view = "front"
        self._is_drawing = False
        self._last_xy: Optional[Tuple[int, int]] = None
        self._move_counter = 0

        # freehand
        self._freehand_points_front: List[Tuple[int, int]] = []
        self._freehand_points_back: List[Tuple[int, int]] = []
        self._preview_ids_front: List[int] = []
        self._preview_ids_back: List[int] = []

        # brush preview
        self._brush_circle_front: Optional[int] = None
        self._brush_circle_back: Optional[int] = None
        self._last_mouse_front: Optional[Tuple[int, int]] = None
        self._last_mouse_back: Optional[Tuple[int, int]] = None

        self._build_ui()
        self._render_all()

    # ---------- external API ----------
    def set_items(self, items: List[PhotoItem], active_index: int = 0):
        self.items = items
        if not self.items:
            self.active_index = 0
            self.active_var.set("active: (none)")
        else:
            self.active_index = max(0, min(active_index, len(self.items) - 1))
            self.active_var.set(f"active: {self.active_index+1}")
        self._render_all()

    def set_active_index(self, idx: int):
        if not self.items:
            self.active_index = 0
            self.active_var.set("active: (none)")
            self._render_all()
            return
        self.active_index = max(0, min(idx, len(self.items) - 1))
        self.active_var.set(f"active: {self.active_index+1}")
        self._render_all()

    def get_cover_percent_for_item(self, idx: int) -> float:
        if not self.items:
            return 0.0
        idx = max(0, min(idx, len(self.items) - 1))
        it = self.items[idx]
        f_cov = compute_cover_percent(self.front_region, it.paint_front, self.weights_front)
        b_cov = compute_cover_percent(self.back_region, it.paint_back, self.weights_back)
        return float(f_cov + b_cov)

    def get_coverage_masks_np_for_item(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.items:
            # fallback
            zf = np.zeros((self.Hf, self.Wf), np.uint8)
            zb = np.zeros((self.Hb, self.Wb), np.uint8)
            return zf, zb
        idx = max(0, min(idx, len(self.items) - 1))
        it = self.items[idx]
        f = np.array(it.paint_front, dtype=np.uint8)
        b = np.array(it.paint_back, dtype=np.uint8)
        f = (f > 0).astype(np.uint8) * 255
        b = (b > 0).astype(np.uint8) * 255
        return f, b

    # “合計cover%” は（重複を無視して）単純加算で表示（最大100にクリップ）
    def get_cover_percent_total(self) -> float:
        if not self.items:
            return 0.0
        total = sum(self.get_cover_percent_for_item(i) for i in range(len(self.items)))
        return float(min(total, 100.0))

    # ---------- internal ----------
    def _compute_initial_scale(self) -> float:
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        max_w = max(self.Wf, self.Wb)
        max_h = max(self.Hf, self.Hb)
        s_w = (sw * 0.9) / (2 * max_w)
        s_h = (sh * 0.75) / max_h
        s = min(s_w, s_h, DISPLAY_SCALE_DEFAULT)
        return max(0.3, min(s, 1.0))

    def _alpha_to_body_mask(self, rgba_img: Image.Image) -> Image.Image:
        a = rgba_img.split()[-1]
        return a.point(lambda p: 255 if p > 0 else 0).convert("L")

    def _clip_to_body(self, which: str, imgL: Image.Image):
        body = self.body_front if which == "front" else self.body_back
        clipped = ImageChops.multiply(imgL, body)
        imgL.paste(clipped)

    def _canvas_to_img(self, cx: int, cy: int) -> Tuple[int, int]:
        return int(cx / self.scale), int(cy / self.scale)

    def _in_bounds(self, which: str, ix: int, iy: int) -> bool:
        w, h = (self.Wf, self.Hf) if which == "front" else (self.Wb, self.Hb)
        return 0 <= ix < w and 0 <= iy < h

    def _build_ui(self):
        # 2-row toolbar
        toolbar = ttk.Frame(self, padding=6)
        toolbar.pack(side="top", fill="x")

        row1 = ttk.Frame(toolbar)
        row1.pack(side="top", fill="x")

        ttk.Label(row1, text="Tool:").pack(side="left")
        ttk.Radiobutton(row1, text="Brush", value="brush", variable=self.tool, command=self._on_tool_change).pack(side="left", padx=4)
        ttk.Radiobutton(row1, text="Freehand", value="freehand", variable=self.tool, command=self._on_tool_change).pack(side="left", padx=4)

        ttk.Separator(row1, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(row1, text="Mode:").pack(side="left")
        ttk.Radiobutton(row1, text="Paint", value="paint", variable=self.mode).pack(side="left", padx=4)
        ttk.Radiobutton(row1, text="Erase", value="erase", variable=self.mode).pack(side="left", padx=4)

        ttk.Separator(row1, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(row1, textvariable=self.active_var, font=("Arial", 11, "bold")).pack(side="left", padx=(8, 0))
        ttk.Label(row1, textvariable=self.cover_var, font=("Arial", 11, "bold")).pack(side="left", padx=10)

        row2 = ttk.Frame(toolbar)
        row2.pack(side="top", fill="x", pady=(6, 0))

        ttk.Label(row2, text="Brush size:").pack(side="left")
        self.brush_slider = ttk.Scale(
            row2, from_=1, to=20, variable=self.brush_size, orient="horizontal",
            command=self._on_brush_slider, length=220
        )
        self.brush_slider.pack(side="left", padx=6)
        ttk.Label(row2, textvariable=self.brush_size, width=3).pack(side="left")

        main = ttk.Frame(self, padding=6)
        main.pack(side="top", fill="both", expand=True)

        # FRONT
        front_frame = ttk.Frame(main)
        front_frame.pack(side="left", padx=(0, 10))
        ttk.Label(front_frame, text="FRONT").pack(side="top")

        self.canvas_front = tk.Canvas(
            front_frame,
            width=int(self.Wf * self.scale),
            height=int(self.Hf * self.scale),
            bg="#222222",
            highlightthickness=2,
            highlightbackground="#66ccff",
        )
        self.canvas_front.pack(side="top")

        # BACK
        back_frame = ttk.Frame(main)
        back_frame.pack(side="left")
        ttk.Label(back_frame, text="BACK").pack(side="top")

        self.canvas_back = tk.Canvas(
            back_frame,
            width=int(self.Wb * self.scale),
            height=int(self.Hb * self.scale),
            bg="#222222",
            highlightthickness=2,
            highlightbackground="#444444",
        )
        self.canvas_back.pack(side="top")

        for which, cv in (("front", self.canvas_front), ("back", self.canvas_back)):
            cv.bind("<ButtonPress-1>", lambda e, w=which: self._on_down(e, w))
            cv.bind("<B1-Motion>", lambda e, w=which: self._on_move(e, w))
            cv.bind("<ButtonRelease-1>", lambda e, w=which: self._on_up(e, w))
            cv.bind("<Motion>", lambda e, w=which: self._on_motion(e, w))
            cv.bind("<Leave>", lambda e, w=which: self._on_leave(e, w))

        self._on_tool_change()

    def _on_brush_slider(self, _val):
        self._refresh_brush_preview()

    def _on_tool_change(self):
        if self.tool.get() == "freehand":
            self.brush_slider.state(["disabled"])
            self._hide_brush_preview()
        else:
            self.brush_slider.state(["!disabled"])
            self._freehand_points_front.clear()
            self._freehand_points_back.clear()
            self._clear_preview("front")
            self._clear_preview("back")
            self._refresh_brush_preview()
        self._render_all()

    def _update_active_highlight(self):
        if self.active_view == "front":
            self.canvas_front.config(highlightbackground="#66ccff")
            self.canvas_back.config(highlightbackground="#444444")
        else:
            self.canvas_front.config(highlightbackground="#444444")
            self.canvas_back.config(highlightbackground="#66ccff")

    def _compose_view_multilayer(self, sil: Image.Image, which: str) -> Image.Image:
        w, h = sil.size
        comp = sil.copy()

        # 全写真のpaintを色付きで合成
        for idx, it in enumerate(self.items):
            paintL = it.paint_front if which == "front" else it.paint_back
            if paintL.getbbox() is None:
                continue
            r, g, b, a = it.color_rgba
            overlay = Image.new("RGBA", (w, h), (r, g, b, 0))
            alpha = paintL.point(lambda p: a if p > 0 else 0)
            overlay.putalpha(alpha)
            comp = Image.alpha_composite(comp, overlay)

        vw, vh = int(w * self.scale), int(h * self.scale)
        return comp.resize((vw, vh), resample=RESAMPLE_VIEW)

    def _render_front(self):
        img = self._compose_view_multilayer(self.front_sil, "front")
        self._tk_front = ImageTk.PhotoImage(img)
        self.canvas_front.delete("all")
        self.canvas_front.create_image(0, 0, anchor="nw", image=self._tk_front)
        self._draw_preview("front")
        self._draw_or_update_brush_circle("front")

    def _render_back(self):
        img = self._compose_view_multilayer(self.back_sil, "back")
        self._tk_back = ImageTk.PhotoImage(img)
        self.canvas_back.delete("all")
        self.canvas_back.create_image(0, 0, anchor="nw", image=self._tk_back)
        self._draw_preview("back")
        self._draw_or_update_brush_circle("back")

    def _render_all(self, update_cover: bool = True):
        self._render_front()
        self._render_back()
        if update_cover:
            self._update_cover_total()
        self._update_active_highlight()

    def _clear_preview(self, which: str):
        cv = self.canvas_front if which == "front" else self.canvas_back
        ids = self._preview_ids_front if which == "front" else self._preview_ids_back
        for cid in ids:
            try:
                cv.delete(cid)
            except Exception:
                pass
        if which == "front":
            self._preview_ids_front = []
        else:
            self._preview_ids_back = []

    def _draw_preview(self, which: str):
        if self.tool.get() != "freehand":
            self._clear_preview(which)
            return
        pts = self._freehand_points_front if which == "front" else self._freehand_points_back
        cv = self.canvas_front if which == "front" else self.canvas_back
        self._clear_preview(which)
        if len(pts) < 2:
            return
        flat = [v for p in pts for v in p]
        cid = cv.create_line(*flat, fill="cyan", width=2)
        if which == "front":
            self._preview_ids_front.append(cid)
        else:
            self._preview_ids_back.append(cid)

    def _hide_brush_preview(self):
        for which in ("front", "back"):
            cv = self.canvas_front if which == "front" else self.canvas_back
            attr = "_brush_circle_front" if which == "front" else "_brush_circle_back"
            cid = getattr(self, attr)
            if cid is not None:
                try:
                    cv.delete(cid)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._last_mouse_front = None
        self._last_mouse_back = None

    def _refresh_brush_preview(self):
        self._draw_or_update_brush_circle("front")
        self._draw_or_update_brush_circle("back")

    def _draw_or_update_brush_circle(self, which: str):
        if self.tool.get() != "brush":
            return
        r_img = int(self.brush_size.get())
        r_cv = int(r_img * self.scale)

        if which == "front":
            cv = self.canvas_front
            last = self._last_mouse_front
            attr = "_brush_circle_front"
        else:
            cv = self.canvas_back
            last = self._last_mouse_back
            attr = "_brush_circle_back"

        if last is None:
            return

        x, y = last
        x0, y0, x1, y1 = x - r_cv, y - r_cv, x + r_cv, y + r_cv

        cid = getattr(self, attr)
        if cid is None:
            cid = cv.create_oval(x0, y0, x1, y1, outline="yellow", width=2)
            setattr(self, attr, cid)
        else:
            cv.coords(cid, x0, y0, x1, y1)
        cv.tag_raise(cid)

    def _update_cover_total(self):
        if not self.items:
            self.cover_var.set("cover%: 0.00")
            return
        total = self.get_cover_percent_total()
        active = self.get_cover_percent_for_item(self.active_index) if self.items else 0.0
        self.cover_var.set(f"cover% total: {total:.2f}   (active {active:.2f})")

    def _set_active_view(self, which: str):
        if self.active_view != which:
            self.active_view = which
            self._update_active_highlight()

    def _active_item(self) -> Optional[PhotoItem]:
        if not self.items:
            return None
        if not (0 <= self.active_index < len(self.items)):
            return None
        return self.items[self.active_index]

    def _draw_brush_at(self, which: str, cx: int, cy: int):
        it = self._active_item()
        if it is None:
            return
        ix, iy = self._canvas_to_img(cx, cy)
        if not self._in_bounds(which, ix, iy):
            return

        paint = it.paint_front if which == "front" else it.paint_back
        fill_val = 255 if self.mode.get() == "paint" else 0
        r = int(self.brush_size.get())
        draw = ImageDraw.Draw(paint)
        draw.ellipse([ix - r, iy - r, ix + r, iy + r], fill=fill_val)
        self._clip_to_body(which, paint)

    def _stroke(self, which: str, x0: int, y0: int, x1: int, y1: int):
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        for i in range(steps + 1):
            x = int(x0 + dx * i / steps)
            y = int(y0 + dy * i / steps)
            self._draw_brush_at(which, x, y)

    def _freehand_commit(self, which: str):
        it = self._active_item()
        if it is None:
            return

        pts = self._freehand_points_front if which == "front" else self._freehand_points_back
        if len(pts) < 4:
            return

        img_pts = [self._canvas_to_img(cx, cy) for (cx, cy) in pts]
        paint = it.paint_front if which == "front" else it.paint_back
        fill_val = 255 if self.mode.get() == "paint" else 0
        draw = ImageDraw.Draw(paint)
        draw.polygon(img_pts, fill=fill_val)
        self._clip_to_body(which, paint)

    def _on_motion(self, event, which: str):
        if which == "front":
            self._last_mouse_front = (event.x, event.y)
        else:
            self._last_mouse_back = (event.x, event.y)
        self._draw_or_update_brush_circle(which)

    def _on_leave(self, _event, which: str):
        if which == "front":
            self._last_mouse_front = None
            if self._brush_circle_front is not None:
                try:
                    self.canvas_front.delete(self._brush_circle_front)
                except Exception:
                    pass
                self._brush_circle_front = None
        else:
            self._last_mouse_back = None
            if self._brush_circle_back is not None:
                try:
                    self.canvas_back.delete(self._brush_circle_back)
                except Exception:
                    pass
                self._brush_circle_back = None

    def _on_down(self, event, which: str):
        self._set_active_view(which)
        self._on_motion(event, which)

        if self.tool.get() == "freehand":
            if which == "front":
                self._freehand_points_front = [(event.x, event.y)]
            else:
                self._freehand_points_back = [(event.x, event.y)]
            self._render_all(update_cover=False)
            return

        self._is_drawing = True
        self._last_xy = (event.x, event.y)
        self._draw_brush_at(which, event.x, event.y)
        self._render_all(update_cover=False)

    def _on_move(self, event, which: str):
        self._set_active_view(which)
        self._on_motion(event, which)

        if self.tool.get() == "freehand":
            pts = self._freehand_points_front if which == "front" else self._freehand_points_back
            pts.append((event.x, event.y))
            self._render_all(update_cover=False)
            return

        if not self._is_drawing or self._last_xy is None:
            return

        x0, y0 = self._last_xy
        self._stroke(which, x0, y0, event.x, event.y)
        self._last_xy = (event.x, event.y)

        self._move_counter = (self._move_counter + 1) % 3
        if self._move_counter == 0:
            self._render_all(update_cover=False)

    def _on_up(self, event, which: str):
        self._set_active_view(which)
        self._on_motion(event, which)

        if self.tool.get() == "freehand":
            self._freehand_commit(which)
            if which == "front":
                self._freehand_points_front.clear()
            else:
                self._freehand_points_back.clear()
            self._clear_preview(which)
            self._render_all(update_cover=True)
            return

        self._is_drawing = False
        self._last_xy = None
        self._render_all(update_cover=True)


# ---------- Result Window ----------
class ResultWindowMulti(tk.Toplevel):
    def __init__(self, master: tk.Tk, *,
                 per_photo: List[dict],
                 total_summary: str):
        super().__init__(master)
        self.title("vIGA Results (Multi)")
        self.geometry("1500x920")

        self._imgtk_refs: List[ImageTk.PhotoImage] = []

        top = ttk.Frame(self, padding=8)
        top.pack(side="top", fill="x")
        ttk.Button(top, text="Close", command=self.destroy).pack(side="right")

        nb = ttk.Notebook(self)
        nb.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        # total tab
        total_frame = ttk.Frame(nb)
        nb.add(total_frame, text="TOTAL")
        txt = tk.Text(total_frame, height=10, wrap="word")
        txt.pack(side="top", fill="x")
        txt.insert("end", total_summary)
        txt.config(state="disabled")

        # each photo tab
        for i, item in enumerate(per_photo):
            frame = ttk.Frame(nb)
            nb.add(frame, text=f"Photo {i+1}")

            body = ttk.Frame(frame, padding=8)
            body.pack(side="top", fill="both", expand=True)

            imgs = ttk.Frame(body)
            imgs.pack(side="top", fill="both", expand=True)

            max_w, max_h = 320, 320

            def add_col(parent, title: str, im: Optional[Image.Image]):
                col = ttk.Frame(parent)
                col.pack(side="left", fill="both", expand=True, padx=8)
                ttk.Label(col, text=title, font=("Arial", 12, "bold")).pack(anchor="center", pady=(0, 6))
                if im is None:
                    ttk.Label(col, text="(not available)").pack()
                    return
                im_fit = pil_to_fit(im, max_w, max_h)
                tkimg = ImageTk.PhotoImage(im_fit)
                self._imgtk_refs.append(tkimg)
                ttk.Label(col, image=tkimg).pack(anchor="center")

            add_col(imgs, "Original", item.get("orig"))
            add_col(imgs, "lIGA Overlay", item.get("liga_overlay"))
            add_col(imgs, "Skin Overlay", item.get("skin_overlay"))
            add_col(imgs, "Skin-only", item.get("skin_only"))

            t2 = tk.Text(body, height=8, wrap="word")
            t2.pack(side="bottom", fill="x", pady=(10, 0))
            t2.insert("end", item.get("summary", ""))
            t2.config(state="disabled")


# ---------- Main GUI ----------
class VigaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("vIGA Pipeline MVP (Multi Photo + Coverage -> lIGA/skin -> IGA/BSA)")
        self.geometry("1500x860")

        self.items: List[PhotoItem] = []
        self.active_index: int = 0

        self._thumb_widgets: List[tk.Frame] = []

        self._build_ui()
        self._sync_painter()

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # top controls
        top = ttk.Frame(root)
        top.pack(side="top", fill="x")

        ttk.Button(top, text="画像を追加（最大5）", command=self.on_add_images).pack(side="left")
        ttk.Button(top, text="クリア", command=self.on_clear_images).pack(side="left", padx=(8, 0))

        self.status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.status_var).pack(side="right", padx=10)
        ttk.Button(top, text="実行", command=self.on_run).pack(side="right")

        body = ttk.Frame(root)
        body.pack(side="top", fill="both", expand=True, pady=(10, 0))

        # left: thumbnails list + big preview
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=False)

        ttk.Label(left, text="入力画像（クリックで選択）").pack(anchor="w")

        # scrollable thumbs
        self.thumb_canvas = tk.Canvas(left, width=340, height=360, bg="#111111", highlightthickness=1)
        self.thumb_canvas.pack(side="top", fill="x")

        self.thumb_scroll = ttk.Scrollbar(left, orient="vertical", command=self.thumb_canvas.yview)
        self.thumb_scroll.pack(side="right", fill="y")
        self.thumb_canvas.configure(yscrollcommand=self.thumb_scroll.set)

        self.thumb_inner = tk.Frame(self.thumb_canvas, bg="#111111")
        self.thumb_canvas.create_window((0, 0), window=self.thumb_inner, anchor="nw")

        def _on_config(_e):
            self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all"))
        self.thumb_inner.bind("<Configure>", _on_config)

        # big preview
        ttk.Label(left, text="選択中プレビュー").pack(anchor="w", pady=(10, 0))
        self.photo_canvas = tk.Canvas(left, width=340, height=340, bg="#111111", highlightthickness=1)
        self.photo_canvas.pack(side="top", fill="x")
        self._big_preview_tk: Optional[ImageTk.PhotoImage] = None

        # right: coverage painter
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        self.painter = CoveragePainter(right)
        self.painter.pack(fill="both", expand=True)

        # bottom: short summary
        bottom = ttk.Frame(root, padding=(0, 10, 0, 0))
        bottom.pack(side="bottom", fill="x")
        self.result_text = tk.StringVar(value="結果: (まだ実行していません)")
        ttk.Label(bottom, textvariable=self.result_text, font=("Arial", 11, "bold")).pack(anchor="w")

    def _sync_painter(self):
        self.painter.set_items(self.items, self.active_index)
        self._render_big_preview()

    def _render_big_preview(self):
        self.photo_canvas.delete("all")
        if not self.items:
            self._big_preview_tk = None
            return
        it = self.items[self.active_index]
        im = load_pil_with_exif(it.path)
        im = pil_to_fit(im, 340, 340)
        self._big_preview_tk = ImageTk.PhotoImage(im)
        self.photo_canvas.create_image(0, 0, anchor="nw", image=self._big_preview_tk)

    def _rebuild_thumbs(self):
        for w in self._thumb_widgets:
            try:
                w.destroy()
            except Exception:
                pass
        self._thumb_widgets = []

        for idx, it in enumerate(self.items):
            border = rgba_to_hex(it.color_rgba) if idx == self.active_index else "#333333"
            fr = tk.Frame(self.thumb_inner, bg="#111111", highlightthickness=3, highlightbackground=border)
            fr.pack(side="top", fill="x", pady=6, padx=6)
            self._thumb_widgets.append(fr)

            # thumbnail image
            if it.thumb_tk is None:
                im = load_pil_with_exif(it.path)
                im = pil_to_fit(im, 300, 180)
                it.thumb_tk = ImageTk.PhotoImage(im)

            lbl = tk.Label(fr, image=it.thumb_tk, bg="#111111")
            lbl.pack(side="top")

            info = tk.Label(fr, text=f"Photo {idx+1}  (color)", fg=rgba_to_hex(it.color_rgba),
                            bg="#111111", anchor="w")
            info.pack(side="top", fill="x")

            def _make_cb(i: int):
                return lambda _e=None: self.on_select_photo(i)

            lbl.bind("<Button-1>", _make_cb(idx))
            info.bind("<Button-1>", _make_cb(idx))
            fr.bind("<Button-1>", _make_cb(idx))

        self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all"))

    def on_select_photo(self, idx: int):
        if not self.items:
            return
        self.active_index = max(0, min(idx, len(self.items) - 1))
        self._rebuild_thumbs()
        self._sync_painter()

    def on_add_images(self):
        if len(self.items) >= MAX_PHOTOS:
            messagebox.showinfo("max reached", f"画像は最大 {MAX_PHOTOS} 枚までにしてるよ！")
            return

        paths = filedialog.askopenfilenames(
            title="画像を選択（複数可）",
            filetypes=[
                ("images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.heic *.heif *.JPG *.JPEG *.PNG"),
                ("all", "*.*"),
            ],
        )
        if not paths:
            return

        remaining = MAX_PHOTOS - len(self.items)
        paths = list(paths)[:remaining]

        for p in paths:
            color = PHOTO_COLORS_RGBA[len(self.items) % len(PHOTO_COLORS_RGBA)]
            # 各写真に専用のpaint maskを持たせる
            front = Image.new("L", self.painter.front_sil.size, 0)
            back = Image.new("L", self.painter.back_sil.size, 0)
            self.items.append(PhotoItem(path=str(p), color_rgba=color, paint_front=front, paint_back=back))

        if self.items:
            self.active_index = len(self.items) - len(paths)  # 追加した最初のやつにフォーカス
        self._rebuild_thumbs()
        self._sync_painter()

    def on_clear_images(self):
        if not self.items:
            return
        if not messagebox.askyesno("Clear", "画像と塗りを全部クリアする？"):
            return
        self.items = []
        self.active_index = 0
        self._rebuild_thumbs()
        self._sync_painter()
        self.result_text.set("結果: (まだ実行していません)")

    def on_run(self):
        if not self.items:
            messagebox.showwarning("No image", "先に画像を追加してね！")
            return

        self.status_var.set("実行中…")
        self.update_idletasks()

        try:
            per_photo_tabs: List[dict] = []

            total_bsa = 0.0
            # IGA合成（lesion pixels重み）
            w_sum = 0.0
            wm_sum = 0.0

            run_dirs: List[str] = []

            for i, it in enumerate(self.items):
                cover_i = self.painter.get_cover_percent_for_item(i)
                cov_f, cov_b = self.painter.get_coverage_masks_np_for_item(i)

                # cover=0でも回したいならコメントアウトしてね
                if cover_i <= 0.0:
                    # その写真は範囲未入力としてスキップ（安全策）
                    continue

                res = run_inference(
                    image_path=it.path,
                    cover_percent=cover_i,
                    coverage_front_mask=cov_f,
                    coverage_back_mask=cov_b,
                    device_str=C.DEFAULT_DEVICE,
                )

                run_dirs.append(res.run_dir)

                total_bsa += float(res.estimated_lesion_bsa_percent)

                # lesion pixels in skin を重みとして iga_mean を合成
                w = float(getattr(res, "liga_lesion_pixels_in_skin", 0))
                if w > 0:
                    w_sum += w
                    wm_sum += w * float(res.iga_mean)

                orig = load_pil_with_exif(it.path)
                liga_overlay = Image.open(res.out_liga_overlay).convert("RGB")

                skin_overlay_pil = None
                skin_only_pil = None

                skin_overlay_path = getattr(res, "out_skin_overlay", None)
                skin_mask_path = getattr(res, "out_skin_mask", None)

                if skin_overlay_path and Path(str(skin_overlay_path)).exists():
                    skin_overlay_pil = Image.open(skin_overlay_path).convert("RGB")

                if skin_mask_path and Path(str(skin_mask_path)).exists():
                    maskL = Image.open(skin_mask_path).convert("L")
                    skin_only_pil = apply_mask_rgb_pil(orig, maskL)

                summary = (
                    f"[Photo {i+1}]\n"
                    f"path={it.path}\n"
                    f"cover%={res.cover_percent:.2f}\n"
                    f"lesion_fraction_in_skin={res.lesion_fraction_in_skin:.3f}\n"
                    f"推定皮疹面積(BSA%)={res.estimated_lesion_bsa_percent:.2f}\n"
                    f"推定IGA={res.iga_int} (mean={res.iga_mean:.2f})\n"
                    f"保存先={res.run_dir}\n"
                )

                per_photo_tabs.append({
                    "orig": orig,
                    "liga_overlay": liga_overlay,
                    "skin_overlay": skin_overlay_pil,
                    "skin_only": skin_only_pil,
                    "summary": summary,
                })

            if not per_photo_tabs:
                messagebox.showwarning("No valid photos", "cover% が入ってる写真が無かったよ（全部0%かも）")
                self.status_var.set("")
                return

            if w_sum > 0:
                iga_mean_total = wm_sum / w_sum
            else:
                iga_mean_total = 0.0

            iga_int_total = int(np.rint(iga_mean_total))
            iga_int_total = max(0, min(4, iga_int_total))

            total_summary = (
                "=== TOTAL (simple sum / overlap double-count) ===\n"
                f"有効写真数={len(per_photo_tabs)} / 全投入={len(self.items)}\n"
                f"推定皮疹面積 合計(BSA%)={total_bsa:.2f}\n"
                f"推定IGA 合成={iga_int_total} (mean={iga_mean_total:.2f}, weighted by lesion_pixels_in_skin)\n"
                f"run_dirs:\n- " + "\n- ".join(run_dirs) + "\n"
            )

            self.result_text.set(
                f"結果: 合計BSA%={total_bsa:.2f} / 合成IGA={iga_int_total} (mean={iga_mean_total:.2f})"
            )

            ResultWindowMulti(self, per_photo=per_photo_tabs, total_summary=total_summary)

            self.status_var.set("完了")

        except Exception as e:
            self.status_var.set("失敗")
            messagebox.showerror("Error", f"実行に失敗したよ:\n{e}")


def main():
    missing = []
    for p in [
        C.CKPT_LIGA, C.CKPT_SKIN,
        C.SIL_FRONT, C.SIL_BACK, C.REGION_FRONT, C.REGION_BACK,
        C.WEIGHTS_JSON
    ]:
        if not Path(p).exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    app = VigaApp()
    app.mainloop()


if __name__ == "__main__":
    main()
