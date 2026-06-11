"""Science-grade UI overlays: scale bar, status bar, toasts, axes gizmo,
and the analysis drawer (inspector / phase diagram / radial profile /
region statistics).

Like overlay.py, everything renders to PIL images; the wgpu wrappers in
wgpu_overlay.py handle the GPU upload. Plot panels rasterize matplotlib
figures through the Agg canvas (no pyplot, no global state).
"""

import time

import numpy as np
from PIL import Image, ImageDraw

from .overlay import Panel, PanelStyle, _rounded
from .themes import DarkTheme

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

SCALEBAR_STYLE = PanelStyle(
    font_size=20, line_height=28, margin=8, min_width=10,
    bg_color=DarkTheme.TRANSPARENT,
    text_color=DarkTheme.C_235_238_245_255,
    accent_color=DarkTheme.C_255_255_255_255,
    toggle_on_color=DarkTheme.C_100_180_255_255,
    toggle_off_color=DarkTheme.C_110_115_130_255,
    dropdown_bg=DarkTheme.C_34_37_50_255,
    dropdown_hover=DarkTheme.C_80_100_140_255,
    slider_btn=DarkTheme.C_64_70_88_255,
    position="bottom-center",
    font_family="sans-serif",
)

STATUS_STYLE = PanelStyle(
    font_size=16, line_height=24, margin=8, min_width=10,
    bg_color=DarkTheme.C_10_12_20_165,
    text_color=DarkTheme.C_200_206_218_255,
    accent_color=DarkTheme.C_100_180_255_255,
    toggle_on_color=DarkTheme.C_100_180_255_255,
    toggle_off_color=DarkTheme.C_110_115_130_255,
    dropdown_bg=DarkTheme.C_34_37_50_255,
    dropdown_hover=DarkTheme.C_80_100_140_255,
    slider_btn=DarkTheme.C_64_70_88_255,
    position="bottom-right",
    font_family="sans-serif",
    radius=10,
)

TOAST_STYLE = PanelStyle(
    font_size=20, line_height=30, margin=10, min_width=10,
    bg_color=DarkTheme.BG_SURFACE,
    text_color=DarkTheme.TEXT_PRIMARY,
    accent_color=DarkTheme.SUCCESS,
    toggle_on_color=DarkTheme.SUCCESS,
    toggle_off_color=DarkTheme.DANGER,
    dropdown_bg=DarkTheme.BG_RAISED,
    dropdown_hover=DarkTheme.ACCENT_DIM,
    slider_btn=DarkTheme.BG_RAISED,
    position="bottom-right",
    font_family="sans-serif",
    radius=12,
)

GIZMO_STYLE = PanelStyle(
    font_size=15, line_height=20, margin=6, min_width=10,
    bg_color=DarkTheme.C_10_12_20_130,
    text_color=DarkTheme.C_220_224_232_255,
    accent_color=DarkTheme.C_255_255_255_255,
    toggle_on_color=DarkTheme.C_100_180_255_255,
    toggle_off_color=DarkTheme.C_110_115_130_255,
    dropdown_bg=DarkTheme.C_34_37_50_255,
    dropdown_hover=DarkTheme.C_80_100_140_255,
    slider_btn=DarkTheme.C_64_70_88_255,
    position="bottom-right",
    font_family="sans-serif",
    radius=10,
)

DRAWER_STYLE = PanelStyle(
    font_size=20, line_height=30, margin=12, min_width=360,
    bg_color=DarkTheme.BG_SURFACE,          # DarkTheme.BG_SURFACE @ panel alpha
    text_color=DarkTheme.TEXT_PRIMARY,     # DarkTheme.TEXT_PRIMARY
    accent_color=DarkTheme.ACCENT,    # DarkTheme.ACCENT
    toggle_on_color=DarkTheme.C_100_180_255_255,
    toggle_off_color=DarkTheme.C_110_115_130_255,
    dropdown_bg=DarkTheme.C_34_37_50_255,
    dropdown_hover=DarkTheme.C_80_100_140_255,
    slider_btn=DarkTheme.C_64_70_88_255,
    position="center-right",
    font_family="sans-serif",
    radius=14,
)


# ---------------------------------------------------------------------------
# Scale bar
# ---------------------------------------------------------------------------

class ScaleBar(Panel):
    """Physical scale bar pinned above the bottom edge.

    The bar length is the largest 1/2/5x10^n kpc value that fits in
    ~22% of the framebuffer width at the reference depth (the distance
    from the camera to the view center), under the current FOV.
    """

    def __init__(self):
        super().__init__(SCALEBAR_STYLE)
        self.enabled = True
        self.anchor_offset = (0, -34)
        self._last_key = None

    def update(self, camera, kpc_per_code, center):
        if not self.enabled:
            return
        from .analysis import nice_scale_bar

        dist_code = float(np.linalg.norm(
            np.asarray(center, dtype=np.float64) - camera.position))
        if not np.isfinite(dist_code) or dist_code <= 0:
            return
        fb_w, fb_h = self._fb_width, self._fb_height
        # World width of one pixel at the reference depth.
        world_h = 2.0 * dist_code * np.tan(np.radians(camera.fov) / 2.0)
        kpc_per_px = world_h * kpc_per_code / max(fb_h, 1)
        target_kpc = kpc_per_px * fb_w * 0.22
        val_kpc, label = nice_scale_bar(target_kpc)
        bar_px = int(val_kpc / max(kpc_per_px, 1e-30))
        if bar_px < 10 or bar_px > fb_w:
            return

        key = (label, bar_px, fb_w, fb_h)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key

        s = self.style
        LH = s.line_height
        th = LH + 14
        tw = bar_px + 4
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        ybar = th - 6
        shadow = DarkTheme.C_0_0_0_180
        for dx, dy in ((1, 1),):
            draw.line([(2 + dx, ybar + dy), (tw - 2 + dx, ybar + dy)],
                      fill=shadow, width=3)
            draw.line([(2 + dx, ybar - 7 + dy), (2 + dx, ybar + dy)],
                      fill=shadow, width=3)
            draw.line([(tw - 2 + dx, ybar - 7 + dy), (tw - 2 + dx, ybar + dy)],
                      fill=shadow, width=3)
        col = s.accent_color
        draw.line([(2, ybar), (tw - 2, ybar)], fill=col, width=3)
        draw.line([(2, ybar - 7), (2, ybar)], fill=col, width=3)
        draw.line([(tw - 2, ybar - 7), (tw - 2, ybar)], fill=col, width=3)
        bbox = draw.textbbox((0, 0), label, font=self._font)
        tx = (tw - (bbox[2] - bbox[0])) // 2
        draw.text((tx + 1, 1), label, fill=shadow, font=self._font)
        draw.text((tx, 0), label, fill=s.text_color, font=self._font)

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def render(self):
        if not self.enabled:
            return
        super().render()


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

class StatusBar(Panel):
    """One-line readout: camera position (kpc), distance to center,
    redshift, active field + units, particle count."""

    def __init__(self):
        super().__init__(STATUS_STYLE)
        self.enabled = True
        self._last_key = None

    def update(self, camera, units, center, field_name, n_vis, n_tot):
        if not self.enabled:
            return
        from .physics import field_unit_label

        k = units.length_to_kpc
        p = camera.position * k
        d = float(np.linalg.norm(
            (np.asarray(center) - camera.position))) * k
        unit = field_unit_label(field_name)
        ftxt = f"{field_name} [{unit}]" if unit else field_name
        ztxt = (f"z={units.redshift:.2f}  " if units.cosmological
                and units.redshift > 1e-3 else "")
        text = (f"({p[0]:,.0f}, {p[1]:,.0f}, {p[2]:,.0f}) kpc   "
                f"d_center={d:,.1f} kpc   {ztxt}"
                f"{ftxt}   {n_vis/1e6:.1f}M/{n_tot/1e6:.1f}M pts")

        key = (text, self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key

        s = self.style
        M = s.margin
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bbox = dummy.textbbox((0, 0), text, font=self._font)
        tw = bbox[2] - bbox[0] + M * 3
        th = s.line_height + M
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius, fill=s.bg_color,
                 outline=DarkTheme.C_255_255_255_26)
        draw.text((M + 4, M // 2 + 1), text, fill=s.text_color, font=self._font)

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def render(self):
        if not self.enabled:
            return
        super().render()


# ---------------------------------------------------------------------------
# Toasts
# ---------------------------------------------------------------------------

class ToastOverlay(Panel):
    """Transient notification stack (top-center), newest on top.

    show() may be called from anywhere in the app; update() rebuilds the
    texture only when the visible set changes (or during fade-out).
    """

    DURATION = 4.0
    FADE = 0.5       # fade-out window (s)
    FADE_IN = 0.2    # fade-in window (s)
    MAX_TOASTS = 5

    def __init__(self):
        super().__init__(TOAST_STYLE)
        self.enabled = True
        self.anchor_offset = (0, -90)  # clear the status bar + gizmo
        self._toasts = []  # (text, t_expire, kind, t_created)

    def show(self, text, kind="info", duration=None):
        now = time.time()
        self._toasts.append((str(text),
                             now + (duration or self.DURATION), kind,
                             now))
        self._toasts = self._toasts[-self.MAX_TOASTS:]

    def push(self, message, type="info", duration_s=None):
        """Section 6.I API: alias of show() with its naming."""
        self.show(message, kind=type, duration=duration_s)

    @property
    def active(self):
        return bool(self._toasts)

    def update(self):
        if not self.enabled:
            return
        now = time.time()
        self._toasts = [t for t in self._toasts if t[1] > now]
        # (kept sorted by creation; expiry prunes in place)
        if not self._toasts:
            self._panel_w = self._panel_h = 0
            self._tex = None
            return

        s = self.style
        M, LH = s.margin, s.line_height
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tw = 10
        for text, *_ in self._toasts:
            bbox = dummy.textbbox((0, 0), text, font=self._font)
            tw = max(tw, bbox[2] - bbox[0] + M * 4)
        row_h = LH + 8
        th = row_h * len(self._toasts) + 4

        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        kind_col = {
            "info": s.text_color,
            "ok": s.toggle_on_color,
            "warn": DarkTheme.C_255_200_110_255,
            "error": s.toggle_off_color,
        }
        y = 0
        for text, t_exp, kind, t_new in reversed(self._toasts):
            alpha = 1.0
            if t_exp - now < self.FADE:
                alpha = max(0.0, (t_exp - now) / self.FADE)
            age = now - t_new
            if age < self.FADE_IN:
                alpha *= age / self.FADE_IN
            bg = tuple(list(s.bg_color[:3]) + [int(s.bg_color[3] * alpha)])
            fg0 = kind_col.get(kind, s.text_color)
            fg = tuple(list(fg0[:3]) + [int(fg0[3] * alpha)])
            _rounded(draw, [(0, y + 2), (tw - 1, y + row_h - 2)], s.radius,
                     fill=bg, outline=(255, 255, 255, int(30 * alpha)))
            # Type-colored left border strip (Section 6.I).
            draw.rectangle([(0, y + 4), (4, y + row_h - 4)], fill=fg)
            draw.text((M + 10, y + 5), text, fill=fg, font=self._font)
            y += row_h

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def render(self):
        if not self.enabled or not self._toasts:
            return
        super().render()


# ---------------------------------------------------------------------------
# Orientation axes gizmo
# ---------------------------------------------------------------------------

class AxesGizmo(Panel):
    """Small corner triad showing the simulation x/y/z axes as seen from
    the current camera orientation (depth-dimmed, painter-sorted)."""

    SIZE = 92  # px at 1080p, DPI-scaled at draw time

    def __init__(self):
        super().__init__(GIZMO_STYLE)
        self.enabled = True
        self.anchor_offset = (0, -52)
        self._last_key = None

    def update(self, camera):
        if not self.enabled:
            return
        r, u, f = camera.right, camera.up, camera.forward
        # Quantize orientation so tiny mouse jitter doesn't re-rasterize.
        key = tuple(int(v * 200) for v in np.concatenate([r, u, f]))
        key = key + (self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key

        size = max(48, int(self.SIZE * self._dpi_scale))
        tw = th = size
        cx = cy = size // 2
        L = size * 0.36
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        s = self.style
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius, fill=s.bg_color,
                 outline=DarkTheme.C_255_255_255_22)

        axes = [
            ("X", np.array([1.0, 0, 0]), (235, 110, 110)),
            ("Y", np.array([0, 1.0, 0]), (120, 220, 130)),
            ("Z", np.array([0, 0, 1.0]), (120, 170, 255)),
        ]
        ents = []
        for label, e, col in axes:
            px = float(e @ r) * L
            py = -float(e @ u) * L
            depth = float(e @ f)
            ents.append((depth, label, px, py, col))
        # Draw back-to-front; dim axes pointing away from the viewer.
        for depth, label, px, py, col in sorted(ents, key=lambda t: -t[0]):
            fade = 0.45 + 0.55 * (1.0 - depth) / 2.0
            c = tuple(int(v * fade) for v in col) + (255,)
            draw.line([(cx, cy), (cx + px, cy + py)], fill=c, width=3)
            lx = cx + px * 1.28 - 4
            ly = cy + py * 1.28 - self.style.font_size // 2
            draw.text((lx, ly), label, fill=c, font=self._font)
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=DarkTheme.C_230_230_235_255)

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def render(self):
        if not self.enabled:
            return
        super().render()


# ---------------------------------------------------------------------------
# Aperture overlay
# ---------------------------------------------------------------------------

APERTURE_STYLE = PanelStyle(
    font_size=18, line_height=26, margin=8, min_width=10,
    bg_color=DarkTheme.TRANSPARENT,
    text_color=DarkTheme.C_235_238_245_255,
    accent_color=DarkTheme.C_120_200_255_255,
    toggle_on_color=DarkTheme.C_120_200_255_255,
    toggle_off_color=DarkTheme.C_110_115_130_255,
    dropdown_bg=DarkTheme.C_34_37_50_255,
    dropdown_hover=DarkTheme.C_80_100_140_255,
    slider_btn=DarkTheme.C_64_70_88_255,
    position="top-left",
    font_family="sans-serif",
)


class ApertureOverlay(Panel):
    """Projected circle marking the analysis aperture sphere.

    The app computes the screen-space center and pixel radius each
    frame (perspective projection of the world-space sphere) and calls
    update(); the panel rasterizes a ring + crosshair + radius label
    into a texture bounding the circle. While the aperture is being
    placed the ring renders dashed.
    """

    MAX_TEX = 2048

    def __init__(self):
        super().__init__(APERTURE_STYLE)
        self.enabled = False
        self._last_key = None
        self._pos_px = (0, 0)

    def _panel_origin(self, tw, th):
        return self._pos_px

    def update(self, cx_px, cy_px, radius_px, label, placing=False):
        if not self.enabled:
            return
        r = float(np.clip(radius_px, 6.0, self.MAX_TEX / 2 - 4))
        key = (int(cx_px), int(cy_px), int(r), label, placing,
               self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key

        pad = 30 + self.style.line_height
        size = int(2 * r) + 2 * pad
        tw = th = min(size, self.MAX_TEX)
        c = tw // 2
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        col = self.style.accent_color if not placing else DarkTheme.C_255_200_110_255
        shadow = DarkTheme.C_0_0_0_150

        if placing:
            # Dashed ring: 48 arc segments, alternating.
            for k in range(0, 48, 2):
                a0, a1 = k * 7.5, (k + 1) * 7.5
                draw.arc([c - r + 1, c - r + 1, c + r - 1, c + r - 1],
                         a0, a1, fill=shadow, width=4)
                draw.arc([c - r, c - r, c + r, c + r], a0, a1, fill=col, width=3)
        else:
            draw.ellipse([c - r + 1, c - r + 1, c + r - 1, c + r - 1],
                         outline=shadow, width=4)
            draw.ellipse([c - r, c - r, c + r, c + r], outline=col, width=3)
        # Center crosshair
        for dx, dy, col2 in ((1, 1, shadow), (0, 0, col)):
            draw.line([(c - 9 + dx, c + dy), (c + 9 + dx, c + dy)],
                      fill=col2, width=2)
            draw.line([(c + dx, c - 9 + dy), (c + dx, c + 9 + dy)],
                      fill=col2, width=2)
        if label:
            bb = draw.textbbox((0, 0), label, font=self._font)
            lx = c - (bb[2] - bb[0]) // 2
            ly = int(c - r) - self.style.line_height - 2
            if ly < 0:
                ly = min(int(c + r) + 6, th - self.style.line_height)
            draw.text((lx + 1, ly + 1), label, fill=shadow, font=self._font)
            draw.text((lx, ly), label, fill=self.style.text_color,
                      font=self._font)

        self._pos_px = (int(cx_px - tw // 2), int(cy_px - th // 2))
        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._pos_px
        self._upload_panel(tw, th, img.tobytes())

    def render(self):
        if not self.enabled:
            return
        super().render()


def vizmo_cache_mb():
    """Size of ~/.cache/vizmo (smoothing-length cache etc.) in MB."""
    import os

    base = os.environ.get("XDG_CACHE_HOME",
                          os.path.expanduser("~/.cache"))
    root = os.path.join(base, "vizmo")
    total = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / 1e6


class ProfilerOverlay(Panel):
    """F10 GPU/CPU frame-time bar chart (Item 4).

    Rows: one per pass with a bar scaled to the slowest pass and the
    ms value right-aligned; total frame time on top. Fed by GPU
    timestamp queries when the device supports them, else the
    renderer's CPU-side pass timings (labelled accordingly).
    """

    BAR_W = 200

    def __init__(self):
        super().__init__(STATUS_STYLE)
        self.enabled = False
        self.style = PanelStyle(**{**STATUS_STYLE.__dict__,
                                   "position": "center-left"})
        self._font = self._font  # keep base font
        self._last_key = None

    def update(self, pass_times, total_ms, gpu_timed):
        if not self.enabled:
            return
        items = tuple(sorted((k, round(v, 2))
                             for k, v in pass_times.items()))
        key = (items, round(total_ms, 1), gpu_timed,
               self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key

        s = self.style
        M, LH = s.margin, s.line_height
        rows = sorted(pass_times.items(), key=lambda kv: -kv[1])
        tw = self.BAR_W + 150
        th = LH * (len(rows) + 2) + M * 2
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius,
                 fill=s.bg_color, outline=DarkTheme.C_255_255_255_26)
        src_lbl = "GPU timestamps" if gpu_timed else "CPU-timed"
        draw.text((M, M), f"frame {total_ms:.1f} ms  ({src_lbl})",
                  fill=s.accent_color, font=self._font)
        y = M + LH
        vmax = max([v for _, v in rows] + [1e-6])
        for name, v in rows:
            draw.text((M, y), name, fill=s.text_color, font=self._font)
            bx0 = M + 80
            bw = int(self.BAR_W * v / vmax)
            _rounded(draw, [(bx0, y + 4), (bx0 + max(bw, 2), y + LH - 6)],
                     4, fill=DarkTheme.C_61_126_255_220)
            txt = f"{v:.2f} ms"
            bb = draw.textbbox((0, 0), txt, font=self._font)
            draw.text((tw - M - (bb[2] - bb[0]), y), txt,
                      fill=s.text_color, font=self._font)
            y += LH
        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def render(self):
        if not self.enabled:
            return
        super().render()


class SightlinesOverlay(Panel):
    """Projected absorption sightlines drawn as labelled 2D segments.

    The app passes a list of (x0, y0, x1, y1, label, color) tuples in
    framebuffer pixels each frame; rebuild is keyed on the rounded
    coordinates so static views don't re-rasterize.
    """

    def __init__(self):
        super().__init__(APERTURE_STYLE)
        self.enabled = False
        self._last_key = None
        self._pos_px = (0, 0)

    def _panel_origin(self, tw, th):
        return self._pos_px

    def update(self, segments):
        if not self.enabled or not segments:
            self._tex = None
            return
        key = tuple((int(a), int(b), int(c), int(d), lbl)
                    for a, b, c, d, lbl, _ in segments) + (
                        self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key
        tw, th = max(self._fb_width, 4), max(self._fb_height, 4)
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        for x0, y0, x1, y1, lbl, color in segments:
            draw.line([(x0 + 1, y0 + 1), (x1 + 1, y1 + 1)],
                      fill=DarkTheme.C_0_0_0_150, width=4)
            draw.line([(x0, y0), (x1, y1)], fill=color, width=2)
            for x, y in ((x0, y0), (x1, y1)):
                draw.ellipse([x - 4, y - 4, x + 4, y + 4],
                             outline=color, width=2)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            draw.text((mx + 7, my + 1), lbl, fill=DarkTheme.C_0_0_0_150,
                      font=self._font)
            draw.text((mx + 6, my), lbl, fill=color, font=self._font)
        self._pos_px = (0, 0)
        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = 0, 0
        self._upload_panel(tw, th, img.tobytes())

    def render(self):
        if not self.enabled:
            return
        super().render()


# ---------------------------------------------------------------------------
# Analysis drawer
# ---------------------------------------------------------------------------

def _fig_to_image(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(buf.copy(), "RGBA")


def _dark_axes(fig, ax):
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0.04, 0.05, 0.09, 0.55))
    for spine in ax.spines.values():
        spine.set_color((0.75, 0.78, 0.85, 0.8))
    ax.tick_params(colors=(0.82, 0.85, 0.9), labelsize=9)
    ax.xaxis.label.set_color((0.88, 0.9, 0.95))
    ax.yaxis.label.set_color((0.88, 0.9, 0.95))
    ax.title.set_color((0.92, 0.94, 0.97))


class AnalysisDrawer(Panel):
    """Right-side science drawer hosting one of four tools:

    - inspector: physical properties of a picked particle
    - phase:     mass-weighted 2D phase diagram (preset pairs)
    - profile:   radial profile of a field about the view center
    - stats:     aggregate properties of a sphere about the center

    The app calls set_mode()/refresh() and forwards clicks; compute
    happens through vizmo.analysis on demand and results are cached
    until refresh() or a mode change.
    """

    PROFILE_FIELDS = ["Density", "RotationCurve", "VelocityDispersion3D",
                      "EnclosedMass", "AngularMomentum", "Temperature",
                      "RadialVelocity", "MetallicityZsun", "Pressure",
                      "TcoolOverTff", "CoolingTime", "Entropy",
                      "VelocityMagnitude"]

    def __init__(self):
        super().__init__(DRAWER_STYLE)
        self.enabled = False
        self.mode = None
        self._buttons = []  # (x0, y0, x1, y1, action) in panel-local px
        self._phase_idx = 0
        self._profile_idx = 0
        self._stats_radius_kpc = None
        self._picked_index = None
        self._cache = {}
        self._last_key = None
        # Analysis scope: None = global, else dict(center=(3,) code
        # units, radius_kpc=float, center_mode=str). Set by the
        # aperture tool; every tool computes within it when use_scope.
        self.scope = None
        self.use_scope = True
        # Filters tool state: candidate-field index for "Add" and a
        # per-field percentile cache (0..100, subsampled) used for
        # robust range nudging on wildly log-distributed fields.
        self._filter_field_idx = 0
        self._pctiles = {}
        # Profile/phase/stats/spectrum tool state.
        self._profile_split = False     # temperature-phase split tracks
        self._phase_custom = None       # None=presets, else [xf, yf]
        self._phase_weight_idx = 0      # index into PHASE_WEIGHTINGS
        self._show_halo = False         # stats: append halo_properties
        # Last computed results, for the export buttons.
        self._last_profile = None       # (r, prof_or_tracks, field, unit)
        self._last_phase = None         # phase dict
        self._last_ps = None            # power spectrum dict
        self._last_orbit = None         # (orbit_result, props) from orbit.py
        # Halo list / inspector state (Section 4.G).
        self.catalog = None             # dict from catalog.load_catalog
        self._halo_sort = ("M_halo", True)   # column, descending
        self._halo_filter_idx = 0
        self._halo_threshold = 10.0          # log10 Msun marker cut
        self._halo_selected = None           # catalog row index
        # Spectrum viewer state (Section 3.D).
        self._spec_compare = False
        self._spec_ion = None
        # Phase brushing (Item 1) + overlays (Items 2/3/5C).
        self.brush_active = False
        self.brush_kind = "rect"        # rect | polygon | ellipse
        self.brush_points = []          # clicks in DATA coords
        self.brush_shape = None         # committed shape dict
        self.brush_mask = None          # boolean particle mask
        self.brush_only = False         # analysis uses mask
        self.obs_datasets = []          # [(name, x, y, labels), ...]
        self.show_precip_line = False
        self.tvir_K = None
        self._phase_geom = None         # plot-pixel <-> data mapping
        self._nfw_fit = None            # (rho_s, r_s) overlay on profile

    # -- filters helpers -----------------------------------------------------

    FILTER_CANDIDATES = ["Temperature", "NumberDensity", "Density",
                         "RadialVelocity", "VelocityMagnitude",
                         "MetallicityZsun", "RadiusFromCenter",
                         "StarFormationRate", "Masses"]

    def _filter_fields(self, data):
        avail = data.available_fields_with_derived()
        out = [f for f in self.FILTER_CANDIDATES if f in avail]
        out += [f for f in avail if f not in out]
        return out

    def _percentiles(self, data, field):
        key = (field, data.n_particles)
        if key not in self._pctiles:
            vals = np.asarray(data.get_field(field), dtype=np.float64)
            n = len(vals)
            if n > 2_000_000:
                rng = np.random.default_rng(0)
                vals = vals[rng.choice(n, size=2_000_000, replace=False)]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                vals = np.zeros(1)
            self._pctiles[key] = np.percentile(vals, np.arange(101))
        return self._pctiles[key]

    @staticmethod
    def _fmt_val(v):
        a = abs(v)
        if a != 0 and (a >= 1e4 or a < 1e-2):
            return f"{v:.2e}"
        return f"{v:.4g}"

    def set_scope(self, center, radius_kpc, center_mode="densest",
                  region=None):
        self.scope = {
            "center": np.asarray(center, dtype=np.float64),
            "radius_kpc": float(radius_kpc),
            "center_mode": center_mode,
            "region": region,
        }
        self.use_scope = True
        self.refresh()

    def clear_scope(self):
        self.scope = None
        self.refresh()

    def _active_scope(self):
        """(center, radius_kpc) or (None, None) for global."""
        if self.scope is not None and self.use_scope:
            return self.scope["center"], self.scope["radius_kpc"]
        return None, None

    def _active_region(self):
        """SelectionRegion when an aperture shape is active, else None."""
        if self.scope is not None and self.use_scope:
            return self.scope.get("region")
        return None

    # -- state management ---------------------------------------------------

    def toggle(self, mode):
        """Open drawer in `mode`, or close it if already showing it."""
        if self.enabled and self.mode == mode:
            self.enabled = False
            self.mode = None
        else:
            self.enabled = True
            self.mode = mode
        return self.enabled

    def open_inspector(self, index):
        self._picked_index = index
        self._cache.pop("inspector", None)
        self.enabled = True
        self.mode = "inspector"

    def refresh(self):
        self._cache.clear()
        self._last_key = None

    # -- compute ------------------------------------------------------------

    def _phase_presets(self, data):
        from .analysis import available_phase_presets

        presets = available_phase_presets(data)
        return presets or [("Masses", "Masses")]

    def _profile_fields(self, data):
        avail = set(data.available_fields_with_derived())
        out = [f for f in self.PROFILE_FIELDS if f == "Density" or f in avail]
        return out or ["Density"]

    def _content_image(self, data):
        """Return (PIL image, caption) for the current mode, cached."""
        from . import analysis

        scale = max(0.6, self._dpi_scale)
        sc_center, sc_radius = self._active_scope()
        sc_key = (None if sc_center is None
                  else (tuple(np.round(sc_center, 3)), round(sc_radius, 3)))
        if self.mode == "phase":
            from .analysis import PHASE_WEIGHTINGS

            if self._phase_custom is not None:
                xf, yf = self._phase_custom
            else:
                presets = self._phase_presets(data)
                self._phase_idx %= len(presets)
                xf, yf = presets[self._phase_idx]
            wgt = PHASE_WEIGHTINGS[self._phase_weight_idx
                                   % len(PHASE_WEIGHTINGS)]
            bkey = (None if self.brush_mask is None
                    else int(self.brush_mask.sum()))
            okey = (len(self.obs_datasets), self.show_precip_line,
                    self.tvir_K, self.brush_active,
                    len(self.brush_points),
                    None if self.brush_shape is None
                    else tuple(sorted(self.brush_shape.items()))
                    if self.brush_shape["kind"] != "polygon"
                    else len(self.brush_shape["points"]))
            key = ("phase", xf, yf, wgt, data.n_particles, sc_key,
                   bkey, okey)
            if key not in self._cache:
                ph = analysis.phase_histogram(
                    data, xf, yf, center=sc_center, radius_kpc=sc_radius,
                    weighting=wgt, region=self._active_region())
                self._last_phase = ph
                if self.brush_mask is not None and ph is not None:
                    sub = analysis.phase_histogram(
                        data, xf, yf, weighting=wgt,
                        region=self._active_region(),
                        brush_premask=self.brush_mask)
                    self._brush_marginals = (
                        analysis.phase_marginals(sub, 20)
                        if sub is not None else None)
                else:
                    self._brush_marginals = None
                self._cache[key] = self._render_phase(ph, scale)
            return self._cache[key], f"{xf} vs {yf}"

        if self.mode == "profile":
            fields = self._profile_fields(data)
            self._profile_idx %= len(fields)
            f = fields[self._profile_idx]
            split = (self._profile_split
                     and f not in analysis.SPECIAL_PROFILES
                     and "Temperature" in data.available_fields_with_derived())
            key = ("profile", f, split, data.n_particles, sc_key,
                   self.brush_only, self._nfw_fit)
            if key not in self._cache:
                if split:
                    r, tracks, unit = analysis.radial_profile_by_phase(
                        data, f, center=sc_center, r_max_kpc=sc_radius)
                    self._last_profile = (r, tracks, f, unit)
                    self._cache[key] = self._render_profile_tracks(
                        r, tracks, unit, f, scale)
                else:
                    bm = (self.brush_mask if self.brush_only else None)
                    r, prof, unit = analysis.radial_profile(
                        data, f, center=sc_center, r_max_kpc=sc_radius,
                        region=self._active_region(), brush_mask=bm)
                    self._last_profile = (r, prof, f, unit)
                    nfw = (self._nfw_fit if f == "Density" else None)
                    self._cache[key] = self._render_profile(
                        r, prof, unit, f, scale, nfw=nfw)
            return self._cache[key], f"{f}(r)"

        if self.mode == "orbit":
            if self._last_orbit is None:
                return None, ""
            key = ("orbit", id(self._last_orbit))
            if key not in self._cache:
                o, props = self._last_orbit
                self._cache[key] = self._render_orbit(o, props, scale)
            return self._cache[key], ""

        if self.mode == "specview":
            sls = [s for s in getattr(self, "sightlines", [])
                   if s.trident_spectrum_path
                   and __import__("os").path.exists(
                       s.trident_spectrum_path)]
            if not sls:
                return None, ""
            key = ("specview", tuple(s.label for s in sls),
                   self._spec_compare)
            if key not in self._cache:
                self._cache[key] = self._render_spectra(
                    sls if self._spec_compare else sls[-1:], scale)
            return self._cache[key], f"{len(sls)} spectra"

        if self.mode == "spectrum":
            key = ("spectrum", data.n_particles, sc_key)
            if key not in self._cache:
                from .power_spectrum import power_spectrum

                ps = power_spectrum(data, center=sc_center,
                                    radius_kpc=sc_radius)
                self._last_ps = ps
                self._cache[key] = self._render_spectrum(ps, scale)
            box = (f"box {self._last_ps['box_kpc']:.0f} kpc"
                   if self._last_ps else "")
            return self._cache[key], box
        return None, ""

    AXL, AXB, AXW, AXH = 0.16, 0.14, 0.62, 0.66  # fixed axes box

    def _render_phase(self, ph, scale):
        from matplotlib.figure import Figure
        from matplotlib.colors import LogNorm

        W = int(470 * scale)
        Hpx = int(410 * scale)
        fig = Figure(figsize=(W / 100, Hpx / 100), dpi=100)
        ax = fig.add_axes([self.AXL, self.AXB, self.AXW, self.AXH])
        if ph is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            _dark_axes(fig, ax)
            return _fig_to_image(fig)
        H = ph["H"].T
        vmax = H.max()
        with np.errstate(invalid="ignore"):
            ax.imshow(
                H, origin="lower", aspect="auto", cmap="inferno",
                norm=LogNorm(vmin=max(vmax * 1e-6, 1e-30),
                             vmax=max(vmax, 1e-29)),
                extent=[ph["xedges"][0], ph["xedges"][-1],
                        ph["yedges"][0], ph["yedges"][-1]],
                interpolation="nearest")
        ax.set_xlabel(ph["xlabel"], fontsize=10)
        ax.set_ylabel(ph["ylabel"], fontsize=10)

        # Marginal histograms (Item 2), brushed overlay in green.
        from .analysis import phase_marginals

        mx, my = phase_marginals(ph, n_bins=20)
        axt = fig.add_axes([self.AXL, self.AXB + self.AXH + 0.01,
                            self.AXW, 0.10])
        axr = fig.add_axes([self.AXL + self.AXW + 0.01, self.AXB,
                            0.10, self.AXH])
        xs = np.linspace(ph["xedges"][0], ph["xedges"][-1],
                         len(mx) + 1)
        ys = np.linspace(ph["yedges"][0], ph["yedges"][-1],
                         len(my) + 1)
        axt.bar(xs[:-1], mx, width=np.diff(xs), align="edge",
                color=(0.24, 0.49, 1.0, 0.6))
        axr.barh(ys[:-1], my, height=np.diff(ys), align="edge",
                 color=(0.24, 0.49, 1.0, 0.6))
        if self.brush_mask is not None and getattr(
                self, "_brush_marginals", None) is not None:
            bmx, bmy = self._brush_marginals
            axt.bar(xs[:-1], bmx, width=np.diff(xs), align="edge",
                    color=(0.15, 0.79, 0.48, 0.8))
            axr.barh(ys[:-1], bmy, height=np.diff(ys), align="edge",
                     color=(0.15, 0.79, 0.48, 0.8))
        for a in (axt, axr):
            a.set_xticks([])
            a.set_yticks([])
            a.set_facecolor((0, 0, 0, 0))
            for sp in a.spines.values():
                sp.set_visible(False)
        axt.set_xlim(ph["xedges"][0], ph["xedges"][-1])
        axr.set_ylim(ph["yedges"][0], ph["yedges"][-1])

        # Observational overlays (Item 3).
        obs_colors = [(0.96, 0.65, 0.14), (0.9, 0.24, 0.24),
                      (0.15, 0.79, 0.48), (0.24, 0.49, 1.0)]
        for k, (name, ox, oy, labels) in enumerate(self.obs_datasets):
            c = obs_colors[k % len(obs_colors)]
            ax.scatter(ox, oy, s=18, color=c, edgecolors="black",
                       linewidths=0.5, zorder=5,
                       label=f"Obs: {name[:18]}")
            if labels is not None:
                for xx, yy, lb in zip(ox, oy, labels):
                    ax.annotate(str(lb)[:2], (xx, yy), fontsize=6,
                                color="white",
                                xytext=(3, 3),
                                textcoords="offset points")
        # Precipitation threshold + T_vir (Item 5C) on n_H-T axes.
        if (self.show_precip_line and ph["xfield"] == "NumberDensity"
                and ph["yfield"] == "Temperature"):
            from .analysis import tcool_tff_unity_locus

            ln, lT = tcool_tff_unity_locus()
            ok = np.isfinite(lT)
            ax.plot(ln[ok], lT[ok], ls="--", lw=1.2, color="white",
                    label="t_cool/t_ff = 1")
        if self.tvir_K is not None and ph["yfield"] == "Temperature":
            yv = (np.log10(self.tvir_K) if ph["ylog"]
                  else self.tvir_K)
            ax.axhline(yv, ls="--", lw=1.2, color=(0.96, 0.65, 0.14),
                       label="T_vir")
        # Brush shape preview / committed shape.
        shp = self.brush_shape
        if shp is None and len(self.brush_points) >= 2 \
                and self.brush_kind == "polygon":
            pts = self.brush_points
            ax.plot([p_[0] for p_ in pts], [p_[1] for p_ in pts],
                    color=(0.24, 0.49, 1.0), lw=1.2)
        if shp is not None:
            acc = (0.24, 0.49, 1.0)
            if shp["kind"] == "rect":
                from matplotlib.patches import Rectangle

                ax.add_patch(Rectangle(
                    (min(shp["x0"], shp["x1"]),
                     min(shp["y0"], shp["y1"])),
                    abs(shp["x1"] - shp["x0"]),
                    abs(shp["y1"] - shp["y0"]),
                    fill=True, facecolor=acc + (0.2,),
                    edgecolor=acc, lw=1.4))
            elif shp["kind"] == "ellipse":
                from matplotlib.patches import Ellipse as _El

                ax.add_patch(_El((shp["cx"], shp["cy"]),
                                 2 * shp["rx"], 2 * shp["ry"],
                                 fill=True, facecolor=acc + (0.2,),
                                 edgecolor=acc, lw=1.4))
            elif shp["kind"] == "polygon":
                from matplotlib.patches import Polygon as _Pg

                ax.add_patch(_Pg(shp["points"], closed=True,
                                 fill=True, facecolor=acc + (0.2,),
                                 edgecolor=acc, lw=1.4))
        if (self.obs_datasets or self.show_precip_line
                or self.tvir_K is not None):
            leg = ax.legend(fontsize=6, framealpha=0.2,
                            labelcolor="white", loc="upper right")
            leg.get_frame().set_facecolor((0.1, 0.12, 0.18))
        ax.set_xlim(ph["xedges"][0], ph["xedges"][-1])
        ax.set_ylim(ph["yedges"][0], ph["yedges"][-1])
        _dark_axes(fig, ax)
        # Pixel<->data mapping for brush clicks (fixed axes => exact).
        self._phase_geom = {
            "img_wh": (W, Hpx),
            "ax_px": (self.AXL * W, (1 - self.AXB - self.AXH) * Hpx,
                      self.AXW * W, self.AXH * Hpx),
            "xlim": (ph["xedges"][0], ph["xedges"][-1]),
            "ylim": (ph["yedges"][0], ph["yedges"][-1]),
        }
        return _fig_to_image(fig)

    def _render_profile(self, r, prof, unit, field, scale, nfw=None):
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4.7 * scale, 3.9 * scale), dpi=100)
        ax = fig.add_subplot(111)
        ok = np.isfinite(prof)
        if ok.sum() > 1:
            ax.plot(r[ok], prof[ok], lw=2.0, color=(0.42, 0.72, 1.0))
            if nfw is not None:
                rho_s, r_s = nfw
                x = r[ok] / r_s
                ax.plot(r[ok], rho_s / (x * (1 + x) ** 2), ls="--",
                        lw=1.4, color="white",
                        label=f"NFW r_s={r_s:.0f} kpc")
                leg = ax.legend(fontsize=7, framealpha=0.2,
                                labelcolor="white")
                leg.get_frame().set_facecolor((0.1, 0.12, 0.18))
            ax.set_xscale("log")
            vals = prof[ok]
            if (vals > 0).all() and vals.max() / max(vals.min(), 1e-300) > 30:
                ax.set_yscale("log")
            ax.grid(alpha=0.18, which="both")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_xlabel("r [kpc]", fontsize=10)
        ax.set_ylabel(f"{field} [{unit}]" if unit else field, fontsize=10)
        _dark_axes(fig, ax)
        fig.tight_layout(pad=1.2)
        return _fig_to_image(fig)

    PHASE_COLORS = {"cold": (0.35, 0.65, 1.0), "warm": (0.4, 0.85, 0.5),
                    "warm-hot": (1.0, 0.7, 0.25), "hot": (0.95, 0.35, 0.3)}

    def _render_profile_tracks(self, r, tracks, unit, field, scale):
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4.7 * scale, 3.9 * scale), dpi=100)
        ax = fig.add_subplot(111)
        any_pos, all_pos = False, True
        for name, prof in tracks.items():
            ok = np.isfinite(prof)
            if ok.sum() > 1:
                ax.plot(r[ok], prof[ok], lw=1.8,
                        color=self.PHASE_COLORS.get(name, (0.8, 0.8, 0.8)),
                        label=name)
                any_pos = True
                vals = prof[ok]
                all_pos = all_pos and (vals > 0).all()
        if any_pos:
            ax.set_xscale("log")
            if all_pos:
                ax.set_yscale("log")
            ax.grid(alpha=0.18, which="both")
            leg = ax.legend(fontsize=8, framealpha=0.2, labelcolor="white")
            leg.get_frame().set_facecolor((0.1, 0.12, 0.18))
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_xlabel("r [kpc]", fontsize=10)
        ax.set_ylabel(f"{field} [{unit}]" if unit else field, fontsize=10)
        _dark_axes(fig, ax)
        fig.tight_layout(pad=1.2)
        return _fig_to_image(fig)

    def _render_orbit(self, orbit_result, props, scale):
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4.7 * scale, 5.2 * scale), dpi=100)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        o = orbit_result
        sc = ax1.scatter(o["pos"][:, 0], o["pos"][:, 1], c=o["t"], s=2,
                         cmap="viridis")
        ax1.scatter([o["pos"][0, 0]], [o["pos"][0, 1]], c="white", s=24,
                    marker="o", zorder=5)
        ax1.set_xlabel("x [kpc]", fontsize=9)
        ax1.set_ylabel("y [kpc]", fontsize=9)
        ax1.set_aspect("equal", adjustable="datalim")
        cb = fig.colorbar(sc, ax=ax1, pad=0.02)
        cb.set_label("t [Gyr]", color=(0.88, 0.9, 0.95), fontsize=8)
        cb.ax.tick_params(colors=(0.82, 0.85, 0.9), labelsize=7)
        ax2.plot(o["t"], o["r"], lw=1.8, color=(0.42, 0.72, 1.0))
        ax2.set_xlabel("t [Gyr]", fontsize=9)
        ax2.set_ylabel("r [kpc]", fontsize=9)
        ax2.grid(alpha=0.18)
        for ax in (ax1, ax2):
            _dark_axes(fig, ax)
        fig.tight_layout(pad=1.0)
        return _fig_to_image(fig)

    def _render_spectra(self, sightlines, scale):
        """Flux vs wavelength for one (or, in Compare mode, several)
        Trident spectra: continuum dashes at 1.0, absorption troughs
        (flux < 0.9) shaded, distinct color per sightline."""
        from matplotlib.figure import Figure

        from .spectro import load_spectrum, compare_color

        fig = Figure(figsize=(4.7 * scale, 3.4 * scale), dpi=100)
        ax = fig.add_subplot(111)
        for k, sl in enumerate(sightlines):
            try:
                w, flux = load_spectrum(sl.trident_spectrum_path)
            except Exception:
                continue
            c = compare_color(k)
            ax.plot(w, flux, lw=1.0, color=c, label=sl.label)
            ax.fill_between(w, flux, 1.0, where=flux < 0.9,
                            color=c, alpha=0.25, linewidth=0)
        # Fitted Voigt components as colored display Gaussians.
        for k, sl in enumerate(sightlines):
            comps = (sl.extra or {}).get("voigt_components") or []
            if comps:
                from .spectro import voigt_gaussian_profile

                try:
                    w0, _ = load_spectrum(sl.trident_spectrum_path)
                    wmid = float(np.median(w0))
                    v = (w0 / wmid - 1.0) * 299792.458
                    for j, (logN, b, v0) in enumerate(comps):
                        prof = voigt_gaussian_profile(v, logN, b, v0)
                        ax.plot(w0, prof, lw=1.0,
                                color=compare_color(j + 1), ls="--")
                except Exception:
                    pass
        ax.axhline(1.0, ls="--", lw=0.8, color=(0.7, 0.72, 0.78))
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("wavelength [A]", fontsize=9)
        ax.set_ylabel("F / F_continuum", fontsize=9)
        if len(sightlines) > 1:
            leg = ax.legend(fontsize=7, framealpha=0.2,
                            labelcolor="white")
            leg.get_frame().set_facecolor((0.1, 0.12, 0.18))
        _dark_axes(fig, ax)
        fig.tight_layout(pad=1.1)
        return _fig_to_image(fig)

    def _render_spectrum(self, ps, scale):
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4.7 * scale, 3.9 * scale), dpi=100)
        ax = fig.add_subplot(111)
        if ps is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        else:
            ok = np.isfinite(ps["pk"]) & (ps["pk"] > 0) & (ps["n_modes"] > 0)
            ax.loglog(ps["k"][ok], ps["pk"][ok], lw=2.0,
                      color=(0.42, 0.72, 1.0))
            ax.grid(alpha=0.18, which="both")
            ax.set_title(f"{ps['field']}  ({ps['n_grid']}^3 CIC)",
                         fontsize=9)
        ax.set_xlabel("k [1/kpc]", fontsize=10)
        ax.set_ylabel("P(k) [kpc^3]", fontsize=10)
        _dark_axes(fig, ax)
        fig.tight_layout(pad=1.2)
        return _fig_to_image(fig)

    # -- drawing ------------------------------------------------------------

    def _update_filters(self, data):
        """Custom layout for the filters tool: one card per active
        filter (field, live range, nudge buttons) + an add row."""
        s = self.style
        M, LH = s.margin, s.line_height
        self._buttons = []
        filters = data.filters
        fields = self._filter_fields(data)
        self._filter_field_idx %= max(len(fields), 1)
        cand = fields[self._filter_field_idx] if fields else "?"

        key = ("filters", tuple((f["field"], round(f["lo"], 6), round(f["hi"], 6))
                                for f in filters),
               cand, self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key

        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tw = max(s.min_width, int(430 * max(0.6, self._dpi_scale) * 1.2))
        header_h = LH + 10
        row_h = 2 * LH + 10
        th = header_h + row_h * max(len(filters), 0) + LH + 18 + M
        if not filters:
            th += LH

        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius, fill=s.bg_color,
                 outline=DarkTheme.C_255_255_255_30)
        draw.text((M, 6), "Field filters", fill=s.accent_color, font=self._font)
        cw = LH - 6
        cx0, cy0 = tw - M - cw, 5
        _rounded(draw, [(cx0, cy0), (cx0 + cw, cy0 + cw)], 6,
                 fill=DarkTheme.C_60_34_40_255, outline=DarkTheme.C_255_255_255_40)
        draw.line([(cx0 + 6, cy0 + 6), (cx0 + cw - 6, cy0 + cw - 6)],
                  fill=DarkTheme.C_235_160_160_255, width=2)
        draw.line([(cx0 + cw - 6, cy0 + 6), (cx0 + 6, cy0 + cw - 6)],
                  fill=DarkTheme.C_235_160_160_255, width=2)
        self._buttons.append((cx0, cy0, cx0 + cw, cy0 + cw, "close"))
        draw.line([(M, header_h - 2), (tw - M, header_h - 2)],
                  fill=DarkTheme.C_255_255_255_30, width=1)

        def btn(x, y, label, action, w=None, fill=None):
            bb = dummy.textbbox((0, 0), label, font=self._font)
            bw = w if w is not None else bb[2] - bb[0] + 18
            _rounded(draw, [(x, y), (x + bw, y + LH - 4)], 7,
                     fill=fill or s.slider_btn, outline=DarkTheme.C_255_255_255_45)
            draw.text((x + (bw - bb[2] + bb[0]) // 2, y - 2), label,
                      fill=s.text_color, font=self._font)
            self._buttons.append((x, y, x + bw, y + LH - 4, action))
            return bw

        y = header_h + 6
        if not filters:
            draw.text((M, y), "No filters active", fill=DarkTheme.C_168_174_188_255,
                      font=self._font)
            y += LH
        for i, f in enumerate(filters):
            draw.text((M, y), f["field"], fill=s.text_color, font=self._font)
            bx = tw - M - (LH - 4)
            btn(bx, y + 2, "x", ("f_del", i), w=LH - 4,
                fill=DarkTheme.C_60_34_40_255)
            y += LH
            bx = M
            bx += btn(bx, y + 2, "-", ("f_lo", i, -5), w=LH) + 4
            bx += btn(bx, y + 2, "+", ("f_lo", i, +5), w=LH) + 10
            rng_txt = f"{self._fmt_val(f['lo'])} .. {self._fmt_val(f['hi'])}"
            draw.text((bx, y), rng_txt, fill=DarkTheme.C_168_174_188_255,
                      font=self._font)
            bb = dummy.textbbox((0, 0), rng_txt, font=self._font)
            bx += bb[2] - bb[0] + 10
            bx += btn(bx, y + 2, "-", ("f_hi", i, -5), w=LH) + 4
            btn(bx, y + 2, "+", ("f_hi", i, +5), w=LH)
            y += LH + 10

        # Add row: cycle candidate field, then add at [5th, 95th] pct.
        bx = M
        bx += btn(bx, y + 2, "<", ("f_field", -1), w=LH + 6) + 6
        cand_w = int(tw * 0.42)
        draw.text((bx + 4, y), cand, fill=s.text_color, font=self._font)
        bx += cand_w
        bx += btn(bx, y + 2, ">", ("f_field", +1), w=LH + 6) + 10
        btn(bx, y + 2, "Add", ("f_add", cand), fill=DarkTheme.C_40_62_90_255)

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def update(self, data):
        if not self.enabled or self.mode is None:
            return
        # Field list snapshot for the X/Y cycling buttons (on_click has
        # no data handle).
        self._all_fields = data.available_fields_with_derived()
        if self.mode == "filters":
            self._update_filters(data)
            return
        from . import analysis

        s = self.style
        M, LH = s.margin, s.line_height
        self._buttons = []

        titles = {"inspector": "Particle inspector", "phase": "Phase diagram",
                  "profile": "Radial profile", "stats": "Region statistics",
                  "spectrum": "Power spectrum",
                  "orbit": "Orbit integration",
                  "sightline": "Absorption sightlines",
                  "slice": "Slice plane",
                  "isosurface": "Isosurface",
                  "streamlines": "Streamlines",
                  "volume": "Volume rendering",
                  "regions": "Regions (boolean)",
                  "halos": "Halo list",
                  "haloinspect": "Halo inspector",
                  "specview": "Spectrum viewer"}
        title = titles.get(self.mode, "")

        plot_img, caption = (None, "")
        rows = []
        if self.mode in ("phase", "profile", "spectrum", "orbit",
                         "specview"):
            plot_img, caption = self._content_image(data)
            if self.mode == "orbit" and plot_img is None:
                rows = [("No orbit yet", "Shift+click a particle,"),
                        ("", "then press Compute")]
            elif self.mode == "orbit":
                _, props = self._last_orbit
                rows = []
        elif self.mode == "inspector":
            if self._picked_index is None:
                rows = [("No particle picked", "Shift+click to pick")]
            else:
                if "inspector" not in self._cache:
                    self._cache["inspector"] = analysis.particle_summary(
                        data, self._picked_index)
                rows = self._cache["inspector"]
        elif self.mode == "orbit" and self._last_orbit is not None:
            _, props = self._last_orbit
            rows = [
                ("r_apo", f"{props['r_apo']:,.1f} kpc"),
                ("r_peri", f"{props['r_peri']:,.1f} kpc"),
                ("eccentricity", f"{props['eccentricity']:.3f}"),
                ("T_orb", f"{props['T_orb']:.2f} Gyr"),
                ("E_mean", f"{props['E_mean']:,.0f} km2/s2"),
            ]
            if props.get("circularity") is not None:
                rows.append(("circularity", f"{props['circularity']:+.2f}"))
        elif self.mode == "regions":
            rs = getattr(self, "region_set", None)
            if rs is None or not rs.entries:
                rows = [("No regions", "place aperture (M), then Add")]
            else:
                rows = []
                for i, e in enumerate(rs.entries):
                    op = "    " if i == 0 else f"{e['op']:>4s}"
                    shape = e["region"].kind[0].upper()
                    rows.append((f"{op} [{shape}] {e['name']}",
                                 "on" if e["visible"] else "off"))
        elif self.mode == "streamlines":
            st = getattr(self, "stream_state", None) or {}
            rows = [
                ("Seeds", f"{st.get('n_seeds', 256)}"
                          + (" (sphere surface)" if st.get("surface")
                             else " (random)")),
                ("Step x", f"{st.get('step_mult', 1.0):.2f}"),
                ("Max steps", f"{st.get('max_steps', 200)}"),
                ("Field", st.get("field", "Velocities")),
                ("Color by", st.get("color_by", "|v|")),
                ("Lines", f"{st.get('n_lines', 0)}"
                          + (" computing..." if st.get("busy") else "")),
            ]
        elif self.mode == "volume":
            st = getattr(self, "volume_state", None) or {}
            rows = [
                ("Mode", "MIP" if st.get("mip") else
                 "emission-absorption"),
                ("Resolution", f"{st.get('res', 128)}^3"),
                ("Step x", f"{st.get('step_mult', 1.0):.2f}"),
                ("Field", st.get("field", "Masses")),
                ("Backend", "GPU voxelize" if st.get("used_gpu")
                 else "CPU voxelize"),
                ("TF points", f"{st.get('n_tf', 3)} "
                              "(TF-/TF+ moves the knee)"),
            ]
        elif self.mode == "isosurface":
            st = getattr(self, "iso_state", None)
            if st is None or not st.get("surfaces"):
                rows = [("No surfaces", "press Add to extract"),
                        ("Field", (st or {}).get("field", "?")),
                        ("Resolution",
                         f"{(st or {}).get('res', 128)}^3")]
            else:
                rows = [("Field", st.get("field", "?")),
                        ("Resolution", f"{st.get('res', 128)}^3"),
                        ("Backend",
                         "computing..." if st.get("busy") else "ready")]
                for i, s in enumerate(st["surfaces"]):
                    rows.append((f"  S{i + 1} level",
                                 f"{s['level']:.3g} "
                                 f"(op {s['opacity']:.1f})"))
        elif self.mode == "slice":
            st = getattr(self, "slice_state", None)
            if st is None or not st.get("active"):
                rows = [("Slice inactive", "Shift+Z to activate")]
            else:
                rows = [
                    ("Normal", st.get("normal_label", "?")),
                    ("Offset", f"{st.get('offset_kpc', 0.0):+,.1f} kpc"),
                    ("Size", f"{st.get('size_kpc', 0.0):,.0f} kpc"),
                    ("Resolution", f"{st.get('res', 512)}^2"),
                    ("Opacity", f"{st.get('opacity', 0.85):.2f}"),
                    ("Backend", "GPU" if st.get("used_gpu") else "CPU"),
                    ("Drag", "Ctrl+drag moves along normal"),
                ]
        elif self.mode == "halos":
            from .catalog import (apply_halo_filter, parse_halo_filter,
                                  mass_threshold_mask, sort_halo_indices)

            if self.catalog is None or not len(self.catalog["halo_id"]):
                rows = [("No catalog", "launch with --catalog FILE")]
            else:
                cat = self.catalog
                presets = [None, "M_halo>1e12", "type=0"]
                ftxt = presets[self._halo_filter_idx % len(presets)]
                mask = (apply_halo_filter(cat, parse_halo_filter(ftxt))
                        & mass_threshold_mask(cat, self._halo_threshold))
                col, desc = self._halo_sort
                order = [i for i in sort_halo_indices(cat, col, desc)
                         if mask[i]][:20]
                self._halo_order = order
                arrow = "v" if desc else "^"
                rows = [(f"sort {col}{arrow}  filt "
                         f"{ftxt or 'none'}",
                         f"M>1e{self._halo_threshold:.0f}")]
                for i in order:
                    t = "C" if cat["type"][i] == 0 else "S"
                    rows.append(
                        (f"#{int(cat['halo_id'][i])} [{t}] "
                         f"logM={np.log10(max(cat['M_halo'][i], 1)):.1f}",
                         f"R200={cat['R_200'][i]:,.0f}"))
        elif self.mode == "haloinspect":
            cat = self.catalog
            i = self._halo_selected
            if cat is None or i is None:
                rows = [("No halo selected", "pick from the list")]
            else:
                kind = ("CENTRAL" if cat["type"][i] == 0
                        else "SATELLITE")
                rows = [
                    (f"Halo #{int(cat['halo_id'][i])}", kind),
                    ("M_halo", f"{cat['M_halo'][i]:.3e} Msun"),
                    ("M_star", f"{cat['M_star'][i]:.3e} Msun"),
                    ("SFR", f"{cat['SFR'][i]:.3g} Msun/yr"),
                    ("R_200", f"{cat['R_200'][i]:,.1f} kpc"),
                ]
        elif self.mode == "specview":
            sls = getattr(self, "sightlines", [])
            if not sls:
                rows = [("No sightlines", "Shift+A to place one")]
            else:
                rows = []
                # NB: do not name this loop variable `s` — it would
                # shadow the panel style bound above (live crash:
                # 'Sightline' object has no attribute 'min_width').
                for sl_ in sls[-3:]:
                    rows.append((sl_.label,
                                 f"b={sl_.impact_b_kpc:.0f} kpc"
                                 if sl_.impact_b_kpc is not None else ""))
                    if sl_.NHI and sl_.NHI > 0:
                        rows.append(("  log N(HI) fast",
                                     f"{np.log10(sl_.NHI):.2f} cm^-2"))
                    if (sl_.trident_spectrum_path
                            and __import__("os").path.exists(
                                sl_.trident_spectrum_path)):
                        rows.append(("  Trident", "spectrum on disk"))
        elif self.mode == "sightline":
            sls = getattr(self, "sightlines", [])
            if not sls:
                rows = [("No sightlines", "Shift+A, then click 2 points")]
            else:
                rows = []
                # Same shadowing hazard as above: keep `s` = style.
                for sl_ in sls:
                    rows.append((sl_.label,
                                 f"b={sl_.impact_b_kpc:.0f} kpc"
                                 if sl_.impact_b_kpc is not None else ""))
                    def _lg(v):
                        return (f"{np.log10(v):.2f}" if v and v > 0
                                else "-")
                    rows.append(("  log N(HI)", _lg(sl_.NHI)))
                    rows.append(("  log N(OVI)*", _lg(sl_.N_OVI)))
                    rows.append(("  log N(CIV)*", _lg(sl_.N_CIV)))
                rows.append(("* CIE approx", "use Trident for science"))
        elif self.mode == "stats":
            sc_center, sc_radius = self._active_scope()
            r_use = sc_radius if sc_radius is not None else self._stats_radius_kpc
            sck = (None if sc_center is None
                   else tuple(np.round(sc_center, 3)))
            key = ("stats", r_use, data.n_particles, sck,
                   self.brush_only)
            if key not in self._cache:
                bm = (self.brush_mask if self.brush_only else None)
                self._cache[key] = analysis.region_stats(
                    data, center=sc_center, radius_kpc=r_use,
                    region=self._active_region(), brush_mask=bm)
            rows, used_r = self._cache[key]
            if sc_radius is None:
                self._stats_radius_kpc = used_r
            if self._show_halo:
                hkey = ("halo",) + key[1:]
                if hkey not in self._cache:
                    try:
                        self._cache[hkey] = analysis.halo_properties(
                            data, center=sc_center, radius_kpc=used_r)
                    except Exception as e:
                        self._cache[hkey] = [("Halo props failed", str(e)[:40])]
                rows = list(rows) + [("--- halo ---", "")] + self._cache[hkey]

        scope_tag = ""
        if self.scope is not None:
            scope_tag = (f"aperture R={self.scope['radius_kpc']:.0f} kpc "
                         f"[{self.scope['center_mode']}]"
                         if self.use_scope else "global")
        key = (self.mode, caption, tuple(rows), self._fb_width,
               self._fb_height, id(plot_img), scope_tag)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key

        # Measure
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tw = s.min_width
        if plot_img is not None:
            tw = max(tw, plot_img.width + M * 2)
        kv_w = 0
        for a, b in rows:
            ba = dummy.textbbox((0, 0), str(a), font=self._font)
            bb = dummy.textbbox((0, 0), str(b), font=self._font)
            kv_w = max(kv_w, ba[2] - ba[0])
            tw = max(tw, ba[2] - ba[0] + bb[2] - bb[0] + 40 + M * 2)
        header_h = LH + 10
        footer_h = (2 * LH + 8 if self.mode == "phase"
                    else LH + 8 if self.mode in ("profile", "stats")
                    else 6)
        body_h = (plot_img.height + 8 + LH * len(rows)
                  if plot_img is not None
                  else LH * max(len(rows), 1) + 8)
        extra_h = LH if self.mode == "inspector" and self._picked_index is not None else 0
        scope_h = LH + 4 if self.scope is not None else 0
        th = header_h + body_h + footer_h + extra_h + scope_h + M

        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius, fill=s.bg_color,
                 outline=DarkTheme.C_255_255_255_30)

        # Header: title + close box
        draw.text((M, 6), title, fill=s.accent_color, font=self._font)
        cw = LH - 6
        cx0, cy0 = tw - M - cw, 5
        _rounded(draw, [(cx0, cy0), (cx0 + cw, cy0 + cw)], 6,
                 fill=DarkTheme.C_60_34_40_255, outline=DarkTheme.C_255_255_255_40)
        draw.line([(cx0 + 6, cy0 + 6), (cx0 + cw - 6, cy0 + cw - 6)],
                  fill=DarkTheme.C_235_160_160_255, width=2)
        draw.line([(cx0 + cw - 6, cy0 + 6), (cx0 + 6, cy0 + cw - 6)],
                  fill=DarkTheme.C_235_160_160_255, width=2)
        self._buttons.append((cx0, cy0, cx0 + cw, cy0 + cw, "close"))
        draw.line([(M, header_h - 2), (tw - M, header_h - 2)],
                  fill=DarkTheme.C_255_255_255_30, width=1)

        y = header_h + 4
        if plot_img is not None:
            self._plot_img_pos = ((tw - plot_img.width) // 2, y)
            img.alpha_composite(plot_img, ((tw - plot_img.width) // 2, y))
            y += plot_img.height + 4
            for a, b in rows:
                draw.text((M, y + 2), str(a), fill=DarkTheme.C_168_174_188_255,
                          font=self._font)
                draw.text((M + kv_w + 28, y + 2), str(b), fill=s.text_color,
                          font=self._font)
                y += LH
        else:
            for a, b in rows:
                draw.text((M, y + 2), str(a), fill=DarkTheme.C_168_174_188_255,
                          font=self._font)
                draw.text((M + kv_w + 28, y + 2), str(b), fill=s.text_color,
                          font=self._font)
                y += LH
            y += 8

        # Inspector extra action: center the view on the picked particle
        if self.mode == "inspector" and self._picked_index is not None:
            bw = 0
            label = "Center view here"
            bbox = dummy.textbbox((0, 0), label, font=self._font)
            bw = bbox[2] - bbox[0] + 24
            bx0 = M
            _rounded(draw, [(bx0, y), (bx0 + bw, y + LH - 2)], 8,
                     fill=DarkTheme.C_40_62_90_255, outline=DarkTheme.C_255_255_255_45)
            draw.text((bx0 + 12, y + 1), label, fill=s.text_color,
                      font=self._font)
            self._buttons.append((bx0, y, bx0 + bw, y + LH - 2, "center_on_pick"))
            y += LH

        # Scope row: aperture/global toggle + center-mode cycling.
        if self.scope is not None:
            bx = M
            lbl = "Aperture" if self.use_scope else "Global"
            bb = dummy.textbbox((0, 0), lbl, font=self._font)
            bw = bb[2] - bb[0] + 20
            fill = DarkTheme.C_40_62_90_255 if self.use_scope else s.slider_btn
            _rounded(draw, [(bx, y + 2), (bx + bw, y + LH - 2)], 8,
                     fill=fill, outline=DarkTheme.C_255_255_255_45)
            draw.text((bx + 10, y), lbl, fill=s.text_color, font=self._font)
            self._buttons.append((bx, y + 2, bx + bw, y + LH - 2, "toggle_scope"))
            bx += bw + 8
            if self.use_scope:
                lbl2 = self.scope["center_mode"]
                bb = dummy.textbbox((0, 0), lbl2, font=self._font)
                bw2 = bb[2] - bb[0] + 20
                _rounded(draw, [(bx, y + 2), (bx + bw2, y + LH - 2)], 8,
                         fill=s.slider_btn, outline=DarkTheme.C_255_255_255_45)
                draw.text((bx + 10, y), lbl2, fill=s.text_color, font=self._font)
                self._buttons.append((bx, y + 2, bx + bw2, y + LH - 2,
                                      "cycle_center"))
                bx += bw2 + 8
                draw.text((bx + 4, y + 1),
                          f"R={self.scope['radius_kpc']:.0f} kpc",
                          fill=DarkTheme.C_168_174_188_255, font=self._font)
            y += LH + 4

        # Footer controls
        def fbtn(bx, lbl, action, active=False):
            bb = dummy.textbbox((0, 0), lbl, font=self._font)
            bw = max(bb[2] - bb[0] + 16, LH + 8)
            fill = DarkTheme.C_40_62_90_255 if active else s.slider_btn
            _rounded(draw, [(bx, y + 2), (bx + bw, y + LH - 2)], 8,
                     fill=fill, outline=DarkTheme.C_255_255_255_45)
            draw.text((bx + (bw - bb[2] + bb[0]) // 2, y), lbl,
                      fill=s.text_color, font=self._font)
            self._buttons.append((bx, y + 2, bx + bw, y + LH - 2, action))
            return bx + bw + 6

        if self.mode == "profile":
            bx = M
            bx = fbtn(bx, "<", "prev")
            bx = fbtn(bx, ">", "next")
            bx = fbtn(bx, "Split", "profile_split", active=self._profile_split)
            bx = fbtn(bx, "CSV", "profile_csv")
            bx = fbtn(bx, "NFW", "profile_nfw",
                      active=self._nfw_fit is not None)
            if self.brush_mask is not None:
                bx = fbtn(bx, "BrushOnly", "brush_only",
                          active=self.brush_only)
            draw.text((bx + 4, y + 1), caption, fill=DarkTheme.C_168_174_188_255,
                      font=self._font)
        elif self.mode == "phase":
            from .analysis import PHASE_WEIGHTINGS

            wgt = PHASE_WEIGHTINGS[self._phase_weight_idx
                                   % len(PHASE_WEIGHTINGS)]
            bx = M
            bx = fbtn(bx, "<", "prev")
            bx = fbtn(bx, ">", "next")
            bx = fbtn(bx, "X>", "phase_x", active=self._phase_custom is not None)
            bx = fbtn(bx, "Y>", "phase_y", active=self._phase_custom is not None)
            bx = fbtn(bx, f"W:{wgt}", "phase_w")
            bx = fbtn(bx, "Save", "phase_save")
            y += LH
            self._panel_h_extra = LH
            bx = M
            bx = fbtn(bx, "Brush", "brush_toggle",
                      active=self.brush_active)
            bx = fbtn(bx, self.brush_kind[:4], "brush_kind")
            if self.brush_kind == "polygon" and self.brush_points:
                bx = fbtn(bx, "Close", "brush_close")
            if self.brush_mask is not None:
                bx = fbtn(bx, "Clear", "brush_clear")
            bx = fbtn(bx, "Obs+", "obs_add")
            if self.obs_datasets:
                bx = fbtn(bx, "Obs-", "obs_clear")
            bx = fbtn(bx, "tc/tff", "precip_line",
                      active=self.show_precip_line)
            bx = fbtn(bx, "Tvir", "tvir_line",
                      active=self.tvir_K is not None)
        elif self.mode == "orbit":
            bx = M
            bx = fbtn(bx, "Compute", "orbit_compute")
            bx = fbtn(bx, "CSV", "orbit_csv")
            bx = fbtn(bx, "Stream", "orbit_stream")
            bx = fbtn(bx, "galpy", "orbit_galpy")
        elif self.mode == "regions":
            bx = M
            bx = fbtn(bx, "Add", "rg_add")
            bx = fbtn(bx, "Op", "rg_op")
            bx = fbtn(bx, "Del", "rg_del")
            bx = fbtn(bx, "Submit", "rg_submit")
            bx = fbtn(bx, "Save", "rg_save")
            bx = fbtn(bx, "Load", "rg_load")
        elif self.mode == "streamlines":
            bx = M
            bx = fbtn(bx, "Go", "sl_compute")
            bx = fbtn(bx, "N-", "sl_seeds_down")
            bx = fbtn(bx, "N+", "sl_seeds_up")
            bx = fbtn(bx, "Step", "sl_step")
            bx = fbtn(bx, "Field", "sl_field")
            bx = fbtn(bx, "Surf", "sl_surface")
        elif self.mode == "volume":
            bx = M
            bx = fbtn(bx, "Go", "vol_compute")
            bx = fbtn(bx, "Res", "vol_res")
            bx = fbtn(bx, "MIP", "vol_mip")
            bx = fbtn(bx, "St-", "vol_step_down")
            bx = fbtn(bx, "St+", "vol_step_up")
            bx = fbtn(bx, "TF-", "vol_tf_down")
            bx = fbtn(bx, "TF+", "vol_tf_up")
        elif self.mode == "isosurface":
            bx = M
            bx = fbtn(bx, "Add", "iso_add")
            bx = fbtn(bx, "Lv-", "iso_down")
            bx = fbtn(bx, "Lv+", "iso_up")
            bx = fbtn(bx, "Res", "iso_res")
            bx = fbtn(bx, "Op", "iso_op")
            bx = fbtn(bx, "OBJ", "iso_obj")
            bx = fbtn(bx, "Clear", "iso_clear")
        elif self.mode == "slice":
            bx = M
            bx = fbtn(bx, "Axis", "slice_axis")
            bx = fbtn(bx, "-", "slice_back")
            bx = fbtn(bx, "+", "slice_fwd")
            bx = fbtn(bx, "Op-", "slice_op_down")
            bx = fbtn(bx, "Op+", "slice_op_up")
            bx = fbtn(bx, "Res", "slice_res")
        elif self.mode == "halos":
            bx = M
            bx = fbtn(bx, "Sort", "halo_sort")
            bx = fbtn(bx, "Filt", "halo_filter")
            bx = fbtn(bx, "T-", "halo_thr_down")
            bx = fbtn(bx, "T+", "halo_thr_up")
            bx = fbtn(bx, "CSV", "halo_csv")
        elif self.mode == "haloinspect":
            bx = M
            bx = fbtn(bx, "Fly to", "halo_fly")
            bx = fbtn(bx, "Set aperture", "halo_aperture")
            bx = fbtn(bx, "Profile", "halo_profile")
            bx = fbtn(bx, "Back", "halo_back")
        elif self.mode == "specview":
            bx = M
            bx = fbtn(bx, "Compare", "spec_compare",
                      active=self._spec_compare)
            bx = fbtn(bx, "Voigt", "spec_voigt")
            bx = fbtn(bx, "PDF", "spec_pdf")
        elif self.mode == "sightline":
            bx = M
            bx = fbtn(bx, "CSV", "sightline_csv")
            bx = fbtn(bx, "View", "spec_open")
            bx = fbtn(bx, "Trident", "sightline_trident")
            bx = fbtn(bx, "Clear", "sightline_clear")
        elif self.mode == "spectrum":
            bx = M
            bx = fbtn(bx, "CSV", "spectrum_csv")
            draw.text((bx + 6, y + 1), caption, fill=DarkTheme.C_168_174_188_255,
                      font=self._font)
        elif self.mode == "stats":
            bx = M
            bx = fbtn(bx, "R/2", "r_half")
            bx = fbtn(bx, "Rx2", "r_double")
            bx = fbtn(bx, "Halo", "stats_halo", active=self._show_halo)
            bx = fbtn(bx, "JSON", "stats_json")
            bx = fbtn(bx, "TeX", "stats_latex")
            bx = fbtn(bx, "Clip", "stats_clip")
            bx = fbtn(bx, "Diff", "stats_compare")
            if self.brush_mask is not None:
                bx = fbtn(bx, "BrushOnly", "brush_only",
                          active=self.brush_only)

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def phase_click_to_data(self, lx, ly):
        """Panel-local click -> phase data coords (or None outside the
        axes box). Uses the fixed-fraction axes geometry recorded by
        _render_phase plus the plot image's paste position."""
        g = self._phase_geom
        pi = getattr(self, "_plot_img_pos", None)
        if g is None or pi is None:
            return None
        px = lx - pi[0]
        py = ly - pi[1]
        ax_x, ax_y, ax_w, ax_h = g["ax_px"]
        if not (ax_x <= px <= ax_x + ax_w and ax_y <= py <= ax_y + ax_h):
            return None
        fx = (px - ax_x) / ax_w
        fy = 1.0 - (py - ax_y) / ax_h
        x0, x1 = g["xlim"]
        y0, y1 = g["ylim"]
        return (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))

    def brush_click(self, lx, ly):
        """Handle a click in brush mode; returns "submit" when a shape
        is completed (rect/ellipse two-click), True when consumed."""
        pt = self.phase_click_to_data(lx, ly)
        if pt is None:
            return False
        self.brush_points.append(pt)
        if self.brush_kind in ("rect", "ellipse") \
                and len(self.brush_points) >= 2:
            (xa, ya), (xb, yb) = self.brush_points[:2]
            if self.brush_kind == "rect":
                self.brush_shape = {"kind": "rect", "x0": xa, "x1": xb,
                                    "y0": ya, "y1": yb}
            else:
                self.brush_shape = {"kind": "ellipse", "cx": xa,
                                    "cy": ya, "rx": abs(xb - xa),
                                    "ry": abs(yb - ya)}
            self.brush_points = []
            self.refresh()
            return "submit"
        self.refresh()
        return True

    def close_polygon(self):
        if self.brush_kind == "polygon" and len(self.brush_points) >= 3:
            self.brush_shape = {"kind": "polygon",
                                "points": list(self.brush_points)}
            self.brush_points = []
            self.refresh()
            return "submit"
        return False

    def on_click(self, x, y):
        """Returns an action string handled by the app, True if consumed,
        or False if the click missed the drawer."""
        if not self.enabled:
            return False
        lx, ly = x - self._panel_x, y - self._panel_y
        if lx < 0 or lx > self._panel_w or ly < 0 or ly > self._panel_h:
            return False
        # Brush mode: clicks inside the phase plot place shape points.
        if (self.mode == "phase" and self.brush_active):
            r = self.brush_click(lx, ly)
            if r:
                return ("brush_submit" if r == "submit" else True)
        # Halo list: clicking a table row opens the inspector.
        if (self.mode == "halos" and self.catalog is not None
                and getattr(self, "_halo_order", None)):
            LH = self.style.line_height
            header_h = LH + 10
            row0 = header_h + 4 + LH  # skip the sort/filter status row
            idx = int((ly - row0) // LH)
            if 0 <= idx < len(self._halo_order) and ly >= row0:
                in_footer = ly > self._panel_h - LH - 8
                if not in_footer:
                    self._halo_selected = int(self._halo_order[idx])
                    self.mode = "haloinspect"
                    self._last_key = None
                    return True
        for x0, y0, x1, y1, action in self._buttons:
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                if action == "close":
                    self.enabled = False
                    self.mode = None
                    return True
                if action == "prev":
                    if self.mode == "phase":
                        self._phase_idx -= 1
                        self._phase_custom = None
                    else:
                        self._profile_idx -= 1
                    self._last_key = None
                    return True
                if action == "next":
                    if self.mode == "phase":
                        self._phase_idx += 1
                        self._phase_custom = None
                    else:
                        self._profile_idx += 1
                    self._last_key = None
                    return True
                if action == "r_half":
                    self._stats_radius_kpc = (self._stats_radius_kpc or 100.0) / 2.0
                    if self.scope is not None and self.use_scope:
                        self.scope["radius_kpc"] /= 2.0
                    self.refresh()
                    return True
                if action == "r_double":
                    self._stats_radius_kpc = (self._stats_radius_kpc or 100.0) * 2.0
                    if self.scope is not None and self.use_scope:
                        self.scope["radius_kpc"] *= 2.0
                    self.refresh()
                    return True
                if (self.mode == "halos"
                        and isinstance(action, str)
                        and action == "close"):
                    pass  # fall through to standard close below
                if action == "halo_sort":
                    cols = ["M_halo", "M_star", "R_200", "halo_id"]
                    col, desc = self._halo_sort
                    if desc:
                        self._halo_sort = (col, False)
                    else:
                        self._halo_sort = (
                            cols[(cols.index(col) + 1) % len(cols)], True)
                    self._last_key = None
                    return True
                if action == "halo_filter":
                    self._halo_filter_idx += 1
                    self._last_key = None
                    return True
                if action in ("halo_thr_down", "halo_thr_up"):
                    self._halo_threshold = float(np.clip(
                        self._halo_threshold
                        + (0.5 if action == "halo_thr_up" else -0.5),
                        10.0, 14.0))
                    self._last_key = None
                    return True
                if action == "halo_back":
                    self.mode = "halos"
                    self._last_key = None
                    return True
                if action == "spec_compare":
                    self._spec_compare = not self._spec_compare
                    self.refresh()
                    return True
                if action == "spec_open":
                    self.mode = "specview"
                    self.refresh()
                    return True
                if action == "toggle_scope":
                    self.use_scope = not self.use_scope
                    self.refresh()
                    return True
                if action == "profile_split":
                    self._profile_split = not self._profile_split
                    self._last_key = None
                    return True
                if action in ("phase_x", "phase_y"):
                    fields = getattr(self, "_all_fields", None) or ["Masses"]
                    if self._phase_custom is None:
                        # Seed custom mode from a sane default pair.
                        self._phase_custom = [fields[0],
                                              fields[min(1, len(fields) - 1)]]
                    i = 0 if action == "phase_x" else 1
                    cur = self._phase_custom[i]
                    j = (fields.index(cur) + 1) % len(fields) \
                        if cur in fields else 0
                    self._phase_custom[i] = fields[j]
                    self._last_key = None
                    return True
                if action == "brush_toggle":
                    self.brush_active = not self.brush_active
                    self.brush_points = []
                    self._last_key = None
                    return True
                if action == "brush_kind":
                    kinds = ["rect", "polygon", "ellipse"]
                    self.brush_kind = kinds[
                        (kinds.index(self.brush_kind) + 1) % 3]
                    self.brush_points = []
                    self._last_key = None
                    return True
                if action == "brush_close":
                    return ("brush_submit" if self.close_polygon()
                            else True)
                if action == "precip_line":
                    self.show_precip_line = not self.show_precip_line
                    self.refresh()
                    return True
                if action == "phase_w":
                    self._phase_weight_idx += 1
                    self._last_key = None
                    return True
                if action == "brush_only":
                    self.brush_only = not self.brush_only
                    self.refresh()
                    return True
                if action == "stats_halo":
                    self._show_halo = not self._show_halo
                    self._last_key = None
                    return True
                if (isinstance(action, tuple) and action
                        and action[0] == "f_field"):
                    self._filter_field_idx += action[1]
                    self._last_key = None
                    return True
                return action
        return True

    def render(self):
        if not self.enabled:
            return
        super().render()


# ---------------------------------------------------------------------------
# Colormap browser (Section 6.G) + visual field picker (Section 6.F)
# ---------------------------------------------------------------------------

CMAP_CATEGORIES = {
    "Perceptual": ["viridis", "plasma", "inferno", "magma", "cividis",
                   "cmr.rainforest", "cmr.ember", "cmr.cosmic"],
    "Diverging": ["RdBu_r", "coolwarm", "bwr", "seismic",
                  "cmr.fusion", "cmr.iceburn", "cmr.redshift"],
    "Sequential": ["Blues", "Reds", "Greens", "YlOrRd", "hot",
                   "cmr.freeze", "cmr.sunburst"],
    "Cyclic": ["hsv", "twilight", "twilight_shifted"],
    "CMasher": [],  # filled lazily from the cmasher module
}


def _cmasher_names():
    try:
        import cmasher

        return [f"cmr.{m}" for m in cmasher.cm.cmap_d
                if not m.endswith("_r")]
    except Exception:
        return []


class ColormapBrowserPanel(Panel):
    """Tabbed colormap browser: gradient swatches per category,
    Reversed toggle, click applies immediately (the app handles the
    returned ('apply_cmap', name) action)."""

    COLS = 3
    SW_W, SW_H = 100, 12

    def __init__(self):
        super().__init__(DRAWER_STYLE)
        self.enabled = False
        self.tab = "Perceptual"
        self.reversed = False
        self.active_cmap = "magma"
        self._buttons = []
        self._last_key = None
        if not CMAP_CATEGORIES["CMasher"]:
            CMAP_CATEGORIES["CMasher"] = _cmasher_names()[:24]

    def names_for_tab(self, tab=None):
        names = list(CMAP_CATEGORIES.get(tab or self.tab, []))
        if self.reversed:
            # Toggle the _r suffix rather than stacking it (RdBu_r's
            # reverse is RdBu, not the invalid RdBu_r_r).
            names = [n[:-2] if n.endswith("_r") else n + "_r"
                     for n in names]
        return names

    def _swatch(self, name):
        import matplotlib

        try:
            cmap = matplotlib.colormaps[name]
        except KeyError:
            return None
        x = np.linspace(0, 1, self.SW_W)
        rgba = (np.asarray(cmap(x)) * 255).astype(np.uint8)
        return np.repeat(rgba[None, :, :], self.SW_H, axis=0)

    def update(self, _data=None):
        if not self.enabled:
            return
        s = self.style
        M, LH = s.margin, s.line_height
        names = self.names_for_tab()
        key = (self.tab, self.reversed, self.active_cmap,
               self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key
        self._buttons = []

        cell_w = self.SW_W + 16
        cell_h = self.SW_H + LH
        rows = (len(names) + self.COLS - 1) // self.COLS
        tw = max(self.COLS * cell_w + 2 * M, 380)
        th = LH + 10 + LH + rows * cell_h + M * 2

        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius,
                 fill=s.bg_color, outline=DarkTheme.C_255_255_255_30)
        draw.text((M, 6), "Colormaps", fill=s.accent_color,
                  font=self._font)
        cw = LH - 6
        cx0 = tw - M - cw
        _rounded(draw, [(cx0, 5), (cx0 + cw, 5 + cw)], 6,
                 fill=DarkTheme.C_60_34_40_255)
        draw.line([(cx0 + 6, 11), (cx0 + cw - 6, cw - 1)],
                  fill=DarkTheme.C_235_160_160_255, width=2)
        draw.line([(cx0 + cw - 6, 11), (cx0 + 6, cw - 1)],
                  fill=DarkTheme.C_235_160_160_255, width=2)
        self._buttons.append((cx0, 5, cx0 + cw, 5 + cw, "close"))

        # Tabs + reversed toggle
        y = LH + 8
        bx = M
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        for tab in CMAP_CATEGORIES:
            bb = dummy.textbbox((0, 0), tab, font=self._font)
            bw = bb[2] - bb[0] + 14
            fill = (DarkTheme.ACCENT_DIM if tab == self.tab
                    else s.slider_btn)
            _rounded(draw, [(bx, y), (bx + bw, y + LH - 6)], 7,
                     fill=fill)
            draw.text((bx + 7, y - 2), tab, fill=s.text_color,
                      font=self._font)
            self._buttons.append((bx, y, bx + bw, y + LH - 6,
                                  ("tab", tab)))
            bx += bw + 5
        lbl = "rev" if not self.reversed else "REV"
        bb = dummy.textbbox((0, 0), lbl, font=self._font)
        bw = bb[2] - bb[0] + 14
        _rounded(draw, [(tw - M - bw, y), (tw - M, y + LH - 6)], 7,
                 fill=(DarkTheme.ACCENT_DIM if self.reversed
                       else s.slider_btn))
        draw.text((tw - M - bw + 7, y - 2), lbl, fill=s.text_color,
                  font=self._font)
        self._buttons.append((tw - M - bw, y, tw - M, y + LH - 6,
                              "reverse"))
        y += LH + 4

        for i, name in enumerate(names):
            col, row = i % self.COLS, i // self.COLS
            x0 = M + col * cell_w
            y0 = y + row * cell_h
            sw = self._swatch(name)
            if sw is not None:
                img.paste(Image.fromarray(sw, "RGBA"), (x0, y0))
            border = (DarkTheme.ACCENT if name == self.active_cmap
                      else DarkTheme.C_255_255_255_50)
            draw.rectangle([(x0 - 1, y0 - 1),
                            (x0 + self.SW_W, y0 + self.SW_H)],
                           outline=border)
            short = name if len(name) <= 16 else name[:15] + "…"
            draw.text((x0, y0 + self.SW_H + 1), short,
                      fill=DarkTheme.C_168_174_188_255, font=self._font)
            self._buttons.append((x0, y0, x0 + self.SW_W,
                                  y0 + self.SW_H + LH,
                                  ("apply_cmap", name)))

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def on_click(self, x, y):
        if not self.enabled:
            return False
        lx, ly = x - self._panel_x, y - self._panel_y
        if not (0 <= lx <= self._panel_w and 0 <= ly <= self._panel_h):
            return False
        for x0, y0, x1, y1, action in self._buttons:
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                if action == "close":
                    self.enabled = False
                    return True
                if action == "reverse":
                    self.reversed = not self.reversed
                    self._last_key = None
                    return True
                if isinstance(action, tuple) and action[0] == "tab":
                    self.tab = action[1]
                    self._last_key = None
                    return True
                if isinstance(action, tuple) and action[0] == "apply_cmap":
                    self.active_cmap = action[1]
                    self._last_key = None
                    return action
                return True
        return True

    def render(self):
        if not self.enabled:
            return
        super().render()


FIELD_CATEGORIES = {
    "HYDRO": ["Density", "Masses"],
    "THERMAL": ["Temperature", "Pressure", "Entropy", "CoolingTime",
                "FreeFallTime", "TcoolOverTff", "SoundSpeed"],
    "KINEMATIC": ["VelocityMagnitude", "RadialVelocity",
                  "SpecificAngularMomentum", "AngularMomentumZ",
                  "MachNumber"],
    "CHEMICAL": ["MetallicityZsun", "OxygenAbundance",
                 "MagnesiumAbundance", "IronAbundance",
                 "AlphaEnhancement"],
    "MAGNETIC": ["MagneticFieldMagnitude", "AlfvenSpeed", "AlfvenMach",
                 "PlasmaBeta"],
    "STELLAR": ["StellarAge", "HIDensity", "JeansLength", "JeansMass"],
}


class FieldPickerPanel(Panel):
    """Searchable, categorized field picker (Section 6.F).

    The app feeds available fields via set_fields(); type-to-search
    filters case-insensitively; clicking a row returns
    ('pick_field', name) for the app to apply to the active dropdown
    target. Recent picks persist for the session.
    """

    def __init__(self):
        super().__init__(DRAWER_STYLE)
        self.enabled = False
        self.search = ""
        self.target = "_sd_field"   # which app state key receives the pick
        self.recent = []
        self._fields = []
        self._buttons = []
        self._last_key = None

    def set_fields(self, fields):
        self._fields = list(fields)

    def filtered(self):
        """(category, [names]) honoring the search filter; RAW collects
        fields not claimed by any category."""
        q = self.search.lower()
        claimed = set()
        out = []
        if self.recent and not q:
            out.append(("RECENT", [f for f in self.recent
                                   if f in self._fields][:6]))
        for cat, names in FIELD_CATEGORIES.items():
            hits = [n for n in names
                    if n in self._fields and q in n.lower()]
            claimed.update(names)
            if hits:
                out.append((cat, hits))
        raw = [n for n in self._fields
               if n not in claimed and q in n.lower()]
        if raw:
            out.append(("RAW", raw))
        return out

    def on_char(self, char):
        if not self.enabled:
            return False
        self.search += char
        self._last_key = None
        return True

    def on_backspace(self):
        if self.enabled and self.search:
            self.search = self.search[:-1]
            self._last_key = None
            return True
        return False

    def update(self, _data=None):
        if not self.enabled:
            return
        s = self.style
        M, LH = s.margin, s.line_height
        groups = self.filtered()
        key = (self.search, tuple((c, tuple(n)) for c, n in groups),
               self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key
        self._buttons = []

        n_rows = sum(1 + len(names) for _, names in groups)
        tw = 360
        th = min(LH * (n_rows + 2) + 3 * M, 720)
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius,
                 fill=s.bg_color, outline=DarkTheme.C_255_255_255_30)
        hint = self.search or "type to search..."
        draw.text((M, 6), f"Field picker: {hint}",
                  fill=s.accent_color, font=self._font)
        cw = LH - 6
        cx0 = tw - M - cw
        _rounded(draw, [(cx0, 5), (cx0 + cw, 5 + cw)], 6,
                 fill=DarkTheme.C_60_34_40_255)
        self._buttons.append((cx0, 5, cx0 + cw, 5 + cw, "close"))

        y = LH + 8
        for cat, names in groups:
            if y > th - LH:
                break
            draw.text((M, y), cat, fill=DarkTheme.TEXT_SECONDARY,
                      font=self._font)
            y += LH
            for n in names:
                if y > th - LH:
                    break
                draw.text((M + 16, y), n, fill=s.text_color,
                          font=self._font)
                # Formula/description tooltip for derived fields,
                # right-aligned and dimmed (Section 6.F).
                from .physics import DERIVED_FIELDS

                df = DERIVED_FIELDS.get(n)
                if df is not None:
                    desc = df.description
                    if len(desc) > 30:
                        desc = desc[:29] + "…"
                    dummy2 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
                    bb = dummy2.textbbox((0, 0), desc, font=self._font)
                    draw.text((tw - M - (bb[2] - bb[0]), y), desc,
                              fill=DarkTheme.TEXT_DISABLED,
                              font=self._font)
                self._buttons.append((M, y, tw - M, y + LH,
                                      ("pick_field", n)))
                y += LH

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def on_click(self, x, y):
        if not self.enabled:
            return False
        lx, ly = x - self._panel_x, y - self._panel_y
        if not (0 <= lx <= self._panel_w and 0 <= ly <= self._panel_h):
            return False
        for x0, y0, x1, y1, action in self._buttons:
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                if action == "close":
                    self.enabled = False
                    return True
                if isinstance(action, tuple) and action[0] == "pick_field":
                    name = action[1]
                    self.recent = ([name] + [r for r in self.recent
                                             if r != name])[:6]
                    self.enabled = False
                    return action
                return True
        return True

    def render(self):
        if not self.enabled:
            return
        super().render()


# ---------------------------------------------------------------------------
# Right analysis dock (Section 6.H)
# ---------------------------------------------------------------------------

DOCK_SECTIONS = [
    ("INSPECTOR", "inspector", "I"),
    ("PHASE DIAGRAM", "phase", "G"),
    ("RADIAL PROFILE", "profile", "J"),
    ("REGION STATS", "stats", "U"),
    ("ORBIT", "orbit", "O"),
    ("SIGHTLINES", "sightline", "sA"),
    ("POWER SPECTRUM", "spectrum", "sK"),
]


class RightDock(Panel):
    """Accordion access point for every analysis tool (Section 6.H).

    Clicking a section header opens that tool's floating drawer (the
    drawer remains the single rendering surface — per spec, the dock
    is an always-visible alternative access point). The expanded
    section mirrors drawer.mode; opening one collapses the others by
    construction. A non-collapsible QUICK STATS footer shows the five
    most recently computed headline values.
    """

    WIDTH = 300

    def __init__(self):
        super().__init__(DRAWER_STYLE)
        self.enabled = False
        self.style = PanelStyle(**{**DRAWER_STYLE.__dict__,
                                   "position": "top-right",
                                   "min_width": self.WIDTH})
        self.anchor_offset = (0, 34)  # below the menu bar
        from collections import deque

        self.quick_stats = deque(maxlen=5)
        self._buttons = []
        self._last_key = None

    def push_stat(self, label, value):
        self.quick_stats.appendleft((label, value))
        self._last_key = None

    def update(self, active_mode=None):
        if not self.enabled:
            return
        s = self.style
        M, LH = s.margin, s.line_height
        key = (active_mode, tuple(self.quick_stats),
               self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key
        self._buttons = []

        n_rows = len(DOCK_SECTIONS) + 2 + max(len(self.quick_stats), 1)
        tw = self.WIDTH
        th = LH * n_rows + 3 * M
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius,
                 fill=s.bg_color, outline=DarkTheme.C_255_255_255_30)
        draw.text((M, 6), "Analysis", fill=s.accent_color,
                  font=self._font)
        y = LH + 6
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        for name, mode, badge in DOCK_SECTIONS:
            active = mode == active_mode
            if active:
                _rounded(draw, [(4, y), (tw - 4, y + LH - 2)], 8,
                         fill=DarkTheme.ACCENT_DIM)
            caret = "v" if active else ">"
            draw.text((M, y), f"{caret} {name}",
                      fill=s.text_color if active
                      else DarkTheme.C_168_174_188_255, font=self._font)
            bb = dummy.textbbox((0, 0), badge, font=self._font)
            _rounded(draw, [(tw - M - (bb[2] - bb[0]) - 12, y + 2),
                            (tw - M, y + LH - 4)], 6,
                     fill=s.slider_btn)
            draw.text((tw - M - (bb[2] - bb[0]) - 6, y), badge,
                      fill=s.text_color, font=self._font)
            self._buttons.append((0, y, tw, y + LH, ("dock", mode)))
            y += LH
        # QUICK STATS footer
        y += 4
        draw.line([(M, y), (tw - M, y)], fill=DarkTheme.C_255_255_255_30)
        draw.text((M, y + 2), "QUICK STATS",
                  fill=DarkTheme.TEXT_SECONDARY, font=self._font)
        y += LH
        if not self.quick_stats:
            draw.text((M, y), "(nothing computed yet)",
                      fill=DarkTheme.TEXT_DISABLED, font=self._font)
            y += LH
        for label, value in self.quick_stats:
            draw.text((M, y), label, fill=DarkTheme.C_168_174_188_255,
                      font=self._font)
            bb = dummy.textbbox((0, 0), str(value), font=self._font)
            draw.text((tw - M - (bb[2] - bb[0]), y), str(value),
                      fill=s.text_color, font=self._font)
            y += LH

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def on_click(self, x, y):
        if not self.enabled:
            return False
        lx, ly = x - self._panel_x, y - self._panel_y
        if not (0 <= lx <= self._panel_w and 0 <= ly <= self._panel_h):
            return False
        for x0, y0, x1, y1, action in self._buttons:
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                return action
        return True

    def render(self):
        if not self.enabled:
            return
        super().render()


VIZMO_LOGO = [
 "██╗   ██╗██╗███████╗███╗   ███╗ ██████╗ ",
 "██║   ██║██║╚══███╔╝████╗ ████║██╔═══██╗",
 "██║   ██║██║  ███╔╝ ██╔████╔██║██║   ██║",
 "╚██╗ ██╔╝██║ ███╔╝  ██║╚██╔╝██║██║   ██║",
 " ╚████╔╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝",
 "  ╚═══╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ",
]


class WelcomeOverlay(Panel):
    """Full-canvas welcome splash (Section 6.J).

    Shown when vizmo is launched through the bare-command chooser;
    dismissed by Escape, any click outside an interactive zone, or
    automatically after a recent file is picked. Recents rows return
    ('welcome_open', path) actions; the Open button returns
    'open_file' for the standard dialog dispatch.
    """

    def __init__(self):
        super().__init__(DRAWER_STYLE)
        self.enabled = False
        self.style = PanelStyle(**{**DRAWER_STYLE.__dict__,
                                   "position": "center",
                                   "font_family": "monospace"})
        self._buttons = []
        self._last_key = None

    def _recent_rows(self):
        import datetime
        import os

        from .recentfiles import RecentFiles

        rows = []
        for p in RecentFiles().get()[:10]:
            name = os.path.basename(p)
            if len(name) > 30:
                name = name[:29] + "…"
            try:
                st = os.stat(p)
                size = (f"{st.st_size / 1e9:.1f} GB"
                        if st.st_size > 1e9
                        else f"{st.st_size / 1e6:.0f} MB")
                date = datetime.date.fromtimestamp(
                    st.st_mtime).isoformat()
            except OSError:
                size, date = "?", ""
            rows.append((p, name, size, date))
        return rows

    def update(self, _data=None):
        if not self.enabled:
            return
        import platform

        s = self.style
        M, LH = s.margin, s.line_height
        recents = self._recent_rows()
        key = (tuple(r[0] for r in recents), self._fb_width,
               self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key
        self._buttons = []

        tw = 640
        th = (len(VIZMO_LOGO) + 4) * LH + LH * max(len(recents) + 2, 6) \
            + 5 * LH + 3 * M
        img = Image.new("RGBA", (tw, th), DarkTheme.BG_SURFACE)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius,
                 fill=DarkTheme.BG_SURFACE, outline=DarkTheme.BORDER)
        y = M
        for line in VIZMO_LOGO:
            bb = draw.textbbox((0, 0), line, font=self._font)
            draw.text(((tw - bb[2] + bb[0]) // 2, y), line,
                      fill=DarkTheme.ACCENT, font=self._font)
            y += LH
        tag = "real-time quantitative simulation explorer — v0.8"
        bb = draw.textbbox((0, 0), tag, font=self._font)
        draw.text(((tw - bb[2] + bb[0]) // 2, y), tag,
                  fill=DarkTheme.TEXT_SECONDARY, font=self._font)
        y += 2 * LH

        col_x = tw // 2
        draw.line([(col_x, y), (col_x, th - 3 * LH)],
                  fill=DarkTheme.BORDER)
        # Left: recents
        draw.text((M, y), "RECENT FILES",
                  fill=DarkTheme.TEXT_SECONDARY, font=self._font)
        ry = y + LH
        if not recents:
            draw.text((M, ry), "No recent files",
                      fill=DarkTheme.TEXT_DISABLED, font=self._font)
        for path, name, size, date in recents:
            draw.text((M, ry), name, fill=DarkTheme.TEXT_PRIMARY,
                      font=self._font)
            meta = f"{size}  {date}"
            bb = draw.textbbox((0, 0), meta, font=self._font)
            draw.text((col_x - M - (bb[2] - bb[0]), ry), meta,
                      fill=DarkTheme.TEXT_DISABLED, font=self._font)
            self._buttons.append((M, ry, col_x - M, ry + LH,
                                  ("welcome_open", path)))
            ry += LH
        # Right: open + quickstart + versions
        rx = col_x + M
        _rounded(draw, [(rx, y), (tw - M, y + int(1.4 * LH))], 8,
                 fill=DarkTheme.ACCENT)
        lbl = "Open Simulation File"
        bb = draw.textbbox((0, 0), lbl, font=self._font)
        draw.text((rx + (tw - M - rx - bb[2] + bb[0]) // 2,
                   y + LH // 5), lbl, fill=DarkTheme.TEXT_PRIMARY,
                  font=self._font)
        self._buttons.append((rx, y, tw - M, y + int(1.4 * LH),
                              "open_file"))
        qy = y + 2 * LH
        draw.text((rx, qy), "or drag a file onto this window",
                  fill=DarkTheme.TEXT_DISABLED, font=self._font)
        qy += LH
        draw.text((rx, qy), "QUICKSTART",
                  fill=DarkTheme.TEXT_SECONDARY, font=self._font)
        qy += LH
        for tip in ("vizmo snapshot.hdf5 — basic launch",
                    "Shift+Z — slice through the galaxy",
                    "M — draw analysis aperture"):
            draw.text((rx, qy), tip, fill=DarkTheme.TEXT_PRIMARY,
                      font=self._font)
            qy += LH
        try:
            import meshoid

            mver = getattr(meshoid, "__version__", "?")
        except Exception:
            mver = "?"
        import wgpu as _wgpu

        vline = (f"py {platform.python_version()} | wgpu "
                 f"{_wgpu.__version__} | meshoid {mver}")
        draw.text((M, th - LH - 4), vline,
                  fill=DarkTheme.TEXT_DISABLED, font=self._font)

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def on_click(self, x, y):
        if not self.enabled:
            return False
        lx, ly = x - self._panel_x, y - self._panel_y
        if not (0 <= lx <= self._panel_w and 0 <= ly <= self._panel_h):
            self.enabled = False  # click outside dismisses
            return True
        for x0, y0, x1, y1, action in self._buttons:
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                self.enabled = False
                return action
        return True

    def render(self):
        if not self.enabled:
            return
        super().render()


SPINNER_FRAMES = ["|", "/", "-", "\\"]


class LoadingOverlay(Panel):
    """Startup loading screen (Section 8.A): centered panel with a
    spinner, 400px ACCENT progress bar, and the live load status."""

    def __init__(self):
        super().__init__(DRAWER_STYLE)
        self.enabled = True
        self.style = PanelStyle(**{**DRAWER_STYLE.__dict__,
                                   "position": "center"})
        self._last_key = None

    def update(self, basename, frame, progress, status):
        if not self.enabled:
            return
        s = self.style
        M, LH = s.margin, s.line_height
        spin = SPINNER_FRAMES[(frame // 3) % len(SPINNER_FRAMES)]
        key = (basename, spin, round(progress, 3), status,
               self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key
        tw, th = 460, 4 * LH + 2 * M
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius,
                 fill=DarkTheme.BG_SURFACE, outline=DarkTheme.BORDER)
        draw.text((M, M), f"{spin}  Loading {basename}...",
                  fill=DarkTheme.TEXT_PRIMARY, font=self._font)
        bx, by = M + 10, M + int(1.6 * LH)
        bw = 400
        _rounded(draw, [(bx, by), (bx + bw, by + 12)], 6,
                 fill=DarkTheme.BG_RAISED)
        fill_w = int(bw * max(0.0, min(progress, 1.0)))
        if fill_w > 4:
            _rounded(draw, [(bx, by), (bx + fill_w, by + 12)], 6,
                     fill=DarkTheme.ACCENT)
        draw.text((M, by + 18), status or "...",
                  fill=DarkTheme.TEXT_SECONDARY, font=self._font)
        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def render(self):
        if not self.enabled:
            return
        super().render()


# ---------------------------------------------------------------------------
# Timeline scrubber (Section 4.C), movie recorder (10.B), About,
# keyboard-shortcuts browser
# ---------------------------------------------------------------------------

class TimelineScrubber(Panel):
    """Bottom-edge series timeline: ruler with z ticks, per-snapshot
    circles, playhead, transport buttons, fps readout."""

    HEIGHT = 50

    def __init__(self):
        super().__init__(STATUS_STYLE)
        self.enabled = False
        self.style = PanelStyle(**{**STATUS_STYLE.__dict__,
                                   "position": "bottom-center"})
        self.anchor_offset = (0, 10)
        self.player = None       # series.TimelinePlayer set by the app
        self._buttons = []
        self._track = (0, 0)     # (x0, width) for click mapping
        self._last_key = None

    def update(self):
        if not self.enabled or self.player is None:
            return
        tp = self.player
        s = self.style
        fbw = max(self._fb_width, 100)
        tw = fbw - 8
        th = self.HEIGHT
        cur = tp.snaps[tp.index]
        key = (tp.index, tp.playing, round(tp.fps, 1), tw,
               self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key
        self._buttons = []

        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (tw, th)], fill=DarkTheme.BG_SURFACE)
        draw.rectangle([(0, 0), (tw, 1)], fill=DarkTheme.BORDER)

        # Row 2 (top 30px): the ruler.
        track_w = int(tw * 0.9) - 220
        tx0 = 180
        ty = 16
        self._track = (tx0, track_w)
        draw.line([(tx0, ty), (tx0 + track_w, ty)],
                  fill=DarkTheme.BORDER, width=2)
        zs = [sn.redshift for sn in tp.snaps]
        z0, z1 = max(zs), min(zs)
        span = max(z0 - z1, 1e-12)

        def x_of(z):
            return tx0 + (z0 - z) / span * track_w

        # Major ticks every dz=1.0 (labeled), minor every 0.2.
        z = np.floor(z1 / 0.2) * 0.2
        while z <= z0 + 1e-9:
            big = abs(z - round(z)) < 1e-9
            xx = x_of(z)
            draw.line([(xx, ty - (6 if big else 3)),
                       (xx, ty + (6 if big else 3))],
                      fill=DarkTheme.TEXT_SECONDARY, width=1)
            if big:
                draw.text((xx - 8, ty + 6), f"z={z:.0f}",
                          fill=DarkTheme.TEXT_SECONDARY,
                          font=self._font)
            z += 0.2
        # Snapshot circles + playhead.
        for i, sn in enumerate(tp.snaps):
            xx = x_of(sn.redshift)
            col = (DarkTheme.ACCENT if i == tp.index
                   else DarkTheme.TEXT_SECONDARY)
            draw.ellipse([xx - 4, ty - 4, xx + 4, ty + 4], fill=col)
        xx = x_of(cur.redshift)
        draw.line([(xx, 2), (xx, 28)], fill=DarkTheme.ACCENT, width=2)

        # Left info; right fps.
        draw.text((8, 6), f"z={cur.redshift:.2f} | t={cur.time_gyr:.2f} Gyr",
                  fill=DarkTheme.TEXT_PRIMARY, font=self._font)
        draw.text((tx0 + track_w + 14, 6), f"{tp.fps:.0f} fps",
                  fill=DarkTheme.TEXT_SECONDARY, font=self._font)
        self._buttons.append((tx0 + track_w + 10, 2,
                              tx0 + track_w + 80, 26, "tl_fps"))

        # Row 1 (bottom 20px): transport, centered.
        labels = [("|<", "tl_first"), ("<", "tl_back"),
                  ("||" if tp.playing else ">", "tl_play"),
                  (">", "tl_fwd"), (">|", "tl_last")]
        bx = tw // 2 - 70
        for lbl, act in labels:
            draw.rectangle([(bx, 30), (bx + 24, 48)],
                           fill=DarkTheme.BG_RAISED)
            bb = draw.textbbox((0, 0), lbl, font=self._font)
            draw.text((bx + (24 - bb[2] + bb[0]) // 2, 31), lbl,
                      fill=DarkTheme.TEXT_PRIMARY, font=self._font)
            self._buttons.append((bx, 30, bx + 24, 48, act))
            bx += 28

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def on_click(self, x, y):
        if not self.enabled or self.player is None:
            return False
        lx, ly = x - self._panel_x, y - self._panel_y
        if not (0 <= lx <= self._panel_w and 0 <= ly <= self._panel_h):
            return False
        for x0, y0, x1, y1, act in self._buttons:
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                return act
        tx0, tweff = self._track
        if 4 <= ly <= 28 and tx0 - 6 <= lx <= tx0 + tweff + 6:
            return ("tl_jump",
                    self.player.nearest_index(lx - tx0, tweff))
        return True

    def render(self):
        if not self.enabled:
            return
        super().render()


class MovieRecorderPanel(Panel):
    """Movie recorder (Section 10.B): manual / orbit / series modes,
    fps + resolution + format, ffmpeg auto-assembly on stop."""

    MODES = ["Manual", "Orbit", "Series"]
    FORMATS = (["MP4 (H.264)", "PNG sequence", "ProRes 422"]
               if __import__("sys").platform == "darwin"
               else ["MP4 (H.264)", "PNG sequence"])
    RESOLUTIONS = ["720p", "1080p", "4K"]

    def __init__(self):
        super().__init__(DRAWER_STYLE)
        self.enabled = False
        self.style = PanelStyle(**{**DRAWER_STYLE.__dict__,
                                   "position": "center-left"})
        self.recording = False
        self.mode_idx = 0
        self.fps = 24
        self.res_idx = 1
        self.fmt_idx = 0
        self.orbit_speed = 30.0
        self.orbit_duration = 12.0
        self.frames_per_snap = 1
        self.series_available = False
        self._buttons = []
        self._last_key = None

    def update(self, _data=None):
        if not self.enabled:
            return
        s = self.style
        M, LH = s.margin, s.line_height
        key = (self.recording, self.mode_idx, self.fps, self.res_idx,
               self.fmt_idx, self.orbit_speed, self.orbit_duration,
               self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key
        self._buttons = []

        rows = 8 + (2 if self.MODES[self.mode_idx] == "Orbit" else 0)
        tw, th = 320, LH * rows + 3 * M + 30
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius,
                 fill=DarkTheme.BG_SURFACE, outline=DarkTheme.BORDER)
        draw.text((M, 6), "Movie recorder", fill=DarkTheme.ACCENT,
                  font=self._font)
        y = LH + 8
        # Start/stop button.
        fill = DarkTheme.DANGER if self.recording else DarkTheme.SUCCESS
        lbl = "STOP RECORDING" if self.recording else "START RECORDING"
        _rounded(draw, [(M, y), (tw - M, y + 34)], 8, fill=fill)
        bb = draw.textbbox((0, 0), lbl, font=self._font)
        draw.text(((tw - bb[2] + bb[0]) // 2, y + 6), lbl,
                  fill=DarkTheme.TEXT_PRIMARY, font=self._font)
        self._buttons.append((M, y, tw - M, y + 34, "mv_toggle"))
        y += 40

        def row(label, value, action):
            nonlocal y
            draw.text((M, y), label, fill=DarkTheme.TEXT_SECONDARY,
                      font=self._font)
            bb = draw.textbbox((0, 0), str(value), font=self._font)
            _rounded(draw, [(tw - M - bb[2] + bb[0] - 16, y),
                            (tw - M, y + LH - 4)], 6,
                     fill=DarkTheme.BG_RAISED)
            draw.text((tw - M - bb[2] + bb[0] - 8, y), str(value),
                      fill=DarkTheme.TEXT_PRIMARY, font=self._font)
            self._buttons.append((tw - M - bb[2] + bb[0] - 16, y,
                                  tw - M, y + LH - 4, action))
            y += LH

        mode = self.MODES[self.mode_idx]
        if mode == "Series" and not self.series_available:
            mode += " (n/a)"
        row("Mode", mode, "mv_mode")
        row("FPS", self.fps, "mv_fps")
        row("Resolution", self.RESOLUTIONS[self.res_idx], "mv_res")
        row("Format", self.FORMATS[self.fmt_idx], "mv_fmt")
        if self.MODES[self.mode_idx] == "Orbit":
            row("Orbit deg/s", f"{self.orbit_speed:.0f}", "mv_ospeed")
            row("Duration s", f"{self.orbit_duration:.0f}", "mv_odur")
        if self.MODES[self.mode_idx] == "Series":
            row("Frames/snap", self.frames_per_snap, "mv_fpsnap")

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def on_click(self, x, y):
        if not self.enabled:
            return False
        lx, ly = x - self._panel_x, y - self._panel_y
        if not (0 <= lx <= self._panel_w and 0 <= ly <= self._panel_h):
            return False
        for x0, y0, x1, y1, act in self._buttons:
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                if act == "mv_mode" and not self.recording:
                    self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
                    if (self.MODES[self.mode_idx] == "Series"
                            and not self.series_available):
                        self.mode_idx = 0
                    self._last_key = None
                    return True
                if act == "mv_fps" and not self.recording:
                    cyc = [12, 24, 30, 60]
                    self.fps = cyc[(cyc.index(self.fps) + 1) % len(cyc)] \
                        if self.fps in cyc else 24
                    self._last_key = None
                    return True
                if act == "mv_res" and not self.recording:
                    self.res_idx = (self.res_idx + 1) % len(self.RESOLUTIONS)
                    self._last_key = None
                    return ("mv_res_changed", self.RESOLUTIONS[self.res_idx])
                if act == "mv_fmt" and not self.recording:
                    self.fmt_idx = (self.fmt_idx + 1) % len(self.FORMATS)
                    self._last_key = None
                    return True
                if act == "mv_ospeed" and not self.recording:
                    cyc = [15.0, 30.0, 60.0, 90.0]
                    self.orbit_speed = cyc[
                        (cyc.index(self.orbit_speed) + 1) % len(cyc)] \
                        if self.orbit_speed in cyc else 30.0
                    self._last_key = None
                    return True
                if act == "mv_odur" and not self.recording:
                    cyc = [6.0, 12.0, 24.0, 48.0]
                    self.orbit_duration = cyc[
                        (cyc.index(self.orbit_duration) + 1) % len(cyc)] \
                        if self.orbit_duration in cyc else 12.0
                    self._last_key = None
                    return True
                if act == "mv_fpsnap" and not self.recording:
                    self.frames_per_snap = (self.frames_per_snap % 4) + 1
                    self._last_key = None
                    return True
                return act
        return True

    def render(self):
        if not self.enabled:
            return
        super().render()


def ffmpeg_command(fps, fmt_label, frames_pattern, out_path):
    """The ffmpeg invocation for a recorded PNG sequence."""
    if fmt_label.startswith("ProRes"):
        return ["ffmpeg", "-y", "-r", str(fps), "-i", frames_pattern,
                "-c:v", "prores_ks", "-profile:v", "2", out_path]
    return ["ffmpeg", "-y", "-r", str(fps), "-i", frames_pattern,
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            out_path]


class AboutPanel(Panel):
    """Help > About: version table + credits."""

    def __init__(self):
        super().__init__(DRAWER_STYLE)
        self.enabled = False
        self.style = PanelStyle(**{**DRAWER_STYLE.__dict__,
                                   "position": "center",
                                   "font_family": "monospace"})
        self._last_key = None

    @staticmethod
    def version_rows():
        import platform

        rows = [("Python", platform.python_version())]
        for mod in ("wgpu", "meshoid", "numpy", "astropy", "numba",
                    "trident"):
            try:
                m = __import__(mod)
                rows.append((mod, getattr(m, "__version__", "?")))
            except ImportError:
                rows.append((mod, "not installed"))
        return rows

    def update(self, _data=None):
        if not self.enabled:
            return
        s = self.style
        M, LH = s.margin, s.line_height
        rows = self.version_rows()
        key = ("about", self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key
        tw = 340
        th = (len(VIZMO_LOGO) + len(rows) + 7) * LH + 2 * M
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius,
                 fill=DarkTheme.BG_SURFACE, outline=DarkTheme.BORDER)
        y = M
        for line in VIZMO_LOGO:
            bb = draw.textbbox((0, 0), line, font=self._font)
            draw.text(((tw - bb[2] + bb[0]) // 2, y), line,
                      fill=DarkTheme.ACCENT, font=self._font)
            y += LH
        try:
            from importlib.metadata import version as _ver

            v = _ver("vizmo")
        except Exception:
            v = "dev"
        for line in (f"v{v}", "Interactive 3D simulation explorer"):
            bb = draw.textbbox((0, 0), line, font=self._font)
            draw.text(((tw - bb[2] + bb[0]) // 2, y), line,
                      fill=DarkTheme.TEXT_PRIMARY, font=self._font)
            y += LH
        draw.line([(M, y + 2), (tw - M, y + 2)], fill=DarkTheme.BORDER)
        y += 8
        for name, ver in rows:
            draw.text((M, y), name, fill=DarkTheme.TEXT_SECONDARY,
                      font=self._font)
            bb = draw.textbbox((0, 0), ver, font=self._font)
            draw.text((tw - M - bb[2] + bb[0], y), ver,
                      fill=DarkTheme.TEXT_PRIMARY, font=self._font)
            y += LH
        draw.line([(M, y + 2), (tw - M, y + 2)], fill=DarkTheme.BORDER)
        y += 8
        for line in ("Built by Tanmay Singh (ASU STARs Lab)",
                     "and Mike Grudic. Core GPU renderer", 
                     "by mikegrudic."):
            draw.text((M, y), line, fill=DarkTheme.TEXT_SECONDARY,
                      font=self._font)
            y += LH
        self._close_zone = (tw - M - 60, y, tw - M, y + LH)
        _rounded(draw, [self._close_zone[:2], self._close_zone[2:]], 6,
                 fill=DarkTheme.BG_RAISED)
        draw.text((tw - M - 50, y), "Close",
                  fill=DarkTheme.TEXT_PRIMARY, font=self._font)
        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def on_click(self, x, y):
        if not self.enabled:
            return False
        lx, ly = x - self._panel_x, y - self._panel_y
        if not (0 <= lx <= self._panel_w and 0 <= ly <= self._panel_h):
            self.enabled = False
            return True
        self.enabled = False  # any click closes
        return True

    def render(self):
        if not self.enabled:
            return
        super().render()


class KeyboardShortcutsPanel(Panel):
    """Help > Keyboard Shortcuts: categorized, scrollable, searchable
    listing of keymap.py."""

    def __init__(self):
        super().__init__(DRAWER_STYLE)
        self.enabled = False
        self.style = PanelStyle(**{**DRAWER_STYLE.__dict__,
                                   "position": "center"})
        self.search = ""
        self.scroll = 0
        self._last_key = None

    def filtered(self):
        from .keymap import KEYBINDINGS, KEY_CATEGORIES

        q = self.search.lower()
        out = []
        for cat in KEY_CATEGORIES:
            hits = [(k, d) for k, d, c in KEYBINDINGS
                    if c == cat and (q in k.lower() or q in d.lower())]
            if hits:
                out.append((cat, hits))
        return out

    def on_char(self, char):
        if not self.enabled:
            return False
        self.search += char
        self.scroll = 0
        self._last_key = None
        return True

    def on_backspace(self):
        if self.enabled and self.search:
            self.search = self.search[:-1]
            self._last_key = None
            return True
        return False

    def on_scroll(self, dy):
        if not self.enabled:
            return False
        self.scroll = max(0, self.scroll - int(dy) * 3)
        self._last_key = None
        return True

    def update(self, _data=None):
        if not self.enabled:
            return
        s = self.style
        M, LH = s.margin, s.line_height
        groups = self.filtered()
        key = (self.search, self.scroll,
               tuple((c, len(h)) for c, h in groups),
               self._fb_width, self._fb_height)
        if key == self._last_key and self._tex is not None:
            return
        self._last_key = key
        tw, th = 380, 600
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius,
                 fill=DarkTheme.BG_SURFACE, outline=DarkTheme.BORDER)
        hint = self.search or "type to filter..."
        draw.text((M, 6), f"Shortcuts: {hint}",
                  fill=DarkTheme.ACCENT, font=self._font)
        # Flatten rows then window by scroll.
        flat = []
        for cat, hits in groups:
            flat.append(("__cat__", cat))
            flat.extend(hits)
        n_vis = (th - LH - 2 * M) // LH
        total = len(flat)
        window = flat[self.scroll:self.scroll + n_vis]
        y = LH + 6
        for item in window:
            if item[0] == "__cat__":
                draw.line([(M, y + LH // 2), (tw - M, y + LH // 2)],
                          fill=DarkTheme.BORDER)
                draw.text((M, y), item[1],
                          fill=DarkTheme.TEXT_SECONDARY, font=self._font)
            else:
                k, d = item
                draw.text((M, y), k, fill=DarkTheme.ACCENT,
                          font=self._font)
                if len(d) > 34:
                    d = d[:33] + "…"
                bb = draw.textbbox((0, 0), d, font=self._font)
                draw.text((tw - M - bb[2] + bb[0], y), d,
                          fill=DarkTheme.TEXT_PRIMARY, font=self._font)
            y += LH
        # Scrollbar.
        if total > n_vis:
            frac = n_vis / total
            top = self.scroll / total
            sb_y0 = int(LH + top * (th - 2 * LH))
            sb_h = max(int(frac * (th - 2 * LH)), 20)
            _rounded(draw, [(tw - 8, sb_y0), (tw - 4, sb_y0 + sb_h)],
                     2, fill=DarkTheme.TEXT_DISABLED)
        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def on_click(self, x, y):
        if not self.enabled:
            return False
        lx, ly = x - self._panel_x, y - self._panel_y
        if not (0 <= lx <= self._panel_w and 0 <= ly <= self._panel_h):
            self.enabled = False
            return True
        return True

    def render(self):
        if not self.enabled:
            return
        super().render()
