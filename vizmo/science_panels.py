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

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

SCALEBAR_STYLE = PanelStyle(
    font_size=20, line_height=28, margin=8, min_width=10,
    bg_color=(0, 0, 0, 0),
    text_color=(235, 238, 245, 255),
    accent_color=(255, 255, 255, 255),
    toggle_on_color=(100, 180, 255, 255),
    toggle_off_color=(110, 115, 130, 255),
    dropdown_bg=(34, 37, 50, 255),
    dropdown_hover=(80, 100, 140, 255),
    slider_btn=(64, 70, 88, 255),
    position="bottom-center",
    font_family="sans-serif",
)

STATUS_STYLE = PanelStyle(
    font_size=16, line_height=24, margin=8, min_width=10,
    bg_color=(10, 12, 20, 165),
    text_color=(200, 206, 218, 255),
    accent_color=(100, 180, 255, 255),
    toggle_on_color=(100, 180, 255, 255),
    toggle_off_color=(110, 115, 130, 255),
    dropdown_bg=(34, 37, 50, 255),
    dropdown_hover=(80, 100, 140, 255),
    slider_btn=(64, 70, 88, 255),
    position="bottom-right",
    font_family="sans-serif",
    radius=10,
)

TOAST_STYLE = PanelStyle(
    font_size=20, line_height=30, margin=10, min_width=10,
    bg_color=(18, 22, 34, 222),
    text_color=(232, 236, 244, 255),
    accent_color=(120, 200, 140, 255),
    toggle_on_color=(120, 200, 140, 255),
    toggle_off_color=(200, 120, 120, 255),
    dropdown_bg=(34, 37, 50, 255),
    dropdown_hover=(80, 100, 140, 255),
    slider_btn=(64, 70, 88, 255),
    position="top-center",
    font_family="sans-serif",
    radius=12,
)

GIZMO_STYLE = PanelStyle(
    font_size=15, line_height=20, margin=6, min_width=10,
    bg_color=(10, 12, 20, 130),
    text_color=(220, 224, 232, 255),
    accent_color=(255, 255, 255, 255),
    toggle_on_color=(100, 180, 255, 255),
    toggle_off_color=(110, 115, 130, 255),
    dropdown_bg=(34, 37, 50, 255),
    dropdown_hover=(80, 100, 140, 255),
    slider_btn=(64, 70, 88, 255),
    position="bottom-right",
    font_family="sans-serif",
    radius=10,
)

DRAWER_STYLE = PanelStyle(
    font_size=20, line_height=30, margin=12, min_width=360,
    bg_color=(14, 16, 26, 235),
    text_color=(228, 231, 238, 255),
    accent_color=(100, 180, 255, 255),
    toggle_on_color=(100, 180, 255, 255),
    toggle_off_color=(110, 115, 130, 255),
    dropdown_bg=(34, 37, 50, 255),
    dropdown_hover=(80, 100, 140, 255),
    slider_btn=(64, 70, 88, 255),
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
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        ybar = th - 6
        shadow = (0, 0, 0, 180)
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
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius, fill=s.bg_color,
                 outline=(255, 255, 255, 26))
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

    DURATION = 3.2
    FADE = 0.6

    def __init__(self):
        super().__init__(TOAST_STYLE)
        self.enabled = True
        self._toasts = []  # (text, t_expire, kind)

    def show(self, text, kind="info", duration=None):
        now = time.time()
        self._toasts.append((str(text), now + (duration or self.DURATION), kind))
        self._toasts = self._toasts[-4:]

    @property
    def active(self):
        return bool(self._toasts)

    def update(self):
        if not self.enabled:
            return
        now = time.time()
        self._toasts = [t for t in self._toasts if t[1] > now]
        if not self._toasts:
            self._panel_w = self._panel_h = 0
            self._tex = None
            return

        s = self.style
        M, LH = s.margin, s.line_height
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tw = 10
        for text, _, _ in self._toasts:
            bbox = dummy.textbbox((0, 0), text, font=self._font)
            tw = max(tw, bbox[2] - bbox[0] + M * 4)
        row_h = LH + 8
        th = row_h * len(self._toasts) + 4

        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        kind_col = {
            "info": s.text_color,
            "ok": s.toggle_on_color,
            "warn": (255, 200, 110, 255),
            "error": s.toggle_off_color,
        }
        y = 0
        for text, t_exp, kind in reversed(self._toasts):
            alpha = 1.0
            if t_exp - now < self.FADE:
                alpha = max(0.0, (t_exp - now) / self.FADE)
            bg = tuple(list(s.bg_color[:3]) + [int(s.bg_color[3] * alpha)])
            fg0 = kind_col.get(kind, s.text_color)
            fg = tuple(list(fg0[:3]) + [int(fg0[3] * alpha)])
            _rounded(draw, [(0, y + 2), (tw - 1, y + row_h - 2)], s.radius,
                     fill=bg, outline=(255, 255, 255, int(30 * alpha)))
            draw.text((M + 6, y + 5), text, fill=fg, font=self._font)
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
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        s = self.style
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius, fill=s.bg_color,
                 outline=(255, 255, 255, 22))

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
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(230, 230, 235, 255))

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
    bg_color=(0, 0, 0, 0),
    text_color=(235, 238, 245, 255),
    accent_color=(120, 200, 255, 255),
    toggle_on_color=(120, 200, 255, 255),
    toggle_off_color=(110, 115, 130, 255),
    dropdown_bg=(34, 37, 50, 255),
    dropdown_hover=(80, 100, 140, 255),
    slider_btn=(64, 70, 88, 255),
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
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        col = self.style.accent_color if not placing else (255, 200, 110, 255)
        shadow = (0, 0, 0, 150)

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
                      "EnclosedMass", "Temperature", "RadialVelocity",
                      "MetallicityZsun", "Pressure", "VelocityMagnitude"]

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

    def set_scope(self, center, radius_kpc, center_mode="densest"):
        self.scope = {
            "center": np.asarray(center, dtype=np.float64),
            "radius_kpc": float(radius_kpc),
            "center_mode": center_mode,
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
            presets = self._phase_presets(data)
            self._phase_idx %= len(presets)
            xf, yf = presets[self._phase_idx]
            key = ("phase", xf, yf, data.n_particles, sc_key)
            if key not in self._cache:
                ph = analysis.phase_histogram(
                    data, xf, yf, center=sc_center, radius_kpc=sc_radius)
                self._cache[key] = self._render_phase(ph, scale)
            return self._cache[key], f"{xf} vs {yf}"

        if self.mode == "profile":
            fields = self._profile_fields(data)
            self._profile_idx %= len(fields)
            f = fields[self._profile_idx]
            key = ("profile", f, data.n_particles, sc_key)
            if key not in self._cache:
                r, prof, unit = analysis.radial_profile(
                    data, f, center=sc_center, r_max_kpc=sc_radius)
                self._cache[key] = self._render_profile(r, prof, unit, f, scale)
            return self._cache[key], f"{f}(r)"
        return None, ""

    def _render_phase(self, ph, scale):
        from matplotlib.figure import Figure
        from matplotlib.colors import LogNorm

        fig = Figure(figsize=(4.7 * scale, 4.1 * scale), dpi=100)
        ax = fig.add_subplot(111)
        if ph is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        else:
            H = ph["H"].T
            vmax = H.max()
            with np.errstate(invalid="ignore"):
                im = ax.imshow(
                    H, origin="lower", aspect="auto", cmap="inferno",
                    norm=LogNorm(vmin=max(vmax * 1e-6, 1e-30), vmax=max(vmax, 1e-29)),
                    extent=[ph["xedges"][0], ph["xedges"][-1],
                            ph["yedges"][0], ph["yedges"][-1]],
                    interpolation="nearest",
                )
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_label("mass [Msun]", color=(0.88, 0.9, 0.95), fontsize=9)
            cb.ax.tick_params(colors=(0.82, 0.85, 0.9), labelsize=8)
            cb.outline.set_edgecolor((0.75, 0.78, 0.85, 0.8))
            ax.set_xlabel(ph["xlabel"], fontsize=10)
            ax.set_ylabel(ph["ylabel"], fontsize=10)
        _dark_axes(fig, ax)
        fig.tight_layout(pad=1.2)
        return _fig_to_image(fig)

    def _render_profile(self, r, prof, unit, field, scale):
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4.7 * scale, 3.9 * scale), dpi=100)
        ax = fig.add_subplot(111)
        ok = np.isfinite(prof)
        if ok.sum() > 1:
            ax.plot(r[ok], prof[ok], lw=2.0, color=(0.42, 0.72, 1.0))
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

    # -- drawing ------------------------------------------------------------

    def update(self, data):
        if not self.enabled or self.mode is None:
            return
        from . import analysis

        s = self.style
        M, LH = s.margin, s.line_height
        self._buttons = []

        titles = {"inspector": "Particle inspector", "phase": "Phase diagram",
                  "profile": "Radial profile", "stats": "Region statistics"}
        title = titles.get(self.mode, "")

        plot_img, caption = (None, "")
        rows = []
        if self.mode in ("phase", "profile"):
            plot_img, caption = self._content_image(data)
        elif self.mode == "inspector":
            if self._picked_index is None:
                rows = [("No particle picked", "Shift+click to pick")]
            else:
                if "inspector" not in self._cache:
                    self._cache["inspector"] = analysis.particle_summary(
                        data, self._picked_index)
                rows = self._cache["inspector"]
        elif self.mode == "stats":
            sc_center, sc_radius = self._active_scope()
            r_use = sc_radius if sc_radius is not None else self._stats_radius_kpc
            sck = (None if sc_center is None
                   else tuple(np.round(sc_center, 3)))
            key = ("stats", r_use, data.n_particles, sck)
            if key not in self._cache:
                self._cache[key] = analysis.region_stats(
                    data, center=sc_center, radius_kpc=r_use)
            rows, used_r = self._cache[key]
            if sc_radius is None:
                self._stats_radius_kpc = used_r

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
        footer_h = LH + 8 if self.mode in ("phase", "profile", "stats") else 6
        body_h = (plot_img.height + 8 if plot_img is not None
                  else LH * max(len(rows), 1) + 8)
        extra_h = LH if self.mode == "inspector" and self._picked_index is not None else 0
        scope_h = LH + 4 if self.scope is not None else 0
        th = header_h + body_h + footer_h + extra_h + scope_h + M

        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius, fill=s.bg_color,
                 outline=(255, 255, 255, 30))

        # Header: title + close box
        draw.text((M, 6), title, fill=s.accent_color, font=self._font)
        cw = LH - 6
        cx0, cy0 = tw - M - cw, 5
        _rounded(draw, [(cx0, cy0), (cx0 + cw, cy0 + cw)], 6,
                 fill=(60, 34, 40, 255), outline=(255, 255, 255, 40))
        draw.line([(cx0 + 6, cy0 + 6), (cx0 + cw - 6, cy0 + cw - 6)],
                  fill=(235, 160, 160, 255), width=2)
        draw.line([(cx0 + cw - 6, cy0 + 6), (cx0 + 6, cy0 + cw - 6)],
                  fill=(235, 160, 160, 255), width=2)
        self._buttons.append((cx0, cy0, cx0 + cw, cy0 + cw, "close"))
        draw.line([(M, header_h - 2), (tw - M, header_h - 2)],
                  fill=(255, 255, 255, 30), width=1)

        y = header_h + 4
        if plot_img is not None:
            img.alpha_composite(plot_img, ((tw - plot_img.width) // 2, y))
            y += plot_img.height + 4
        else:
            for a, b in rows:
                draw.text((M, y + 2), str(a), fill=(168, 174, 188, 255),
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
                     fill=(40, 62, 90, 255), outline=(255, 255, 255, 45))
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
            fill = (40, 62, 90, 255) if self.use_scope else s.slider_btn
            _rounded(draw, [(bx, y + 2), (bx + bw, y + LH - 2)], 8,
                     fill=fill, outline=(255, 255, 255, 45))
            draw.text((bx + 10, y), lbl, fill=s.text_color, font=self._font)
            self._buttons.append((bx, y + 2, bx + bw, y + LH - 2, "toggle_scope"))
            bx += bw + 8
            if self.use_scope:
                lbl2 = self.scope["center_mode"]
                bb = dummy.textbbox((0, 0), lbl2, font=self._font)
                bw2 = bb[2] - bb[0] + 20
                _rounded(draw, [(bx, y + 2), (bx + bw2, y + LH - 2)], 8,
                         fill=s.slider_btn, outline=(255, 255, 255, 45))
                draw.text((bx + 10, y), lbl2, fill=s.text_color, font=self._font)
                self._buttons.append((bx, y + 2, bx + bw2, y + LH - 2,
                                      "cycle_center"))
                bx += bw2 + 8
                draw.text((bx + 4, y + 1),
                          f"R={self.scope['radius_kpc']:.0f} kpc",
                          fill=(168, 174, 188, 255), font=self._font)
            y += LH + 4

        # Footer controls
        if self.mode in ("phase", "profile"):
            bx = M
            for lbl, action in (("<", "prev"), (">", "next")):
                bw = LH + 8
                _rounded(draw, [(bx, y + 2), (bx + bw, y + LH - 2)], 8,
                         fill=s.slider_btn, outline=(255, 255, 255, 45))
                bb = dummy.textbbox((0, 0), lbl, font=self._font)
                draw.text((bx + (bw - bb[2] + bb[0]) // 2, y), lbl,
                          fill=s.text_color, font=self._font)
                self._buttons.append((bx, y + 2, bx + bw, y + LH - 2, action))
                bx += bw + 8
            draw.text((bx + 6, y + 1), caption, fill=(168, 174, 188, 255),
                      font=self._font)
        elif self.mode == "stats":
            bx = M
            for lbl, action in (("R/2", "r_half"), ("Rx2", "r_double")):
                bb = dummy.textbbox((0, 0), lbl, font=self._font)
                bw = bb[2] - bb[0] + 20
                _rounded(draw, [(bx, y + 2), (bx + bw, y + LH - 2)], 8,
                         fill=s.slider_btn, outline=(255, 255, 255, 45))
                draw.text((bx + 10, y), lbl, fill=s.text_color, font=self._font)
                self._buttons.append((bx, y + 2, bx + bw, y + LH - 2, action))
                bx += bw + 8

        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = self._panel_origin(tw, th)
        self._upload_panel(tw, th, img.tobytes())

    def on_click(self, x, y):
        """Returns an action string handled by the app, True if consumed,
        or False if the click missed the drawer."""
        if not self.enabled:
            return False
        lx, ly = x - self._panel_x, y - self._panel_y
        if lx < 0 or lx > self._panel_w or ly < 0 or ly > self._panel_h:
            return False
        for x0, y0, x1, y1, action in self._buttons:
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                if action == "close":
                    self.enabled = False
                    self.mode = None
                    return True
                if action == "prev":
                    if self.mode == "phase":
                        self._phase_idx -= 1
                    else:
                        self._profile_idx -= 1
                    self._last_key = None
                    return True
                if action == "next":
                    if self.mode == "phase":
                        self._phase_idx += 1
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
                if action == "toggle_scope":
                    self.use_scope = not self.use_scope
                    self.refresh()
                    return True
                return action
        return True

    def render(self):
        if not self.enabled:
            return
        super().render()
