"""UI overlay panels rendered as PIL images. The actual GPU upload +
draw step lives in wgpu_overlay.py; this file contains only the
backend-agnostic widget layout, hit-testing, and PIL rendering.
"""

from PIL import Image, ImageDraw, ImageFont
from .themes import DarkTheme
from dataclasses import dataclass


def _get_font(size, family="monospace"):
    try:
        import matplotlib.font_manager as fm
        path = fm.findfont(fm.FontProperties(family=family))
        if path:
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _rounded(draw, box, radius, fill=None, outline=None, width=1):
    """rounded_rectangle with a guard for degenerate boxes."""
    (x0, y0), (x1, y1) = box
    if x1 <= x0 or y1 <= y0:
        return
    r = max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill, outline=outline, width=width)


@dataclass
class PanelStyle:
    font_size: int
    line_height: int
    margin: int
    min_width: int
    bg_color: tuple
    text_color: tuple
    accent_color: tuple
    toggle_on_color: tuple
    toggle_off_color: tuple
    dropdown_bg: tuple
    dropdown_hover: tuple
    slider_btn: tuple
    field_bg: tuple = DarkTheme.C_30_30_30_255
    field_active: tuple = DarkTheme.C_50_50_80_255
    position: str = "top-right"  # "top-right", "top-left", "bottom-left", or "center"
    font_family: str = "monospace"
    radius: int = 10  # panel corner radius (px at 1080p reference)


DEV_STYLE = PanelStyle(
    font_size=14, line_height=20, margin=8, min_width=200,
    bg_color=DarkTheme.C_0_0_0_200,
    text_color=DarkTheme.C_0_255_0_255,
    accent_color=DarkTheme.C_0_255_0_255,
    toggle_on_color=DarkTheme.C_0_200_0_255,
    toggle_off_color=DarkTheme.C_150_50_50_255,
    dropdown_bg=DarkTheme.C_40_40_40_255,
    dropdown_hover=DarkTheme.C_80_80_120_255,
    slider_btn=DarkTheme.C_80_80_80_255,
    position="top-right",
)

SINK_STYLE = PanelStyle(
    font_size=14, line_height=20, margin=8, min_width=220,
    bg_color=DarkTheme.C_20_10_30_220,
    text_color=DarkTheme.C_220_220_255_255,
    accent_color=DarkTheme.C_180_180_255_255,
    toggle_on_color=DarkTheme.C_120_160_255_255,
    toggle_off_color=DarkTheme.C_100_100_110_255,
    dropdown_bg=DarkTheme.C_40_40_60_255,
    dropdown_hover=DarkTheme.C_80_80_120_255,
    slider_btn=DarkTheme.C_80_80_100_255,
    position="top-right",
)

HELP_STYLE = PanelStyle(
    font_size=18, line_height=26, margin=14, min_width=420,
    bg_color=DarkTheme.C_14_16_26_238,
    text_color=DarkTheme.C_228_231_238_255,
    accent_color=DarkTheme.C_120_190_255_255,
    toggle_on_color=DarkTheme.C_120_190_255_255,
    toggle_off_color=DarkTheme.C_150_150_160_255,
    dropdown_bg=DarkTheme.C_30_30_40_255,
    dropdown_hover=DarkTheme.C_80_100_140_255,
    slider_btn=DarkTheme.C_70_75_90_255,
    position="center",
    font_family="sans-serif",
    radius=14,
)

USER_STYLE = PanelStyle(
    font_size=28, line_height=38, margin=14, min_width=280,
    bg_color=DarkTheme.C_16_18_28_170,
    text_color=DarkTheme.C_228_231_238_255,
    accent_color=DarkTheme.C_100_180_255_255,
    toggle_on_color=DarkTheme.C_100_180_255_255,
    toggle_off_color=DarkTheme.C_110_115_130_255,
    dropdown_bg=DarkTheme.C_34_37_50_255,
    dropdown_hover=DarkTheme.C_80_100_140_255,
    slider_btn=DarkTheme.C_64_70_88_255,
    position="bottom-left",
    font_family="sans-serif",
    radius=14,
)

TOOLBAR_STYLE = PanelStyle(
    font_size=22, line_height=34, margin=10, min_width=10,
    bg_color=DarkTheme.C_16_18_28_170,
    text_color=DarkTheme.C_228_231_238_255,
    accent_color=DarkTheme.C_100_180_255_255,
    toggle_on_color=DarkTheme.C_100_180_255_255,
    toggle_off_color=DarkTheme.C_110_115_130_255,
    dropdown_bg=DarkTheme.C_34_37_50_255,
    dropdown_hover=DarkTheme.C_80_100_140_255,
    slider_btn=DarkTheme.C_64_70_88_255,
    position="top-left",
    font_family="sans-serif",
    radius=12,
)


class Panel:
    """Base class for PIL-rendered UI panels uploaded as GPU textures.

    Subclasses override build_items() to define widget content and
    on_widget_click() to handle interactions.
    """

    # Reference height for DPI scaling (styles are designed for 1080p)
    _REF_HEIGHT = 1080

    def __init__(self, style):
        self.style = style
        self._base_style = style  # unscaled original
        self._tex = None
        self._font = _get_font(style.font_size, style.font_family)
        self._dpi_scale = 1.0
        self._widgets = []
        self._dropdown_open = None
        self._dropdown_scroll = {}
        # Subclasses can opt into a minimize button by appending a
        # ("button", "[-]", "_toggle_minimize") item to render_panel.
        self._minimized = False
        self._fb_width = 1
        self._fb_height = 1
        # Pixel nudge applied after anchoring (dx right, dy down) so two
        # panels can share an anchor without overlapping.
        self.anchor_offset = (0, 0)
        self._panel_x = 0
        self._panel_y = 0
        self._panel_w = 0
        self._panel_h = 0
        self._last_items_key = None  # for dirty-flag caching

    def set_framebuffer_size(self, w, h):
        self._fb_width = w
        self._fb_height = h
        # DPI scaling: scale UI relative to 1080p reference
        new_scale = max(h, 1) / self._REF_HEIGHT
        if abs(new_scale - self._dpi_scale) > 0.01:
            self._dpi_scale = new_scale
            bs = self._base_style
            self.style = PanelStyle(
                font_size=max(8, int(bs.font_size * new_scale)),
                line_height=max(12, int(bs.line_height * new_scale)),
                margin=max(4, int(bs.margin * new_scale)),
                min_width=max(10, int(bs.min_width * new_scale)),
                bg_color=bs.bg_color, text_color=bs.text_color,
                accent_color=bs.accent_color,
                toggle_on_color=bs.toggle_on_color,
                toggle_off_color=bs.toggle_off_color,
                dropdown_bg=bs.dropdown_bg,
                dropdown_hover=bs.dropdown_hover,
                slider_btn=bs.slider_btn,
                field_bg=bs.field_bg,
                field_active=bs.field_active,
                position=bs.position,
                font_family=bs.font_family,
                radius=max(4, int(bs.radius * new_scale)),
            )
            self._font = _get_font(self.style.font_size, self.style.font_family)
            self._last_items_key = None  # force re-render

    def render_panel(self, items):
        """Measure, draw, and upload a list of widget items. Skips if unchanged."""
        # Dirty check: hash items + dropdown state + framebuffer size to skip re-render
        items_key = (
            tuple(tuple(item) if not isinstance(item, tuple) else item for item in items),
            self._dropdown_open,
            tuple(sorted(self._dropdown_scroll.items())),
            self._fb_width, self._fb_height,
        )
        if items_key == self._last_items_key and self._tex is not None:
            return
        self._last_items_key = items_key

        s = self.style
        M, LH = s.margin, s.line_height

        # Measure width
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        max_w = s.min_width
        kv_key_w = 0
        for item in items:
            if item[0] == "kv":
                bbox = draw.textbbox((0, 0), item[1], font=self._font)
                kv_key_w = max(kv_key_w, bbox[2] - bbox[0])
        toggle_pill_w = int(LH * 1.5)
        for item in items:
            t = item[0]
            if t == "kv":
                bbox = draw.textbbox((0, 0), item[2], font=self._font)
                max_w = max(max_w, kv_key_w + 24 + bbox[2] - bbox[0] + M * 2)
            elif t in ("text", "field"):
                label = f"{item[1]}: {item[2]}" if len(item) > 2 else item[1]
                bbox = draw.textbbox((0, 0), label, font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + M * 4)
            elif t == "button":
                bbox = draw.textbbox((0, 0), item[1], font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + M * 4 + 16)
            elif t == "button_row":
                row_w = M
                for entry in item[1]:
                    bbox = draw.textbbox((0, 0), entry[0], font=self._font)
                    row_w += (bbox[2] - bbox[0]) + 28 + 8
                max_w = max(max_w, row_w)
            elif t == "toggle":
                bbox = draw.textbbox((0, 0), item[1], font=self._font)
                max_w = max(max_w, toggle_pill_w + 10 + bbox[2] - bbox[0] + M * 4)
            elif t == "dropdown":
                bbox = draw.textbbox((0, 0), f"v {item[1]}: {item[2]}", font=self._font)
                max_w = max(max_w, bbox[2] - bbox[0] + M * 4)
            elif t == "slider":
                bbox = draw.textbbox((0, 0), f"{item[1]}: {item[2]:.2f}", font=self._font)
                # text + both buttons + at least 90px of visible track
                max_w = max(max_w, bbox[2] - bbox[0] + 2 * (LH + 10) + 90 + M * 4)
            elif t == "ptype_row":
                # Vertical stack: width is just the widest single label
                # (plus the indent + padding), not the sum of all of them.
                bbox = draw.textbbox((0, 0), f"{item[1]}:", font=self._font)
                header_w = bbox[2] - bbox[0] + M * 4
                labels = item[5] if len(item) > 5 else {}
                widest = 0
                for p in sorted(item[2]):
                    txt = str(labels.get(p, p))
                    tb = draw.textbbox((0, 0), txt, font=self._font)
                    widest = max(widest, tb[2] - tb[0])
                button_w = max(LH - 4, widest + 8)
                max_w = max(max_w, header_w, M + 16 + button_w + M * 2)

        # Count lines including dropdown expansion + ptype_row stack.
        n_lines = len(items)
        dropdown_extra = 0
        ptype_extra = 0
        max_dd_visible = max(4, (self._fb_height // LH) - n_lines - 4)
        for item in items:
            if item[0] == "ptype_row":
                # Header + one row per available ptype (the header counts
                # as the 1 already in n_lines, so just add the N rows).
                ptype_extra += len(item[2])
            elif item[0] == "dropdown" and self._dropdown_open and item[4] == self._dropdown_open:
                n_opts = len(item[3])
                dropdown_extra = min(n_opts, max_dd_visible)
                if n_opts > max_dd_visible:
                    dropdown_extra += 2

        tw = max_w + M * 2
        th = (n_lines + dropdown_extra + ptype_extra) * LH + M * 2

        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        # Panel chrome: rounded card with a hairline border.
        _rounded(draw, [(0, 0), (tw - 1, th - 1)], s.radius, fill=s.bg_color,
                 outline=DarkTheme.C_255_255_255_30, width=1)
        y = M
        self._widgets = []
        r_widget = max(4, s.radius // 2)

        for item in items:
            t = item[0]

            if t == "text":
                draw.text((M, y), item[1], fill=s.text_color, font=self._font)
                y += LH

            elif t == "kv":
                # Two-column row: key (accent) | description (text color).
                draw.text((M, y), item[1], fill=s.accent_color, font=self._font)
                draw.text((M + kv_key_w + 24, y), item[2], fill=s.text_color, font=self._font)
                y += LH

            elif t == "button":
                _, label, key = item
                bbox = draw.textbbox((0, 0), label, font=self._font)
                bw = bbox[2] - bbox[0] + 20
                _rounded(draw, [(M, y + 2), (M + bw, y + LH - 2)], r_widget,
                         fill=s.slider_btn, outline=DarkTheme.C_255_255_255_60)
                draw.text((M + 10, y), label, fill=s.text_color, font=self._font)
                self._widgets.append((y, y + LH, "button", key, M, M + bw))
                y += LH

            elif t == "button_row":
                # Horizontal row of rounded buttons, each with its own
                # x hit range. item[1] = [(label, key), ...]; an entry
                # may carry a third "active" flag for a lit state.
                x = M
                for entry in item[1]:
                    label, bkey = entry[0], entry[1]
                    active = len(entry) > 2 and entry[2]
                    bbox = draw.textbbox((0, 0), label, font=self._font)
                    bw = bbox[2] - bbox[0] + 28
                    fill = s.accent_color if active else s.slider_btn
                    txt_col = DarkTheme.C_15_18_28_255 if active else s.text_color
                    _rounded(draw, [(x, y + 2), (x + bw, y + LH - 2)], r_widget,
                             fill=fill, outline=DarkTheme.C_255_255_255_60)
                    draw.text((x + 14, y), label, fill=txt_col, font=self._font)
                    self._widgets.append((y, y + LH, "hbutton", bkey, x, x + bw))
                    x += bw + 8
                y += LH

            elif t == "toggle":
                _, label, state, key = item
                # Switch-style pill with a sliding knob.
                pill_w = int(LH * 1.5)
                pill_h = LH - 8
                py0 = y + (LH - pill_h) // 2
                track = s.toggle_on_color if state else DarkTheme.C_60_64_78_255
                _rounded(draw, [(M, py0), (M + pill_w, py0 + pill_h)],
                         pill_h // 2, fill=track)
                kr = pill_h - 4
                kx = M + pill_w - kr - 2 if state else M + 2
                draw.ellipse([kx, py0 + 2, kx + kr, py0 + 2 + kr],
                             fill=DarkTheme.C_240_242_248_255)
                draw.text((M + pill_w + 10, y), label, fill=s.text_color, font=self._font)
                self._widgets.append((y, y + LH, "toggle", key))
                y += LH

            elif t == "dropdown":
                _, label, current, options, key = item
                is_open = self._dropdown_open == key
                arrow = "v" if is_open else ">"
                row_bg = s.dropdown_hover if is_open else None
                if row_bg:
                    _rounded(draw, [(M - 4, y + 1), (tw - M + 4, y + LH - 1)],
                             r_widget, fill=DarkTheme.C_255_255_255_18)
                draw.text((M, y), f"{arrow} {label}: ", fill=s.accent_color, font=self._font)
                lab_w = draw.textlength(f"{arrow} {label}: ", font=self._font)
                draw.text((M + lab_w, y), str(current), fill=s.text_color, font=self._font)
                self._widgets.append((y, y + LH, "dropdown_header", key))
                y += LH
                if is_open:
                    n_opts = len(options)
                    scrollable = n_opts > max_dd_visible
                    scroll = self._dropdown_scroll.get(key, 0)
                    if scrollable:
                        scroll = max(0, min(scroll, n_opts - max_dd_visible))
                        self._dropdown_scroll[key] = scroll
                        arrow_color = s.accent_color if scroll > 0 else DarkTheme.C_80_80_80_255
                        draw.text((M + 30, y), f"^ ({scroll} more)", fill=arrow_color, font=self._font)
                        self._widgets.append((y, y + LH, "dropdown_scroll", key, -3))
                        y += LH
                    vis_start = scroll if scrollable else 0
                    vis_end = min(vis_start + max_dd_visible, n_opts)
                    for i, opt in enumerate(options[vis_start:vis_end]):
                        if opt == current:
                            _rounded(draw, [(M + 10, y), (tw - M, y + LH - 1)],
                                     r_widget, fill=s.dropdown_hover)
                        elif i % 2 == 0:
                            draw.rectangle([(M + 10, y), (tw - M, y + LH - 1)],
                                           fill=DarkTheme.C_255_255_255_10)
                        draw.text((M + 15, y), str(opt), fill=s.text_color, font=self._font)
                        self._widgets.append((y, y + LH, "dropdown_item", key, opt))
                        y += LH
                    if scrollable:
                        remaining = n_opts - vis_end
                        arrow_color = s.accent_color if remaining > 0 else DarkTheme.C_80_80_80_255
                        draw.text((M + 30, y), f"v ({remaining} more)", fill=arrow_color, font=self._font)
                        self._widgets.append((y, y + LH, "dropdown_scroll", key, 3))
                        y += LH

            elif t == "slider":
                _, label, value, vmin, vmax, key = item
                btn_w = LH
                # - button
                _rounded(draw, [(M, y + 2), (M + btn_w, y + LH - 2)], r_widget,
                         fill=s.slider_btn)
                mw = draw.textlength("-", font=self._font)
                draw.text((M + (btn_w - mw) / 2, y), "-", fill=s.text_color, font=self._font)
                self._widgets.append((y, y + LH, "slider_dec", key, vmin, vmax))
                # + button
                rx = tw - M - btn_w
                _rounded(draw, [(rx, y + 2), (tw - M, y + LH - 2)], r_widget,
                         fill=s.slider_btn)
                pw = draw.textlength("+", font=self._font)
                draw.text((rx + (btn_w - pw) / 2, y), "+", fill=s.text_color, font=self._font)
                self._widgets.append((y, y + LH, "slider_inc", key, vmin, vmax))
                # Label left of the track, track fills the remaining gap.
                text = f"{label}: {value:.2f}"
                tx_text = M + btn_w + 10
                draw.text((tx_text, y), text, fill=s.text_color, font=self._font)
                tw_text = draw.textlength(text, font=self._font)
                tx0, tx1 = int(tx_text + tw_text + 10), rx - 10
                tcy = y + LH // 2
                if tx1 - tx0 > 12 and vmax > vmin:
                    frac = min(1.0, max(0.0, (value - vmin) / (vmax - vmin)))
                    _rounded(draw, [(tx0, tcy - 3), (tx1, tcy + 3)], 3,
                             fill=DarkTheme.C_50_54_68_255)
                    fx = tx0 + int((tx1 - tx0) * frac)
                    if fx > tx0:
                        _rounded(draw, [(tx0, tcy - 3), (fx, tcy + 3)], 3,
                                 fill=s.accent_color)
                y += LH

            elif t == "ptype_row":
                _, label, available, selected, key, labels = item
                # Header line + one chip per available ptype, stacked
                # vertically and indented under the header.
                draw.text((M, y), f"{label}:", fill=s.text_color, font=self._font)
                y += LH
                indent = M + 16
                for p in sorted(available):
                    txt = str(labels.get(p, p))
                    tb = draw.textbbox((0, 0), txt, font=self._font)
                    tw_ = tb[2] - tb[0]
                    box_w = max(LH - 4, tw_ + 14)
                    on = p in selected
                    fill = s.toggle_on_color if on else s.field_bg
                    txt_col = DarkTheme.C_15_18_28_255 if on else s.text_color
                    _rounded(draw, [(indent, y + 2), (indent + box_w, y + LH - 4)],
                             (LH - 6) // 2, fill=fill,
                             outline=DarkTheme.C_255_255_255_60)
                    draw.text(
                        (indent + (box_w - tw_) // 2, y),
                        txt,
                        fill=txt_col,
                        font=self._font,
                    )
                    self._widgets.append((y, y + LH, "ptype_tick", key, p, indent, indent + box_w))
                    y += LH

            elif t == "field":
                _, label, value, key = item
                active = getattr(self, '_editing', None) == key
                label_text = f"{label}:"
                bbox = draw.textbbox((0, 0), label_text, font=self._font)
                label_w = bbox[2] - bbox[0] + 10
                draw.text((M, y), label_text, fill=s.text_color, font=self._font)
                field_x = M + label_w
                field_bg = s.field_active if active else s.field_bg
                _rounded(draw, [(field_x, y), (tw - M, y + LH - 2)], r_widget,
                         fill=field_bg,
                         outline=s.accent_color if active else DarkTheme.C_255_255_255_40)
                draw.text((field_x + 8, y), value,
                          fill=s.accent_color if active else s.text_color, font=self._font)
                self._widgets.append((y, y + LH, "field", key))
                y += LH

        # Finalize: compute panel bounds, upload texture, build VAO
        self._panel_w = tw
        self._panel_h = th
        data = img.tobytes()

        self._panel_x, self._panel_y = self._panel_origin(tw, th)

        # Drop shadow (Item 1): paste the panel over a +3px offset dark
        # rounded rect. Widget hit-testing keeps the unshadowed size.
        canvas = Image.new("RGBA", (tw + 4, th + 4), DarkTheme.TRANSPARENT)
        cdraw = ImageDraw.Draw(canvas)
        _rounded(cdraw, [(3, 3), (tw + 2, th + 2)], s.radius,
                 fill=DarkTheme.SHADOW)
        canvas.alpha_composite(img, (0, 0))
        self._upload_panel(tw + 4, th + 4, canvas.tobytes())

    def _panel_origin(self, tw, th):
        """Top-left pixel of the panel for the style's anchor position."""
        fb_w, fb_h = self._fb_width, self._fb_height
        pos = self.style.position
        if pos == "top-right":
            x, y = fb_w - tw - 10, 10
        elif pos == "top-left":
            x, y = 10, 10
        elif pos == "top-center":
            x, y = (fb_w - tw) // 2, 10
        elif pos == "center":
            x, y = (fb_w - tw) // 2, (fb_h - th) // 2
        elif pos == "center-right":
            x, y = fb_w - tw - 10, (fb_h - th) // 2
        elif pos == "center-left":
            x, y = 10, (fb_h - th) // 2
        elif pos == "bottom-right":
            x, y = fb_w - tw - 10, fb_h - th - 10
        elif pos == "bottom-center":
            x, y = (fb_w - tw) // 2, fb_h - th - 10
        else:  # bottom-left
            x, y = 10, fb_h - th - 10
        ox, oy = self.anchor_offset
        return x + int(ox), y + int(oy)

    def _upload_panel(self, tw, th, data):
        """Upload PIL image to GPU and build vertex data.
        Concrete subclass (WGPU panel mixin) overrides this with the
        wgpu upload path; the base method is a no-op so headless tests
        and any code that constructs a Panel without a backend works.
        """
        pass

    def _hit_test(self, x, y):
        """Convert screen coords to panel-local and find widget. Returns widget tuple or None."""
        lx = x - self._panel_x
        ly = y - self._panel_y
        if lx < 0 or lx > self._panel_w or ly < 0 or ly > self._panel_h:
            if self._dropdown_open:
                self._dropdown_open = None
                return "outside_close"
            return None
        for widget in self._widgets:
            if widget[0] <= ly < widget[1]:
                # ptype_tick widgets share a row; disambiguate by x bounds.
                if widget[2] == "ptype_tick":
                    if widget[5] <= lx < widget[6]:
                        return widget
                    continue
                # Toolbar-style horizontal buttons also share a row.
                if widget[2] == "hbutton":
                    if widget[4] <= lx < widget[5]:
                        return widget
                    continue
                return widget
        return "inside_miss"

    def _handle_base_click(self, widget, lx):
        """Handle common widget click types. Returns True if handled."""
        wtype = widget[2]
        if wtype == "button":
            key = widget[3]
            if key == "_toggle_minimize":
                self._minimized = not self._minimized
                return True
            if isinstance(key, str) and key.startswith("_sec_"):
                getattr(self, key)()
                return True
        if wtype == "dropdown_header":
            key = widget[3]
            self._dropdown_open = None if self._dropdown_open == key else key
            return True
        if wtype == "dropdown_scroll":
            key, delta = widget[3], widget[4]
            self._dropdown_scroll[key] = self._dropdown_scroll.get(key, 0) + delta
            return True
        if wtype in ("slider_dec", "slider_inc"):
            key, vmin, vmax = widget[3], widget[4], widget[5]
            if lx < self._panel_w // 3:
                return ("slider_dec", key, vmin, vmax)
            elif lx > self._panel_w * 2 // 3:
                return ("slider_inc", key, vmin, vmax)
            return True  # clicked in middle of slider, consume but do nothing
        return None

    def render(self):
        """Render hook overridden by the wgpu panel mixin."""
        pass

    def release(self):
        self._tex = None

    def on_scroll(self, yoffset):
        """Handle scroll for open dropdowns. Returns True if consumed."""
        if self._dropdown_open:
            key = self._dropdown_open
            delta = -3 if yoffset > 0 else 3
            self._dropdown_scroll[key] = self._dropdown_scroll.get(key, 0) + delta
            return True
        return False


class DevOverlay(Panel):
    """Developer HUD with performance info and interactive toggles."""

    def __init__(self):
        super().__init__(DEV_STYLE)
        self.enabled = False
        self._last_message = ""

    def update(self, renderer, camera, fps, render_mode_name, cmap_name, timings, message, **kwargs):
        if not self.enabled:
            return
        self._camera = camera
        self._last_message = message

        items = []
        n_vis = renderer.n_particles
        n_tot = renderer.n_total
        scale = "log" if renderer.log_scale else "linear"

        smooth_fps = kwargs.get('smooth_fps', 0)
        pid_str = f"  (PID: {smooth_fps:.0f})" if smooth_fps > 0 else ""
        items.append(("text", f"FPS: {fps:.0f}{pid_str}"))
        items.append(("text", f"Particles: {n_vis:,} / {n_tot:,}"))
        items.append(("text", f"Budget: {renderer._subsample_max_per_frame/1e6:.1f}M"))
        items.append(("text", f"Cull: {timings.get('cull',0)*1000:.0f}ms  Upload: {timings.get('upload',0)*1000:.0f}ms  Render: {timings.get('render',0)*1000:.0f}ms"))
        if renderer.log_scale:
            range_str = f"10^{renderer.qty_min:.2f} .. 10^{renderer.qty_max:.2f}"
        else:
            range_str = f"{renderer.qty_min:.3g} .. {renderer.qty_max:.3g}"
        items.append(("text", f"Render: {render_mode_name}  Scale: {scale}"))
        items.append(("text", f"Range: {range_str}"))
        items.append(("text", f"Colormap: {cmap_name}"))
        items.append(("text", f"Pos: ({camera.position[0]:.2f}, {camera.position[1]:.2f}, {camera.position[2]:.2f})"))
        items.append(("text", f"Speed: {camera.speed:.3g}"))
        if getattr(renderer, "n_stars", 0) > 0:
            ext = "ON" if getattr(renderer, "star_extinction_enabled", False) else "OFF"
            band = getattr(renderer, "star_band", "?")
            items.append(("text", f"Stars: {renderer.n_stars}  Band: {band}  Ext: {ext}  (B/⇧B cycle, O toggle)"))
        items.append(("text", ""))

        items.append(("toggle", "Invert Mouse", camera.invert_mouse, "cam:invert_mouse"))
        items.append(("toggle", "Skip Vsync", renderer.skip_vsync, "skip_vsync"))
        items.append(("toggle", "Auto LOD", renderer.auto_lod, "auto_lod"))
        if renderer.auto_lod:
            items.append(("slider", "Target FPS", renderer.target_fps, 1.0, 60.0, "target_fps"))
            items.append(("slider", "LOD Smooth (s)", renderer.auto_lod_smooth, 0.05, 2.0, "auto_lod_smooth"))
            items.append(("slider", "PID Kp", renderer.pid_Kp, 0.0, 10.0, "pid_Kp"))
            items.append(("slider", "PID Ki", renderer.pid_Ki, 0.0, 2.0, "pid_Ki"))
            items.append(("slider", "PID Kd", renderer.pid_Kd, 0.0, 2.0, "pid_Kd"))
        items.append(("slider", "Hsml Scale", renderer.hsml_scale, 0.1, 5.0, "hsml_scale"))
        items.append(("slider", "Multigrid Levels",
                      float(getattr(renderer, "multigrid_levels", 1)),
                      1.0, 8.0, "multigrid_levels"))
        # Sink/star controls live in the dedicated panel now (toggle with K).
        if getattr(renderer, "n_stars", 0) > 0:
            items.append(("text", "Sink controls: press K"))

        if message:
            items.append(("text", message))

        mem = kwargs.get("mem")
        if mem:
            items.append(("text", "--- memory ---"))
            for k, v in mem.items():
                items.append(("kv", k, v))
        self.render_panel(items)

    def render(self):
        if not self.enabled:
            return
        super().render()

    def on_click(self, x, y, renderer):
        if not self.enabled:
            return False
        hit = self._hit_test(x, y)
        if hit is None:
            return False
        if hit == "outside_close" or hit == "inside_miss":
            return True

        widget = hit
        wtype = widget[2]
        lx = x - self._panel_x

        # Common widget handling
        base = self._handle_base_click(widget, lx)
        if base is True:
            return True
        if isinstance(base, tuple):
            # Slider
            action, key, vmin, vmax = base
            cur = getattr(renderer, key, 1.0)
            # Log-scale unbounded sliders: each click multiplies/divides
            # by a fixed factor. Useful for quantities spanning many
            # orders of magnitude (star brightness, world radius).
            if key in ("star_world_radius", "star_intensity"):
                factor = 1.5
                if action == "slider_dec":
                    setattr(renderer, key, max(cur / factor, 1e-12))
                else:
                    setattr(renderer, key, cur * factor)
                return True
            if key == "multigrid_levels":
                # Integer slider; rebuilds bin state via the renderer
                # so the change applies immediately on the next frame.
                cur_i = int(round(cur))
                new_i = cur_i - 1 if action == "slider_dec" else cur_i + 1
                new_i = max(int(vmin), min(int(vmax), new_i))
                if hasattr(renderer, "set_multigrid_levels"):
                    renderer.set_multigrid_levels(new_i)
                else:
                    setattr(renderer, key, new_i)
                return True
            step = max((vmax - vmin) / 20, 0.01)
            if action == "slider_dec":
                setattr(renderer, key, max(vmin, cur - step))
            else:
                setattr(renderer, key, min(vmax, cur + step))
            return True

        if wtype == "toggle":
            key = widget[3]
            if key.startswith("cam:"):
                attr = key[4:]
                setattr(self._camera, attr, not getattr(self._camera, attr))
            else:
                setattr(renderer, key, not getattr(renderer, key))
            return True

        return True


class SinkOverlay(Panel):
    """Panel for sink/star marker rendering: pick mode (realistic PSF
    vs. flat marker) and tune the per-mode controls. Toggled with K."""

    def __init__(self):
        super().__init__(SINK_STYLE)
        self.enabled = False
        # Numeric-field text-edit state (mirrors UserMenu's pattern):
        # self._editing is the key of the field currently being edited,
        # or None. _edit_buffer holds the digits typed so far. The
        # renderer ref is captured at click time so on_key/on_char can
        # commit without re-routing through wgpu_app.
        self._editing = None
        self._edit_buffer = ""
        self._renderer_ref = None

    def update(self, renderer):
        if not self.enabled:
            return
        self._renderer_ref = renderer
        if self._minimized:
            self.render_panel([("button", "[+] Sink", "_toggle_minimize")])
            return
        if getattr(renderer, "n_stars", 0) == 0:
            self.render_panel([
                ("button", "[-] Sink", "_toggle_minimize"),
                ("text", "(no sink/star particles loaded)"),
            ])
            return

        avail_fields = sorted((getattr(renderer, "_star_fields", {}) or {}).keys())
        size_opts = avail_fields if avail_fields else [renderer.sink_size_field]
        color_opts = ["None"] + avail_fields

        items = [
            ("button", "[-] Sink", "_toggle_minimize"),
            ("text", f"Sink rendering — {renderer.n_stars:,} particles"),
        ]
        items.append(("toggle", "Marker Mode (off=PSF)",
                      bool(renderer.sink_marker_mode), "sink_marker_mode"))
        items.append(("dropdown", "Size Field",
                      renderer.sink_size_field, size_opts, "sink_size_field"))
        items.append(("slider", "Star Radius",
                      renderer.star_world_radius, 0.001, 5.0, "star_world_radius"))
        items.append(("slider", "Size Exponent",
                      renderer.sink_size_exponent, 0.0, 2.0, "sink_size_exponent"))

        if renderer.sink_marker_mode:
            items.append(("slider", "Opacity",
                          renderer.sink_opacity, 0.0, 1.0, "sink_opacity"))
            items.append(("slider", "Border Width",
                          renderer.sink_border_frac, 0.0, 0.5, "sink_border_frac"))
            items.append(("dropdown", "Color Field",
                          renderer.sink_color_field, color_opts, "sink_color_field"))
            if renderer.sink_color_field == "None":
                # No field selected → fall back to a uniform RGB fill.
                items.append(("text", "Fill color"))
                items.append(("slider", "  R", renderer.sink_fill_r, 0.0, 1.0, "sink_fill_r"))
                items.append(("slider", "  G", renderer.sink_fill_g, 0.0, 1.0, "sink_fill_g"))
                items.append(("slider", "  B", renderer.sink_fill_b, 0.0, 1.0, "sink_fill_b"))
            else:
                from .colormaps import AVAILABLE_COLORMAPS
                items.append(("dropdown", "Cmap", renderer.sink_cmap_name,
                              list(AVAILABLE_COLORMAPS), "sink_cmap"))
                # Min/Max are bare numeric ranges; the slider widget can't
                # span the dynamic range of arbitrary sink data, but ±20-
                # range click-stepping handles most cases. Switch the Log
                # toggle to dial in log10 space.
                prefix = "log " if renderer.sink_log_scale else ""
                if renderer.sink_log_scale:
                    vmin_lo, vmax_lo = -20.0, 20.0
                else:
                    ext = max(abs(renderer.sink_qty_min), abs(renderer.sink_qty_max), 1.0)
                    vmin_lo, vmax_lo = -10.0 * ext, 10.0 * ext
                items.append(("slider", f"{prefix}Min",
                              renderer.sink_qty_min, vmin_lo, vmax_lo, "sink_qty_min"))
                items.append(("slider", f"{prefix}Max",
                              renderer.sink_qty_max, vmin_lo, vmax_lo, "sink_qty_max"))
                items.append(("toggle", "Log scale", renderer.sink_log_scale, "sink_log_scale"))
            items.append(("text", "Border color"))
            items.append(("slider", "  R", renderer.sink_border_r, 0.0, 1.0, "sink_border_r"))
            items.append(("slider", "  G", renderer.sink_border_g, 0.0, 1.0, "sink_border_g"))
            items.append(("slider", "  B", renderer.sink_border_b, 0.0, 1.0, "sink_border_b"))
        else:
            items.append(("slider", "Star Intensity",
                          renderer.star_intensity, 0.0, 100.0, "star_intensity"))
            band = getattr(renderer, "star_band", "?")
            ext = "ON" if getattr(renderer, "star_extinction_enabled", False) else "OFF"
            items.append(("text", f"Band: {band}  (B/⇧B to cycle)"))
            items.append(("text", f"Extinction: {ext}  (O to toggle)"))

        # ---- Trajectories (RAMSES sink_*.txt files) ----
        traj_data = getattr(renderer, "_sink_trajectory_data", None) or {}
        if traj_data:
            avail_ids = sorted(traj_data.keys())
            max_n = min(8, len(avail_ids))
            n_active = len(getattr(renderer, "_traj_slots", []))
            items.append(("text", f"Trajectories ({len(avail_ids)} sink IDs available)"))
            items.append(("slider", "# Trajectories",
                          float(n_active), 0.0, float(max_n), "_n_trajectories"))
            items.append(("slider", "Line width (px)",
                          float(getattr(renderer, "_traj_line_width", 2.0)),
                          0.5, 20.0, "_traj_line_width"))
            a_start = float(getattr(renderer, "_traj_start_aexp", 0.0))
            # Show the corresponding redshift so users who don't think
            # in scale factors can still read the cutoff at a glance.
            # z = 1/a - 1; clamp a away from 0 so we don't divide.
            z_start = (1.0 / max(a_start, 1e-6) - 1.0) if a_start > 0 else float("inf")
            z_str = f"z={z_start:.2f}" if a_start > 0 else "z=∞"
            items.append(("text", f"  Trim trajectories before  ({z_str})"))
            if self._editing == "_traj_start_aexp":
                a_display = self._edit_buffer + "_"
            else:
                a_display = f"{a_start:.4f}"
            items.append(("field", "  Start scale factor (a)",
                          a_display, "_traj_start_aexp"))
            id_opts = [str(i) for i in avail_ids]
            for i, slot in enumerate(renderer._traj_slots):
                sid = slot.get("sink_id")
                items.append(("dropdown", f"  T{i+1} ID",
                              str(sid) if sid is not None else "(none)",
                              id_opts, f"_traj_id_{i}"))
                items.append(("slider", f"    R", slot["r"], 0.0, 1.0, f"_traj_r_{i}"))
                items.append(("slider", f"    G", slot["g"], 0.0, 1.0, f"_traj_g_{i}"))
                items.append(("slider", f"    B", slot["b"], 0.0, 1.0, f"_traj_b_{i}"))

        self.render_panel(items)

    def render(self):
        if not self.enabled:
            return
        super().render()

    def on_click(self, x, y, renderer):
        if not self.enabled:
            return False
        hit = self._hit_test(x, y)
        if hit is None:
            return False
        if hit in ("outside_close", "inside_miss"):
            return True

        widget = hit
        wtype = widget[2]
        lx = x - self._panel_x

        base = self._handle_base_click(widget, lx)
        if base is True:
            return True
        if isinstance(base, tuple):
            action, key, vmin, vmax = base
            # Trajectory controls have keys "_n_trajectories", "_traj_r_<i>" etc.
            # Route them through the renderer's setters so the buffers
            # get rebuilt immediately.
            if key == "_n_trajectories":
                cur = len(renderer._traj_slots)
                renderer.set_n_trajectories(cur - 1 if action == "slider_dec" else cur + 1)
                return True
            if key == "_traj_start_aexp":
                cur_a = float(getattr(renderer, "_traj_start_aexp", 0.0))
                step = (vmax - vmin) / 20
                new_a = max(vmin, cur_a - step) if action == "slider_dec" else min(vmax, cur_a + step)
                renderer.set_traj_start_aexp(new_a)
                return True
            if key == "_traj_line_width":
                cur_w = float(getattr(renderer, "_traj_line_width", 2.0))
                step = (vmax - vmin) / 20
                new_w = max(vmin, cur_w - step) if action == "slider_dec" else min(vmax, cur_w + step)
                renderer.set_traj_line_width(new_w)
                return True
            if key.startswith("_traj_") and "_" in key[6:]:
                # _traj_<r|g|b>_<i>
                _, _, channel, idx_str = key.split("_", 3)
                idx = int(idx_str)
                if 0 <= idx < len(renderer._traj_slots):
                    cur_v = renderer._traj_slots[idx][channel]
                    step = (vmax - vmin) / 20
                    new_v = max(vmin, cur_v - step) if action == "slider_dec" else min(vmax, cur_v + step)
                    kw = {channel: new_v}
                    renderer.set_traj_slot_color(idx, **kw)
                return True
            cur = getattr(renderer, key, 1.0)
            # Log-scale clicks for quantities spanning many decades.
            if key in ("star_world_radius", "star_intensity"):
                factor = 1.5
                if action == "slider_dec":
                    setattr(renderer, key, max(cur / factor, 1e-12))
                else:
                    setattr(renderer, key, cur * factor)
                return True
            step = max((vmax - vmin) / 20, 0.01)
            if action == "slider_dec":
                setattr(renderer, key, max(vmin, cur - step))
            else:
                setattr(renderer, key, min(vmax, cur + step))
            return True

        if wtype == "toggle":
            key = widget[3]
            setattr(renderer, key, not getattr(renderer, key))
            return True

        if wtype == "dropdown_item":
            key, value = widget[3], widget[4]
            if key == "sink_cmap":
                from .colormaps import colormap_to_texture_data
                renderer.sink_cmap_name = value
                renderer.set_sink_colormap(colormap_to_texture_data(value))
            elif key == "sink_size_field":
                renderer.set_sink_size_field(value)
            elif key == "sink_color_field":
                renderer.set_sink_color_field(value)
            elif key.startswith("_traj_id_"):
                idx = int(key.split("_")[-1])
                try:
                    sid = int(value)
                except ValueError:
                    sid = None
                renderer.set_traj_slot_id(idx, sid)
            self._dropdown_open = None
            return True

        if wtype == "field":
            key = widget[3]
            self._editing = key
            if key == "_traj_start_aexp":
                self._edit_buffer = f"{getattr(renderer, '_traj_start_aexp', 0.0):.4f}"
            else:
                self._edit_buffer = ""
            return True
        return True

    def on_key(self, key, action):
        """Handle text-field edits (Escape / Enter / Backspace)."""
        import glfw
        if self._editing is None:
            return False
        if action not in (glfw.PRESS, glfw.REPEAT):
            return True
        if key == glfw.KEY_ESCAPE:
            self._editing = None
            self._edit_buffer = ""
            return True
        if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            self._commit_edit()
            return True
        if key == glfw.KEY_BACKSPACE:
            self._edit_buffer = self._edit_buffer[:-1]
            return True
        return True

    def on_char(self, codepoint):
        """Append typed digits / exponent characters to the edit buffer."""
        if self._editing is None:
            return False
        ch = chr(codepoint)
        if ch in "0123456789.eE+-":
            self._edit_buffer += ch
            return True
        if ch in ("\r", "\n"):
            self._commit_edit()
            return True
        return False

    def _commit_edit(self):
        if self._editing and self._edit_buffer and self._renderer_ref is not None:
            try:
                val = float(self._edit_buffer)
                if self._editing == "_traj_start_aexp":
                    # Scale factor must be non-negative; allow values >1
                    # (the filter just clips everything in that case).
                    self._renderer_ref.set_traj_start_aexp(max(0.0, val))
            except ValueError:
                pass
        self._editing = None
        self._edit_buffer = ""


class ToolbarOverlay(Panel):
    """Top-left toolbar with one-click actions for the most common
    operations. on_click returns the action key string for the app to
    dispatch ("auto_range", "screenshot", "record", "bookmark",
    "help"), or True/False for consumed/ignored clicks.
    """

    def __init__(self):
        super().__init__(TOOLBAR_STYLE)
        self.enabled = True

    def update(self, recording=False, orbiting=False, drawer_mode=None,
               aperture=False):
        if not self.enabled:
            return
        rec_label = "Stop" if recording else "Rec"
        self.render_panel([
            ("button_row", [
                ("Auto-range", "auto_range"),
                ("Screenshot", "screenshot"),
                ("Figure", "publication"),
                (rec_label, "record", recording),
                ("Help", "help"),
            ]),
            ("button_row", [
                ("Inspect", "inspector", drawer_mode == "inspector"),
                ("Phase", "phase", drawer_mode == "phase"),
                ("Profile", "profile", drawer_mode == "profile"),
                ("Stats", "stats", drawer_mode == "stats"),
                ("Filters", "filters", drawer_mode == "filters"),
                ("Aperture", "aperture", aperture),
                ("Orbit", "orbit", orbiting),
                ("Cutout", "export_region"),
                ("FITS", "fits_map"),
            ]),
        ])

    def render(self):
        if not self.enabled:
            return
        super().render()

    def on_click(self, x, y):
        if not self.enabled:
            return False
        hit = self._hit_test(x, y)
        if hit is None:
            return False
        if hit in ("outside_close", "inside_miss"):
            return hit == "inside_miss"
        if hit[2] == "hbutton":
            return hit[3]
        return True


class HelpOverlay(Panel):
    """Centered keybinding cheatsheet, toggled with F1 or H."""

    def __init__(self):
        super().__init__(HELP_STYLE)
        self.enabled = False

    def update(self):
        if not self.enabled:
            return
        from .keymap import KEYBINDINGS

        items = [("text", "vizmo controls"), ("text", "")]
        items += [("kv", k, desc) for k, desc in KEYBINDINGS]
        items += [("text", ""), ("text", "F1 / H / Esc to close")]
        self.render_panel(items)

    def render(self):
        if not self.enabled:
            return
        super().render()

    def on_click(self, x, y, _renderer=None):
        """Any click inside the panel closes it."""
        if not self.enabled:
            return False
        hit = self._hit_test(x, y)
        if hit is None:
            return False
        self.enabled = False
        return True


class UserMenu(Panel):
    """Always-visible user menu with weight field, limits, scale, colorbar."""

    SECTION_KEYS = ("types", "mode", "fields", "color", "lod")

    def __init__(self):
        super().__init__(USER_STYLE)
        self.show_colorbar = False
        self._editing = None
        self._edit_buffer = ""
        self._app_ref = None
        self._cbar_tex = None
        # Left-sidebar sections (Item 3): open/closed state persisted
        # in the session settings.
        from .session import load_settings

        saved = load_settings().get("sidebar_sections", {})
        self.sections = {k: bool(saved.get(k, True))
                         for k in self.SECTION_KEYS}
        self.pending_tool = None  # render-mode icon row tool requests

    def _toggle_section(self, key):
        self.sections[key] = not self.sections[key]
        self._last_items_key = None
        try:
            from .session import load_settings, save_settings

            st = load_settings()
            st["sidebar_sections"] = self.sections
            save_settings(st)
        except Exception:
            pass

    def _sec_types(self):
        self._toggle_section("types")

    def _sec_mode(self):
        self._toggle_section("mode")

    def _sec_fields(self):
        self._toggle_section("fields")

    def _sec_color(self):
        self._toggle_section("color")

    def _sec_lod(self):
        self._toggle_section("lod")

    def on_key(self, key, action):
        import glfw
        if self._editing is None:
            return False
        if action not in (glfw.PRESS, glfw.REPEAT):
            return True
        if key == glfw.KEY_ESCAPE:
            self._editing = None
            self._edit_buffer = ""
            return True
        if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            if self._app_ref is not None:
                self._commit_edit(self._app_ref)
            return True
        if key == glfw.KEY_BACKSPACE:
            self._edit_buffer = self._edit_buffer[:-1]
            return True
        return True

    def on_char(self, codepoint, app):
        if self._editing is None:
            return False
        ch = chr(codepoint)
        if ch in "0123456789.eE+-":
            self._edit_buffer += ch
            return True
        if ch in ("\r", "\n"):
            self._commit_edit(app)
            return True
        return False

    def _slot_index(self, key):
        """Return 0 for L: prefix, 1 for C: prefix, None otherwise."""
        if key.startswith("L:"):
            return 0
        if key.startswith("C:"):
            return 1
        return None

    def _handle_slot_dropdown(self, app, slot_idx, field_key, value):
        """Handle a dropdown selection for a composite slot."""
        s = app._slot[slot_idx]
        if field_key == "mode":
            s["mode"] = value
            s["resolve"] = {"SurfaceDensity": 0, "WeightedAverage": 1, "WeightedVariance": 2}[value]
        elif field_key == "weight":
            s["weight"] = value
        elif field_key == "weight2":
            s["weight2"] = value
        elif field_key == "op":
            s["op"] = value
        elif field_key == "data":
            s["data"] = value
        elif field_key == "proj":
            s["proj"] = value
        # Reset limits to trigger auto-range on next _apply_render_mode
        s["min"] = -1.0
        s["max"] = 3.0

    def _commit_edit(self, app):
        if self._editing and self._edit_buffer:
            try:
                val = float(self._edit_buffer)
                slot_idx = self._slot_index(self._editing)
                if slot_idx is not None:
                    field_key = self._editing[2:]
                    if field_key.startswith("log "):
                        field_key = field_key[4:]
                    if "Min" in field_key or "min" in field_key:
                        app._slot[slot_idx]["min"] = val
                    elif "Max" in field_key or "max" in field_key:
                        app._slot[slot_idx]["max"] = val
                elif self._editing == "min":
                    app.renderer.qty_min = val
                elif self._editing == "max":
                    app.renderer.qty_max = val
            except ValueError:
                pass
        self._editing = None
        self._edit_buffer = ""

    def _build_slot_items(self, slot, prefix, all_fields, vf_set, slot_modes, vprojs):
        """Build widget items for a composite slot."""
        items = []
        s = slot
        items.append(("dropdown", f"{prefix}Mode", s["mode"], slot_modes, f"{prefix}mode"))
        items.append(("dropdown", f"{prefix}Weight", s["weight"], all_fields, f"{prefix}weight"))
        if s["mode"] == "SurfaceDensity":
            items.append(("dropdown", f"{prefix}Op", s["op"], self._SD_OPS, f"{prefix}op"))
            items.append(("dropdown", f"{prefix}Field2", s["weight2"], ["None"] + all_fields, f"{prefix}weight2"))
        else:
            items.append(("dropdown", f"{prefix}Data", s["data"], all_fields, f"{prefix}data"))
        # Vector projection
        uses_vec = (s["weight"] in vf_set
                    or (s["mode"] == "SurfaceDensity" and s["weight2"] in vf_set)
                    or (s["mode"] != "SurfaceDensity" and s["data"] in vf_set))
        if uses_vec:
            items.append(("dropdown", f"{prefix}Proj", s["proj"], vprojs, f"{prefix}proj"))
        # Limits
        lo = f"{s['min']:.2f}" if s["log"] else f"{s['min']:.3g}"
        hi = f"{s['max']:.2f}" if s["log"] else f"{s['max']:.3g}"
        if self._editing == f"{prefix}min":
            lo = self._edit_buffer + "_"
        if self._editing == f"{prefix}max":
            hi = self._edit_buffer + "_"
        lbl = "log " if s["log"] else ""
        items.append(("field", f"{prefix}{lbl}Min", lo, f"{prefix}min"))
        items.append(("field", f"{prefix}{lbl}Max", hi, f"{prefix}max"))
        items.append(("toggle", f"{prefix}Log", s["log"], f"{prefix}log"))
        return items

    def update(self, renderer, cmap_name, colormaps,
               sd_fields=None, sd_field="Masses",
               sd_field2="None", sd_op="*", sd_ops=None,
               render_modes=None, render_mode_name="SurfaceDensity",
               wa_data_field="Masses",
               vector_fields=None, vector_projection="LOS", vector_projections=None,
               composite_slots=None,
               available_ptypes=None, selected_ptypes=None,
               ptype_labels=None,
               fov=None, cam_speed=None, **kwargs):
        self._SD_OPS = sd_ops or ["*"]
        if self._minimized:
            self.render_panel([("button", "[+] Splat", "_toggle_minimize")])
            return
        items = [("button", "[-] Splat", "_toggle_minimize")]

        # All field dropdowns include vector field names
        vf_set = set(vector_fields or [])
        all_fields = list(sd_fields or [])
        for vf in (vector_fields or []):
            if vf not in all_fields:
                all_fields.append(vf)
        vprojs = vector_projections or ["LOS", "|v|", "|v|^2"]

        ptype_counts = kwargs.get("ptype_counts") or {}

        def _hdr(label, key):
            caret = "v" if self.sections[key] else ">"
            items.append(("button", f"{caret} {label}", f"_sec_{key}"))

        _hdr("PARTICLE TYPES", "types")
        if self.sections["types"] and available_ptypes is not None:
            items.append((
                "ptype_row", "Types",
                set(available_ptypes), set(selected_ptypes or set()),
                "ptypes", dict(ptype_labels or {}),
            ))
            for pt in sorted(selected_ptypes or []):
                name = (ptype_labels or {}).get(pt, f"type {pt}")
                n = ptype_counts.get(pt, 0)
                items.append(("kv", f"  {pt} {name}",
                              f"{n / 1e6:.1f}M" if n else "-"))

        _hdr("RENDER MODE", "mode")
        if self.sections["mode"] and render_modes and len(render_modes) > 1:
            short = {"SurfaceDensity": "SD", "WeightedAverage": "WA",
                     "WeightedVariance": "WV", "Composite": "CO"}
            items.append(("button_row", [
                (short.get(m, m[:2]), ("set_mode", m),
                 m == render_mode_name) for m in render_modes]))
            items.append(("button_row", [
                ("SL", ("tool", "slice")), ("IS", ("tool", "isosurface")),
                ("ST", ("tool", "streamlines")),
                ("VL", ("tool", "volume"))]))

        _hdr("FIELDS", "fields")

        if not self.sections["fields"]:
            composite_slots = None if render_mode_name != "Composite" else composite_slots
        if (render_mode_name == "Composite" and composite_slots
                and self.sections["fields"]):
            # Two stacked slot panels
            slot_modes = ["SurfaceDensity", "WeightedAverage", "WeightedVariance"]
            items.append(("text", "--- Lightness ---"))
            items.extend(self._build_slot_items(composite_slots[0], "L:", all_fields, vf_set, slot_modes, vprojs))
            items.append(("text", "--- Color ---"))
            items.extend(self._build_slot_items(composite_slots[1], "C:", all_fields, vf_set, slot_modes, vprojs))
            items.append(("dropdown", "Cmap", cmap_name, colormaps, "colormap"))
            items.append(("toggle", "Colorbar", self.show_colorbar, "colorbar"))
        else:
            # Single-field mode
            if (self.sections["fields"] and all_fields
                    and len(all_fields) > 1):
                items.append(("dropdown", "Weight", sd_field, all_fields, "sd_field"))
                if render_mode_name == "SurfaceDensity":
                    items.append(("dropdown", "Op", sd_op, sd_ops or ["*"], "sd_op"))
                    items.append(("dropdown", "Field 2", sd_field2, ["None"] + all_fields, "sd_field2"))
                else:
                    items.append(("dropdown", "Data", wa_data_field, all_fields, "wa_data_field"))

            _hdr("COLOR", "color")
            uses_vector = self.sections["color"] and (sd_field in vf_set
                           or (render_mode_name == "SurfaceDensity" and sd_field2 in vf_set)
                           or (render_mode_name != "SurfaceDensity" and wa_data_field in vf_set))
            if uses_vector:
                items.append(("dropdown", "Proj", vector_projection, vprojs, "vector_projection"))
            if self.sections["color"]:
                items.append(("dropdown", "Cmap", cmap_name, colormaps, "colormap"))

            if self._editing == "min":
                lo_display = self._edit_buffer + "_"
            elif renderer.log_scale:
                lo_display = f"{renderer.qty_min:.2f}"
            else:
                lo_display = f"{renderer.qty_min:.3g}"

            if self._editing == "max":
                hi_display = self._edit_buffer + "_"
            elif renderer.log_scale:
                hi_display = f"{renderer.qty_max:.2f}"
            else:
                hi_display = f"{renderer.qty_max:.3g}"

            prefix = "log " if renderer.log_scale else ""
            if self.sections["color"]:
                items.append(("field", f"{prefix}Min", lo_display, "min"))
                items.append(("field", f"{prefix}Max", hi_display, "max"))
                items.append(("toggle", "Log scale", renderer.log_scale, "log_scale"))
                items.append(("toggle", "Colorbar", self.show_colorbar, "colorbar"))

        # View controls (both modes). Speed is adjusted multiplicatively,
        # so its track is drawn permanently half-filled.
        _hdr("VIEW & LOD", "lod")
        perf = kwargs.get("perf")
        if self.sections["lod"] and perf:
            items.append(("kv", "  perf", perf))
        if not self.sections["lod"]:
            fov = None
            cam_speed = None
        if fov is not None:
            items.append(("slider", "FOV °", float(fov), 10.0, 140.0, "view_fov"))
        if cam_speed is not None:
            items.append(("slider", "Speed", float(cam_speed), 0.0,
                          max(1e-12, 2.0 * float(cam_speed)), "view_speed"))
        if self.sections["lod"]:
            items.append(("slider", "Smoothing", float(renderer.hsml_scale), 0.1, 5.0, "hsml_scale"))

        self._cmap_name = cmap_name
        if render_mode_name == "Composite" and composite_slots:
            # Colorbar uses slot 1 (color channel) limits
            s1 = composite_slots[1]
            self._lo_str = f"{s1['min']:.2f}" if s1["log"] else f"{s1['min']:.3g}"
            self._hi_str = f"{s1['max']:.2f}" if s1["log"] else f"{s1['max']:.3g}"
        else:
            self._lo_str = lo_display.rstrip("_")
            self._hi_str = hi_display.rstrip("_")
        self._renderer = renderer
        self.render_panel(items)

        if self.show_colorbar:
            self._build_colorbar()

    def _build_colorbar(self):
        fb_w, fb_h = self._fb_width, self._fb_height
        LH = self.style.line_height
        cbar_h = max(fb_h // 4, 100)
        cbar_w = 30
        label_pad = 10
        label_w = 200
        total_w = cbar_w + label_pad + label_w
        total_h = cbar_h + LH

        img = Image.new("RGBA", (total_w, total_h), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)

        cbar_top = LH // 2
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(self._cmap_name)
            for j in range(cbar_h):
                t = 1.0 - j / max(cbar_h - 1, 1)
                rgba = cmap(t)
                c = tuple(int(v * 255) for v in rgba[:3]) + (255,)
                draw.rectangle([(0, cbar_top + j), (cbar_w, cbar_top + j)], fill=c)
        except Exception:
            draw.rectangle([(0, cbar_top), (cbar_w, cbar_top + cbar_h)],
                           fill=DarkTheme.C_128_128_128_255)
        draw.rectangle([(0, cbar_top), (cbar_w, cbar_top + cbar_h)], outline=self.style.text_color)

        label_x = cbar_w + label_pad
        draw.text((label_x, cbar_top - 4), self._hi_str, fill=self.style.text_color, font=self._font)
        draw.text((label_x, cbar_top + cbar_h - LH + 4), self._lo_str, fill=self.style.text_color, font=self._font)

        # Hand the rendered colorbar PIL image off to the wgpu mixin
        # which actually uploads it to the GPU.
        self._cbar_data = (total_w, total_h, img.tobytes())

    def on_click(self, x, y, app):
        hit0 = self._hit_test(x, y)
        if (isinstance(hit0, tuple) and len(hit0) > 3
                and hit0[2] == "hbutton"
                and isinstance(hit0[3], tuple)):
            kind, val = hit0[3]
            if kind == "set_mode" and app is not None:
                app._render_mode_name = val
                app._apply_render_mode()
            elif kind == "tool":
                self.pending_tool = val
            self._last_items_key = None
            return True
        self._app_ref = app
        if self._editing:
            self._commit_edit(app)

        hit = self._hit_test(x, y)
        if hit is None:
            return False
        if hit == "outside_close":
            return True
        if hit == "inside_miss":
            return True

        widget = hit
        wtype = widget[2]

        # Common widget handling
        base = self._handle_base_click(widget, x - self._panel_x)
        if base is True:
            return True
        if isinstance(base, tuple):
            action, key, vmin, vmax = base
            if key == "view_fov":
                app.camera.adjust_fov(-5.0 if action == "slider_dec" else +5.0)
            elif key == "view_speed":
                factor = 1.5
                app.camera.speed = (
                    app.camera.speed / factor if action == "slider_dec" else app.camera.speed * factor
                )
            elif key == "hsml_scale":
                step = (vmax - vmin) / 20
                cur = app.renderer.hsml_scale
                new = max(vmin, cur - step) if action == "slider_dec" else min(vmax, cur + step)
                app.renderer.hsml_scale = new
            return True

        if wtype == "ptype_tick":
            p = widget[4]
            if hasattr(app, "_toggle_particle_type"):
                app._toggle_particle_type(p)
            return True

        if wtype == "field":
            key = widget[3]
            self._editing = key
            # Pre-fill edit buffer from the right source
            slot_idx = self._slot_index(key)
            if slot_idx is not None:
                s = app._slot[slot_idx]
                field_key = key[2:]  # strip "L:" or "C:" prefix
                # strip "log " prefix from field key if present
                if field_key.startswith("log "):
                    field_key = field_key[4:]
                if field_key.endswith("Min") or field_key.endswith("min"):
                    self._edit_buffer = f"{s['min']:.4g}"
                elif field_key.endswith("Max") or field_key.endswith("max"):
                    self._edit_buffer = f"{s['max']:.4g}"
            else:
                r = app.renderer
                if key == "min":
                    self._edit_buffer = f"{r.qty_min:.4g}"
                elif key == "max":
                    self._edit_buffer = f"{r.qty_max:.4g}"
            return True

        if wtype == "toggle":
            key = widget[3]
            slot_idx = self._slot_index(key)
            if slot_idx is not None:
                app._slot[slot_idx]["log"] = 1 - app._slot[slot_idx]["log"]
            elif key == "log_scale":
                app.renderer.log_scale = 1 - app.renderer.log_scale
                app._needs_auto_range = True
            elif key == "colorbar":
                self.show_colorbar = not self.show_colorbar
            return True

        if wtype == "dropdown_item":
            key, value = widget[3], widget[4]
            # Composite slot dropdowns
            slot_idx = self._slot_index(key)
            if slot_idx is not None:
                self._handle_slot_dropdown(app, slot_idx, key[2:], value)
            elif key == "render_mode":
                app._render_mode_name = value
                # Set sensible defaults for WeightedVariance
                if value == "WeightedVariance" and app._vector_fields:
                    app._sd_field = "Masses"
                    app._wa_data_field = app._vector_fields[0]  # e.g. "Velocities"
                    app._vector_projection = "LOS"
                app._apply_render_mode()
            elif key == "sd_field":
                app._set_sd_field(value)
            elif key == "sd_field2":
                app._sd_field2 = value
                app._apply_render_mode()
            elif key == "sd_op":
                app._sd_op = value
                if app._sd_field2 != "None":
                    app._apply_render_mode()
            elif key == "wa_data_field":
                app._wa_data_field = value
                app._apply_render_mode()
            elif key == "vector_projection":
                app._vector_projection = value
                app._apply_render_mode()
            elif key == "colormap":
                from .colormaps import AVAILABLE_COLORMAPS
                idx = AVAILABLE_COLORMAPS.index(value) if value in AVAILABLE_COLORMAPS else 0
                app._cmap_idx = idx
                app._set_colormap(value)
            self._dropdown_open = None
            return True

        return True

    # render() / release() come from Panel; the wgpu mixin overrides
    # render to draw the optional colorbar quad alongside the panel.


# ---------------------------------------------------------------------------
# Menu bar (Section 6.D)
# ---------------------------------------------------------------------------

class MenuItem:
    """One row of a dropdown menu."""

    def __init__(self, label="", shortcut="", action=None,
                 separator=False, submenu=None):
        self.label = label
        self.shortcut = shortcut
        self.action = action          # action string dispatched by the app
        self.separator = separator
        self.submenu = submenu        # list[MenuItem] | None


def _sep():
    return MenuItem(separator=True)


def build_default_menus(recent_paths=()):
    """The five standard menus as pure data. The app maps each item's
    action string onto its existing handlers."""
    export_items = [
        MenuItem("Screenshot", "P", "screenshot"),
        MenuItem("Publication Figure", "Ctrl+P", "publication"),
        MenuItem("HDF5 Cutout", "Ctrl+E", "export_region"),
        MenuItem("FITS Map", "Ctrl+M", "fits_map"),
        MenuItem("VTK File", "", "export_vtk"),
        MenuItem("LaTeX Table", "", "stats_latex"),
        MenuItem("All Data (ZIP)", "Ctrl+Shift+E", "export_zip"),
    ]
    recents = [MenuItem(("..." + p[-34:]) if len(p) > 34 else p, "",
                        ("open_recent", p))
               for p in recent_paths] or [MenuItem("(empty)", "", None)]
    return {
        "File": [
            MenuItem("Open...", "Ctrl+O", "open_file"),
            MenuItem("Open Recent", "", submenu=recents),
            _sep(),
            MenuItem("Export", "", submenu=list(export_items)),
            _sep(),
            MenuItem("Quit", "Ctrl+Q", "quit"),
        ],
        "View": [
            MenuItem("Render Mode", "", submenu=[
                MenuItem("Surface Density", "", ("mode", "SurfaceDensity")),
                MenuItem("Weighted Average", "", ("mode", "WeightedAverage")),
                MenuItem("Weighted Variance", "", ("mode", "WeightedVariance")),
                MenuItem("Composite", "", ("mode", "Composite")),
                MenuItem("Slice Plane", "Shift+Z", "slice"),
                MenuItem("Isosurface", "Shift+I", "isosurface_open"),
                MenuItem("Streamlines", "Shift+V", "streamlines_open"),
                MenuItem("Volume", "Shift+W", "volume_open"),
            ]),
            _sep(),
            MenuItem("Split Screen", "Shift+S", "split"),
            _sep(),
            MenuItem("Colormap Browser", "", "colormap_browser"),
            MenuItem("Field Picker", "", "field_picker"),
            _sep(),
            MenuItem("Hide All UI", "Tab", "hide_ui"),
            MenuItem("Science Chrome", "F9", "chrome"),
            MenuItem("Dev Overlay", "\\", "dev_overlay"),
            MenuItem("GPU Profiling", "F10", "profiler"),
        ],
        "Analysis": [
            MenuItem("Inspector", "I", ("drawer", "inspector")),
            MenuItem("Phase Diagram", "G", ("drawer", "phase")),
            MenuItem("Radial Profile", "J", ("drawer", "profile")),
            MenuItem("Region Statistics", "U", ("drawer", "stats")),
            MenuItem("Power Spectrum", "Shift+K", ("drawer", "spectrum")),
            MenuItem("Orbit Integration", "O", ("drawer", "orbit")),
            MenuItem("Sightlines", "Shift+A", "sightline_mode"),
            MenuItem("Sightline List", "", ("drawer", "sightline")),
            MenuItem("Regions", "Ctrl+R", ("drawer", "regions")),
            MenuItem("Field Filters", "F", ("drawer", "filters")),
        ],
        "Export": list(export_items),
        "Help": [
            MenuItem("Help Browser", "F1", "help"),
            MenuItem("Keyboard Shortcuts", "", "help"),
            MenuItem("About", "", "about"),
        ],
    }


MENUBAR_STYLE = PanelStyle(
    font_size=15, line_height=26, margin=10, min_width=10,
    bg_color=DarkTheme.BG_SURFACE,       # DarkTheme.BG_SURFACE
    text_color=DarkTheme.TEXT_PRIMARY,  # DarkTheme.TEXT_PRIMARY
    accent_color=DarkTheme.ACCENT,  # DarkTheme.ACCENT
    toggle_on_color=DarkTheme.ACCENT,
    toggle_off_color=DarkTheme.TEXT_SECONDARY,
    dropdown_bg=DarkTheme.BG_RAISED,    # DarkTheme.BG_RAISED
    dropdown_hover=DarkTheme.ACCENT_DIM,
    slider_btn=DarkTheme.BG_RAISED,
    position="top-left",
    font_family="sans-serif",
    radius=0,
)


class MenuBar(Panel):
    """Top menu bar + dropdown rendering and hit-testing.

    State machine: closed -> menu open (index) -> submenu open
    (index, row). on_click returns the activated item's action (str or
    tuple) for the app to dispatch, True when the click was consumed
    by the bar/menu chrome, or False when it missed entirely (the
    caller should close on miss). Escape closes via on_escape().
    """

    HEIGHT = 26

    def __init__(self, menus=None):
        super().__init__(MENUBAR_STYLE)
        self.enabled = True
        self.menus = menus or build_default_menus()
        self.open_menu = None       # menu name or None
        self.open_submenu = None    # row index whose submenu is open
        self._title_zones = []      # (x0, x1, name)
        self._row_zones = []        # (y0, y1, item, in_submenu)
        self._dd_origin = (0, 0)
        self._sub_origin = (0, 0)

    def get_menu(self, name):
        return self.menus.get(name, [])

    def on_escape(self):
        if self.open_menu is not None:
            self.open_menu = None
            self.open_submenu = None
            self._last_items_key = None
            return True
        return False

    # -- drawing ------------------------------------------------------------

    def update(self):
        if not self.enabled:
            return
        fb_w = max(self._fb_width, 4)
        s = self.style
        bar_h = int(self.HEIGHT * max(self._dpi_scale, 1.0))
        names = list(self.menus.keys())
        key = (tuple(names), self.open_menu, self.open_submenu,
               fb_w, self._fb_height)
        if key == self._last_items_key and self._tex is not None:
            return
        self._last_items_key = key

        # Measure dropdown if open.
        dd_items = self.menus.get(self.open_menu, []) if self.open_menu else []
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        row_h = s.line_height
        dd_w = 0
        for it in dd_items:
            if it.separator:
                continue
            bb = dummy.textbbox((0, 0), it.label, font=self._font)
            sc = dummy.textbbox((0, 0), it.shortcut or "", font=self._font)
            w = (bb[2] - bb[0]) + (sc[2] - sc[0]) + 70
            dd_w = max(dd_w, w)
        dd_h = sum(8 if it.separator else row_h for it in dd_items) + 8

        sub_items = []
        if (self.open_menu and self.open_submenu is not None
                and 0 <= self.open_submenu < len(dd_items)
                and dd_items[self.open_submenu].submenu):
            sub_items = dd_items[self.open_submenu].submenu
        sub_w = 0
        for it in sub_items:
            bb = dummy.textbbox((0, 0), it.label, font=self._font)
            sc = dummy.textbbox((0, 0), it.shortcut or "", font=self._font)
            sub_w = max(sub_w, (bb[2] - bb[0]) + (sc[2] - sc[0]) + 70)
        sub_h = sum(8 if it.separator else row_h for it in sub_items) + 8

        th = bar_h + (dd_h if dd_items else 0) + max(sub_h - dd_h, 0) + 4
        tw = fb_w
        img = Image.new("RGBA", (tw, th), DarkTheme.TRANSPARENT)
        draw = ImageDraw.Draw(img)
        # Bar background + 1px bottom border (DarkTheme.BORDER).
        draw.rectangle([(0, 0), (tw, bar_h - 1)], fill=s.bg_color)
        draw.rectangle([(0, bar_h - 1), (tw, bar_h)],
                       fill=DarkTheme.BORDER)

        self._title_zones = []
        x = s.margin
        for name in names:
            bb = dummy.textbbox((0, 0), name, font=self._font)
            w = bb[2] - bb[0] + 2 * s.margin
            if name == self.open_menu:
                draw.rectangle([(x - 4, 2), (x + w - s.margin, bar_h - 3)],
                               fill=DarkTheme.ACCENT_DIM)
            draw.text((x, (bar_h - s.font_size) // 2 - 2), name,
                      fill=s.text_color, font=self._font)
            self._title_zones.append((x - 4, x + w - s.margin, name))
            x += w + 4

        # Dropdown.
        self._row_zones = []
        if dd_items:
            zone = self._title_zones[names.index(self.open_menu)]
            dx = min(zone[0], tw - dd_w - 6)
            dy = bar_h + 2
            self._dd_origin = (dx, dy)
            draw.rounded_rectangle([dx, dy, dx + dd_w, dy + dd_h],
                                   radius=6, fill=s.dropdown_bg,
                                   outline=DarkTheme.BORDER)
            yy = dy + 4
            for i, it in enumerate(dd_items):
                if it.separator:
                    draw.line([(dx + 8, yy + 3), (dx + dd_w - 8, yy + 3)],
                              fill=DarkTheme.BORDER, width=1)
                    yy += 8
                    continue
                if i == self.open_submenu:
                    draw.rectangle([(dx + 2, yy), (dx + dd_w - 2, yy + row_h)],
                                   fill=DarkTheme.ACCENT_DIM)
                draw.text((dx + 10, yy + 2), it.label, fill=s.text_color,
                          font=self._font)
                tail = "▶" if it.submenu else (it.shortcut or "")
                if tail:
                    bb = dummy.textbbox((0, 0), tail, font=self._font)
                    draw.text((dx + dd_w - 10 - (bb[2] - bb[0]), yy + 2),
                              tail, fill=DarkTheme.TEXT_SECONDARY,
                              font=self._font)
                self._row_zones.append((yy, yy + row_h, i, it, False))
                yy += row_h

            if sub_items:
                row_y = next(z[0] for z in self._row_zones
                             if z[2] == self.open_submenu)
                sx = min(dx + dd_w + 2, tw - sub_w - 4)
                sy = min(row_y, th - sub_h - 2)
                self._sub_origin = (sx, sy)
                draw.rounded_rectangle([sx, sy, sx + sub_w, sy + sub_h],
                                       radius=6, fill=s.dropdown_bg,
                                       outline=DarkTheme.BORDER)
                yy = sy + 4
                for it in sub_items:
                    if it.separator:
                        yy += 8
                        continue
                    draw.text((sx + 10, yy + 2), it.label,
                              fill=s.text_color, font=self._font)
                    if it.shortcut:
                        bb = dummy.textbbox((0, 0), it.shortcut,
                                            font=self._font)
                        draw.text((sx + sub_w - 10 - (bb[2] - bb[0]),
                                   yy + 2), it.shortcut,
                                  fill=DarkTheme.TEXT_SECONDARY,
                                  font=self._font)
                    self._row_zones.append((yy, yy + row_h, -1, it, True))
                    yy += row_h

        self._bar_h = bar_h
        self._panel_w, self._panel_h = tw, th
        self._panel_x, self._panel_y = 0, 0
        self._upload_panel(tw, th, img.tobytes())

    # -- interaction --------------------------------------------------------

    def on_click(self, x, y):
        if not self.enabled:
            return False
        bar_h = getattr(self, "_bar_h", self.HEIGHT)
        if y < bar_h:
            for x0, x1, name in self._title_zones:
                if x0 <= x <= x1:
                    self.open_menu = (None if self.open_menu == name
                                      else name)
                    self.open_submenu = None
                    self._last_items_key = None
                    return True
            return self.open_menu is not None and self._close()
        if self.open_menu is None:
            return False
        for y0, y1, idx, it, in_sub in self._row_zones:
            ox = self._sub_origin[0] if in_sub else self._dd_origin[0]
            # x bound check against the owning dropdown panel
            if y0 <= y <= y1 and x >= ox - 4:
                if it.submenu and not in_sub:
                    self.open_submenu = idx
                    self._last_items_key = None
                    return True
                if it.action is None:
                    return True
                self._close()
                return it.action
        return self._close()

    def _close(self):
        self.open_menu = None
        self.open_submenu = None
        self._last_items_key = None
        return True
