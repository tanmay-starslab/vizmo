"""wgpu-based UI overlay: renders PIL panel images as textured quads.

Provides WGPUDevOverlay and WGPUUserMenu. All PIL rendering and widget
logic is inherited from the base classes; this module only handles the
final GPU upload + draw step.
"""

import numpy as np
import wgpu
from pathlib import Path

from .overlay import DevOverlay, SinkOverlay, UserMenu, HelpOverlay, ToolbarOverlay
from .science_panels import (
    ScaleBar,
    StatusBar,
    ToastOverlay,
    AxesGizmo,
    AnalysisDrawer,
    ApertureOverlay,
    SightlinesOverlay,
    ProfilerOverlay,
)

SHADER_DIR = Path(__file__).parent / "shaders"


class WGPUPanelBackend:
    """Handles wgpu texture upload and quad rendering for a single panel."""

    def __init__(self, device, present_format):
        self.device = device
        self._tex = None
        self._tex_view = None
        self._tex_size = (0, 0)
        self._vbo = None
        self._sampler = device.create_sampler(mag_filter="linear", min_filter="linear")

        shader = device.create_shader_module(code=(SHADER_DIR / "text.wgsl").read_text())
        self._bgl = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "float"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": "filtering"}},
        ])
        layout = device.create_pipeline_layout(bind_group_layouts=[self._bgl])

        self._pipeline = device.create_render_pipeline(
            layout=layout,
            vertex={
                "module": shader, "entry_point": "vs_main",
                "buffers": [{
                    "array_stride": 16, "step_mode": "vertex",
                    "attributes": [
                        {"format": "float32x2", "offset": 0, "shader_location": 0},
                        {"format": "float32x2", "offset": 8, "shader_location": 1},
                    ],
                }],
            },
            primitive={"topology": "triangle-list"},
            fragment={
                "module": shader, "entry_point": "fs_main",
                "targets": [{"format": present_format, "blend": {
                    "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
                    "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha"},
                }}],
            },
        )
        self._bind_group = None

    def upload(self, tw, th, rgba_data, verts):
        """Upload texture + vertex data."""
        dev = self.device
        if self._tex_size != (tw, th):
            self._tex = dev.create_texture(
                size=(tw, th, 1), format="rgba8unorm",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            self._tex_view = self._tex.create_view()
            self._tex_size = (tw, th)
            self._bind_group = dev.create_bind_group(
                layout=self._bgl,
                entries=[
                    {"binding": 0, "resource": self._tex_view},
                    {"binding": 1, "resource": self._sampler},
                ],
            )

        dev.queue.write_texture(
            {"texture": self._tex, "mip_level": 0, "origin": (0, 0, 0)},
            rgba_data, {"bytes_per_row": tw * 4, "rows_per_image": th}, (tw, th, 1),
        )

        vert_bytes = verts.astype(np.float32).tobytes()
        if self._vbo is None or self._vbo.size < len(vert_bytes):
            self._vbo = dev.create_buffer_with_data(
                data=vert_bytes, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST)
        else:
            dev.queue.write_buffer(self._vbo, 0, vert_bytes)

    def render(self, render_pass):
        """Draw panel quad into an active render pass."""
        if self._bind_group is None or self._vbo is None:
            return
        render_pass.set_pipeline(self._pipeline)
        render_pass.set_bind_group(0, self._bind_group)
        render_pass.set_vertex_buffer(0, self._vbo)
        render_pass.draw(6)


class _WGPUPanelMixin:
    """Mixin that overrides Panel's GPU methods with wgpu equivalents."""

    def _init_wgpu(self, device, present_format):
        self._wgpu_backend = WGPUPanelBackend(device, present_format)

    def _upload_panel(self, tw, th, data):
        """Upload the PIL panel image into a wgpu texture and build the
        textured-quad vertex buffer for this frame's draw call.

        Quad placement derives from the same pixel origin used for
        hit-testing (Panel._panel_origin), so clicks and pixels can
        never disagree about where a panel is.
        """
        self._tex = True  # satisfy dirty-flag check in render_panel
        fb_w, fb_h = max(self._fb_width, 1), max(self._fb_height, 1)
        px, py = self._panel_origin(tw, th)
        x1 = px / fb_w * 2 - 1.0
        x2 = (px + tw) / fb_w * 2 - 1.0
        y2 = 1.0 - py / fb_h * 2
        y1 = 1.0 - (py + th) / fb_h * 2

        verts = np.array([
            x1, y1, 0, 1,  x2, y1, 1, 1,  x1, y2, 0, 0,
            x2, y1, 1, 1,  x2, y2, 1, 0,  x1, y2, 0, 0,
        ], dtype=np.float32).reshape(6, 4)

        self._wgpu_backend.upload(tw, th, data, verts)

    def render(self):
        """No-op for base Panel.render() — actual drawing via render_to_pass()."""
        pass

    def render_to_pass(self, render_pass):
        """Draw panel into an active wgpu render pass."""
        self._wgpu_backend.render(render_pass)

    def release(self):
        pass


class WGPUDevOverlay(_WGPUPanelMixin, DevOverlay):
    def __init__(self, device, present_format):
        DevOverlay.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUSinkOverlay(_WGPUPanelMixin, SinkOverlay):
    def __init__(self, device, present_format):
        SinkOverlay.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUHelpOverlay(_WGPUPanelMixin, HelpOverlay):
    def __init__(self, device, present_format):
        HelpOverlay.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUToolbar(_WGPUPanelMixin, ToolbarOverlay):
    def __init__(self, device, present_format):
        ToolbarOverlay.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUScaleBar(_WGPUPanelMixin, ScaleBar):
    def __init__(self, device, present_format):
        ScaleBar.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUStatusBar(_WGPUPanelMixin, StatusBar):
    def __init__(self, device, present_format):
        StatusBar.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUToastOverlay(_WGPUPanelMixin, ToastOverlay):
    def __init__(self, device, present_format):
        ToastOverlay.__init__(self)
        self._init_wgpu(device, present_format)

    def render_to_pass(self, render_pass):
        if self._toasts:
            self._wgpu_backend.render(render_pass)


class WGPUAxesGizmo(_WGPUPanelMixin, AxesGizmo):
    def __init__(self, device, present_format):
        AxesGizmo.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUAnalysisDrawer(_WGPUPanelMixin, AnalysisDrawer):
    def __init__(self, device, present_format):
        AnalysisDrawer.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUApertureOverlay(_WGPUPanelMixin, ApertureOverlay):
    def __init__(self, device, present_format):
        ApertureOverlay.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUSightlinesOverlay(_WGPUPanelMixin, SightlinesOverlay):
    def __init__(self, device, present_format):
        SightlinesOverlay.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUProfilerOverlay(_WGPUPanelMixin, ProfilerOverlay):
    def __init__(self, device, present_format):
        ProfilerOverlay.__init__(self)
        self._init_wgpu(device, present_format)


class WGPUUserMenu(_WGPUPanelMixin, UserMenu):
    def __init__(self, device, present_format):
        UserMenu.__init__(self)
        self._init_wgpu(device, present_format)
        self._cbar_backend = WGPUPanelBackend(device, present_format)

    def _build_colorbar(self):
        """Override: build colorbar PIL image and upload via wgpu."""
        from PIL import Image, ImageDraw
        fb_w, fb_h = self._fb_width, self._fb_height
        LH = self.style.line_height
        cbar_h = max(fb_h // 4, 100)
        cbar_w = 30
        label_pad = 10
        label_w = 200
        total_w = cbar_w + label_pad + label_w
        total_h = cbar_h + LH

        img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
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
                           fill=(128, 128, 128, 255))
        draw.rectangle([(0, cbar_top), (cbar_w, cbar_top + cbar_h)],
                       outline=self.style.text_color)

        label_x = cbar_w + label_pad
        draw.text((label_x, cbar_top - 4), self._hi_str,
                  fill=self.style.text_color, font=self._font)
        draw.text((label_x, cbar_top + cbar_h - LH + 4), self._lo_str,
                  fill=self.style.text_color, font=self._font)

        data = img.tobytes()
        px_w = total_w / fb_w * 2
        px_h = total_h / fb_h * 2
        x1 = -1.0 + 0.01
        x2 = x1 + px_w
        y1 = -px_h / 2
        y2 = px_h / 2

        verts = np.array([
            x1, y1, 0, 1,  x2, y1, 1, 1,  x1, y2, 0, 0,
            x2, y1, 1, 1,  x2, y2, 1, 0,  x1, y2, 0, 0,
        ], dtype=np.float32).reshape(6, 4)

        self._cbar_backend.upload(total_w, total_h, data, verts)

    def render_to_pass(self, render_pass):
        """Draw panel + colorbar into a wgpu render pass."""
        self._wgpu_backend.render(render_pass)
        if self.show_colorbar:
            self._cbar_backend.render(render_pass)
