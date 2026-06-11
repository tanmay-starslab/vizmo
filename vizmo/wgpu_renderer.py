"""wgpu-based renderer: additive accumulation + resolve."""

import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import wgpu

SHADER_DIR = Path(__file__).parent / "shaders"


@dataclass
class RenderMode:
    """Defines how the two accumulation textures are combined in the
    resolve pass. Fragment shaders write
        out_numerator   = sigma * quantity
        out_denominator = sigma           (where sigma = mass * W(r) / h^2)
    The resolve shader displays the denominator (resolve_mode=0), the
    ratio numerator/denominator (resolve_mode=1), or the variance
    sqrt(<f^2> - <f>^2) (resolve_mode=2).
    """

    name: str
    weight_field: str
    qty_field: str
    resolve_mode: int

    @staticmethod
    def surface_density(weight_field="Masses"):
        return RenderMode(
            name=f"Sigma {weight_field}",
            weight_field=weight_field,
            qty_field=weight_field,
            resolve_mode=0,
        )

    @staticmethod
    def mass_weighted_average(qty_field, weight_field="Masses"):
        return RenderMode(
            name=f"<{qty_field}>",
            weight_field=weight_field,
            qty_field=qty_field,
            resolve_mode=1,
        )

    @staticmethod
    def weighted_variance(qty_field, weight_field="Masses"):
        return RenderMode(
            name=f"sigma({qty_field})",
            weight_field=weight_field,
            qty_field=qty_field,
            resolve_mode=2,
        )


_COMMON_WGSL = (SHADER_DIR / "common.wgsl").read_text()


def _load_wgsl(name, include_common=False):
    src = (SHADER_DIR / name).read_text()
    if include_common:
        src = _COMMON_WGSL + "\n" + src
    return src


def _make_bind_group(dev, layout, buffers):
    """Create bind group from layout + ordered list of buffers."""
    return dev.create_bind_group(
        layout=layout,
        entries=[{"binding": i, "resource": {"buffer": b}} for i, b in enumerate(buffers)],
    )


def _storage_bgl(dev, n_buffers, visibility):
    """Create bind group layout with N read-only-storage buffer entries."""
    return dev.create_bind_group_layout(
        entries=[
            {"binding": i, "visibility": visibility, "buffer": {"type": "read-only-storage"}} for i in range(n_buffers)
        ]
    )


def _make_render_pipeline(dev, layout, shader, targets):
    """Create render pipeline for instanced quads (triangle-strip, no vertex buffers)."""
    return dev.create_render_pipeline(
        layout=layout,
        vertex={"module": shader, "entry_point": "vs_main", "buffers": []},
        primitive={"topology": "triangle-strip", "strip_index_format": "uint32"},
        fragment={"module": shader, "entry_point": "fs_main", "targets": targets},
    )


def _additive_blend():
    """Additive blending for accumulation passes."""
    return {
        "color": {"operation": "add", "src_factor": "one", "dst_factor": "one"},
        "alpha": {"operation": "add", "src_factor": "one", "dst_factor": "one"},
    }


# Accumulation texture format.
# r32float with additive blending requires "float32-blendable" feature (not on macOS Metal).
# rgba16float is blendable by default everywhere. Shaders output vec4 so both formats work.
ACCUM_FORMAT = "r32float"
ACCUM_FORMAT_FALLBACK = "rgba16float"


class WGPURenderer:
    """Splat renderer backed by wgpu."""

    KERNELS = ["cubic_spline", "wendland_c2", "gaussian", "quartic", "sphere"]

    def __init__(self, device, canvas_context=None, present_format=None):
        self.device = device
        self.canvas_context = canvas_context
        self.present_format = present_format or "bgra8unorm"

        features = device.features
        if "float32-blendable" in features:
            self._accum_format = ACCUM_FORMAT
        else:
            self._accum_format = ACCUM_FORMAT_FALLBACK
            print(f"  wgpu: float32-blendable not available, using {ACCUM_FORMAT_FALLBACK}")
        self._timestamp_supported = False
        self._last_render_ms = 0.0

        # HUD / UI state
        self.n_particles = 0
        self.n_total = 0
        self.n_stars = 0
        self.qty_min = -1.0
        self.qty_max = 3.0
        self.resolve_mode = 0
        self.log_scale = 1
        self.kernel = "cubic_spline"
        self.auto_lod = True
        self.target_fps = 20.0
        self.auto_lod_smooth = 0.3
        # PID gains for the auto-LOD controller (tunable from the dev panel).
        self.pid_Kp = 1.0
        self.pid_Ki = 0.0
        self.pid_Kd = 0.0
        self.skip_vsync = False
        self.hsml_scale = 1.0
        # Multigrid LOD: 1 = disabled (single-level, original behavior).
        # When > 1, particles are routed to a pyramid of accum textures
        # (level 0 = full res, level k = full_res / 2^k) by their kernel
        # screen radius, and a cascade upsample folds coarser levels back
        # into the finest before resolve.
        self.multigrid_levels = 6
        self.multigrid_n_grid_kernel = 4.0

        # Timing
        self._last_cull_ms = 0.0
        self._last_upload_ms = 0.0
        self._viewport_width = 1024

        # CPU-side particle data (for grid construction only)
        self._all_pos = None
        self._all_hsml = None
        self._all_mass = None
        self._all_qty = None

        # GPU resources
        self._accum_textures = None
        self._accum_textures2 = None  # second set for composite mode
        self._accum_size = (0, 0)
        self._accum_size2 = (0, 0)
        # Multigrid pyramid: list of accum-texture dicts, level 0 unused
        # (it aliases self._accum_textures); level k has res/2^k.
        # Rebuilt by _ensure_fbo when multigrid_levels or size changes.
        self._accum_pyramid = []
        self._accum_pyramid_levels = 0
        self._cascade_bgs = []  # bind groups, one per non-zero level
        # CPU-side star particle data, populated by upload_stars().
        self.n_stars = 0
        self._star_positions = None
        self._star_masses = None

        # Colormap
        self._colormap_tex = None
        self._colormap_tex_view = None
        self._colormap_sampler = None
        self.colormap_tex = None

        # Cached bind groups for resolve / composite passes. Invalidated
        # whenever the FBO triple is recreated or the colormap is reloaded.
        self._resolve_bg = None
        self._composite_bg = None

        self._init_pipelines()

    def _init_pipelines(self):
        """Compile all shader modules and create pipeline layouts."""
        dev = self.device

        # Shader modules (render shaders include common.wgsl for Camera, quad_corner, eval_kernel)
        self._splat_subsample_shader = dev.create_shader_module(
            code=_load_wgsl("splat_subsample.wgsl", include_common=True)
        )
        self._resolve_shader = dev.create_shader_module(code=_load_wgsl("resolve.wgsl"))
        self._composite_shader = dev.create_shader_module(code=_load_wgsl("composite.wgsl"))
        self._cascade_shader = dev.create_shader_module(code=_load_wgsl("cascade.wgsl"))
        self._mg_bin_shader = dev.create_shader_module(code=_load_wgsl("multigrid_bin.wgsl"))

        # Camera uniform: view + proj + viewport + kernel_id + pad
        # + view_rot (rotation-only view matrix used by the
        # precision-preserving splat path) = 208 B
        self._camera_buf = dev.create_buffer(size=208, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        # Resolve params (qty_min, qty_max, mode, log_scale)
        self._resolve_params_buf = dev.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        # Composite params
        self._composite_params_buf = dev.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        VF = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        V = wgpu.ShaderStage.VERTEX

        # bg0: camera + subsample-cull params (both visible to vertex+fragment)
        self._splat_subsample_bgl0 = dev.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": VF, "buffer": {"type": "uniform"}},
                {"binding": 1, "visibility": V, "buffer": {"type": "uniform"}},
            ]
        )
        # bg1: pos / hsml / mass / qty / index / bases / pos_lo per chunk
        self._splat_bgl1 = _storage_bgl(dev, 7, V)

        self._splat_subsample_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._splat_subsample_bgl0, self._splat_bgl1]
        )

        accum_targets = [{"format": self._accum_format, "blend": _additive_blend()}] * 3
        self._splat_subsample_pipeline = _make_render_pipeline(
            dev, self._splat_subsample_layout, self._splat_subsample_shader, accum_targets
        )

        # Cascade pipeline: reads 3 coarse accum textures, additively
        # blends bilinear samples into the next-finer level.
        F = wgpu.ShaderStage.FRAGMENT
        self._cascade_bgl = dev.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": F,
                    "texture": {"sample_type": "unfilterable-float" if self._accum_format == "r32float" else "float"},
                },
                {
                    "binding": 1,
                    "visibility": F,
                    "texture": {"sample_type": "unfilterable-float" if self._accum_format == "r32float" else "float"},
                },
                {
                    "binding": 2,
                    "visibility": F,
                    "texture": {"sample_type": "unfilterable-float" if self._accum_format == "r32float" else "float"},
                },
            ]
        )
        cascade_layout = dev.create_pipeline_layout(bind_group_layouts=[self._cascade_bgl])
        # Multigrid binning compute pipelines.
        C = wgpu.ShaderStage.COMPUTE
        self._mg_bin_bgl = dev.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": C, "buffer": {"type": "uniform"}},
                {"binding": 1, "visibility": C, "buffer": {"type": "read-only-storage"}},
                {"binding": 2, "visibility": C, "buffer": {"type": "read-only-storage"}},
                {"binding": 3, "visibility": C, "buffer": {"type": "storage"}},
                {"binding": 4, "visibility": C, "buffer": {"type": "storage"}},
                {"binding": 5, "visibility": C, "buffer": {"type": "storage"}},
                {"binding": 6, "visibility": C, "buffer": {"type": "storage"}},
            ]
        )
        mg_layout = dev.create_pipeline_layout(bind_group_layouts=[self._mg_bin_bgl])
        self._mg_count_pipeline = dev.create_compute_pipeline(
            layout=mg_layout, compute={"module": self._mg_bin_shader, "entry_point": "cs_count"}
        )
        self._mg_build_pipeline = dev.create_compute_pipeline(
            layout=mg_layout, compute={"module": self._mg_bin_shader, "entry_point": "cs_build"}
        )
        self._mg_scatter_pipeline = dev.create_compute_pipeline(
            layout=mg_layout, compute={"module": self._mg_bin_shader, "entry_point": "cs_scatter"}
        )

        self._cascade_pipeline = dev.create_render_pipeline(
            layout=cascade_layout,
            vertex={"module": self._cascade_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-list"},
            fragment={"module": self._cascade_shader, "entry_point": "fs_main", "targets": accum_targets},
        )

        # Subsample state set later via set_subsample_chunks. The
        # cap-based auto-LOD is the sole LOD knob — there is no
        # separate stride. Each draw renders min(n_total, cap)
        # particles, with each splat enlarged by (n_total/cap)^(1/3)
        # to compensate for the missing neighbors.
        self._subsample_chunks = None
        self._subsample_max_per_frame = 4_000_000
        self._slot_subsample_bgs = [None, None]
        self._active_subsample_slot = None
        self._world_offset = None

        # ---- Realistic stars overlay ----
        self._star_shader = dev.create_shader_module(code=_load_wgsl("star_realistic.wgsl"))
        self._star_params_buf = dev.create_buffer(size=80, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._star_bgl0 = dev.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": VF, "buffer": {"type": "uniform"}},
                {"binding": 1, "visibility": VF, "buffer": {"type": "uniform"}},
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d-array"},
                },
                {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
                # Sink-marker colormap (1D texture sampled in marker mode).
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "1d"},
                },
                {"binding": 5, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
            ]
        )
        # Load baked UBVRI PSFs (5 layers × 162×162) and upload as a
        # texture_2d_array. Falls back to a single-layer Gaussian if the
        # asset is missing.
        self._star_psf_tex, self._star_psf_view, self._star_psf_sampler = self._load_star_psf_texture()
        self._star_bgl1 = _storage_bgl(dev, 1, V)
        star_layout = dev.create_pipeline_layout(bind_group_layouts=[self._star_bgl0, self._star_bgl1])
        # Additive blend over the resolved (sRGB / unorm) screen target.
        star_blend = {
            "color": {"operation": "add", "src_factor": "one", "dst_factor": "one"},
            "alpha": {"operation": "add", "src_factor": "one", "dst_factor": "one"},
        }
        self._star_pipeline = dev.create_render_pipeline(
            layout=star_layout,
            vertex={"module": self._star_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-strip", "strip_index_format": "uint32"},
            fragment={
                "module": self._star_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.present_format, "blend": star_blend}],
            },
        )
        # Marker mode uses standard alpha blending so the fill/border
        # actually cover the gas underneath. With additive blending the
        # border would be invisible (black border adds zero).
        marker_blend = {
            "color": {
                "operation": "add",
                "src_factor": "src-alpha",
                "dst_factor": "one-minus-src-alpha",
            },
            "alpha": {
                "operation": "add",
                "src_factor": "one",
                "dst_factor": "one-minus-src-alpha",
            },
        }
        self._sink_marker_pipeline = dev.create_render_pipeline(
            layout=star_layout,
            vertex={"module": self._star_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-strip", "strip_index_format": "uint32"},
            fragment={
                "module": self._star_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.present_format, "blend": marker_blend}],
            },
        )
        self._star_buf = None
        self._star_bg0 = None
        self._star_bg1 = None
        self._star_buf_dirty = True

        # ---- Sink trajectories (colored thick line ribbons) ----
        # Thick lines via instanced quads: each segment between
        # consecutive trajectory points becomes a 4-vertex triangle
        # strip extruded perpendicular to the segment in screen space.
        # `line_width` (pixels) controls the half-width offset. The
        # trajectory points sit in a storage buffer indexed by
        # instance_index.
        traj_src = """
            struct Camera {
                view: mat4x4<f32>,
                proj: mat4x4<f32>,
                viewport_size: vec2<f32>,
                kernel_id: u32,
                _pad: u32,
            };
            struct TrajParams {
                color: vec4<f32>,
                line_width: f32,
                _pad0: f32,
                _pad1: f32,
                _pad2: f32,
            };
            @group(0) @binding(0) var<uniform> camera: Camera;
            @group(0) @binding(1) var<uniform> params: TrajParams;
            @group(1) @binding(0) var<storage, read> points: array<vec4<f32>>;

            @vertex
            fn vs_main(@builtin(vertex_index) vid: u32,
                       @builtin(instance_index) iid: u32) -> @builtin(position) vec4<f32> {
                let p0 = points[iid].xyz;
                let p1 = points[iid + 1u].xyz;
                // Strip vertices: 0=p0 left, 1=p1 left, 2=p0 right, 3=p1 right.
                let is_end = (vid & 1u) == 1u;
                let side = select(1.0, -1.0, vid >= 2u);
                let p = select(p0, p1, is_end);
                let clip  = camera.proj * (camera.view * vec4<f32>(p,  1.0));
                let c0    = camera.proj * (camera.view * vec4<f32>(p0, 1.0));
                let c1    = camera.proj * (camera.view * vec4<f32>(p1, 1.0));
                // Behind the camera: collapse to a point so the quad gets
                // clipped instead of flailing across the screen.
                if (c0.w <= 0.0 || c1.w <= 0.0) {
                    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
                }
                let n0 = c0.xy / c0.w;
                let n1 = c1.xy / c1.w;
                let dx = n1 - n0;
                if (length(dx) < 1.0e-8) {
                    return clip;
                }
                let dir  = normalize(dx);
                let perp = vec2<f32>(-dir.y, dir.x) * side;
                // Pixel half-width → NDC offset (per-axis division by
                // viewport keeps the line a fixed pixel thickness even
                // when the viewport's aspect changes).
                let half_w_ndc = params.line_width * 0.5 / camera.viewport_size;
                let off = perp * half_w_ndc;
                var out = clip;
                out.x = out.x + off.x * clip.w;
                out.y = out.y + off.y * clip.w;
                return out;
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return params.color;
            }
        """
        self._traj_shader = dev.create_shader_module(code=traj_src)
        self._traj_bgl0 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": VF, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": VF, "buffer": {"type": "uniform"}},
        ])
        self._traj_bgl1 = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX,
             "buffer": {"type": "read-only-storage"}},
        ])
        traj_layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._traj_bgl0, self._traj_bgl1]
        )
        traj_blend = {
            "color": {
                "operation": "add",
                "src_factor": "src-alpha",
                "dst_factor": "one-minus-src-alpha",
            },
            "alpha": {
                "operation": "add",
                "src_factor": "one",
                "dst_factor": "one-minus-src-alpha",
            },
        }
        self._traj_pipeline = dev.create_render_pipeline(
            layout=traj_layout,
            vertex={
                "module": self._traj_shader,
                "entry_point": "vs_main",
                "buffers": [],
            },
            primitive={"topology": "triangle-strip", "strip_index_format": "uint32"},
            fragment={
                "module": self._traj_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.present_format, "blend": traj_blend}],
            },
        )
        # Loaded once by wgpu_app from data.sink_trajectories. Maps
        # sink_id → (positions ndarray (N,3), aexp ndarray (N,)).
        self._sink_trajectory_data = {}
        # Per-slot selection state. SinkOverlay mutates these; renderer
        # rebuilds line buffers via _rebuild_trajectories(). Each slot:
        # {"sink_id": int|None, "r": f, "g": f, "b": f}.
        self._traj_slots = []
        # Built per-trajectory line buffers (vbo + bind group). Mirrors
        # _traj_slots in length but only includes valid entries.
        self._traj_render = []
        # Global start-time cutoff: only include trajectory samples with
        # aexp >= _traj_start_aexp. 0 keeps the whole track; bumping
        # toward 1 trims off early-universe portions that wander far
        # from the current view at galaxy-scale zoom.
        self._traj_start_aexp = 0.0
        # Trajectory line thickness in screen pixels.
        self._traj_line_width = 2.0

        # Tunables — wired into the dev panel later.
        # Physical billboard radius, in world (snapshot) units. The
        # apparent screen size grows as 1/d as the camera approaches,
        # like a real spherical object. Tune from the dev panel.
        self.star_world_radius = 0.2
        self.star_intensity = 10.0
        # Sink/star marker rendering: switch the shader's fragment branch
        # from a PSF lookup to a flat filled disk with a contrasting
        # border ring. World-radius sizing (with √M scaling and pixel
        # floor) is shared between both modes. Fill color comes from
        # sampling a colormap at the per-sink data value (mass by
        # default), with min/max/log controls mirroring the splat path.
        # Border color stays RGB-configurable.
        self.sink_marker_mode = True
        self.sink_border_r = 0.0
        self.sink_border_g = 0.0
        self.sink_border_b = 0.0
        self.sink_border_frac = 0.2
        self.sink_cmap_name = "viridis"
        self.sink_qty_min = 0.0
        self.sink_qty_max = 1.0
        self.sink_log_scale = True
        # Per-sink field selection. Size defaults to the synthesized
        # luminosity (cluster R² for yt sinks, ZAMS L for Gadget), which
        # preserves the existing physical-radius sizing. Color defaults
        # to None → solid fill from sink_fill_rgb sliders (black at
        # startup; user-configurable). Picking any field swaps to a
        # colormap-driven fill.
        self.sink_size_field = "StarLuminosity"
        self.sink_color_field = "None"
        self.sink_fill_r = 0.0
        self.sink_fill_g = 0.0
        self.sink_fill_b = 0.0
        # r_world = world_radius * pow(size_field_value, size_exponent).
        # 0.5 reproduces the original sqrt-area scaling (radius ∝ √L);
        # 1.0 makes radius linear in the field; 0 is constant-size dots.
        self.sink_size_exponent = 0.5
        # Marker-mode fill opacity. PSF mode is additive and ignores it.
        self.sink_opacity = 1.0
        self._star_fields = {}
        self._sink_colormap_tex = None
        self._sink_colormap_tex_view = None
        self._sink_colormap_sampler = dev.create_sampler(
            mag_filter="linear", min_filter="linear",
            address_mode_u="clamp-to-edge",
        )
        # UBVRI dust opacities (cm^2/g, MRN-style averages). Σ in
        # Msun/pc^2 → multiply by 2.0834e-4 to get code-unit κ such
        # that τ = κ_code * Σ.
        _S2C = 2.0834e-4
        self.STAR_BANDS = [
            ("U", 700.0 * _S2C),
            ("B", 440.0 * _S2C),
            ("V", 200.0 * _S2C),
            ("R", 140.0 * _S2C),
            ("I", 100.0 * _S2C),
        ]
        # Index 5 is reserved for the RGB composite mode (I→R, V→G, B→B
        # mapping — standard astronomy false-color recipe).
        self.STAR_RGB_BANDS = (4, 2, 1)  # PSF/κ layer indices for (R, G, B)
        self.star_band_idx = 2  # default V
        self.star_extinction_kappa = self.STAR_BANDS[self.star_band_idx][1]
        # Off by default: building the gas KDTree and computing per-star
        # column densities is expensive on large snapshots and pointless
        # for users who never toggle extinction on. Both the tree build
        # and the column update are gated on this flag.
        self.star_extinction_enabled = False

    N_STAR_MODES = 6  # 5 single-band + 1 RGB composite

    @property
    def star_band(self):
        if self.star_band_idx >= len(self.STAR_BANDS):
            return "RGB"
        return self.STAR_BANDS[self.star_band_idx][0]

    def cycle_star_band(self, step=1):
        self.star_band_idx = (self.star_band_idx + step) % self.N_STAR_MODES
        if self.star_band_idx < len(self.STAR_BANDS):
            self.star_extinction_kappa = self.STAR_BANDS[self.star_band_idx][1]
        self._star_buf_dirty = True
        print(f"  [stars] band={self.star_band}")

    def toggle_star_extinction(self):
        self.star_extinction_enabled = not self.star_extinction_enabled
        # Drop the column cache so the next frame recomputes (when
        # turning on) or stops attenuating (when turning off).
        self._star_columns = None
        self._star_columns_cam_pos = None
        self._star_buf_dirty = True
        print(f"  [stars] extinction {'ON' if self.star_extinction_enabled else 'OFF'}")

    def set_sink_colormap(self, rgba_data):
        """Set the colormap for sink-marker fills, independent of the
        gas-splat colormap. Same RGBA uint8 (N, 4) interface."""
        dev = self.device
        n = len(rgba_data)
        self._sink_colormap_tex = dev.create_texture(
            size=(n, 1, 1),
            format="rgba8unorm",
            dimension="1d",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._sink_colormap_tex_view = self._sink_colormap_tex.create_view()
        # Invalidate the cached bind group so it gets rebuilt with the
        # new texture on the next render.
        self._star_bg0 = None
        dev.queue.write_texture(
            {"texture": self._sink_colormap_tex, "mip_level": 0, "origin": (0, 0, 0)},
            rgba_data.tobytes(),
            {"bytes_per_row": n * 4, "rows_per_image": 1},
            (n, 1, 1),
        )

    def set_colormap(self, rgba_data):
        """Set colormap from RGBA uint8 array of shape (N, 4).

        Args:
            rgba_data: numpy array of shape (N, 4) dtype uint8
        """
        dev = self.device
        n = len(rgba_data)

        self._colormap_tex = dev.create_texture(
            size=(n, 1, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._colormap_tex_view = self._colormap_tex.create_view()
        # Invalidate any cached resolve/composite bind groups that
        # referenced the previous colormap.
        self._resolve_bg = None
        self._composite_bg = None
        dev.queue.write_texture(
            {"texture": self._colormap_tex, "mip_level": 0, "origin": (0, 0, 0)},
            rgba_data.tobytes(),
            {"bytes_per_row": n * 4, "rows_per_image": 1},
            (n, 1, 1),
        )

        self._colormap_sampler = dev.create_sampler(
            mag_filter="linear",
            min_filter="linear",
            address_mode_u="clamp-to-edge",
        )

        # Rebuild resolve/composite pipelines and bind groups since they reference the colormap
        self._build_resolve_pipeline()
        self._build_composite_pipeline()
        self.colormap_tex = True  # compatibility flag

    def _build_resolve_pipeline(self):
        """Build resolve pipeline and bind group layout."""
        if self._colormap_tex is None:
            return
        dev = self.device

        self._resolve_bgl = dev.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}},
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
                {"binding": 5, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
            ]
        )

        resolve_layout = dev.create_pipeline_layout(bind_group_layouts=[self._resolve_bgl])

        self._resolve_pipeline = dev.create_render_pipeline(
            layout=resolve_layout,
            vertex={"module": self._resolve_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-list"},
            fragment={
                "module": self._resolve_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.present_format}],
            },
        )

    def _build_composite_pipeline(self):
        """Build composite pipeline and bind group layout."""
        if self._colormap_tex is None:
            return
        dev = self.device

        self._composite_bgl = dev.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}},
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {
                    "binding": 6,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "unfilterable-float"},
                },
                {"binding": 7, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {"sample_type": "float"}},
                {"binding": 8, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {"type": "filtering"}},
            ]
        )

        composite_layout = dev.create_pipeline_layout(bind_group_layouts=[self._composite_bgl])

        self._composite_pipeline = dev.create_render_pipeline(
            layout=composite_layout,
            vertex={"module": self._composite_shader, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": "triangle-list"},
            fragment={
                "module": self._composite_shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.present_format}],
            },
        )

    # ---- Data management ----

    @staticmethod
    def _safe_f32(arr):
        """Convert *arr* to float32, rescaling if values exceed float32 range.

        Returns (f32_array, scale_factor) where the true values are
        ``f32_array * scale_factor``.  For arrays that already fit in
        float32 the scale factor is 1.0.
        """
        arr = np.asarray(arr, dtype=np.float64)
        amax = np.max(np.abs(arr))
        if amax == 0 or amax < np.finfo(np.float32).max:
            return arr.astype(np.float32), 1.0
        # Normalise so the peak absolute value maps to 1.0; the caller
        # must compensate via a GPU-side scale uniform.
        scale = float(amax)
        return (arr / scale).astype(np.float32), scale

    def set_particles(self, positions, hsml, masses, quantity=None):
        """Store CPU-side particle data. The actual GPU upload is
        performed by GPUCompute.upload_subsample_only on the first
        canvas tick.
        """
        # Stored as float64 so the GPU upload path (which builds the
        # DSFUN90 hi/lo position split) starts from full source
        # precision rather than a pre-truncated f32 copy.
        self._all_pos = np.asarray(positions, dtype=np.float64)
        self._all_hsml, self._hsml_norm = self._safe_f32(hsml)
        self._all_mass, self._mass_norm = self._safe_f32(masses)
        if quantity is None:
            quantity = masses
        self._all_qty, self._qty_norm = self._safe_f32(quantity)
        self.n_total = len(masses)

    def update_weights(self, masses, quantity=None):
        """Update the renderer's CPU mass/qty arrays after a field swap.
        The GPU side must be re-uploaded by GPUCompute.upload_weights.
        """
        self._all_mass, self._mass_norm = self._safe_f32(masses)
        if quantity is None:
            quantity = masses
        self._all_qty, self._qty_norm = self._safe_f32(quantity)

    def set_subsample_chunks(self, chunks, world_offset=None):
        """Configure compute-driven splat: render directly from per-chunk
        source particle buffers, with hash-stride + frustum cull happening
        in the vertex shader.

        Args:
            chunks: list of dicts {"pos","hsml","mass","qty","n","start"}
                from GPUCompute, or None to disable.
            world_offset: (3,) float32. Subtracted from camera position
                before building view matrix and cull params, to match the
                world-origin shift applied to particle positions on
                upload (avoids float32 precision loss for cosmological
                snapshots).
        """
        if chunks is None:
            self._subsample_chunks = None
            self._world_offset = None
            self._slot_subsample_bgs = [None, None]
            self._active_subsample_slot = None
            return
        # Keep the offset in float64: the splat path's hi/lo precision
        # split is undermined if the offset itself is truncated to f32.
        self._world_offset = np.asarray(world_offset, dtype=np.float64) if world_offset is not None else None
        # Reset slot bindings whenever the base chunks change.
        self._slot_subsample_bgs = [None, None]
        self._active_subsample_slot = None
        dev = self.device
        out = []
        n_levels = max(1, int(self.multigrid_levels))
        for cb in chunks:
            # One params buffer (and bg0) per multigrid level. With
            # n_levels=1 this collapses to the original single-level
            # binding.
            params_bufs = []
            bg0s = []
            for _ in range(n_levels):
                # 128 B = 112 prior + 16 for cam_pos_lo (vec3 + pad).
                pb = dev.create_buffer(size=128, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
                params_bufs.append(pb)
                bg0s.append(_make_bind_group(dev, self._splat_subsample_bgl0, [self._camera_buf, pb]))
            # Per-chunk multigrid binning state. Sized for MAX_LEVELS=16.
            MAX_LV = 16
            bin_params_buf = dev.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            counts_buf = dev.create_buffer(size=MAX_LV * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            bases_buf = dev.create_buffer(size=MAX_LV * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            # Identity bases (all zero) for the n_levels==1 path so the
            # default splat shader binding works without the bin pass.
            dev.queue.write_buffer(bases_buf, 0, b"\x00" * (MAX_LV * 4))
            indirect_buf = dev.create_buffer(
                size=MAX_LV * 16,
                usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.COPY_DST),
            )
            bg1 = _make_bind_group(
                dev,
                self._splat_bgl1,
                [cb["pos"], cb["hsml"], cb["mass"], cb["qty"], cb["index"], bases_buf, cb["pos_lo"]],
            )
            mg_bg = dev.create_bind_group(
                layout=self._mg_bin_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": bin_params_buf}},
                    {"binding": 1, "resource": {"buffer": cb["pos"]}},
                    {"binding": 2, "resource": {"buffer": cb["hsml"]}},
                    {"binding": 3, "resource": {"buffer": counts_buf}},
                    {"binding": 4, "resource": {"buffer": bases_buf}},
                    {"binding": 5, "resource": {"buffer": cb["index"]}},
                    {"binding": 6, "resource": {"buffer": indirect_buf}},
                ],
            )
            out.append(
                {
                    "bg0": bg0s[0],
                    "bg0s": bg0s,
                    "bg1": bg1,
                    "params_buf": params_bufs[0],
                    "params_bufs": params_bufs,
                    "pos": cb["pos"],
                    "pos_lo": cb["pos_lo"],
                    "hsml": cb["hsml"],
                    "index": cb["index"],
                    "n": int(cb["n"]),
                    "start": int(cb["start"]),
                    "bin_params_buf": bin_params_buf,
                    "counts_buf": counts_buf,
                    "bases_buf": bases_buf,
                    "indirect_buf": indirect_buf,
                    "mg_bg": mg_bg,
                }
            )
        self._subsample_chunks = out
        self._subsample_n_levels = n_levels
        # Stash the raw source chunks so set_multigrid_levels() can
        # rebuild the per-level bind groups without the caller having
        # to re-upload.
        self._subsample_source_chunks = list(chunks)
        self._subsample_source_offset = world_offset

    def set_multigrid_levels(self, n_levels):
        """Change the multigrid level count and rebuild the per-chunk
        bin state. Cheap (only allocates a few small uniform/storage
        buffers per chunk) and safe to call between frames.

        Preserves slot bind groups across the rebuild — the underlying
        source pos/hsml/mass/qty buffers are unchanged, so the app's
        composite-slot mass/qty bind groups remain valid; we just need
        to rebuild the parallel bgs that reference the (new) chunk
        objects' pos/hsml/index/bases buffers.
        """
        n_levels = max(1, int(n_levels))
        if n_levels == int(getattr(self, "_subsample_n_levels", 1)):
            self.multigrid_levels = n_levels
            return
        self.multigrid_levels = n_levels
        if getattr(self, "_subsample_source_chunks", None) is None:
            return

        # Capture the active slot index before set_subsample_chunks wipes it.
        prev_active = self._active_subsample_slot
        # Drop the previous chunk dicts and rebuild fresh — this gives
        # us new bases_buf storage initialized to all zeros, which is
        # critical when going from multigrid back to n_levels=1 (the
        # splat shader reads s_bases[0] and would otherwise see stale
        # base offsets left over from the prior bin pass).
        self.set_subsample_chunks(self._subsample_source_chunks, world_offset=self._subsample_source_offset)
        if self._accum_size[0] > 0:
            self._ensure_pyramid(*self._accum_size)
        # Restore active slot index so the next set_subsample_slot_chunks
        # call (or render) sees the prior selection.
        self._active_subsample_slot = prev_active
        # NOTE: prev_slot_bgs intentionally dropped — the bg1 layout is
        # unchanged but the bgs reference the OLD chunk dicts' pos/hsml/
        # index/bases buffers. If we kept them, vertex shader reads of
        # s_index/s_bases would see stale (freed) buffers. The caller
        # must re-issue set_subsample_slot_chunks() after this returns;
        # data_manager.py does so on its next render-state refresh.

    def set_subsample_slot_chunks(self, slot_idx, slot_chunks):
        """Bind a composite slot's mass/qty per-chunk buffers. Pos+hsml
        come from the shared self._subsample_chunks; only mass+qty are
        per-slot. Builds a parallel set of bg1 bind groups; the active
        slot is selected via set_active_subsample_slot().
        """
        if not hasattr(self, "_slot_subsample_bgs"):
            self._slot_subsample_bgs = [None, None]
        bgs = []
        for ck, sc in zip(self._subsample_chunks, slot_chunks):
            bg = _make_bind_group(
                self.device,
                self._splat_bgl1,
                [ck["pos"], ck["hsml"], sc["mass"], sc["qty"], ck["index"], ck["bases_buf"], ck["pos_lo"]],
            )
            bgs.append(bg)
        self._slot_subsample_bgs[slot_idx] = bgs

    def set_active_subsample_slot(self, slot_idx):
        """Pick which composite slot's mass/qty bind groups the next
        _render_accum will use. Pass None for the default (chunks' own
        mass/qty, used outside composite mode).
        """
        self._active_subsample_slot = slot_idx

    def set_subsample_max_per_frame(self, max_inst):
        """Set the per-frame instance cap for the compute-driven splat
        path. wgpu_app raises this gradually as refinement progresses
        and resets it on camera motion.
        """
        self._subsample_max_per_frame = max(int(max_inst), 1000)

    def _write_subsample_params(self, camera, stride):
        """Write each chunk's params uniform. Must be called BEFORE
        beginning the render pass (write_buffer cannot run mid-pass).

        `stride` is the (possibly fractional) coarsening factor:
            stride = n_total / budget
        where budget is the total number of instances dispatched across
        all chunks. Each rendered particle stands in for `stride`
        particles' worth of mass.
        """
        if self._subsample_chunks is None:
            return
        import struct as _struct

        ratio = float(stride)
        # User-controlled hsml multiplier (overlay slider) composes with
        # the stride-derived scaling so each splat both fills its share
        # of the subsample volume and honors the manual size knob.
        h_scale = (ratio ** (1.0 / 3.0)) * float(self.hsml_scale) * getattr(self, "_hsml_norm", 1.0)
        mass_scale = ratio * getattr(self, "_mass_norm", 1.0)  # each sampled particle stands in for `stride`
        fov_rad = float(np.radians(camera.fov))
        # Apply the same world-origin shift as the view matrix, in
        # f64, then split into a DSFUN90 hi/lo pair so the shader's
        # `(pos_hi - cam_hi) + (pos_lo - cam_lo)` cancellation reaches
        # ~14 digits of precision. Without the lo term, sub-f32-ULP
        # camera moves vanish in the cast and the camera snaps to the
        # same ~1e-7 * box_extent grid as the particles.
        offset = getattr(self, "_world_offset", None)
        cam_pos_f64 = np.asarray(camera.position, dtype=np.float64)
        if offset is not None:
            cam_pos_f64 = cam_pos_f64 - np.asarray(offset, dtype=np.float64)
        cam_hi = cam_pos_f64.astype(np.float32)
        cam_lo = (cam_pos_f64 - cam_hi.astype(np.float64)).astype(np.float32)
        cam_bytes = _struct.pack(
            "fff f fff f fff f fff f fff f",
            float(cam_hi[0]),
            float(cam_hi[1]),
            float(cam_hi[2]),
            0.0,
            float(camera.forward[0]),
            float(camera.forward[1]),
            float(camera.forward[2]),
            0.0,
            float(camera.right[0]),
            float(camera.right[1]),
            float(camera.right[2]),
            0.0,
            float(camera.up[0]),
            float(camera.up[1]),
            float(camera.up[2]),
            0.0,
            float(cam_lo[0]),
            float(cam_lo[1]),
            float(cam_lo[2]),
            0.0,
        )
        n_levels = int(getattr(self, "_subsample_n_levels", 1))
        full_res_w = float(self._accum_size[0]) if self._accum_size[0] > 0 else 1.0
        n_grid_kernel = float(self.multigrid_n_grid_kernel)
        for ck in self._subsample_chunks:
            for lvl in range(n_levels):
                tail = _struct.pack(
                    "ffII ffII fff f",
                    fov_rad,
                    float(camera.aspect),
                    int(stride),
                    int(ck["n"]),
                    h_scale,
                    mass_scale,
                    int(lvl),
                    int(n_levels),
                    full_res_w,
                    n_grid_kernel,
                    0.0,
                    0.0,
                )
                pbs = ck.get("params_bufs", [ck["params_buf"]])
                self.device.queue.write_buffer(pbs[lvl], 0, cam_bytes + tail)

    def upload_stars(self, positions, masses, luminosity=None):
        """Upload star particle data.

        Keeps the CPU-side count and the position/mass arrays around so
        a future star-rendering path can pick them up without changing
        the call site. The GPU bind groups for the old dedicated star
        shader pipeline have been removed.
        """
        self.n_stars = len(masses)
        # Invalidate any cached extinction columns — star set has changed.
        self._star_columns = None
        self._star_columns_cam_pos = None
        if self.n_stars == 0:
            return
        self._star_positions = positions.astype(np.float64)
        self._star_masses = masses.astype(np.float32)
        if luminosity is None:
            # Fallback to mass as a stand-in luminosity proxy.
            self._star_luminosity = self._star_masses.copy()
        else:
            self._star_luminosity = np.asarray(luminosity, dtype=np.float32)
        self._star_buf_dirty = True

    def set_extinction_gas(self, positions, masses, hsml):
        """Provide gas particle data used to compute per-star extinction
        columns via a direct port of starforge_tools.star_gas_columns.

        Stored as float64 because the KDTree query in star_gas_columns
        runs on these arrays directly.
        """
        self._ext_xgas = np.ascontiguousarray(positions, dtype=np.float64)
        self._ext_mgas = np.ascontiguousarray(masses, dtype=np.float64)
        self._ext_hgas = np.ascontiguousarray(hsml, dtype=np.float64)
        self._ext_tree = None
        self._ext_bins = None  # list of (idx, tree, h_max_in_bin)
        self._ext_hmax = float(self._ext_hgas.max()) if self._ext_hgas.size else 0.0
        self._star_columns = None
        self._star_columns_cam_pos = None

    def _update_star_columns(self, camera):
        """Recompute per-star gas column densities when the camera has
        translated. Pure rotation preserves every star→camera sightline
        (parallel-projection approximation along the camera forward axis,
        which only depends on the camera position relative to the stars
        when forward = normalize(mean_star_pos - cam_pos)), so we key the
        cache on camera.position alone.

        Direct port of starforge_tools.star_gas_columns; result lives in
        self._star_columns and is consumed by the future star draw path.
        """
        if getattr(self, "n_stars", 0) == 0:
            return
        if not self.star_extinction_enabled:
            return
        if getattr(self, "_ext_xgas", None) is None:
            return

        cam_pos = np.asarray(camera.position, dtype=np.float64)
        prev = self._star_columns_cam_pos
        if self._star_columns is not None and prev is not None and np.array_equal(prev, cam_pos):
            return  # rotation-only frame — cached columns are still valid

        # Try the tree-accelerated path (scipy required); fall back to brute force.
        try:
            columns = self._star_columns_tree(cam_pos)
        except ImportError:
            columns = self._star_columns_brute(cam_pos)

        self._star_columns = columns.astype(np.float32)
        self._star_columns_cam_pos = cam_pos.copy()

    def _build_ext_hbins(self):
        """Bin gas by smoothing length into octave-wide bins and build a
        KDTree per bin. This keeps the per-bin h_max within ~2x of the
        bin's h_min, so segment queries don't degenerate when a small
        number of outlier particles have huge h."""
        from scipy.spatial import cKDTree

        h = self._ext_hgas
        if h.size == 0:
            self._ext_bins = []
            return
        pos_h = h[h > 0.0]
        if pos_h.size == 0:
            self._ext_bins = []
            return
        h_min = float(pos_h.min())
        h_max = float(h.max())
        n_oct = max(1, int(np.ceil(np.log2(max(h_max / h_min, 1.0)))) + 1)
        edges = h_min * (2.0 ** np.arange(n_oct + 1))
        edges[-1] = max(edges[-1], h_max * 1.0001)
        bins = []
        for i in range(n_oct):
            lo, hi = edges[i], edges[i + 1]
            idx = np.where((h >= lo) & (h < hi))[0]
            if idx.size == 0:
                continue
            tree = cKDTree(self._ext_xgas[idx])
            bins.append((idx, tree, float(h[idx].max())))
        self._ext_bins = bins

    def _star_columns_tree(self, cam_pos):
        """KDTree-accelerated per-star column densities.

        Gas is binned by h (octave-wide). For each bin, we batch the
        segment-subdivision query points from ALL stars into a single
        cKDTree.query_ball_point(workers=-1) call — small per-(star,bin)
        calls don't give scipy's thread pool enough work to amortize
        spawn overhead, so batching per bin is what actually engages
        multiple cores.
        """
        if self._ext_bins is None:
            self._build_ext_hbins()
        if not self._ext_bins:
            return np.zeros(self._star_positions.shape[0], dtype=np.float64)

        xstar = self._star_positions
        xgas = self._ext_xgas
        mgas = self._ext_mgas
        hgas2 = self._ext_hgas * self._ext_hgas
        inv_pi = 1.0 / np.pi
        n_stars = xstar.shape[0]
        columns = np.zeros(n_stars, dtype=np.float64)

        # Per-star geometry, computed once.
        rays = cam_pos[None, :] - xstar  # (n_stars, 3)
        d_obs_arr = np.linalg.norm(rays, axis=1)  # (n_stars,)
        safe = d_obs_arr > 0.0
        ray_dirs = np.zeros_like(rays)
        ray_dirs[safe] = rays[safe] / d_obs_arr[safe, None]

        # Accumulate per-star candidate index lists across all bins.
        star_cands = [[] for _ in range(n_stars)]

        for bin_idx, tree, h_bin in self._ext_bins:
            seg_len = max(2.0 * h_bin, 1e-30)
            centers_list = []
            owner_list = []  # which star each center belongs to
            max_radius = 0.0
            for s in range(n_stars):
                if not safe[s]:
                    continue
                d_obs = d_obs_arr[s]
                n_chunks = max(1, int(np.ceil(d_obs / seg_len)))
                if n_chunks > 4096:
                    n_chunks = 4096
                step = d_obs / n_chunks
                ts = (np.arange(n_chunks) + 0.5) * step
                centers = xstar[s][None, :] + ts[:, None] * ray_dirs[s][None, :]
                centers_list.append(centers)
                owner_list.append(np.full(n_chunks, s, dtype=np.int32))
                r_here = 0.5 * step + h_bin
                if r_here > max_radius:
                    max_radius = r_here
            if not centers_list:
                continue
            centers_cat = np.concatenate(centers_list, axis=0)
            owners_cat = np.concatenate(owner_list)

            # One big query per bin — enough work for the thread pool.
            lists = tree.query_ball_point(centers_cat, r=max_radius, workers=-1)

            # Distribute results back to their owning stars.
            # lists[i] is a python list of local (in-bin) indices.
            # Group by owner, dedupe, map to global indices.
            # (cKDTree returns a numpy object array of lists.)
            per_star_local = [[] for _ in range(n_stars)]
            for i, lst in enumerate(lists):
                if lst:
                    per_star_local[owners_cat[i]].append(lst)
            for s in range(n_stars):
                if per_star_local[s]:
                    local = np.unique(np.concatenate([np.asarray(l, dtype=np.int64) for l in per_star_local[s]]))
                    if local.size:
                        star_cands[s].append(bin_idx[local])

        for s in range(n_stars):
            if not star_cands[s]:
                continue
            idx = np.concatenate(star_cands[s])  # bins are disjoint
            d_obs = d_obs_arr[s]
            ray_dir = ray_dirs[s]
            rcand = xgas[idx] - xstar[s]
            tcand = rcand @ ray_dir
            r2 = np.einsum("ij,ij->i", rcand, rcand)
            b2 = np.maximum(r2 - tcand * tcand, 0.0)
            h2c = hgas2[idx]
            mask = (tcand > 0.0) & (tcand < d_obs) & (b2 < h2c)
            if not np.any(mask):
                continue
            b2m = b2[mask]
            h2m = h2c[mask]
            q2 = b2m / h2m
            w = 1.0 - q2
            w = w * w
            columns[s] = np.sum(mgas[idx][mask] * (3.0 * inv_pi / h2m) * w)

        return columns

    def _star_columns_brute(self, cam_pos):
        """Brute-force fallback: chunk-vectorized over stars."""
        xstar = self._star_positions
        xgas = self._ext_xgas
        mgas = self._ext_mgas
        hgas2 = self._ext_hgas * self._ext_hgas
        inv_pi = 1.0 / np.pi
        n_stars = xstar.shape[0]
        columns = np.zeros(n_stars, dtype=np.float64)

        n_gas = xgas.shape[0]
        # ~256 MB cap for the (chunk, n_gas) scratch arrays.
        chunk = max(1, int(32 * 1024 * 1024 / max(n_gas, 1)))
        for s0 in range(0, n_stars, chunk):
            s1 = min(s0 + chunk, n_stars)
            xs = xstar[s0:s1]  # (C, 3)
            ray = cam_pos[None, :] - xs  # (C, 3)
            d_obs = np.linalg.norm(ray, axis=1)  # (C,)
            safe = d_obs > 0.0
            if not np.any(safe):
                continue
            ray_dir = np.zeros_like(ray)
            ray_dir[safe] = ray[safe] / d_obs[safe, None]

            # r[c, g, :] = xgas[g] - xs[c]
            # t[c, g] = r · ray_dir[c]
            # |r|² = |xgas|² - 2 xgas·xs + |xs|²
            xs_dot_xs = np.einsum("ij,ij->i", xs, xs)  # (C,)
            xg_dot_xg = np.einsum("ij,ij->i", xgas, xgas)  # (n_gas,)
            xg_dot_xs = xgas @ xs.T  # (n_gas, C)
            r2 = (xg_dot_xg[:, None] - 2.0 * xg_dot_xs + xs_dot_xs[None, :]).T  # (C, n_gas)

            # t = (xgas - xs) · ray_dir = xgas·ray_dir - xs·ray_dir
            xs_dot_dir = np.einsum("ij,ij->i", xs, ray_dir)  # (C,)
            xg_dot_dir = xgas @ ray_dir.T  # (n_gas, C)
            t = (xg_dot_dir - xs_dot_dir[None, :]).T  # (C, n_gas)

            b2 = np.maximum(r2 - t * t, 0.0)
            mask = (t > 0.0) & (t < d_obs[:, None]) & (b2 < hgas2[None, :])
            mask &= safe[:, None]
            if not np.any(mask):
                continue
            q2 = np.where(mask, b2 / hgas2[None, :], 0.0)
            w = 1.0 - q2
            w = w * w
            contrib = np.where(mask, mgas[None, :] * (3.0 * inv_pi / hgas2[None, :]) * w, 0.0)
            columns[s0:s1] = contrib.sum(axis=1)

        return columns

    # ---- Accumulation textures ----

    def _ensure_fbo(self, width, height, which=1):
        """Create or resize the accumulation texture triple."""
        if which == 1:
            if self._accum_size != (width, height) or self._accum_textures is None:
                self._accum_textures = self._create_accum_textures(width, height)
                self._accum_size = (width, height)
                # Resolve bind group references slot-1 accum views.
                self._resolve_bg = None
                self._composite_bg = None
                self._accum_pyramid = []
                self._accum_pyramid_levels = 0
                self._cascade_bgs = []
            self._ensure_pyramid(width, height)
        else:
            if self._accum_size2 == (width, height) and self._accum_textures2 is not None:
                return
            self._accum_textures2 = self._create_accum_textures(width, height)
            self._accum_size2 = (width, height)
            # Composite bind group references slot-2 accum views.
            self._composite_bg = None

    def _dispatch_multigrid_bin(self, camera, encoder, budget, n_total_chunks):
        """Run count → build → scatter for every chunk so each level's
        draw_indirect args and per-level index ranges are ready before
        the splat passes begin.
        """
        import struct as _struct

        dev = self.device
        n_levels = int(self._subsample_n_levels)
        full_res_w = float(self._accum_size[0]) if self._accum_size[0] > 0 else 1.0
        n_grid_kernel = float(self.multigrid_n_grid_kernel)
        offset = getattr(self, "_world_offset", None)
        cam_pos = camera.position - offset if offset is not None else camera.position
        proj = camera.projection_matrix()
        proj11 = float(proj[1, 1])
        # Reuse the same h_scale derivation as the splat path so the
        # binner sees the kernel size the splat will actually draw.
        ratio = max(n_total_chunks / max(budget, 1), 1.0)
        h_scale = (ratio ** (1.0 / 3.0)) * float(self.hsml_scale) * getattr(self, "_hsml_norm", 1.0)

        zero_counts = b"\x00" * (16 * 4)
        for ck in self._subsample_chunks:
            n_to_consider = max(1, int(round(budget * ck["n"] / n_total_chunks)))
            n_to_consider = min(n_to_consider, int(ck["n"]))
            params = _struct.pack(
                "fff f fff f I I f f f f f f",
                float(cam_pos[0]),
                float(cam_pos[1]),
                float(cam_pos[2]),
                0.0,
                float(camera.forward[0]),
                float(camera.forward[1]),
                float(camera.forward[2]),
                0.0,
                int(n_to_consider),
                int(n_levels),
                full_res_w,
                n_grid_kernel,
                proj11,
                h_scale,
                0.0,
                0.0,
            )
            dev.queue.write_buffer(ck["bin_params_buf"], 0, params)
            dev.queue.write_buffer(ck["counts_buf"], 0, zero_counts)

        # Workgroup count cap is 65535 per dimension; with 256-thread
        # groups that's ~16.7 M particles per dispatch. Use a 2D grid
        # so chunks of up to ~4G particles fit in one dispatch.
        for ck in self._subsample_chunks:
            n_to_consider = max(1, int(round(budget * ck["n"] / n_total_chunks)))
            n_to_consider = min(n_to_consider, int(ck["n"]))
            n_groups = (n_to_consider + 255) // 256
            gx = min(n_groups, 65535)
            gy = (n_groups + 65535 - 1) // 65535
            cp = encoder.begin_compute_pass()
            cp.set_bind_group(0, ck["mg_bg"])
            cp.set_pipeline(self._mg_count_pipeline)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.set_pipeline(self._mg_build_pipeline)
            cp.dispatch_workgroups(1, 1, 1)
            cp.set_pipeline(self._mg_scatter_pipeline)
            cp.dispatch_workgroups(gx, gy, 1)
            cp.end()

    def _ensure_pyramid(self, width, height):
        """Build (or rebuild) the multigrid pyramid for slot-1 accum.

        Levels [1..n_levels-1] are coarser than the full-res slot-1
        accum textures (which serve as level 0). Bind groups for the
        cascade pass are built here too — each binds level k's textures
        as inputs (the cascade fragment shader writes additively into
        level k-1, which is the bound render target).
        """
        n_levels = max(1, int(self.multigrid_levels))
        if (
            n_levels == self._accum_pyramid_levels
            and self._accum_pyramid is not None
            and len(self._accum_pyramid) == max(0, n_levels - 1)
            and self._accum_size == (width, height)
        ):
            return
        self._accum_pyramid = []
        self._cascade_bgs = []
        for k in range(1, n_levels):
            w = max(1, width >> k)
            h = max(1, height >> k)
            self._accum_pyramid.append(self._create_accum_textures(w, h))
        # Build per-level cascade bind groups: input = level k textures.
        for k in range(1, n_levels):
            tex = self._accum_pyramid[k - 1]["views"]
            bg = self.device.create_bind_group(
                layout=self._cascade_bgl,
                entries=[
                    {"binding": 0, "resource": tex[0]},
                    {"binding": 1, "resource": tex[1]},
                    {"binding": 2, "resource": tex[2]},
                ],
            )
            self._cascade_bgs.append(bg)
        self._accum_pyramid_levels = n_levels

    def _create_accum_textures(self, width, height):
        """Create a triple of accumulation textures (num, den, sq)."""
        dev = self.device
        textures = []
        views = []
        for _ in range(3):
            tex = dev.create_texture(
                size=(width, height, 1),
                format=self._accum_format,
                usage=(
                    wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC
                ),
            )
            textures.append(tex)
            views.append(tex.create_view())
        return {"textures": textures, "views": views}

    # ---- Camera uniforms ----

    def _write_camera_uniforms(self, camera, width, height):
        """Write camera data to the uniform buffer."""
        # view and proj are column-major in WGSL (same as OpenGL).
        # Apply the world-origin shift (subsample mode pre-translates
        # particles by self._world_offset to avoid float32 precision loss
        # in the view-matrix multiply on cosmological-scale snapshots).
        offset = getattr(self, "_world_offset", None)
        if offset is not None:
            saved = camera.position
            camera.position = saved - np.asarray(offset, dtype=np.float64)
            try:
                view_raw = camera.view_matrix()
            finally:
                camera.position = saved
        else:
            view_raw = camera.view_matrix()
        view = np.ascontiguousarray(view_raw.T, dtype=np.float32)
        # Rotation-only view matrix: the splat_subsample path feeds it
        # a precision-preserving camera-relative offset (rel) instead of
        # the absolute position, so the translation column must be
        # zeroed. Built from the original (un-shifted) view matrix
        # because rotation is independent of camera position.
        view_rot_raw = camera.view_matrix().copy()
        view_rot_raw[:3, 3] = 0.0  # zero translation column (column-major source)
        view_rot = np.ascontiguousarray(view_rot_raw.T, dtype=np.float32)
        proj = np.ascontiguousarray(camera.projection_matrix().T, dtype=np.float32)

        kernel_id = self.KERNELS.index(self.kernel) if self.kernel in self.KERNELS else 0

        data = np.zeros(52, dtype=np.float32)  # 208 bytes / 4
        data[0:16] = view.ravel()
        data[16:32] = proj.ravel()
        data[32] = float(width)
        data[33] = float(height)
        # view_rot at byte offset 144 → float index 36
        data[36:52] = view_rot.ravel()
        # kernel_id and pad as uint32
        data_bytes = bytearray(data.tobytes())
        # Write kernel_id at offset 136 (byte 34*4=136)
        import struct

        struct.pack_into("I", data_bytes, 136, kernel_id)
        struct.pack_into("I", data_bytes, 140, 0)  # padding

        self.device.queue.write_buffer(self._camera_buf, 0, data_bytes)

    # ---- Render passes ----

    def _load_star_psf_texture(self):
        """Load the baked UBVRI PSF stack into a 2D texture array.

        Asset: vizmo/assets/star_psf_ubvri.npy, shape (5, H, W) float32.
        Falls back to a single-layer 2D Gaussian if the asset is missing.
        """
        dev = self.device
        asset = Path(__file__).parent / "assets" / "star_psf_ubvri.npy"
        if asset.exists():
            psf = np.load(asset).astype(np.float32)  # (5, H, W)
        else:
            print(f"  [stars] PSF asset not found at {asset}; using Gaussian fallback")
            H = 64
            yy, xx = np.mgrid[-1 : 1 : H * 1j, -1 : 1 : H * 1j]
            g = np.exp(-8.0 * (xx * xx + yy * yy)).astype(np.float32)
            g /= g.sum()
            psf = np.stack([g] * 5)
        n_layers, H, W = psf.shape
        # Per-layer normalization to peak=1 so star brightness is set
        # by the shader's intensity*flux factor, not by the PSF energy.
        peaks = psf.reshape(n_layers, -1).max(axis=1).reshape(n_layers, 1, 1)
        psf = psf / np.where(peaks > 0, peaks, 1.0)

        # r16float is universally filterable; r32float requires the
        # float32-filterable feature (not present on this backend).
        psf16 = psf.astype(np.float16)
        tex = dev.create_texture(
            size=(W, H, n_layers),
            format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="2d",
        )
        dev.queue.write_texture(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            psf16.tobytes(),
            {"offset": 0, "bytes_per_row": W * 2, "rows_per_image": H},
            (W, H, n_layers),
        )
        view = tex.create_view(dimension="2d-array")
        sampler = dev.create_sampler(
            mag_filter="linear",
            min_filter="linear",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
        )
        return tex, view, sampler

    def set_sink_trajectory_data(self, trajectories):
        """Hand the renderer the per-sink-id positions dict from the
        data manager. Resets the slot list so the next UI interaction
        can pick from the new sink IDs."""
        self._sink_trajectory_data = dict(trajectories or {})
        self._traj_slots = []
        self._rebuild_trajectories()

    def _default_traj_slot(self, slot_idx):
        """Pick a sensible default sink_id + color for a freshly-added
        slot. Color defaults to white; user dials RGB from the panel
        when they want to distinguish multiple tracks."""
        ids = sorted(self._sink_trajectory_data.keys())
        sid = ids[slot_idx % len(ids)] if ids else None
        return {"sink_id": sid, "r": 1.0, "g": 1.0, "b": 1.0}

    def set_n_trajectories(self, n):
        """Resize the slot list to *n*. Existing slots are preserved;
        new slots get default sink_id + colour."""
        n = max(0, int(n))
        while len(self._traj_slots) < n:
            self._traj_slots.append(self._default_traj_slot(len(self._traj_slots)))
        if len(self._traj_slots) > n:
            self._traj_slots = self._traj_slots[:n]
        self._rebuild_trajectories()

    def set_traj_slot_id(self, idx, sink_id):
        if 0 <= idx < len(self._traj_slots):
            self._traj_slots[idx]["sink_id"] = sink_id
            self._rebuild_trajectories()

    def set_traj_slot_color(self, idx, r=None, g=None, b=None):
        if not (0 <= idx < len(self._traj_slots)):
            return
        slot = self._traj_slots[idx]
        if r is not None: slot["r"] = float(r)
        if g is not None: slot["g"] = float(g)
        if b is not None: slot["b"] = float(b)
        self._rebuild_trajectories()

    def set_traj_start_aexp(self, a):
        """Clip every drawn trajectory to samples with aexp >= a."""
        self._traj_start_aexp = float(a)
        self._rebuild_trajectories()

    def set_traj_line_width(self, w):
        """Set trajectory thickness (in screen pixels) and rebuild the
        per-trajectory uniform that carries it."""
        self._traj_line_width = max(0.5, float(w))
        self._rebuild_trajectories()

    def _rebuild_trajectories(self):
        """Reconcile self._traj_render with self._traj_slots — build a
        storage buffer of trajectory points plus a color/line-width
        uniform for each populated slot. Each segment between
        consecutive points becomes one instance of a 4-vertex quad."""
        import struct
        self._traj_render = []
        offset = getattr(self, "_world_offset", None)
        for slot in self._traj_slots:
            sid = slot.get("sink_id")
            if sid is None:
                continue
            entry = self._sink_trajectory_data.get(sid)
            if entry is None:
                continue
            pos, aexp = entry
            if self._traj_start_aexp > 0.0:
                mask = aexp >= self._traj_start_aexp
                pos = pos[mask]
            if len(pos) < 2:
                continue
            p = np.asarray(pos, dtype=np.float32)
            if offset is not None:
                p = p - np.asarray(offset, dtype=np.float32)
            # vec4 per point — vec3 in std430 storage would stride to 16
            # anyway, so just pad with a zero w to keep layout obvious.
            verts = np.zeros((len(p), 4), dtype=np.float32)
            verts[:, :3] = p
            sbo = self.device.create_buffer_with_data(
                data=verts.tobytes(),
                usage=wgpu.BufferUsage.STORAGE,
            )
            color = (slot["r"], slot["g"], slot["b"], 1.0)
            params = struct.pack(
                "f" * 8,
                *color,
                float(self._traj_line_width),
                0.0, 0.0, 0.0,
            )
            pbuf = self.device.create_buffer_with_data(
                data=params,
                usage=wgpu.BufferUsage.UNIFORM,
            )
            bg0 = self.device.create_bind_group(
                layout=self._traj_bgl0,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._camera_buf}},
                    {"binding": 1, "resource": {"buffer": pbuf}},
                ],
            )
            bg1 = self.device.create_bind_group(
                layout=self._traj_bgl1,
                entries=[{"binding": 0, "resource": {"buffer": sbo}}],
            )
            self._traj_render.append({
                "sbo": sbo, "bg0": bg0, "bg1": bg1,
                "n_segments": len(p) - 1,
            })

    def _encode_trajectories(self, encoder, screen_view):
        """Draw active trajectories as instanced thick-line quads."""
        if not self._traj_render or screen_view is None:
            return
        rp = encoder.begin_render_pass(color_attachments=[{
            "view": screen_view,
            "load_op": "load",
            "store_op": "store",
        }])
        rp.set_pipeline(self._traj_pipeline)
        for entry in self._traj_render:
            rp.set_bind_group(0, entry["bg0"])
            rp.set_bind_group(1, entry["bg1"])
            rp.draw(4, entry["n_segments"], 0, 0)
        rp.end()

    def _sink_field(self, which, fallback):
        """Resolve a sink field-selection attribute to an ndarray. Falls
        back to *fallback* when the selection is "None", unknown, or the
        wrong length (e.g. before the field dict has been uploaded)."""
        name = getattr(self, which, None)
        if name is None or name == "None":
            return fallback
        arr = self._star_fields.get(name)
        if arr is None or len(arr) != self.n_stars:
            return fallback
        return arr

    def set_sink_size_field(self, name):
        """Pick which per-sink scalar drives the marker/PSF size."""
        self.sink_size_field = name
        self._star_buf_dirty = True

    def set_sink_color_field(self, name):
        """Pick which per-sink scalar feeds the marker colormap, and
        re-auto-range min/max for the new field. "None" → solid black
        fill (colormap path skipped in the shader)."""
        self.sink_color_field = name
        self._star_buf_dirty = True
        if name == "None":
            return
        arr = self._star_fields.get(name)
        if arr is None or len(arr) == 0:
            return
        valid = arr[np.isfinite(arr)]
        if self.sink_log_scale:
            valid = valid[valid > 0]
        if len(valid) == 0:
            return
        if self.sink_log_scale:
            self.sink_qty_min = float(np.log10(valid.min()))
            self.sink_qty_max = float(np.log10(valid.max()))
        else:
            self.sink_qty_min = float(valid.min())
            self.sink_qty_max = float(valid.max())
        if self.sink_qty_max - self.sink_qty_min < 1e-6:
            self.sink_qty_max = self.sink_qty_min + 1.0

    def _ensure_star_buffer(self):
        """(Re)build the per-star storage buffer holding (xyz, attenuated L)."""
        n = getattr(self, "n_stars", 0)
        if n == 0:
            return False
        if self._star_positions is None or self._star_luminosity is None:
            return False

        # World-origin shift mirrors the splat path so float32 precision
        # holds at cosmological scales.
        offset = getattr(self, "_world_offset", None)
        pos = self._star_positions
        if offset is not None:
            pos = pos - np.asarray(offset, dtype=np.float64)
        # Track the offset so we rebuild if it changes after upload.
        self._star_buf_offset_used = None if offset is None else np.asarray(offset, dtype=np.float64).copy()

        L0 = self._sink_field("sink_size_field", self._star_luminosity).astype(np.float32, copy=False)
        cols = getattr(self, "_star_columns", None)
        have_ext = self.star_extinction_enabled and cols is not None and len(cols) == n

        # Single-band channel: uses the *currently selected* band's κ
        # (or no extinction if disabled or in RGB mode where the single
        # value is just the bolometric L for fallback display).
        lum_single = L0.copy()
        if have_ext and self.star_band_idx < len(self.STAR_BANDS):
            tau = self.star_extinction_kappa * cols
            lum_single = lum_single * np.exp(-tau).astype(np.float32)

        # RGB triple: always computed using the fixed (I, V, B) → (R, G, B)
        # mapping so cycling into composite mode is instantaneous.
        rgb = np.empty((n, 3), dtype=np.float32)
        for ch, layer in enumerate(self.STAR_RGB_BANDS):
            kappa_ch = self.STAR_BANDS[layer][1]
            l = L0.copy()
            if have_ext:
                l = l * np.exp(-kappa_ch * cols).astype(np.float32)
            rgb[:, ch] = l

        # Layout: three vec4 per star.
        # vec4[0] = (x, y, z, L_raw)            ← drives billboard size
        # vec4[1] = (L_R_att, L_G_att, L_B_att, L_single_att)
        # vec4[2] = (sink_data_value, _, _, _)  ← marker-mode colormap lookup
        # The shader divides each L_*_att by L_raw to get a per-channel
        # attenuation factor in [0,1], so per-pixel surface brightness
        # is independent of intrinsic luminosity (which goes entirely
        # into the billboard area via radius ∝ sqrt(L)).
        data = np.zeros((n, 12), dtype=np.float32)
        data[:, 0:3] = pos.astype(np.float32)
        data[:, 3] = L0  # raw bolometric, for sizing
        data[:, 4:7] = rgb
        data[:, 7] = lum_single
        # Marker color driver — whatever per-sink field the user picked.
        # When the color field is "None" the shader skips the colormap
        # sample entirely; the array still has to be filled with
        # *something*, so default to L0 (the size driver) to avoid
        # branching here.
        sink_data = self._sink_field("sink_color_field", L0).astype(np.float32, copy=False)
        if len(sink_data) != n:
            sink_data = L0
        data[:, 8] = sink_data

        nbytes = data.nbytes
        if self._star_buf is None or self._star_buf.size < nbytes:
            self._star_buf = self.device.create_buffer(
                size=nbytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self._star_bg1 = self.device.create_bind_group(
                layout=self._star_bgl1,
                entries=[{"binding": 0, "resource": {"buffer": self._star_buf}}],
            )
        self.device.queue.write_buffer(self._star_buf, 0, data.tobytes())

        if self._star_bg0 is None and self._sink_colormap_tex_view is not None:
            self._star_bg0 = self.device.create_bind_group(
                layout=self._star_bgl0,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._camera_buf}},
                    {"binding": 1, "resource": {"buffer": self._star_params_buf}},
                    {"binding": 2, "resource": self._star_psf_view},
                    {"binding": 3, "resource": self._star_psf_sampler},
                    {"binding": 4, "resource": self._sink_colormap_tex_view},
                    {"binding": 5, "resource": self._sink_colormap_sampler},
                ],
            )
        if self._star_bg0 is None:
            return False
        self._star_buf_dirty = False
        return True

    def _encode_star_overlay(self, encoder, screen_view):
        """Draw realistic-stars billboards over the resolved screen.

        Updates the per-star extinction column cache (cheap on rotation-
        only frames), refreshes the storage buffer when stars or columns
        changed, then issues an instanced quad draw additively blended
        into `screen_view`.
        """
        if getattr(self, "n_stars", 0) == 0:
            return
        cam = getattr(self, "_last_camera", None)
        if cam is None or screen_view is None:
            return

        prev_cam_pos = self._star_columns_cam_pos
        self._update_star_columns(cam)
        # Force a rebuild if either columns refreshed or the world-origin
        # offset changed since the last buffer build.
        cur_off = getattr(self, "_world_offset", None)
        used_off = getattr(self, "_star_buf_offset_used", None)
        offset_changed = (cur_off is None) != (used_off is None) or (
            cur_off is not None
            and used_off is not None
            and not np.array_equal(np.asarray(cur_off, dtype=np.float64), used_off)
        )
        if self._star_buf_dirty or self._star_columns_cam_pos is not prev_cam_pos or offset_changed:
            self._star_buf_dirty = True
        if self._star_buf_dirty:
            if not self._ensure_star_buffer():
                return

        import struct

        # 20 f32s = 80 bytes. Layout matches WGSL StarParams:
        #   00 world_radius, intensity, band_idx, marker_mode
        #   16 border_rgb (vec3, 16-aligned), border_frac
        #   32 sink_qty_min, _max, log_scale, color_active
        #   48 fill_rgb (vec3, 16-aligned), _pad
        #   64 size_exponent, opacity, _pad, _pad
        color_active = 0.0 if self.sink_color_field == "None" else 1.0
        params = struct.pack(
            "f" * 20,
            float(self.star_world_radius),
            float(self.star_intensity),
            float(self.star_band_idx),
            1.0 if self.sink_marker_mode else 0.0,
            float(self.sink_border_r), float(self.sink_border_g), float(self.sink_border_b),
            float(self.sink_border_frac),
            float(self.sink_qty_min),
            float(self.sink_qty_max),
            1.0 if self.sink_log_scale else 0.0,
            float(color_active),
            float(self.sink_fill_r), float(self.sink_fill_g), float(self.sink_fill_b),
            0.0,
            float(self.sink_size_exponent),
            float(self.sink_opacity),
            0.0, 0.0,
        )
        self.device.queue.write_buffer(self._star_params_buf, 0, params)

        rp = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": screen_view,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": "load",
                    "store_op": "store",
                }
            ],
        )
        pipeline = (
            self._sink_marker_pipeline if self.sink_marker_mode
            else self._star_pipeline
        )
        rp.set_pipeline(pipeline)
        rp.set_bind_group(0, self._star_bg0)
        rp.set_bind_group(1, self._star_bg1)
        rp.draw(4, int(self.n_stars), 0, 0)
        rp.end()

    def _render_accum(self, camera, width, height, accum_textures, encoder=None):
        """Render additive accumulation pass into given textures.

        Renders:
          - real particles + anisotropic summaries → full-res `accum_textures`
          - isotropic summary splats → half-res `_accum_textures_lo`

        If `encoder` is None, a fresh command encoder is created and
        submitted at the end of the call (legacy path used by screenshot).
        Otherwise the accum pass is appended to the supplied encoder and
        the caller is responsible for submission.
        """
        self._write_camera_uniforms(camera, width, height)

        # Compute the per-chunk dispatch budget BEFORE beginning the
        # render pass — wgpu forbids write_buffer mid-pass, so all
        # uniform writes must happen now.
        budget = 0
        n_total_chunks = 0
        eff_stride = 1.0
        if self._subsample_chunks is not None:
            n_total_chunks = sum(ck["n"] for ck in self._subsample_chunks)
            cap = self._subsample_max_per_frame
            budget = min(n_total_chunks, cap)
            budget = max(1, budget)
            # Each rendered splat stands in for `eff_stride` particles.
            eff_stride = max(n_total_chunks / max(budget, 1), 1.0)
            self._write_subsample_params(camera, eff_stride)

        dev = self.device
        owns_encoder = encoder is None
        if owns_encoder:
            encoder = dev.create_command_encoder()

        n_levels = max(1, int(getattr(self, "_subsample_n_levels", 1)))

        # Multigrid: bin pass writes per-chunk index buffer + indirect
        # args. Must run before begin_render_pass.
        if n_levels > 1 and self._subsample_chunks is not None and accum_textures is self._accum_textures:
            self._dispatch_multigrid_bin(camera, encoder, budget, n_total_chunks)

        # Pre-resolve the slot bg list once.
        slot = self._active_subsample_slot
        slot_bgs = (
            self._slot_subsample_bgs[slot] if slot is not None and self._slot_subsample_bgs[slot] is not None else None
        )

        # Build a list of per-level (views, bg0_index) pairs. Level 0
        # is the supplied accum_textures (full res); higher levels come
        # from the multigrid pyramid attached to the slot-1 set.
        level_views = [accum_textures["views"]]
        if n_levels > 1 and accum_textures is self._accum_textures:
            for lvl in range(1, n_levels):
                level_views.append(self._accum_pyramid[lvl - 1]["views"])
        elif n_levels > 1:
            # Composite slot or other accum target: fall back to single
            # level for now (multigrid only wired into slot-1 path).
            n_levels = 1

        for lvl in range(n_levels):
            views = level_views[lvl]
            _tsw_a = self.ts_writes(0, 1)
            render_pass = encoder.begin_render_pass(
                **({"timestamp_writes": _tsw_a} if _tsw_a else {}),
                color_attachments=[
                    {"view": views[0], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
                    {"view": views[1], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
                    {"view": views[2], "clear_value": (0, 0, 0, 0), "load_op": "clear", "store_op": "store"},
                ]
            )
            if self._subsample_chunks is not None:
                render_pass.set_pipeline(self._splat_subsample_pipeline)
                for i, ck in enumerate(self._subsample_chunks):
                    instances = max(1, int(round(budget * ck["n"] / n_total_chunks)))
                    instances = min(instances, ck["n"])
                    bg0 = ck.get("bg0s", [ck["bg0"]])[lvl]
                    render_pass.set_bind_group(0, bg0)
                    bg1 = slot_bgs[i] if slot_bgs is not None else ck["bg1"]
                    render_pass.set_bind_group(1, bg1)
                    if n_levels > 1:
                        render_pass.draw_indirect(ck["indirect_buf"], lvl * 16)
                    else:
                        render_pass.draw(4, instances, 0, 0)
            render_pass.end()

        # Cascade: fold coarser levels into finer ones, ending at level 0
        # which the resolve pass consumes.
        if n_levels > 1:
            for lvl in range(n_levels - 1, 0, -1):
                target_views = level_views[lvl - 1]
                rp = encoder.begin_render_pass(
                    color_attachments=[
                        {"view": target_views[0], "load_op": "load", "store_op": "store"},
                        {"view": target_views[1], "load_op": "load", "store_op": "store"},
                        {"view": target_views[2], "load_op": "load", "store_op": "store"},
                    ]
                )
                rp.set_pipeline(self._cascade_pipeline)
                rp.set_bind_group(0, self._cascade_bgs[lvl - 1])
                rp.draw(3, 1, 0, 0)
                rp.end()

        if owns_encoder:
            dev.queue.submit([encoder.finish()])

    def render(self, camera, width, height, encoder=None, screen_view=None, skip_accum=False):
        """Render particle splats via additive accumulation + resolve.

        If `encoder` is provided, the accum + resolve passes are appended
        to it and the caller owns submission. `screen_view` must then also
        be supplied (the swapchain texture view to render into). When both
        are None the legacy path is used: this method creates its own
        encoder, acquires the current swapchain texture, and submits.

        `skip_accum=True` reuses the prior frame's accumulation textures
        (still resident on the GPU) and only re-runs the resolve pass.
        Used by the main loop on UI-only dirty frames so typing into a
        text field doesn't trigger an N-particle re-accum.
        """
        self._viewport_width = width
        self._last_camera = camera
        if self._colormap_tex is None:
            return
        if self.n_particles == 0 and self._subsample_chunks is None:
            return

        self._ensure_fbo(width, height, which=1)

        owns_encoder = encoder is None
        if owns_encoder:
            if self.canvas_context is None:
                return
            encoder = self.device.create_command_encoder()
            current_tex = self.canvas_context.get_current_texture()
            screen_view = current_tex.create_view()

        # Accum pass — appended to the shared encoder. Skipped on
        # UI-only dirty frames where the prior accum textures are still
        # valid.
        if not skip_accum:
            self._render_accum(camera, width, height, self._accum_textures, encoder=encoder)

        import struct

        resolve_data = struct.pack("ffII", self.qty_min, self.qty_max, self.resolve_mode, self.log_scale)
        self.device.queue.write_buffer(self._resolve_params_buf, 0, resolve_data)

        if self._resolve_bg is None:
            self._resolve_bg = self.device.create_bind_group(
                layout=self._resolve_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._resolve_params_buf}},
                    {"binding": 1, "resource": self._accum_textures["views"][0]},
                    {"binding": 2, "resource": self._accum_textures["views"][1]},
                    {"binding": 3, "resource": self._accum_textures["views"][2]},
                    {"binding": 4, "resource": self._colormap_tex_view},
                    {"binding": 5, "resource": self._colormap_sampler},
                ],
            )

        _tsw = self.ts_writes(2, 3)
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": screen_view,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ],
            **({"timestamp_writes": _tsw} if _tsw else {}),
        )
        render_pass.set_pipeline(self._resolve_pipeline)
        render_pass.set_bind_group(0, self._resolve_bg)
        render_pass.draw(3, 1, 0, 0)  # fullscreen triangle
        render_pass.end()
        self._encode_trajectories(encoder, screen_view)
        self._encode_star_overlay(encoder, screen_view)
        if owns_encoder:
            self.device.queue.submit([encoder.finish()])

    def render_composite(
        self, camera, width, height, mode1, min1, max1, log1, mode2, min2, max2, log2, encoder=None, screen_view=None
    ):
        """Composite two pre-filled FBOs.

        Like `render()`, accepts an optional external encoder + swapchain
        view so the caller can bundle this pass with overlay/UI passes
        into a single submit.
        """
        self._viewport_width = width
        self._ensure_fbo(width, height, which=1)
        self._ensure_fbo(width, height, which=2)

        owns_encoder = encoder is None
        if owns_encoder:
            if self.canvas_context is None:
                return
            current_tex = self.canvas_context.get_current_texture()
            screen_view = current_tex.create_view()
            encoder = self.device.create_command_encoder()

        import struct

        comp_data = struct.pack("ffIIffII", min1, max1, mode1, log1, min2, max2, mode2, log2)
        self.device.queue.write_buffer(self._composite_params_buf, 0, comp_data)

        if self._composite_bg is None:
            self._composite_bg = self.device.create_bind_group(
                layout=self._composite_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._composite_params_buf}},
                    {"binding": 1, "resource": self._accum_textures["views"][0]},
                    {"binding": 2, "resource": self._accum_textures["views"][1]},
                    {"binding": 3, "resource": self._accum_textures["views"][2]},
                    {"binding": 4, "resource": self._accum_textures2["views"][0]},
                    {"binding": 5, "resource": self._accum_textures2["views"][1]},
                    {"binding": 6, "resource": self._accum_textures2["views"][2]},
                    {"binding": 7, "resource": self._colormap_tex_view},
                    {"binding": 8, "resource": self._colormap_sampler},
                ],
            )

        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": screen_view,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ],
        )
        render_pass.set_pipeline(self._composite_pipeline)
        render_pass.set_bind_group(0, self._composite_bg)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()
        self._encode_trajectories(encoder, screen_view)
        self._encode_star_overlay(encoder, screen_view)
        if owns_encoder:
            self.device.queue.submit([encoder.finish()])

    def screenshot(self, path, width, height, camera, composite_args=None, quiet=False):
        """Render one frame at (width, height) into an offscreen RGBA8
        texture and save it to `path`.

        Args:
            path: output file path. Format inferred from extension by PIL.
            width, height: framebuffer size in pixels.
            camera: Camera instance.
            composite_args: optional tuple
                (m1, lo1, hi1, log1, m2, lo2, hi2, log2)
                to render in composite mode. None → single-slot resolve.
            quiet: suppress the "Saved screenshot" line (frame recording).
        """
        if self._colormap_tex is None:
            raise RuntimeError("screenshot: colormap not set")

        dev = self.device

        # 1) Accumulate particles into the FBO triple. In composite mode
        # the caller is responsible for having uploaded both slots; we
        # render slot 0 into _accum_textures and slot 1 into
        # _accum_textures2 (matching the live composite path).
        if composite_args is not None:
            self._ensure_fbo(width, height, which=1)
            self._ensure_fbo(width, height, which=2)
            self.set_active_subsample_slot(0)
            self._write_camera_uniforms(camera, width, height)
            self._render_accum(camera, width, height, self._accum_textures)
            self.set_active_subsample_slot(1)
            self._write_camera_uniforms(camera, width, height)
            self._render_accum(camera, width, height, self._accum_textures2)
            self.set_active_subsample_slot(None)
        else:
            self._ensure_fbo(width, height, which=1)
            self._write_camera_uniforms(camera, width, height)
            self._render_accum(camera, width, height, self._accum_textures)

        # 2) Allocate an offscreen target in the same format the resolve
        # / composite pipeline was built against (the swapchain
        # present_format). We'll byte-swap on the CPU side if needed.
        out_tex = dev.create_texture(
            size=(width, height, 1),
            format=self.present_format,
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC),
        )
        out_view = out_tex.create_view()

        # 3) Run the resolve (or composite) pass into out_tex. The two
        # paths build their own bind groups against the existing
        # _resolve_pipeline / _composite_pipeline.
        import struct

        if composite_args is not None:
            m1, lo1, hi1, log1, m2, lo2, hi2, log2 = composite_args
            comp_data = struct.pack("ffIIffII", lo1, hi1, m1, log1, lo2, hi2, m2, log2)
            dev.queue.write_buffer(self._composite_params_buf, 0, comp_data)
            bind_group = dev.create_bind_group(
                layout=self._composite_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._composite_params_buf}},
                    {"binding": 1, "resource": self._accum_textures["views"][0]},
                    {"binding": 2, "resource": self._accum_textures["views"][1]},
                    {"binding": 3, "resource": self._accum_textures["views"][2]},
                    {"binding": 4, "resource": self._accum_textures2["views"][0]},
                    {"binding": 5, "resource": self._accum_textures2["views"][1]},
                    {"binding": 6, "resource": self._accum_textures2["views"][2]},
                    {"binding": 7, "resource": self._colormap_tex_view},
                    {"binding": 8, "resource": self._colormap_sampler},
                ],
            )
            pipeline = self._composite_pipeline
        else:
            resolve_data = struct.pack("ffII", self.qty_min, self.qty_max, self.resolve_mode, self.log_scale)
            dev.queue.write_buffer(self._resolve_params_buf, 0, resolve_data)
            bind_group = dev.create_bind_group(
                layout=self._resolve_bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._resolve_params_buf}},
                    {"binding": 1, "resource": self._accum_textures["views"][0]},
                    {"binding": 2, "resource": self._accum_textures["views"][1]},
                    {"binding": 3, "resource": self._accum_textures["views"][2]},
                    {"binding": 4, "resource": self._colormap_tex_view},
                    {"binding": 5, "resource": self._colormap_sampler},
                ],
            )
            pipeline = self._resolve_pipeline

        encoder = dev.create_command_encoder()
        rp = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": out_view,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": "clear",
                    "store_op": "store",
                }
            ]
        )
        rp.set_pipeline(pipeline)
        rp.set_bind_group(0, bind_group)
        rp.draw(3, 1, 0, 0)
        rp.end()
        # Overlay sinks/stars on top of the resolved gas, mirroring the
        # live render path. _encode_star_overlay reads _last_camera —
        # set it explicitly since the screenshot path doesn't go
        # through the normal render() entry that updates it.
        self._last_camera = camera
        self._encode_trajectories(encoder, out_view)
        self._encode_star_overlay(encoder, out_view)
        dev.queue.submit([encoder.finish()])

        # 4) Read back. read_texture requires bytes_per_row to be a
        # multiple of 256, so we round up and crop the padding.
        bytes_per_row = ((width * 4 + 255) // 256) * 256
        data = dev.queue.read_texture(
            {"texture": out_tex, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": bytes_per_row},
            (width, height, 1),
        )
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, bytes_per_row // 4, 4)
        arr = arr[:, :width, :]
        # Convert BGRA → RGBA if needed.
        if "bgra" in self.present_format:
            arr = arr[:, :, [2, 1, 0, 3]]
        from PIL import Image

        Image.fromarray(arr, mode="RGBA").save(path)
        if not quiet:
            print(f"  Saved screenshot: {path}")

    def _read_accum_texture_r(self, texture, size=None):
        """Read back an accumulation texture's red channel as float32 array.

        Args:
            texture: GPU texture to read.
            size: optional (w, h) tuple. Defaults to `self._accum_size`.
        """
        w, h = size if size is not None else self._accum_size
        fmt = self._accum_format

        if fmt == "r32float":
            bpp = 4
        elif fmt == "rgba16float":
            bpp = 8  # 4 channels * 2 bytes
        else:
            bpp = 4

        data = self.device.queue.read_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": w * bpp},
            (w, h, 1),
        )

        if fmt == "r32float":
            return np.frombuffer(data, dtype=np.float32)
        elif fmt == "rgba16float":
            # Interpret as float16, take every 4th element (R channel)
            all_channels = np.frombuffer(data, dtype=np.float16)
            return all_channels[0::4].astype(np.float32)
        else:
            return np.frombuffer(data, dtype=np.float32)

    def read_accum_range(self, mass_weighted=True):
        """Read back accumulation textures and compute the qty range.

        Args:
            mass_weighted: when True the entropy maximization weights bins
                by accumulated mass (used for surface-density / lightness
                slots so very dense regions dominate the level choice).
                When False the entropy is computed on raw value counts
                (better for color slots where we want the dynamic range
                of the field itself, not where the mass piles up).
        """
        if self._accum_textures is None:
            return self.qty_min, self.qty_max

        w, h = self._accum_size
        if w == 0 or h == 0:
            return self.qty_min, self.qty_max

        den = self._read_accum_texture_r(self._accum_textures["textures"][1])
        mask = den > 1e-30

        if self.resolve_mode == 2:
            num = self._read_accum_texture_r(self._accum_textures["textures"][0])
            sq = self._read_accum_texture_r(self._accum_textures["textures"][2])
            with np.errstate(invalid="ignore"):
                mean = np.where(mask, num / den, 0)
                mean_sq = np.where(mask, sq / den, 0)
            vals = np.sqrt(np.maximum(mean_sq - mean * mean, 0))[mask]
            mass = den[mask]
        elif self.resolve_mode == 1:
            num = self._read_accum_texture_r(self._accum_textures["textures"][0])
            with np.errstate(invalid="ignore"):
                vals = np.where(mask, num / den, 0)[mask]
            mass = den[mask]
        else:
            vals = den[mask]
            mass = vals

        if len(vals) == 0:
            return self.qty_min, self.qty_max

        has_negative = (vals < 0).any()

        if has_negative:
            n = len(vals)
            k_lo = max(n // 100, 0)
            k_hi = min(99 * n // 100, n - 1)
            partitioned = np.partition(vals, (k_lo, k_hi))
            lim_lo = float(partitioned[k_lo])
            lim_hi = float(partitioned[k_hi])
        else:
            from vizmo.field_ops import max_entropy_limits

            entropy_weights = mass if mass_weighted else np.ones_like(vals)
            lim_lo, lim_hi = max_entropy_limits(vals, entropy_weights, log_scale=bool(self.log_scale))

        if self.log_scale and not has_negative:
            if lim_lo <= 0:
                lim_lo = float(vals[vals > 0].min()) if (vals > 0).any() else 1e-10
            lo = float(np.log10(max(lim_lo, 1e-30)))
            hi = float(np.log10(max(lim_hi, 1e-30)))
            if hi - lo < 0.1:
                mid = (hi + lo) / 2
                lo, hi = mid - 1, mid + 1
        else:
            if self.log_scale and has_negative:
                self.log_scale = 0
            lo, hi = lim_lo, lim_hi
            if hi - lo < 1e-30:
                mid = (hi + lo) / 2
                lo, hi = mid - 1, mid + 1

        return lo, hi

    def release(self):
        """Drop GPU resource references; wgpu garbage-collects them."""
        self._subsample_chunks = None
        self._slot_subsample_bgs = [None, None]
        self._star_positions = None
        self._star_masses = None
        self._accum_textures = None
        self._accum_textures2 = None


# ---------------------------------------------------------------------------
# Slice plane (Section 5.A)
# ---------------------------------------------------------------------------

def plane_basis(normal):
    """Orthonormal in-plane basis (e1, e2) for a plane normal.

    Args:
        normal: (3,) array, need not be unit length.

    Returns:
        (e1, e2, n_hat) float64 unit vectors.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-30)
    ref = (np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9
           else np.array([0.0, 1.0, 0.0]))
    e1 = np.cross(ref, n)
    e1 /= max(np.linalg.norm(e1), 1e-30)
    e2 = np.cross(n, e1)
    return e1, e2, n


def slab_cull_to_plane(pos, hsml, values, center, normal, half_size):
    """Select particles whose kernels intersect the slice plane and
    transform them into plane coordinates.

    Returns (pos_h (N,4) float32 [e1, e2, dist_to_plane, hsml],
    vals (N,) float32). Particles outside the grid footprint (padded
    by hsml) are dropped.
    """
    e1, e2, n = plane_basis(normal)
    center = np.asarray(center, dtype=np.float64)
    rel = np.asarray(pos, dtype=np.float64) - center[None, :]
    d_n = rel @ n
    h = np.asarray(hsml, dtype=np.float64)
    keep = np.abs(d_n) < h
    if not keep.any():
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32))
    rel = rel[keep]
    u = rel @ e1
    v = rel @ e2
    hk = h[keep]
    inside = ((np.abs(u) < half_size + hk)
              & (np.abs(v) < half_size + hk))
    u, v = u[inside], v[inside]
    dz = d_n[keep][inside]
    hk = hk[inside]
    vals = np.asarray(values, dtype=np.float64)[keep][inside]
    pos_h = np.stack([u, v, dz, hk], axis=1).astype(np.float32)
    return pos_h, vals.astype(np.float32)


def _kernel_m4_np(u):
    out = np.zeros_like(u)
    m1 = u < 0.5
    m2 = (u >= 0.5) & (u < 1.0)
    out[m1] = 1.0 - 6.0 * u[m1] ** 2 + 6.0 * u[m1] ** 3
    out[m2] = 2.0 * (1.0 - u[m2]) ** 3
    return out * (8.0 / np.pi)


def compute_slice_grid_cpu(pos_h, vals, half_size, res):
    """CPU reference for the slice.wgsl kernel-weighted reconstruction.

    Scatter implementation: each particle deposits W(r)/h^3 weights
    into the pixels inside its kernel footprint, identical math to the
    shader's per-pixel gather. Empty pixels return NaN.

    Args:
        pos_h: (N, 4) plane-frame [u, v, dist_to_plane, hsml].
        vals: (N,) field values.
        half_size: half extent of the grid (world units).
        res: output resolution.

    Returns:
        (res, res) float64 grid, row-major [v, u] to match the shader's
        y*res + x layout.
    """
    num = np.zeros((res, res))
    den = np.zeros((res, res))
    if len(pos_h) == 0:
        return np.full((res, res), np.nan)
    cell = 2.0 * half_size / res
    centers = -half_size + (np.arange(res) + 0.5) * cell
    u, v, dz, h = (pos_h[:, 0].astype(np.float64),
                   pos_h[:, 1].astype(np.float64),
                   pos_h[:, 2].astype(np.float64),
                   pos_h[:, 3].astype(np.float64))
    vals = np.asarray(vals, dtype=np.float64)
    for i in range(len(u)):
        # In-plane kernel footprint radius.
        r_in2 = h[i] ** 2 - dz[i] ** 2
        if r_in2 <= 0:
            continue
        r_in = np.sqrt(r_in2)
        i0 = max(int(np.searchsorted(centers, u[i] - r_in)) - 1, 0)
        i1 = min(int(np.searchsorted(centers, u[i] + r_in)) + 1, res)
        j0 = max(int(np.searchsorted(centers, v[i] - r_in)) - 1, 0)
        j1 = min(int(np.searchsorted(centers, v[i] + r_in)) + 1, res)
        if i0 >= i1 or j0 >= j1:
            continue
        du = centers[i0:i1] - u[i]
        dv = centers[j0:j1] - v[i]
        r = np.sqrt(du[None, :] ** 2 + dv[:, None] ** 2 + dz[i] ** 2)
        w = _kernel_m4_np(r / h[i]) / h[i] ** 3
        num[j0:j1, i0:i1] += vals[i] * w
        den[j0:j1, i0:i1] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(den > 0, num / den, np.nan)
    return grid


def slice_grid_to_fits(grid, center_kpc, size_kpc, path, field="Masses",
                       unit="", normal_label="z"):
    """Write a slice grid as FITS with a linear-kpc WCS.

    CRPIX at the image center, CDELT = size_kpc / res — the same
    conventions as export_fits_map, so CARTA/DS9 read it directly.
    """
    from astropy.io import fits as pyfits

    grid = np.asarray(grid, dtype=np.float32)
    res = grid.shape[0]
    hdu = pyfits.PrimaryHDU(grid)
    hd = hdu.header
    hd["BUNIT"] = unit or "code units"
    hd["FIELD"] = field
    hd["CTYPE1"] = "LINEAR"
    hd["CTYPE2"] = "LINEAR"
    hd["CUNIT1"] = "kpc"
    hd["CUNIT2"] = "kpc"
    hd["CRPIX1"] = res / 2 + 0.5
    hd["CRPIX2"] = res / 2 + 0.5
    hd["CRVAL1"] = 0.0
    hd["CRVAL2"] = 0.0
    hd["CDELT1"] = size_kpc / res
    hd["CDELT2"] = size_kpc / res
    hd["SLICENRM"] = normal_label
    hd["CENX"] = float(center_kpc[0])
    hd["CENY"] = float(center_kpc[1])
    hd["CENZ"] = float(center_kpc[2])
    hd["ORIGIN"] = "vizmo slice plane (kernel-weighted reconstruction)"
    hdu.writeto(path, overwrite=True)
    return path


_SLICE_QUAD_WGSL = """
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) uv: vec2<f32>) -> VSOut {
    var out: VSOut;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@group(0) @binding(0) var t_slice: texture_2d<f32>;
@group(0) @binding(1) var s_slice: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(t_slice, s_slice, uv);
}
"""

# Cap on slab-culled particles sent to the GPU gather (keeps the
# res^2 * N inner loop bounded; ~65k * 512^2 = 1.7e10 MACs, tens of
# ms on an Apple-class GPU for a one-shot recompute).
SLICE_GPU_MAX_PARTICLES = 65536


class SlicePlaneRenderer:
    """Slice-plane compute + textured-quad blit.

    compute() tries the slice.wgsl GPU gather first and falls back to
    the identical-math CPU scatter (compute_slice_grid_cpu) on any
    failure — documented fallback per Section 5.A. The result grid is
    colormapped on the CPU and drawn as a textured quad whose corners
    the app projects into NDC each frame.
    """

    def __init__(self, device, present_format):
        self.device = device
        self.grid = None           # last computed (res, res) grid
        self.used_gpu = False
        self._tex = None
        self._tex_size = (0, 0)
        self._sampler = device.create_sampler(mag_filter="linear",
                                              min_filter="linear")
        shader = device.create_shader_module(code=_SLICE_QUAD_WGSL)
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
                        {"format": "float32x2", "offset": 0,
                         "shader_location": 0},
                        {"format": "float32x2", "offset": 8,
                         "shader_location": 1},
                    ],
                }],
            },
            primitive={"topology": "triangle-list"},
            fragment={
                "module": shader, "entry_point": "fs_main",
                "targets": [{"format": present_format, "blend": {
                    "color": {"src_factor": "src-alpha",
                              "dst_factor": "one-minus-src-alpha"},
                    "alpha": {"src_factor": "one",
                              "dst_factor": "one-minus-src-alpha"},
                }}],
            },
        )
        self._bind_group = None
        self._vbo = None
        # Lazy compute pipeline (built on first GPU compute attempt).
        self._compute_pipeline = None
        self._compute_bgl = None

    # -- compute ------------------------------------------------------------

    def _ensure_compute_pipeline(self):
        if self._compute_pipeline is not None:
            return
        dev = self.device
        shader = dev.create_shader_module(code=_load_wgsl("slice.wgsl"))
        self._compute_bgl = dev.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": "storage"}},
        ])
        layout = dev.create_pipeline_layout(
            bind_group_layouts=[self._compute_bgl])
        self._compute_pipeline = dev.create_compute_pipeline(
            layout=layout,
            compute={"module": shader, "entry_point": "cs_main"})

    def _compute_gpu(self, pos_h, vals, half_size, res):
        dev = self.device
        self._ensure_compute_pipeline()
        n = len(pos_h)
        params = np.zeros(4, dtype=np.float32)
        params[0] = half_size
        params.view(np.uint32)[1] = res
        params.view(np.uint32)[2] = n
        pbuf = dev.create_buffer_with_data(
            data=params.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM)
        posbuf = dev.create_buffer_with_data(
            data=np.ascontiguousarray(pos_h, dtype=np.float32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE)
        valbuf = dev.create_buffer_with_data(
            data=np.ascontiguousarray(vals, dtype=np.float32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE)
        outbuf = dev.create_buffer(
            size=res * res * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        bg = dev.create_bind_group(
            layout=self._compute_bgl,
            entries=[
                {"binding": 0, "resource": {"buffer": pbuf}},
                {"binding": 1, "resource": {"buffer": posbuf}},
                {"binding": 2, "resource": {"buffer": valbuf}},
                {"binding": 3, "resource": {"buffer": outbuf}},
            ])
        enc = dev.create_command_encoder()
        cpass = enc.begin_compute_pass()
        cpass.set_pipeline(self._compute_pipeline)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((res + 7) // 8, (res + 7) // 8)
        cpass.end()
        dev.queue.submit([enc.finish()])
        raw = dev.queue.read_buffer(outbuf)
        grid = np.frombuffer(raw, dtype=np.float32).reshape(res, res).copy()
        grid[grid <= -1.0e29] = np.nan  # empty-pixel sentinel
        return grid.astype(np.float64)

    def compute(self, pos, hsml, values, center, normal, half_size,
                res=512):
        """Recompute the slice grid. GPU first, CPU fallback."""
        pos_h, vals = slab_cull_to_plane(pos, hsml, values, center,
                                         normal, half_size)
        if len(pos_h) > SLICE_GPU_MAX_PARTICLES:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(pos_h), size=SLICE_GPU_MAX_PARTICLES,
                             replace=False)
            pos_h, vals = pos_h[sel], vals[sel]
        try:
            self.grid = self._compute_gpu(pos_h, vals, half_size, res)
            self.used_gpu = True
        except Exception as e:
            # Documented CPU fallback: same kernel math, numpy scatter.
            print(f"  slice: GPU compute unavailable ({e}); CPU fallback")
            self.grid = compute_slice_grid_cpu(pos_h, vals, half_size, res)
            self.used_gpu = False
        return self.grid

    # -- draw ---------------------------------------------------------------

    def upload_colormapped(self, colormap_rgba, vmin, vmax, log_scale,
                           opacity=0.85):
        """Map the grid through a (256, 4) colormap LUT and upload."""
        if self.grid is None:
            return
        g = self.grid
        vals = np.where(np.isfinite(g), g, np.nan)
        if log_scale:
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = np.log10(np.where(vals > 0, vals, np.nan))
        t = (vals - vmin) / max(vmax - vmin, 1e-30)
        idx = np.clip((t * 255), 0, 255)
        nanmask = ~np.isfinite(idx)
        idx = np.where(nanmask, 0, idx).astype(np.uint8)
        rgba = colormap_rgba[idx]
        rgba[..., 3] = np.where(nanmask, 0,
                                int(np.clip(opacity, 0, 1) * 255))
        rgba = np.ascontiguousarray(rgba)
        res = rgba.shape[0]
        dev = self.device
        if self._tex_size != (res, res):
            self._tex = dev.create_texture(
                size=(res, res, 1), format="rgba8unorm",
                usage=(wgpu.TextureUsage.TEXTURE_BINDING
                       | wgpu.TextureUsage.COPY_DST))
            self._tex_size = (res, res)
            self._bind_group = dev.create_bind_group(
                layout=self._bgl,
                entries=[
                    {"binding": 0, "resource": self._tex.create_view()},
                    {"binding": 1, "resource": self._sampler},
                ])
        dev.queue.write_texture(
            {"texture": self._tex, "mip_level": 0, "origin": (0, 0, 0)},
            rgba.tobytes(), {"bytes_per_row": res * 4,
                             "rows_per_image": res},
            (res, res, 1))

    def render_to_pass(self, rpass, ndc_corners):
        """Draw the slice quad. ndc_corners: 4 (x, y) NDC points in
        the order (-u-v, +u-v, +u+v, -u+v) of plane coordinates."""
        if self._bind_group is None:
            return
        c = ndc_corners
        verts = np.array([
            c[0][0], c[0][1], 0, 0,  c[1][0], c[1][1], 1, 0,
            c[3][0], c[3][1], 0, 1,
            c[1][0], c[1][1], 1, 0,  c[2][0], c[2][1], 1, 1,
            c[3][0], c[3][1], 0, 1,
        ], dtype=np.float32)
        dev = self.device
        vb = verts.tobytes()
        if self._vbo is None or self._vbo.size < len(vb):
            self._vbo = dev.create_buffer_with_data(
                data=vb, usage=(wgpu.BufferUsage.VERTEX
                                | wgpu.BufferUsage.COPY_DST))
        else:
            dev.queue.write_buffer(self._vbo, 0, vb)
        rpass.set_pipeline(self._pipeline)
        rpass.set_bind_group(0, self._bind_group)
        rpass.set_vertex_buffer(0, self._vbo)
        rpass.draw(6)


def slice_plane_mode():
    """RenderMode marker for the slice plane (Section 5.A)."""
    return RenderMode(name="SlicePlane", weight_field="",
                      qty_field="", resolve_mode=-2)


RenderMode.slice_plane = staticmethod(slice_plane_mode)


# ---------------------------------------------------------------------------
# Isosurface extraction (Section 5.B) + split-screen state (Section 5.F)
# ---------------------------------------------------------------------------

def voxelize_particles(pos, mass, center, half_size, n_grid):
    """CIC-voxelize particle masses onto an n_grid^3 mesh (code units).

    Thin wrapper over power_spectrum.cic_deposit so the isosurface and
    P(k) paths share one deposition kernel. Returns the 3D grid; total
    deposited mass equals the in-box particle mass (CIC conserves).
    """
    from .power_spectrum import cic_deposit

    return cic_deposit(pos, mass, center, half_size, n_grid)


def extract_isosurface(grid, level, center, half_size):
    """Marching cubes on a voxel grid -> world-space mesh.

    Args:
        grid: (n, n, n) voxelized field.
        level: iso threshold in grid units.
        center: (3,) world center of the cube (code units).
        half_size: half extent of the cube.

    Returns:
        (verts (V, 3) world code units, faces (F, 3) int, normals (V, 3)).
    """
    from skimage.measure import marching_cubes

    n = grid.shape[0]
    cell = 2.0 * half_size / n
    verts, faces, normals, _ = marching_cubes(grid, level=level)
    # Voxel index -> world: cell centers at (i + 0.5) * cell - half.
    world = (verts + 0.5) * cell - half_size + np.asarray(center)[None, :]
    return world, faces.astype(np.uint32), normals.astype(np.float32)


def mesh_to_obj(path, verts, faces):
    """Write a mesh as Wavefront OBJ (1-based face indices)."""
    with open(path, "w") as f:
        f.write("# vizmo isosurface export\n")
        for v in verts:
            f.write(f"v {v[0]:.6g} {v[1]:.6g} {v[2]:.6g}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    return path


ISO_COLORS = [(0.35, 0.65, 1.0), (1.0, 0.62, 0.25),
              (0.45, 0.9, 0.5), (0.95, 0.4, 0.75)]


class IsosurfaceRenderer:
    """Up to 4 simultaneous Phong-shaded isosurface meshes.

    Voxelization + marching cubes run on the CPU (the caller may do so
    in a background thread); this class owns the GPU vertex buffers and
    the render pipeline. Vertices are uploaded relative to a reference
    point and the full-precision translation is folded into the MVP on
    the CPU in float64, so float32 vertex coordinates stay accurate at
    cosmological box offsets.
    """

    MAX_SURFACES = 4

    def __init__(self, device, present_format):
        self.device = device
        self.surfaces = []  # dicts: vbo, n_verts, color, opacity, level
        shader = device.create_shader_module(
            code=_load_wgsl("isosurface.wgsl"))
        self._bgl = device.create_bind_group_layout(entries=[
            {"binding": 0,
             "visibility": (wgpu.ShaderStage.VERTEX
                            | wgpu.ShaderStage.FRAGMENT),
             "buffer": {"type": "uniform"}},
        ])
        layout = device.create_pipeline_layout(
            bind_group_layouts=[self._bgl])
        self._pipeline = device.create_render_pipeline(
            layout=layout,
            vertex={
                "module": shader, "entry_point": "vs_main",
                "buffers": [{
                    "array_stride": 24, "step_mode": "vertex",
                    "attributes": [
                        {"format": "float32x3", "offset": 0,
                         "shader_location": 0},
                        {"format": "float32x3", "offset": 12,
                         "shader_location": 1},
                    ],
                }],
            },
            primitive={"topology": "triangle-list"},
            fragment={
                "module": shader, "entry_point": "fs_main",
                "targets": [{"format": present_format, "blend": {
                    "color": {"src_factor": "src-alpha",
                              "dst_factor": "one-minus-src-alpha"},
                    "alpha": {"src_factor": "one",
                              "dst_factor": "one-minus-src-alpha"},
                }}],
            },
        )

    def set_surface(self, idx, verts, faces, normals, color, opacity,
                    level, ref=None):
        """Upload a mesh as surface `idx` (expanded triangle soup)."""
        if ref is None:
            ref = verts.mean(axis=0)
        tri_v = (verts[faces.reshape(-1)] - ref[None, :]).astype(np.float32)
        tri_n = normals[faces.reshape(-1)].astype(np.float32)
        inter = np.empty((len(tri_v), 6), dtype=np.float32)
        inter[:, :3] = tri_v
        inter[:, 3:] = tri_n
        dev = self.device
        vbo = dev.create_buffer_with_data(
            data=inter.tobytes(), usage=wgpu.BufferUsage.VERTEX)
        ubuf = dev.create_buffer(
            size=96, usage=(wgpu.BufferUsage.UNIFORM
                            | wgpu.BufferUsage.COPY_DST))
        bg = dev.create_bind_group(
            layout=self._bgl,
            entries=[{"binding": 0, "resource": {"buffer": ubuf}}])
        entry = {"vbo": vbo, "n": len(tri_v), "ubuf": ubuf, "bg": bg,
                 "color": color, "opacity": opacity, "level": level,
                 "ref": np.asarray(ref, dtype=np.float64),
                 "verts": verts, "faces": faces}
        while len(self.surfaces) <= idx:
            self.surfaces.append(None)
        self.surfaces[idx] = entry

    def clear(self):
        self.surfaces = []

    def write_uniforms(self, camera):
        """Per-surface MVP (float64 fold of the ref translation)."""
        view = camera.view_matrix().astype(np.float64)
        proj = camera.projection_matrix().astype(np.float64)
        for s in self.surfaces:
            if s is None:
                continue
            t = np.eye(4)
            t[:3, 3] = s["ref"]
            mvp = (proj @ view @ t).astype(np.float32)
            buf = np.zeros(24, dtype=np.float32)
            buf[:16] = mvp.T.reshape(-1)  # wgsl column-major
            buf[16:19] = s["color"]
            buf[19] = s["opacity"]
            buf[20:23] = (-camera.forward).astype(np.float32)
            self.device.queue.write_buffer(s["ubuf"], 0, buf.tobytes())

    def render_to_pass(self, rpass):
        for s in self.surfaces:
            if s is None or s["n"] == 0:
                continue
            rpass.set_pipeline(self._pipeline)
            rpass.set_bind_group(0, s["bg"])
            rpass.set_vertex_buffer(0, s["vbo"])
            rpass.draw(s["n"])


@dataclass
class RenderState:
    """Per-viewport display state for split-screen mode (Section 5.F).

    Each half of a split view owns one of these; mutating one side
    never touches the other (verified in tests/test_split.py).
    """

    field: str = "Masses"
    colormap: str = "magma"
    render_mode: str = "SurfaceDensity"
    qty_min: float = -1.0
    qty_max: float = 3.0
    log_scale: int = 1


SPLIT_MODES = [None, "lr", "tb"]


def _append_split_resolve(renderer, encoder, screen_view, width, height,
                          left_state, right_state, orientation="lr"):
    """Two viewport-restricted resolve passes for split-screen mode.

    Left/top half resolves accumulation set 1 with the main colormap;
    right/bottom half resolves set 2 (the composite slot-1
    accumulation) with the renderer's split colormap. Params for each
    side come from its RenderState. The caller must have filled both
    accum sets (the composite accumulation path does exactly that).
    """
    import struct

    dev = renderer.device
    if getattr(renderer, "_split_params_bufs", None) is None:
        renderer._split_params_bufs = [
            dev.create_buffer(size=16,
                              usage=(wgpu.BufferUsage.UNIFORM
                                     | wgpu.BufferUsage.COPY_DST))
            for _ in range(2)]
        renderer._split_bgs = [None, None]

    states = [left_state, right_state]
    accums = [renderer._accum_textures, renderer._accum_textures2]
    cmap_views = [renderer._colormap_tex_view,
                  getattr(renderer, "_split_cmap_view", None)
                  or renderer._colormap_tex_view]
    for i in (0, 1):
        st = states[i]
        resolve_mode = {"SurfaceDensity": 0, "WeightedAverage": 1,
                        "WeightedVariance": 2}.get(st.render_mode, 0)
        dev.queue.write_buffer(
            renderer._split_params_bufs[i], 0,
            struct.pack("ffII", st.qty_min, st.qty_max, resolve_mode,
                        st.log_scale))
        if renderer._split_bgs[i] is None and accums[i] is not None:
            renderer._split_bgs[i] = dev.create_bind_group(
                layout=renderer._resolve_bgl,
                entries=[
                    {"binding": 0,
                     "resource": {"buffer": renderer._split_params_bufs[i]}},
                    {"binding": 1, "resource": accums[i]["views"][0]},
                    {"binding": 2, "resource": accums[i]["views"][1]},
                    {"binding": 3, "resource": accums[i]["views"][2]},
                    {"binding": 4, "resource": cmap_views[i]},
                    {"binding": 5,
                     "resource": renderer._colormap_sampler},
                ])

    rpass = encoder.begin_render_pass(color_attachments=[{
        "view": screen_view, "clear_value": (0, 0, 0, 1),
        "load_op": "clear", "store_op": "store"}])
    for i in (0, 1):
        if renderer._split_bgs[i] is None:
            continue
        if orientation == "lr":
            x, y = (0 if i == 0 else width // 2), 0
            w, h = width // 2, height
        else:
            x, y = 0, (0 if i == 0 else height // 2)
            w, h = width, height // 2
        rpass.set_viewport(float(x), float(y), float(w), float(h),
                           0.0, 1.0)
        rpass.set_scissor_rect(x, y, w, h)
        rpass.set_pipeline(renderer._resolve_pipeline)
        rpass.set_bind_group(0, renderer._split_bgs[i])
        rpass.draw(3, 1, 0, 0)
    rpass.end()


def set_split_colormap(renderer, rgba_data):
    """Upload the right/bottom pane's colormap as a separate texture."""
    dev = renderer.device
    tex = dev.create_texture(
        size=(rgba_data.shape[0], 1, 1), format="rgba8unorm",
        usage=(wgpu.TextureUsage.TEXTURE_BINDING
               | wgpu.TextureUsage.COPY_DST))
    dev.queue.write_texture(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        np.ascontiguousarray(rgba_data).tobytes(),
        {"bytes_per_row": rgba_data.shape[0] * 4, "rows_per_image": 1},
        (rgba_data.shape[0], 1, 1))
    renderer._split_cmap_view = tex.create_view()
    renderer._split_bgs = [None, None]  # rebind with the new texture


WGPURenderer.append_split_resolve = _append_split_resolve
WGPURenderer.set_split_colormap = set_split_colormap


# ---------------------------------------------------------------------------
# Streamlines (5.C), field-line arrows (5.D), volume rendering (5.E)
# ---------------------------------------------------------------------------

def integrate_streamlines(pos, vec, hsml, rho, seeds, step_size,
                          max_steps=200, k_neighbors=32, bounds=None,
                          v_floor=1.0, n_workers=8):
    """RK4 streamline integration through an SPH-sampled vector field.

    The local vector at a point is the kernel-weighted average of the
    k nearest particles' vectors:
        v(x) = sum_j m_j v_j W(|x - x_j|/h_j) / rho_j / sum_j (...)
    simplified here to inverse-kernel weighting without the mass/rho
    factor cancelling in the normalized average (identical for the
    direction field; magnitudes follow the local particle values).

    Args:
        pos: (N, 3) particle positions (code units).
        vec: (N, 3) particle vector field (velocity or B).
        hsml: (N,) smoothing lengths.
        rho: (N,) densities (unused weighting hook, may be None).
        seeds: (S, 3) seed points.
        step_size: integration step (code units).
        max_steps: maximum steps per line.
        k_neighbors: neighbors per interpolation.
        bounds: optional (center, radius) sphere; integration stops on
            exit (the aperture clamp — tested).
        v_floor: stop when |v| falls below this (field units).
        n_workers: thread pool size.

    Returns:
        list of dicts: {"points": (n, 3), "values": (n,) |v|}.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scipy.spatial import cKDTree

    pos = np.asarray(pos, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    h = np.asarray(hsml, dtype=np.float64)
    tree = cKDTree(pos)
    if bounds is not None:
        b_center = np.asarray(bounds[0], dtype=np.float64)
        b_r = float(bounds[1])

    def sample(x):
        d, idx = tree.query(x, k=k_neighbors)
        d = np.atleast_1d(d)
        idx = np.atleast_1d(idx)
        w = _kernel_m4_np(np.minimum(d / np.maximum(h[idx], 1e-30), 0.999))
        w += 1e-30
        return (vec[idx] * w[:, None]).sum(axis=0) / w.sum()

    def trace(seed):
        x = np.asarray(seed, dtype=np.float64).copy()
        pts = [x.copy()]
        vals = []
        v0 = sample(x)
        vals.append(np.linalg.norm(v0))
        for _ in range(max_steps):
            if bounds is not None and np.linalg.norm(x - b_center) > b_r:
                break
            v1 = sample(x)
            sp = np.linalg.norm(v1)
            if sp < v_floor:
                break
            k1 = v1 / sp
            k2v = sample(x + 0.5 * step_size * k1)
            k2 = k2v / max(np.linalg.norm(k2v), 1e-30)
            k3v = sample(x + 0.5 * step_size * k2)
            k3 = k3v / max(np.linalg.norm(k3v), 1e-30)
            k4v = sample(x + step_size * k3)
            k4 = k4v / max(np.linalg.norm(k4v), 1e-30)
            x = x + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            pts.append(x.copy())
            vals.append(sp)
        return {"points": np.array(pts),
                "values": np.array(vals[:len(pts)])}

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        return list(ex.map(trace, seeds))


def fibonacci_sphere(n, center, radius):
    """n seed points on a sphere surface (Fibonacci lattice)."""
    i = np.arange(n, dtype=np.float64)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    d = np.stack([np.sin(phi) * np.cos(theta),
                  np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)
    return np.asarray(center)[None, :] + radius * d


class StreamlineRenderer:
    """GPU polylines for streamlines: one line_strip draw per line,
    per-vertex RGBA from the color-by field through the colormap."""

    def __init__(self, device, present_format):
        self.device = device
        self.lines = []  # dicts: vbo, n, ref
        shader = device.create_shader_module(
            code=_load_wgsl("streamlines.wgsl"))
        self._bgl = device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX,
             "buffer": {"type": "uniform"}}])
        layout = device.create_pipeline_layout(
            bind_group_layouts=[self._bgl])
        self._pipeline = device.create_render_pipeline(
            layout=layout,
            vertex={"module": shader, "entry_point": "vs_main",
                    "buffers": [{
                        "array_stride": 28, "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0,
                             "shader_location": 0},
                            {"format": "float32x4", "offset": 12,
                             "shader_location": 1}]}]},
            primitive={"topology": "line-strip"},
            fragment={"module": shader, "entry_point": "fs_main",
                      "targets": [{"format": present_format, "blend": {
                          "color": {"src_factor": "src-alpha",
                                    "dst_factor": "one-minus-src-alpha"},
                          "alpha": {"src_factor": "one",
                                    "dst_factor": "one-minus-src-alpha"}}}]},
        )
        self._ubuf = device.create_buffer(
            size=64, usage=(wgpu.BufferUsage.UNIFORM
                            | wgpu.BufferUsage.COPY_DST))
        self._bg = device.create_bind_group(
            layout=self._bgl,
            entries=[{"binding": 0, "resource": {"buffer": self._ubuf}}])
        self._ref = np.zeros(3)

    def set_lines(self, traces, colormap_rgba, vmin, vmax,
                  log_scale=True):
        self.lines = []
        if not traces:
            return
        self._ref = np.mean([t["points"][0] for t in traces], axis=0)
        for t in traces:
            pts = t["points"]
            if len(pts) < 2:
                continue
            vals = t["values"].astype(np.float64)
            if log_scale:
                with np.errstate(invalid="ignore", divide="ignore"):
                    vals = np.log10(np.where(vals > 0, vals, np.nan))
            x = (vals - vmin) / max(vmax - vmin, 1e-30)
            idx = np.clip(np.nan_to_num(x) * 255, 0, 255).astype(np.uint8)
            rgba = colormap_rgba[idx].astype(np.float32) / 255.0
            rgba[:, 3] = 0.9
            inter = np.empty((len(pts), 7), dtype=np.float32)
            inter[:, :3] = pts - self._ref[None, :]
            inter[:, 3:] = rgba[:len(pts)]
            vbo = self.device.create_buffer_with_data(
                data=inter.tobytes(), usage=wgpu.BufferUsage.VERTEX)
            self.lines.append({"vbo": vbo, "n": len(pts)})

    def write_uniforms(self, camera):
        view = camera.view_matrix().astype(np.float64)
        proj = camera.projection_matrix().astype(np.float64)
        t = np.eye(4)
        t[:3, 3] = self._ref
        mvp = (proj @ view @ t).astype(np.float32)
        self.device.queue.write_buffer(
            self._ubuf, 0, mvp.T.copy().tobytes())

    def render_to_pass(self, rpass):
        for ln in self.lines:
            rpass.set_pipeline(self._pipeline)
            rpass.set_bind_group(0, self._bg)
            rpass.set_vertex_buffer(0, ln["vbo"])
            rpass.draw(ln["n"])


class ArrowRenderer:
    """Instanced procedural arrow glyphs (max 5000 instances)."""

    MAX_ARROWS = 5000
    VERTS_PER = 30

    def __init__(self, device, present_format):
        self.device = device
        self.n_instances = 0
        shader = device.create_shader_module(code=_load_wgsl("arrows.wgsl"))
        self._bgl = device.create_bind_group_layout(entries=[
            {"binding": 0,
             "visibility": (wgpu.ShaderStage.VERTEX
                            | wgpu.ShaderStage.FRAGMENT),
             "buffer": {"type": "uniform"}}])
        layout = device.create_pipeline_layout(
            bind_group_layouts=[self._bgl])
        self._pipeline = device.create_render_pipeline(
            layout=layout,
            vertex={"module": shader, "entry_point": "vs_main",
                    "buffers": [{
                        "array_stride": 40, "step_mode": "instance",
                        "attributes": [
                            {"format": "float32x3", "offset": 0,
                             "shader_location": 0},
                            {"format": "float32x3", "offset": 12,
                             "shader_location": 1},
                            {"format": "float32x4", "offset": 24,
                             "shader_location": 2}]}]},
            primitive={"topology": "triangle-list"},
            fragment={"module": shader, "entry_point": "fs_main",
                      "targets": [{"format": present_format, "blend": {
                          "color": {"src_factor": "src-alpha",
                                    "dst_factor": "one-minus-src-alpha"},
                          "alpha": {"src_factor": "one",
                                    "dst_factor": "one-minus-src-alpha"}}}]},
        )
        self._ubuf = device.create_buffer(
            size=80, usage=(wgpu.BufferUsage.UNIFORM
                            | wgpu.BufferUsage.COPY_DST))
        self._bg = device.create_bind_group(
            layout=self._bgl,
            entries=[{"binding": 0, "resource": {"buffer": self._ubuf}}])
        self._vbo = None
        self._ref = np.zeros(3)
        self._radius = 1.0

    def set_arrows(self, positions, vectors, colors, arrow_length):
        """positions (N,3) code units; vectors (N,3) normalized scale
        applied by caller; colors (N,4) float 0-1."""
        n = min(len(positions), self.MAX_ARROWS)
        if n == 0:
            self.n_instances = 0
            return
        sel = (np.random.default_rng(0).choice(len(positions), n,
                                               replace=False)
               if len(positions) > n else np.arange(n))
        self._ref = positions[sel].mean(axis=0)
        inter = np.empty((n, 10), dtype=np.float32)
        inter[:, :3] = positions[sel] - self._ref[None, :]
        inter[:, 3:6] = vectors[sel]
        inter[:, 6:] = colors[sel]
        self._vbo = self.device.create_buffer_with_data(
            data=inter.tobytes(), usage=wgpu.BufferUsage.VERTEX)
        self.n_instances = n
        self._radius = arrow_length * 0.06

    def write_uniforms(self, camera):
        if self.n_instances == 0:
            return
        view = camera.view_matrix().astype(np.float64)
        proj = camera.projection_matrix().astype(np.float64)
        t = np.eye(4)
        t[:3, 3] = self._ref
        mvp = (proj @ view @ t).astype(np.float32)
        buf = np.zeros(20, dtype=np.float32)
        buf[:16] = mvp.T.reshape(-1)
        buf[16] = self._radius
        self.device.queue.write_buffer(self._ubuf, 0, buf.tobytes())

    def render_to_pass(self, rpass):
        if self.n_instances == 0 or self._vbo is None:
            return
        rpass.set_pipeline(self._pipeline)
        rpass.set_bind_group(0, self._bg)
        rpass.set_vertex_buffer(0, self._vbo)
        rpass.draw(self.VERTS_PER, self.n_instances)


VOXELIZE_FIXED_SCALE = 1.0e6


class VolumeRenderer:
    """Emission-absorption / MIP ray marcher over a voxelized field.

    Voxelization: GPU compute (voxelize.wgsl, fixed-point u32 atomics)
    with a CPU CIC fallback. The 3D texture and the 1D transfer
    function feed volume.wgsl's fullscreen ray-march pass. Resolution
    auto-halves while the camera moves (LOD hook driven by the app).
    """

    def __init__(self, device, present_format):
        self.device = device
        self.enabled = False
        self.mode = 0          # 0 = emission-absorption, 1 = MIP
        self.step_mult = 1.0
        self.max_steps = 512
        self.res = 128
        self.vmin = 0.0
        self.vmax = 1.0
        self.log_scale = True
        self._tex3d = None
        self._tex3d_res = 0
        self._tf_tex = None
        self.tf_points = [(0.0, 0.0), (0.6, 0.05), (1.0, 0.8)]
        self._grid = None
        self.center = None
        self.half_size = None
        self.used_gpu_voxelize = False

        shader = device.create_shader_module(code=_load_wgsl("volume.wgsl"))
        self._bgl = device.create_bind_group_layout(entries=[
            {"binding": 0,
             "visibility": (wgpu.ShaderStage.VERTEX
                            | wgpu.ShaderStage.FRAGMENT),
             "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "float",
                         "view_dimension": "3d"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": "filtering"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "float",
                         "view_dimension": "1d"}},
            {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": "filtering"}},
        ])
        layout = device.create_pipeline_layout(
            bind_group_layouts=[self._bgl])
        self._pipeline = device.create_render_pipeline(
            layout=layout,
            vertex={"module": shader, "entry_point": "vs_main",
                    "buffers": []},
            primitive={"topology": "triangle-list"},
            fragment={"module": shader, "entry_point": "fs_main",
                      "targets": [{"format": present_format, "blend": {
                          "color": {"src_factor": "src-alpha",
                                    "dst_factor": "one-minus-src-alpha"},
                          "alpha": {"src_factor": "one",
                                    "dst_factor": "one-minus-src-alpha"}}}]},
        )
        self._ubuf = device.create_buffer(
            size=112, usage=(wgpu.BufferUsage.UNIFORM
                             | wgpu.BufferUsage.COPY_DST))
        self._sampler = device.create_sampler(
            mag_filter="linear", min_filter="linear",
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            address_mode_w="clamp-to-edge")
        self._bg = None
        self._vox_pipeline = None
        self._vox_bgl = None

    # -- voxelization ---------------------------------------------------

    def _voxelize_gpu(self, pos, mass, center, half_size, n_grid):
        dev = self.device
        if self._vox_pipeline is None:
            shader = dev.create_shader_module(
                code=_load_wgsl("voxelize.wgsl"))
            self._vox_bgl = dev.create_bind_group_layout(entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": "uniform"}},
                {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": "read-only-storage"}},
                {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": "read-only-storage"}},
                {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": "storage"}},
            ])
            layout = dev.create_pipeline_layout(
                bind_group_layouts=[self._vox_bgl])
            self._vox_pipeline = dev.create_compute_pipeline(
                layout=layout,
                compute={"module": shader, "entry_point": "cs_main"})
        n = len(pos)
        pos4 = np.zeros((n, 4), dtype=np.float32)
        pos4[:, :3] = pos
        params = np.zeros(8, dtype=np.float32)
        params[0:3] = center
        params[3] = half_size
        params.view(np.uint32)[4] = n_grid
        params.view(np.uint32)[5] = n
        # Fixed-point scale chosen so the total deposited mass stays
        # far below u32 overflow per voxel.
        scale = VOXELIZE_FIXED_SCALE / max(float(mass.max()), 1e-30)
        params[6] = scale
        pbuf = dev.create_buffer_with_data(
            data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        posb = dev.create_buffer_with_data(
            data=pos4.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        mb = dev.create_buffer_with_data(
            data=np.ascontiguousarray(mass, dtype=np.float32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE)
        outb = dev.create_buffer(
            size=n_grid**3 * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        bg = dev.create_bind_group(
            layout=self._vox_bgl,
            entries=[{"binding": i, "resource": {"buffer": b}}
                     for i, b in enumerate([pbuf, posb, mb, outb])])
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._vox_pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((n + 255) // 256)
        cp.end()
        dev.queue.submit([enc.finish()])
        raw = dev.queue.read_buffer(outb)
        fixed = np.frombuffer(raw, dtype=np.uint32).astype(np.float64)
        return (fixed / scale).reshape(n_grid, n_grid, n_grid)

    def voxelize(self, pos, mass, vals, center, half_size, n_grid=None):
        """Voxelize a (mass-weighted) field; GPU first, CPU fallback."""
        if n_grid is None:
            n_grid = self.res
        try:
            grid_m = self._voxelize_gpu(pos, mass, center, half_size,
                                        n_grid)
            self.used_gpu_voxelize = True
        except Exception as e:
            print(f"  volume: GPU voxelize unavailable ({e}); CPU CIC")
            grid_m = voxelize_particles(pos, mass, center, half_size,
                                        n_grid)
            self.used_gpu_voxelize = False
        if vals is None:
            grid = grid_m
        else:
            try:
                grid_f = (self._voxelize_gpu(pos, mass * vals, center,
                                             half_size, n_grid)
                          if self.used_gpu_voxelize else
                          voxelize_particles(pos, mass * vals, center,
                                             half_size, n_grid))
            except Exception:
                grid_f = voxelize_particles(pos, mass * vals, center,
                                            half_size, n_grid)
            with np.errstate(invalid="ignore", divide="ignore"):
                grid = np.where(grid_m > 0, grid_f / grid_m, 0.0)
        self.set_grid(grid, center, half_size)
        return grid

    def set_grid(self, grid, center, half_size):
        """Upload a voxel grid as the 3D texture (+ display range)."""
        self._grid = grid
        self.center = np.asarray(center, dtype=np.float64)
        self.half_size = float(half_size)
        disp = grid.astype(np.float64)
        if self.log_scale:
            with np.errstate(invalid="ignore", divide="ignore"):
                disp = np.log10(np.where(disp > 0, disp, np.nan))
        fin = disp[np.isfinite(disp)]
        if fin.size:
            self.vmin = float(np.percentile(fin, 5))
            self.vmax = float(np.percentile(fin, 99.8))
        tex_data = np.nan_to_num(
            disp, nan=(self.vmin - 10.0)).astype(np.float32)
        n = grid.shape[0]
        dev = self.device
        if self._tex3d_res != n:
            self._tex3d = dev.create_texture(
                size=(n, n, n), format="r32float", dimension="3d",
                usage=(wgpu.TextureUsage.TEXTURE_BINDING
                       | wgpu.TextureUsage.COPY_DST))
            self._tex3d_res = n
            self._bg = None
        # wgpu 3D write: z-major layout, rows_per_image = height.
        dev.queue.write_texture(
            {"texture": self._tex3d, "mip_level": 0,
             "origin": (0, 0, 0)},
            np.ascontiguousarray(
                np.transpose(tex_data, (2, 1, 0))).tobytes(),
            {"bytes_per_row": n * 4, "rows_per_image": n},
            (n, n, n))
        self.upload_transfer_function()

    def upload_transfer_function(self, colormap_rgba=None):
        """Build the 1D RGBA transfer function from the piecewise-
        linear opacity control points + a colormap ramp."""
        n = 256
        x = np.linspace(0, 1, n)
        pts = sorted(self.tf_points)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        alpha = np.interp(x, xs, ys)
        if colormap_rgba is None:
            import matplotlib

            cmap = matplotlib.colormaps["inferno"]
            rgb = (cmap(x)[:, :3] * 255).astype(np.uint8)
        else:
            rgb = colormap_rgba[:, :3]
        tf = np.zeros((n, 4), dtype=np.uint8)
        tf[:, :3] = rgb
        tf[:, 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        dev = self.device
        if self._tf_tex is None:
            self._tf_tex = dev.create_texture(
                size=(n, 1, 1), format="rgba8unorm", dimension="1d",
                usage=(wgpu.TextureUsage.TEXTURE_BINDING
                       | wgpu.TextureUsage.COPY_DST))
            self._bg = None
        dev.queue.write_texture(
            {"texture": self._tf_tex, "mip_level": 0,
             "origin": (0, 0, 0)},
            np.ascontiguousarray(tf).tobytes(),
            {"bytes_per_row": n * 4, "rows_per_image": 1}, (n, 1, 1))

    def render_to_pass(self, rpass, camera):
        if (self._tex3d is None or self._tf_tex is None
                or self.center is None):
            return
        if self._bg is None:
            self._bg = self.device.create_bind_group(
                layout=self._bgl,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._ubuf}},
                    {"binding": 1,
                     "resource": self._tex3d.create_view(
                         dimension="3d")},
                    {"binding": 2, "resource": self._sampler},
                    {"binding": 3,
                     "resource": self._tf_tex.create_view(
                         dimension="1d")},
                    {"binding": 4, "resource": self._sampler},
                ])
        buf = np.zeros(28, dtype=np.float32)
        buf[0:3] = camera.position
        buf[3] = np.radians(camera.fov)
        buf[4:7] = camera.forward
        buf[7] = camera.aspect
        buf[8:11] = camera.right
        buf[11] = self.step_mult
        buf[12:15] = camera.up
        buf[15] = self.vmin
        buf[16:19] = self.center
        buf[19] = self.vmax
        buf[20] = self.half_size
        buf[21] = float(self._tex3d_res)
        buf.view(np.uint32)[22] = self.mode
        buf.view(np.uint32)[23] = self.max_steps
        self.device.queue.write_buffer(self._ubuf, 0, buf.tobytes())
        rpass.set_pipeline(self._pipeline)
        rpass.set_bind_group(0, self._bg)
        rpass.draw(3)


# ---------------------------------------------------------------------------
# GPU pass timing via timestamp queries (Item 4)
# ---------------------------------------------------------------------------

def _init_gpu_timing(renderer):
    """Create the timestamp QuerySet (8 slots) when the device has the
    timestamp-query feature. Returns True on success."""
    dev = renderer.device
    try:
        if "timestamp-query" not in dev.features:
            return False
        renderer._ts_query_set = dev.create_query_set(
            type="timestamp", count=8)
        renderer._ts_resolve_buf = dev.create_buffer(
            size=8 * 8, usage=(wgpu.BufferUsage.QUERY_RESOLVE
                               | wgpu.BufferUsage.COPY_SRC))
        renderer.pass_times_ms = {}
        renderer._ts_ema = {}
        renderer._gpu_timing_ok = True
        return True
    except Exception as e:
        print(f"  gpu timing unavailable: {e}")
        renderer._gpu_timing_ok = False
        return False


def _ts_writes(renderer, begin_idx, end_idx):
    """timestamp_writes descriptor for a render pass, or None."""
    if not getattr(renderer, "_gpu_timing_ok", False):
        return None
    return {"query_set": renderer._ts_query_set,
            "beginning_of_pass_write_index": begin_idx,
            "end_of_pass_write_index": end_idx}


def _read_gpu_timings(renderer):
    """Resolve + read the query set; update the EMA dict. Call at most
    ~1 Hz — does a blocking readback."""
    if not getattr(renderer, "_gpu_timing_ok", False):
        return {}
    try:
        dev = renderer.device
        enc = dev.create_command_encoder()
        enc.resolve_query_set(renderer._ts_query_set, 0, 8,
                              renderer._ts_resolve_buf, 0)
        dev.queue.submit([enc.finish()])
        raw = np.frombuffer(dev.queue.read_buffer(
            renderer._ts_resolve_buf), dtype=np.uint64)
        pairs = {"accum": (0, 1), "resolve": (2, 3)}
        for name, (i0, i1) in pairs.items():
            if raw[i1] > raw[i0]:
                ms = float(raw[i1] - raw[i0]) / 1e6
                prev = renderer._ts_ema.get(name, ms)
                renderer._ts_ema[name] = 0.8 * prev + 0.2 * ms
        renderer.pass_times_ms = dict(renderer._ts_ema)
        return renderer.pass_times_ms
    except Exception:
        renderer._gpu_timing_ok = False
        return {}


WGPURenderer.init_gpu_timing = _init_gpu_timing
WGPURenderer.read_gpu_timings = _read_gpu_timings
WGPURenderer.ts_writes = _ts_writes
