"""wgpu backend application: GLFW window + WGPURenderer + UI overlays."""

import time
import atexit
import numpy as np
import glfw

import wgpu
from wgpu.utils.glfw_present_info import get_glfw_present_info

from .camera import Camera
from .data_manager import SnapshotData
from .wgpu_renderer import WGPURenderer, RenderMode
from .colormaps import colormap_to_texture_data, AVAILABLE_COLORMAPS
from .field_ops import (
    resolve_field,
    compute_weights,
    compute_slot_fields,
    combine_fields,
    make_default_app_state,
    is_los_stale,
    SD_OPS,
    RENDER_MODES,
    VECTOR_PROJECTIONS,
)


def run_wgpu_app(
    snapshot_path,
    width=1920,
    height=1080,
    fov=90.0,
    fullscreen=False,
    screenshot=None,
    no_stars=False,
    types=None,
    center=None,
    center_on=None,
    radius=None,
    colormap=None,
    screenshot_dir=None,
    field=None,
    mode=None,
    filters=None,
):
    """Run the vizmo application with the wgpu backend.

    If `screenshot` is set to a path, the canvas loop runs just long
    enough for GPU init + auto-range to complete, takes a screenshot,
    and exits without entering the interactive loop.
    """
    import os

    snapshot_path = os.path.abspath(snapshot_path)
    if screenshot_dir:
        os.makedirs(screenshot_dir, exist_ok=True)

    # Initialize GLFW without OpenGL (wgpu uses Vulkan/Metal)
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    atexit.register(glfw.terminate)

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)  # No OpenGL context
    glfw.window_hint(glfw.RESIZABLE, True)

    monitor = glfw.get_primary_monitor() if fullscreen else None
    window = glfw.create_window(width, height, "vizmo [wgpu]", monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    # Create wgpu device and canvas context
    present_info = get_glfw_present_info(window)
    canvas_context = wgpu.gpu.get_canvas_context(present_info)

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = adapter.info
    print(f"  wgpu adapter: {info.get('description', 'unknown')}")
    print(f"    vendor:  {info.get('vendor', '?')}")
    print(f"    device:  {info.get('device', '?')}")
    print(f"    backend: {info.get('backend_type', info.get('adapter_type', '?'))}")
    print(f"    driver:  {info.get('driver', info.get('driver_description', '?'))}")

    # Request float32-blendable if available
    req_features = set()
    if "float32-blendable" in adapter.features:
        req_features.add("float32-blendable")
    if "timestamp-query" in adapter.features:
        req_features.add("timestamp-query")

    # Request max storage buffer size the adapter supports (need ~1.6 GB for 134M particles)
    max_ssbo = min(adapter.limits["max-storage-buffer-binding-size"], 2**32 - 1)
    max_buf = min(adapter.limits["max-buffer-size"], 2**38)
    device = adapter.request_device_sync(
        required_features=req_features,
        required_limits={
            "max_storage_buffer_binding_size": max_ssbo,
            "max_buffer_size": max_buf,
        },
    )

    fb_w, fb_h = glfw.get_framebuffer_size(window)
    canvas_context.set_physical_size(fb_w, fb_h)
    # Force linear format — colormap data from matplotlib is already sRGB-encoded,
    # so we must NOT apply sRGB gamma again (bgra8unorm-srgb would double-gamma).
    present_format = "bgra8unorm"
    canvas_context.configure(device=device, format=present_format)

    # Load data
    print(f"Loading {snapshot_path}...")
    def _hsml_progress(msg):
        try:
            glfw.set_window_title(window, f"vizmo [wgpu] | {msg}")
            glfw.poll_events()
        except Exception:
            pass

    data = SnapshotData(snapshot_path, particle_types=types, hsml_progress=_hsml_progress)
    print(f"  {data.n_particles:,} particles loaded (types {data.particle_types})")

    # Camera
    camera = Camera(fov=fov, aspect=width / height)
    boxsize = data.header.get("BoxSize", None)
    camera.auto_scale(data.positions, masses=data.get_field("Masses"), boxsize=boxsize)
    if center is not None or center_on is not None:
        from .framing import find_center, frame_camera

        view_center = (
            np.asarray(center, dtype=np.float64)
            if center is not None
            else find_center(data, center_on)
        )
        used_r = frame_camera(camera, view_center, data.positions, radius=radius)
        # Re-derive flight speed from the framed radius so a tight zoom
        # onto one halo doesn't inherit box-scale movement speed.
        camera.speed = used_r / 5.0
        data.set_view_center(view_center)
        print(
            f"  View centered on ({view_center[0]:.1f}, {view_center[1]:.1f}, "
            f"{view_center[2]:.1f})  r={used_r:.1f}"
        )

    # Renderer
    renderer = WGPURenderer(device, canvas_context, present_format)
    renderer._viewport_width = width  # use window size for LOD, not retina framebuffer size
    if getattr(data, "is_structured_grid", False):
        # Hsml from these readers is cbrt(cell_volume) — i.e. the cell's
        # half-side. Widening to 2× smooths across cell faces and hides
        # the AMR grid pattern.
        renderer.hsml_scale = 2.0
    from vizmo.data_manager import _YtFile
    if isinstance(data._file, _YtFile) and data.n_stars > 0:
        # _read_yt stored StarLuminosity = R_phys² (in code_length²) for
        # cluster particles, so sqrt(L)·world_radius gives world_radius·R_phys.
        # Default the multiplier to 1 so the physical scaling shows as-is.
        renderer.star_world_radius = 1.0

    # Persisted user settings (colormap, mouse, target FPS). CLI flags
    # take precedence over saved values.
    from .session import load_settings, save_settings

    _settings = load_settings()
    if colormap is None and _settings.get("colormap") in AVAILABLE_COLORMAPS:
        colormap = _settings["colormap"]
    if "invert_mouse" in _settings:
        camera.invert_mouse = bool(_settings["invert_mouse"])

    # Colormap
    start_cmap = colormap if colormap in AVAILABLE_COLORMAPS else (colormap or "magma")
    try:
        rgba = colormap_to_texture_data(start_cmap)
    except Exception:
        print(f"  Unknown colormap {start_cmap!r}; using magma")
        start_cmap = "magma"
        rgba = colormap_to_texture_data(start_cmap)
    renderer.set_colormap(rgba)
    if "target_fps" in _settings:
        try:
            renderer.target_fps = float(_settings["target_fps"])
        except (TypeError, ValueError):
            pass
    # Sink-marker colormap. Independent default; user picks the active
    # colour field from the sink panel (default "None" → black fill).
    sink_rgba = colormap_to_texture_data(renderer.sink_cmap_name)
    renderer.set_sink_colormap(sink_rgba)
    if data.n_stars > 0:
        # Hand the renderer the per-sink field dict so the size/colour
        # selectors can resolve names to arrays at draw time.
        renderer._star_fields = data.star_fields
    # Per-sink trajectory positions (RAMSES SINK1/sink_*.txt files). The
    # sink panel will let the user pick which ones to draw.
    renderer.set_sink_trajectory_data(getattr(data, "sink_trajectories", {}))

    # Load particles. set_particles only stores arrays + n_total; the
    # actual GPU upload happens in the background-built grid + GPUCompute
    # init below.
    weights = data.get_field("Masses")
    renderer.set_particles(data.positions, data.hsml, weights)

    # Render + present an empty first frame immediately so the window
    # comes up responsive. The GPU upload follows in the canvas tick.
    try:
        renderer.render(camera, fb_w, fb_h)
        canvas_context.present()
        glfw.poll_events()
    except Exception:
        pass

    from .gpu_compute import GPUCompute

    gpu_compute = None
    glfw.set_window_title(window, "vizmo [wgpu] | Initializing...")

    # UI overlays
    from .wgpu_overlay import (
        WGPUDevOverlay,
        WGPUSinkOverlay,
        WGPUUserMenu,
        WGPUHelpOverlay,
        WGPUToolbar,
        WGPUScaleBar,
        WGPUStatusBar,
        WGPUToastOverlay,
        WGPUAxesGizmo,
        WGPUAnalysisDrawer,
    )
    from .physics import UnitSystem

    overlay = WGPUDevOverlay(device, present_format)
    sink_panel = WGPUSinkOverlay(device, present_format)
    sink_panel.enabled = data.n_stars > 0
    user_menu = WGPUUserMenu(device, present_format)
    help_panel = WGPUHelpOverlay(device, present_format)
    toolbar = WGPUToolbar(device, present_format)
    scale_bar = WGPUScaleBar(device, present_format)
    status_bar = WGPUStatusBar(device, present_format)
    toasts = WGPUToastOverlay(device, present_format)
    gizmo = WGPUAxesGizmo(device, present_format)
    drawer = WGPUAnalysisDrawer(device, present_format)
    units = UnitSystem(data.header)
    # Deferred Shift+click pick: the KD-tree build can take seconds on
    # a 10M+ particle pool, so the click only queues the ray and shows
    # a toast; the main loop runs the pick one frame later.
    # `purpose` is "inspect" (open inspector) or "aperture" (place the
    # analysis-aperture center).
    _pending_pick = {"ray": None, "frames": 0, "purpose": "inspect"}

    # Analysis aperture: a world-space sphere the user places with the
    # mouse (M key). While `placing`, clicks move the center and scroll
    # resizes; M again submits — the center is then refined inside the
    # sphere (densest by default) and all drawer tools compute within it.
    from .wgpu_overlay import WGPUApertureOverlay

    aperture_panel = WGPUApertureOverlay(device, present_format)
    from .wgpu_overlay import WGPUSightlinesOverlay

    sightline_overlay = WGPUSightlinesOverlay(device, present_format)
    # Absorption sightlines: Shift+A toggles placement mode (click two
    # particles to define the segment). Computed columns live on the
    # Sightline objects; the drawer's "sightline" tool lists them.
    _sightlines = {"list": [], "placing": False, "pending_start": None,
                   "trident_procs": []}
    drawer.sightlines = _sightlines["list"]
    _aperture = {
        "active": False,      # submitted and in use
        "placing": False,     # interactive placement in progress
        "center": None,       # (3,) code units
        "radius": None,       # code units
        "center_mode": 0,     # index into analysis.CENTER_MODES
        "shape_idx": 0,       # index into _APERTURE_SHAPES
    }
    # Shape catalogue for the aperture tool (Tab cycles while placing).
    # Each entry: (name, ring RGBA). Non-sphere shapes are oriented by
    # the camera at submit time (axis = camera forward).
    _APERTURE_SHAPES = [
        ("sphere", (120, 200, 255, 255)),
        ("cylinder", (240, 220, 90, 255)),
        ("box", (235, 110, 235, 255)),
        ("slab", (250, 160, 70, 255)),
        ("cone", (110, 230, 130, 255)),
        ("ellipsoid", (240, 240, 240, 255)),
    ]

    def _build_aperture_region():
        """Materialize the current aperture as a SelectionRegion,
        oriented by the camera for the axis-dependent shapes."""
        from . import selection as selmod

        name = _APERTURE_SHAPES[_aperture["shape_idx"]][0]
        c = np.asarray(_aperture["center"], dtype=np.float64)
        R = float(_aperture["radius"])
        fwd = camera.forward.astype(np.float64)
        if name == "cylinder":
            return selmod.Cylinder(c, fwd, R, R)
        if name == "box":
            return selmod.Box(c - R, c + R)
        if name == "slab":
            return selmod.Slab(c, fwd, R / 2.0)
        if name == "cone":
            return selmod.Cone(c, fwd, 30.0, 2.0 * R)
        if name == "ellipsoid":
            return selmod.Ellipsoid(c, [R, 0.7 * R, 0.5 * R])
        return selmod.Sphere(c, R)

    def _focus_center():
        """Center used by orbit / N / F2-F4: aperture when active."""
        if _aperture["active"] and _aperture["center"] is not None:
            return np.asarray(_aperture["center"], dtype=np.float64)
        return data.get_view_center()

    def _world_to_screen(p, fbw, fbh):
        """Project a world point to framebuffer px. None when behind."""
        rel = np.asarray(p, dtype=np.float64) - camera.position
        zf = float(rel @ camera.forward)
        if zf <= 1e-12:
            return None
        xr = float(rel @ camera.right)
        yu = float(rel @ camera.up)
        tan_half = np.tan(np.radians(camera.fov) / 2.0)
        nx = xr / (zf * tan_half * camera.aspect)
        ny = yu / (zf * tan_half)
        return ((nx + 1.0) * 0.5 * fbw, (1.0 - ny) * 0.5 * fbh, zf)

    def _submit_aperture():
        """Refine the center inside the sphere and hand it to the drawer."""
        from .analysis import CENTER_MODES, find_center_in_region

        mode_name = CENTER_MODES[_aperture["center_mode"] % len(CENTER_MODES)]
        try:
            refined = find_center_in_region(
                data, _aperture["center"], _aperture["radius"], mode_name)
        except Exception as e:
            toasts.show(f"Center refine failed: {e}", "error")
            refined = np.asarray(_aperture["center"], dtype=np.float64)
        _aperture["center"] = refined
        _aperture["placing"] = False
        _aperture["active"] = True
        r_kpc = _aperture["radius"] * units.length_to_kpc
        region = _build_aperture_region()
        shape_name = _APERTURE_SHAPES[_aperture["shape_idx"]][0]
        drawer.set_scope(refined, r_kpc, mode_name, region=region)
        toasts.show(
            f"Aperture set: {shape_name}, R={r_kpc:.0f} kpc, "
            f"center={mode_name}", "ok")
    _timings = {"cull": 0, "upload": 0, "render": 0}
    _last_message = ""
    _render_mode = RenderMode.surface_density("Masses")
    _cmap_idx = AVAILABLE_COLORMAPS.index(start_cmap) if start_cmap in AVAILABLE_COLORMAPS else 0
    _s = make_default_app_state(data)
    _sd_fields = _s["sd_fields"]
    _vector_fields = _s["vector_fields"]
    _sd_field = _s["sd_field"]
    _sd_field2 = _s["sd_field2"]
    _sd_op = _s["sd_op"]
    _render_mode_name = _s["render_mode_name"]
    _wa_data_field = _s["wa_data_field"]
    _RENDER_MODES = RENDER_MODES
    _SD_OPS = SD_OPS
    _VECTOR_PROJECTIONS = VECTOR_PROJECTIONS
    _vector_projection = _s["vector_projection"]
    _composite = _s["composite"]
    _slot = _s["slot"]

    # --field / --mode CLI startup view. A non-mass field implies a
    # mass-weighted average unless the user pinned the mode explicitly.
    if field is not None:
        if field not in _sd_fields:
            print(f"  Unknown field {field!r}; available: {', '.join(_sd_fields)}")
            field = None
    if field is not None or mode is not None:
        eff_mode = mode or (
            "WeightedAverage" if field not in (None, "Masses") else "SurfaceDensity"
        )
        _render_mode_name = eff_mode
        if eff_mode in ("WeightedAverage", "WeightedVariance"):
            _wa_data_field = field or _wa_data_field
        else:
            _sd_field = field or _sd_field
        print(f"  Startup view: {eff_mode}({field or _sd_field})")
    # --filter FIELD:LO:HI (repeatable)
    if filters:
        parsed = []
        for spec in filters:
            try:
                fname, lo, hi = spec.rsplit(":", 2)
                parsed.append({"field": fname, "lo": float(lo), "hi": float(hi)})
            except ValueError:
                print(f"  Bad --filter spec {spec!r} (want FIELD:LO:HI); ignored")
        if parsed:
            data.set_filters(parsed)
            print(f"  {len(parsed)} filter(s) active")
    _startup_view_pending = (field is not None or mode is not None
                             or bool(data.filters))

    # Stars
    if no_stars:
        data.n_stars = 0
    if data.n_stars > 0:
        renderer.upload_stars(data.star_positions, data.star_masses, luminosity=getattr(data, "star_luminosity", None))
        if data.n_particles > 0:
            renderer.set_extinction_gas(data.positions, data.masses, data.hsml)
        print(f"  {data.n_stars} star particles loaded")

    # Input callbacks
    def _auto_range_composite_slot(slot_idx, label):
        s = _state["_slot"][slot_idx]
        renderer.resolve_mode = s["resolve"]
        renderer.log_scale = s["log"]

        gpu_ready_now = gpu_compute is not None and getattr(gpu_compute, "_upload_ready", False)
        fb_w_, fb_h_ = glfw.get_framebuffer_size(window)

        if gpu_ready_now:
            # GPU subsample path: upload this slot's mass/qty into the
            # composite slot's per-chunk buffers, bind them as the active
            # slot, and let _render_accum dispatch the splat draws.
            sorted_mass, sorted_qty = _ensure_slot_sorted(slot_idx)
            slot_id = _slot_sorted[slot_idx][0]
            slot_chunks = gpu_compute.upload_subsample_slot(slot_idx, slot_id, sorted_mass, sorted_qty)
            renderer.set_subsample_slot_chunks(slot_idx, slot_chunks)
            renderer.set_active_subsample_slot(slot_idx)
            renderer.n_particles = min(renderer.n_total, renderer._subsample_max_per_frame)
        else:
            # CPU fallback (subsample mode keeps the renderer empty here;
            # the GPU path is required to actually see anything)
            w, q = app_proxy._compute_slot(s)
            renderer.update_weights(w, q)

        renderer._ensure_fbo(fb_w_, fb_h_, which=1)
        renderer._write_camera_uniforms(camera, fb_w_, fb_h_)
        renderer._render_accum(camera, fb_w_, fb_h_, renderer._accum_textures)
        # Lightness (slot 0): mass-weighted entropy, so the surface density
        # range is set by where the actual mass is. Color (slot 1): raw
        # entropy on field values, so the dynamic range is the full spread
        # of the field itself rather than wherever it happens to be dense.
        lo, hi = renderer.read_accum_range(mass_weighted=(slot_idx == 0))
        s["min"] = lo
        s["max"] = hi
        print(f"Auto-range {label}: {lo:.3g} .. {hi:.3g}")

    def _adjust_range(rend, factor):
        mid = (rend.qty_min + rend.qty_max) / 2
        half = (rend.qty_max - rend.qty_min) / 2 * factor
        rend.qty_min = mid - half
        rend.qty_max = mid + half

    def key_callback(win, key, scancode, action, mods):
        nonlocal _cmap_idx, needs_auto_range, ui_hidden
        nonlocal subsample_cap_ceiling, last_subsample_cap
        nonlocal dirty, ui_dirty, idle_streak
        if action in (glfw.PRESS, glfw.REPEAT):
            idle_streak = 0
        # Keystrokes routed into the user_menu (text-field editing) only
        # mutate UI state — no scene re-render needed. The main loop's
        # ui-only dirty path will redraw the panel without re-running
        # particle accumulation.
        if user_menu.on_key(key, action):
            ui_dirty = True
        if sink_panel.on_key(key, action):
            ui_dirty = True
            return
        # Any other PRESS may mutate scene state (R, L, ',', '.', mode
        # toggles, etc.), so force a full re-render.
        if action == glfw.PRESS:
            dirty = True
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                # Esc cancels aperture placement, then closes panels;
                # quits only when nothing is in the way.
                if _aperture["placing"]:
                    _aperture["placing"] = False
                    if not _aperture["active"]:
                        aperture_panel.enabled = False
                    toasts.show("Aperture placement cancelled")
                elif help_panel.enabled:
                    help_panel.enabled = False
                elif drawer.enabled:
                    drawer.enabled = False
                    drawer.mode = None
                else:
                    glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_R:
                if _state["_composite"]:
                    _auto_range_composite_slot(1, "Color")
                else:
                    lo, hi = renderer.read_accum_range()
                    renderer.qty_min = lo
                    renderer.qty_max = hi
                    print(f"Auto-range: {lo:.3g} .. {hi:.3g}")
            elif key == glfw.KEY_T:
                if _state["_composite"]:
                    _auto_range_composite_slot(0, "Lightness")
            elif key == glfw.KEY_L:
                renderer.log_scale = 1 - renderer.log_scale
                needs_auto_range = True
            elif key == glfw.KEY_PERIOD:
                # Raise the auto-LOD subsample-cap ceiling. The cap
                # itself adapts within [floor, ceiling]; this knob is
                # the ceiling. SUBSAMPLE_CAP_HARD_LIMIT is the hard
                # upper bound for safe GPU buffer / dispatch limits.
                subsample_cap_ceiling = min(SUBSAMPLE_CAP_HARD_LIMIT, int(subsample_cap_ceiling * 2))
                print(f"Subsample cap ceiling: " f"{subsample_cap_ceiling/1e6:.1f}M")
            elif key == glfw.KEY_COMMA:
                subsample_cap_ceiling = max(500_000, subsample_cap_ceiling // 2)
                # Clamp the running cap so the change is felt
                # immediately, not just at the next motion frame.
                last_subsample_cap = min(last_subsample_cap, subsample_cap_ceiling)
                renderer.set_subsample_max_per_frame(min(renderer._subsample_max_per_frame, subsample_cap_ceiling))
                print(f"Subsample cap ceiling: " f"{subsample_cap_ceiling/1e6:.1f}M")
            elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
                _adjust_range(renderer, 0.8)
            elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
                _adjust_range(renderer, 1.25)
            elif key == glfw.KEY_C:
                nonlocal _cmap_idx
                _cmap_idx = (_cmap_idx + 1) % len(AVAILABLE_COLORMAPS)
                from .colormaps import colormap_to_texture_data

                renderer.set_colormap(colormap_to_texture_data(AVAILABLE_COLORMAPS[_cmap_idx]))
                _state["_cmap_idx"] = _cmap_idx
            elif key == glfw.KEY_P:
                if mods & glfw.MOD_CONTROL:
                    _export_publication()
                else:
                    _take_screenshot()
            elif key == glfw.KEY_E and (mods & glfw.MOD_CONTROL):
                _export_region_cutout()
            elif key == glfw.KEY_F1 or key == glfw.KEY_H:
                help_panel.enabled = not help_panel.enabled
            elif key == glfw.KEY_BACKSLASH:
                overlay.enabled = not overlay.enabled
            elif key == glfw.KEY_K:
                if mods & glfw.MOD_SHIFT:
                    drawer.toggle("spectrum")
                else:
                    sink_panel.enabled = not sink_panel.enabled
            elif key == glfw.KEY_TAB:
                if _aperture["placing"]:
                    _aperture["shape_idx"] = (
                        _aperture["shape_idx"] + 1) % len(_APERTURE_SHAPES)
                    toasts.show(
                        f"Aperture shape: "
                        f"{_APERTURE_SHAPES[_aperture['shape_idx']][0]}")
                else:
                    ui_hidden = not ui_hidden
            elif key == glfw.KEY_B:
                renderer.cycle_star_band(-1 if (mods & glfw.MOD_SHIFT) else 1)
            elif key == glfw.KEY_O:
                if mods & glfw.MOD_SHIFT:
                    renderer.toggle_star_extinction()
                else:
                    drawer.toggle("orbit")
            elif key == glfw.KEY_LEFT_BRACKET:
                print(f"FOV: {camera.adjust_fov(-5.0):.0f}°")
            elif key == glfw.KEY_RIGHT_BRACKET:
                print(f"FOV: {camera.adjust_fov(+5.0):.0f}°")
            elif key == glfw.KEY_V:
                _toggle_recording()
            elif key == glfw.KEY_G:
                drawer.toggle("phase")
            elif key == glfw.KEY_J:
                drawer.toggle("profile")
            elif key == glfw.KEY_U:
                drawer.refresh()
                drawer.toggle("stats")
            elif key == glfw.KEY_I:
                drawer.toggle("inspector")
            elif key == glfw.KEY_F:
                drawer.toggle("filters")
            elif key == glfw.KEY_M and (mods & glfw.MOD_CONTROL):
                _export_fits_map()
            elif key == glfw.KEY_M:
                if mods & glfw.MOD_SHIFT:
                    # Shift+M: drop the aperture, back to global scope.
                    _aperture["active"] = False
                    _aperture["placing"] = False
                    aperture_panel.enabled = False
                    drawer.clear_scope()
                    toasts.show("Aperture cleared (global scope)")
                elif _aperture["placing"]:
                    if _aperture["center"] is None:
                        toasts.show("Click to place the center first", "warn")
                    else:
                        _submit_aperture()
                else:
                    _aperture["placing"] = True
                    if _aperture["center"] is None:
                        _aperture["center"] = data.get_view_center().copy()
                    if _aperture["radius"] is None:
                        d0 = float(np.linalg.norm(
                            camera.position - _aperture["center"]))
                        _aperture["radius"] = max(d0 * 0.15, 1e-9)
                    aperture_panel.enabled = True
                    toasts.show(
                        "Aperture: click=center, scroll=size, M=set, Esc=cancel",
                        duration=5.0)
            elif key == glfw.KEY_A and (mods & glfw.MOD_SHIFT):
                _sightlines["placing"] = not _sightlines["placing"]
                _sightlines["pending_start"] = None
                if _sightlines["placing"]:
                    toasts.show(
                        "Sightline mode: click start point, then end point",
                        duration=5.0)
                else:
                    toasts.show("Sightline mode off")
            elif key == glfw.KEY_F9:
                vis = not scale_bar.enabled
                scale_bar.enabled = vis
                status_bar.enabled = vis
                gizmo.enabled = vis
                toasts.show(f"Science chrome {'on' if vis else 'off'}")
            elif key == glfw.KEY_N:
                center = _focus_center()
                if mods & glfw.MOD_SHIFT:
                    # Approach: fly to 1/3 of the current distance.
                    rel = camera.position - center
                    d = np.linalg.norm(rel)
                    if d > 0:
                        camera.fly_to(position=center + rel / 3.0,
                                      look_at=center, duration=1.4)
                        toasts.show("Approaching view center")
                else:
                    camera.fly_to(look_at=center, duration=0.8)
                    toasts.show("Looking at view center")
            elif key == glfw.KEY_Y:
                if camera.orbit is not None:
                    camera.stop_orbit()
                    toasts.show("Orbit off")
                else:
                    speed_scale = -1.0 if (mods & glfw.MOD_SHIFT) else 1.0
                    if camera.start_orbit(_focus_center(),
                                          angular_speed=0.25 * speed_scale):
                        toasts.show("Orbiting view center (Y stops)", "ok")
            elif key in (glfw.KEY_F2, glfw.KEY_F3, glfw.KEY_F4):
                center = _focus_center()
                d = float(np.linalg.norm(camera.position - center))
                if d <= 0:
                    d = camera.speed * 5.0
                axis = {glfw.KEY_F2: np.array([1.0, 0, 0]),
                        glfw.KEY_F3: np.array([0, 1.0, 0]),
                        glfw.KEY_F4: np.array([0, 0, 1.0])}[key]
                sign = -1.0 if (mods & glfw.MOD_SHIFT) else 1.0
                up = (np.array([0, 0, 1.0]) if key != glfw.KEY_F4
                      else np.array([0, 1.0, 0]))
                camera.fly_to(position=center + sign * d * axis,
                              look_at=center, up=up, duration=1.2)
                toasts.show(f"View along {'-' if sign < 0 else '+'}"
                            f"{'XYZ'[key - glfw.KEY_F2]}")
            elif glfw.KEY_1 <= key <= glfw.KEY_9:
                slot = str(key - glfw.KEY_0)
                from .session import camera_pose, fly_to_pose, save_bookmarks

                if mods & glfw.MOD_SHIFT:
                    _bookmarks[slot] = camera_pose(camera)
                    save_bookmarks(snapshot_path, _bookmarks)
                    print(f"Bookmark {slot} saved")
                    toasts.show(f"Bookmark {slot} saved", "ok")
                elif slot in _bookmarks:
                    fly_to_pose(camera, _bookmarks[slot])
                    print(f"Bookmark {slot} restored")
                    toasts.show(f"Bookmark {slot}")
                else:
                    toasts.show(f"Bookmark {slot} empty (Shift+{slot} saves)", "warn")
        # [ and ] repeat while held for a smooth zoom.
        if action == glfw.REPEAT:
            if key == glfw.KEY_LEFT_BRACKET:
                camera.adjust_fov(-5.0)
                dirty = True
            elif key == glfw.KEY_RIGHT_BRACKET:
                camera.adjust_fov(+5.0)
                dirty = True
        # Ctrl chords (Ctrl+E export, Ctrl+P figure) must not leak into
        # flight controls (E is roll).
        if not (mods & glfw.MOD_CONTROL):
            camera.on_key(key, action)

    glfw.set_key_callback(window, key_callback)

    # Shared mutable state dict — the app proxy reads/writes this,
    # and the main loop syncs closure locals from it each frame.
    _state = {
        "_sd_field": _sd_field,
        "_sd_field2": _sd_field2,
        "_sd_op": _sd_op,
        "_render_mode_name": _render_mode_name,
        "_wa_data_field": _wa_data_field,
        "_vector_fields": _vector_fields,
        "_vector_projection": _vector_projection,
        "_los_camera_pos": None,
        "_cmap_idx": _cmap_idx,
        "_slot": _slot,
        "_needs_auto_range": False,
        "_composite": _composite,
    }

    def _apply_filters(w):
        """Zero the weights of particles excluded by active filters."""
        if data.filters:
            return (w * data.filter_mask()).astype(np.float32)
        return w

    class _AppProxy:
        """Live proxy that reads/writes shared state dict."""

        def __getattr__(self, name):
            if name == "renderer":
                return renderer
            if name == "camera":
                return camera
            if name in _state:
                return _state[name]
            raise AttributeError(name)

        def __setattr__(self, name, value):
            _state[name] = value

        def _project_field(self, field_name):
            if field_name in _state["_vector_fields"] and _state["_vector_projection"] == "LOS":
                _state["_los_camera_pos"] = camera.position.copy()
            return resolve_field(
                field_name,
                _state["_vector_fields"],
                data,
                _state["_vector_projection"],
                camera.forward,
                camera_position=camera.position,
            )

        def _compute_weights(self):
            if _state["_sd_field"] in _state["_vector_fields"] and _state["_vector_projection"] == "LOS":
                _state["_los_camera_pos"] = camera.position.copy()
            return compute_weights(
                _state["_sd_field"],
                _state.get("_sd_field2", "None"),
                _state.get("_sd_op", "*"),
                _state["_vector_fields"],
                data,
                _state["_vector_projection"],
                camera.forward,
                camera_position=camera.position,
            )

        def _compute_slot(self, slot):
            """Compute weights and qty for a composite slot dict."""
            w, q = compute_slot_fields(
                slot, _state["_vector_fields"], data, camera.forward, camera_position=camera.position
            )
            return _apply_filters(w), q

        def _apply_render_mode(self, auto_range=True):
            nonlocal _render_mode, needs_auto_range
            _state["_los_camera_pos"] = None

            mode = _state["_render_mode_name"]
            _state["_composite"] = mode == "Composite"

            if mode == "Composite":
                _render_mode = RenderMode(name="Composite", weight_field="", qty_field="", resolve_mode=-1)
                # Pre-populate slot caches then auto-range
                for si in range(2):
                    _ensure_slot_sorted(si)
                _auto_range_composite_slot(0, "Lightness")
                _auto_range_composite_slot(1, "Color")
                return

            if mode in ("WeightedAverage", "WeightedVariance"):
                weights = self._project_field(_state["_sd_field"])
                qty = self._project_field(_state["_wa_data_field"])
                if mode == "WeightedVariance":
                    _render_mode = RenderMode.weighted_variance(_state["_wa_data_field"], _state["_sd_field"])
                    renderer.resolve_mode = 2
                else:
                    _render_mode = RenderMode.mass_weighted_average(_state["_wa_data_field"], _state["_sd_field"])
                    renderer.resolve_mode = 1
            else:
                weights = self._compute_weights()
                qty = None
                _render_mode = RenderMode.surface_density(_state["_sd_field"])
                renderer.resolve_mode = 0

            renderer.update_weights(_apply_filters(weights), qty)
            print(
                f"  [diag] mode={mode} n_total={renderer.n_total} "
                f"n_particles={renderer.n_particles} "
                f"resolve_mode={renderer.resolve_mode} "
                f"chunks={renderer._subsample_chunks is not None} "
                f"max_per_frame={renderer._subsample_max_per_frame}",
                flush=True,
            )
            if gpu_compute is not None:
                _t = time.perf_counter()
                gpu_compute.upload_weights(renderer._all_mass, renderer._all_qty)
                print(f"  [diag] upload_weights: {(time.perf_counter()-_t)*1000:.0f}ms", flush=True)
                # GPU LOS projection isn't wired into the subsample path
                # (would need to update qty across every per-chunk buffer
                # per camera rotation). The CPU _project_field above
                # already produced the right LOS-projected qty for the
                # current camera orientation; subsequent rotations will
                # render with stale qty until the user re-applies the
                # render mode (the canvas tick re-fires _apply_render_mode
                # when LOS staleness is detected).
            if auto_range:
                needs_auto_range = True

        def _set_sd_field(self, value):
            _state["_sd_field"] = value
            self._apply_render_mode()

        def _toggle_particle_type(self, p):
            """Toggle a particle type in the loaded pool. Triggers a reload."""
            cur = set(data.particle_types)
            if p in cur:
                cur.discard(p)
            else:
                cur.add(p)
            new_types = sorted(cur)
            _state["_pending_ptype_reload"] = new_types

        def _set_colormap(self, name):
            from .colormaps import colormap_to_texture_data as _cmap_data

            renderer.set_colormap(_cmap_data(name))
            _state["_cmap_idx"] = AVAILABLE_COLORMAPS.index(name) if name in AVAILABLE_COLORMAPS else 0

    app_proxy = _AppProxy()

    def _cursor_to_fb(win):
        """Convert GLFW cursor position (window coords) to framebuffer pixel coords."""
        cx, cy = glfw.get_cursor_pos(win)
        ww, wh = glfw.get_window_size(win)
        fw, fh = glfw.get_framebuffer_size(win)
        return cx * fw / max(ww, 1), cy * fh / max(wh, 1)

    def mouse_button_callback(win, button, action, mods):
        nonlocal dirty, idle_streak
        # Any mouse click can mutate UI-internal state (open a dropdown,
        # focus a text field, drag a slider) that isn't reflected in
        # `state_sig`. Force at least the next frame to render so the
        # idle short-circuit doesn't swallow the visible response.
        dirty = True
        idle_streak = 0
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            x, y = _cursor_to_fb(win)
            if help_panel.enabled and help_panel.on_click(x, y):
                return
            shift_held = (
                glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                or glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            )
            if _sightlines["placing"] and not shift_held:
                fw, fh = glfw.get_framebuffer_size(win)
                nx = 2.0 * x / max(fw, 1) - 1.0
                ny = 1.0 - 2.0 * y / max(fh, 1)
                tan_half = np.tan(np.radians(camera.fov) / 2.0)
                ray = (camera.forward
                       + nx * tan_half * camera.aspect * camera.right
                       + ny * tan_half * camera.up)
                _pending_pick["ray"] = ray / np.linalg.norm(ray)
                _pending_pick["frames"] = 2
                _pending_pick["purpose"] = "sightline"
                if getattr(data, "_pick_tree", None) is None:
                    toasts.show("Building spatial index...", "info")
                return
            if _aperture["placing"] and not shift_held:
                # Aperture placement: a plain click drops the center on
                # the particle under the cursor (mouse-look is disabled
                # for the duration of placement).
                fw, fh = glfw.get_framebuffer_size(win)
                nx = 2.0 * x / max(fw, 1) - 1.0
                ny = 1.0 - 2.0 * y / max(fh, 1)
                tan_half = np.tan(np.radians(camera.fov) / 2.0)
                ray = (
                    camera.forward
                    + nx * tan_half * camera.aspect * camera.right
                    + ny * tan_half * camera.up
                )
                _pending_pick["ray"] = ray / np.linalg.norm(ray)
                _pending_pick["frames"] = 2
                _pending_pick["purpose"] = "aperture"
                if getattr(data, "_pick_tree", None) is None:
                    toasts.show("Building spatial index...", "info")
                return
            if shift_held:
                # Shift+click: pick the particle under the cursor. The
                # actual KD-tree query runs from the main loop a frame
                # later so the "building index" toast can present first.
                fw, fh = glfw.get_framebuffer_size(win)
                nx = 2.0 * x / max(fw, 1) - 1.0
                ny = 1.0 - 2.0 * y / max(fh, 1)
                tan_half = np.tan(np.radians(camera.fov) / 2.0)
                ray = (
                    camera.forward
                    + nx * tan_half * camera.aspect * camera.right
                    + ny * tan_half * camera.up
                )
                _pending_pick["ray"] = ray / np.linalg.norm(ray)
                _pending_pick["frames"] = 2
                _pending_pick["purpose"] = "inspect"
                if getattr(data, "_pick_tree", None) is None:
                    toasts.show("Building spatial index...", "info")
                return
            dr_action = drawer.on_click(x, y)
            if dr_action:
                if dr_action == "center_on_pick" and drawer._picked_index is not None:
                    new_c = data.positions[drawer._picked_index].copy()
                    data.set_view_center(new_c)
                    scale_bar._last_key = None
                    app_proxy._apply_render_mode(auto_range=False)
                    toasts.show("View center moved to picked particle", "ok")
                elif dr_action == "sightline_csv":
                    if _sightlines["list"]:
                        import os as _os

                        from .spectro import sightlines_to_csv

                        out = _os.path.join(
                            screenshot_dir or ".",
                            f"vizmo_sightlines_{int(time.time())}.csv")
                        sightlines_to_csv(out, _sightlines["list"])
                        toasts.show(
                            f"Sightlines saved: {_os.path.basename(out)}",
                            "ok")
                    else:
                        toasts.show("No sightlines yet", "warn")
                elif dr_action == "sightline_trident":
                    if not _sightlines["list"]:
                        toasts.show("No sightlines yet", "warn")
                    else:
                        try:
                            from .spectro import launch_trident_spectrum

                            proc = launch_trident_spectrum(
                                _sightlines["list"][-1], snapshot_path,
                                os.path.join(screenshot_dir or ".",
                                             "spectra"))
                            _sightlines["trident_procs"].append(
                                (_sightlines["list"][-1], proc))
                            toasts.show(
                                f"Trident running for "
                                f"{_sightlines['list'][-1].label}...", "info",
                                duration=6.0)
                        except ImportError as e:
                            toasts.show(str(e), "error", duration=8.0)
                elif dr_action == "sightline_clear":
                    _sightlines["list"].clear()
                    sightline_overlay.enabled = False
                    drawer.refresh()
                elif dr_action == "orbit_compute":
                    _compute_orbit(stream=False)
                elif dr_action == "orbit_stream":
                    _compute_orbit(stream=True)
                elif dr_action in ("profile_csv", "phase_save",
                                   "stats_json", "spectrum_csv",
                                   "orbit_csv"):
                    _handle_drawer_export(dr_action)
                elif (isinstance(dr_action, tuple) and dr_action
                        and str(dr_action[0]).startswith("f_")):
                    _handle_filter_action(dr_action)
                elif dr_action == "cycle_center" and _aperture["active"]:
                    from .analysis import CENTER_MODES

                    _aperture["center_mode"] = (
                        _aperture["center_mode"] + 1) % len(CENTER_MODES)
                    _submit_aperture()
                return
            tb_action = toolbar.on_click(x, y)
            if tb_action:
                if tb_action == "auto_range":
                    if _state["_composite"]:
                        _auto_range_composite_slot(0, "Lightness")
                        _auto_range_composite_slot(1, "Color")
                    else:
                        _state["_needs_auto_range"] = True
                elif tb_action == "screenshot":
                    _take_screenshot()
                elif tb_action == "publication":
                    _export_publication()
                elif tb_action == "record":
                    _toggle_recording()
                elif tb_action in ("inspector", "phase", "profile", "stats",
                                   "filters"):
                    if tb_action == "stats":
                        drawer.refresh()
                    drawer.toggle(tb_action)
                elif tb_action == "aperture":
                    if _aperture["placing"]:
                        if _aperture["center"] is not None:
                            _submit_aperture()
                    elif _aperture["active"]:
                        _aperture["active"] = False
                        aperture_panel.enabled = False
                        drawer.clear_scope()
                        toasts.show("Aperture cleared (global scope)")
                    else:
                        _aperture["placing"] = True
                        if _aperture["center"] is None:
                            _aperture["center"] = data.get_view_center().copy()
                        if _aperture["radius"] is None:
                            d0 = float(np.linalg.norm(
                                camera.position - _aperture["center"]))
                            _aperture["radius"] = max(d0 * 0.15, 1e-9)
                        aperture_panel.enabled = True
                        toasts.show(
                            "Aperture: click=center, scroll=size, M=set, Esc=cancel",
                            duration=5.0)
                elif tb_action == "orbit":
                    if camera.orbit is not None:
                        camera.stop_orbit()
                        toasts.show("Orbit off")
                    elif camera.start_orbit(_focus_center()):
                        toasts.show("Orbiting view center", "ok")
                elif tb_action == "export_region":
                    _export_region_cutout()
                elif tb_action == "fits_map":
                    _export_fits_map()
                elif tb_action == "help":
                    help_panel.enabled = not help_panel.enabled
                return
            if user_menu.on_click(x, y, app_proxy):
                return
            if sink_panel.enabled and sink_panel.on_click(x, y, renderer):
                return
            if overlay.enabled and overlay.on_click(x, y, renderer):
                return
        camera.on_mouse_button(button, action)

    def cursor_callback(win, x, y):
        camera.on_cursor(x, y)

    def scroll_callback(win, xoffset, yoffset):
        nonlocal dirty, ui_dirty, idle_streak
        idle_streak = 0
        if user_menu.on_scroll(yoffset):
            ui_dirty = True
            return
        # Aperture placement: scroll resizes the sphere.
        if _aperture["placing"]:
            _aperture["radius"] *= 1.1 ** yoffset
            dirty = True
            return
        # Ctrl+scroll: optical zoom (FOV). Plain scroll: flight speed.
        ctrl_held = (
            glfw.get_key(win, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
            or glfw.get_key(win, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
        )
        if ctrl_held:
            dirty = True
            camera.adjust_fov(-2.0 * yoffset)
            return
        # Camera speed change: full re-render
        dirty = True
        camera.on_scroll(yoffset)

    def char_callback(win, codepoint):
        nonlocal ui_dirty, idle_streak
        # Text-field editing only mutates UI state — no scene re-render.
        ui_dirty = True
        idle_streak = 0
        user_menu.on_char(codepoint, app_proxy)
        sink_panel.on_char(codepoint)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_char_callback(window, char_callback)

    needs_auto_range = True
    ui_hidden = False
    # Camera bookmarks: per-snapshot poses persisted under ~/.config/vizmo.
    from .session import load_bookmarks

    _bookmarks = load_bookmarks(snapshot_path)
    if _bookmarks:
        print(f"  {len(_bookmarks)} camera bookmark(s) loaded (1-9 to jump, Shift+1-9 to save)")

    # Frame recording (V key): dump every presented frame as a PNG into
    # a timestamped directory for movie assembly.
    _recording = {"dir": None, "frame": 0}

    def _toggle_recording():
        import os

        if _recording["dir"] is None:
            base = screenshot_dir or "."
            _recording["dir"] = os.path.join(base, f"vizmo_rec_{int(time.time())}")
            os.makedirs(_recording["dir"], exist_ok=True)
            _recording["frame"] = 0
            print(f"Recording frames to {_recording['dir']}/")
            toasts.show("Recording started", "ok")
        else:
            d, n = _recording["dir"], _recording["frame"]
            _recording["dir"] = None
            print(f"Recording stopped: {n} frames in {d}/")
            toasts.show(f"Recording stopped: {n} frames", "ok")
            print(f"  Make a movie with e.g.: ffmpeg -framerate 30 -i {d}/frame_%05d.png -pix_fmt yuv420p out.mp4")

    # Main loop state
    last_time = time.perf_counter()
    frame_count = 0
    fps_time = time.perf_counter()
    # Wall-clock duration of the previous renderer.render() call. This is
    # the only signal that reflects real GPU latency (encode/submit are
    # async). Used by the subsample cap-growth gate so we don't overshoot
    # into a backlogged swapchain.
    last_render_ms = 0.0
    smooth_render_ms = 0.0  # EMA of last_render_ms
    fps = 0.0

    # Idle-frame short-circuit. We re-render only when something visible
    # could have changed. Tracked: camera movement, framebuffer resize,
    # _state mutations (qty range, log, render mode, colormap, composite
    # slot params), the per-frame subsample cap (progressive refinement),
    # current LOD, and pending auto-range. When none changed we skip
    # render+overlay+submit entirely and sleep briefly.
    # `dirty` forces a full scene re-render (accum + resolve). `ui_dirty`
    # forces a UI-only redraw (resolve over the existing accum textures
    # + overlay/menu repaint), which is cheap enough to be responsive
    # even on 100M+ particle snapshots while text is being typed.
    dirty = True
    ui_dirty = False
    prev_state_sig = None
    prev_cap = None
    prev_n_particles = None
    prev_fb_size = (0, 0)
    # Number of consecutive idle frames seen. We only start sleeping
    # after a few in a row so a transient `moved=False` frame in the
    # middle of a slow rotation doesn't introduce a visible stutter.
    idle_streak = 0
    IDLE_STREAK_THRESHOLD = 6
    # Tracks the camera position from the previous frame so we can
    # detect translations independently of rotations. Per-particle LOS
    # is invariant under rotation, so only translations should freeze
    # the slot LOS cache.
    prev_camera_pos = camera.position.copy()

    # Cap-based auto-LOD state
    was_moving = False
    # Per-frame instance cap for the subsample splat path. Initialized
    # to 4M; afterwards adapted to the highest cap that sustained target FPS.
    last_subsample_cap = 4_000_000
    # Auto-LOD ceiling for the per-frame subsample cap. The cap adapts
    # within [floor, ceiling]; the , / . keys halve / double this.
    # SUBSAMPLE_CAP_HARD_LIMIT is the upper bound (safe GPU dispatch /
    # buffer headroom on commodity hardware).
    SUBSAMPLE_CAP_HARD_LIMIT = 200_000_000
    subsample_cap_ceiling = 16_000_000
    # Don't grow the cap before the user has moved the camera at least
    # once — startup has no FPS history to validate growth against.
    has_moved_ever = False
    # Frames to wait before firing the post-init auto-range. Set when
    # GPU subsample chunks are first wired up.
    pending_auto_range_frames = 0
    smooth_fps_ema = 0.0

    # Pre-sort and cache slot weight arrays (done once per slot config change)
    _slot_sorted = [None, None]  # (slot_id, sorted_mass, sorted_qty) per slot

    def _ensure_slot_sorted(slot_idx, skip_los_recompute=False):
        """Ensure sorted mass/qty arrays are cached for this slot. Returns (mass, qty).

        For LOS vector qty fields, qty was historically a placeholder — the
        GPU LOS projection path filled it in. The subsample-mode pipeline
        doesn't run GPU LOS projection, so we always compute qty on the CPU
        here. Per-particle LOS is invariant under camera rotation but goes
        stale on translation; the canvas loop calls _apply_render_mode again
        when translation moves the camera past the LOS staleness threshold.

        `skip_los_recompute=True` makes the cache key ignore the current
        camera position for LOS vector slots (any cached entry, regardless
        of where it was computed, will be reused). The per-frame composite
        render path uses this while the camera is translating to avoid a
        ~1.5 s/frame CPU reprojection on every micro-translation — the
        stale projection is acceptable while the user is moving; once they
        stop, the gate opens and the next frame refreshes once.
        """
        sl = _state["_slot"][slot_idx]
        # When the slot uses an LOS-projected vector field, the cached
        # projection depends on the camera position — include a quantized
        # position in the cache key so translation invalidates the entry.
        is_los_vec = sl.get("proj") == "LOS" and (
            sl["weight"] in _state["_vector_fields"]
            or sl.get("weight2", "None") in _state["_vector_fields"]
            or (sl["mode"] in ("WeightedAverage", "WeightedVariance") and sl["data"] in _state["_vector_fields"])
        )
        if is_los_vec and not skip_los_recompute:
            # Per-particle LOS depends only on camera *position*
            # (rotation leaves the camera→particle direction unchanged).
            pos_key = tuple(int(round(float(c) * 1000)) for c in camera.position)
        else:
            pos_key = None
        slot_id = (
            sl["weight"],
            sl.get("weight2", "None"),
            sl.get("op", "*"),
            sl["mode"],
            sl["data"],
            sl.get("proj", "LOS"),
            pos_key,
        )
        cached = _slot_sorted[slot_idx]
        if cached is not None and cached[0] == slot_id:
            return cached[1], cached[2]

        # Compute weight (and optional second weight field) on the CPU.
        # The GPU upload path is in raw particle order, so no sort is
        # applied here.
        proj = sl.get("proj", "LOS")
        vf = _state["_vector_fields"]
        w = resolve_field(sl["weight"], vf, data, proj, camera.forward, camera_position=camera.position)
        w2_name = sl.get("weight2", "None")
        if w2_name != "None":
            w2 = resolve_field(w2_name, vf, data, proj, camera.forward, camera_position=camera.position)
            w = combine_fields(w, w2, sl.get("op", "*"))
        sm = _apply_filters(w.astype(np.float32))

        if sl["mode"] in ("WeightedAverage", "WeightedVariance"):
            q = resolve_field(sl["data"], vf, data, proj, camera.forward, camera_position=camera.position)
            sq = q.astype(np.float32)
        else:
            sq = sm

        _slot_sorted[slot_idx] = (slot_id, sm, sq)
        return sm, sq

    def _render_composite_frame(fb_w, fb_h, encoder=None, screen_view=None, skip_los_recompute=False, skip_accum=False):
        """Render two fields into separate FBOs and composite them.

        If `encoder` and `screen_view` are provided, all sub-passes (two
        accum passes + one composite resolve) are appended to the shared
        encoder and the caller submits. Otherwise the legacy path runs
        each sub-call with its own encoder+submit.

        `skip_los_recompute=True` (passed from the main loop while the
        camera is translating) makes the LOS vector slot cache ignore the
        current camera position, so we don't pay a ~1.5 s/frame CPU
        reprojection on every micro-translation. The cached LOS field is
        slightly stale during motion; on stop the gate opens and the next
        frame recomputes once with the fresh position.
        """
        renderer._ensure_fbo(fb_w, fb_h, which=1)
        renderer._ensure_fbo(fb_w, fb_h, which=2)

        gpu_ready_now = gpu_compute is not None and getattr(gpu_compute, "_upload_ready", False)
        accum_sets = [renderer._accum_textures, renderer._accum_textures2]

        if not skip_accum:
            for i in range(2):
                sl = _state["_slot"][i]
                if gpu_ready_now:
                    # Subsample path: ensure the slot's sorted mass/qty is
                    # cached for the *current* camera direction (the cache
                    # key includes a quantized fwd for LOS slots), then
                    # upload to the slot's per-chunk buffers and render.
                    sorted_mass, sorted_qty = _ensure_slot_sorted(i, skip_los_recompute=skip_los_recompute)
                    slot_id = _slot_sorted[i][0]
                    slot_chunks = gpu_compute.upload_subsample_slot(i, slot_id, sorted_mass, sorted_qty)
                    renderer.set_subsample_slot_chunks(i, slot_chunks)
                    renderer.set_active_subsample_slot(i)
                    renderer.n_particles = min(renderer.n_total, renderer._subsample_max_per_frame)
                else:
                    w, q = app_proxy._compute_slot(sl)
                    renderer.update_weights(w, q)

                renderer._write_camera_uniforms(camera, fb_w, fb_h)
                renderer._render_accum(camera, fb_w, fb_h, accum_sets[i], encoder=encoder)
            # Reset active slot so non-composite passes use the default path.
            renderer.set_active_subsample_slot(None)

        s0, s1 = _state["_slot"][0], _state["_slot"][1]
        renderer.render_composite(
            camera,
            fb_w,
            fb_h,
            s0["resolve"],
            s0["min"],
            s0["max"],
            s0["log"],
            s1["resolve"],
            s1["min"],
            s1["max"],
            s1["log"],
            encoder=encoder,
            screen_view=screen_view,
        )

    def _take_screenshot(path=None, quiet=False):
        """Render the current view at full framebuffer resolution into
        an offscreen texture and save it. Filename defaults to
        vizmo_<timestamp>.png in --screenshot-dir (or cwd).
        """
        import os

        if path is None:
            path = f"vizmo_{int(time.time())}.png"
            if screenshot_dir:
                path = os.path.join(screenshot_dir, path)
        fb_w_, fb_h_ = glfw.get_framebuffer_size(window)
        if _state["_composite"]:
            s0 = _state["_slot"][0]
            s1 = _state["_slot"][1]
            comp = (s0["resolve"], s0["min"], s0["max"], s0["log"], s1["resolve"], s1["min"], s1["max"], s1["log"])
            renderer.screenshot(path, fb_w_, fb_h_, camera, composite_args=comp, quiet=quiet)
        else:
            renderer.screenshot(path, fb_w_, fb_h_, camera, quiet=quiet)
        if not quiet:
            toasts.show(f"Screenshot saved: {os.path.basename(path)}", "ok")
        return os.path.abspath(path)

    def _export_publication():
        """Render a clean frame and burn in colorbar + scale bar + caption."""
        import os
        import tempfile

        ts = int(time.time())
        tmp = os.path.join(tempfile.gettempdir(), f"vizmo_raw_{ts}.png")
        out = os.path.join(screenshot_dir or ".", f"vizmo_fig_{ts}.png")
        try:
            _take_screenshot(tmp, quiet=True)
            from .export import annotate_screenshot
            from .physics import field_unit_label

            if _state["_composite"]:
                s1 = _state["_slot"][1]
                qmin, qmax, qlog = s1["min"], s1["max"], bool(s1["log"])
                field = s1["data"] if s1["mode"] != "SurfaceDensity" else s1["weight"]
            else:
                qmin, qmax = renderer.qty_min, renderer.qty_max
                qlog = bool(renderer.log_scale)
                field = (
                    _state["_wa_data_field"]
                    if _state["_render_mode_name"] in ("WeightedAverage", "WeightedVariance")
                    else _state["_sd_field"]
                )
            center = data.get_view_center()
            dist = float(np.linalg.norm(camera.position - center))
            fb_w_, fb_h_ = glfw.get_framebuffer_size(window)
            world_h = 2.0 * dist * np.tan(np.radians(camera.fov) / 2.0)
            kpc_per_px = world_h * units.length_to_kpc / max(fb_h_, 1)
            zcap = (f"z={units.redshift:.2f}   "
                    if units.cosmological and units.redshift > 1e-3 else "")
            caption = (f"{os.path.basename(snapshot_path)}   {zcap}"
                       f"{_state['_render_mode_name']}")
            annotate_screenshot(
                tmp, out,
                cmap_name=AVAILABLE_COLORMAPS[_state["_cmap_idx"]],
                qty_min=qmin, qty_max=qmax, log_scale=qlog,
                field_label=field, unit_label=field_unit_label(field),
                kpc_per_px=kpc_per_px, caption=caption,
            )
            print(f"  Publication figure: {out}")
            toasts.show(f"Figure saved: {os.path.basename(out)}", "ok")
        except Exception as e:
            toasts.show(f"Figure export failed: {e}", "error")
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _export_region_cutout():
        """Write the sphere around the view center to an HDF5 cutout."""
        import os

        try:
            from .export import export_region

            radius = drawer._stats_radius_kpc  # None -> auto (25th pct)
            out = os.path.join(screenshot_dir or ".",
                               f"vizmo_region_{int(time.time())}.hdf5")
            path, n = export_region(data, radius_kpc=radius, path=out)
            print(f"  Region cutout: {path} ({n:,} particles)")
            toasts.show(f"Cutout saved: {n:,} particles", "ok")
        except Exception as e:
            toasts.show(f"Cutout export failed: {e}", "error")

    def _export_fits_map():
        """Kernel-projected FITS map of the active field over the
        aperture (or a default sphere about the view center), along
        the camera's dominant axis."""
        import os

        try:
            from .export import export_fits_map

            if _aperture["active"]:
                center = _aperture["center"]
                radius_kpc = _aperture["radius"] * units.length_to_kpc
            else:
                center = data.get_view_center()
                radius_kpc = None
            fwd = camera.forward
            axis = "xyz"[int(np.argmax(np.abs(fwd)))]
            field = (
                _state["_wa_data_field"]
                if _state["_render_mode_name"] in ("WeightedAverage", "WeightedVariance")
                else _state["_sd_field"]
            )
            out = os.path.join(screenshot_dir or ".",
                               f"vizmo_map_{field}_{int(time.time())}.fits")
            toasts.show(f"Computing {field} map...", "info")
            fpath, ppath = export_fits_map(
                data, field=field, center=center,
                radius_kpc=radius_kpc, path=out, axis=axis)
            print(f"  FITS map: {fpath}")
            toasts.show(f"FITS map saved: {os.path.basename(fpath)}", "ok")
        except Exception as e:
            toasts.show(f"FITS export failed: {e}", "error")

    def _compute_orbit(stream=False):
        """Integrate the picked particle's orbit (or a mock stream) in
        an NFW potential fit to the matter around the focus center."""
        from . import orbit as orbmod

        idx = drawer._picked_index
        if idx is None:
            toasts.show("Shift+click a particle first", "warn")
            return
        try:
            center = _focus_center()
            region = drawer._active_region()
            toasts.show("Fitting potential + integrating...", "info")
            phi = orbmod.build_potential_from_snapshot(
                data.positions, data.masses, units,
                method="nfw_fit", aperture_region=region, center=center)
            kpc = units.length_to_kpc
            p0 = (data.positions[idx] - center) * kpc
            v_all = data.get_vector_field("Velocities")
            v0 = np.asarray(v_all[idx], dtype=np.float64) * units.velocity_to_kms
            if stream:
                st = orbmod.generate_mock_stream(
                    p0, v0, phi, m_satellite=1e9, n_tracers=60,
                    t_end_gyr=2.0, n_steps=300)
                o = st["satellite_orbit"]
                props = orbmod.compute_orbital_properties(o, phi=phi)
                n_strip = int(np.isfinite(st["stripping_time"]).sum())
                toasts.show(
                    f"Stream: r_J={st['r_jacobi']:.2f} kpc, "
                    f"{n_strip}/60 tracers stripped", "ok")
            else:
                o = orbmod.integrate_orbit(p0, v0, phi, t_end_gyr=2.0)
                props = orbmod.compute_orbital_properties(o, phi=phi)
                toasts.show(
                    f"Orbit: peri={props['r_peri']:.1f} apo="
                    f"{props['r_apo']:.1f} kpc e={props['eccentricity']:.2f}",
                    "ok")
            drawer._last_orbit = (o, props)
            drawer.refresh()
        except Exception as e:
            toasts.show(f"Orbit failed: {e}", "error")

    def _handle_drawer_export(action):
        """Write the drawer's last computed result to disk."""
        import os

        base = screenshot_dir or "."
        ts = int(time.time())
        try:
            if action == "profile_csv" and drawer._last_profile is not None:
                from .analysis import profile_to_csv

                r, prof, fld, unit = drawer._last_profile
                out = os.path.join(base, f"vizmo_profile_{fld}_{ts}.csv")
                profile_to_csv(out, r, prof, fld, unit)
                toasts.show(f"Profile saved: {os.path.basename(out)}", "ok")
            elif action == "phase_save" and drawer._last_phase is not None:
                from astropy.io import fits as pyfits

                ph = drawer._last_phase
                out = os.path.join(
                    base, f"vizmo_phase_{ph['xfield']}_{ph['yfield']}_{ts}.fits")
                hdu = pyfits.PrimaryHDU(ph["H"].T.astype(np.float32))
                hd = hdu.header
                hd["XFIELD"] = ph["xfield"]
                hd["YFIELD"] = ph["yfield"]
                hd["XLOG"] = ph["xlog"]
                hd["YLOG"] = ph["ylog"]
                hd["WEIGHT"] = ph["weighting"]
                hd["XMIN"], hd["XMAX"] = ph["xedges"][0], ph["xedges"][-1]
                hd["YMIN"], hd["YMAX"] = ph["yedges"][0], ph["yedges"][-1]
                hdu.writeto(out, overwrite=True)
                toasts.show(f"Phase histogram saved: {os.path.basename(out)}", "ok")
            elif action == "stats_json":
                from .analysis import region_stats, halo_properties, stats_to_json
                from .physics import UnitSystem as _US

                sc_center, sc_radius = drawer._active_scope()
                r_use = (sc_radius if sc_radius is not None
                         else drawer._stats_radius_kpc)
                rows, used_r = region_stats(data, center=sc_center,
                                            radius_kpc=r_use)
                halo_rows = halo_properties(data, center=sc_center,
                                            radius_kpc=used_r)
                meta = {
                    "snapshot": snapshot_path,
                    "redshift": float(units.redshift),
                    "radius_kpc": used_r,
                    "center_code_units": (
                        None if sc_center is None
                        else [float(v) for v in sc_center]),
                    "particle_types": list(data.particle_types),
                    "timestamp": ts,
                }
                out = os.path.join(base, f"vizmo_stats_{ts}.json")
                stats_to_json(out, rows, halo_rows, meta)
                toasts.show(f"Stats saved: {os.path.basename(out)}", "ok")
            elif action == "orbit_csv" and drawer._last_orbit is not None:
                from .orbit import orbit_to_csv

                out = os.path.join(base, f"vizmo_orbit_{ts}.csv")
                orbit_to_csv(out, drawer._last_orbit[0])
                toasts.show(f"Orbit saved: {os.path.basename(out)}", "ok")
            elif action == "spectrum_csv" and drawer._last_ps is not None:
                from .power_spectrum import power_spectrum_to_csv

                out = os.path.join(base, f"vizmo_pk_{ts}.csv")
                power_spectrum_to_csv(out, drawer._last_ps)
                toasts.show(f"P(k) saved: {os.path.basename(out)}", "ok")
            else:
                toasts.show("Nothing to export yet", "warn")
        except Exception as e:
            toasts.show(f"Export failed: {e}", "error")

    def _handle_filter_action(action):
        """Mutate data.filters from a drawer action and re-render.

        Ranges nudge in percentile space (5-point steps from a cached
        subsampled percentile table) so the controls behave sensibly on
        wildly log-distributed fields.
        """
        kind = action[0]
        filters = list(data.filters)
        if kind == "f_add":
            field = action[1]
            pct = drawer._percentiles(data, field)
            filters.append({"field": field, "lo": float(pct[5]),
                            "hi": float(pct[95]), "plo": 5, "phi": 95})
        elif kind == "f_del":
            i = action[1]
            if 0 <= i < len(filters):
                filters.pop(i)
        elif kind in ("f_lo", "f_hi"):
            i, step = action[1], action[2]
            if not (0 <= i < len(filters)):
                return
            f = filters[i]
            pct = drawer._percentiles(data, f["field"])
            if kind == "f_lo":
                f["plo"] = int(np.clip(f.get("plo", 5) + step, 0,
                                       f.get("phi", 95) - 1))
                f["lo"] = float(pct[f["plo"]])
            else:
                f["phi"] = int(np.clip(f.get("phi", 95) + step,
                                       f.get("plo", 5) + 1, 100))
                f["hi"] = float(pct[f["phi"]])
        else:
            return
        data.set_filters(filters)
        try:
            app_proxy._apply_render_mode(auto_range=False)
        except Exception as e:
            toasts.show(f"Filter apply failed: {e}", "error")
            return
        drawer.refresh()
        if filters:
            n_vis = int(data.filter_mask().sum())
            toasts.show(
                f"Filters: {n_vis/1e6:.2f}M / {data.n_particles/1e6:.1f}M "
                f"particles pass", "ok")
        else:
            toasts.show("Filters cleared", "ok")

    print("vizmo [wgpu] running. WASD=move, mouse=look, F1/H=help, ESC=quit, R=auto-range, P=screenshot.")

    while not glfw.window_should_close(window):
        now = time.perf_counter()
        dt_raw = now - last_time
        last_time = now
        # Clamp dt so a single slow frame can't catapult the camera by
        # 10x the normal step. The auto-LOD PID can briefly oscillate
        # frame time on cost-curve discontinuities (e.g. multigrid level
        # boundaries), and an unclamped dt turns that into visible
        # position lurches. 1/15 s matches the default target FPS.
        dt = min(dt_raw, 1.0 / 15.0)

        # FPS
        frame_count += 1
        if now - fps_time > 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now
            n_vis = renderer.n_particles
            n_tot = renderer.n_total
            init_msg = " | Initializing GPU..." if gpu_compute is None else ""
            rec_msg = f" | REC {_recording['frame']}" if _recording["dir"] is not None else ""
            snap_name = os.path.basename(snapshot_path)
            glfw.set_window_title(
                window,
                f"vizmo — {snap_name} | {fps:.0f} fps | "
                f"{n_vis/1e6:.1f}M/{n_tot/1e6:.1f}M | "
                f"fov {camera.fov:.0f}° | spd {camera.speed:.3g} | "
                f"{AVAILABLE_COLORMAPS[_state['_cmap_idx']]}{rec_msg}{init_msg}",
            )

        glfw.poll_events()

        # Deferred Shift+click pick: runs after the click frame has
        # presented (so the "building index" toast is visible during
        # the KD-tree build).
        if _pending_pick["ray"] is not None:
            if _pending_pick["frames"] > 0:
                _pending_pick["frames"] -= 1
                dirty = True
            else:
                ray = _pending_pick["ray"]
                _pending_pick["ray"] = None
                from .analysis import pick_particle

                try:
                    idx = pick_particle(camera.position, ray, data,
                                        fov_deg=camera.fov)
                except Exception as e:
                    idx = None
                    toasts.show(f"Pick failed: {e}", "error")
                if idx is None:
                    toasts.show("No particle under cursor", "warn")
                elif _pending_pick["purpose"] == "sightline":
                    p_pick = data.positions[idx].copy()
                    if _sightlines["pending_start"] is None:
                        _sightlines["pending_start"] = p_pick
                        toasts.show("Sightline start set — click end point")
                    else:
                        from .spectro import (Sightline,
                                              compute_los_column_densities)

                        n_sl = len(_sightlines["list"]) + 1
                        sl = Sightline(
                            start=_sightlines["pending_start"],
                            end=p_pick, label=f"SL-{n_sl:03d}")
                        _sightlines["pending_start"] = None
                        try:
                            compute_los_column_densities(sl, data)
                            _sightlines["list"].append(sl)
                            sightline_overlay.enabled = True
                            drawer.enabled = True
                            drawer.mode = "sightline"
                            toasts.show(sl.summary(), "ok", duration=6.0)
                        except Exception as e:
                            toasts.show(f"Sightline failed: {e}", "error")
                elif _pending_pick["purpose"] == "aperture":
                    _aperture["center"] = data.positions[idx].copy()
                    toasts.show("Aperture center placed (M to set)", "ok")
                else:
                    drawer.open_inspector(idx)
                    toasts.show(f"Picked particle {idx:,}", "ok")
                dirty = True

        # Particle-type reload requested from the UI tickboxes.
        pending_types = _state.get("_pending_ptype_reload")
        if pending_types is not None:
            _state["_pending_ptype_reload"] = None
            print(f"Reloading particle types: {pending_types}")
            prev_types = list(data.particle_types)
            try:
                data.set_particle_types(pending_types)
            except Exception as e:
                # A type with missing/unreadable fields must not kill the
                # session — restore the previous selection and surface the
                # error in the HUD instead.
                _last_message = f"Type load failed: {e}"
                print(f"  {_last_message}")
                toasts.show(_last_message, "error", duration=5.0)
                try:
                    data.set_particle_types(prev_types)
                except Exception:
                    data.set_particle_types([])

            # Refresh available scalar/vector field lists (intersection
            # across selected types).
            _sd_fields = data.available_fields_with_derived()
            _vector_fields = data.available_vector_fields()
            _state["_vector_fields"] = _vector_fields
            if _state["_sd_field"] not in _sd_fields:
                _state["_sd_field"] = "Masses"
            if _state["_sd_field2"] not in (["None"] + _sd_fields):
                _state["_sd_field2"] = "None"
            if _state["_wa_data_field"] not in _sd_fields:
                _state["_wa_data_field"] = "Masses"
            for sl in _state["_slot"]:
                if sl["weight"] not in _sd_fields:
                    sl["weight"] = "Masses"
                if sl.get("weight2", "None") not in (["None"] + _sd_fields):
                    sl["weight2"] = "None"
                if sl["data"] not in _sd_fields:
                    sl["data"] = "Masses"

            # Re-upload particles to the renderer and force GPU compute
            # to re-init on the next frame so the subsample pipeline
            # picks up the new pool.
            weights = data.get_field("Masses")
            renderer.set_particles(data.positions, data.hsml, weights)
            try:
                renderer.set_subsample_chunks(None)
            except Exception:
                pass
            if gpu_compute is not None:
                try:
                    gpu_compute.release()
                except Exception:
                    pass
            gpu_compute = None
            _slot_sorted[0] = None
            _slot_sorted[1] = None
            needs_auto_range = True
            dirty = True

        # GPU subsample pipeline init (one-time upload, first frame).
        gpu_ready = getattr(gpu_compute, "_upload_ready", False)
        if not gpu_ready and gpu_compute is None:
            try:
                gpu_compute = GPUCompute(device)
                gpu_compute.upload_subsample_only(
                    renderer._all_pos, renderer._all_hsml, renderer._all_mass, renderer._all_qty
                )
                renderer.set_subsample_chunks(gpu_compute.get_chunk_bufs(), world_offset=gpu_compute.get_pos_offset())
                # Prime renderer.n_particles so the first render() call
                # doesn't early-out before the user has moved.
                renderer.n_particles = min(renderer.n_total, renderer._subsample_max_per_frame)
                print("  GPU subsample pipeline initialized")
                # Apply the --field/--mode startup view now that the
                # subsample buffers exist to receive its weights.
                if _startup_view_pending:
                    _startup_view_pending = False
                    try:
                        app_proxy._apply_render_mode(auto_range=False)
                    except Exception as e:
                        print(f"  Startup view failed: {e}")
                # Defer the post-init auto-range a few frames so the
                # GPU has actually rendered something into the FBO
                # before we read it back. The pre-init auto-range that
                # fires from set_particles ran against an empty FBO
                # so its qty range is garbage.
                pending_auto_range_frames = 3
                needs_auto_range = False
                gpu_ready = True
            except Exception as e:
                print(f"  GPU compute init failed: {e}")
                gpu_compute = None

        # Camera movement
        moved = camera.update(dt)
        # Per-particle LOS depends on camera position only, not on
        # orientation, so detect translation separately from rotation.
        translated = bool(np.any(camera.position != prev_camera_pos))
        prev_camera_pos = camera.position.copy()

        # Subsample per-frame instance cap. Init to 4M (used only at the
        # very first frame); after that the cap learns the highest value
        # that sustained the target FPS during motion ("last sustainable")
        # and resets to that on motion start. Stopped frames grow the cap
        # for progressive refinement, but only after the user has moved
        # at least once and only when the previous frame met its target
        # (so we never over-extend with no FPS history to validate).
        target_ms = 1000.0 / max(renderer.target_fps, 1.0)
        # Time-based proportional controller. Framerate-independent: all
        # gains are in seconds, so behavior is identical at 15/30/60 fps
        # targets — only the granularity of corrections changes.
        #
        #   tau_smooth_s   EMA time constant for render_ms (s)
        #   tau_resp_s     controller response time constant (s)
        #   max_rate_log2  hard ceiling on cap change rate (log2/s)
        #
        # Tuned against tests/bench_orbit_pid.py on a 134 M-particle
        # snapshot under big workload swings; cuts ms std ~50% vs the
        # old bang-bang controller at every target FPS.
        TAU_SMOOTH_S = 0.05
        TAU_RESP_S = 0.10
        MAX_RATE_LOG2 = 5.0
        if last_render_ms > 0 and dt > 0:
            alpha = 1.0 - np.exp(-dt / TAU_SMOOTH_S)
            smooth_render_ms = (
                smooth_render_ms + alpha * (last_render_ms - smooth_render_ms)
                if smooth_render_ms > 0
                else last_render_ms
            )
        if moved:
            has_moved_ever = True
            if not was_moving:
                # Resume from the last cap proven sustainable for
                # interactive motion. The on_submitted_work_done_sync
                # call after each stationary frame guarantees no
                # refinement frame is in flight, so this cap takes
                # effect on the very next render with zero drain wait.
                renderer.set_subsample_max_per_frame(last_subsample_cap)
                # Reset the smoothed render-time EMA so the just-elapsed
                # multi-second stationary frame doesn't poison the cap
                # update on the first few motion frames.
                smooth_render_ms = 0.0
            if smooth_render_ms > 0 and dt > 0:
                cur_cap = renderer._subsample_max_per_frame
                err = (smooth_render_ms - target_ms) / target_ms
                log_step = -err * dt / TAU_RESP_S
                max_step = MAX_RATE_LOG2 * dt
                if log_step > max_step:
                    log_step = max_step
                elif log_step < -max_step:
                    log_step = -max_step
                new_cap = int(cur_cap * (2.0**log_step))
                new_cap = max(1, min(subsample_cap_ceiling, new_cap))
                last_subsample_cap = new_cap
                renderer.set_subsample_max_per_frame(new_cap)
        elif has_moved_ever:
            # Stopped after at least one motion: refine progressively.
            # The growth gate uses last_render_ms (wall-clock duration of
            # the previous renderer.render() call), which captures real
            # GPU latency via get_current_texture's swapchain block.
            # Python's `dt` is a useless signal here — Metal queues
            # encode/submit asynchronously and Python keeps racing
            # ahead, so dt stays small even when the GPU is buried.
            # Stationary refinement is allowed to take long-ish frames —
            # the user is sitting still waiting for full detail. The cap
            # only stops growing if a frame takes more than ~750 ms,
            # which is the threshold past which input feels "stuck".
            REFINE_FRAME_MS = 750.0
            if 0 < last_render_ms <= REFINE_FRAME_MS:
                # Stationary refinement target: actual full detail. The
                # user-controlled `subsample_cap_ceiling` only governs
                # the cap during motion (where high FPS matters); when
                # sitting still we're allowed to push up to the hard
                # GPU limit and the snapshot's particle count.
                stationary_cap = min(SUBSAMPLE_CAP_HARD_LIMIT, renderer.n_total)
                cur = renderer._subsample_max_per_frame
                if cur < stationary_cap:
                    renderer.set_subsample_max_per_frame(min(int(cur * 1.4) + 1, stationary_cap))
                # Promote the refined cap to last_subsample_cap only if
                # the frame we just rendered was above target FPS. That
                # way `last_subsample_cap` always reflects "the largest
                # cap proven sustainable for interactive motion", and
                # resuming motion restores that quality without ever
                # putting the user above their target frame budget.
                if 0 < last_render_ms <= target_ms and cur > last_subsample_cap:
                    last_subsample_cap = min(cur, subsample_cap_ceiling)

        # Per-particle LOS depends only on camera position, so the
        # cached projection is invalid only when the camera has
        # *translated* (rotation is free). Skip the recompute while
        # translation is in progress — it's too expensive (full CPU LOS
        # pass over every particle) to run mid-flight on large
        # snapshots; once the user stops translating, the gate opens
        # and the next frame refreshes once.
        if not translated:
            if is_los_stale(
                _state["_render_mode_name"],
                _state["_wa_data_field"],
                _state["_sd_field"],
                _state.get("_sd_field2", "None"),
                _vector_fields,
                _state["_vector_projection"],
                _state.get("_los_camera_pos"),
                camera.position,
            ):
                app_proxy._apply_render_mode(auto_range=True)
                dirty = True

        # Keep n_particles in sync with the live cap so the overlay /
        # dirty signature reflect what's actually being drawn.
        if renderer._subsample_chunks is not None:
            renderer.n_particles = min(renderer.n_total, renderer._subsample_max_per_frame)

        was_moving = moved

        # Resize handling
        new_fb_w, new_fb_h = glfw.get_framebuffer_size(window)
        if (new_fb_w, new_fb_h) != (fb_w, fb_h):
            fb_w, fb_h = new_fb_w, new_fb_h
            canvas_context.set_physical_size(fb_w, fb_h)
            camera.aspect = fb_w / max(fb_h, 1)
            win_w, _ = glfw.get_window_size(window)
            renderer._viewport_width = win_w  # LOD uses window size, not retina

        # --- Idle-frame short-circuit ---
        # Build a signature of all state that affects the rendered image.
        # If nothing has changed since the last presented frame and we
        # aren't carrying any pending work, skip the entire render path
        # (no swapchain acquire, no encode, no submit) and sleep briefly
        # so the loop doesn't burn CPU.
        try:
            _slot0 = _state["_slot"][0]
            _slot1 = _state["_slot"][1]
            state_sig = (
                _state.get("_composite"),
                _slot0.get("min"),
                _slot0.get("max"),
                _slot0.get("log"),
                _slot0.get("resolve"),
                _slot1.get("min"),
                _slot1.get("max"),
                _slot1.get("log"),
                _slot1.get("resolve"),
                _state.get("_render_mode_name"),
                _state.get("_cmap_idx"),
                _state.get("_wa_data_field"),
                _state.get("_sd_field"),
                _state.get("_sd_field2"),
                _state.get("_sd_op"),
                _state.get("_vector_projection"),
                renderer.qty_min,
                renderer.qty_max,
                renderer.log_scale,
                renderer.hsml_scale,
                ui_hidden,
            )
        except Exception:
            state_sig = None  # be conservative: render
        cap_now = renderer._subsample_max_per_frame
        n_particles_now = renderer.n_particles
        fb_size_now = (fb_w, fb_h)

        scene_dirty = (
            dirty
            or moved
            or needs_auto_range
            or pending_auto_range_frames > 0
            or state_sig is None
            or state_sig != prev_state_sig
            or cap_now != prev_cap
            or n_particles_now != prev_n_particles
            or fb_size_now != prev_fb_size
        )
        # ui_dirty alone is enough to require a frame, but lets us skip
        # the (very expensive on big snapshots) accum pass. Active
        # toasts animate (fade-out), so they hold the UI path open.
        frame_dirty = scene_dirty or ui_dirty or toasts.active
        skip_accum_this_frame = (ui_dirty or toasts.active) and not scene_dirty

        if not frame_dirty:
            # Require a few consecutive idle frames before we actually
            # start sleeping. A single `moved=False` tick in the middle
            # of a slow rotation (e.g. between mouse-delta events) would
            # otherwise insert a 5ms sleep and cause visible stutter,
            # most noticeably in LOS variance / composite modes whose
            # CPU-cached projection already updates coarsely.
            idle_streak += 1
            if idle_streak >= IDLE_STREAK_THRESHOLD:
                # Don't increment frame_count for slept frames so the
                # FPS counter isn't inflated by no-op iterations.
                frame_count = max(frame_count - 1, 0)
                time.sleep(0.005)
                continue
            # Otherwise fall through and render anyway (cheap, since
            # nothing changed the GPU work is essentially the same as
            # the last frame).
        else:
            idle_streak = 0

        # Render. Time the call wall-clock — most of this is
        # get_current_texture blocking on the swapchain present, which
        # is the only signal that reflects real GPU latency (Python's
        # encode + submit are async w.r.t. the GPU).
        _t_render = time.perf_counter()

        # Single encoder + single submit per frame: acquire the swapchain
        # texture once, append accum/resolve and the overlay/UI passes to
        # the same command encoder, submit at the end. This halves the
        # per-frame wgpu FFI roundtrips vs. the old "render submits, then
        # overlay submits" path.
        _frame_encoder = None
        _frame_screen_view = None
        try:
            current_tex = canvas_context.get_current_texture()
            _frame_screen_view = current_tex.create_view()
            _frame_encoder = device.create_command_encoder()
        except Exception:
            import traceback

            traceback.print_exc()

        try:
            if _frame_encoder is not None:
                if _state["_composite"]:
                    _render_composite_frame(
                        fb_w,
                        fb_h,
                        encoder=_frame_encoder,
                        screen_view=_frame_screen_view,
                        skip_los_recompute=translated,
                        skip_accum=skip_accum_this_frame,
                    )
                else:
                    renderer.render(
                        camera,
                        fb_w,
                        fb_h,
                        encoder=_frame_encoder,
                        screen_view=_frame_screen_view,
                        skip_accum=skip_accum_this_frame,
                    )
        except Exception as e:
            print(f"Render error: {e}")
            import traceback

            traceback.print_exc()
            # Ensure we still present something (avoid black flash). The
            # legacy fallback path uses its own encoder/submit.
            try:
                renderer.render(camera, fb_w, fb_h)
            except Exception:
                pass
        last_render_ms = (time.perf_counter() - _t_render) * 1000.0
        # Auto-range on first frame
        if pending_auto_range_frames > 0:
            pending_auto_range_frames -= 1
            if pending_auto_range_frames == 0:
                needs_auto_range = True
        if needs_auto_range and not _state["_composite"]:
            lo, hi = renderer.read_accum_range()
            renderer.qty_min = lo
            renderer.qty_max = hi
            print(f"Auto-range: {lo:.3g} .. {hi:.3g}")
            # The pre-built frame encoder already encoded a resolve with
            # the *stale* qty range. Drop it so we don't submit stale
            # pixels on top of the freshly-auto-ranged image. The legacy
            # render() call below uses its own encoder and submits
            # immediately; the overlay block will fall back to its own
            # encoder since _frame_encoder is now None.
            _frame_encoder = None
            _frame_screen_view = None
            try:
                renderer.render(camera, fb_w, fb_h)
            except Exception:
                pass
            needs_auto_range = False
            # Headless screenshot mode: take the shot now that init,
            # auto-range, and one full render have completed, then exit.
            if screenshot is not None:
                _take_screenshot(screenshot)
                glfw.set_window_should_close(window, True)

        # Update timing stats
        cull_s = getattr(renderer, "_last_cull_ms", 0) / 1000
        upload_s = getattr(renderer, "_last_upload_ms", 0) / 1000
        render_s = getattr(renderer, "_last_render_ms", 0) / 1000
        alpha = 0.2
        if cull_s > 0 or upload_s > 0 or render_s > 0:
            _timings["cull"] = _timings["cull"] * (1 - alpha) + cull_s * alpha
            _timings["upload"] = _timings["upload"] * (1 - alpha) + upload_s * alpha
            _timings["render"] = _timings["render"] * (1 - alpha) + render_s * alpha

        # Sync auto-range request from proxy
        if _state["_needs_auto_range"]:
            needs_auto_range = True
            _state["_needs_auto_range"] = False

        # Overlay rendering pass (on top of the resolved image)
        # Skip overlay + present on odd frames when skip_vsync is on
        _do_present = not renderer.skip_vsync or frame_count % 2 == 0
        # If the auto-range branch dropped the prebuilt frame encoder,
        # we need a fresh one (with its own swapchain view) just for the
        # overlay pass.
        if not ui_hidden and _do_present and _frame_encoder is None:
            try:
                current_tex = canvas_context.get_current_texture()
                _frame_screen_view = current_tex.create_view()
                _frame_encoder = device.create_command_encoder()
            except Exception:
                import traceback

                traceback.print_exc()
        if not ui_hidden and _do_present and _frame_encoder is not None:
            try:
                screen_view = _frame_screen_view

                overlay.set_framebuffer_size(fb_w, fb_h)
                sink_panel.set_framebuffer_size(fb_w, fb_h)
                user_menu.set_framebuffer_size(fb_w, fb_h)
                help_panel.set_framebuffer_size(fb_w, fb_h)
                toolbar.set_framebuffer_size(fb_w, fb_h)
                for p in (scale_bar, status_bar, toasts, gizmo, drawer):
                    p.set_framebuffer_size(fb_w, fb_h)
                toolbar.update(
                    recording=_recording["dir"] is not None,
                    orbiting=camera.orbit is not None,
                    drawer_mode=drawer.mode if drawer.enabled else None,
                    aperture=_aperture["active"] or _aperture["placing"],
                )

                # Science chrome (each panel dirty-checks internally)
                view_center = data.get_view_center()
                scale_bar.update(camera, units.length_to_kpc, view_center)
                active_field = (
                    _state["_wa_data_field"]
                    if _state["_render_mode_name"] in ("WeightedAverage", "WeightedVariance")
                    else _state["_sd_field"]
                )
                status_bar.update(camera, units, view_center, active_field,
                                  renderer.n_particles, renderer.n_total)
                gizmo.update(camera)
                toasts.update()
                if drawer.enabled:
                    drawer.update(data)

                # Sightline overlay (projected segments).
                if sightline_overlay.enabled and _sightlines["list"]:
                    sightline_overlay.set_framebuffer_size(fb_w, fb_h)
                    segs = []
                    for sl in _sightlines["list"]:
                        pa = _world_to_screen(sl.start, fb_w, fb_h)
                        pb = _world_to_screen(sl.end, fb_w, fb_h)
                        if pa is not None and pb is not None:
                            segs.append((pa[0], pa[1], pb[0], pb[1],
                                         sl.label, sl.color))
                    sightline_overlay.update(segs)

                # Trident completion polling (sentinel files).
                if _sightlines["trident_procs"]:
                    done_now = []
                    for sl_t, proc in _sightlines["trident_procs"]:
                        sent = sl_t.extra.get("trident_done_sentinel")
                        if sent and os.path.exists(sent):
                            done_now.append((sl_t, proc))
                            toasts.show(
                                f"Trident done: {sl_t.label} -> "
                                f"{os.path.basename(sl_t.trident_spectrum_path)}",
                                "ok", duration=8.0)
                        elif proc.poll() is not None and not (
                                sent and os.path.exists(sent)):
                            done_now.append((sl_t, proc))
                            toasts.show(
                                f"Trident failed for {sl_t.label} "
                                f"(exit {proc.returncode})", "error",
                                duration=8.0)
                    for item in done_now:
                        _sightlines["trident_procs"].remove(item)

                # Aperture sphere indicator (projected circle).
                _aperture["visible"] = False
                if aperture_panel.enabled and _aperture["center"] is not None:
                    aperture_panel.set_framebuffer_size(fb_w, fb_h)
                    res = _world_to_screen(_aperture["center"], fb_w, fb_h)
                    if res is not None:
                        cx_px, cy_px, zf = res
                        tan_half = np.tan(np.radians(camera.fov) / 2.0)
                        r_px = (_aperture["radius"] / (zf * tan_half)
                                * fb_h / 2.0)
                        r_kpc = _aperture["radius"] * units.length_to_kpc
                        shp, ring = _APERTURE_SHAPES[_aperture["shape_idx"]]
                        lbl = f"{shp}  R = {r_kpc:,.3g} kpc"
                        if _aperture["placing"]:
                            lbl += "  (M=set, Tab=shape)"
                        aperture_panel.style.accent_color = ring
                        aperture_panel.update(cx_px, cy_px, r_px, lbl,
                                              placing=_aperture["placing"])
                        _aperture["visible"] = True

                smooth_fps_val = smooth_fps_ema if smooth_fps_ema > 0 else fps
                # Only rebuild overlay texture at ~4Hz to avoid PIL cost every frame
                _overlay_age = getattr(overlay, "_last_update_time", 0)
                if now - _overlay_age > 0.25:
                    overlay._last_update_time = now
                    was_enabled = overlay.enabled
                    overlay.enabled = True
                    init_status = "Initializing: uploading to GPU..." if gpu_compute is None else ""
                    overlay_message = init_status if init_status else _last_message
                    overlay.update(
                        renderer,
                        camera,
                        fps,
                        _render_mode.name,
                        AVAILABLE_COLORMAPS[_state["_cmap_idx"]],
                        _timings,
                        overlay_message,
                        smooth_fps=smooth_fps_val,
                    )
                    overlay.enabled = was_enabled
                if sink_panel.enabled:
                    sink_panel.update(renderer)
                if help_panel.enabled:
                    help_panel.update()
                user_menu.update(
                    renderer,
                    AVAILABLE_COLORMAPS[_state["_cmap_idx"]],
                    AVAILABLE_COLORMAPS,
                    sd_fields=_sd_fields,
                    sd_field=_state["_sd_field"],
                    sd_field2=_state["_sd_field2"],
                    sd_op=_state["_sd_op"],
                    sd_ops=_SD_OPS,
                    render_modes=_RENDER_MODES,
                    render_mode_name=_state["_render_mode_name"],
                    wa_data_field=_state["_wa_data_field"],
                    vector_fields=_vector_fields,
                    vector_projection=_state["_vector_projection"],
                    vector_projections=_VECTOR_PROJECTIONS,
                    composite_slots=_state["_slot"] if _state["_composite"] else None,
                    available_ptypes=data.available_types,
                    selected_ptypes=list(data.particle_types),
                    ptype_labels=data.ptype_labels,
                    fov=camera.fov,
                    cam_speed=camera.speed,
                )

                rpass = _frame_encoder.begin_render_pass(
                    color_attachments=[
                        {
                            "view": screen_view,
                            "load_op": "load",
                            "store_op": "store",
                        }
                    ]
                )
                user_menu.render_to_pass(rpass)
                toolbar.render_to_pass(rpass)
                if scale_bar.enabled:
                    scale_bar.render_to_pass(rpass)
                if status_bar.enabled:
                    status_bar.render_to_pass(rpass)
                if gizmo.enabled:
                    gizmo.render_to_pass(rpass)
                if aperture_panel.enabled and _aperture.get("visible"):
                    aperture_panel.render_to_pass(rpass)
                if sightline_overlay.enabled and _sightlines["list"]:
                    sightline_overlay.render_to_pass(rpass)
                if sink_panel.enabled:
                    sink_panel.render_to_pass(rpass)
                if overlay.enabled:
                    overlay.render_to_pass(rpass)
                if drawer.enabled:
                    drawer.render_to_pass(rpass)
                if help_panel.enabled:
                    help_panel.render_to_pass(rpass)
                toasts.render_to_pass(rpass)
                rpass.end()
            except Exception:
                import traceback

                traceback.print_exc()

        # Submit the single per-frame encoder (covers accum + resolve +
        # optional overlay/UI). Then present.
        if _frame_encoder is not None:
            try:
                device.queue.submit([_frame_encoder.finish()])
            except Exception:
                import traceback

                traceback.print_exc()

        if _do_present:
            canvas_context.present()

        # Frame recording: capture every presented frame. Keeps the
        # scene marked dirty so the idle short-circuit doesn't freeze
        # the recording while the camera is still.
        if _do_present and _recording["dir"] is not None:
            import os

            fp = os.path.join(_recording["dir"], f"frame_{_recording['frame']:05d}.png")
            try:
                _take_screenshot(fp, quiet=True)
                _recording["frame"] += 1
            except Exception as e:
                print(f"Recording frame failed ({e}); recording stopped")
                _recording["dir"] = None

        # On stationary refinement frames, block here until the GPU has
        # actually retired the work we just submitted. The point: when
        # the user resumes motion, there must be NO in-flight refinement
        # frame queued behind us. If we let refinement keep racing
        # ahead, the next motion-start get_current_texture() blocks for
        # hundreds of ms waiting for a stale 100M-particle accum to
        # drain. Synchronizing here pushes that wait into the stationary
        # period (where it doesn't matter — the user is sitting still)
        # and makes resume instantaneous.
        if (not moved) and has_moved_ever and _frame_encoder is not None:
            try:
                device.queue.on_submitted_work_done_sync()
            except Exception:
                pass

        # Frame committed: snapshot state for the next idle-check.
        prev_state_sig = state_sig
        prev_cap = cap_now
        prev_n_particles = n_particles_now
        prev_fb_size = fb_size_now
        # While recording, every frame must render+capture even if the
        # camera is still, so the movie has constant pacing.
        dirty = _recording["dir"] is not None
        ui_dirty = False

    # Persist user-tunable settings for the next session.
    try:
        _settings.update(
            {
                "colormap": AVAILABLE_COLORMAPS[_state["_cmap_idx"]],
                "invert_mouse": bool(camera.invert_mouse),
                "target_fps": float(renderer.target_fps),
            }
        )
        save_settings(_settings)
    except Exception as e:
        print(f"  Settings save skipped: {e}")

    # Cleanup
    renderer.release()
    data.close()
    glfw.destroy_window(window)
