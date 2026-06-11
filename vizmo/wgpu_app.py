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
    series=False,
    split_snapshot=None,
    catalog=None,
    show_welcome=False,
):
    """Run the vizmo application with the wgpu backend.

    If `screenshot` is set to a path, the canvas loop runs just long
    enough for GPU init + auto-range to complete, takes a screenshot,
    and exits without entering the interactive loop.
    """
    import os

    snapshot_path = os.path.abspath(snapshot_path)
    # --series: the positional path is a directory; discover the time
    # series and start from the first (highest-z) snapshot.
    _series = {"snaps": [], "index": 0, "pending": None}
    # In-place snapshot swap queue: any code (series arrows, File >
    # Open, Open Recent) may set a path here; the main loop performs
    # the reload exactly as the series switcher does.
    _pending_load = {"path": None, "label": ""}
    _split_snapshot_path = split_snapshot
    _catalog_path = catalog
    if series:
        from .series import discover_snapshots

        _series["snaps"] = discover_snapshots(snapshot_path)
        _series["player_pending"] = True
        if not _series["snaps"]:
            raise FileNotFoundError(
                f"--series: no snapshots found under {snapshot_path}")
        snapshot_path = _series["snaps"][0].path
        print(f"  Series: {len(_series['snaps'])} snapshots, starting "
              f"z={_series['snaps'][0].redshift:.2f}")
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

    # Async startup (Section 8.A): the snapshot loads in a background
    # thread while a spinner + progress-bar loading screen renders, so
    # the window is alive and responsive from the first frame. The
    # render loop proper never starts until the data exists, so no
    # downstream code ever sees a half-loaded object.
    import threading as _threading

    from .wgpu_overlay import WGPULoadingOverlay

    _load = {"data": None, "error": None,
             "progress": {"status": "opening file...", "progress": 0.0}}

    def _load_worker():
        import time as _t

        t0 = _t.time()
        try:
            _load["data"] = SnapshotData(
                snapshot_path, particle_types=types,
                hsml_progress=None, progress=_load["progress"])
            _load["dt"] = _t.time() - t0
        except Exception as e:
            _load["error"] = e

    _threading.Thread(target=_load_worker, daemon=True).start()
    _loading_panel = WGPULoadingOverlay(device, present_format)
    _lframe = 0
    while _load["data"] is None and _load["error"] is None:
        glfw.poll_events()
        if glfw.window_should_close(window):
            raise SystemExit(0)
        fbw_l, fbh_l = glfw.get_framebuffer_size(window)
        try:
            _loading_panel.set_framebuffer_size(fbw_l, fbh_l)
            _loading_panel.update(
                os.path.basename(snapshot_path), _lframe,
                _load["progress"].get("progress", 0.0),
                _load["progress"].get("status", ""))
            tex = canvas_context.get_current_texture()
            enc = device.create_command_encoder()
            rp = enc.begin_render_pass(color_attachments=[{
                "view": tex.create_view(),
                "clear_value": (0, 0, 0, 1),
                "load_op": "clear", "store_op": "store"}])
            _loading_panel.render_to_pass(rp)
            rp.end()
            device.queue.submit([enc.finish()])
            canvas_context.present()
        except Exception:
            pass
        _lframe += 1
        time.sleep(1.0 / 30.0)
    if _load["error"] is not None:
        raise _load["error"]
    data = _load["data"]
    data._hsml_progress = _hsml_progress
    print(f"  {data.n_particles:,} particles loaded "
          f"(types {data.particle_types}) in {_load.get('dt', 0):.1f}s")
    _load_done_toast = (f"Loaded {data.n_particles / 1e6:.1f}M particles "
                        f"in {_load.get('dt', 0):.1f}s")
    data2 = None
    if _split_snapshot_path:
        # --split: second dataset loaded for side-by-side comparison.
        # Rendering its particles needs a second GPU pipeline (not yet
        # implemented); the right pane shows this snapshot's slot-1
        # field meanwhile.
        data2 = SnapshotData(os.path.abspath(_split_snapshot_path),
                             particle_types=types)
        print(f"  [split] second snapshot loaded: "
              f"{data2.n_particles:,} particles")

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
    renderer.init_gpu_timing()
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

    import tracemalloc

    tracemalloc.start()
    from .wgpu_overlay import WGPUProfilerOverlay

    profiler_panel = WGPUProfilerOverlay(device, present_format)
    from .overlay import build_default_menus
    from .recentfiles import RecentFiles
    from .wgpu_overlay import (WGPUMenuBar, WGPUColormapBrowser,
                               WGPUFieldPicker)

    _recents = RecentFiles()
    _recents.add(snapshot_path)
    # (toasts exists below; fire the load toast after panel creation)
    menubar = WGPUMenuBar(device, present_format,
                          build_default_menus(_recents.get()))
    from .overlay import MenuItem as _MI

    _menus = {"File": menubar.menus["File"]}
    _menus["Edit"] = [_MI("Undo", "Ctrl+Z", "undo"),
                      _MI("Redo", "Ctrl+Shift+Z", "redo")]
    for k in ("View", "Analysis", "Export", "Help"):
        _menus[k] = menubar.menus[k]
    menubar.menus = _menus
    from .wgpu_overlay import WGPURightDock

    from .wgpu_overlay import WGPUWelcomeOverlay

    welcome = WGPUWelcomeOverlay(device, present_format)
    welcome.enabled = bool(show_welcome)
    from .undo import UndoStack
    from .wgpu_overlay import (WGPUTimelineScrubber, WGPUMovieRecorder,
                               WGPUAboutPanel, WGPUShortcutsPanel)

    timeline = WGPUTimelineScrubber(device, present_format)
    movie_panel = WGPUMovieRecorder(device, present_format)
    about_panel = WGPUAboutPanel(device, present_format)
    shortcuts_panel = WGPUShortcutsPanel(device, present_format)
    undo = UndoStack()

    def _undo_state():
        return (dict(_aperture), list(data.filters),
                _state["_render_mode_name"])

    def _undo_push(action):
        undo.push(action, *_undo_state())

    def _undo_restore(entry):
        ap = entry["prev_aperture"]
        for k in ("active", "placing", "center", "radius",
                  "center_mode", "shape_idx"):
            if k in ap:
                _aperture[k] = ap[k]
        aperture_panel.enabled = bool(_aperture.get("center") is not None
                                      and (_aperture["active"]
                                           or _aperture["placing"]))
        if _aperture["active"] and _aperture["center"] is not None:
            try:
                _submit_aperture()
            except Exception:
                pass
        else:
            drawer.clear_scope()
        data.set_filters(entry["prev_filters"])
        _state["_render_mode_name"] = entry["prev_render_mode"]
        try:
            app_proxy._apply_render_mode(auto_range=False)
        except Exception:
            pass
        drawer.refresh()

    right_dock = WGPURightDock(device, present_format)
    cmap_browser = WGPUColormapBrowser(device, present_format)
    field_picker = WGPUFieldPicker(device, present_format)
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
    toasts.show(_load_done_toast, "ok")
    if _series.get("player_pending"):
        from .series import TimelinePlayer

        _series["player"] = TimelinePlayer(_series["snaps"])
        timeline.player = _series["player"]
        timeline.enabled = True
        movie_panel.series_available = True

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
    from .wgpu_renderer import HaloMarkerRenderer

    halo_markers = HaloMarkerRenderer(device, present_format)
    _halo_sel = {"id": None}

    def _refresh_halo_markers():
        from .catalog import mass_threshold_mask

        if drawer.catalog is None:
            return
        halo_markers.set_halos(
            drawer.catalog, units.length_to_kpc,
            mask=mass_threshold_mask(drawer.catalog,
                                     drawer._halo_threshold),
            selected_id=_halo_sel["id"])

    if _catalog_path:
        try:
            from .catalog import load_catalog

            drawer.catalog = load_catalog(_catalog_path, units)
            print(f"  Catalog: {len(drawer.catalog['halo_id'])} halos")
            _refresh_halo_markers()
        except Exception as e:
            print(f"  Catalog load failed: {e}")

    # Drag-and-drop: dropping an HDF5 file loads it in place.
    def _drop_callback(win, paths):
        for p_ in paths:
            if str(p_).endswith((".hdf5", ".h5")):
                _pending_load["path"] = str(p_)
                _pending_load["label"] = os.path.basename(str(p_))
                return

    try:
        glfw.set_drop_callback(window, _drop_callback)
    except Exception:
        pass

    # Slice plane (Section 5.A). Shift+Z activates / cycles the normal;
    # Ctrl+drag translates the plane along its normal; the drawer's
    # "slice" tool exposes axis/offset/opacity/resolution controls.
    from .wgpu_renderer import SlicePlaneRenderer, plane_basis

    slice_renderer = SlicePlaneRenderer(device, present_format)
    _SLICE_NORMALS = [("camera-right", None), ("+X", (1.0, 0, 0)),
                      ("+Y", (0, 1.0, 0)), ("+Z", (0, 0, 1.0))]
    _slice = {
        "active": False, "normal_idx": 3, "offset": 0.0,  # code units
        "res": 256, "opacity": 0.85, "dirty": False,
        "normal_label": "+Z", "used_gpu": False,
        "size_kpc": 0.0, "offset_kpc": 0.0,
    }
    drawer.slice_state = _slice

    from .wgpu_renderer import (IsosurfaceRenderer, ISO_COLORS,
                                voxelize_particles, extract_isosurface,
                                mesh_to_obj, RenderState, SPLIT_MODES)

    iso_renderer = IsosurfaceRenderer(device, present_format)
    _iso = {"surfaces": [], "res": 128, "field": "Masses",
            "busy": False, "grid": None, "half": None, "center": None}
    drawer.iso_state = _iso

    def _iso_recompute(add_new=False):
        import threading

        if _iso["busy"]:
            toasts.show("Isosurface busy...", "warn")
            return
        center = _focus_center()
        d_cam = float(np.linalg.norm(camera.position - center))
        half = max(d_cam * 0.6, 1e-6)
        field = (_state["_wa_data_field"]
                 if _state["_render_mode_name"] in
                 ("WeightedAverage", "WeightedVariance")
                 else _state["_sd_field"])
        _iso["field"] = field
        vals = (None if field == "Masses"
                else np.asarray(data.get_field(field), dtype=np.float64)
                * data.masses)
        pos = data.positions
        masses = data.masses
        res = _iso["res"]
        _iso["busy"] = True

        def work():
            try:
                grid_m = voxelize_particles(pos, masses, center,
                                            half, res)
                if vals is None:
                    grid = grid_m
                else:
                    grid_f = voxelize_particles(pos, vals, center,
                                                half, res)
                    with np.errstate(invalid="ignore",
                                     divide="ignore"):
                        grid = np.where(grid_m > 0, grid_f / grid_m, 0.0)
                _iso["grid"] = grid
                _iso["half"] = half
                _iso["center"] = center
                if add_new or not _iso["surfaces"]:
                    pos_g = grid[grid > 0]
                    lev = (float(np.percentile(pos_g, 50))
                           if pos_g.size else 0.5)
                    if len(_iso["surfaces"]) < iso_renderer.MAX_SURFACES:
                        _iso["surfaces"].append(
                            {"level": lev, "opacity": 0.55})
                for i, s in enumerate(_iso["surfaces"]):
                    v, fc, nm = extract_isosurface(grid, s["level"],
                                                   center, half)
                    iso_renderer.set_surface(
                        i, v, fc, nm, ISO_COLORS[i % len(ISO_COLORS)],
                        s["opacity"], s["level"])
                toasts.show(
                    f"Isosurface ready: {len(_iso['surfaces'])} "
                    f"surface(s) at {res}^3", "ok")
            except Exception as e:
                toasts.show(f"Isosurface failed: {e}", "error")
            finally:
                _iso["busy"] = False
                drawer.refresh()

        threading.Thread(target=work, daemon=True).start()
        toasts.show(f"Voxelizing {res}^3 in background...", "info")

    _split = {"mode_idx": 0,
              "left": RenderState(),
              "right": RenderState(colormap="viridis"),
              "snapshot2": None}

    # Streamlines (5.C) + field-line arrows (5.D) + volume (5.E).
    from .wgpu_renderer import (StreamlineRenderer, ArrowRenderer,
                                VolumeRenderer, integrate_streamlines,
                                fibonacci_sphere)

    stream_renderer = StreamlineRenderer(device, present_format)
    arrow_renderer = ArrowRenderer(device, present_format)
    volume_renderer = VolumeRenderer(device, present_format)
    from .regions_panel import RegionSet, MaskRegion

    region_set = RegionSet()
    drawer.region_set = region_set

    orbit_trail_renderer = StreamlineRenderer(device, present_format)
    _orbit_trail = {"visible": False}
    _stream = {"n_seeds": 256, "step_mult": 1.0, "max_steps": 200,
               "field": "Velocities", "color_by": "|v|",
               "surface": False, "visible": False, "busy": False,
               "n_lines": 0}
    _volume = {"res": 128, "step_mult": 1.0, "mip": False,
               "field": "Masses", "used_gpu": False, "n_tf": 3}
    drawer.stream_state = _stream
    drawer.volume_state = _volume

    def _stream_compute():
        import threading

        if _stream["busy"]:
            return
        if _stream["field"] not in data.available_vector_fields():
            toasts.show(f"No vector field {_stream['field']}", "warn")
            return
        center = _focus_center()
        d_cam = float(np.linalg.norm(camera.position - center))
        radius = (drawer.scope["radius_kpc"] / units.length_to_kpc
                  if (drawer.scope and drawer.use_scope)
                  else d_cam * 0.5)
        pos = data.positions
        vec = np.asarray(data.get_vector_field(_stream["field"]),
                         dtype=np.float64)
        h = data.hsml
        step = float(np.median(h)) * 0.5 * _stream["step_mult"]
        n = _stream["n_seeds"]
        if _stream["surface"]:
            seeds = fibonacci_sphere(n, center, radius)
        else:
            rng = np.random.default_rng(0)
            d3 = rng.standard_normal((n, 3))
            d3 /= np.linalg.norm(d3, axis=1)[:, None]
            seeds = center + radius * rng.random((n, 1)) ** (1 / 3) * d3
        _stream["busy"] = True
        toasts.show(f"Integrating {n} streamlines...", "info")

        def work():
            try:
                # Subsample the interpolation pool for tractable KD
                # queries on 10M+ snapshots.
                np_pool = len(pos)
                if np_pool > 2_000_000:
                    rng2 = np.random.default_rng(1)
                    sub = rng2.choice(np_pool, 2_000_000, replace=False)
                else:
                    sub = slice(None)
                traces = integrate_streamlines(
                    pos[sub], vec[sub], h[sub], None, seeds, step,
                    max_steps=_stream["max_steps"],
                    bounds=(center, radius))
                vals = np.concatenate(
                    [t["values"] for t in traces if len(t["values"])])
                pos_v = vals[vals > 0]
                vmin = (np.log10(np.percentile(pos_v, 5))
                        if pos_v.size else 0.0)
                vmax = (np.log10(np.percentile(pos_v, 99))
                        if pos_v.size else 1.0)
                from .colormaps import colormap_to_texture_data as _ctd

                lut = _ctd(AVAILABLE_COLORMAPS[_state["_cmap_idx"]])
                stream_renderer.set_lines(traces, lut, vmin, vmax)
                _stream["n_lines"] = len(stream_renderer.lines)
                _stream["visible"] = True
                toasts.show(
                    f"Streamlines: {_stream['n_lines']} lines", "ok")
            except Exception as e:
                toasts.show(f"Streamlines failed: {e}", "error")
            finally:
                _stream["busy"] = False
                drawer.refresh()

        threading.Thread(target=work, daemon=True).start()

    def _arrows_compute():
        if "MagneticField" not in data.available_vector_fields():
            toasts.show("No MagneticField in selection", "warn")
            return
        center = _focus_center()
        d_cam = float(np.linalg.norm(camera.position - center))
        radius = d_cam * 0.5
        rel = np.linalg.norm(data.positions - center[None, :], axis=1)
        keep = np.flatnonzero(rel < radius)
        if keep.size == 0:
            toasts.show("No particles in range", "warn")
            return
        b = np.asarray(data.get_vector_field("MagneticField"),
                       dtype=np.float64)[keep]
        bmag = np.linalg.norm(b, axis=1)
        arrow_len = radius * 0.04
        with np.errstate(invalid="ignore", divide="ignore"):
            dirs = np.where(bmag[:, None] > 0, b / bmag[:, None], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lb = np.log10(np.where(bmag > 0, bmag, np.nan))
        fin = lb[np.isfinite(lb)]
        lo, hi = ((np.percentile(fin, 5), np.percentile(fin, 99))
                  if fin.size else (0, 1))
        x = np.clip(np.nan_to_num((lb - lo) / max(hi - lo, 1e-30)),
                    0, 1)
        from .colormaps import colormap_to_texture_data as _ctd

        lut = _ctd(AVAILABLE_COLORMAPS[_state["_cmap_idx"]])
        cols = lut[(x * 255).astype(np.uint8)].astype(np.float32) / 255
        cols[:, 3] = 0.85
        scale = arrow_len * (0.4 + 0.6 * x)
        arrow_renderer.set_arrows(data.positions[keep],
                                  dirs * scale[:, None], cols,
                                  arrow_len)
        toasts.show(
            f"B-field arrows: {arrow_renderer.n_instances}", "ok")

    def _volume_compute():
        import threading

        center = _focus_center()
        d_cam = float(np.linalg.norm(camera.position - center))
        half = d_cam * 0.6
        field = (_state["_wa_data_field"]
                 if _state["_render_mode_name"] in
                 ("WeightedAverage", "WeightedVariance")
                 else _state["_sd_field"])
        _volume["field"] = field
        vals = (None if field == "Masses"
                else np.asarray(data.get_field(field), dtype=np.float64))
        volume_renderer.res = _volume["res"]
        toasts.show(f"Voxelizing {_volume['res']}^3...", "info")

        def work():
            try:
                volume_renderer.voxelize(
                    data.positions, data.masses.astype(np.float64),
                    vals, center, half, _volume["res"])
                _volume["used_gpu"] = volume_renderer.used_gpu_voxelize
                volume_renderer.enabled = True
                toasts.show(
                    f"Volume ready ({'GPU' if _volume['used_gpu'] else 'CPU'}"
                    f" voxelize)", "ok")
            except Exception as e:
                toasts.show(f"Volume failed: {e}", "error")
            finally:
                drawer.refresh()

        threading.Thread(target=work, daemon=True).start()

    def _slice_normal():
        name, vec = _SLICE_NORMALS[_slice["normal_idx"]]
        if vec is None:
            return camera.right.astype(np.float64), name
        return np.asarray(vec, dtype=np.float64), name

    def _recompute_slice():
        if not _slice["active"]:
            return
        normal, name = _slice_normal()
        _slice["normal_label"] = name
        center = _focus_center() + _slice["offset"] * normal
        d_cam = float(np.linalg.norm(camera.position - _focus_center()))
        half = max(d_cam * 0.6, 1e-6)
        field = (_state["_wa_data_field"]
                 if _state["_render_mode_name"] in
                 ("WeightedAverage", "WeightedVariance")
                 else _state["_sd_field"])
        try:
            vals = (data.masses if field == "Masses"
                    else np.asarray(data.get_field(field)))
            slice_renderer.compute(data.positions, data.hsml, vals,
                                   center, normal, half,
                                   res=_slice["res"])
            _slice["used_gpu"] = slice_renderer.used_gpu
            _slice["size_kpc"] = 2 * half * units.length_to_kpc
            _slice["offset_kpc"] = _slice["offset"] * units.length_to_kpc
            g = slice_renderer.grid
            fin = g[np.isfinite(g)]
            if fin.size and renderer.log_scale:
                pos_v = fin[fin > 0]
                vmin = (np.log10(np.percentile(pos_v, 1))
                        if pos_v.size else 0.0)
                vmax = (np.log10(np.percentile(pos_v, 99.9))
                        if pos_v.size else 1.0)
            elif fin.size:
                vmin, vmax = (float(np.percentile(fin, 1)),
                              float(np.percentile(fin, 99.9)))
            else:
                vmin, vmax = 0.0, 1.0
            from .colormaps import colormap_to_texture_data

            lut = colormap_to_texture_data(
                AVAILABLE_COLORMAPS[_state["_cmap_idx"]])
            slice_renderer.upload_colormapped(
                lut, vmin, vmax, renderer.log_scale,
                opacity=_slice["opacity"])
            _slice["_geom"] = (center.copy(), normal.copy(), half)
            drawer.refresh()
        except Exception as e:
            toasts.show(f"Slice failed: {e}", "error")
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

        _undo_push("aperture change")
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
    _mem_stats = None
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
        if (action == glfw.PRESS and key == glfw.KEY_BACKSPACE
                and field_picker.on_backspace()):
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
                if welcome.enabled:
                    welcome.enabled = False
                elif menubar.on_escape():
                    pass
                elif cmap_browser.enabled:
                    cmap_browser.enabled = False
                elif field_picker.enabled:
                    field_picker.enabled = False
                elif _slice["active"]:
                    _slice["active"] = False
                    if drawer.mode == "slice":
                        drawer.enabled = False
                        drawer.mode = None
                    toasts.show("Slice plane off")
                elif _aperture["placing"]:
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
                if mods & glfw.MOD_SHIFT:
                    try:
                        from .export import export_all_data

                        profiles = {}
                        if drawer._last_profile is not None:
                            r, prof, fld, unit = drawer._last_profile
                            profiles[fld] = (r, prof, fld, unit)
                        sc_center, sc_radius = drawer._active_scope()
                        from .analysis import (halo_properties,
                                               region_stats)

                        rows, used_r = region_stats(
                            data, center=sc_center, radius_kpc=sc_radius)
                        halo_rows = halo_properties(
                            data, center=sc_center, radius_kpc=used_r)
                        meta = {"snapshot": snapshot_path,
                                "redshift": float(units.redshift),
                                "radius_kpc": used_r}
                        zp = export_all_data(
                            screenshot_dir or ".", profiles=profiles,
                            stats_rows=rows, halo_rows=halo_rows,
                            phase=drawer._last_phase,
                            sightlines=_sightlines["list"] or None,
                            metadata=meta)
                        toasts.show(
                            f"Bundle: {os.path.basename(zp)}", "ok",
                            duration=6.0)
                    except Exception as e:
                        toasts.show(f"Bundle failed: {e}", "error")
                else:
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
                    was_open = drawer.enabled and drawer.mode == "orbit"
                    drawer.toggle("orbit")
                    if was_open and orbit_trail_renderer.lines:
                        # Closing the panel: O also toggles the trail.
                        _orbit_trail["visible"] = not _orbit_trail["visible"]
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
                if _slice["active"] and slice_renderer.grid is not None:
                    from .wgpu_renderer import slice_grid_to_fits
                    from .physics import field_unit_label

                    geom = _slice.get("_geom")
                    center_kpc = (geom[0] * units.length_to_kpc
                                  if geom else np.zeros(3))
                    field = (_state["_wa_data_field"]
                             if _state["_render_mode_name"] in
                             ("WeightedAverage", "WeightedVariance")
                             else _state["_sd_field"])
                    out = os.path.join(
                        screenshot_dir or ".",
                        f"vizmo_slice_{field}_{int(time.time())}.fits")
                    slice_grid_to_fits(
                        slice_renderer.grid, center_kpc,
                        _slice["size_kpc"], out, field=field,
                        unit=field_unit_label(field),
                        normal_label=_slice["normal_label"])
                    print(f"  Slice FITS: {out}")
                    toasts.show(
                        f"Slice FITS: {os.path.basename(out)}", "ok")
                else:
                    _export_fits_map()
            elif key == glfw.KEY_M:
                if mods & glfw.MOD_SHIFT:
                    # Shift+M: drop the aperture, back to global scope.
                    _undo_push("clear aperture")
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
            elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT) and _series["snaps"]:
                step = -1 if key == glfw.KEY_LEFT else 1
                new_i = _series["index"] + step
                if 0 <= new_i < len(_series["snaps"]):
                    _series["pending"] = new_i
                    if _series.get("player"):
                        _series["player"].index = new_i
                else:
                    toasts.show("End of series", "warn")
            elif key == glfw.KEY_SPACE and _series.get("player"):
                playing = _series["player"].toggle_play()
                toasts.show("Playing" if playing else "Paused")
            elif key == glfw.KEY_HOME and _series.get("player"):
                _series["player"].home()
                _series["pending"] = _series["player"].index
            elif key == glfw.KEY_END and _series.get("player"):
                _series["player"].end()
                _series["pending"] = _series["player"].index
            elif (key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL)
                    and (mods & glfw.MOD_SHIFT)):
                entry = undo.redo(*_undo_state())
                if entry is None:
                    toasts.show("Nothing to redo", "warn")
                else:
                    _undo_restore(entry)
                    toasts.show(f"Redid: {entry['action']}", "ok")
            elif (key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL)
                    and not _slice["active"]):
                entry = undo.undo(*_undo_state())
                if entry is None:
                    toasts.show("Nothing to undo", "warn")
                else:
                    _undo_restore(entry)
                    toasts.show(f"Undid: {entry['action']}", "ok")
            elif key == glfw.KEY_V and (mods & glfw.MOD_CONTROL):
                movie_panel.enabled = not movie_panel.enabled
            elif key == glfw.KEY_I and (mods & glfw.MOD_SHIFT):
                drawer.toggle("isosurface")
                if drawer.mode == "isosurface" and not _iso["surfaces"]:
                    _iso_recompute(add_new=True)
            elif key == glfw.KEY_Z and (mods & glfw.MOD_SHIFT):
                if (drawer.mode == "isosurface" and _iso["surfaces"]
                        and _iso["grid"] is not None):
                    s = _iso["surfaces"][-1]
                    g = _iso["grid"]
                    gmax = float(g.max()) if g.size else 1.0
                    s["level"] = s["level"] * (10 ** 0.5)
                    if s["level"] > gmax:
                        pos_g = g[g > 0]
                        s["level"] = (float(np.percentile(pos_g, 10))
                                      if pos_g.size else gmax / 100)
                    _iso_recompute()
                elif not _slice["active"]:
                    _slice["active"] = True
                    drawer.enabled = True
                    drawer.mode = "slice"
                else:
                    _slice["normal_idx"] = ((_slice["normal_idx"] + 1)
                                            % len(_SLICE_NORMALS))
                _recompute_slice()
                toasts.show(
                    f"Slice plane: normal {_slice['normal_label']} "
                    f"(Shift+Z cycles, Esc closes)")
            elif key == glfw.KEY_V and (mods & glfw.MOD_SHIFT):
                if not stream_renderer.lines:
                    drawer.enabled = True
                    drawer.mode = "streamlines"
                    _stream_compute()
                else:
                    _stream["visible"] = not _stream["visible"]
                    toasts.show(
                        f"Streamlines "
                        f"{'on' if _stream['visible'] else 'off'}")
            elif key == glfw.KEY_B and (mods & glfw.MOD_SHIFT):
                if "MagneticField" not in data.available_vector_fields():
                    toasts.show("MagneticField not loaded", "warn")
                elif arrow_renderer.n_instances == 0:
                    _arrows_compute()
                    _stream["field"] = "MagneticField"
                else:
                    arrow_renderer.n_instances = 0
                    toasts.show("B-field arrows off")
            elif key == glfw.KEY_W and (mods & glfw.MOD_SHIFT):
                if not volume_renderer.enabled:
                    drawer.enabled = True
                    drawer.mode = "volume"
                    _volume_compute()
                else:
                    volume_renderer.mode = 1 - volume_renderer.mode
                    _volume["mip"] = volume_renderer.mode == 1
                    drawer.refresh()
                    toasts.show(
                        "Volume: MIP" if _volume["mip"]
                        else "Volume: emission-absorption")
            elif key == glfw.KEY_S and (mods & glfw.MOD_SHIFT):
                _split["mode_idx"] = (_split["mode_idx"] + 1) % len(SPLIT_MODES)
                mode_now = SPLIT_MODES[_split["mode_idx"]]
                if mode_now is not None:
                    _state["_render_mode_name"] = "Composite"
                    app_proxy._apply_render_mode(auto_range=False)
                    from .colormaps import colormap_to_texture_data as _ctd

                    renderer.set_split_colormap(
                        _ctd(_split["right"].colormap))
                    toasts.show(
                        f"Split screen: {mode_now.upper()} — left=slot0, "
                        f"right=slot1 (edit via Composite controls)")
                else:
                    toasts.show("Split screen off")
            elif key == glfw.KEY_R and (mods & glfw.MOD_CONTROL):
                drawer.toggle("regions")
            elif key == glfw.KEY_O and (mods & glfw.MOD_CONTROL):
                _dispatch_menu_action("open_file")
            elif key == glfw.KEY_Q and (mods & glfw.MOD_CONTROL):
                _dispatch_menu_action("quit")
            elif key == glfw.KEY_H and (mods & glfw.MOD_CONTROL):
                drawer.toggle("halos")
            elif (key == glfw.KEY_D and (mods & glfw.MOD_CONTROL)
                    and (mods & glfw.MOD_SHIFT)):
                right_dock.enabled = not right_dock.enabled
            elif key == glfw.KEY_F10:
                profiler_panel.enabled = not profiler_panel.enabled
                toasts.show(
                    f"GPU profiler "
                    f"{'on' if profiler_panel.enabled else 'off'}")
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
            for fname in (_state["_sd_field"], _state["_wa_data_field"]):
                _ensure_gpu_derived(fname)
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
            if welcome.enabled:
                w_action = welcome.on_click(x, y)
                if w_action:
                    if (isinstance(w_action, tuple)
                            and w_action[0] == "welcome_open"):
                        _pending_load["path"] = w_action[1]
                        _pending_load["label"] = os.path.basename(
                            w_action[1])
                    elif w_action == "open_file":
                        _dispatch_menu_action("open_file")
                    return
            mb_action = menubar.on_click(x, y)
            if mb_action:
                if mb_action is not True:
                    _dispatch_menu_action(mb_action)
                return
            for pnl in (about_panel, shortcuts_panel):
                if pnl.enabled and pnl.on_click(x, y):
                    return
            tl_action = (timeline.on_click(x, y)
                         if timeline.enabled else False)
            if tl_action:
                tp = _series.get("player")
                if tp and tl_action is not True:
                    if tl_action == "tl_play":
                        tp.toggle_play()
                    elif tl_action == "tl_back":
                        tp.step(-1)
                        _series["pending"] = tp.index
                    elif tl_action == "tl_fwd":
                        tp.step(+1)
                        _series["pending"] = tp.index
                    elif tl_action == "tl_first":
                        tp.home()
                        _series["pending"] = tp.index
                    elif tl_action == "tl_last":
                        tp.end()
                        _series["pending"] = tp.index
                    elif tl_action == "tl_fps":
                        cyc = [6.0, 12.0, 24.0, 30.0]
                        tp.fps = (cyc[(cyc.index(tp.fps) + 1) % len(cyc)]
                                  if tp.fps in cyc else 24.0)
                    elif (isinstance(tl_action, tuple)
                            and tl_action[0] == "tl_jump"):
                        tp.index = tl_action[1]
                        _series["pending"] = tp.index
                    timeline._last_key = None
                return
            mv_action = (movie_panel.on_click(x, y)
                         if movie_panel.enabled else False)
            if mv_action:
                if mv_action == "mv_toggle":
                    _movie_toggle()
                elif (isinstance(mv_action, tuple)
                        and mv_action[0] == "mv_res_changed"
                        and mv_action[1] == "4K"):
                    toasts.show("4K recording requires significant "
                                "GPU memory.", "warn", duration=6.0)
                return
            dk_action = right_dock.on_click(x, y)
            if dk_action:
                if isinstance(dk_action, tuple) and dk_action[0] == "dock":
                    drawer.toggle(dk_action[1])
                return
            cb_action = cmap_browser.on_click(x, y)
            if cb_action:
                if (isinstance(cb_action, tuple)
                        and cb_action[0] == "apply_cmap"):
                    try:
                        from .colormaps import (
                            colormap_to_texture_data as _ctd)

                        renderer.set_colormap(_ctd(cb_action[1]))
                        if cb_action[1] in AVAILABLE_COLORMAPS:
                            _state["_cmap_idx"] = AVAILABLE_COLORMAPS.index(
                                cb_action[1])
                        toasts.show(f"Colormap: {cb_action[1]}", "ok")
                    except Exception as e:
                        toasts.show(f"Colormap failed: {e}", "error")
                return
            fp_action = field_picker.on_click(x, y)
            if fp_action:
                if (isinstance(fp_action, tuple)
                        and fp_action[0] == "pick_field"):
                    _state[field_picker.target] = fp_action[1]
                    app_proxy._apply_render_mode()
                    toasts.show(f"Field: {fp_action[1]}", "ok")
                return
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
                if (drawer.catalog is not None
                        and halo_markers.n_vertices > 0):
                    fw_, fh_ = glfw.get_framebuffer_size(win)
                    hid = halo_markers.pick(_world_to_screen, fw_, fh_,
                                            x, y, max_px=12.0)
                    if hid is not None:
                        rows = np.flatnonzero(
                            np.asarray(drawer.catalog["halo_id"]) == hid)
                        if rows.size:
                            drawer._halo_selected = int(rows[0])
                            _halo_sel["id"] = hid
                            _refresh_halo_markers()
                            drawer.enabled = True
                            drawer.mode = "haloinspect"
                            drawer.refresh()
                            toasts.show(f"Halo #{hid}", "ok")
                            return
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
                elif dr_action in ("rg_add", "rg_op", "rg_del",
                                   "rg_submit", "rg_save", "rg_load"):
                    try:
                        if dr_action == "rg_add":
                            if _aperture["center"] is None:
                                toasts.show(
                                    "Place an aperture first (M)", "warn")
                            else:
                                region_set.add(_build_aperture_region())
                                toasts.show(
                                    f"Region {len(region_set.entries)} "
                                    f"added", "ok")
                        elif dr_action == "rg_op" and region_set.entries:
                            region_set.cycle_op(
                                len(region_set.entries) - 1)
                        elif dr_action == "rg_del" and region_set.entries:
                            region_set.remove(len(region_set.entries) - 1)
                        elif dr_action == "rg_submit":
                            if not region_set.entries:
                                toasts.show("No regions to submit", "warn")
                            else:
                                mask_region = MaskRegion(region_set)
                                first = region_set.entries[0]["region"]
                                c0 = getattr(first, "center",
                                             getattr(first, "apex", None))
                                if c0 is None:
                                    c0 = data.get_view_center()
                                r_kpc = (_aperture["radius"]
                                         * units.length_to_kpc
                                         if _aperture["radius"]
                                         else 100.0)
                                drawer.set_scope(np.asarray(c0), r_kpc,
                                                 "regions",
                                                 region=mask_region)
                                n_in = int(mask_region.contains(
                                    data.positions[::100]).sum()) * 100
                                toasts.show(
                                    f"Boolean scope active (~{n_in:,} "
                                    f"particles)", "ok")
                        elif dr_action == "rg_save":
                            out = region_set.save(snapshot_path)
                            toasts.show(
                                f"Regions saved: "
                                f"{os.path.basename(out)}", "ok")
                        elif dr_action == "rg_load":
                            region_set.load(snapshot_path)
                            toasts.show(
                                f"{len(region_set.entries)} regions "
                                f"loaded", "ok")
                    except Exception as e:
                        toasts.show(f"Regions: {e}", "error")
                    drawer.refresh()
                elif dr_action in ("sl_compute", "sl_seeds_down",
                                   "sl_seeds_up", "sl_step", "sl_field",
                                   "sl_surface"):
                    if dr_action == "sl_seeds_down":
                        _stream["n_seeds"] = max(64,
                                                 _stream["n_seeds"] // 2)
                    elif dr_action == "sl_seeds_up":
                        _stream["n_seeds"] = min(1024,
                                                 _stream["n_seeds"] * 2)
                    elif dr_action == "sl_step":
                        cyc = [0.5, 1.0, 2.0]
                        _stream["step_mult"] = cyc[
                            (cyc.index(_stream["step_mult"])
                             if _stream["step_mult"] in cyc else 0 + 1)
                            % len(cyc)]
                    elif dr_action == "sl_field":
                        vfs = data.available_vector_fields()
                        if vfs:
                            i = (vfs.index(_stream["field"]) + 1
                                 if _stream["field"] in vfs else 0)
                            _stream["field"] = vfs[i % len(vfs)]
                    elif dr_action == "sl_surface":
                        _stream["surface"] = not _stream["surface"]
                    if dr_action == "sl_compute":
                        _stream_compute()
                    drawer.refresh()
                elif dr_action in ("vol_compute", "vol_res", "vol_mip",
                                   "vol_step_down", "vol_step_up",
                                   "vol_tf_down", "vol_tf_up"):
                    if dr_action == "vol_res":
                        cyc = [64, 128, 256, 512]
                        _volume["res"] = cyc[(cyc.index(_volume["res"])
                                              + 1) % len(cyc)]
                        _volume_compute()
                    elif dr_action == "vol_mip":
                        volume_renderer.mode = 1 - volume_renderer.mode
                        _volume["mip"] = volume_renderer.mode == 1
                    elif dr_action == "vol_step_down":
                        volume_renderer.step_mult = max(
                            0.25, volume_renderer.step_mult / 1.5)
                        _volume["step_mult"] = volume_renderer.step_mult
                    elif dr_action == "vol_step_up":
                        volume_renderer.step_mult = min(
                            4.0, volume_renderer.step_mult * 1.5)
                        _volume["step_mult"] = volume_renderer.step_mult
                    elif dr_action in ("vol_tf_down", "vol_tf_up"):
                        # Move the transfer-function knee (middle
                        # control point) up/down in opacity.
                        pts = volume_renderer.tf_points
                        if len(pts) >= 3:
                            x, y = pts[1]
                            y = float(np.clip(
                                y + (0.05 if dr_action == "vol_tf_up"
                                     else -0.05), 0.0, 1.0))
                            pts[1] = (x, y)
                            volume_renderer.upload_transfer_function()
                    elif dr_action == "vol_compute":
                        _volume_compute()
                    drawer.refresh()
                elif dr_action in ("iso_add", "iso_down", "iso_up",
                                   "iso_res", "iso_op", "iso_obj",
                                   "iso_clear"):
                    if dr_action == "iso_add":
                        _iso_recompute(add_new=True)
                    elif dr_action in ("iso_down", "iso_up") and _iso["surfaces"]:
                        f = 10 ** (0.25 if dr_action == "iso_up" else -0.25)
                        _iso["surfaces"][-1]["level"] *= f
                        _iso_recompute()
                    elif dr_action == "iso_res":
                        cyc = [64, 128, 256]
                        _iso["res"] = cyc[(cyc.index(_iso["res"]) + 1)
                                          % len(cyc)]
                        _iso_recompute()
                    elif dr_action == "iso_op" and _iso["surfaces"]:
                        s = _iso["surfaces"][-1]
                        s["opacity"] = round((s["opacity"] + 0.15) % 1.05, 2)
                        if s["opacity"] < 0.1:
                            s["opacity"] = 0.15
                        _iso_recompute()
                    elif dr_action == "iso_obj":
                        sfc = [s for s in iso_renderer.surfaces
                               if s is not None]
                        if sfc:
                            out = os.path.join(
                                screenshot_dir or ".",
                                f"vizmo_iso_{int(time.time())}.obj")
                            mesh_to_obj(out, sfc[-1]["verts"],
                                        sfc[-1]["faces"])
                            toasts.show(
                                f"OBJ: {os.path.basename(out)}", "ok")
                    elif dr_action == "iso_clear":
                        _iso["surfaces"].clear()
                        iso_renderer.clear()
                        drawer.refresh()
                elif dr_action in ("slice_axis", "slice_back", "slice_fwd",
                                   "slice_op_down", "slice_op_up",
                                   "slice_res"):
                    if dr_action == "slice_axis":
                        _slice["normal_idx"] = ((_slice["normal_idx"] + 1)
                                                % len(_SLICE_NORMALS))
                    elif dr_action == "slice_back":
                        geom = _slice.get("_geom")
                        half = geom[2] if geom else camera.speed
                        _slice["offset"] -= 0.05 * 2 * half
                    elif dr_action == "slice_fwd":
                        geom = _slice.get("_geom")
                        half = geom[2] if geom else camera.speed
                        _slice["offset"] += 0.05 * 2 * half
                    elif dr_action == "slice_op_down":
                        _slice["opacity"] = max(0.0,
                                                _slice["opacity"] - 0.1)
                    elif dr_action == "slice_op_up":
                        _slice["opacity"] = min(1.0,
                                                _slice["opacity"] + 0.1)
                    elif dr_action == "slice_res":
                        cyc = [128, 256, 512, 1024]
                        _slice["res"] = cyc[(cyc.index(_slice["res"]) + 1)
                                            % len(cyc)]
                    if not _slice["active"]:
                        _slice["active"] = True
                    _recompute_slice()
                elif dr_action is True and drawer.mode == "halos":
                    _refresh_halo_markers()
                elif dr_action in ("halo_fly", "halo_aperture",
                                   "halo_profile", "halo_csv"):
                    cat = drawer.catalog
                    i = drawer._halo_selected
                    kpc = units.length_to_kpc
                    if dr_action == "halo_csv" and cat is not None:
                        from .catalog import (apply_halo_filter,
                                              halos_to_csv,
                                              mass_threshold_mask,
                                              parse_halo_filter)
                        import datetime as _dt

                        mask = mass_threshold_mask(
                            cat, drawer._halo_threshold)
                        out = os.path.join(
                            screenshot_dir or ".",
                            f"halos_{_dt.date.today():%Y%m%d}.csv")
                        halos_to_csv(out, cat, mask)
                        toasts.show(
                            f"Halos CSV: {os.path.basename(out)}", "ok")
                    elif cat is not None and i is not None:
                        hpos = np.array([cat['x'][i], cat['y'][i],
                                         cat['z'][i]]) / kpc
                        r200_code = cat['R_200'][i] / kpc
                        if dr_action == "halo_fly":
                            camera.fly_to(
                                position=hpos + camera.forward
                                * (-3.0 * r200_code),
                                look_at=hpos, duration=1.2)
                            toasts.show(
                                f"Flying to halo "
                                f"#{int(cat['halo_id'][i])}")
                        elif dr_action == "halo_aperture":
                            _aperture["center"] = hpos
                            _aperture["radius"] = r200_code
                            _aperture["shape_idx"] = 0
                            aperture_panel.enabled = True
                            _submit_aperture()
                        elif dr_action == "halo_profile":
                            data.set_view_center(hpos)
                            drawer.set_scope(hpos, cat['R_200'][i],
                                             "halo")
                            drawer.mode = "profile"
                            drawer.refresh()
                elif dr_action == "spec_voigt":
                    sls = [s for s in _sightlines["list"]
                           if s.trident_spectrum_path
                           and os.path.exists(s.trident_spectrum_path)]
                    if not sls:
                        toasts.show("Run Trident first", "warn")
                    else:
                        try:
                            from .spectro import launch_voigtfit

                            proc = launch_voigtfit(
                                sls[-1], redshift=units.redshift,
                                output_dir=os.path.join(
                                    screenshot_dir or ".", "spectra"))
                            _sightlines["trident_procs"].append(
                                (sls[-1], proc))
                            toasts.show(
                                f"VoigtFit running for {sls[-1].label}...",
                                "info", duration=6.0)
                        except (ImportError, ValueError) as e:
                            toasts.show(str(e), "error", duration=8.0)
                elif dr_action == "spec_pdf":
                    sls = [s for s in _sightlines["list"]
                           if s.trident_spectrum_path
                           and os.path.exists(s.trident_spectrum_path)]
                    if not sls:
                        toasts.show("Run Trident first", "warn")
                    else:
                        try:
                            from .spectro import save_spectrum_pdf
                            import datetime as _dt

                            sl_ = sls[-1]
                            out = os.path.join(
                                screenshot_dir or ".", "spectra",
                                f"{sl_.label}_{_dt.date.today():%Y%m%d}"
                                ".pdf")
                            os.makedirs(os.path.dirname(out),
                                        exist_ok=True)
                            save_spectrum_pdf(sl_, out, snapshot_path,
                                              units.redshift)
                            toasts.show(
                                f"PDF: {os.path.basename(out)}", "ok")
                        except Exception as e:
                            toasts.show(f"PDF failed: {e}", "error")
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
                                   "stats_json", "stats_latex",
                                   "spectrum_csv", "orbit_csv"):
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

    _slice_drag = {"last_y": None}
    _hover = {"tip": "", "t": 0.0}

    _MODE_TIPS = {"SurfaceDensity": "Surface density projection",
                  "WeightedAverage": "Mass-weighted line-of-sight average",
                  "WeightedVariance": "LOS standard deviation (dispersion)",
                  "Composite": "Dual-channel lightness x color",
                  "slice": "Interactive slice plane (Shift+Z)",
                  "isosurface": "Marching-cubes isosurfaces (Shift+I)",
                  "streamlines": "RK4 velocity streamlines (Shift+V)",
                  "volume": "Emission-absorption ray marching (Shift+W)"}

    def _update_hover_tip(x, y):
        # Status-bar tooltips (Section 6.E): mode icon row hover shows
        # the full mode name; field-picker row hover shows the formula.
        tip = ""
        hit = user_menu._hit_test(x, y)
        if (isinstance(hit, tuple) and len(hit) > 3
                and hit[2] == "hbutton" and isinstance(hit[3], tuple)):
            kind, val = hit[3]
            if kind in ("set_mode", "tool"):
                tip = _MODE_TIPS.get(val, val)
        elif field_picker.enabled:
            lx = x - field_picker._panel_x
            ly = y - field_picker._panel_y
            for x0, y0, x1, y1, action in field_picker._buttons:
                if (x0 <= lx <= x1 and y0 <= ly <= y1
                        and isinstance(action, tuple)
                        and action[0] == "pick_field"):
                    from .physics import DERIVED_FIELDS

                    df = DERIVED_FIELDS.get(action[1])
                    if df is not None:
                        tip = f"{action[1]}: {df.description}"
                    break
        if tip != _hover["tip"]:
            _hover["tip"] = tip
            status_bar._last_key = None

    def cursor_callback(win, x, y):
        # Ctrl+LMB drag in slice mode: translate the plane along its
        # normal (world step scaled to the slice size per pixel).
        ctrl = (glfw.get_key(win, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
                or glfw.get_key(win, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS)
        lmb = glfw.get_mouse_button(
            win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        if _slice["active"] and ctrl and lmb:
            if _slice_drag["last_y"] is not None:
                dy = y - _slice_drag["last_y"]
                geom = _slice.get("_geom")
                half = geom[2] if geom else camera.speed
                _slice["offset"] += -dy * (2.0 * half) / 600.0
                _recompute_slice()
            _slice_drag["last_y"] = y
            return
        _slice_drag["last_y"] = None
        now_h = time.perf_counter()
        if now_h - _hover["t"] > 0.1:  # 10 Hz throttle
            _hover["t"] = now_h
            fx, fy = _cursor_to_fb(win)
            _update_hover_tip(fx, fy)
        camera.on_cursor(x, y)

    def scroll_callback(win, xoffset, yoffset):
        nonlocal dirty, ui_dirty, idle_streak
        idle_streak = 0
        if shortcuts_panel.enabled and shortcuts_panel.on_scroll(yoffset):
            ui_dirty = True
            return
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
        if shortcuts_panel.enabled and shortcuts_panel.on_char(
                chr(codepoint)):
            return
        if field_picker.enabled and field_picker.on_char(chr(codepoint)):
            return
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

    # --split second-snapshot pipeline: a dedicated GPUCompute holds
    # snapshot 2 (positions shifted so both centers coincide); during
    # split frames the renderer accumulates pane 2 from these chunks.
    gpu_compute2 = None
    _split_chunks2 = {"chunks": None, "offset": None, "n": 0}

    def _ensure_split_pipeline():
        nonlocal gpu_compute2
        if data2 is None or gpu_compute2 is not None:
            return
        try:
            from .framing import find_center as _fc2

            c1 = data.get_view_center()
            try:
                c2 = _fc2(data2, center_on or "densest")
            except Exception:
                c2 = data2.get_view_center()
            shift = (np.asarray(c1, dtype=np.float64)
                     - np.asarray(c2, dtype=np.float64))
            pos2 = data2.positions + shift[None, :]
            gpu_compute2 = GPUCompute(device)
            m2 = data2.masses.astype(np.float32)
            gpu_compute2.upload_subsample_only(
                pos2, data2.hsml, m2, m2)
            _split_chunks2["chunks"] = gpu_compute2.get_chunk_bufs()
            _split_chunks2["offset"] = gpu_compute2.get_pos_offset()
            _split_chunks2["n"] = data2.n_particles
            toasts.show(
                f"Split: snapshot 2 on GPU "
                f"({data2.n_particles / 1e6:.1f}M)", "ok")
        except Exception as e:
            toasts.show(f"Split pipeline failed: {e}", "error")
            gpu_compute2 = False  # do not retry every frame

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
                right_dock.push_stat(
                    "orbit e", f"{props['eccentricity']:.2f}")
            drawer._last_orbit = (o, props)
            # 3D orbit trail: reuse the streamline line-strip pipeline,
            # colored by time through the active colormap, positioned
            # back in absolute code units about the potential center.
            try:
                from .colormaps import colormap_to_texture_data as _ctd

                kpc = units.length_to_kpc
                trail_pts = center[None, :] + o["pos"] / kpc
                trace = {"points": trail_pts,
                         "values": o["t"] - o["t"].min() + 1e-6}
                lut = _ctd(AVAILABLE_COLORMAPS[_state["_cmap_idx"]])
                tmax = float(o["t"].max() - o["t"].min())
                orbit_trail_renderer.set_lines(
                    [trace], lut, 0.0, max(np.log10(max(tmax, 1e-6)), 1e-6),
                    log_scale=False)
                # linear time coloring: rescale manually
                orbit_trail_renderer.set_lines(
                    [trace], lut, 0.0, max(tmax, 1e-6), log_scale=False)
                _orbit_trail["visible"] = True
            except Exception:
                pass
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
            elif action == "stats_latex":
                from .analysis import region_stats, halo_properties
                from .export import export_latex_table

                sc_center, sc_radius = drawer._active_scope()
                r_use = (sc_radius if sc_radius is not None
                         else drawer._stats_radius_kpc)
                rows, used_r = region_stats(data, center=sc_center,
                                            radius_kpc=r_use)
                halo_rows = halo_properties(data, center=sc_center,
                                            radius_kpc=used_r)
                out = os.path.join(base, f"vizmo_stats_{ts}.tex")
                export_latex_table(list(rows) + halo_rows, out)
                toasts.show(f"LaTeX table: {os.path.basename(out)}", "ok")
            elif action == "spectrum_csv" and drawer._last_ps is not None:
                from .power_spectrum import power_spectrum_to_csv

                out = os.path.join(base, f"vizmo_pk_{ts}.csv")
                power_spectrum_to_csv(out, drawer._last_ps)
                toasts.show(f"P(k) saved: {os.path.basename(out)}", "ok")
            else:
                toasts.show("Nothing to export yet", "warn")
        except Exception as e:
            toasts.show(f"Export failed: {e}", "error")

    def _ensure_gpu_derived(name):
        # Renderer GPU-buffer bypass: compute Temperature/NumberDensity
        # via derived_fields.wgsl and inject into the data cache under
        # the exact derived-field keys, so every consumer (weights,
        # profiles, phase, filters) reads the GPU result and the CPU
        # physics path is skipped. Failures fall back silently (one
        # warning toast).
        if name not in ("Temperature", "NumberDensity"):
            return
        key = f"derived/{name}/{tuple(data.particle_types)}"
        if key in data._cache or gpu_compute is None:
            return
        raw = set(data.available_fields())
        if not {"InternalEnergy", "Density"} <= raw:
            return
        try:
            u = data.get_field("InternalEnergy")
            xe = (data.get_field("ElectronAbundance")
                  if "ElectronAbundance" in raw
                  else np.full(data.n_particles, 1.158, dtype=np.float32))
            rho = data.get_field("Density")
            t_gpu, nh_gpu = gpu_compute.compute_derived_fields(
                u, xe, rho, units)
            tkey = f"derived/Temperature/{tuple(data.particle_types)}"
            nkey = f"derived/NumberDensity/{tuple(data.particle_types)}"
            data._cache[tkey] = t_gpu
            data._cache[nkey] = nh_gpu
            print("  [gpu] derived fields computed on GPU")
        except Exception as e:
            print(f"  [gpu] derived-field bypass failed: {e}")
            if not _state.get("_gpu_derived_warned"):
                _state["_gpu_derived_warned"] = True
                toasts.show(f"GPU derived fields unavailable ({e}); "
                            f"CPU fallback", "warn")

    def _dispatch_menu_action(action):
        """Map MenuBar action strings onto the existing handlers."""
        import os as _os

        if isinstance(action, tuple):
            kind = action[0]
            if kind == "mode":
                _undo_push(f"mode -> {action[1]}")
                _state["_render_mode_name"] = action[1]
                app_proxy._apply_render_mode()
            elif kind == "drawer":
                drawer.toggle(action[1])
            elif kind == "open_recent":
                _pending_load["path"] = action[1]
                _pending_load["label"] = _os.path.basename(action[1])
            return
        if action == "open_file":
            try:
                import tkinter as tk
                from tkinter import filedialog

                root = tk.Tk()
                root.withdraw()
                sel = filedialog.askopenfilename(
                    title="Open snapshot",
                    filetypes=[("HDF5 snapshots", "*.hdf5 *.h5"),
                               ("All files", "*")])
                root.destroy()
                if sel:
                    _pending_load["path"] = sel
                    _pending_load["label"] = _os.path.basename(sel)
            except Exception as e:
                toasts.show(f"Open failed: {e}", "error")
        elif action == "quit":
            glfw.set_window_should_close(window, True)
        elif action == "screenshot":
            _take_screenshot()
        elif action == "publication":
            _export_publication()
        elif action == "export_region":
            _export_region_cutout()
        elif action == "fits_map":
            _export_fits_map()
        elif action == "export_vtk":
            try:
                from .export import export_vtk
                from .physics import UnitSystem as _US

                fields = {"Masses": data.masses}
                out = _os.path.join(screenshot_dir or ".",
                                    f"vizmo_{int(time.time())}.vtu")
                export_vtk(data.positions * units.length_to_kpc,
                           fields, out,
                           aperture_region=drawer._active_region())
                toasts.show(f"VTU: {_os.path.basename(out)}", "ok")
            except Exception as e:
                toasts.show(f"VTU failed: {e}", "error")
        elif action == "stats_latex":
            _handle_drawer_export("stats_latex")
        elif action == "export_zip":
            toasts.show("Use Ctrl+Shift+E for the ZIP bundle", "info")
        elif action == "split":
            toasts.show("Use Shift+S to cycle split modes", "info")
        elif action == "slice":
            toasts.show("Use Shift+Z to activate the slice plane",
                        "info")
        elif action in ("isosurface_open", "streamlines_open",
                        "volume_open"):
            drawer.toggle(action.replace("_open", ""))
        elif action == "colormap_browser":
            cmap_browser.enabled = not cmap_browser.enabled
        elif action == "field_picker":
            field_picker.set_fields(_sd_fields)
            field_picker.search = ""
            field_picker.enabled = not field_picker.enabled
        elif action == "hide_ui":
            pass  # Tab handles this; menu entry is documentation
        elif action == "chrome":
            scale_bar.enabled = not scale_bar.enabled
            status_bar.enabled = scale_bar.enabled
            gizmo.enabled = scale_bar.enabled
        elif action == "dev_overlay":
            overlay.enabled = not overlay.enabled
        elif action == "profiler":
            profiler_panel.enabled = not profiler_panel.enabled
        elif action == "sightline_mode":
            _sightlines["placing"] = True
            toasts.show("Sightline mode: click two points")
        elif action == "help":
            help_panel.enabled = not help_panel.enabled
        elif action == "undo":
            entry = undo.undo(*_undo_state())
            if entry is not None:
                _undo_restore(entry)
                toasts.show(f"Undid: {entry['action']}", "ok")
        elif action == "redo":
            entry = undo.redo(*_undo_state())
            if entry is not None:
                _undo_restore(entry)
                toasts.show(f"Redid: {entry['action']}", "ok")
        elif action == "movie":
            movie_panel.enabled = not movie_panel.enabled
        elif action == "shortcuts":
            shortcuts_panel.enabled = not shortcuts_panel.enabled
            shortcuts_panel.search = ""
        elif action == "about":
            about_panel.enabled = not about_panel.enabled

    def _movie_toggle():
        # Start/stop per movie-panel settings; on stop, assemble with
        # ffmpeg in a background thread (or print the command).
        import datetime as _dt
        import shutil as _sh
        import threading as _th

        if not movie_panel.recording:
            base = screenshot_dir or "."
            _recording["dir"] = os.path.join(
                base, f"vizmo_rec_{int(time.time())}")
            os.makedirs(_recording["dir"], exist_ok=True)
            _recording["frame"] = 0
            movie_panel.recording = True
            movie_panel._last_key = None
            mode = movie_panel.MODES[movie_panel.mode_idx]
            if mode == "Orbit":
                camera.start_orbit(
                    _focus_center(),
                    angular_speed=np.radians(movie_panel.orbit_speed))
                movie_panel._orbit_stop_at = (
                    time.monotonic() + movie_panel.orbit_duration)
            elif mode == "Series" and _series.get("player"):
                _series["player"].fps = movie_panel.fps
                _series["player"].index = 0
                _series["pending"] = 0
                _series["player"].toggle_play()
            toasts.show(f"Recording ({mode})...", "ok")
            return
        movie_panel.recording = False
        movie_panel._last_key = None
        camera.stop_orbit()
        d, n = _recording["dir"], _recording["frame"]
        _recording["dir"] = None
        fmt = movie_panel.FORMATS[movie_panel.fmt_idx]
        fps = movie_panel.fps
        if fmt == "PNG sequence" or n == 0:
            toasts.show(f"Recording stopped: {n} frames in {d}/", "ok")
            return
        from .science_panels import ffmpeg_command

        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".mov" if fmt.startswith("ProRes") else ".mp4"
        out = os.path.join(screenshot_dir or ".",
                           f"vizmo_{stamp}{ext}")
        cmd = ffmpeg_command(fps, fmt,
                             os.path.join(d, "frame_%05d.png"), out)
        if _sh.which("ffmpeg") is None:
            toasts.show("ffmpeg not found - run: " + " ".join(cmd),
                        "warn", duration=12.0)
            return

        def _encode():
            import subprocess as _sp

            toasts.show("Encoding...", "info", duration=4.0)
            r = _sp.run(cmd, capture_output=True)
            if r.returncode == 0:
                toasts.show(
                    f"Movie saved: {os.path.basename(out)} "
                    f"({n} frames, {n / fps:.1f}s at {fps}fps)", "ok",
                    duration=8.0)
            else:
                toasts.show("ffmpeg failed - see terminal", "error")
                print(r.stderr.decode()[-500:])

        _th.Thread(target=_encode, daemon=True).start()

    def _handle_filter_action(action):
        """Mutate data.filters from a drawer action and re-render.

        Ranges nudge in percentile space (5-point steps from a cached
        subsampled percentile table) so the controls behave sensibly on
        wildly log-distributed fields.
        """
        kind = action[0]
        _undo_push(f"filter {action[0][2:]}")
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
            # Once-per-second profiler + memory refresh (Items 4/5).
            if profiler_panel.enabled:
                gpu_times = renderer.read_gpu_timings()
                gpu_ok = getattr(renderer, "_gpu_timing_ok", False)
                times = dict(gpu_times) if gpu_ok and gpu_times else {
                    "cull": _timings["cull"] * 1000,
                    "upload": _timings["upload"] * 1000,
                    "render": _timings["render"] * 1000,
                }
                profiler_panel.set_framebuffer_size(fb_w, fb_h)
                profiler_panel.update(times, last_render_ms,
                                      gpu_ok and bool(gpu_times))
            if overlay.enabled:
                from .science_panels import vizmo_cache_mb

                cur, peak = tracemalloc.get_traced_memory()
                gpu_mb = (gpu_compute.gpu_buffer_bytes() / 1e6
                          if gpu_compute is not None else 0.0)
                by_type = "  ".join(
                    f"{p_}:{(sl.stop - sl.start) / 1e6:.1f}M"
                    for p_, sl in sorted(
                        getattr(data, "_type_slices", {}).items()))
                _mem_stats = {
                    "py heap peak": f"{peak / 1e6:,.0f} MB",
                    "gpu buffers": f"{gpu_mb:,.0f} MB",
                    "hsml cache": f"{vizmo_cache_mb():,.0f} MB",
                    "particles": by_type or "-",
                }
            else:
                _mem_stats = None
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
                            if sl.NHI:
                                right_dock.push_stat(
                                    f"{sl.label} log NHI",
                                    f"{np.log10(sl.NHI):.1f}")
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

        # Timeline playback (monotonic tick) + orbit-record auto-stop.
        tp_ = _series.get("player")
        if tp_ is not None and tp_.playing:
            if tp_.tick(time.monotonic()):
                _series["pending"] = tp_.index
            dirty = True
        if (movie_panel.recording
                and getattr(movie_panel, "_orbit_stop_at", None)
                and time.monotonic() > movie_panel._orbit_stop_at):
            movie_panel._orbit_stop_at = None
            _movie_toggle()

        # Snapshot-series navigation queues a path like any other
        # in-place load.
        if _series["pending"] is not None:
            new_i = _series["pending"]
            _series["pending"] = None
            info = _series["snaps"][new_i]
            _pending_load["path"] = info.path
            _pending_load["label"] = (
                f"snapshot {info.snap_num} (z={info.redshift:.2f})")
            _series["index"] = new_i
            if _series.get("player"):
                _series["player"].index = new_i
                timeline._last_key = None

        # In-place snapshot swap (series arrows, File > Open, recents).
        if _pending_load["path"] is not None:
            _new_path = _pending_load["path"]
            _pending_load["path"] = None
            toasts.show(f"Loading {_pending_load['label'] or _new_path}...",
                        "info")
            try:
                from .framing import find_center as _fc

                old_center = data.get_view_center().copy()
                cam_offset = camera.position - old_center
                new_data = SnapshotData(
                    _new_path, particle_types=list(data.particle_types),
                    hsml_progress=_hsml_progress)
                data.close()
                data = new_data
                snapshot_path = _new_path
                units = UnitSystem(data.header)
                _recents.add(_new_path)
                menubar.menus = build_default_menus(_recents.get())
                menubar._last_items_key = None
                try:
                    new_center = _fc(data, center_on or "densest")
                except Exception:
                    new_center = data.get_view_center()
                data.set_view_center(new_center)
                camera.fly_to(position=new_center + cam_offset,
                              look_at=new_center, duration=0.4)
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
                _sd_fields = data.available_fields_with_derived()
                _state["_vector_fields"] = data.available_vector_fields()
                drawer.refresh()
                drawer.sightlines = _sightlines["list"]
                needs_auto_range = True
                dirty = True
                toasts.show(
                    f"Loaded {os.path.basename(_new_path)}: "
                    f"{data.n_particles / 1e6:.1f}M particles", "ok")
            except Exception as e:
                toasts.show(f"Snapshot load failed: {e}", "error",
                            duration=6.0)

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
        volume_renderer.lod_motion = bool(moved)
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
                    if (SPLIT_MODES[_split["mode_idx"]] is not None
                            and data2 is not None):
                        # Right pane = snapshot 2: swap the subsample
                        # source to gpu_compute2's chunks, accumulate
                        # into texture set 2, then restore.
                        _ensure_split_pipeline()
                        if _split_chunks2["chunks"]:
                            saved_chunks = renderer._subsample_chunks
                            saved_n = renderer.n_total
                            renderer.set_subsample_chunks(
                                _split_chunks2["chunks"],
                                world_offset=_split_chunks2["offset"])
                            renderer.n_particles = min(
                                _split_chunks2["n"],
                                renderer._subsample_max_per_frame)
                            renderer._ensure_fbo(fb_w, fb_h, which=2)
                            renderer._write_camera_uniforms(
                                camera, fb_w, fb_h)
                            renderer._render_accum(
                                camera, fb_w, fb_h,
                                renderer._accum_textures2,
                                encoder=_frame_encoder)
                            renderer.set_subsample_chunks(
                                saved_chunks,
                                world_offset=gpu_compute.get_pos_offset()
                                if gpu_compute else None)
                            renderer.n_total = saved_n
                    if SPLIT_MODES[_split["mode_idx"]] is not None:
                        # Split screen: overdraw the composite with two
                        # viewport-restricted resolves (left/top =
                        # slot 0, right/bottom = slot 1, each with its
                        # own RenderState + colormap).
                        s0, s1 = _state["_slot"][0], _state["_slot"][1]
                        for st, sl in ((_split["left"], s0),
                                       (_split["right"], s1)):
                            st.qty_min, st.qty_max = sl["min"], sl["max"]
                            st.log_scale = int(sl["log"])
                            st.render_mode = sl["mode"]
                            st.field = sl.get("data", sl["weight"])
                        renderer.append_split_resolve(
                            _frame_encoder, _frame_screen_view,
                            fb_w, fb_h, _split["left"], _split["right"],
                            orientation=SPLIT_MODES[_split["mode_idx"]])
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
                for p in (scale_bar, status_bar, toasts, gizmo, drawer,
                          menubar, cmap_browser, field_picker):
                    p.set_framebuffer_size(fb_w, fb_h)
                menubar.update()
                if timeline.enabled:
                    timeline.set_framebuffer_size(fb_w, fb_h)
                    timeline.update()
                for pnl in (movie_panel, about_panel, shortcuts_panel):
                    if pnl.enabled:
                        pnl.set_framebuffer_size(fb_w, fb_h)
                        pnl.update()
                if welcome.enabled:
                    welcome.set_framebuffer_size(fb_w, fb_h)
                    welcome.update()
                right_dock.set_framebuffer_size(fb_w, fb_h)
                if right_dock.enabled:
                    right_dock.update(
                        drawer.mode if drawer.enabled else None)
                if cmap_browser.enabled:
                    cmap_browser.update()
                if field_picker.enabled:
                    field_picker.set_fields(_sd_fields)
                    field_picker.update()
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
                status_bar.update(camera, units, view_center,
                                  _hover["tip"] or active_field,
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
                        vsent = sl_t.extra.get("voigt_done_sentinel")
                        if vsent and os.path.exists(vsent):
                            try:
                                from .spectro import (
                                    parse_voigtfit_components)

                                comps = parse_voigtfit_components(
                                    open(sl_t.extra["voigt_log"]).read())
                                sl_t.extra["voigt_components"] = comps
                                drawer.refresh()
                                toasts.show(
                                    f"VoigtFit: {len(comps)} components "
                                    f"for {sl_t.label}", "ok",
                                    duration=8.0)
                            except Exception as e:
                                toasts.show(f"VoigtFit parse: {e}",
                                            "error")
                            done_now.append((sl_t, proc))
                            continue
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
                        mem=_mem_stats,
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
                    ptype_counts={p_: sl.stop - sl.start
                                  for p_, sl in getattr(
                                      data, "_type_slices", {}).items()},
                    perf=(f"{last_render_ms:.0f} ms | {fps:.0f} fps | "
                          f"LOD {min(renderer.n_particles / max(renderer.n_total, 1), 1.0) * 100:.0f}%"),
                )
                # Render-mode icon row tool requests (SL/IS/ST/VL).
                if user_menu.pending_tool:
                    tool = user_menu.pending_tool
                    user_menu.pending_tool = None
                    if tool == "slice":
                        if not _slice["active"]:
                            _slice["active"] = True
                            drawer.enabled = True
                            drawer.mode = "slice"
                            _recompute_slice()
                    elif tool == "isosurface":
                        drawer.toggle("isosurface")
                        if (drawer.mode == "isosurface"
                                and not _iso["surfaces"]):
                            _iso_recompute(add_new=True)
                    else:
                        drawer.toggle(tool)

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
                if iso_renderer.surfaces and not _iso["busy"]:
                    iso_renderer.write_uniforms(camera)
                    iso_renderer.render_to_pass(rpass)
                if volume_renderer.enabled:
                    volume_renderer.render_to_pass(rpass, camera)
                if _stream["visible"] and stream_renderer.lines:
                    stream_renderer.write_uniforms(camera)
                    stream_renderer.render_to_pass(rpass)
                if halo_markers.n_vertices > 0:
                    halo_markers.write_uniforms(camera)
                    halo_markers.render_to_pass(rpass)
                if _orbit_trail["visible"] and orbit_trail_renderer.lines:
                    orbit_trail_renderer.write_uniforms(camera)
                    orbit_trail_renderer.render_to_pass(rpass)
                if arrow_renderer.n_instances > 0:
                    arrow_renderer.write_uniforms(camera)
                    arrow_renderer.render_to_pass(rpass)
                if _slice["active"] and slice_renderer.grid is not None:
                    geom = _slice.get("_geom")
                    if geom is not None:
                        c0, nrm, half = geom
                        from .wgpu_renderer import plane_basis as _pb

                        e1, e2, _ = _pb(nrm)
                        corners = []
                        ok_all = True
                        for su, sv in ((-1, -1), (1, -1), (1, 1),
                                       (-1, 1)):
                            wpt = c0 + su * half * e1 + sv * half * e2
                            res_p = _world_to_screen(wpt, fb_w, fb_h)
                            if res_p is None:
                                ok_all = False
                                break
                            corners.append(
                                (res_p[0] / fb_w * 2 - 1.0,
                                 1.0 - res_p[1] / fb_h * 2))
                        if ok_all:
                            slice_renderer.render_to_pass(rpass, corners)
                if sink_panel.enabled:
                    sink_panel.render_to_pass(rpass)
                if overlay.enabled:
                    overlay.render_to_pass(rpass)
                if drawer.enabled:
                    drawer.render_to_pass(rpass)
                if help_panel.enabled:
                    help_panel.render_to_pass(rpass)
                if profiler_panel.enabled:
                    profiler_panel.render_to_pass(rpass)
                if right_dock.enabled:
                    right_dock.render_to_pass(rpass)
                menubar.render_to_pass(rpass)
                if cmap_browser.enabled:
                    cmap_browser.render_to_pass(rpass)
                if field_picker.enabled:
                    field_picker.render_to_pass(rpass)
                if timeline.enabled:
                    timeline.render_to_pass(rpass)
                for pnl in (movie_panel, about_panel, shortcuts_panel):
                    if pnl.enabled:
                        pnl.render_to_pass(rpass)
                if welcome.enabled:
                    welcome.render_to_pass(rpass)
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
