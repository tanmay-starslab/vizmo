# Changelog

## 0.11.0 - 2026-06-10

### Added
- Phase-diagram brushing (Shift+P): rect / polygon / ellipse
  selections drawn in data coordinates on the fixed-geometry phase
  plot (two-click rect/ellipse, click-vertices + Close for polygon);
  submit computes the particle mask (log-axis aware, NaN-safe),
  highlights the selection in the 3D view as a capped 500k point
  cloud, shows an 80px LOS inset heat map, and toasts the selected
  fraction. Escape clears. Brush-only toggles thread the mask into
  radial profiles and region statistics.
- Marginal histograms on the phase diagram (top + right, 20 bins,
  accent bars) with green brushed-subset overlays.
- Observational CSV overlays on the phase diagram (columns named
  after the displayed fields + optional label), multiple datasets in
  cycling colors, legend, Obs- to clear - built for COS-IGrM
  constraints on n_H-T.
- Science buttons: t_cool/t_ff = 1 precipitation threshold locus
  (bisection on the TN01 cooling function; rises with density as
  tested), T_vir line from the aperture rotation curve, NFW density-
  profile fit overlay (rho_s/r_s recovered to 2% in tests), stats
  clipboard export (pyperclip fallback) + compare-with-previous diff
  toasts, orbit-to-galpy script export bridging the fitted NFW
  potential into the galpy ecosystem.

### Notes
- Split-screen second-snapshot rendering shipped in v0.9.0; the
  architectural pools refactor specd alongside it was intentionally
  skipped (working pipeline, zero new capability). Half labels ride
  the RenderState fields. Brush points are 1px (point-list topology
  has no size control in WebGPU).

## 0.10.0 - 2026-06-10

### Added
- Timeline scrubber (4.C): bottom-edge ruler with dz=0.2/1.0 ticks,
  per-snapshot circles, accent playhead, click-to-jump, transport
  buttons (first/back/play-pause/fwd/last), fps cycling, z/t readout;
  Space play/pause + Home/End jumps; monotonic-clock playback driving
  the in-place snapshot loader.
- Movie recorder (10.B, Ctrl+V / File > Export > Movie...): Manual /
  Orbit (deg/s + duration, auto-stop) / Series (player-driven) modes,
  fps/resolution/format (ProRes on macOS only, 4K memory warning),
  background ffmpeg assembly with copyable-command fallback and a
  frames/duration success toast.
- About panel (real version table incl. not-installed markers,
  credits) and a categorized, searchable, scrollable Keyboard
  Shortcuts browser; keymap entries now carry categories.
- Undo system: 20-deep undo/redo of aperture, filter, and render-mode
  changes (Ctrl+Z / Ctrl+Shift+Z, new Edit menu), deep-copied
  snapshots with exact round-tripping.

## 0.9.0 — 2026-06-10

### Deep-infrastructure completion — nothing deferred remains
- Async startup (8.A): the snapshot loads in a background thread
  behind a live loading screen (spinner, 400px accent progress bar,
  per-PartType status from a shared progress dict); the main loop
  starts only once data exists; load-time toast on completion.
- GPU derived-field bypass: Temperature/n_H computed by
  derived_fields.wgsl and injected into the field cache, skipping the
  CPU physics path for every consumer. Fixed a real 65,535-workgroup
  dispatch-limit bug (17.1M particles) with 2D dispatch in both
  compute shaders.
- --split second-snapshot GPU pipeline: snapshot 2 uploads into its
  own GPUCompute with centers aligned; split frames accumulate the
  right pane from those chunks (source swap + restore per frame).
- Volume motion-LOD: step doubling + half max_steps while the camera
  moves, automatic restore on idle.
- Hover tooltips: mode names and derived-field formulas in the
  status bar (10 Hz hover tracking).

## 0.8.0 — 2026-06-10

### Added
- Halo list + inspector (Ctrl+H, --catalog GROUPCAT): sortable
  (M_halo/M_star/R_200/id, asc/desc cycling), range-filter syntax
  (M_halo>1e12 / bare halo id), 10^10–10^14 marker mass threshold,
  CSV export; inspector with CENTRAL/SATELLITE badge and Fly-to /
  Set-aperture(R_200) / Profile actions wired to the camera,
  aperture, and profile machinery.
- Spectrum viewer (sightline panel > View): Trident .h5/.fits
  parsing, continuum dashes, absorption-trough shading, Compare mode
  overlaying all sightlines in distinct cycling colors, fast-N
  readouts alongside; VoigtFit component-line parser shipped+tested.
- Welcome chooser: `vizmo` with no arguments prints recents +
  quickstart and opens the file dialog (Section 6.J chooser form).

### Completed in follow-up (same day) — deferred list cleared
- 3D halo cross-hair markers: line-list renderer, 0.10 R_200 crosses,
  plasma-colored by log M_halo over 10^10..10^14, threshold-masked,
  white-highlighted selection; Shift+click within 12 px of a marker
  opens the halo inspector (particle pick falls through otherwise).
- VoigtFit subprocess launcher (sentinel pattern, graceful when not
  installed) + fitted components parsed into the sightline and drawn
  as dashed display-Gaussian overlays on the spectrum plot.
- Spectrum Save-PDF: continuum, trough shading, component markers,
  metadata footer (snapshot, z, impact parameter).
- Full-canvas welcome splash: ASCII logo, recents with size/date,
  accent Open button, quickstart tips, version line; click-outside/
  Escape dismissal; shown after the bare-command chooser launch.
- Drag-and-drop: dropping an .hdf5/.h5 onto the window loads it in
  place via the shared pending-load queue (glfw drop callback).

## 0.7.1 — 2026-06-10

### Performance infrastructure
- vizmo/fast_ops.py: numba-parallel kernel interpolation, LOS column
  density (kernel-column table interp), and radial-profile binning
  with pure-numpy fallbacks; equivalence-tested to 1e-8..1e-10.
- shaders/derived_fields.wgsl + GPUCompute.compute_derived_fields():
  Temperature and n_H computed on the GPU, verified against the CPU
  physics to 2e-4 on Metal.
- DataManager async-load API: is_loaded / load_progress /
  load_status markers and get_field_async()/field_ready() per-field
  background futures (2-worker pool).

### Completed in follow-up (same day)
- F10 GPU profiling overlay: timestamp queries on the accum + resolve
  passes (timestamp-query feature, EMA smoothing, 1 Hz readback) with
  a CPU-timed fallback bar chart when unsupported.
- Dev-overlay memory section (\\ key): tracemalloc heap peak, GPU
  buffer bytes (registry scan), per-type particle counts, hsml cache
  directory size.
- spectro + analysis hot paths rewired onto fast_ops (numba) with the
  numpy fallbacks intact.

### Still deferred
- Non-blocking startup with loading screen; renderer GPU-buffer
  bypass for derived fields.

## 0.7.0 — 2026-06-10

### Added
- **Regions boolean panel** (`Ctrl+R`, Section 1.B): up to 8 named
  regions built from the aperture tool's shapes, OR/AND/NOT chain
  evaluation, Submit-all turns the combined mask into the active
  analysis scope (profiles/phase/stats/exports all honor it),
  Save/Load to `~/.config/vizmo/regions_<hash>.json`.
- **3D orbit trails**: computed orbits draw as time-colored line
  strips over the particle render (O toggles once computed).
- **Toast system per Section 6.I**: bottom-right stack, max 5,
  0.2 s fade-in / 0.5 s fade-out, type-colored left border strip,
  `push()` API.
- **Phase 5 render modes** (previous sessions this cycle): slice
  plane (Shift+Z, GPU gather + CPU fallback, FITS export),
  isosurfaces (Shift+I, marching cubes + Phong, 4 surfaces, OBJ),
  split screen (Shift+S), streamlines (Shift+V, RK4 + KDTree),
  magnetic-field arrow glyphs (Shift+B, instanced), volume ray
  marching (Shift+W, emission-absorption + MIP, GPU fixed-point CIC).
- Orbit integration module, absorption sightlines + Trident
  launcher, snapshot series navigation, Subfind/Rockstar catalog
  loaders, LaTeX/VTU/ZIP exports, checksum hsml cache, DarkTheme
  tokens (versions 0.5.0–0.6.0).

### Fixed
- WGSL reserved keyword `ref` in arrows.wgsl crashed startup.
- wgpu sampler kwargs (`address_mode_u/v/w`).
- Velocity anisotropy beta no longer contaminated by coherent
  radial flows.
- Snapshot-number parsing no longer eats the '5' in '.hdf5'.

### Known gaps (tracked for next cycle)
- Async snapshot I/O (8.A), top menu bar (6.D), left-sidebar
  restructure (6.E), right accordion dock (6.H), welcome screen
  (6.J), colormap browser / field picker (6.F/6.G), halo
  marker/list UI (4.F/4.G), Spectrum Viewer panel (3.D),
  region wireframe draw_overlay(), --split second-snapshot
  particle pipeline, volume motion-LOD + click-to-add TF editor.
