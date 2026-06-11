# Changelog

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
