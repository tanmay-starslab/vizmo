# Changelog

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

### Not included (deferred)
- Non-blocking startup with loading screen, renderer GPU-buffer
  bypass for derived fields, GPU timestamp profiling overlay (F10),
  memory overlay, spectro/analysis rewiring onto fast_ops.

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
