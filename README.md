# vizmo

![Animated demo of vizmo flying through a galaxy simulation](vizmo_demo.gif)

Real-time 3D fly-through explorer for unstructured simulation data. Loads simulation snapshot data and renders interactive surface density maps, mass-weighted averages, velocity dispersions, and composite lightness x color maps on the GPU via WebGPU.

Beyond rendering, vizmo is a quantitative explorer: it derives physical
fields (temperature, n_H, pressure, entropy, radial velocity, |B|, Z/Zsun,
stellar ages) from raw snapshot data with full cosmological unit handling,
and puts interactive science tools in the window — particle inspection by
clicking, phase diagrams, radial profiles, region statistics, a physical
scale bar, publication-figure export, and HDF5 cutout export.

Codes whose outputs have been successfully loaded to date:
- GIZMO
- AREPO
- Gadget-3
- Athena
- RAMSES
- ART

If your data format is parseable by `yt` then vizmo should be able to load it.

This is a **vibecoded** app. I have no idea how its front- or backends work, and am not sure if I could really properly support or get it running for you 10 years from now! But it would not exist otherwise. Software is weird now. It is what it is. 

But I hope you enjoy it and please feel free to report bugs, request features, or develop your own and send a PR. Making it work for more data formats is well within scope, just ask!

## Installation

### pypi

```bash
pip install vizmo
```

### Local source

```bash
pip install -e .
```

### Requirements

- Python >= 3.9
- WebGPU-capable GPU (Metal on macOS, Vulkan on Linux, D3D12 on Windows)

## Quickstart

Run `vizmo` with a snapshot file as the argument:

```bash
vizmo path/to/snapshot.hdf5
```

For gizmo/arepo/gadget: The snapshot must contain at least one `PartTypeN` group with `Coordinates`. Masses come from a per-particle `Masses` dataset or, when absent (TNG-style dark matter), from the header `MassTable`. Smoothing lengths are taken from `SmoothingLength`/`KernelMaxRadius`/`Hsml`/`StellarHsml`/`SubfindHsml` if present, otherwise from a per-type softening in the header (e.g. `SofteningTypeN`, `SofteningTable`), and as a last resort computed on the fly via meshoid — that last-resort result is cached under `~/.cache/vizmo` so it only happens once per snapshot. Star particles (`PartType5`) are rendered as point sources when present.

### Particle type selection

A row of six tickboxes labeled `0`–`5` at the top of the user menu controls which `PartTypeN` groups are loaded into the active particle pool. Disabled (greyed out) ticks correspond to types that are not present in the snapshot. Toggling a tick reloads the pool, refreshes the **Weight**/**Field 2**/**Data** dropdowns to show only the scalar/vector fields that are common to *all* selected types, and re-runs auto-range. The default selection is `PartType0` if present, otherwise the first available type.

### CLI options

```
vizmo snapshot.hdf5 [--width 1920] [--height 1080] [--fov 90]
                    [--fullscreen] [--screenshot OUT.png]
                    [--profile OUT.pstats]
                    [--types 0,1,4] [--center X,Y,Z | --center-on MODE]
                    [--radius R] [--colormap NAME] [--screenshot-dir DIR]
```

- `--types 0,1,4` — initial `PartTypeN` selection (default: gas only).
- `--center X,Y,Z` — aim the starting view at this point (snapshot coords).
- `--center-on {densest,potential,com,median}` — auto-find the starting
  center: highest-density particle, potential minimum, center of mass, or
  mass-weighted median. Falls back to `median` when the needed fields are
  missing.
- `--radius R` — starting camera distance from the center; also rescales
  flight speed to the framed region.
- `--colormap NAME` — starting colormap.
- `--screenshot-dir DIR` — where `P`-key screenshots and `V`-key frame
  recordings go (default: cwd).

Centering on the densest gas of a TNG halo cutout, for example:

```bash
vizmo cutout.hdf5 --center-on densest --radius 300 --colormap inferno
```

## Rendering backend

vizmo uses a WebGPU backend ([wgpu-py](https://github.com/pygfx/wgpu-py)) with GPU-resident particle data. Compute shaders perform frustum culling, LOD selection, and per-cell summary gathering with zero CPU↔GPU per-frame transfer. On unified-memory systems (Apple Silicon) field switches are also near-zero copy.

The renderer uses progressive refinement and an auto-LOD subsample cap that adapts within a user-controlled ceiling to keep interaction smooth during motion and sharpen on idle.

## Algorithms

Getting this to work on 100M+ particle datasets has been tricky. Here is what has worked so far:
- Adaptive random subsampling, rescaling masses and radii appropriately, wired up to a PID loop to achieve a smooth framerate.
- Fully-sampled renders when the camera stops, resuming adaptive subsampling when it moves again.
- Multi-grid kernel splatting, implementing the algorithm of [meshoid](https://github.com/mikegrudic/meshoid) on the GPU, so that particles appearing large on the screen do not have to splat a huge number of pixels.
- GPU-side brute-force frustum culling. Experimented extensively with tree-based algorithms, but at ~100M bruteforce works fine.

## Science tools

**Derived physical fields.** Every field menu (Weight / Field 2 / Data,
all render modes) lists physically meaningful fields computed on demand
from the raw snapshot with full cosmological (a, h) unit handling:
`Temperature` (K, from internal energy + electron abundance),
`NumberDensity` (n_H in cm^-3), `Pressure` (P/k_B in K cm^-3),
`Entropy` (T/n^2/3), `SoundSpeed`, `VelocityMagnitude`,
`RadialVelocity` (about the view center, positive = outflow),
`RadiusFromCenter`, `MagneticFieldMagnitude` (uG),
`MetallicityZsun`, and `StellarAge` (Gyr, flat-LCDM).
Color a TNG halo by temperature in two clicks.

**Analysis drawer** (right side, toolbar buttons or keys):
- **Inspector** (`I`, or `Shift+Click` any particle) — physical
  properties of the particle under the cursor: n_H, T, Z, v_r, SFR,
  |v|, ID, distance to center; one click re-centers the view on it.
- **Phase diagram** (`G`) — mass-weighted 2D histograms with preset
  pairs (n_H–T, n_H–P, T–Z, r–T, r–v_r), auto log axes.
- **Radial profile** (`J`) — shell density or mass-weighted profiles
  (T, v_r, Z, P, |v|) about the view center.
- **Region statistics** (`U`) — total/per-type mass, half-mass radius,
  SFR, <T>, <Z>, <v_r> inside an adjustable sphere.

**Science chrome** (`F9` toggles): physical scale bar (pc/kpc/Mpc,
1-2-5 rounded), status bar (camera position in kpc, distance to
center, redshift, active field + units, particle counts), and an
orientation triad showing the simulation axes.

**Analysis aperture** (`M`, or toolbar **Aperture**): draw a sphere in
the scene — click drops the center on the particle under the cursor,
scroll resizes, `M` submits, `Shift+M` clears. On submit the center is
refined inside the sphere by your choice of densest particle,
potential minimum, Power+03 shrinking sphere, or center of mass
(cycled from the drawer). Phase diagrams, profiles, and statistics
then compute inside the aperture (per-tool Aperture/Global toggle),
and Orbit / go-to-center / axis views pivot about it. Profiles include
shell density, enclosed mass, rotation curve v_c = sqrt(GM(<r)/r), and
3D velocity dispersion.

**Halo properties** (Stats panel, `Halo` button): spherical-overdensity
masses M200c/R200c and M500c/R500c from the header cosmology, NFW
concentration fit, Bullock spin, velocity anisotropy beta (coherent
radial flows correctly excluded), gas / cold-gas / baryon fractions vs
cosmic, boundary mass flux dM/dt, and an approximate circularity D/T
when Potential is present. Load `--types 0,1,4` for meaningful virial
masses. `JSON` exports everything with metadata.

**Profiles**: shell density, enclosed mass, rotation curve, 3D velocity
dispersion, specific angular momentum |sum m r x v|/M, plus any derived
field (t_cool/t_ff, entropy, abundances, ...). `Split` overlays
cold/warm/warm-hot/hot temperature-phase tracks; `CSV` exports.

**Phase diagrams**: preset pairs or any-X vs any-Y (`X>` / `Y>` cycle
every field), weighting selectable between mass / volume / SFR /
number (`W:` button), `Save` writes the 2D histogram as FITS.

**Power spectrum** (`Shift+K`): shell-averaged P(k) of the density (or
any field) contrast via CIC + FFT inside the aperture cube; CSV export.

**Selection regions** (programmatic + `--region-file` foundation):
sphere, box, cylinder, slab, cone, ellipsoid primitives with boolean
union / intersect / subtract composition (vizmo.selection), JSON
persistence — the geometric base for multi-shape apertures.

**Multi-shape apertures**: while placing (`M`), `Tab` cycles
sphere / cylinder / box / slab / cone / ellipsoid (each with its own
ring color); axis shapes orient along the camera. Center refinement
offers densest / potential / shrinking-sphere / CoM / FoF most-massive
group / stellar CoM.

**Orbit integration** (`O`): Shift+click a particle, press Compute —
an NFW potential is fit to the aperture's enclosed-mass profile and
the orbit integrated with DOP853 (energy-conserving to 0.1%); the
panel plots the time-colored trajectory + r(t) and reports
apo/peri/eccentricity/period/circularity. Stream releases Jacobi-
radius tracers; CSV exports the trail. pytreegrav supported when
installed.

**Absorption sightlines** (`Shift+A`): click two particles — exact
kernel-integrated N_H and N_HI (M4 projected kernel, normalization
verified to 0.2%) plus clearly-labelled CIE-approximate O VI / C IV /
Mg II columns appear instantly; segments render as labelled overlay
lines; the sightline panel lists everything with CSV export and a
background **Trident** launcher for real spectra (polled, toasted).

**Snapshot series** (`vizmo ./snapdir/ --series`): Left/Right arrows
step through time with the center re-tracked and the camera carried
over; Subfind/Rockstar catalogs load via vizmo.catalog.

**Exports**: LaTeX stats tables (TeX button), dependency-free VTU for
ParaView, and `Ctrl+Shift+E` bundles every computed product (profiles,
stats JSON + TeX, phase FITS, sightline columns, README) into one ZIP.

**Field filters** (`F`, or toolbar **Filters**): Firefly-style range
cuts on any raw or derived field — e.g. show only gas with
T < 3x10^4 K, or only inflowing material (RadialVelocity < 0).
Filters stack, work in every render mode, and nudge their bounds in
percentile steps so they behave on fields spanning 8 decades. Also
available headlessly: `--filter Temperature:0:3e4` (repeatable).

**Exports:**
- `Ctrl+P` / toolbar **Figure** — publication-ready PNG with colorbar,
  ticks, units, scale bar, and metadata caption burned in.
- `Ctrl+E` / toolbar **Cutout** — write the sphere around the view
  center to a standalone Gadget-style HDF5 (round-trips through vizmo
  and standard tools; `.csv` also supported).
- `Ctrl+M` / toolbar **FITS** — kernel-weighted projected map
  (meshoid GridSurfaceDensity) of the active field over the aperture,
  written as FITS with linear-kpc WCS + PNG quicklook. Surface density
  in Msun/kpc^2; other fields as mass-weighted projections.
- `V` — frame recording for movies (pairs well with Orbit).

CMasher perceptually-uniform colormaps (cmr.rainforest, cmr.ember,
cmr.cosmic, ...) join the matplotlib set in the colormap cycle.

## Controls

Press `F1` or `H` in-app for this list. The top-left toolbar gives
one-click access to all of it: Auto-range / Screenshot / Figure / Rec /
Help and Inspect / Phase / Profile / Stats / Orbit / Cutout.

**Camera:**
- `W/A/S/D` — Move forward/left/back/right
- `Z/X` — Move up/down
- `Q/E` — Roll left/right
- Mouse (click + drag) — Look around
- Scroll wheel — Adjust flight speed
- `Ctrl`+Scroll or `[/]` — Optical zoom (FOV 10–140°)
- `1-9` — Fly to camera bookmark (eased); `Shift+1-9` — save bookmark
  (persisted per snapshot under `~/.config/vizmo`)
- `N` / `Shift+N` — Look at the view center / fly toward it
- `Y` / `Shift+Y` — Orbit the view center (reverse with Shift); any
  manual input disengages
- `F2/F3/F4` — Snap to +X/+Y/+Z axis views (`Shift` for negative)

**Visualization & analysis:**
- `Tab` — Hide/show all UI
- `C` — Cycle colormap
- `L` — Toggle log/linear scale
- `R` — Auto-range color scale (composite: Color slot)
- `T` — Auto-range composite Lightness slot
- `+/-` — Contract/expand color range
- `,/.` — Lower/raise the auto-LOD subsample-cap ceiling
- `Shift+Click` — Pick particle (opens inspector)
- `I` / `G` / `J` / `U` / `F` — Inspector / Phase diagram / Radial
  profile / Region stats / Field filters
- `M` / `Shift+M` — Place analysis aperture / clear it
- `Ctrl+M` — Export FITS map
- `F9` — Science chrome (scale bar, status bar, axes triad)
- `P` — Save screenshot; `Ctrl+P` — publication figure
- `Ctrl+E` — Export region cutout (HDF5)
- `V` — Start/stop frame recording (PNG sequence + ffmpeg hint)
- `F1` or `H` — Help panel
- `\` — Toggle dev overlay
- `K` — Toggle sink/star panel
- `Esc` — Close panel/drawer, then quit

Colormap, mouse inversion, and target FPS persist across sessions in
`~/.config/vizmo/config.json`.

## Render Modes

Select from the **Mode** dropdown in the user menu:

- **SurfaceDensity** — Projected surface density of a weight field. Supports combining two fields with arithmetic operators (Op / Field 2).
- **WeightedAverage** — Mass-weighted line-of-sight average of a data field.
- **WeightedVariance** — Mass-weighted line-of-sight standard deviation (e.g. velocity dispersion).
- **Composite** — CoolMap-style dual-field visualization. Encodes one field in lightness and another in colormap hue. Each channel has independent render mode, field selection, limits, and scaling.

### Vector Fields

3D vector fields (e.g. Velocities) are automatically detected. When selected as a weight or data field, a **Proj** dropdown appears:

- **LOS** — Per-particle line-of-sight component (dot product with the unit vector from the camera to the particle). Invariant under camera rotation; recomputed when the camera translates past a small threshold.
- **|v|** — Euclidean norm.
- **|v|^2** — Squared norm.

## Architecture

```
vizmo/
  app.py            - CLI entry point
  wgpu_app.py       - Main loop, key actions, progressive refinement, auto-LOD
  wgpu_renderer.py  - WGPURenderer: RenderMode, accumulate + resolve + composite passes
  gpu_compute.py    - GPUCompute: GPU-resident data, compute cull/LOD/gather
  wgpu_overlay.py   - wgpu panel rendering (all overlays)
  overlay.py        - Panel/PanelStyle base, DevOverlay, UserMenu, Toolbar, Help
  science_panels.py - Scale bar, status bar, toasts, axes gizmo, analysis drawer
  physics.py        - UnitSystem (code->physical) + derived-field registry
  analysis.py       - Ray picking, radial profiles, phase histograms, region stats
  export.py         - Publication-figure annotation, HDF5/CSV region cutouts
  framing.py        - View centering (densest/potential/com/median) + camera framing
  session.py        - Bookmarks, settings persistence, eased pose restore
  keymap.py         - Canonical keybinding list (help panel + README)
  camera.py         - 6DOF camera, eased fly-to transitions, orbit autopilot
  data_manager.py   - HDF5 I/O with lazy loading and cosmological corrections
  field_ops.py      - Field arithmetic and vector projections
  colormaps.py      - Matplotlib colormap to GPU texture
  shaders/          - WGSL shaders (common, splat_subsample, resolve, composite,
                      star, text)
```
