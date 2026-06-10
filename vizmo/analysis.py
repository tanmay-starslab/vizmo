"""Quantitative analysis backend: particle picking, radial profiles,
phase-space histograms, and region statistics.

Everything here is pure numpy/scipy on SnapshotData arrays — no GPU and
no UI — so it is unit-testable headlessly. The science panels and the
inspector drive these functions.
"""

import numpy as np

from .physics import UnitSystem, DERIVED_FIELDS, field_unit_label


# ---------------------------------------------------------------------------
# Spatial index + picking
# ---------------------------------------------------------------------------

def ensure_kdtree(data):
    """Build (once) and return a cKDTree over data.positions.

    Cached on the SnapshotData instance, invalidated when the particle
    pool size changes (type toggle / reload).
    """
    from scipy.spatial import cKDTree

    tree = getattr(data, "_pick_tree", None)
    if tree is not None and tree.n == data.n_particles:
        return tree
    tree = cKDTree(data.positions)
    data._pick_tree = tree
    return tree


def pick_particle(origin, direction, data, fov_deg=90.0, n_steps=64,
                  angular_tol_deg=1.5):
    """Pick the particle nearest to a view ray.

    Marches log-spaced sample points along the ray, queries the KD-tree
    for the nearest particle to each sample, and selects the candidate
    with the smallest *angular* offset from the ray (ties resolved by
    distance). Angular selection matches what the user perceives as
    "under the cursor" in a perspective projection.

    Args:
        origin: (3,) camera position (code units).
        direction: (3,) unit view ray through the cursor.
        data: SnapshotData.
        fov_deg: current FOV — bounds the search cone.
        n_steps: ray sample count.
        angular_tol_deg: max accepted angular offset from the ray.

    Returns:
        int particle index into the concatenated pool, or None.
    """
    if data.n_particles == 0:
        return None
    tree = ensure_kdtree(data)
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(np.linalg.norm(direction), 1e-30)

    # Ray extent: from very near to the far side of the particle cloud.
    span = data.positions.max(axis=0) - data.positions.min(axis=0)
    far = float(np.linalg.norm(span)) or 1.0
    near = max(far * 1e-5, 1e-6)
    ts = np.geomspace(near, 2.0 * far, n_steps)
    samples = origin[None, :] + ts[:, None] * direction[None, :]

    _, idxs = tree.query(samples, k=1, workers=-1)
    idxs = np.unique(idxs)
    cand = data.positions[idxs] - origin[None, :]
    dist = np.linalg.norm(cand, axis=1)
    ok = dist > near
    if not ok.any():
        return None
    idxs, cand, dist = idxs[ok], cand[ok], dist[ok]
    along = cand @ direction
    ok = along > 0  # in front of the camera
    if not ok.any():
        return None
    idxs, cand, dist, along = idxs[ok], cand[ok], dist[ok], along[ok]
    perp = np.linalg.norm(cand - along[:, None] * direction[None, :], axis=1)
    ang = np.degrees(np.arctan2(perp, along))
    # Particles with large kernels are visually bigger: allow a looser
    # angular match scaled by their apparent kernel size.
    h = data.hsml[idxs]
    ang_kernel = np.degrees(np.arctan2(h, np.maximum(along, 1e-30)))
    eff = ang / np.maximum(1.0, ang_kernel / angular_tol_deg)
    best = int(np.argmin(eff))
    if eff[best] > angular_tol_deg:
        return None
    return int(idxs[best])


def particle_ptype(data, index):
    """PartType int for a concatenated-pool index."""
    for p, sl in getattr(data, "_type_slices", {}).items():
        if sl.start <= index < sl.stop:
            return p
    return None


def particle_summary(data, index):
    """Human-readable (label, value) rows for one particle, in physical
    units where a conversion is known."""
    units = UnitSystem(data.header)
    p = particle_ptype(data, index)
    pos = data.positions[index]
    center = data.get_view_center()
    r_kpc = float(np.linalg.norm(pos - center) * units.length_to_kpc)

    label = data.ptype_labels.get(p, str(p)) if p is not None else "?"
    rows = [
        ("Type", f"PartType{p} ({label})"),
        ("Index", f"{index:,}"),
        ("r -> center", f"{r_kpc:,.2f} kpc"),
        ("Mass", f"{float(data.masses[index]) * units.mass_to_msun:,.3g} Msun"),
    ]

    sl = data._type_slices.get(p)
    local = index - sl.start if sl is not None else index

    def raw(field):
        try:
            grp = data._file[f"PartType{p}"]
            if field in grp:
                return np.asarray(grp[field][local])
        except Exception:
            pass
        return None

    rho = raw("Density")
    if rho is not None:
        rho_cgs = float(rho) * units.density_to_cgs
        nh = rho_cgs * 0.76 / 1.67262192369e-24
        rows.append(("n_H", f"{nh:.3g} cm^-3"))
    for name in ("Temperature", "MetallicityZsun", "RadialVelocity"):
        if name in DERIVED_FIELDS:
            try:
                vals = data.get_field(name)
                rows.append((name, f"{float(vals[index]):.4g} "
                             f"{field_unit_label(name)}"))
            except Exception:
                pass
    sfr = raw("StarFormationRate")
    if sfr is not None and float(sfr) > 0:
        rows.append(("SFR", f"{float(sfr):.3g} Msun/yr"))
    vel = raw("Velocities")
    if vel is not None:
        v = np.asarray(vel, dtype=np.float64) * units.velocity_to_kms
        rows.append(("|v|", f"{np.linalg.norm(v):,.1f} km/s"))
    pid = raw("ParticleIDs")
    if pid is not None:
        rows.append(("ID", str(int(pid))))
    return rows


# ---------------------------------------------------------------------------
# Region centering (aperture-aware)
# ---------------------------------------------------------------------------

def shrinking_sphere_center(positions, masses, center, radius,
                            shrink_factor=0.9, min_particles=500):
    """Power et al. (2003) / pynbody-style shrinking-sphere center.

    Iteratively recomputes the center of mass inside a sphere whose
    radius shrinks by `shrink_factor` per iteration, until fewer than
    `min_particles` remain (or the radius collapses). Robust against
    substructure that biases a plain center of mass.

    All inputs in code units; returns (3,) float64 center.
    """
    center = np.asarray(center, dtype=np.float64).copy()
    radius = float(radius)
    pos = positions
    m = np.asarray(masses, dtype=np.float64)
    idx = np.arange(len(pos))
    for _ in range(200):
        d = np.linalg.norm(pos[idx] - center[None, :], axis=1)
        keep = d <= radius
        if keep.sum() < max(min_particles, 8):
            break
        idx = idx[keep]
        mw = m[idx]
        center = (pos[idx] * mw[:, None]).sum(axis=0) / mw.sum()
        radius *= shrink_factor
    return center


def find_center_in_region(data, center, radius_code, mode="densest"):
    """Refined center within a sphere (code units).

    Modes: 'densest' (peak Density / SubfindDensity), 'potential'
    (potential minimum), 'shrinking' (shrinking-sphere COM), 'com'
    (plain center of mass). Falls back gracefully when the needed
    field is missing. Returns (3,) float64 code-unit center.
    """
    center = np.asarray(center, dtype=np.float64)
    pos = data.positions
    d = np.linalg.norm(pos - center[None, :], axis=1)
    inside = d <= float(radius_code)
    if not inside.any():
        return center
    idx = np.flatnonzero(inside)

    if mode == "shrinking":
        return shrinking_sphere_center(pos, data.masses, center, radius_code)
    if mode == "com":
        m = data.masses[idx].astype(np.float64)
        return (pos[idx] * m[:, None]).sum(axis=0) / m.sum()

    field = None
    avail = set(data.available_fields())
    if mode == "potential" and "Potential" in avail:
        vals = np.asarray(data.get_field("Potential"))[idx]
        return pos[idx[int(np.argmin(vals))]].astype(np.float64)
    for name in ("Density", "SubfindDensity", "SubfindDMDensity"):
        if name in avail:
            field = name
            break
    if field is not None:
        vals = np.asarray(data.get_field(field))[idx]
        return pos[idx[int(np.argmax(vals))]].astype(np.float64)
    # Last resort: mass-weighted shrinking sphere.
    return shrinking_sphere_center(pos, data.masses, center, radius_code)


CENTER_MODES = ["densest", "potential", "shrinking", "com"]


# ---------------------------------------------------------------------------
# Radial profiles
# ---------------------------------------------------------------------------

# Gravitational constant in kpc (km/s)^2 / Msun
G_KPC_KMS2_MSUN = 4.30091e-6

# Profile quantities computed from geometry/kinematics rather than a
# per-particle scalar field.
SPECIAL_PROFILES = ("Density", "EnclosedMass", "RotationCurve",
                    "VelocityDispersion3D")


def radial_profile(data, field, center=None, r_min_kpc=None, r_max_kpc=None,
                   n_bins=40, max_samples=4_000_000):
    """Mass-weighted radial profile of `field` about `center`.

    Special quantities (SPECIAL_PROFILES):
      Density              — true shell density (Msun/kpc^3)
      EnclosedMass         — M(<r) (Msun) over the loaded particle pool
      RotationCurve        — v_c = sqrt(G M(<r)/r) (km/s); note this
                             uses only the loaded particle types
      VelocityDispersion3D — mass-weighted 3D velocity dispersion about
                             the mean shell velocity (km/s)

    `r_max_kpc` doubles as the aperture clamp: when given, only
    particles inside it contribute.

    Returns:
        (r_kpc, prof, unit_label) — bin centers (geometric), profile
        values, and a y-axis unit string.
    """
    units = UnitSystem(data.header)
    if center is None:
        center = data.get_view_center()
    center = np.asarray(center, dtype=np.float64)

    pos = data.positions
    n = len(pos)
    if n == 0:
        return np.zeros(0), np.zeros(0), ""
    if n > max_samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, size=max_samples, replace=False)
        frac = n / max_samples
    else:
        sel = slice(None)
        frac = 1.0

    r_code = np.linalg.norm(pos[sel] - center[None, :], axis=1)
    r_kpc = r_code * units.length_to_kpc
    mass = data.masses[sel].astype(np.float64) * units.mass_to_msun * frac

    if r_max_kpc is None:
        r_max_kpc = float(np.percentile(r_kpc, 99.0))
    if r_min_kpc is None:
        r_min_kpc = max(r_max_kpc * 1e-3, float(np.percentile(r_kpc, 0.1)))
    edges = np.geomspace(max(r_min_kpc, 1e-6), r_max_kpc, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    which = np.digitize(r_kpc, edges) - 1
    ok = (which >= 0) & (which < n_bins)

    msum = np.bincount(which[ok], weights=mass[ok], minlength=n_bins)
    if field == "Density":
        vol = 4.0 / 3.0 * np.pi * np.diff(edges**3)
        prof = msum / vol
        unit = "Msun/kpc^3"
    elif field in ("EnclosedMass", "RotationCurve"):
        # Include mass interior to the first bin edge as well.
        m_inner = mass[r_kpc < edges[0]].sum()
        enc = m_inner + np.cumsum(msum)
        if field == "EnclosedMass":
            prof = enc
            unit = "Msun"
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                prof = np.sqrt(G_KPC_KMS2_MSUN * enc / edges[1:])
            unit = "km/s"
            # v_c is evaluated at the outer bin edge.
            centers = edges[1:]
        return centers, prof, unit
    elif field == "VelocityDispersion3D":
        vel = (np.asarray(data.get_vector_field("Velocities"),
                          dtype=np.float64)[sel] * units.velocity_to_kms)
        wsum = msum
        vmean = np.empty((n_bins, 3))
        for k in range(3):
            s = np.bincount(which[ok], weights=(mass * vel[:, k])[ok],
                            minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                vmean[:, k] = s / wsum
        dv2 = np.zeros(len(r_kpc))
        vm = np.where(np.isfinite(vmean), vmean, 0.0)
        for k in range(3):
            dv2 += (vel[:, k] - vm[np.clip(which, 0, n_bins - 1), k]) ** 2
        s2 = np.bincount(which[ok], weights=(mass * dv2)[ok], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = np.sqrt(s2 / wsum)
        unit = "km/s"
        prof[wsum == 0] = np.nan
        return centers, prof, unit
    else:
        vals = np.asarray(data.get_field(field), dtype=np.float64)[sel]
        wsum = np.bincount(which[ok], weights=(mass * vals)[ok], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = wsum / msum
        unit = field_unit_label(field)
    prof[msum == 0] = np.nan
    return centers, prof, unit


# ---------------------------------------------------------------------------
# Phase diagrams
# ---------------------------------------------------------------------------

PHASE_PRESETS = [
    ("NumberDensity", "Temperature"),
    ("NumberDensity", "Pressure"),
    ("Temperature", "MetallicityZsun"),
    ("RadiusFromCenter", "Temperature"),
    ("RadiusFromCenter", "RadialVelocity"),
]


def available_phase_presets(data):
    fields = set(data.available_fields_with_derived())
    return [(x, y) for x, y in PHASE_PRESETS if x in fields and y in fields]


def phase_histogram(data, xfield, yfield, n_bins=128, max_samples=4_000_000,
                    center=None, radius_kpc=None):
    """Mass-weighted 2D histogram of two fields.

    Log-scales an axis automatically when its values are all-positive
    and span more than 2.5 decades. When `center` (code units) and
    `radius_kpc` are given, only particles inside that sphere
    contribute (aperture scope).

    Returns dict with H (n_bins x n_bins, mass in Msun), x/y edges,
    xlog/ylog flags, and axis labels.
    """
    units = UnitSystem(data.header)
    n = data.n_particles
    if n == 0:
        return None
    if n > max_samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, size=max_samples, replace=False)
        frac = n / max_samples
    else:
        sel = np.arange(n)
        frac = 1.0

    if center is not None and radius_kpc is not None:
        r_code = radius_kpc / max(units.length_to_kpc, 1e-30)
        d = np.linalg.norm(
            data.positions[sel] - np.asarray(center, dtype=np.float64)[None, :],
            axis=1)
        sel = sel[d <= r_code]
        if sel.size == 0:
            return None

    x = np.asarray(data.get_field(xfield), dtype=np.float64)[sel]
    y = np.asarray(data.get_field(yfield), dtype=np.float64)[sel]
    w = data.masses[sel].astype(np.float64) * units.mass_to_msun * frac

    def prep(v):
        finite = np.isfinite(v)
        pos = v > 0
        if pos[finite].all() and finite.any():
            lo, hi = v[finite].min(), v[finite].max()
            if lo > 0 and hi / max(lo, 1e-300) > 10**2.5:
                return np.log10(np.where(pos, v, np.nan)), True
        return v, False

    xv, xlog = prep(x)
    yv, ylog = prep(y)
    ok = np.isfinite(xv) & np.isfinite(yv)
    if not ok.any():
        return None
    xv, yv, w = xv[ok], yv[ok], w[ok]

    def edges(v):
        lo, hi = np.percentile(v, [0.05, 99.95])
        if hi <= lo:
            hi = lo + 1.0
        pad = (hi - lo) * 0.03
        return np.linspace(lo - pad, hi + pad, n_bins + 1)

    xe, ye = edges(xv), edges(yv)
    H, _, _ = np.histogram2d(xv, yv, bins=(xe, ye), weights=w)

    def label(name, is_log):
        u = field_unit_label(name)
        s = f"{name} [{u}]" if u else name
        return f"log10 {s}" if is_log else s

    return {
        "H": H, "xedges": xe, "yedges": ye, "xlog": xlog, "ylog": ylog,
        "xlabel": label(xfield, xlog), "ylabel": label(yfield, ylog),
        "xfield": xfield, "yfield": yfield,
    }


# ---------------------------------------------------------------------------
# Region statistics
# ---------------------------------------------------------------------------

def region_stats(data, center=None, radius_kpc=None):
    """Aggregate physical properties of a sphere around `center`.

    Returns a list of (label, value-string) rows ready for display.
    """
    units = UnitSystem(data.header)
    if center is None:
        center = data.get_view_center()
    center = np.asarray(center, dtype=np.float64)

    pos = data.positions
    if len(pos) == 0:
        return [("Particles", "0")], radius_kpc or 0.0
    r_kpc = np.linalg.norm(pos - center[None, :], axis=1) * units.length_to_kpc
    if radius_kpc is None:
        radius_kpc = float(np.percentile(r_kpc, 50.0))
    inside = r_kpc <= radius_kpc
    n_in = int(inside.sum())

    rows = [("Radius", f"{radius_kpc:,.1f} kpc"),
            ("Particles", f"{n_in:,}")]
    if n_in == 0:
        return rows, radius_kpc

    mass = data.masses.astype(np.float64) * units.mass_to_msun
    m_in = mass[inside]
    m_tot = float(m_in.sum())
    rows.append(("Total mass", f"{m_tot:.3e} Msun"))

    # Per-ptype masses
    for p, sl in sorted(getattr(data, "_type_slices", {}).items()):
        sel = inside[sl]
        if sel.any():
            mp = float(mass[sl][sel].sum())
            lbl = data.ptype_labels.get(p, f"type{p}")
            rows.append((f"  {lbl}", f"{mp:.3e} Msun"))

    # Half-mass radius
    order = np.argsort(r_kpc[inside])
    cm = np.cumsum(m_in[order])
    i_half = int(np.searchsorted(cm, 0.5 * m_tot))
    r_half = float(r_kpc[inside][order][min(i_half, n_in - 1)])
    rows.append(("Half-mass r", f"{r_half:,.1f} kpc"))

    fields = set(data.available_fields())
    def mw_mean(vals):
        return float((m_in * vals[inside]).sum() / m_tot)

    if "StarFormationRate" in fields:
        sfr = np.asarray(data.get_field("StarFormationRate"), dtype=np.float64)
        rows.append(("SFR", f"{float(sfr[inside].sum()):.3g} Msun/yr"))
    try:
        if "Temperature" in data.available_fields_with_derived():
            t = np.asarray(data.get_field("Temperature"), dtype=np.float64)
            rows.append(("<T> (mass-wtd)", f"{mw_mean(t):.3g} K"))
    except Exception:
        pass
    try:
        if "MetallicityZsun" in data.available_fields_with_derived():
            z = np.asarray(data.get_field("MetallicityZsun"), dtype=np.float64)
            rows.append(("<Z>", f"{mw_mean(z):.3g} Zsun"))
    except Exception:
        pass
    try:
        if "RadialVelocity" in data.available_fields_with_derived():
            vr = np.asarray(data.get_field("RadialVelocity"), dtype=np.float64)
            rows.append(("<v_r>", f"{mw_mean(vr):+.1f} km/s"))
    except Exception:
        pass
    return rows, radius_kpc


# ---------------------------------------------------------------------------
# Scale bar helper
# ---------------------------------------------------------------------------

def nice_scale_bar(target_kpc):
    """Largest 1/2/5 x 10^n kpc value <= target. Returns (value, label)."""
    if target_kpc <= 0 or not np.isfinite(target_kpc):
        return 1.0, "1 kpc"
    exp = np.floor(np.log10(target_kpc))
    base = target_kpc / 10**exp
    for m in (5.0, 2.0, 1.0):
        if base >= m:
            val = m * 10**exp
            break
    else:
        val = 10**exp
    if val >= 1000.0:
        return val, f"{val/1000:g} Mpc"
    if val >= 1.0:
        return val, f"{val:g} kpc"
    return val, f"{val*1000:g} pc"
