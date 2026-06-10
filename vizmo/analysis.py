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
KMS_PER_KPC_TO_PER_YR = 1.0227e-9  # (km/s)/kpc in 1/yr

# Profile quantities computed from geometry/kinematics rather than a
# per-particle scalar field.
SPECIAL_PROFILES = ("Density", "EnclosedMass", "RotationCurve",
                    "VelocityDispersion3D", "AngularMomentum")

# Temperature phases for phase-split profiles (K).
TEMPERATURE_PHASES = [
    ("cold", 0.0, 1e4),
    ("warm", 1e4, 1e5),
    ("warm-hot", 1e5, 1e7),
    ("hot", 1e7, np.inf),
]


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
    elif field == "AngularMomentum":
        # Mass-weighted specific angular momentum |sum m r x v| / sum m
        # per shell, in kpc km/s about `center`.
        vel = (np.asarray(data.get_vector_field("Velocities"),
                          dtype=np.float64)[sel] * units.velocity_to_kms)
        rvec = (pos[sel] - center[None, :]) * units.length_to_kpc
        j = np.cross(rvec, vel)
        comp = np.empty((n_bins, 3))
        for k in range(3):
            s = np.bincount(which[ok], weights=(mass * j[:, k])[ok],
                            minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                comp[:, k] = s / msum
        prof = np.linalg.norm(comp, axis=1)
        unit = "kpc km/s"
        prof[msum == 0] = np.nan
        return centers, prof, unit
    else:
        vals = np.asarray(data.get_field(field), dtype=np.float64)[sel]
        wsum = np.bincount(which[ok], weights=(mass * vals)[ok], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = wsum / msum
        unit = field_unit_label(field)
    prof[msum == 0] = np.nan
    return centers, prof, unit


def radial_profile_by_phase(data, field, center=None, r_min_kpc=None,
                            r_max_kpc=None, n_bins=40):
    """Phase-split radial profiles: one track per temperature phase
    (cold <1e4 K, warm 1e4-1e5, warm-hot 1e5-1e7, hot >1e7).

    Works by temporarily restricting the particle selection via a
    boolean mask on Temperature; returns (r_kpc, {phase: prof}, unit).
    Only valid for mass-weighted field profiles (not special profiles).
    """
    units = UnitSystem(data.header)
    if center is None:
        center = data.get_view_center()
    center = np.asarray(center, dtype=np.float64)
    t = np.asarray(data.get_field("Temperature"), dtype=np.float64)
    pos = data.positions
    r_kpc_all = np.linalg.norm(pos - center[None, :], axis=1) * units.length_to_kpc
    mass = data.masses.astype(np.float64) * units.mass_to_msun
    vals = np.asarray(data.get_field(field), dtype=np.float64)

    if r_max_kpc is None:
        r_max_kpc = float(np.percentile(r_kpc_all, 99.0))
    if r_min_kpc is None:
        r_min_kpc = max(r_max_kpc * 1e-3, float(np.percentile(r_kpc_all, 0.1)))
    edges = np.geomspace(max(r_min_kpc, 1e-6), r_max_kpc, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    which = np.digitize(r_kpc_all, edges) - 1
    ok = (which >= 0) & (which < n_bins)

    tracks = {}
    for name, tlo, thi in TEMPERATURE_PHASES:
        sel = ok & (t >= tlo) & (t < thi)
        msum = np.bincount(which[sel], weights=mass[sel], minlength=n_bins)
        if field == "Density":
            vol = 4.0 / 3.0 * np.pi * np.diff(edges**3)
            prof = msum / vol
        else:
            wsum = np.bincount(which[sel], weights=(mass * vals)[sel],
                               minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                prof = wsum / msum
        prof[msum == 0] = np.nan
        tracks[name] = prof
    unit = ("Msun/kpc^3" if field == "Density" else field_unit_label(field))
    return centers, tracks, unit


def profile_to_csv(path, r_kpc, prof_or_tracks, field, unit):
    """Write a profile (single array or {phase: array}) to CSV."""
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if isinstance(prof_or_tracks, dict):
            names = list(prof_or_tracks)
            w.writerow(["r_kpc"] + [f"{field}_{n} [{unit}]" for n in names])
            for i, r in enumerate(r_kpc):
                w.writerow([f"{r:.6g}"] + [f"{prof_or_tracks[n][i]:.6g}"
                                           for n in names])
        else:
            w.writerow(["r_kpc", f"{field} [{unit}]"])
            for r, p in zip(r_kpc, prof_or_tracks):
                w.writerow([f"{r:.6g}", f"{p:.6g}"])
    return path


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


PHASE_WEIGHTINGS = ["mass", "volume", "SFR", "number"]


def phase_histogram(data, xfield, yfield, n_bins=128, max_samples=4_000_000,
                    center=None, radius_kpc=None, weighting="mass"):
    """Weighted 2D histogram of two fields.

    `weighting`: "mass" (Msun), "volume" (kpc^3, from m/rho), "SFR"
    (Msun/yr), or "number" (raw counts). Log-scales an axis
    automatically when its values are all-positive and span more than
    2.5 decades. When `center` (code units) and `radius_kpc` are given,
    only particles inside that sphere contribute (aperture scope).

    Returns dict with H (n_bins x n_bins), x/y edges, xlog/ylog flags,
    axis labels, and the weighting label.
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
    if weighting == "volume" and "Density" in data.available_fields():
        rho = np.asarray(data.get_field("Density"), dtype=np.float64)[sel]
        m_code = data.masses[sel].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(rho > 0, m_code / rho, 0.0) * units.length_to_kpc**3 * frac
        wlabel = "volume [kpc^3]"
    elif weighting == "SFR" and "StarFormationRate" in data.available_fields():
        w = (np.asarray(data.get_field("StarFormationRate"),
                        dtype=np.float64)[sel] * frac)
        wlabel = "SFR [Msun/yr]"
    elif weighting == "number":
        w = np.full(len(sel), frac)
        wlabel = "count"
    else:
        weighting = "mass"
        w = data.masses[sel].astype(np.float64) * units.mass_to_msun * frac
        wlabel = "mass [Msun]"

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
        "weighting": weighting, "wlabel": wlabel,
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
# Halo properties (virial quantities, structure, kinematics)
# ---------------------------------------------------------------------------

def critical_density_msun_kpc3(units):
    """rho_crit(z) = 3 H(z)^2 / (8 pi G) in Msun/kpc^3."""
    h0 = units.h * 100.0  # km/s/Mpc
    om = units.omega_m if units.omega_m > 0 else 0.3
    ol = units.omega_l
    z = max(units.redshift, 0.0)
    hz2 = h0**2 * (om * (1 + z) ** 3 + ol)  # (km/s/Mpc)^2
    hz2_kpc = hz2 / 1000.0**2  # (km/s/kpc)^2
    return 3.0 * hz2_kpc / (8.0 * np.pi * G_KPC_KMS2_MSUN)


def _nfw_mass_ratio(x):
    """mu(x) = ln(1+x) - x/(1+x)."""
    return np.log(1.0 + x) - x / (1.0 + x)


def fit_nfw_concentration(r_kpc, m_enc, r_vir_kpc, m_vir):
    """Fit c to M(<r)/M_vir = mu(c r/R_vir)/mu(c) by least squares in
    log M. Returns c (float) or None when the fit fails."""
    from scipy.optimize import curve_fit

    ok = (r_kpc > 0.01 * r_vir_kpc) & (r_kpc <= r_vir_kpc) & (m_enc > 0)
    if ok.sum() < 5:
        return None
    x = r_kpc[ok] / r_vir_kpc
    y = np.log10(m_enc[ok] / m_vir)

    def model(xx, c):
        return np.log10(_nfw_mass_ratio(c * xx) / _nfw_mass_ratio(c))

    try:
        popt, _ = curve_fit(model, x, y, p0=[8.0], bounds=(1.0, 100.0))
        return float(popt[0])
    except Exception:
        return None


def halo_properties(data, center=None, radius_kpc=None, max_samples=6_000_000):
    """Structural + kinematic halo properties about `center`.

    Computed over the loaded particle pool (load types 0,1,4 for
    meaningful virial masses):

      M200c/R200c, M500c/R500c   spherical-overdensity masses
      c_NFW                      NFW concentration fit to M(<r)
      lambda_Bullock             spin J / (sqrt(2) M200 V200 R200)
      beta_anisotropy            1 - sigma_t^2 / (2 sigma_r^2)
      f_gas, f_cold, f_baryon    inside min(radius, R200c if found)
      dM/dt at boundary          shell [0.9, 1.0] R mass flux (outflow +)
      D/T (|eps|>0.7 proxy)      circularity fraction when Potential
                                 is available (approximate j_circ(E))

    Returns a list of (label, value-string) rows.
    """
    units = UnitSystem(data.header)
    if center is None:
        center = data.get_view_center()
    center = np.asarray(center, dtype=np.float64)
    n = data.n_particles
    if n == 0:
        return []
    if n > max_samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, size=max_samples, replace=False)
        frac = n / max_samples
    else:
        sel = np.arange(n)
        frac = 1.0

    pos = data.positions[sel]
    mass = data.masses[sel].astype(np.float64) * units.mass_to_msun * frac
    r_kpc = np.linalg.norm(pos - center[None, :], axis=1) * units.length_to_kpc
    if radius_kpc is None:
        radius_kpc = float(np.percentile(r_kpc, 75.0))

    order = np.argsort(r_kpc)
    r_sorted = r_kpc[order]
    m_cum = np.cumsum(mass[order])
    rho_crit = critical_density_msun_kpc3(units)

    rows = []

    def so_mass(delta):
        """Spherical-overdensity radius/mass at delta x rho_crit."""
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_rho = m_cum / (4.0 / 3.0 * np.pi * r_sorted**3)
        target = delta * rho_crit
        # Search outside 10 kpc (avoid resolution-noise crossings).
        valid = r_sorted > 10.0
        if not valid.any():
            return None, None
        below = valid & (mean_rho < target)
        if not below.any():
            return None, None
        i = int(np.argmax(below))  # first crossing below target
        return float(r_sorted[i]), float(m_cum[i])

    r200, m200 = so_mass(200.0)
    r500, m500 = so_mass(500.0)
    if r200 is not None:
        rows.append(("M200c", f"{m200:.3e} Msun"))
        rows.append(("R200c", f"{r200:,.0f} kpc"))
    if r500 is not None:
        rows.append(("M500c", f"{m500:.3e} Msun"))
        rows.append(("R500c", f"{r500:,.0f} kpc"))

    # Working radius for everything below.
    r_use = min(radius_kpc, r200) if r200 is not None else radius_kpc
    inside = r_kpc <= r_use
    if inside.sum() < 16:
        return rows
    m_in = mass[inside]
    m_tot = float(m_in.sum())

    # NFW concentration
    if r200 is not None:
        c = fit_nfw_concentration(r_sorted, m_cum, r200, m200)
        if c is not None:
            rows.append(("c_NFW", f"{c:.1f}"))

    # Kinematics
    vel = (np.asarray(data.get_vector_field("Velocities"),
                      dtype=np.float64)[sel] * units.velocity_to_kms)
    v_in = vel[inside]
    vbulk = (m_in[:, None] * v_in).sum(axis=0) / m_tot
    dv = v_in - vbulk
    rvec = (pos[inside] - center[None, :]) * units.length_to_kpc
    rr = np.linalg.norm(rvec, axis=1)
    rhat = rvec / np.maximum(rr, 1e-12)[:, None]
    vr = (dv * rhat).sum(axis=1)
    sig2_tot = float((m_in * (dv * dv).sum(axis=1)).sum() / m_tot)
    vr_mean = float((m_in * vr).sum() / m_tot)
    sig2_r = float((m_in * (vr - vr_mean) ** 2).sum() / m_tot)
    sig2_t = max(sig2_tot - sig2_r, 0.0)
    if sig2_r > 0:
        beta = 1.0 - sig2_t / (2.0 * sig2_r)
        rows.append(("beta_anisotropy", f"{beta:+.2f}"))

    # Bullock spin (about r_use sphere; uses M200/V200/R200 when found)
    j_tot = np.linalg.norm((m_in[:, None] * np.cross(rvec, dv)).sum(axis=0))
    if r200 is not None and m200 > 0:
        v200 = np.sqrt(G_KPC_KMS2_MSUN * m200 / r200)
        lam = j_tot / (np.sqrt(2.0) * m200 * v200 * r200)
        rows.append(("lambda_spin", f"{lam:.3f}"))

    # Baryon budget (per-type masses over the full pool inside r_use)
    r_all = (np.linalg.norm(data.positions - center[None, :], axis=1)
             * units.length_to_kpc)
    in_all = r_all <= r_use
    mass_all = data.masses.astype(np.float64) * units.mass_to_msun
    m_by_type = {}
    for p, sl in sorted(getattr(data, "_type_slices", {}).items()):
        seln = in_all[sl]
        if seln.any():
            m_by_type[p] = float(mass_all[sl][seln].sum())
    m_gas = m_by_type.get(0, 0.0)
    m_star = m_by_type.get(4, 0.0)
    m_total_all = sum(m_by_type.values())
    if m_gas > 0 and (m_gas + m_star) > 0:
        rows.append(("f_gas", f"{m_gas / (m_gas + m_star):.3f}"))
    if m_gas > 0 and 0 in data.particle_types:
        try:
            t = np.asarray(data.get_field("Temperature"), dtype=np.float64)
            sl0 = data._type_slices[0]
            cold = in_all[sl0] & (t[sl0] < 2e4)
            f_cold = float(mass_all[sl0][cold].sum()) / m_gas
            rows.append(("f_cold(T<2e4K)", f"{f_cold:.3f}"))
        except Exception:
            pass
    omega_b = float(data.header.get("OmegaBaryon", 0) or 0)
    omega_m = units.omega_m
    if m_total_all > 0 and omega_b > 0 and omega_m > 0 and (m_gas + m_star) > 0:
        f_b = (m_gas + m_star) / m_total_all
        rows.append(("f_baryon", f"{f_b:.3f} (cosmic {omega_b/omega_m:.3f})"))

    # Boundary mass flux in the [0.9, 1.0] r_use shell (outflow > 0)
    sh = rr >= 0.9 * r_use
    if sh.sum() > 8:
        dr = 0.1 * r_use
        mdot = float((m_in[sh] * vr[sh]).sum()) * KMS_PER_KPC_TO_PER_YR / dr
        rows.append(("dM/dt boundary", f"{mdot:+.2f} Msun/yr"))

    # Disk-to-total proxy from circularity (needs Potential).
    if "Potential" in set(data.available_fields()):
        try:
            phi = np.asarray(data.get_field("Potential"),
                             dtype=np.float64)[sel][inside]
            e_spec = phi + 0.5 * (dv * dv).sum(axis=1)
            jz = rvec[:, 0] * dv[:, 1] - rvec[:, 1] * dv[:, 0]
            jmag = np.linalg.norm(np.cross(rvec, dv), axis=1)
            nb = 40
            qe = np.quantile(e_spec, np.linspace(0, 1, nb + 1))
            ebin = np.clip(np.searchsorted(qe, e_spec) - 1, 0, nb - 1)
            jcirc = np.ones(nb)
            for b in range(nb):
                m_b = ebin == b
                if m_b.any():
                    jcirc[b] = max(np.percentile(jmag[m_b], 95), 1e-12)
            eps = jz / jcirc[ebin]
            dt = float(m_in[np.abs(eps) > 0.7].sum() / m_tot)
            rows.append(("D/T (|eps|>0.7)", f"{dt:.2f} (approx)"))
        except Exception:
            pass
    return rows


def stats_to_json(path, rows, halo_rows, meta):
    """Write stats + halo properties + metadata to a JSON file."""
    import json

    out = {
        "metadata": meta,
        "region_stats": {k: v for k, v in rows},
        "halo_properties": {k: v for k, v in halo_rows},
    }
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    import os

    os.replace(tmp, path)
    return path


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
