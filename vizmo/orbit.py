"""Orbit integration in potentials built from the loaded snapshot.

Builds a gravitational potential (NFW fit to the enclosed-mass profile,
or pytreegrav direct summation when installed), integrates single or
ensemble orbits with scipy's DOP853, derives orbital properties
(apo/peri, eccentricity, period, circularity), and generates mock tidal
streams via Jacobi-radius tracer release.

Everything works in physical units: positions in kpc, velocities in
km/s, times in Gyr, masses in Msun.
"""

from __future__ import annotations

import numpy as np

from .analysis import G_KPC_KMS2_MSUN

# 1 km/s = 1.0227e-3 kpc/Gyr... precisely: 1 km/s * 1 Gyr in kpc
KMS_GYR_TO_KPC = 1.02271217  # kpc per (km/s * Gyr)


# ---------------------------------------------------------------------------
# Potential construction
# ---------------------------------------------------------------------------

def _nfw_potential_factory(rho_s: float, r_s: float):
    """Analytic NFW potential phi(r) in (km/s)^2.

    phi(r) = -4 pi G rho_s r_s^3 ln(1 + r/r_s) / r, with the r -> 0
    limit -4 pi G rho_s r_s^2.
    """
    pref = 4.0 * np.pi * G_KPC_KMS2_MSUN * rho_s * r_s**3

    def phi(xyz):
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        r = np.linalg.norm(xyz, axis=1)
        out = np.where(
            r > 1e-8 * r_s,
            -pref * np.log(1.0 + r / r_s) / np.maximum(r, 1e-12),
            -4.0 * np.pi * G_KPC_KMS2_MSUN * rho_s * r_s**2,
        )
        return out if out.size > 1 else float(out[0])

    phi.kind = "nfw"
    phi.rho_s = rho_s
    phi.r_s = r_s
    return phi


def fit_nfw_to_mass_profile(r_kpc: np.ndarray, m_enc_msun: np.ndarray):
    """Fit (rho_s, r_s) of an NFW profile to an enclosed-mass profile.

    M_NFW(<r) = 4 pi rho_s r_s^3 [ln(1+x) - x/(1+x)], x = r/r_s.
    Fit is least-squares in log10 M. Returns (rho_s [Msun/kpc^3],
    r_s [kpc]).
    """
    from scipy.optimize import curve_fit

    ok = (r_kpc > 0) & (m_enc_msun > 0) & np.isfinite(m_enc_msun)
    r, m = r_kpc[ok], m_enc_msun[ok]
    if len(r) < 5:
        raise ValueError("not enough profile points for an NFW fit")

    def logm(rr, log_rho_s, log_rs):
        rho_s, rs = 10.0**log_rho_s, 10.0**log_rs
        x = rr / rs
        mu = np.log(1.0 + x) - x / (1.0 + x)
        return np.log10(4.0 * np.pi * rho_s * rs**3 *
                        np.maximum(mu, 1e-12))

    p0 = [np.log10(m[-1] / (4 * np.pi * (r[-1] / 5) ** 3)),
          np.log10(r[-1] / 5.0)]
    popt, _ = curve_fit(logm, r, np.log10(m), p0=p0, maxfev=20000)
    return 10.0 ** popt[0], 10.0 ** popt[1]


def build_potential_from_snapshot(
    pos: np.ndarray,
    mass: np.ndarray,
    unit_system,
    method: str = "nfw_fit",
    aperture_region=None,
    center=None,
    theta: float = 0.7,
):
    """Build phi(xyz_kpc) -> (km/s)^2 from snapshot particles.

    Args:
        pos: (N, 3) particle positions in code units.
        mass: (N,) particle masses in code units.
        unit_system: physics.UnitSystem for the snapshot.
        method: "nfw_fit" (analytic NFW fit to M(<r); default),
            "treegrav" (pytreegrav direct/tree summation; requires
            pytreegrav), or "analytic_multipole" (alias of nfw_fit).
        aperture_region: optional SelectionRegion (code units) limiting
            the mass distribution used.
        center: (3,) code-unit center; defaults to the mass-weighted
            mean of the selected particles. The returned phi takes
            positions RELATIVE to this center, in physical kpc.
        theta: Barnes-Hut opening angle (treegrav only).

    Returns:
        callable phi(xyz) with attributes .kind and .center_kpc.
    """
    pos = np.asarray(pos, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    if aperture_region is not None:
        keep = aperture_region.contains(pos)
        pos, m = pos[keep], m[keep]
    if len(pos) < 16:
        raise ValueError("too few particles to build a potential")
    if center is None:
        center = (pos * m[:, None]).sum(axis=0) / m.sum()
    center = np.asarray(center, dtype=np.float64)

    kpc = unit_system.length_to_kpc
    pos_kpc = (pos - center[None, :]) * kpc
    m_msun = m * unit_system.mass_to_msun

    if method == "treegrav":
        try:
            from pytreegrav import Potential
        except ImportError as e:
            raise ImportError(
                "pytreegrav is required for method='treegrav' — "
                "pip install pytreegrav") from e

        def phi(xyz):
            xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
            out = Potential(
                np.vstack([pos_kpc, xyz]),
                np.concatenate([m_msun, np.zeros(len(xyz))]),
                G=G_KPC_KMS2_MSUN, theta=theta,
            )[len(pos_kpc):]
            return out if out.size > 1 else float(out[0])

        phi.kind = "treegrav"
    else:  # nfw_fit / analytic_multipole
        r = np.linalg.norm(pos_kpc, axis=1)
        order = np.argsort(r)
        r_sorted = r[order]
        m_cum = np.cumsum(m_msun[order])
        # 64 log-spaced sample radii spanning the populated range.
        lo = max(np.percentile(r_sorted, 1), 1e-3)
        hi = np.percentile(r_sorted, 99)
        rg = np.geomspace(lo, hi, 64)
        mg = np.interp(rg, r_sorted, m_cum)
        rho_s, r_s = fit_nfw_to_mass_profile(rg, mg)
        phi = _nfw_potential_factory(rho_s, r_s)

    phi.center_kpc = center * kpc
    return phi


# ---------------------------------------------------------------------------
# Orbit integration
# ---------------------------------------------------------------------------

def _gradient(phi, xyz: np.ndarray, eps: float) -> np.ndarray:
    """Central finite-difference gradient of phi, one axis at a time."""
    g = np.empty(3)
    for k in range(3):
        dp = np.zeros(3)
        dp[k] = eps
        g[k] = (phi(xyz + dp) - phi(xyz - dp)) / (2.0 * eps)
    return g


def integrate_orbit(
    pos0: np.ndarray,
    vel0: np.ndarray,
    phi,
    t_end_gyr: float = 2.0,
    n_steps: int = 2000,
    method: str = "DOP853",
    eps_kpc: float | None = None,
) -> dict:
    """Integrate one orbit in the potential phi.

    Args:
        pos0: (3,) initial position in kpc (relative to phi's center).
        vel0: (3,) initial velocity in km/s.
        phi: potential callable (kpc -> (km/s)^2).
        t_end_gyr: integration span in Gyr; negative integrates backward.
        n_steps: output samples.
        method: scipy solve_ivp method.
        eps_kpc: finite-difference step; default 1e-3 * max(|pos0|, 10).

    Returns:
        dict with pos (n,3) kpc, vel (n,3) km/s, t (n,) Gyr, r (n,) kpc,
        E (n,) (km/s)^2, L and Lz (n,) kpc km/s.
    """
    from scipy.integrate import solve_ivp

    pos0 = np.asarray(pos0, dtype=np.float64)
    vel0 = np.asarray(vel0, dtype=np.float64)
    if eps_kpc is None:
        eps_kpc = 1e-3 * max(float(np.linalg.norm(pos0)), 10.0)

    # State y = [x(kpc), v(km/s)]; dx/dt = v * KMS_GYR_TO_KPC,
    # dv/dt = -grad(phi) [ (km/s)^2/kpc ] * (1/KMS_GYR_TO_KPC) per Gyr...
    # Cleaner: work in time units of Gyr with conversion folded in.
    def rhs(t, y):
        x, v = y[:3], y[3:]
        g = _gradient(phi, x, eps_kpc)  # (km/s)^2 / kpc
        return np.concatenate([
            v * KMS_GYR_TO_KPC,          # kpc/Gyr
            -g * KMS_GYR_TO_KPC,         # (km/s)/Gyr
        ])

    t_eval = np.linspace(0.0, t_end_gyr, n_steps)
    sol = solve_ivp(
        rhs, (0.0, t_end_gyr), np.concatenate([pos0, vel0]),
        t_eval=t_eval, method=method, rtol=1e-9, atol=1e-9,
    )
    pos = sol.y[:3].T
    vel = sol.y[3:].T
    r = np.linalg.norm(pos, axis=1)
    pot = np.array([phi(p) for p in pos])
    E = 0.5 * (vel**2).sum(axis=1) + pot
    Lvec = np.cross(pos, vel)
    return {
        "pos": pos, "vel": vel, "t": sol.t, "r": r, "E": E,
        "L": np.linalg.norm(Lvec, axis=1), "Lz": Lvec[:, 2],
    }


def compute_orbital_properties(orbit_result: dict, phi=None) -> dict:
    """Apocenter, pericenter, eccentricity, period, circularity.

    The period is estimated from successive pericenter passages
    (local minima of r); falls back to twice the apo-to-peri travel
    time when fewer than two pericenters are sampled.
    """
    r = orbit_result["r"]
    t = orbit_result["t"]
    r_apo = float(np.max(r))
    r_peri = float(np.min(r))
    ecc = (r_apo - r_peri) / max(r_apo + r_peri, 1e-12)

    # Pericenter passages: local minima below the orbit median.
    interior = (r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])
    minima = np.flatnonzero(interior) + 1
    minima = minima[r[minima] < np.median(r)]
    if len(minima) >= 2:
        T_orb = float(np.mean(np.diff(t[minima])))
    else:
        T_orb = 2.0 * abs(t[int(np.argmax(r))] - t[int(np.argmin(r))])

    E_mean = float(np.mean(orbit_result["E"]))
    Jz = float(np.mean(orbit_result["Lz"]))

    circ = None
    if phi is not None:
        # j_circ(E): solve E = phi(r) + v_c^2/2 with
        # v_c^2 = r dphi/dr on a radial grid.
        rg = np.geomspace(max(r_peri / 10, 1e-2), r_apo * 10, 256)
        pots = np.array([phi(np.array([rr, 0, 0])) for rr in rg])
        dphidr = np.gradient(pots, rg)
        vc2 = np.clip(rg * dphidr, 0, None)
        E_circ = pots + 0.5 * vc2
        j_circ_grid = rg * np.sqrt(vc2)
        order = np.argsort(E_circ)
        j_circ_E = float(np.interp(E_mean, E_circ[order],
                                   j_circ_grid[order]))
        if j_circ_E > 0:
            circ = Jz / j_circ_E
    return {
        "r_apo": r_apo, "r_peri": r_peri, "eccentricity": ecc,
        "T_orb": T_orb, "Jz": Jz, "E_mean": E_mean,
        "circularity": circ,
    }


def integrate_ensemble(
    positions: np.ndarray,
    velocities: np.ndarray,
    phi,
    t_end_gyr: float = 2.0,
    n_steps: int = 500,
    max_particles: int = 200,
) -> list:
    """Integrate many orbits in parallel (thread pool).

    Subsamples to max_particles when more are given. Returns a list of
    integrate_orbit() result dicts.
    """
    from concurrent.futures import ThreadPoolExecutor

    positions = np.atleast_2d(positions)
    velocities = np.atleast_2d(velocities)
    n = len(positions)
    if n > max_particles:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, size=max_particles, replace=False)
        positions, velocities = positions[sel], velocities[sel]

    def one(i):
        return integrate_orbit(positions[i], velocities[i], phi,
                               t_end_gyr=t_end_gyr, n_steps=n_steps)

    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(one, range(len(positions))))


# ---------------------------------------------------------------------------
# Mock tidal stream
# ---------------------------------------------------------------------------

def jacobi_radius(r_kpc: float, m_sat_msun: float, m_enc_msun: float) -> float:
    """r_J = r (m_sat / (3 M(<r)))^(1/3)."""
    return float(r_kpc) * (m_sat_msun / (3.0 * max(m_enc_msun, 1e-12))) ** (1.0 / 3.0)


def generate_mock_stream(
    pos0: np.ndarray,
    vel0: np.ndarray,
    phi,
    m_satellite: float,
    n_tracers: int = 100,
    t_end_gyr: float = 2.0,
    n_steps: int = 500,
    seed: int = 0,
) -> dict:
    """Release tracers around a satellite seed and integrate them.

    Tracers start on a sphere of radius r_J (computed from the
    enclosed mass implied by the potential's circular velocity at
    |pos0|) with isotropic Gaussian velocity offsets of width
    v_circ(r_J location)/3. 'stripping_time' is when each tracer first
    exceeds 2 r_J from the (also integrated) satellite seed orbit.

    Returns dict(positions (n_tracers, n_steps, 3) kpc,
    times (n_steps,) Gyr, stripping_time (n_tracers,) Gyr, r_jacobi).
    """
    rng = np.random.default_rng(seed)
    pos0 = np.asarray(pos0, dtype=np.float64)
    vel0 = np.asarray(vel0, dtype=np.float64)
    r0 = float(np.linalg.norm(pos0))

    # Enclosed mass from the potential's rotation curve: M(<r) =
    # r v_c^2 / G with v_c^2 = r dphi/dr (finite difference).
    eps = max(1e-3 * r0, 1e-3)
    dphidr = (phi(pos0 * (1 + eps / r0)) - phi(pos0 * (1 - eps / r0))) / (2 * eps)
    vc2 = max(r0 * dphidr, 1e-6)
    m_enc = r0 * vc2 / G_KPC_KMS2_MSUN
    r_j = jacobi_radius(r0, m_satellite, m_enc)

    # Satellite seed orbit
    sat = integrate_orbit(pos0, vel0, phi, t_end_gyr=t_end_gyr,
                          n_steps=n_steps)

    sigma_v = np.sqrt(vc2) / 3.0
    dirs = rng.standard_normal((n_tracers, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    tracer_pos0 = pos0[None, :] + r_j * dirs
    tracer_vel0 = vel0[None, :] + sigma_v * rng.standard_normal(
        (n_tracers, 3))

    orbits = integrate_ensemble(tracer_pos0, tracer_vel0, phi,
                                t_end_gyr=t_end_gyr, n_steps=n_steps,
                                max_particles=n_tracers)
    positions = np.stack([o["pos"] for o in orbits])
    times = sat["t"]
    strip = np.full(n_tracers, np.nan)
    for i in range(n_tracers):
        d = np.linalg.norm(positions[i] - sat["pos"], axis=1)
        far = np.flatnonzero(d > 2.0 * r_j)
        if len(far):
            strip[i] = times[far[0]]
    return {"positions": positions, "times": times,
            "stripping_time": strip, "r_jacobi": r_j,
            "satellite_orbit": sat}


def orbit_to_csv(path: str, orbit_result: dict) -> str:
    """Write an orbit trail to CSV."""
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_gyr", "x_kpc", "y_kpc", "z_kpc",
                    "vx_kms", "vy_kms", "vz_kms",
                    "E_km2s2", "L_kpckms", "Lz_kpckms"])
        o = orbit_result
        for i in range(len(o["t"])):
            w.writerow([f"{o['t'][i]:.6g}",
                        *[f"{v:.6g}" for v in o["pos"][i]],
                        *[f"{v:.6g}" for v in o["vel"][i]],
                        f"{o['E'][i]:.6g}", f"{o['L'][i]:.6g}",
                        f"{o['Lz'][i]:.6g}"])
    return path
