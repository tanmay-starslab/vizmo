"""3D power spectrum of particle fields via CIC deposition + FFT.

Pure numpy; headless-testable. Computes P(k) of the density contrast
delta = rho/<rho> - 1 (or of any mass-weighted field) inside a cubic
region, shell-averaged into 1D.
"""

import numpy as np

from .physics import UnitSystem


def cic_deposit(pos, weights, center, half_size, n_grid):
    """Cloud-in-cell deposit of `weights` onto an n_grid^3 mesh.

    `pos` (N,3), `center` (3,), `half_size` scalar — all in the same
    (code) units. Particles outside the cube are dropped. Returns the
    (n_grid, n_grid, n_grid) deposited array.
    """
    pos = np.asarray(pos, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    size = 2.0 * float(half_size)
    # Fractional grid coordinates in [0, n_grid)
    u = (pos - (center - half_size)[None, :]) / size * n_grid
    ok = np.all((u >= 0) & (u < n_grid), axis=1)
    u = u[ok]
    w = np.asarray(weights, dtype=np.float64)[ok]

    grid = np.zeros((n_grid, n_grid, n_grid))
    # CIC: split each particle over the 8 surrounding cells.
    i0 = np.floor(u - 0.5).astype(np.int64)
    f = (u - 0.5) - i0  # weight toward the upper cell
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1.0 - f[:, 0]
        ix = np.clip(i0[:, 0] + dx, 0, n_grid - 1)
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            iy = np.clip(i0[:, 1] + dy, 0, n_grid - 1)
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                iz = np.clip(i0[:, 2] + dz, 0, n_grid - 1)
                np.add.at(grid, (ix, iy, iz), w * wx * wy * wz)
    return grid


def power_spectrum(data, center=None, radius_kpc=None, n_grid=128,
                   field="Masses", n_kbins=32, max_samples=8_000_000):
    """Shell-averaged P(k) of the density contrast inside a cube.

    The cube has half-size = radius_kpc (aperture radius) about
    `center`. For field == "Masses" the transform is of
    delta = rho/<rho> - 1; for other fields, of the mass-weighted
    field contrast f/<f> - 1 on the mesh.

    Returns dict(k=..., pk=..., n_modes=..., k_unit="1/kpc",
    pk_unit="kpc^3", n_grid=..., box_kpc=...). k is the geometric bin
    center; the k range spans the fundamental mode to Nyquist.
    """
    units = UnitSystem(data.header)
    if center is None:
        center = data.get_view_center()
    center = np.asarray(center, dtype=np.float64)
    n = data.n_particles
    if n == 0:
        return None
    if radius_kpc is None:
        r_all = (np.linalg.norm(data.positions - center[None, :], axis=1)
                 * units.length_to_kpc)
        radius_kpc = float(np.percentile(r_all, 50.0))
    half_code = radius_kpc / max(units.length_to_kpc, 1e-30)

    if n > max_samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, size=max_samples, replace=False)
        frac = n / max_samples
    else:
        sel = slice(None)
        frac = 1.0
    pos = data.positions[sel]
    m = data.masses[sel].astype(np.float64) * frac

    grid_m = cic_deposit(pos, m, center, half_code, n_grid)
    if field in (None, "Masses"):
        mean = grid_m.mean()
        if mean <= 0:
            return None
        delta = grid_m / mean - 1.0
    else:
        vals = np.asarray(data.get_field(field), dtype=np.float64)[sel]
        grid_mf = cic_deposit(pos, m * vals, center, half_code, n_grid)
        with np.errstate(invalid="ignore", divide="ignore"):
            fgrid = np.where(grid_m > 0, grid_mf / grid_m, 0.0)
        mean = fgrid[grid_m > 0].mean() if (grid_m > 0).any() else 0.0
        if mean == 0:
            return None
        delta = fgrid / mean - 1.0

    box_kpc = 2.0 * radius_kpc
    dk = np.fft.fftn(delta)
    # P(k) = |delta_k|^2 V / N^6 with the numpy FFT convention
    pk3d = np.abs(dk) ** 2 * (box_kpc**3) / n_grid**6

    kf = 2.0 * np.pi / box_kpc
    freqs = np.fft.fftfreq(n_grid, d=1.0 / n_grid) * kf
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)

    k_ny = kf * n_grid / 2
    edges = np.geomspace(kf, k_ny, n_kbins + 1)
    kcen = np.sqrt(edges[:-1] * edges[1:])
    which = np.digitize(kmag.ravel(), edges) - 1
    ok = (which >= 0) & (which < n_kbins) & (kmag.ravel() > 0)
    psum = np.bincount(which[ok], weights=pk3d.ravel()[ok], minlength=n_kbins)
    cnt = np.bincount(which[ok], minlength=n_kbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        pk = psum / cnt
    return {
        "k": kcen, "pk": pk, "n_modes": cnt,
        "k_unit": "1/kpc", "pk_unit": "kpc^3",
        "n_grid": n_grid, "box_kpc": box_kpc, "field": field or "Masses",
    }


def power_spectrum_to_csv(path, ps):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"k [{ps['k_unit']}]", f"P(k) [{ps['pk_unit']}]",
                    "n_modes"])
        for k, p, nm in zip(ps["k"], ps["pk"], ps["n_modes"]):
            w.writerow([f"{k:.6g}", f"{p:.6g}", int(nm)])
    return path
