"""Numba-accelerated hot paths (Item 3): kernel interpolation, LOS
column density, radial profile binning.

Every function has a pure-numpy twin with identical math; the public
wrappers dispatch to numba when importable and fall back silently
(one-time warning) otherwise. Equivalence is enforced by
tests/test_fast_ops.py to 1e-8 or better.
"""

from __future__ import annotations

import numpy as np

try:
    import numba

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

_warned = [False]


def _warn_once():
    if not _warned[0]:  # pragma: no cover
        print("  fast_ops: numba unavailable; using numpy fallbacks")
        _warned[0] = True


def _kernel_m4_scalar(u):
    if u >= 1.0:
        return 0.0
    if u < 0.5:
        return (8.0 / np.pi) * (1.0 - 6.0 * u * u + 6.0 * u * u * u)
    t = 1.0 - u
    return (8.0 / np.pi) * 2.0 * t * t * t


# ---------------------------------------------------------------------------
# 1. Kernel-weighted interpolation at query points
# ---------------------------------------------------------------------------

def _interp_numpy(query_pos, part_pos, part_vals, part_hsml,
                  neighbor_idx, neighbor_dist):
    m = len(query_pos)
    scalar = part_vals.ndim == 1
    out = np.zeros(m if scalar else (m, part_vals.shape[1]))
    for i in range(m):
        idx = neighbor_idx[i]
        d = neighbor_dist[i]
        h = part_hsml[idx]
        w = np.array([_kernel_m4_scalar(min(d[k] / h[k], 0.999))
                      for k in range(len(idx))]) + 1e-30
        if scalar:
            out[i] = (part_vals[idx] * w).sum() / w.sum()
        else:
            out[i] = (part_vals[idx] * w[:, None]).sum(axis=0) / w.sum()
    return out


if HAVE_NUMBA:
    @numba.njit(cache=True, inline="always")
    def _k_m4(u):
        if u >= 1.0:
            return 0.0
        if u < 0.5:
            return (8.0 / np.pi) * (1.0 - 6.0 * u * u + 6.0 * u * u * u)
        t = 1.0 - u
        return (8.0 / np.pi) * 2.0 * t * t * t

    @numba.njit(parallel=True, cache=True)
    def kernel_weighted_interp_numba(query_pos, part_pos, part_vals,
                                     part_hsml, neighbor_idx,
                                     neighbor_dist):
        m, k = neighbor_idx.shape
        out = np.zeros(m)
        for i in numba.prange(m):
            num = 0.0
            den = 1e-30
            for j in range(k):
                p = neighbor_idx[i, j]
                u = neighbor_dist[i, j] / part_hsml[p]
                if u > 0.999:
                    u = 0.999
                w = _k_m4(u) + 1e-30 / k
                num += part_vals[p] * w
                den += w
            out[i] = num / den
        return out


def kernel_weighted_interpolation(query_pos, part_pos, part_vals,
                                  part_hsml, neighbor_idx,
                                  neighbor_dist):
    """M4 kernel-weighted interpolation at M query points using
    precomputed K-nearest neighbors. Scalar fields only in the numba
    path; vector fields go through numpy."""
    part_vals = np.asarray(part_vals, dtype=np.float64)
    if HAVE_NUMBA and part_vals.ndim == 1:
        return kernel_weighted_interp_numba(
            np.asarray(query_pos, dtype=np.float64),
            np.asarray(part_pos, dtype=np.float64),
            part_vals,
            np.asarray(part_hsml, dtype=np.float64),
            np.asarray(neighbor_idx, dtype=np.int64),
            np.asarray(neighbor_dist, dtype=np.float64))
    _warn_once()
    return _interp_numpy(query_pos, part_pos, part_vals, part_hsml,
                         neighbor_idx, neighbor_dist)


# ---------------------------------------------------------------------------
# 2. LOS column density (matches spectro.py kernel-column math)
# ---------------------------------------------------------------------------

def _los_numpy(a, b, part_pos, part_vals, part_hsml):
    from .spectro import column_kernel

    ab = b - a
    L2 = float(ab @ ab)
    t = np.clip(((part_pos - a[None, :]) @ ab) / max(L2, 1e-300), 0, 1)
    closest = a[None, :] + t[:, None] * ab[None, :]
    d = np.linalg.norm(part_pos - closest, axis=1)
    q = d / part_hsml
    hit = q < 1.0
    ck = column_kernel(q[hit])
    return float((part_vals[hit] * ck / part_hsml[hit] ** 2).sum())


if HAVE_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def los_column_density_numba(a, b, part_pos, part_vals, part_hsml,
                                 cgrid):
        # cgrid: precomputed column-kernel table on q in [0, 1).
        n = len(part_pos)
        ab0 = b[0] - a[0]
        ab1 = b[1] - a[1]
        ab2 = b[2] - a[2]
        L2 = ab0 * ab0 + ab1 * ab1 + ab2 * ab2
        if L2 < 1e-300:
            L2 = 1e-300
        ng = len(cgrid)
        total = 0.0
        for i in numba.prange(n):
            dx = part_pos[i, 0] - a[0]
            dy = part_pos[i, 1] - a[1]
            dz = part_pos[i, 2] - a[2]
            t = (dx * ab0 + dy * ab1 + dz * ab2) / L2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            cx = a[0] + t * ab0 - part_pos[i, 0]
            cy = a[1] + t * ab1 - part_pos[i, 1]
            cz = a[2] + t * ab2 - part_pos[i, 2]
            d = np.sqrt(cx * cx + cy * cy + cz * cz)
            h = part_hsml[i]
            q = d / h
            if q < 1.0:
                # linear interp into the kernel-column table
                x = q * (ng - 1)
                j = int(x)
                if j >= ng - 1:
                    j = ng - 2
                f = x - j
                ck = cgrid[j] * (1.0 - f) + cgrid[j + 1] * f
                total += part_vals[i] * ck / (h * h)
        return total


def los_column_density(sightline_start, sightline_end, part_pos,
                       part_vals, part_hsml):
    """Kernel-column LOS integral: sum_i vals_i c(q_i) / h_i^2 over
    particles whose kernels intersect the segment. Same math as
    spectro.compute_los_column_densities' inner loop."""
    a = np.asarray(sightline_start, dtype=np.float64)
    b = np.asarray(sightline_end, dtype=np.float64)
    pos = np.asarray(part_pos, dtype=np.float64)
    vals = np.asarray(part_vals, dtype=np.float64)
    h = np.asarray(part_hsml, dtype=np.float64)
    if HAVE_NUMBA:
        from .spectro import _CGRID

        return float(los_column_density_numba(a, b, pos, vals, h,
                                              _CGRID))
    _warn_once()
    return _los_numpy(a, b, pos, vals, h)


# ---------------------------------------------------------------------------
# 3. Radial profile binning
# ---------------------------------------------------------------------------

def _profile_numpy(center, pos, mass, field, r_edges):
    r = np.linalg.norm(pos - center[None, :], axis=1)
    which = np.digitize(r, r_edges) - 1
    nb = len(r_edges) - 1
    ok = (which >= 0) & (which < nb)
    counts = np.bincount(which[ok], minlength=nb).astype(np.int64)
    msum = np.bincount(which[ok], weights=mass[ok], minlength=nb)
    fsum = np.bincount(which[ok], weights=(mass * field)[ok],
                       minlength=nb)
    return counts, msum, fsum


if HAVE_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def radial_profile_bins_numba(center, pos, mass, field, r_edges):
        n = len(pos)
        nb = len(r_edges) - 1
        nt = numba.get_num_threads()
        counts_t = np.zeros((nt, nb), dtype=np.int64)
        msum_t = np.zeros((nt, nb))
        fsum_t = np.zeros((nt, nb))
        for i in numba.prange(n):
            tid = numba.get_thread_id()
            dx = pos[i, 0] - center[0]
            dy = pos[i, 1] - center[1]
            dz = pos[i, 2] - center[2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            # binary search into edges
            lo, hi = 0, nb + 1
            while lo < hi:
                mid = (lo + hi) // 2
                if r_edges[mid] <= r:
                    lo = mid + 1
                else:
                    hi = mid
            b = lo - 1
            if 0 <= b < nb:
                counts_t[tid, b] += 1
                msum_t[tid, b] += mass[i]
                fsum_t[tid, b] += mass[i] * field[i]
        return (counts_t.sum(axis=0), msum_t.sum(axis=0),
                fsum_t.sum(axis=0))


def radial_profile_bins(center, pos, mass, field, r_edges):
    """Per-bin (counts, sum m, sum m*f) for radial profiles."""
    center = np.asarray(center, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    r_edges = np.asarray(r_edges, dtype=np.float64)
    if HAVE_NUMBA:
        return radial_profile_bins_numba(center, pos, mass, field,
                                         r_edges)
    _warn_once()
    return _profile_numpy(center, pos, mass, field, r_edges)
