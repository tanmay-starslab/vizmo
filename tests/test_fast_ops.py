"""Numba fast-path equivalence tests (Item 3)."""

import numpy as np
import pytest

from vizmo import fast_ops


def test_los_column_numba_vs_numpy():
    rng = np.random.default_rng(0)
    n = 1000
    pos = rng.uniform(-2, 2, size=(n, 3))
    pos[:, 2] = rng.uniform(-0.5, 0.5, n)  # slab
    vals = rng.random(n)
    h = np.full(n, 0.8)
    a = np.array([0.0, 0.0, -5.0])
    b = np.array([0.0, 0.0, 5.0])
    got = fast_ops.los_column_density(a, b, pos, vals, h)
    ref = fast_ops._los_numpy(a, b, pos, vals, h)
    assert got == pytest.approx(ref, rel=1e-10)
    assert got > 0


def test_radial_profile_numba_vs_numpy():
    rng = np.random.default_rng(1)
    n = 20000
    pos = rng.uniform(-5, 5, size=(n, 3))
    mass = rng.random(n)
    field = rng.standard_normal(n)
    center = np.array([0.3, -0.2, 0.1])
    edges = np.geomspace(0.1, 8.0, 17)
    c1, m1, f1 = fast_ops.radial_profile_bins(center, pos, mass,
                                              field, edges)
    c2, m2, f2 = fast_ops._profile_numpy(center, pos, mass, field,
                                         edges)
    assert (c1 == c2).all()
    assert np.allclose(m1, m2, rtol=1e-10)
    assert np.allclose(f1, f2, rtol=1e-10)


def test_kernel_interp_numba_vs_numpy():
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(2)
    n, m, k = 5000, 50, 16
    pos = rng.uniform(-3, 3, size=(n, 3))
    vals = rng.random(n)
    h = np.full(n, 0.9)
    q = rng.uniform(-2, 2, size=(m, 3))
    d, idx = cKDTree(pos).query(q, k=k)
    got = fast_ops.kernel_weighted_interpolation(q, pos, vals, h,
                                                 idx, d)
    ref = fast_ops._interp_numpy(q, pos, vals, h, idx, d)
    assert np.allclose(got, ref, rtol=1e-8)
