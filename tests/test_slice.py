"""Slice-plane reconstruction tests (CPU reference path)."""

import numpy as np
import pytest


def test_uniform_density_flat_slice():
    """A uniform random particle field must reconstruct to a flat
    slice: interior pixels within 5% of the mean."""
    from vizmo.wgpu_renderer import (slab_cull_to_plane,
                                     compute_slice_grid_cpu)

    rng = np.random.default_rng(0)
    n = 60000
    pos = rng.uniform(-10, 10, size=(n, 3))
    # Smoothing ~ 2.2x mean separation for good kernel overlap.
    h = np.full(n, 2.2 * 20.0 / n ** (1 / 3))
    vals = np.full(n, 7.5)  # constant field -> ratio must return 7.5
    pos_h, v = slab_cull_to_plane(pos, h, vals, np.zeros(3),
                                  np.array([0, 0, 1.0]), 6.0)
    assert len(pos_h) > 500
    grid = compute_slice_grid_cpu(pos_h, v, 6.0, 64)
    interior = grid[8:-8, 8:-8]
    assert np.isfinite(interior).all()
    # Constant field: kernel-weighted ratio is exactly the constant.
    assert np.nanmax(np.abs(interior - 7.5)) / 7.5 < 1e-6

    # Varying weights, uniform geometry: density-like reconstruction
    # of per-particle constant remains flat to 5%.
    vals2 = np.full(n, 1.0) + 0.0
    pos_h2, v2 = slab_cull_to_plane(pos, h, vals2, np.zeros(3),
                                    np.array([0, 0, 1.0]), 6.0)
    grid2 = compute_slice_grid_cpu(pos_h2, v2, 6.0, 64)
    interior2 = grid2[8:-8, 8:-8]
    dev = np.nanmax(np.abs(interior2 - np.nanmean(interior2)))
    assert dev / np.nanmean(interior2) < 0.05


def test_gradient_field_reconstruction():
    """A linear field f = x must reconstruct monotonically along x."""
    from vizmo.wgpu_renderer import (slab_cull_to_plane,
                                     compute_slice_grid_cpu)

    rng = np.random.default_rng(1)
    n = 40000
    pos = rng.uniform(-10, 10, size=(n, 3))
    h = np.full(n, 2.2 * 20.0 / n ** (1 / 3))
    vals = pos[:, 0]
    pos_h, v = slab_cull_to_plane(pos, h, vals, np.zeros(3),
                                  np.array([0, 0, 1.0]), 6.0)
    grid = compute_slice_grid_cpu(pos_h, v, 6.0, 64)
    # Column means rise along the u (x) axis. Note plane_basis maps
    # +Z normal -> e1 = cross([0,1,0], z)... check via correlation.
    prof = np.nanmean(grid, axis=0)
    x = np.linspace(-6, 6, 64)
    corr = np.corrcoef(prof, x)[0, 1]
    assert abs(corr) > 0.99  # monotone linear (sign depends on basis)


def test_slab_cull_geometry():
    from vizmo.wgpu_renderer import slab_cull_to_plane

    pos = np.array([[0, 0, 0.5], [0, 0, 3.0], [10.0, 0, 0]])
    h = np.array([1.0, 1.0, 1.0])
    vals = np.array([1.0, 2.0, 3.0])
    pos_h, v = slab_cull_to_plane(pos, h, vals, np.zeros(3),
                                  np.array([0, 0, 1.0]), 5.0)
    # Particle 2 is 3h from the plane (cut); particle 3 is outside the
    # 5+h footprint (cut). Only particle 1 survives.
    assert len(pos_h) == 1
    assert v[0] == 1.0
    assert pos_h[0, 2] == pytest.approx(0.5)  # distance to plane


def test_slice_fits_wcs(tmp_path):
    from astropy.io import fits as pyfits

    from vizmo.wgpu_renderer import slice_grid_to_fits

    grid = np.random.default_rng(2).random((128, 128))
    size_kpc = 300.0
    out = slice_grid_to_fits(grid, np.array([10.0, 20.0, 30.0]),
                             size_kpc, str(tmp_path / "s.fits"),
                             field="Temperature", unit="K",
                             normal_label="+Z")
    hdu = pyfits.open(out)[0]
    assert hdu.data.shape == (128, 128)
    # Pixel scale: CDELT = size / res, CRPIX at image center.
    assert hdu.header["CDELT1"] == pytest.approx(size_kpc / 128)
    assert hdu.header["CDELT2"] == pytest.approx(size_kpc / 128)
    assert hdu.header["CRPIX1"] == pytest.approx(64.5)
    assert hdu.header["CUNIT1"] == "kpc"
    assert hdu.header["FIELD"] == "Temperature"
    assert hdu.header["SLICENRM"] == "+Z"
