"""Phase-diagram brushing tests (Item 1/2)."""

import numpy as np
import pytest

from .test_physics_analysis import cosmo_snapshot, cosmo_data  # fixtures


def _phase(data):
    from vizmo.analysis import phase_histogram

    return phase_histogram(data, "NumberDensity", "Temperature",
                           n_bins=32)


def test_rectangle_brush_selects_bounds(cosmo_data):
    from vizmo.analysis import phase_selection_mask, _phase_display_values

    ph = _phase(cosmo_data)
    x, y = _phase_display_values(cosmo_data, ph)
    # Fixture is single-valued in (n, T): a rect containing that point
    # selects everything; a rect away from it selects nothing.
    x0, y0 = np.nanmedian(x), np.nanmedian(y)
    m = phase_selection_mask(cosmo_data, ph, {
        "kind": "rect", "x0": x0 - 0.5, "x1": x0 + 0.5,
        "y0": y0 - 0.5, "y1": y0 + 0.5})
    assert m.dtype == bool and len(m) == cosmo_data.n_particles
    assert m.all()
    m2 = phase_selection_mask(cosmo_data, ph, {
        "kind": "rect", "x0": x0 + 2, "x1": x0 + 3,
        "y0": y0, "y1": y0 + 1})
    assert not m2.any()


def test_polygon_brush_triangle(cosmo_data):
    from vizmo.analysis import (phase_selection_mask,
                                _phase_display_values, point_in_polygon)

    ph = _phase(cosmo_data)
    x, y = _phase_display_values(cosmo_data, ph)
    cx, cy = np.nanmedian(x), np.nanmedian(y)
    tri_in = [(cx - 1, cy - 1), (cx + 1, cy - 1), (cx, cy + 1)]
    m = phase_selection_mask(cosmo_data, ph,
                             {"kind": "polygon", "points": tri_in})
    assert m.all()
    tri_out = [(cx + 5, cy), (cx + 6, cy), (cx + 5.5, cy + 1)]
    m2 = phase_selection_mask(cosmo_data, ph,
                              {"kind": "polygon", "points": tri_out})
    assert not m2.any()
    # Pure geometry check.
    inside = point_in_polygon([0.0], [0.25], [(-1, 0), (1, 0), (0, 1)])
    outside = point_in_polygon([2.0], [2.0], [(-1, 0), (1, 0), (0, 1)])
    assert inside[0] and not outside[0]


def test_ellipse_brush(cosmo_data):
    from vizmo.analysis import phase_selection_mask, _phase_display_values

    ph = _phase(cosmo_data)
    x, y = _phase_display_values(cosmo_data, ph)
    cx, cy = np.nanmedian(x), np.nanmedian(y)
    m = phase_selection_mask(cosmo_data, ph, {
        "kind": "ellipse", "cx": cx, "cy": cy, "rx": 0.5, "ry": 0.5})
    assert m.all()


def test_inset_map_nonzero(cosmo_data):
    from vizmo.analysis import brush_inset_histogram

    mask = np.zeros(cosmo_data.n_particles, dtype=bool)
    mask[:500] = True
    H = brush_inset_histogram(cosmo_data, mask, res=16)
    assert H.shape == (16, 16)
    assert H.sum() == 500


def test_marginals_conserve_weight(cosmo_data):
    from vizmo.analysis import phase_marginals

    ph = _phase(cosmo_data)
    mx, my = phase_marginals(ph, n_bins=16)
    assert len(mx) == 16 and len(my) == 16
    assert mx.max() == pytest.approx(1.0)
    assert my.max() == pytest.approx(1.0)


def test_brush_mask_in_profile_and_stats(cosmo_data):
    from vizmo.analysis import radial_profile, region_stats

    mask = np.zeros(cosmo_data.n_particles, dtype=bool)
    mask[::2] = True
    rows_all, _ = region_stats(cosmo_data, radius_kpc=999.0)
    rows_b, _ = region_stats(cosmo_data, radius_kpc=999.0,
                             brush_mask=mask)
    n_all = int(dict(rows_all)["Particles"].replace(",", ""))
    n_b = int(dict(rows_b)["Particles"].replace(",", ""))
    assert n_b == n_all // 2
    r, prof, unit = radial_profile(cosmo_data, "Density",
                                   brush_mask=mask, n_bins=8)
    assert np.isfinite(prof).any()


def test_tcool_tff_locus_physical():
    from vizmo.analysis import tcool_tff_unity_locus

    logn, logT = tcool_tff_unity_locus()
    ok = np.isfinite(logT)
    assert ok.sum() > 30
    # The threshold temperature rises with density (denser gas cools
    # faster, so unity requires hotter gas).
    d = np.diff(logT[ok])
    assert np.median(d) > 0
    # At CGM densities the threshold sits in the 1e4-1e8 K band.
    mid = logT[ok][len(logT[ok]) // 2]
    assert 4.0 < mid < 8.0


def test_nfw_density_fit_recovery():
    from vizmo.analysis import fit_nfw_density_profile

    rho_s, r_s = 5e6, 20.0
    r = np.geomspace(1, 300, 40)
    x = r / r_s
    rho = rho_s / (x * (1 + x) ** 2)
    fs, fr = fit_nfw_density_profile(r, rho)
    assert fs == pytest.approx(rho_s, rel=0.02)
    assert fr == pytest.approx(r_s, rel=0.02)
