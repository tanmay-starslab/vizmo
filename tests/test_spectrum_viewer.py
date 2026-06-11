"""Spectrum viewer support tests (Section 3.D)."""

import numpy as np
import pytest


def test_spectrum_fits_parse(tmp_path):
    from astropy.io import fits as pyfits

    from vizmo.spectro import load_spectrum

    v = np.linspace(-600, 600, 200)
    flux = 1.0 - 0.6 * np.exp(-0.5 * (v / 40.0) ** 2)
    hdu = pyfits.PrimaryHDU(np.vstack([v, flux]))
    path = str(tmp_path / "spec.fits")
    hdu.writeto(path)
    w, f = load_spectrum(path)
    assert len(w) == 200 and len(f) == 200
    assert f.min() < 0.5 < f.max()  # absorption trough present
    # Polyline data is non-empty and finite.
    assert np.isfinite(w).all() and np.isfinite(f).all()


def test_spectrum_h5_parse(tmp_path):
    import h5py

    from vizmo.spectro import load_spectrum

    path = str(tmp_path / "spec.h5")
    with h5py.File(path, "w") as f:
        f["wavelength"] = np.linspace(1000, 1600, 100)
        f["flux"] = np.ones(100)
    w, fl = load_spectrum(path)
    assert len(w) == 100 and fl.max() == 1.0


def test_voigt_component_regex():
    from vizmo.spectro import parse_voigtfit_components

    log = """
    Best-fit parameters:
    HI : logN = 14.23 +/- 0.05, b = 25.4 +/- 2.1, v = -45.2
    HI : logN = 13.10 +/- 0.12, b = 18.0 +/- 4.0, v = 102.7
    chi2 = 1.04
    """
    comps = parse_voigtfit_components(log)
    assert comps == [(14.23, 25.4, -45.2), (13.10, 18.0, 102.7)]
    assert parse_voigtfit_components("no fits here") == []


def test_compare_colors_distinct():
    from vizmo.spectro import compare_color

    c0, c1 = compare_color(0), compare_color(1)
    assert c0 != c1
    assert compare_color(6) == c0  # cycles
