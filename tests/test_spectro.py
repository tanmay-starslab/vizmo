"""Sightline column-density tests."""

import numpy as np
import h5py
import pytest

from .test_physics_analysis import cosmo_snapshot, cosmo_data  # fixtures


def test_column_kernel_normalization():
    """∫ c(q) 2 pi q dq over the kernel support must equal 1 (the 3D
    kernel integrates to unity, so the projected kernel must too)."""
    from vizmo.spectro import column_kernel

    q = np.linspace(0, 1, 20001)
    c = column_kernel(q)
    integral = np.trapezoid(c * 2 * np.pi * q, q)
    assert integral == pytest.approx(1.0, rel=2e-3)


def test_single_particle_column(tmp_path, cache_isolation):
    """Sightline through one particle's center: N_H = m X_H/m_p * c(0)/h^2."""
    from vizmo.data_manager import SnapshotData
    from vizmo.spectro import (Sightline, compute_los_column_densities,
                               column_kernel)
    from vizmo.physics import UnitSystem, PROTONMASS_CGS, XH_DEFAULT

    path = str(tmp_path / "one.hdf5")
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = 100.0
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["NumPart_ThisFile"] = np.array([1, 0, 0, 0, 0, 0])
        h.attrs["Time"] = 1.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["UnitLength_in_cm"] = 3.085678e21
        h.attrs["UnitMass_in_g"] = 1.989e43
        h.attrs["UnitVelocity_in_cm_per_s"] = 1e5
        g = f.create_group("PartType0")
        g["Coordinates"] = np.array([[50.0, 50.0, 50.0]])
        g["Masses"] = np.array([1e-4], dtype=np.float32)
        g["SmoothingLength"] = np.array([2.0], dtype=np.float32)
        g["NeutralHydrogenAbundance"] = np.array([1.0], dtype=np.float32)
    d = SnapshotData(path, particle_types=[0])
    d.set_view_center(np.array([50.0, 50.0, 50.0]))
    units = UnitSystem(d.header)

    sl = Sightline(start=np.array([50.0, 50.0, 0.0]),
                   end=np.array([50.0, 50.0, 100.0]))
    compute_los_column_densities(sl, d)
    h_cm = 2.0 * units.unit_length_cgs
    m_g = 1e-4 * units.unit_mass_cgs
    expected = m_g * XH_DEFAULT / PROTONMASS_CGS * float(
        column_kernel(np.array([0.0]))[0]) / h_cm**2
    assert sl.N_total_H == pytest.approx(expected, rel=1e-6)
    assert sl.NHI == pytest.approx(expected, rel=1e-6)  # fully neutral
    d.close()


def test_missed_sightline_zero(cosmo_data):
    from vizmo.spectro import Sightline, compute_los_column_densities

    # The fixture sphere spans [40, 60]; aim far away.
    sl = Sightline(start=np.array([0.0, 0.0, 0.0]),
                   end=np.array([0.0, 0.0, 100.0]))
    compute_los_column_densities(sl, cosmo_data)
    assert sl.N_total_H == 0.0


def test_cie_fraction_peaks():
    from vizmo.spectro import cie_ion_fraction, CIE_IONS

    for ion, p in CIE_IONS.items():
        t_peak = 10 ** p["logT_peak"]
        f = cie_ion_fraction(ion, np.array([t_peak]))
        assert f[0] == pytest.approx(p["f_peak"], rel=1e-6)
        # Far from the peak the fraction collapses.
        assert cie_ion_fraction(ion, np.array([1e8]))[0] < 0.01 * p["f_peak"]


def test_sightline_through_cosmo_sphere(cosmo_data, tmp_path):
    from vizmo.spectro import (Sightline, compute_los_column_densities,
                               sightlines_to_csv)

    sl = Sightline(start=np.array([50.0, 50.0, 0.0]),
                   end=np.array([50.0, 50.0, 100.0]), label="SL-T")
    compute_los_column_densities(sl, cosmo_data)
    assert sl.N_total_H > 0
    assert sl.impact_b_kpc == pytest.approx(0.0, abs=1e-6)
    # CIV peaks near logT 5.0; fixture gas is 4.7e4 K, so CIV ~ small
    # but nonzero, OVI ~ negligible.
    assert sl.N_CIV > sl.N_OVI
    out = sightlines_to_csv(str(tmp_path / "sl.csv"), [sl])
    assert "SL-T" in open(out).read()
