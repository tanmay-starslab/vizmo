"""Tests for selection primitives, power spectrum, halo properties,
phase-split profiles, and weighted phase histograms."""

import numpy as np
import h5py
import pytest

from .test_physics_analysis import cosmo_snapshot, cosmo_data  # fixtures
from .conftest import make_snapshot


# ---------------------------------------------------------------------------
# Selection primitives
# ---------------------------------------------------------------------------

def test_selection_contains_analytic():
    from vizmo.selection import (Sphere, Box, Cylinder, Slab, Cone,
                                 Ellipsoid)

    pts = np.array([[0, 0, 0], [0.9, 0, 0], [1.1, 0, 0],
                    [0, 0, 2.0], [0, 1.4, 0]])
    s = Sphere([0, 0, 0], 1.0)
    assert list(s.contains(pts)) == [True, True, False, False, False]
    assert s.volume() == pytest.approx(4 / 3 * np.pi)

    b = Box([-1, -1, -1], [1, 1, 1])
    assert list(b.contains(pts)) == [True, True, False, False, False]
    assert b.volume() == pytest.approx(8.0)

    c = Cylinder([0, 0, 0], [0, 0, 1], radius=1.0, half_height=1.5)
    assert list(c.contains(pts)) == [True, True, False, False, False]
    assert c.volume() == pytest.approx(np.pi * 3.0)

    sl = Slab([0, 0, 0], [0, 0, 1], half_thickness=1.0)
    assert list(sl.contains(pts)) == [True, True, True, False, True]

    co = Cone([0, 0, 0], [1, 0, 0], half_angle_deg=45.0, length=2.0)
    assert bool(co.contains(np.array([[1.0, 0.5, 0]]))[0])
    assert not bool(co.contains(np.array([[1.0, 1.5, 0]]))[0])
    assert not bool(co.contains(np.array([[-1.0, 0, 0]]))[0])

    e = Ellipsoid([0, 0, 0], [2.0, 1.0, 0.5])
    assert bool(e.contains(np.array([[1.9, 0, 0]]))[0])
    assert not bool(e.contains(np.array([[0, 0, 0.6]]))[0])
    assert e.volume() == pytest.approx(4 / 3 * np.pi * 1.0)


def test_selection_composite_and_roundtrip(tmp_path):
    from vizmo.selection import (Sphere, CompositeRegion, SelectionRegion,
                                 save_regions, load_regions)

    rng = np.random.default_rng(0)
    pts = rng.uniform(-2, 2, size=(5000, 3))
    shell = CompositeRegion([(None, Sphere([0, 0, 0], 1.0)),
                             ("subtract", Sphere([0, 0, 0], 0.5))])
    m = shell.contains(pts)
    r = np.linalg.norm(pts, axis=1)
    assert (m == ((r <= 1.0) & (r > 0.5))).all()

    path = str(tmp_path / "regions.json")
    save_regions(path, [shell, Sphere([1, 2, 3], 4.0)])
    loaded = load_regions(path)
    assert len(loaded) == 2
    assert (loaded[0].contains(pts) == m).all()
    assert isinstance(loaded[1], Sphere)


# ---------------------------------------------------------------------------
# Power spectrum
# ---------------------------------------------------------------------------

def test_cic_deposit_conserves_mass():
    from vizmo.power_spectrum import cic_deposit

    rng = np.random.default_rng(1)
    pos = rng.uniform(40, 60, size=(20000, 3))
    w = rng.random(20000)
    grid = cic_deposit(pos, w, np.array([50.0, 50.0, 50.0]), 10.0, 32)
    assert grid.sum() == pytest.approx(w.sum(), rel=1e-10)


def test_power_spectrum_basic(cosmo_data, tmp_path):
    from vizmo.power_spectrum import power_spectrum, power_spectrum_to_csv
    from vizmo.physics import UnitSystem

    units = UnitSystem(cosmo_data.header)
    ps = power_spectrum(cosmo_data, radius_kpc=12 * units.length_to_kpc,
                        n_grid=32, n_kbins=12)
    assert ps is not None
    ok = ps["n_modes"] > 0
    assert ok.sum() >= 8
    assert np.isfinite(ps["pk"][ok]).all()
    assert (ps["pk"][ok] >= 0).all()
    # k range: fundamental to Nyquist
    kf = 2 * np.pi / ps["box_kpc"]
    assert ps["k"][0] == pytest.approx(np.sqrt(kf * kf * 32 / 2) , rel=10)
    out = power_spectrum_to_csv(str(tmp_path / "pk.csv"), ps)
    assert open(out).readline().startswith("k [")


# ---------------------------------------------------------------------------
# Halo properties
# ---------------------------------------------------------------------------

def test_halo_m200_point_mass(tmp_path, cache_isolation):
    """All mass concentrated centrally: R200c satisfies
    M / (4/3 pi R^3) = 200 rho_crit analytically."""
    from vizmo.data_manager import SnapshotData
    from vizmo.analysis import halo_properties, critical_density_msun_kpc3
    from vizmo.physics import UnitSystem

    path = str(tmp_path / "pm.hdf5")
    rng = np.random.default_rng(5)
    n_core, n_halo = 4000, 4000
    core = 50.0 + 0.02 * rng.standard_normal((n_core, 3))
    u = rng.random(n_halo)
    rr = 5.0 + 45.0 * u
    vv = rng.standard_normal((n_halo, 3))
    vv /= np.linalg.norm(vv, axis=1)[:, None]
    halo = 50.0 + rr[:, None] * vv
    pos = np.vstack([core, halo])
    m = np.concatenate([np.full(n_core, 5.0e-5),
                        np.full(n_halo, 1e-12)])
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = 100.0
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["NumPart_ThisFile"] = np.array([len(m), 0, 0, 0, 0, 0])
        h.attrs["Time"] = 1.0
        h.attrs["Redshift"] = 0.0
        h.attrs["HubbleParam"] = 0.6774
        h.attrs["Omega0"] = 0.3089
        h.attrs["OmegaLambda"] = 0.6911
        h.attrs["UnitLength_in_cm"] = 3.085678e21
        h.attrs["UnitMass_in_g"] = 1.989e43
        h.attrs["UnitVelocity_in_cm_per_s"] = 1e5
        g = f.create_group("PartType0")
        g["Coordinates"] = pos
        g["Masses"] = m.astype(np.float32)
        g["SmoothingLength"] = np.full(len(m), 0.5, dtype=np.float32)
        g["Velocities"] = np.zeros((len(m), 3), dtype=np.float32)
    d = SnapshotData(path, particle_types=[0])
    d.set_view_center(np.array([50.0, 50.0, 50.0]))
    units = UnitSystem(d.header)
    rows = dict(halo_properties(d, radius_kpc=80.0))
    assert "M200c" in rows
    m200 = float(rows["M200c"].split()[0])
    r200 = float(rows["R200c"].replace(",", "").split()[0])
    m_central = n_core * 5.0e-5 * units.mass_to_msun
    rho_c = critical_density_msun_kpc3(units)
    r200_expected = (3 * m_central / (800 * np.pi * rho_c)) ** (1 / 3)
    assert m200 == pytest.approx(m_central, rel=0.02)
    assert r200 == pytest.approx(r200_expected, rel=0.1)
    # Zero velocities: spin ~ 0
    if "lambda_spin" in rows:
        assert float(rows["lambda_spin"]) < 0.01
    d.close()


def test_halo_properties_on_uniform_sphere(cosmo_data):
    from vizmo.analysis import halo_properties

    # Radius inside the 14.8-kpc particle sphere so the
    # boundary shell is populated.
    rows = dict(halo_properties(cosmo_data, radius_kpc=12.0))
    # Constant-speed pure radial outflow: both dispersions are float32
    # roundoff noise, so beta is either skipped or ~0 — it must never
    # blow up the way the old sigma_tot-based formula did (-6500).
    if "beta_anisotropy" in rows:
        assert abs(float(rows["beta_anisotropy"])) < 0.2
    assert "dM/dt boundary" in rows
    assert float(rows["dM/dt boundary"].split()[0]) > 0  # outflow positive


# ---------------------------------------------------------------------------
# Phase-split profiles + weighted phase histograms
# ---------------------------------------------------------------------------

def test_profile_by_phase_single_temperature(cosmo_data):
    from vizmo.analysis import radial_profile_by_phase

    # Fixture gas is ~47,000 K -> everything in the "warm" track.
    r, tracks, unit = radial_profile_by_phase(cosmo_data, "Density",
                                              n_bins=10)
    assert set(tracks) == {"cold", "warm", "warm-hot", "hot"}
    assert np.isfinite(tracks["warm"]).sum() >= 8
    assert np.isfinite(tracks["cold"]).sum() == 0
    assert np.isfinite(tracks["hot"]).sum() == 0


def test_angular_momentum_profile_radial_flow(cosmo_data):
    from vizmo.analysis import radial_profile

    # Pure radial flow: r x v = 0 everywhere.
    r, L, unit = radial_profile(cosmo_data, "AngularMomentum", n_bins=8)
    ok = np.isfinite(L)
    assert unit == "kpc km/s"
    assert np.nanmax(np.abs(L[ok])) < 1.0


def test_phase_histogram_volume_weighting(cosmo_data):
    from vizmo.analysis import phase_histogram
    from vizmo.physics import UnitSystem

    units = UnitSystem(cosmo_data.header)
    ph = phase_histogram(cosmo_data, "NumberDensity", "Temperature",
                         weighting="volume", n_bins=16)
    rho = np.asarray(cosmo_data.get_field("Density"), dtype=np.float64)
    m = cosmo_data.masses.astype(np.float64)
    vol_expected = (m / rho).sum() * units.length_to_kpc**3
    assert ph["H"].sum() == pytest.approx(vol_expected, rel=1e-3)
    assert ph["wlabel"].startswith("volume")


def test_stats_json_roundtrip(cosmo_data, tmp_path):
    import json

    from vizmo.analysis import region_stats, halo_properties, stats_to_json

    rows, used_r = region_stats(cosmo_data, radius_kpc=10.0)
    halo = halo_properties(cosmo_data, radius_kpc=10.0)
    out = stats_to_json(str(tmp_path / "stats.json"), rows, halo,
                        {"snapshot": "x", "redshift": 0.0})
    payload = json.load(open(out))
    assert payload["region_stats"]["Particles"]
    assert "metadata" in payload


# ---------------------------------------------------------------------------
# Phase 1: region-aware analysis + new center modes
# ---------------------------------------------------------------------------

def test_fof_most_massive_center():
    from vizmo.analysis import fof_most_massive_center

    rng = np.random.default_rng(7)
    # Dense cluster of 600 at (10,10,10), diffuse background of 400.
    cluster = 10.0 + 0.1 * rng.standard_normal((600, 3))
    bg = rng.uniform(0, 20, size=(400, 3))
    pos = np.vstack([cluster, bg])
    m = np.ones(len(pos))
    c = fof_most_massive_center(pos, m)
    assert np.linalg.norm(c - 10.0) < 0.5


def test_stellar_com_center(tmp_path, cache_isolation):
    from vizmo.data_manager import SnapshotData
    from vizmo.analysis import stellar_com_center

    path = str(tmp_path / "sg.hdf5")
    rng = np.random.default_rng(8)
    n_gas, n_star = 500, 300
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = 100.0
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["NumPart_ThisFile"] = np.array([n_gas, 0, 0, 0, n_star, 0])
        h.attrs["Time"] = 1.0
        h.attrs["HubbleParam"] = 0.7
        g = f.create_group("PartType0")
        g["Coordinates"] = 30.0 + rng.standard_normal((n_gas, 3))
        g["Masses"] = np.full(n_gas, 1e-4, dtype=np.float32)
        g["SmoothingLength"] = np.full(n_gas, 1.0, dtype=np.float32)
        s4 = f.create_group("PartType4")
        s4["Coordinates"] = 60.0 + 0.5 * rng.standard_normal((n_star, 3))
        s4["Masses"] = np.full(n_star, 2e-4, dtype=np.float32)
        s4["SubfindHsml"] = np.full(n_star, 1.0, dtype=np.float32)
    d = SnapshotData(path, particle_types=[0, 4])
    c = stellar_com_center(d, np.array([45.0, 45.0, 45.0]), 40.0)
    # Stars live at 60; gas at 30 — stellar center must pick the stars.
    assert np.linalg.norm(c - 60.0) < 1.0
    d.close()


def test_region_aware_analysis(cosmo_data):
    from vizmo.selection import Box, Sphere
    from vizmo.analysis import region_stats, phase_histogram

    c = np.array([50.0, 50.0, 50.0])
    box = Box(c - 5.0, c + 5.0)
    sph = Sphere(c, 5.0)
    rows_b, _ = region_stats(cosmo_data, center=c, radius_kpc=999.0,
                             region=box)
    rows_s, _ = region_stats(cosmo_data, center=c, radius_kpc=999.0,
                             region=sph)
    nb = int(dict(rows_b)["Particles"].replace(",", ""))
    ns = int(dict(rows_s)["Particles"].replace(",", ""))
    # The box circumscribes the sphere.
    assert nb > ns > 0
    ph = phase_histogram(cosmo_data, "NumberDensity", "Temperature",
                         region=sph)
    mass_s = float(dict(rows_s)["Total mass"].split()[0])
    assert ph["H"].sum() == pytest.approx(mass_s, rel=2e-2)
