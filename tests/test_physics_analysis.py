"""Headless tests for the physics (units/derived fields), analysis
(picking/profiles/histograms/stats), export, and camera-autopilot
layers. No GPU required."""

import numpy as np
import h5py
import pytest

from .conftest import make_snapshot


# ---------------------------------------------------------------------------
# Cosmology-aware snapshot fixture (TNG-like header + gas thermodynamics)
# ---------------------------------------------------------------------------

@pytest.fixture
def cosmo_snapshot(tmp_path):
    """Uniform-density gas sphere with known thermodynamics."""
    path = str(tmp_path / "cosmo.hdf5")
    rng = np.random.default_rng(1)
    n = 4000
    # Uniform sphere of radius 10 code units centered at (50,50,50)
    u = rng.random(n)
    r = 10.0 * u ** (1.0 / 3.0)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    pos = 50.0 + r[:, None] * v

    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = 100.0
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["NumPart_ThisFile"] = np.array([n, 0, 0, 0, 0, 0])
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
        g["Masses"] = np.full(n, 1.0e-4, dtype=np.float32)
        g["SmoothingLength"] = np.full(n, 1.0, dtype=np.float32)
        g["Density"] = np.full(n, 2.0e-7, dtype=np.float32)
        # u = 1000 (km/s)^2, fully ionized (x_e ~ 1.16)
        g["InternalEnergy"] = np.full(n, 1000.0, dtype=np.float32)
        g["ElectronAbundance"] = np.full(n, 1.158, dtype=np.float32)
        g["GFM_Metallicity"] = np.full(n, 0.0127, dtype=np.float32)
        # Pure radial outflow at 100 km/s
        rhat = (pos - 50.0) / np.maximum(
            np.linalg.norm(pos - 50.0, axis=1), 1e-12)[:, None]
        g["Velocities"] = (100.0 * rhat).astype(np.float32)
        g["StarFormationRate"] = np.full(n, 0.01, dtype=np.float32)
    return path


@pytest.fixture
def cosmo_data(cosmo_snapshot, cache_isolation):
    from vizmo.data_manager import SnapshotData

    d = SnapshotData(cosmo_snapshot, particle_types=[0])
    d.set_view_center(np.array([50.0, 50.0, 50.0]))
    yield d
    d.close()


# ---------------------------------------------------------------------------
# UnitSystem
# ---------------------------------------------------------------------------

def test_unit_system_cosmological_detection(cosmo_data):
    from vizmo.physics import UnitSystem

    u = UnitSystem(cosmo_data.header)
    assert u.cosmological
    assert u.h == pytest.approx(0.6774)
    assert u.a == pytest.approx(1.0)
    # 1 code length = 1 ckpc/h -> 1/h kpc at a=1
    assert u.length_to_kpc == pytest.approx(1.0 / 0.6774, rel=1e-5)
    # 1 code mass = 1e10 Msun/h
    assert u.mass_to_msun == pytest.approx(1e10 / 0.6774, rel=1e-3)


def test_unit_system_non_cosmological():
    from vizmo.physics import UnitSystem

    u = UnitSystem({"Time": 3.7, "HubbleParam": 1.0})
    assert not u.cosmological
    assert u.a == 1.0 and u.h == 1.0


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------

def test_temperature_fully_ionized(cosmo_data):
    # T = (gamma-1) u mu m_p / k_B with u = 1000 (km/s)^2, x_e = 1.158,
    # X_H = 0.76 -> mu ~ 0.588, T ~ 47,400 K
    t = cosmo_data.get_field("Temperature")
    from vizmo.physics import mean_molecular_weight, PROTONMASS_CGS, BOLTZMANN_CGS

    mu = mean_molecular_weight(1.158, 0.76)
    expected = (2.0 / 3.0) * 1000.0 * 1e10 * mu * PROTONMASS_CGS / BOLTZMANN_CGS
    assert np.allclose(t, expected, rtol=1e-3)
    assert 4e4 < t[0] < 6e4  # sanity: ~47,000 K


def test_radial_velocity_pure_outflow(cosmo_data):
    vr = cosmo_data.get_field("RadialVelocity")
    # Built as 100 km/s pure radial outflow.
    assert np.allclose(vr, 100.0, atol=0.5)
    vmag = cosmo_data.get_field("VelocityMagnitude")
    assert np.allclose(vmag, 100.0, atol=0.5)


def test_metallicity_solar(cosmo_data):
    z = cosmo_data.get_field("MetallicityZsun")
    assert np.allclose(z, 1.0, rtol=1e-5)


def test_derived_fields_listed_and_cached(cosmo_data):
    fields = cosmo_data.available_fields_with_derived()
    for name in ("Temperature", "NumberDensity", "Pressure", "Entropy",
                 "RadialVelocity", "VelocityMagnitude", "MetallicityZsun"):
        assert name in fields
    a = cosmo_data.get_field("Temperature")
    b = cosmo_data.get_field("Temperature")
    assert a is b  # cached


def test_center_change_invalidates_radial_velocity(cosmo_data):
    vr1 = cosmo_data.get_field("RadialVelocity")
    cosmo_data.set_view_center(np.array([0.0, 0.0, 0.0]))
    vr2 = cosmo_data.get_field("RadialVelocity")
    assert not np.allclose(vr1, vr2)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def test_radial_profile_uniform_density(cosmo_data):
    from vizmo.analysis import radial_profile

    r, prof, unit = radial_profile(cosmo_data, "Density", n_bins=12)
    assert unit == "Msun/kpc^3"
    # Uniform sphere: shell density flat in the interior. Compare the
    # middle bins against their own median (edge bins suffer Poisson +
    # boundary effects).
    mid = prof[3:9]
    mid = mid[np.isfinite(mid)]
    assert mid.size >= 4
    assert np.nanstd(mid) / np.nanmean(mid) < 0.35


def test_phase_histogram_mass_conservation(cosmo_data):
    from vizmo.analysis import phase_histogram
    from vizmo.physics import UnitSystem

    ph = phase_histogram(cosmo_data, "NumberDensity", "Temperature", n_bins=32)
    units = UnitSystem(cosmo_data.header)
    total = cosmo_data.masses.astype(np.float64).sum() * units.mass_to_msun
    # Single-valued n/T -> everything lands in one bin; mass conserved.
    assert ph["H"].sum() == pytest.approx(total, rel=1e-3)


def test_pick_particle_hits_target(cosmo_data):
    from vizmo.analysis import pick_particle

    target = cosmo_data.positions[123]
    origin = target + np.array([25.0, 0.0, 0.0])
    ray = (target - origin)
    ray = ray / np.linalg.norm(ray)
    idx = pick_particle(origin, ray, cosmo_data)
    assert idx is not None
    # The picked particle lies within a small angle of the ray.
    d = cosmo_data.positions[idx] - origin
    cosang = float(d @ ray / np.linalg.norm(d))
    assert cosang > 0.999


def test_region_stats_totals(cosmo_data):
    from vizmo.analysis import region_stats
    from vizmo.physics import UnitSystem

    units = UnitSystem(cosmo_data.header)
    r_all = 10.0 * units.length_to_kpc  # whole sphere
    rows, used_r = region_stats(cosmo_data, radius_kpc=r_all * 1.2)
    labels = dict(rows)
    n = cosmo_data.n_particles
    assert labels["Particles"] == f"{n:,}"
    total = float(labels["Total mass"].split()[0])
    expected = n * 1.0e-4 * units.mass_to_msun
    assert total == pytest.approx(expected, rel=1e-2)
    assert "SFR" in labels


def test_nice_scale_bar_rounding():
    from vizmo.analysis import nice_scale_bar

    assert nice_scale_bar(347.0) == (200.0, "200 kpc")
    assert nice_scale_bar(99.0) == (50.0, "50 kpc")
    val, lbl = nice_scale_bar(0.7)
    assert val == pytest.approx(0.5) and lbl == "500 pc"
    val, lbl = nice_scale_bar(12345.0)
    assert val == pytest.approx(10000.0) and lbl == "10 Mpc"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_region_roundtrip(cosmo_data, tmp_path):
    from vizmo.export import export_region
    from vizmo.data_manager import SnapshotData
    from vizmo.physics import UnitSystem

    units = UnitSystem(cosmo_data.header)
    out = str(tmp_path / "region.hdf5")
    path, n = export_region(
        cosmo_data, radius_kpc=5.0 * units.length_to_kpc, path=out)
    assert n > 0
    d2 = SnapshotData(path, particle_types=[0])
    assert d2.n_particles == n
    # Raw thermo fields survive, so derived fields still work.
    assert "Temperature" in d2.available_fields_with_derived()
    with h5py.File(path) as f:
        assert "VizmoCutout" in f
        assert f["Header"].attrs["NumPart_Total"][0] == n
    d2.close()


def test_export_region_csv(cosmo_data, tmp_path):
    from vizmo.export import export_region
    from vizmo.physics import UnitSystem

    units = UnitSystem(cosmo_data.header)
    out = str(tmp_path / "region.csv")
    path, n = export_region(
        cosmo_data, radius_kpc=5.0 * units.length_to_kpc, path=out)
    lines = open(path).read().strip().splitlines()
    assert lines[0] == "ptype,x,y,z,mass"
    assert len(lines) == n + 1


def test_annotate_screenshot(tmp_path):
    from PIL import Image

    from vizmo.export import annotate_screenshot

    src = str(tmp_path / "raw.png")
    Image.new("RGB", (640, 480), (5, 5, 12)).save(src)
    out = annotate_screenshot(
        src, str(tmp_path / "fig.png"),
        cmap_name="magma", qty_min=-6.0, qty_max=-2.0, log_scale=True,
        field_label="Masses", unit_label="code units",
        kpc_per_px=1.5, caption="test snapshot   z=0.00")
    img = Image.open(out)
    assert img.size == (640, 480)
    # The colorbar region on the right edge must no longer be background.
    px = np.asarray(img)
    right = px[:, int(640 * 0.95):, :]
    assert right.std() > 5.0


# ---------------------------------------------------------------------------
# Camera autopilot
# ---------------------------------------------------------------------------

def test_camera_fly_to_lands_exactly():
    from vizmo.camera import Camera

    c = Camera(position=[0, 0, 100.0])
    c.fly_to(position=[100.0, 0, 0], look_at=[0, 0, 0], duration=1.0)
    assert c.in_transit
    for _ in range(20):
        c.update(0.1)
    assert not c.in_transit
    assert np.allclose(c.position, [100.0, 0, 0], atol=1e-9)
    assert np.allclose(c.forward, [-1.0, 0, 0], atol=1e-5)


def test_camera_orbit_preserves_radius():
    from vizmo.camera import Camera

    c = Camera(position=[0, 0, 50.0])
    assert c.start_orbit([0.0, 0.0, 0.0])
    for _ in range(50):
        c.update(0.05)
    assert np.linalg.norm(c.position) == pytest.approx(50.0, rel=1e-9)
    # Forward keeps pointing at the center.
    f = -c.position / np.linalg.norm(c.position)
    assert np.allclose(c.forward, f, atol=1e-5)


def test_manual_input_cancels_autopilot():
    import glfw

    from vizmo.camera import Camera

    c = Camera(position=[0, 0, 100.0])
    c.fly_to(position=[100.0, 0, 0], duration=5.0)
    c.update(0.05)
    assert c.in_transit
    c.on_key(glfw.KEY_W, glfw.PRESS)
    c.update(0.05)
    assert not c.in_transit


# ---------------------------------------------------------------------------
# Aperture analysis (region centering, special profiles, scoping)
# ---------------------------------------------------------------------------

def test_shrinking_sphere_center_finds_clump(cosmo_data):
    from vizmo.analysis import shrinking_sphere_center

    # True center of the uniform sphere is (50,50,50); start offset.
    c = shrinking_sphere_center(
        cosmo_data.positions, cosmo_data.masses,
        center=np.array([53.0, 48.0, 51.0]), radius=15.0)
    assert np.linalg.norm(c - 50.0) < 1.5


def test_find_center_in_region_modes(cosmo_data):
    from vizmo.analysis import find_center_in_region, CENTER_MODES

    for mode in CENTER_MODES:
        c = find_center_in_region(
            cosmo_data, np.array([50.0, 50.0, 50.0]), 12.0, mode)
        assert np.all(np.isfinite(c))
        assert np.linalg.norm(c - 50.0) < 11.0


def test_rotation_curve_keplerian(tmp_path, cache_isolation):
    """All mass in a tiny core -> v_c(r) ~ sqrt(GM/r) outside it."""
    import h5py as _h5
    from vizmo.data_manager import SnapshotData
    from vizmo.analysis import radial_profile, G_KPC_KMS2_MSUN
    from vizmo.physics import UnitSystem

    path = str(tmp_path / "kepler.hdf5")
    rng = np.random.default_rng(3)
    n_core, n_test = 5000, 2000
    core = 50.0 + 0.05 * rng.standard_normal((n_core, 3))
    u = rng.random(n_test)
    r = 1.0 + 9.0 * u
    v = rng.standard_normal((n_test, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    outer = 50.0 + r[:, None] * v
    pos = np.vstack([core, outer])
    m = np.concatenate([np.full(n_core, 1.0e-3), np.full(n_test, 1e-12)])
    with _h5.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = 100.0
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["NumPart_ThisFile"] = np.array([len(m), 0, 0, 0, 0, 0])
        h.attrs["Time"] = 1.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["UnitLength_in_cm"] = 3.085678e21
        h.attrs["UnitMass_in_g"] = 1.989e43
        h.attrs["UnitVelocity_in_cm_per_s"] = 1e5
        g = f.create_group("PartType0")
        g["Coordinates"] = pos
        g["Masses"] = m.astype(np.float32)
        g["SmoothingLength"] = np.full(len(m), 0.5, dtype=np.float32)
    d = SnapshotData(path, particle_types=[0])
    d.set_view_center(np.array([50.0, 50.0, 50.0]))
    units = UnitSystem(d.header)
    rr, vc, unit = radial_profile(d, "RotationCurve", r_min_kpc=1.0,
                                  r_max_kpc=9.0, n_bins=12)
    assert unit == "km/s"
    M = n_core * 1.0e-3 * units.mass_to_msun
    expected = np.sqrt(G_KPC_KMS2_MSUN * M / rr)
    ok = np.isfinite(vc)
    assert np.allclose(vc[ok], expected[ok], rtol=0.05)
    d.close()


def test_velocity_dispersion_pure_radial(cosmo_data):
    from vizmo.analysis import radial_profile

    # Pure 100 km/s radial outflow: shell-mean velocity vector ~ 0, so
    # the 3D dispersion about the mean is ~100 km/s.
    r, sig, unit = radial_profile(cosmo_data, "VelocityDispersion3D",
                                  n_bins=8)
    ok = np.isfinite(sig)
    assert unit == "km/s"
    assert np.allclose(sig[ok][2:], 100.0, rtol=0.15)


def test_phase_histogram_aperture_scoping(cosmo_data):
    from vizmo.analysis import phase_histogram, region_stats
    from vizmo.physics import UnitSystem

    units = UnitSystem(cosmo_data.header)
    r_kpc = 5.0 * units.length_to_kpc
    ph = phase_histogram(cosmo_data, "NumberDensity", "Temperature",
                         center=np.array([50.0, 50.0, 50.0]),
                         radius_kpc=r_kpc)
    rows, _ = region_stats(cosmo_data, center=np.array([50.0, 50.0, 50.0]),
                           radius_kpc=r_kpc)
    total = float(dict(rows)["Total mass"].split()[0])
    assert ph["H"].sum() == pytest.approx(total, rel=2e-2)


# ---------------------------------------------------------------------------
# Field filters
# ---------------------------------------------------------------------------

def test_filter_mask_basic(cosmo_data):
    t0 = float(cosmo_data.get_field("Temperature")[0])
    cosmo_data.set_filters([
        {"field": "Temperature", "lo": t0 * 0.5, "hi": t0 * 2.0}])
    assert cosmo_data.filter_mask().min() == 1.0  # single-T snapshot
    cosmo_data.set_filters([
        {"field": "Temperature", "lo": t0 * 2.0, "hi": t0 * 4.0}])
    assert cosmo_data.filter_mask().max() == 0.0
    cosmo_data.set_filters([])
    assert cosmo_data.filter_mask().min() == 1.0


def test_filter_on_derived_radius(cosmo_data):
    from vizmo.physics import UnitSystem

    units = UnitSystem(cosmo_data.header)
    r5 = 5.0 * units.length_to_kpc
    cosmo_data.set_filters([
        {"field": "RadiusFromCenter", "lo": 0.0, "hi": r5}])
    m = cosmo_data.filter_mask()
    r = cosmo_data.get_field("RadiusFromCenter")
    assert abs(m.sum() - (r <= r5).sum()) <= 1
    # Moving the center invalidates radius-dependent filter masks.
    cosmo_data.set_view_center(np.array([0.0, 0.0, 0.0]))
    m2 = cosmo_data.filter_mask()
    assert m2.sum() < m.sum()
    cosmo_data.set_view_center(np.array([50.0, 50.0, 50.0]))
    cosmo_data.set_filters([])


# ---------------------------------------------------------------------------
# FITS map export
# ---------------------------------------------------------------------------

def test_export_fits_map_mass_conservation(cosmo_data, tmp_path):
    from astropy.io import fits as pyfits
    from vizmo.export import export_fits_map
    from vizmo.physics import UnitSystem

    units = UnitSystem(cosmo_data.header)
    r_kpc = 12.0 * units.length_to_kpc  # whole 10-unit sphere inside
    fpath, ppath = export_fits_map(
        cosmo_data, field="Masses", radius_kpc=r_kpc,
        path=str(tmp_path / "map.fits"), npix=128)
    hdu = pyfits.open(fpath)[0]
    pix_area = hdu.header["CDELT1"] * hdu.header["CDELT2"]
    total = np.nansum(hdu.data) * pix_area
    expected = (cosmo_data.masses.astype(np.float64).sum()
                * units.mass_to_msun)
    assert total == pytest.approx(expected, rel=5e-2)
    assert hdu.header["BUNIT"] == "Msun/kpc^2"


def test_export_fits_map_weighted_field(cosmo_data, tmp_path):
    from astropy.io import fits as pyfits
    from vizmo.export import export_fits_map
    from vizmo.physics import UnitSystem

    units = UnitSystem(cosmo_data.header)
    fpath, _ = export_fits_map(
        cosmo_data, field="Temperature",
        radius_kpc=12.0 * units.length_to_kpc,
        path=str(tmp_path / "tmap.fits"), npix=64)
    img = pyfits.open(fpath)[0].data
    t_expected = float(cosmo_data.get_field("Temperature")[0])
    finite = np.isfinite(img)
    # Mass-weighted projection of a single-valued field returns it.
    assert np.allclose(img[finite], t_expected, rtol=1e-3)


def test_cmasher_colormaps_available():
    from vizmo.colormaps import AVAILABLE_COLORMAPS, colormap_to_texture_data

    cmr = [c for c in AVAILABLE_COLORMAPS if c.startswith("cmr.")]
    assert len(cmr) >= 4
    tex = colormap_to_texture_data(cmr[0])
    assert tex.shape == (256, 4) and tex.dtype == np.uint8
