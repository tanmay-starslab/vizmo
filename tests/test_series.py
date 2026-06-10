"""Snapshot-series discovery and catalog loading tests."""

import numpy as np
import h5py
import pytest

from .conftest import make_snapshot


def _mini_snap(path, z):
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = 100.0
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["NumPart_ThisFile"] = np.array([10, 0, 0, 0, 0, 0])
        h.attrs["Redshift"] = z
        h.attrs["Time"] = 1.0 / (1.0 + z)
        h.attrs["HubbleParam"] = 0.6774
        h.attrs["Omega0"] = 0.3089
        h.attrs["OmegaLambda"] = 0.6911
        g = f.create_group("PartType0")
        g["Coordinates"] = np.random.default_rng(0).random((10, 3)) * 100
        g["Masses"] = np.ones(10, dtype=np.float32)
        g["SmoothingLength"] = np.ones(10, dtype=np.float32)


def test_discover_snapshots_sorted(tmp_path):
    from vizmo.series import discover_snapshots

    _mini_snap(str(tmp_path / "snapshot_002.hdf5"), 0.5)
    _mini_snap(str(tmp_path / "snapshot_000.hdf5"), 4.0)
    sub = tmp_path / "snapdir_001"
    sub.mkdir()
    _mini_snap(str(sub / "snapshot_001.hdf5"), 2.0)
    # A non-snapshot HDF5 must be ignored.
    with h5py.File(str(tmp_path / "groupcat.hdf5"), "w") as f:
        f.create_group("Header")

    snaps = discover_snapshots(str(tmp_path))
    assert len(snaps) == 3
    assert [round(s.redshift, 1) for s in snaps] == [4.0, 2.0, 0.5]
    assert snaps[0].snap_num == 0
    # Cosmic time increases along the series.
    times = [s.time_gyr for s in snaps]
    assert times[0] < times[1] < times[2]
    assert times[2] == pytest.approx(8.6, rel=0.1)  # z=0.5, Planck-like


def test_discover_collapses_multipart(tmp_path):
    from vizmo.series import discover_snapshots

    _mini_snap(str(tmp_path / "snapshot_005.0.hdf5"), 1.0)
    _mini_snap(str(tmp_path / "snapshot_005.1.hdf5"), 1.0)
    snaps = discover_snapshots(str(tmp_path))
    assert len(snaps) == 1
    assert snaps[0].path.endswith(".0.hdf5")


def test_load_catalog_subfind(tmp_path):
    from vizmo.catalog import load_catalog
    from vizmo.physics import UnitSystem

    path = str(tmp_path / "groupcat_099.hdf5")
    n_g = 5
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["Redshift"] = 0.0
        h.attrs["Time"] = 1.0
        h.attrs["HubbleParam"] = 0.6774
        h.attrs["Omega0"] = 0.3089
        h.attrs["OmegaLambda"] = 0.6911
        g = f.create_group("Group")
        g["GroupPos"] = np.arange(n_g * 3, dtype=np.float64).reshape(n_g, 3)
        g["Group_M_Crit200"] = np.array([100.0, 10, 1, 0.1, 0.01])
        g["Group_R_Crit200"] = np.full(n_g, 200.0)
        g["GroupFirstSub"] = np.arange(n_g, dtype=np.int64)
        s = f.create_group("Subhalo")
        smt = np.zeros((n_g, 6))
        smt[:, 4] = 1.0  # 1e10 Msun/h stellar
        s["SubhaloMassType"] = smt

    cat = load_catalog(path)
    units = UnitSystem({"HubbleParam": 0.6774, "Time": 1.0,
                        "Omega0": 0.3089, "OmegaLambda": 0.6911})
    assert len(cat["x"]) == n_g
    # ckpc/h -> kpc conversion on positions.
    assert cat["x"][1] == pytest.approx(3.0 * units.length_to_kpc)
    # 1e10 Msun/h mass convention.
    assert cat["M_halo"][0] == pytest.approx(100.0 * units.mass_to_msun)
    assert cat["M_star"][0] == pytest.approx(1.0 * units.mass_to_msun)
    assert cat["R_200"][0] == pytest.approx(200.0 * units.length_to_kpc)
