"""Shared fixtures: synthetic Gadget/AREPO-style snapshots, no GPU needed."""

import numpy as np
import h5py
import pytest


def make_snapshot(
    path,
    n_gas=200,
    n_dm=0,
    dm_in_masstable=True,
    gas_hsml=True,
    gas_density=True,
    boxsize=100.0,
    seed=42,
):
    """Write a minimal Gadget/AREPO-style HDF5 snapshot."""
    rng = np.random.default_rng(seed)
    mass_table = np.zeros(6)
    if n_dm and dm_in_masstable:
        mass_table[1] = 3.0e-5
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["BoxSize"] = boxsize
        h.attrs["MassTable"] = mass_table
        h.attrs["NumPart_ThisFile"] = np.array([n_gas, n_dm, 0, 0, 0, 0])
        h.attrs["Time"] = 1.0
        h.attrs["HubbleParam"] = 0.7

        if n_gas:
            g = f.create_group("PartType0")
            g["Coordinates"] = rng.random((n_gas, 3)) * boxsize
            g["Masses"] = np.full(n_gas, 1.5e-4, dtype=np.float32)
            if gas_hsml:
                g["SmoothingLength"] = np.full(n_gas, 2.0, dtype=np.float32)
            if gas_density:
                g["Density"] = rng.random(n_gas).astype(np.float32)
            g["Potential"] = -rng.random(n_gas).astype(np.float32)
            g["Velocities"] = rng.standard_normal((n_gas, 3)).astype(np.float32)

        if n_dm:
            g = f.create_group("PartType1")
            g["Coordinates"] = rng.random((n_dm, 3)) * boxsize
            g["SubfindHsml"] = np.full(n_dm, 4.0, dtype=np.float32)
            if not dm_in_masstable:
                g["Masses"] = np.full(n_dm, 3.0e-5, dtype=np.float32)
    return path


@pytest.fixture
def snapshot_gas(tmp_path):
    return make_snapshot(str(tmp_path / "snap_gas.hdf5"))


@pytest.fixture
def snapshot_gas_dm(tmp_path):
    return make_snapshot(str(tmp_path / "snap_gasdm.hdf5"), n_dm=150)


@pytest.fixture
def cache_isolation(tmp_path, monkeypatch):
    """Point the hsml disk cache and config dir at the test tmp dir."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path
