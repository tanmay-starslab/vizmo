"""Loader unit tests: MassTable fallback, hsml resolution, crash-safe
type toggling, disk-cached kernel radii. All headless (no GPU)."""

import numpy as np
import pytest

from vizmo.data_manager import SnapshotData
from .conftest import make_snapshot


def test_basic_gas_load(snapshot_gas):
    d = SnapshotData(snapshot_gas)
    assert d.particle_types == [0]
    assert d.n_particles == 200
    assert d.positions.shape == (200, 3)
    assert np.allclose(d.masses, 1.5e-4)
    assert np.allclose(d.hsml, 2.0)
    d.close()


def test_masses_from_masstable(snapshot_gas_dm):
    """PartType1 has no Masses dataset; header MassTable must fill in."""
    d = SnapshotData(snapshot_gas_dm, particle_types=[1])
    assert d.n_particles == 150
    assert np.allclose(d.masses, 3.0e-5)
    d.close()


def test_subfindhsml_used_for_dm(snapshot_gas_dm):
    """SubfindHsml must be accepted as kernel radius (no KDTree run)."""
    d = SnapshotData(snapshot_gas_dm, particle_types=[1])
    assert np.allclose(d.hsml, 4.0)
    d.close()


def test_combined_types(snapshot_gas_dm):
    d = SnapshotData(snapshot_gas_dm, particle_types=[0, 1])
    assert d.n_particles == 350
    assert d._type_slices[0] == slice(0, 200)
    assert d._type_slices[1] == slice(200, 350)
    d.close()


def test_set_particle_types_failure_reverts(tmp_path):
    """A type with no Masses anywhere must fail without clobbering the pool."""
    path = make_snapshot(str(tmp_path / "bad.hdf5"), n_dm=50, dm_in_masstable=False)
    # Strip the Masses dataset so PartType1 is genuinely unloadable.
    import h5py

    with h5py.File(path, "a") as f:
        del f["PartType1"]["Masses"]
    d = SnapshotData(path, particle_types=[0])
    with pytest.raises(KeyError):
        d.set_particle_types([0, 1])
    assert d.particle_types == [0]
    assert d.n_particles == 200
    assert len(d.positions) == 200
    d.close()


def test_hsml_disk_cache_roundtrip(tmp_path, cache_isolation):
    """KDTree-computed radii must be served from the disk cache on reload."""
    path = make_snapshot(
        str(tmp_path / "nohsml.hdf5"), n_gas=300, gas_hsml=False, gas_density=False
    )
    d1 = SnapshotData(path)
    h1 = d1.hsml.copy()
    d1.close()
    d2 = SnapshotData(path)
    assert np.allclose(d2.hsml, h1)
    d2.close()

    from vizmo.data_manager import _hsml_disk_cache_path
    import os

    assert os.path.isfile(_hsml_disk_cache_path(path, 0, 300))


def test_available_fields_intersection(snapshot_gas_dm):
    """Field list must be the intersection across selected types."""
    d = SnapshotData(snapshot_gas_dm, particle_types=[0])
    assert "Density" in d.available_fields()
    d.set_particle_types([0, 1])
    fields_both = d.available_fields()
    assert "Masses" in fields_both
    assert "Density" not in fields_both  # DM has no Density
    d.close()
