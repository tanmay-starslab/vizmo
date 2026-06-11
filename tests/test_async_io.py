"""Async-load API tests (Item 1, safe subset)."""

import time

import numpy as np
import pytest

from .conftest import make_snapshot


def test_load_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    from vizmo.data_manager import SnapshotData

    path = make_snapshot(str(tmp_path / "s.hdf5"), n_gas=300)
    d = SnapshotData(path, particle_types=[0])
    # Synchronous constructor: by return time the snapshot is loaded
    # and the progress markers are terminal.
    assert d.is_loaded is True
    assert d.load_progress == 1.0
    assert d.load_status == "ready"
    d.close()


def test_get_field_async_nonblocking(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    from vizmo.data_manager import SnapshotData

    path = make_snapshot(str(tmp_path / "s.hdf5"), n_gas=2000)
    d = SnapshotData(path, particle_types=[0])
    # First call submits and may return None (pending) — it must
    # never raise. Poll until ready.
    first = d.get_field_async("Density")
    assert first is None or isinstance(first, np.ndarray)
    for _ in range(200):
        if d.field_ready("Density"):
            break
        time.sleep(0.01)
    assert d.field_ready("Density")
    out = d.get_field_async("Density")
    assert isinstance(out, np.ndarray)
    assert len(out) == d.n_particles
    d.close()


def test_progress_reaches_one_after_type_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    from vizmo.data_manager import SnapshotData

    path = make_snapshot(str(tmp_path / "s.hdf5"), n_gas=300, n_dm=200)
    d = SnapshotData(path, particle_types=[0])
    d.set_particle_types([0, 1])
    assert d.load_progress == 1.0
    assert d.load_status == "ready"
    d.close()
