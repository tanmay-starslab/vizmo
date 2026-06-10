"""Framing tests: center finding and camera placement."""

import numpy as np
import h5py

from vizmo.data_manager import SnapshotData
from vizmo.framing import find_center, frame_camera, CENTER_MODES
from vizmo.camera import Camera
from .conftest import make_snapshot


def _spiked_snapshot(tmp_path):
    """Snapshot where particle 7 is the densest and deepest in potential."""
    path = make_snapshot(str(tmp_path / "spiked.hdf5"), n_gas=50)
    with h5py.File(path, "a") as f:
        g = f["PartType0"]
        dens = g["Density"][:]
        pot = g["Potential"][:]
        dens[7] = 1e6
        pot[7] = -1e6
        del g["Density"], g["Potential"]
        g["Density"] = dens
        g["Potential"] = pot
        spike_pos = g["Coordinates"][7]
    return path, spike_pos


def test_center_densest(tmp_path):
    path, spike_pos = _spiked_snapshot(tmp_path)
    d = SnapshotData(path)
    c = find_center(d, "densest")
    assert np.allclose(c, spike_pos)
    d.close()


def test_center_potential(tmp_path):
    path, spike_pos = _spiked_snapshot(tmp_path)
    d = SnapshotData(path)
    c = find_center(d, "potential")
    assert np.allclose(c, spike_pos)
    d.close()


def test_center_com_and_median(snapshot_gas):
    d = SnapshotData(snapshot_gas)
    for mode in ("com", "median"):
        c = find_center(d, mode)
        assert c.shape == (3,)
        # Uniform random box: both centers land near the middle.
        assert np.all(c > 20) and np.all(c < 80)
    d.close()


def test_center_modes_constant():
    assert set(CENTER_MODES) == {"densest", "potential", "com", "median"}


def test_frame_camera_placement():
    cam = Camera()
    positions = np.random.default_rng(1).random((100, 3)) * 10
    center = np.array([5.0, 5.0, 5.0])
    r = frame_camera(cam, center, positions, radius=42.0)
    assert r == 42.0
    assert np.allclose(cam.position, [5, 5, 47])
    # Looking down -z at the center.
    assert np.allclose(cam.forward, [0, 0, -1])


def test_frame_camera_default_radius():
    cam = Camera()
    positions = np.zeros((10, 3))
    positions[0] = [30, 0, 0]  # extent 30 -> default radius 10
    r = frame_camera(cam, [0, 0, 0], positions)
    assert abs(r - 10.0) < 1e-6
