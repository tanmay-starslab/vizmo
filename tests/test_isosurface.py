"""Isosurface voxelization + marching cubes tests (Section 5.B)."""

import numpy as np
import pytest


def test_voxelization_mass_conservation():
    from vizmo.wgpu_renderer import voxelize_particles

    rng = np.random.default_rng(0)
    pos = rng.uniform(40, 60, size=(30000, 3))
    m = rng.random(30000)
    grid = voxelize_particles(pos, m, np.array([50.0, 50.0, 50.0]),
                              10.0, 64)
    assert grid.sum() == pytest.approx(m.sum(), rel=1e-2)


def test_isosurface_sphere_radius():
    """An analytic unit ball on a 64^3 grid: the 0.5-level marching
    cubes surface lies within 3% of the input radius (validates the
    voxel-index -> world transform; CIC statistics are covered by the
    mass-conservation test, where particle shot noise cannot create
    spurious surface components)."""
    from vizmo.wgpu_renderer import extract_isosurface

    R = 8.0
    half = 1.2 * R
    res = 64
    center = np.array([50.0, 50.0, 50.0])
    cell = 2 * half / res
    coords = -half + (np.arange(res) + 0.5) * cell
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    rr = np.sqrt(X**2 + Y**2 + Z**2)
    grid = (rr < R).astype(np.float64)
    verts, faces, normals = extract_isosurface(grid, 0.5, center, half)
    assert len(verts) > 1000
    assert len(faces) > 1000
    dist = np.linalg.norm(verts - center[None, :], axis=1)
    # Cell size = 0.0375 R, so the band is sub-3% of R everywhere.
    assert np.abs(dist - R).max() / R < 0.03
    assert normals.shape == verts.shape


def test_mesh_obj_roundtrip(tmp_path):
    from vizmo.wgpu_renderer import mesh_to_obj

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    out = mesh_to_obj(str(tmp_path / "m.obj"), verts, faces)
    lines = open(out).read().splitlines()
    assert sum(1 for l in lines if l.startswith("v ")) == 3
    assert "f 1 2 3" in lines  # 1-based indices
