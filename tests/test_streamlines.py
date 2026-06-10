"""Streamline integration tests (Section 5.C)."""

import numpy as np
import pytest

from vizmo.wgpu_renderer import integrate_streamlines, fibonacci_sphere


def _uniform_flow_field(n=20000, v=(100.0, 0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-10, 10, size=(n, 3))
    vec = np.tile(np.asarray(v, dtype=np.float64), (n, 1))
    h = np.full(n, 2.2 * 20.0 / n ** (1 / 3))
    return pos, vec, h


def test_uniform_flow_direction():
    """In a uniform +x flow, streamlines travel along +x to <1 deg."""
    pos, vec, h = _uniform_flow_field()
    seeds = np.array([[0.0, 0.0, 0.0], [-3.0, 2.0, 1.0]])
    traces = integrate_streamlines(pos, vec, h, None, seeds,
                                   step_size=0.3, max_steps=20,
                                   bounds=(np.zeros(3), 50.0))
    for t in traces:
        assert len(t["points"]) > 10
        d = t["points"][-1] - t["points"][0]
        d /= np.linalg.norm(d)
        angle = np.degrees(np.arccos(np.clip(d @ np.array([1.0, 0, 0]),
                                             -1, 1)))
        assert angle < 1.0
        # Speed sampled along the line matches the flow speed.
        assert np.allclose(t["values"], 100.0, rtol=1e-6)


def test_streamline_terminates_outside_aperture():
    """A seed outside the bounds sphere terminates immediately."""
    pos, vec, h = _uniform_flow_field()
    seeds = np.array([[30.0, 0.0, 0.0]])  # outside the r=5 aperture
    traces = integrate_streamlines(pos, vec, h, None, seeds,
                                   step_size=0.3, max_steps=200,
                                   bounds=(np.zeros(3), 5.0))
    # The bounds check fires on the first iteration: one segment max.
    assert len(traces[0]["points"]) <= 2


def test_streamline_stops_on_slow_flow():
    pos, vec, h = _uniform_flow_field(v=(0.1, 0.0, 0.0))
    seeds = np.array([[0.0, 0.0, 0.0]])
    traces = integrate_streamlines(pos, vec, h, None, seeds,
                                   step_size=0.3, max_steps=200,
                                   bounds=(np.zeros(3), 50.0),
                                   v_floor=1.0)
    assert len(traces[0]["points"]) <= 2  # |v| < floor stops at once


def test_fibonacci_sphere_radius():
    pts = fibonacci_sphere(128, np.array([5.0, 5.0, 5.0]), 3.0)
    r = np.linalg.norm(pts - 5.0, axis=1)
    assert np.allclose(r, 3.0, rtol=1e-12)
    # Reasonable uniformity: nearest-neighbor spacing CV < 0.5.
    from scipy.spatial import cKDTree

    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    assert nn.std() / nn.mean() < 0.5
