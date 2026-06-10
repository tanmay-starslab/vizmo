"""Volume rendering tests (Section 5.E)."""

import numpy as np
import pytest


def test_voxelization_mass_conservation():
    from vizmo.wgpu_renderer import voxelize_particles

    rng = np.random.default_rng(3)
    pos = rng.uniform(40, 60, size=(25000, 3))
    m = rng.random(25000)
    grid = voxelize_particles(pos, m, np.array([50.0, 50.0, 50.0]),
                              10.0, 64)
    assert grid.sum() == pytest.approx(m.sum(), rel=1e-2)


def test_texture3d_upload():
    """3D texture creation + upload on a headless wgpu device. Skipped
    when no adapter is available (CI without a GPU)."""
    import wgpu

    try:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance")
        device = adapter.request_device_sync()
    except Exception:
        pytest.skip("no wgpu adapter available")
    n = 32
    tex = device.create_texture(
        size=(n, n, n), format="r32float", dimension="3d",
        usage=(wgpu.TextureUsage.TEXTURE_BINDING
               | wgpu.TextureUsage.COPY_DST))
    data = np.random.default_rng(0).random((n, n, n)).astype(np.float32)
    device.queue.write_texture(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        np.ascontiguousarray(data).tobytes(),
        {"bytes_per_row": n * 4, "rows_per_image": n}, (n, n, n))
    device.queue.submit([])  # flush; raises on validation errors


def test_transfer_function_interpolation():
    """The piecewise-linear TF control points produce a monotone alpha
    ramp between the knots (pure-numpy check of the same math the
    renderer uploads)."""
    pts = sorted([(0.0, 0.0), (0.6, 0.05), (1.0, 0.8)])
    x = np.linspace(0, 1, 256)
    alpha = np.interp(x, [p[0] for p in pts], [p[1] for p in pts])
    assert alpha[0] == 0.0
    assert alpha[-1] == pytest.approx(0.8)
    assert (np.diff(alpha) >= -1e-12).all()  # monotone for these knots
    assert alpha[np.searchsorted(x, 0.6)] == pytest.approx(0.05,
                                                           abs=0.01)
