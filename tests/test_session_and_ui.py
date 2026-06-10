"""Session persistence and headless UI panel tests."""

import numpy as np

from vizmo.camera import Camera
from vizmo.session import (
    load_bookmarks,
    save_bookmarks,
    camera_pose,
    apply_camera_pose,
    load_settings,
    save_settings,
)


def test_bookmark_roundtrip(cache_isolation):
    cam = Camera()
    cam.position = np.array([1.0, 2.0, 3.0])
    cam.speed = 42.0
    cam.fov = 60.0
    save_bookmarks("/tmp/some_snap.hdf5", {"3": camera_pose(cam)})
    bm = load_bookmarks("/tmp/some_snap.hdf5")
    cam2 = Camera()
    apply_camera_pose(cam2, bm["3"])
    assert np.allclose(cam2.position, [1, 2, 3])
    assert cam2.speed == 42.0 and cam2.fov == 60.0


def test_bookmarks_are_per_snapshot(cache_isolation):
    cam = Camera()
    save_bookmarks("/tmp/snap_a.hdf5", {"1": camera_pose(cam)})
    assert load_bookmarks("/tmp/snap_b.hdf5") == {}


def test_settings_roundtrip(cache_isolation):
    save_settings({"colormap": "inferno", "invert_mouse": False})
    s = load_settings()
    assert s["colormap"] == "inferno"
    assert s["invert_mouse"] is False


def test_camera_fov_clamping():
    cam = Camera(fov=90)
    assert cam.adjust_fov(-500) == 10.0
    assert cam.adjust_fov(+500) == 140.0


def test_toolbar_hit_testing():
    from vizmo.overlay import ToolbarOverlay

    tb = ToolbarOverlay()
    tb.set_framebuffer_size(1920, 1080)
    tb.update(recording=False)
    keys = [w[3] for w in tb._widgets if w[2] == "hbutton"]
    assert keys == [
        "auto_range", "screenshot", "publication", "record", "help",
        "inspector", "phase", "profile", "stats", "filters", "aperture",
        "orbit", "export_region",
    ]
    # Click dead-center of each button and check the action comes back.
    for w in tb._widgets:
        x = tb._panel_x + (w[4] + w[5]) // 2
        y = tb._panel_y + (w[0] + w[1]) // 2
        assert tb.on_click(x, y) == w[3]
    # A click far outside is not consumed.
    assert tb.on_click(5000, 5000) is False


def test_help_overlay_layout():
    from vizmo.overlay import HelpOverlay
    from vizmo.keymap import KEYBINDINGS

    h = HelpOverlay()
    h.enabled = True
    h.set_framebuffer_size(1920, 1080)
    h.update()
    # Panel must be centered and big enough for every binding row.
    assert h._panel_h >= len(KEYBINDINGS) * h.style.line_height
    assert abs((h._panel_x + h._panel_w / 2) - 960) <= 2


def test_panels_render_without_gpu():
    """Layout/draw of every panel type must run headless (no wgpu)."""
    from vizmo.overlay import UserMenu, DevOverlay

    class FakeRenderer:
        log_scale = 1
        qty_min = -5.0
        qty_max = -2.0
        hsml_scale = 1.0
        n_particles = 1000
        n_total = 2000
        _subsample_max_per_frame = 4_000_000
        skip_vsync = False
        auto_lod = False
        target_fps = 20.0

    um = UserMenu()
    um.set_framebuffer_size(1920, 1080)
    um.update(
        FakeRenderer(), "magma", ["magma", "viridis"],
        sd_fields=["Masses", "Density"], sd_field="Masses",
        render_modes=["SurfaceDensity", "Composite"],
        render_mode_name="SurfaceDensity",
        sd_ops=["*"], vector_fields=["Velocities"],
        vector_projections=["LOS"],
        available_ptypes=[0, 1], selected_ptypes=[0],
        ptype_labels={0: "gas", 1: "halo"},
        fov=90.0, cam_speed=10.0,
    )
    assert um._panel_w > 0 and any(w[2] == "slider_dec" for w in um._widgets)
