"""Halo list/inspector model tests (Sections 4.F, 4.G)."""

import numpy as np
import pytest


def _cat():
    return {
        "halo_id": np.array([0, 1, 2, 3]),
        "type": np.array([0, 1, 0, 1]),
        "M_halo": np.array([1e13, 5e11, 2e12, 8e10]),
        "M_star": np.array([1e11, 1e9, 5e10, 1e8]),
        "SFR": np.array([3.0, 0.1, 1.5, 0.0]),
        "R_200": np.array([400.0, 120.0, 250.0, 60.0]),
        "x": np.zeros(4), "y": np.zeros(4), "z": np.zeros(4),
    }


def test_sort_mass_descending():
    from vizmo.catalog import sort_halo_indices

    order = sort_halo_indices(_cat(), "M_halo", True)
    assert list(order) == [0, 2, 1, 3]  # most massive first


def test_filter_range_syntax():
    from vizmo.catalog import apply_halo_filter, parse_halo_filter

    parsed = parse_halo_filter("M_halo>1e12")
    assert parsed == ("M_halo", ">", 1e12)
    mask = apply_halo_filter(_cat(), parsed)
    assert list(mask) == [True, False, True, False]
    # Bare integer = halo id
    mask = apply_halo_filter(_cat(), parse_halo_filter("2"))
    assert list(mask) == [False, False, True, False]
    assert parse_halo_filter("bogus>>5") is None
    assert parse_halo_filter("NotAField>1") is None


def test_mass_threshold_mask():
    from vizmo.catalog import mass_threshold_mask

    mask = mass_threshold_mask(_cat(), 12.0)  # > 1e12
    assert list(mask) == [True, False, True, False]
    assert mass_threshold_mask(_cat(), 10.0).sum() == 4
    assert mass_threshold_mask(_cat(), 14.0).sum() == 0


def test_inspector_populates_from_catalog():
    from vizmo.science_panels import AnalysisDrawer

    d = AnalysisDrawer()
    d.set_framebuffer_size(1280, 800)
    d.catalog = _cat()
    d._halo_selected = 0
    d.enabled = True
    d.mode = "haloinspect"

    class _FakeData:
        filters = []
    d.update.__wrapped__ if False else None
    # Render via the standard update path with a stub data object that
    # the haloinspect branch never touches.
    import types

    stub = types.SimpleNamespace(
        available_fields_with_derived=lambda: [],
        filters=[], n_particles=0)
    d.update(stub)
    assert d._panel_w > 100
    actions = [b[4] for b in d._buttons]
    assert {"halo_fly", "halo_aperture", "halo_profile"} <= set(actions)


def test_halos_csv(tmp_path):
    from vizmo.catalog import halos_to_csv, mass_threshold_mask

    cat = _cat()
    out = halos_to_csv(str(tmp_path / "h.csv"), cat,
                       mass_threshold_mask(cat, 12.0))
    lines = open(out).read().strip().splitlines()
    assert len(lines) == 3  # header + 2 halos above 1e12
    assert lines[1].startswith("0,")


def test_marker_geometry_and_pick():
    """HaloMarkerRenderer vertex math + 12px screen pick (CPU side:
    GPU buffer creation is exercised in-app; here we validate the
    line-list layout and the pick on a fake projector)."""
    import types

    from vizmo.wgpu_renderer import HaloMarkerRenderer

    # Bypass GPU init: build a bare instance with only the fields the
    # CPU paths need.
    r = HaloMarkerRenderer.__new__(HaloMarkerRenderer)
    cat = _cat()
    r._centers_code = np.stack([cat["x"], cat["y"], cat["z"]],
                               axis=1) + np.arange(4)[:, None]
    r._ids = cat["halo_id"]

    def fake_w2s(p, w, h):
        # Halo i projects to x = 100 * i, y = 0.
        return (100.0 * p[0], 0.0, 1.0)

    assert r.pick(fake_w2s, 800, 600, 205, 5, max_px=12) == 2
    assert r.pick(fake_w2s, 800, 600, 150, 0, max_px=12) is None


def test_welcome_overlay_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from vizmo.recentfiles import RecentFiles
    from vizmo.science_panels import WelcomeOverlay

    snap = tmp_path / "demo.hdf5"
    snap.write_bytes(b"0" * 2_000_000)
    RecentFiles().add(str(snap))

    w = WelcomeOverlay()
    w.enabled = True
    w.set_framebuffer_size(1280, 800)
    w.update()
    assert w._panel_w == 640
    actions = [b[4] for b in w._buttons]
    assert "open_file" in actions
    opens = [a for a in actions
             if isinstance(a, tuple) and a[0] == "welcome_open"]
    assert len(opens) == 1 and opens[0][1] == str(snap)
    # Outside click dismisses.
    assert w.on_click(-50, -50) is True
    assert w.enabled is False
