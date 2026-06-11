"""GUI structural tests: menu bar, recents, browser, picker."""

import numpy as np
import pytest


def test_menubar_file_menu():
    from vizmo.overlay import MenuBar

    m = MenuBar()
    items = m.get_menu("File")
    assert len(items) >= 5
    labels = [i.label for i in items if not i.separator]
    assert "Open..." in labels
    assert "Quit" in labels
    # Export submenu carries the seven export targets.
    exp = next(i for i in items if i.label == "Export")
    assert exp.submenu is not None
    assert len(exp.submenu) == 8  # incl. Movie...


def test_menubar_all_five_menus():
    from vizmo.overlay import build_default_menus

    menus = build_default_menus()
    assert list(menus) == ["File", "View", "Analysis", "Export", "Help"]
    assert len(menus["Analysis"]) >= 10
    # Every non-separator item carries either an action or a submenu.
    for items in menus.values():
        for it in items:
            if not it.separator:
                assert it.action is not None or it.submenu is not None


def test_menubar_escape_closes():
    from vizmo.overlay import MenuBar

    m = MenuBar()
    m.open_menu = "File"
    m.open_submenu = 1
    assert m.on_escape() is True
    assert m.open_menu is None and m.open_submenu is None
    assert m.on_escape() is False  # already closed


def test_recent_files_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from vizmo.recentfiles import RecentFiles

    r = RecentFiles()
    for i in range(13):
        r.add(str(tmp_path / f"snap_{i}.hdf5"))
    got = r.get()
    assert len(got) == 10  # capped
    assert got[0].endswith("snap_12.hdf5")  # most recent first
    # Re-adding moves to front without duplication.
    r.add(str(tmp_path / "snap_5.hdf5"))
    got = r.get()
    assert got[0].endswith("snap_5.hdf5")
    assert len(got) == 10
    assert len(set(got)) == 10


def test_field_picker_search():
    from vizmo.science_panels import FieldPickerPanel

    p = FieldPickerPanel()
    p.set_fields(["Temperature", "TcoolOverTff", "Density",
                  "RadialVelocity", "SomeRawField"])
    p.search = "t"
    groups = dict(p.filtered())
    flat = [n for names in groups.values() for n in names]
    assert all("t" in n.lower() for n in flat)
    assert "Temperature" in flat and "Density" in flat
    p.search = "zzz"
    assert p.filtered() == []
    # Raw fields not in any category land in RAW.
    p.search = "someraw"
    assert dict(p.filtered())["RAW"] == ["SomeRawField"]


def test_colormap_browser_categories():
    from vizmo.science_panels import (ColormapBrowserPanel,
                                      CMAP_CATEGORIES)

    b = ColormapBrowserPanel()
    assert "viridis" in CMAP_CATEGORIES["Perceptual"]
    assert "coolwarm" in CMAP_CATEGORIES["Diverging"]
    assert len(CMAP_CATEGORIES["CMasher"]) > 5  # filled from cmasher
    b.tab = "Diverging"
    b.reversed = True
    names = b.names_for_tab()
    # Reversal TOGGLES the _r suffix (RdBu_r -> RdBu, coolwarm ->
    # coolwarm_r); every reversed name must resolve to a real cmap.
    for orig, rev in zip(CMAP_CATEGORIES["Diverging"], names):
        if orig.endswith("_r"):
            assert rev == orig[:-2]
        else:
            assert rev == orig + "_r"
        sw = b._swatch(rev)
        assert sw is not None and sw.shape == (b.SW_H, b.SW_W, 4)


def test_sidebar_section_state_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from vizmo.overlay import UserMenu

    m = UserMenu()
    assert m.sections["fields"] is True  # default open
    m._sec_fields()                       # collapse + persist
    m._sec_lod()
    assert m.sections["fields"] is False
    # A fresh instance reloads the persisted state.
    m2 = UserMenu()
    assert m2.sections["fields"] is False
    assert m2.sections["lod"] is False
    assert m2.sections["types"] is True


def test_loading_overlay_layout():
    from vizmo.science_panels import LoadingOverlay, SPINNER_FRAMES

    p = LoadingOverlay()
    p.set_framebuffer_size(1280, 800)
    p.update("snap.hdf5", frame=0, progress=0.47,
             status="Reading PartType0...")
    assert p._panel_w == 460 and p._panel_h > 80
    assert len(SPINNER_FRAMES) == 4
    # Progress edge cases render without error.
    p.update("snap.hdf5", frame=7, progress=0.0, status="")
    p.update("snap.hdf5", frame=13, progress=1.0, status="ready")


def test_data_manager_progress_dict(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    from tests.conftest import make_snapshot
    from vizmo.data_manager import SnapshotData

    path = make_snapshot(str(tmp_path / "s.hdf5"), n_gas=200, n_dm=100)
    prog = {}
    d = SnapshotData(path, particle_types=[0, 1], progress=prog)
    # The shared dict mirrored live status during the load.
    assert "status" in prog and "progress" in prog
    assert 0.0 <= prog["progress"] <= 1.0
    d.close()
