"""Undo/redo stack tests."""

from vizmo.undo import UndoStack, MAX_DEPTH


def test_max_depth():
    u = UndoStack()
    for i in range(25):
        u.push(f"a{i}", {"r": i}, [], "SurfaceDensity")
    assert len(u.undo_stack) == MAX_DEPTH
    assert u.undo_stack[0]["action"] == "a5"   # oldest 5 dropped
    assert u.undo_stack[-1]["action"] == "a24"


def test_undo_restores_aperture():
    u = UndoStack()
    ap0 = {"center": [1, 2, 3], "radius": 10.0}
    u.push("place aperture", ap0, [], "SurfaceDensity")
    ap1 = {"center": [9, 9, 9], "radius": 50.0}
    entry = u.undo(ap1, [], "SurfaceDensity")
    assert entry["prev_aperture"] == ap0
    # Deep copy: mutating the restored dict cannot corrupt the stack.
    entry["prev_aperture"]["radius"] = -1
    assert u.redo_stack[-1]["prev_aperture"]["radius"] == 50.0


def test_undo_redo_roundtrip():
    u = UndoStack()
    u.push("add filter", None, [], "SurfaceDensity")
    live = (None, [{"field": "T", "lo": 0, "hi": 1}], "WeightedAverage")
    e1 = u.undo(*live)
    assert e1["prev_filters"] == []
    e2 = u.redo(e1["prev_aperture"], e1["prev_filters"],
                e1["prev_render_mode"])
    assert e2["prev_filters"] == live[1]
    assert e2["prev_render_mode"] == "WeightedAverage"


def test_new_action_clears_redo():
    u = UndoStack()
    u.push("a", None, [], "m")
    u.undo(None, [], "m")
    assert len(u.redo_stack) == 1
    u.push("b", None, [], "m")
    assert u.redo_stack == []
