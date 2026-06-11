"""Minimal undo/redo stack for aperture, filter, and render-mode
operations (Ctrl+Z / Ctrl+Shift+Z)."""

from __future__ import annotations

import copy

MAX_DEPTH = 20


class UndoStack:
    """Snapshot-style undo: before every undoable mutation the caller
    pushes the *current* state with a description; undo swaps the live
    state with the popped entry (capturing the live state onto the
    redo stack), so undo/redo round-trips exactly."""

    def __init__(self):
        self.undo_stack = []
        self.redo_stack = []

    @staticmethod
    def _snap(action, aperture, filters, render_mode):
        return {
            "action": str(action),
            "prev_aperture": copy.deepcopy(aperture),
            "prev_filters": copy.deepcopy(list(filters or [])),
            "prev_render_mode": render_mode,
        }

    def push(self, action, aperture, filters, render_mode):
        """Record state *before* an undoable action; clears redo."""
        self.undo_stack.append(
            self._snap(action, aperture, filters, render_mode))
        del self.undo_stack[:-MAX_DEPTH]
        self.redo_stack.clear()

    def undo(self, current_aperture, current_filters,
             current_render_mode):
        """Pop the last entry; returns it (the state to restore) or
        None. The current state moves to the redo stack."""
        if not self.undo_stack:
            return None
        entry = self.undo_stack.pop()
        self.redo_stack.append(
            self._snap(entry["action"], current_aperture,
                       current_filters, current_render_mode))
        del self.redo_stack[:-MAX_DEPTH]
        return entry

    def redo(self, current_aperture, current_filters,
             current_render_mode):
        if not self.redo_stack:
            return None
        entry = self.redo_stack.pop()
        self.undo_stack.append(
            self._snap(entry["action"], current_aperture,
                       current_filters, current_render_mode))
        del self.undo_stack[:-MAX_DEPTH]
        return entry
