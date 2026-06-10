"""Canonical keybinding list: single source of truth for the in-app
help panel and the README controls table.
"""

KEYBINDINGS = [
    ("W / A / S / D", "Fly forward / left / back / right"),
    ("Z / X", "Fly up / down"),
    ("Q / E", "Roll counter-clockwise / clockwise"),
    ("Mouse drag (LMB)", "Look around"),
    ("Scroll", "Flight speed up / down"),
    ("Ctrl+Scroll", "Optical zoom (FOV)"),
    ("[ / ]", "Optical zoom in / out (FOV -/+ 5°)"),
    ("R", "Auto-range color scale (Color slot in Composite)"),
    ("T", "Auto-range Lightness slot (Composite mode)"),
    ("L", "Toggle log / linear color scale"),
    ("- / =", "Widen / narrow color range"),
    ("C", "Cycle colormap"),
    (", / .", "Halve / double detail ceiling (subsample cap)"),
    ("P", "Save screenshot"),
    ("B / Shift+B", "Cycle star band forward / back"),
    ("O", "Toggle star dust extinction"),
    ("K", "Toggle sink/star panel"),
    ("\\", "Toggle developer HUD"),
    ("Tab", "Hide / show all UI"),
    ("F1 or H", "Toggle this help"),
    ("Esc", "Quit"),
]
