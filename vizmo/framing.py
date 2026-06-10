"""Initial-view framing helpers: find a physically meaningful center
in the loaded particle pool and aim the camera at it.
"""

import numpy as np

CENTER_MODES = ("densest", "potential", "com", "median")

# Density-like fields in preference order. Plain Density only exists for
# gas; the Subfind* variants cover DM/star types in TNG-style outputs.
_DENSITY_FIELDS = ("Density", "SubfindDensity", "SubfindDMDensity")


def mass_weighted_median(positions, masses=None, subsample=100):
    """Per-axis mass-weighted median from a coarse subsample."""
    n = len(positions)
    step = max(1, n // subsample) if n > subsample else 1
    sub_pos = positions[::step]
    if masses is not None:
        sub_w = np.asarray(masses[::step], dtype=np.float64)
    else:
        sub_w = np.ones(len(sub_pos), dtype=np.float64)
    center = np.empty(3, dtype=np.float64)
    for axis in range(3):
        order = np.argsort(sub_pos[:, axis])
        cw = np.cumsum(sub_w[order])
        half = cw[-1] * 0.5
        idx = min(int(np.searchsorted(cw, half)), len(order) - 1)
        center[axis] = float(sub_pos[order[idx], axis])
    return center


def find_center(data, mode):
    """Return a (3,) float64 center for the requested mode.

    Modes: "densest" (position of the highest-density particle),
    "potential" (potential minimum), "com" (mass-weighted mean),
    "median" (mass-weighted per-axis median). Falls back to "median"
    when the fields a mode needs aren't in the snapshot.
    """
    if mode not in CENTER_MODES:
        raise ValueError(f"Unknown center mode {mode!r}; pick from {CENTER_MODES}")

    if mode == "densest":
        best_val, best_pos = -np.inf, None
        for p in data.particle_types:
            ptype = f"PartType{p}"
            grp = data._file.get(ptype)
            if grp is None:
                continue
            name = next((n for n in _DENSITY_FIELDS if n in grp), None)
            if name is None:
                continue
            dens = np.asarray(data._read_field(ptype, name))
            i = int(np.argmax(dens))
            if dens[i] > best_val:
                pos = np.asarray(data._read_field(ptype, "Coordinates"))
                best_val, best_pos = float(dens[i]), pos[i].astype(np.float64)
        if best_pos is not None:
            return best_pos
        print("  No density field found; falling back to median center")
        mode = "median"

    if mode == "potential":
        best_val, best_pos = np.inf, None
        for p in data.particle_types:
            ptype = f"PartType{p}"
            grp = data._file.get(ptype)
            if grp is None or "Potential" not in grp:
                continue
            pot = np.asarray(data._read_field(ptype, "Potential"))
            i = int(np.argmin(pot))
            if pot[i] < best_val:
                pos = np.asarray(data._read_field(ptype, "Coordinates"))
                best_val, best_pos = float(pot[i]), pos[i].astype(np.float64)
        if best_pos is not None:
            return best_pos
        print("  No Potential field found; falling back to median center")
        mode = "median"

    if mode == "com":
        w = np.asarray(data.masses, dtype=np.float64)
        return np.average(np.asarray(data.positions, dtype=np.float64), weights=w, axis=0)

    return mass_weighted_median(data.positions, data.masses)


def frame_camera(camera, center, positions, radius=None):
    """Place `camera` looking down -z at `center` from `radius` away.

    `radius` defaults to a third of the pool's bounding-box diagonal —
    roughly "whole object in view" at the default 90° FOV.
    """
    center = np.asarray(center, dtype=np.float64)
    if radius is None:
        pmin = positions.min(axis=0)
        pmax = positions.max(axis=0)
        radius = float(np.linalg.norm(pmax - pmin)) / 3.0
    camera.position = center + np.array([0.0, 0.0, radius], dtype=np.float64)
    camera._forward = np.array([0, 0, -1], dtype=np.float32)
    camera._up = np.array([0, 1, 0], dtype=np.float32)
    camera._dirty = True
    return radius
