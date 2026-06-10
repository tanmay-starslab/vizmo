"""Selection region primitives: sphere, box, cylinder, slab, cone,
ellipsoid — each with a vectorized contains() mask, volume(), and
dict round-tripping for persistence.

This is the geometric foundation for multi-shape apertures and
boolean region algebra. The interactive UI currently drives Sphere;
the other shapes are fully functional programmatically (e.g. through
--region-file) and unit-tested.
"""

import numpy as np


class SelectionRegion:
    """Base class. All coordinates/lengths are in snapshot code units."""

    kind = "base"

    def contains(self, xyz):
        raise NotImplementedError

    def volume(self):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @staticmethod
    def from_dict(d):
        cls = _REGION_KINDS[d["kind"]]
        return cls._from_dict(d)


class Sphere(SelectionRegion):
    kind = "sphere"

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)

    def contains(self, xyz):
        d = np.asarray(xyz, dtype=np.float64) - self.center[None, :]
        return (d * d).sum(axis=1) <= self.radius**2

    def volume(self):
        return 4.0 / 3.0 * np.pi * self.radius**3

    def to_dict(self):
        return {"kind": self.kind, "center": self.center.tolist(),
                "radius": self.radius}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["center"], d["radius"])


class Box(SelectionRegion):
    kind = "box"

    def __init__(self, lo, hi):
        self.lo = np.minimum(np.asarray(lo, dtype=np.float64),
                             np.asarray(hi, dtype=np.float64))
        self.hi = np.maximum(np.asarray(lo, dtype=np.float64),
                             np.asarray(hi, dtype=np.float64))

    def contains(self, xyz):
        p = np.asarray(xyz, dtype=np.float64)
        return np.all((p >= self.lo[None, :]) & (p <= self.hi[None, :]),
                      axis=1)

    def volume(self):
        return float(np.prod(self.hi - self.lo))

    def to_dict(self):
        return {"kind": self.kind, "lo": self.lo.tolist(),
                "hi": self.hi.tolist()}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["lo"], d["hi"])


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError("zero-length axis vector")
    return v / n


class Cylinder(SelectionRegion):
    """Finite cylinder: center, axis (unit), radius, half_height."""

    kind = "cylinder"

    def __init__(self, center, axis, radius, half_height):
        self.center = np.asarray(center, dtype=np.float64)
        self.axis = _unit(axis)
        self.radius = float(radius)
        self.half_height = float(half_height)

    def contains(self, xyz):
        d = np.asarray(xyz, dtype=np.float64) - self.center[None, :]
        z = d @ self.axis
        perp2 = (d * d).sum(axis=1) - z**2
        return (np.abs(z) <= self.half_height) & (perp2 <= self.radius**2)

    def volume(self):
        return np.pi * self.radius**2 * 2.0 * self.half_height

    def to_dict(self):
        return {"kind": self.kind, "center": self.center.tolist(),
                "axis": self.axis.tolist(), "radius": self.radius,
                "half_height": self.half_height}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["center"], d["axis"], d["radius"], d["half_height"])


class Slab(SelectionRegion):
    """Infinite plane cut: |(r - center) . normal| <= half_thickness."""

    kind = "slab"

    def __init__(self, center, normal, half_thickness):
        self.center = np.asarray(center, dtype=np.float64)
        self.normal = _unit(normal)
        self.half_thickness = float(half_thickness)

    def contains(self, xyz):
        d = np.asarray(xyz, dtype=np.float64) - self.center[None, :]
        return np.abs(d @ self.normal) <= self.half_thickness

    def volume(self):
        return np.inf

    def to_dict(self):
        return {"kind": self.kind, "center": self.center.tolist(),
                "normal": self.normal.tolist(),
                "half_thickness": self.half_thickness}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["center"], d["normal"], d["half_thickness"])


class Cone(SelectionRegion):
    """One-sided cone: apex, axis, half-angle (deg), optional length."""

    kind = "cone"

    def __init__(self, apex, axis, half_angle_deg, length=np.inf):
        self.apex = np.asarray(apex, dtype=np.float64)
        self.axis = _unit(axis)
        self.half_angle_deg = float(half_angle_deg)
        self.length = float(length)

    def contains(self, xyz):
        d = np.asarray(xyz, dtype=np.float64) - self.apex[None, :]
        z = d @ self.axis
        r = np.linalg.norm(d, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosang = np.where(r > 0, z / r, 1.0)
        cos_half = np.cos(np.radians(self.half_angle_deg))
        return (z >= 0) & (cosang >= cos_half) & (z <= self.length)

    def volume(self):
        if not np.isfinite(self.length):
            return np.inf
        r_base = self.length * np.tan(np.radians(self.half_angle_deg))
        return np.pi * r_base**2 * self.length / 3.0

    def to_dict(self):
        return {"kind": self.kind, "apex": self.apex.tolist(),
                "axis": self.axis.tolist(),
                "half_angle_deg": self.half_angle_deg,
                "length": self.length}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["apex"], d["axis"], d["half_angle_deg"],
                   d.get("length", np.inf))


class Ellipsoid(SelectionRegion):
    """Axis-aligned-by-default ellipsoid; optional rotation matrix."""

    kind = "ellipsoid"

    def __init__(self, center, semi_axes, rotation=None):
        self.center = np.asarray(center, dtype=np.float64)
        self.semi_axes = np.asarray(semi_axes, dtype=np.float64)
        self.rotation = (np.eye(3) if rotation is None
                         else np.asarray(rotation, dtype=np.float64))

    def contains(self, xyz):
        d = np.asarray(xyz, dtype=np.float64) - self.center[None, :]
        local = d @ self.rotation.T
        q = (local / self.semi_axes[None, :]) ** 2
        return q.sum(axis=1) <= 1.0

    def volume(self):
        return 4.0 / 3.0 * np.pi * float(np.prod(self.semi_axes))

    def to_dict(self):
        return {"kind": self.kind, "center": self.center.tolist(),
                "semi_axes": self.semi_axes.tolist(),
                "rotation": self.rotation.tolist()}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["center"], d["semi_axes"], d.get("rotation"))


class CompositeRegion(SelectionRegion):
    """Boolean algebra over regions: ops are applied left to right,
    e.g. terms=[(None, A), ("union", B), ("subtract", C)]."""

    kind = "composite"
    OPS = ("union", "intersect", "subtract")

    def __init__(self, terms):
        self.terms = list(terms)
        if not self.terms:
            raise ValueError("CompositeRegion needs at least one term")

    def contains(self, xyz):
        mask = self.terms[0][1].contains(xyz)
        for op, region in self.terms[1:]:
            m = region.contains(xyz)
            if op == "union":
                mask = mask | m
            elif op == "intersect":
                mask = mask & m
            elif op == "subtract":
                mask = mask & ~m
            else:
                raise ValueError(f"unknown boolean op {op!r}")
        return mask

    def volume(self):
        return np.nan  # not analytic in general

    def to_dict(self):
        return {"kind": self.kind,
                "terms": [[op, r.to_dict()] for op, r in self.terms]}

    @classmethod
    def _from_dict(cls, d):
        terms = [(op, SelectionRegion.from_dict(rd))
                 for op, rd in d["terms"]]
        return cls(terms)


_REGION_KINDS = {c.kind: c for c in
                 (Sphere, Box, Cylinder, Slab, Cone, Ellipsoid,
                  CompositeRegion)}


def load_regions(path):
    """Load a list of regions from a JSON file."""
    import json

    with open(path) as f:
        payload = json.load(f)
    return [SelectionRegion.from_dict(d) for d in payload["regions"]]


def save_regions(path, regions):
    import json
    import os

    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"regions": [r.to_dict() for r in regions]}, f, indent=1)
    os.replace(tmp, path)
    return path
