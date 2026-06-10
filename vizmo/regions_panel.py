"""Multi-region boolean algebra model (Section 1.B).

RegionSet holds named SelectionRegions with per-entry boolean
operators (OR / AND / NOT relative to the running mask, applied left
to right), evaluates the combined mask, and persists to
~/.config/vizmo/regions_<snapshot-hash>.json. The drawer's "regions"
tool renders the list and drives this model; regions are added from
the current aperture shape (M tool), so the placement flow is shared.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from .selection import SelectionRegion

OPS = ["OR", "AND", "NOT"]
REGION_COLORS = [(120, 200, 255, 255), (240, 220, 90, 255),
                 (235, 110, 235, 255), (250, 160, 70, 255),
                 (110, 230, 130, 255), (240, 240, 240, 255),
                 (255, 120, 120, 255), (160, 160, 255, 255)]
MAX_REGIONS = 8


class RegionSet:
    """Ordered list of (name, op, region, visible) entries."""

    def __init__(self):
        self.entries = []  # dicts: name, op, region, visible

    def add(self, region, name=None, op="OR"):
        if len(self.entries) >= MAX_REGIONS:
            raise ValueError(f"at most {MAX_REGIONS} regions")
        if name is None:
            name = f"Region {len(self.entries) + 1}"
        self.entries.append({"name": name, "op": op, "region": region,
                             "visible": True})
        return self.entries[-1]

    def remove(self, idx):
        if 0 <= idx < len(self.entries):
            self.entries.pop(idx)

    def cycle_op(self, idx):
        if 0 <= idx < len(self.entries):
            cur = self.entries[idx]["op"]
            self.entries[idx]["op"] = OPS[(OPS.index(cur) + 1) % len(OPS)]

    def combined_mask(self, xyz):
        """Evaluate the boolean chain over points (N, 3).

        Entry 0 seeds the mask (its op is ignored); each later entry
        applies OR (union), AND (intersection), or NOT (subtraction).
        """
        if not self.entries:
            return np.ones(len(xyz), dtype=bool)
        mask = self.entries[0]["region"].contains(xyz)
        for e in self.entries[1:]:
            m = e["region"].contains(xyz)
            if e["op"] == "OR":
                mask = mask | m
            elif e["op"] == "AND":
                mask = mask & m
            else:  # NOT
                mask = mask & ~m
        return mask

    # -- persistence --------------------------------------------------------

    @staticmethod
    def _path_for(snapshot_path):
        from .session import config_dir

        digest = hashlib.sha1(
            os.path.abspath(snapshot_path).encode()).hexdigest()[:16]
        return os.path.join(config_dir(), f"regions_{digest}.json")

    def save(self, snapshot_path):
        import json

        path = self._path_for(snapshot_path)
        payload = {"regions": [
            {"name": e["name"], "op": e["op"], "visible": e["visible"],
             "region": e["region"].to_dict()} for e in self.entries]}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, path)
        return path

    def load(self, snapshot_path):
        import json

        path = self._path_for(snapshot_path)
        with open(path) as f:
            payload = json.load(f)
        self.entries = [
            {"name": d["name"], "op": d.get("op", "OR"),
             "visible": d.get("visible", True),
             "region": SelectionRegion.from_dict(d["region"])}
            for d in payload["regions"]]
        return path


class MaskRegion(SelectionRegion):
    """Adapter: present a RegionSet's boolean combination as a single
    SelectionRegion so the existing scope plumbing (profiles, phase,
    stats, FITS, cutouts) consumes it unchanged."""

    kind = "mask"

    def __init__(self, region_set):
        self._set = region_set

    def contains(self, xyz):
        return self._set.combined_mask(xyz)

    def volume(self):
        return np.nan

    def to_dict(self):
        return {"kind": "composite",
                "terms": [[None if i == 0 else
                           {"OR": "union", "AND": "intersect",
                            "NOT": "subtract"}[e["op"]],
                           e["region"].to_dict()]
                          for i, e in enumerate(self._set.entries)]}
