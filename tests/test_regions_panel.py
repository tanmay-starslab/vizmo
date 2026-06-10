"""Boolean region-set tests (Section 1.B)."""

import numpy as np

from vizmo.regions_panel import RegionSet, MaskRegion
from vizmo.selection import Sphere


def _pts():
    rng = np.random.default_rng(0)
    return rng.uniform(-3, 3, size=(20000, 3))


def test_or_and_not_masks():
    pts = _pts()
    a = Sphere([0, 0, 0], 1.5)        # big sphere
    b = Sphere([0, 0, 0], 0.8)        # nested inside a
    c = Sphere([2.0, 0, 0], 0.5)      # disjoint from b, overlaps a edge

    rs = RegionSet()
    rs.add(a)
    rs.add(b, op="NOT")
    m = rs.combined_mask(pts)
    r = np.linalg.norm(pts, axis=1)
    assert (m == ((r <= 1.5) & (r > 0.8))).all()   # shell via NOT

    rs2 = RegionSet()
    rs2.add(a)
    rs2.add(c, op="OR")
    m2 = rs2.combined_mask(pts)
    assert (m2 == (a.contains(pts) | c.contains(pts))).all()

    rs3 = RegionSet()
    rs3.add(a)
    rs3.add(b, op="AND")
    m3 = rs3.combined_mask(pts)
    assert (m3 == b.contains(pts)).all()           # nested AND = inner


def test_mask_region_adapter_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    pts = _pts()
    rs = RegionSet()
    rs.add(Sphere([0, 0, 0], 1.0))
    rs.add(Sphere([0.5, 0, 0], 0.5), op="NOT")
    mr = MaskRegion(rs)
    assert (mr.contains(pts) == rs.combined_mask(pts)).all()

    rs.save("/fake/snapshot.hdf5")
    rs2 = RegionSet()
    rs2.load("/fake/snapshot.hdf5")
    assert len(rs2.entries) == 2
    assert rs2.entries[1]["op"] == "NOT"
    assert (rs2.combined_mask(pts) == rs.combined_mask(pts)).all()


def test_op_cycling_and_limits():
    rs = RegionSet()
    rs.add(Sphere([0, 0, 0], 1.0))
    rs.cycle_op(0)
    assert rs.entries[0]["op"] == "AND"
    rs.cycle_op(0)
    assert rs.entries[0]["op"] == "NOT"
    rs.cycle_op(0)
    assert rs.entries[0]["op"] == "OR"
