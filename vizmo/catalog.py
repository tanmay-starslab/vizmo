"""Group/subhalo catalog loading (TNG-Subfind HDF5 and Rockstar ASCII).

Returns plain dicts of numpy arrays (no pandas dependency):
  {"x", "y", "z" (kpc, physical), "M_halo" (Msun), "M_star" (Msun),
   "SFR" (Msun/yr), "R_200" (kpc), "type" (0 central / 1 satellite),
   "halo_id" (int)}
"""

from __future__ import annotations

import os

import numpy as np

from .physics import UnitSystem


def _empty():
    return {k: np.zeros(0) for k in
            ("x", "y", "z", "M_halo", "M_star", "SFR", "R_200",
             "type", "halo_id")}


def load_catalog(catalog_path: str, unit_system=None) -> dict:
    """Load a halo catalog, auto-detecting the format.

    Supported:
    - TNG/Arepo Subfind group catalogs (HDF5 with Group/Subhalo groups
      or flat GroupPos/SubhaloPos datasets). Positions are converted
      from ckpc/h to physical kpc with the catalog's own header
      cosmology (or `unit_system` when the header lacks one).
    - Rockstar .list ASCII (positions in Mpc/h, masses Msun/h).

    Returns the dict described in the module docstring.
    """
    if catalog_path.endswith((".hdf5", ".h5")):
        return _load_subfind(catalog_path, unit_system)
    if catalog_path.endswith(".list"):
        return _load_rockstar(catalog_path, unit_system)
    raise ValueError(f"Unrecognized catalog format: {catalog_path}")


def _load_subfind(path: str, unit_system=None) -> dict:
    import h5py

    out = _empty()
    with h5py.File(path, "r") as f:
        hdr = dict(f["Header"].attrs) if "Header" in f else {}
        units = unit_system or UnitSystem(hdr)
        kpc = units.length_to_kpc
        # 1e10 Msun/h mass convention for TNG catalogs.
        m_conv = units.mass_to_msun

        grp = f.get("Group", f)
        sub = f.get("Subhalo", f)

        gpos = grp.get("GroupPos")
        if gpos is None:
            return out
        gpos = np.asarray(gpos, dtype=np.float64) * kpc
        n_g = len(gpos)

        def garr(names, default=0.0, scale=1.0):
            for n in names:
                if n in grp:
                    return np.asarray(grp[n], dtype=np.float64) * scale
            return np.full(n_g, default)

        m_halo = garr(["Group_M_Crit200", "Group_M_Mean200",
                       "GroupMass"], scale=m_conv)
        r200 = garr(["Group_R_Crit200", "Group_R_Mean200"], scale=kpc)
        sfr_g = garr(["GroupSFR"])

        # Stellar masses via the group's first subhalo when available.
        m_star = np.zeros(n_g)
        if (sub is not f and "SubhaloMassType" in sub
                and "GroupFirstSub" in grp):
            first = np.asarray(grp["GroupFirstSub"], dtype=np.int64)
            smt = np.asarray(sub["SubhaloMassType"], dtype=np.float64)
            ok = (first >= 0) & (first < len(smt))
            m_star[ok] = smt[first[ok], 4] * m_conv

        out = {
            "x": gpos[:, 0], "y": gpos[:, 1], "z": gpos[:, 2],
            "M_halo": m_halo, "M_star": m_star, "SFR": sfr_g,
            "R_200": r200,
            "type": np.zeros(n_g, dtype=np.int64),  # groups = centrals
            "halo_id": np.arange(n_g, dtype=np.int64),
        }
    return out


def _load_rockstar(path: str, unit_system=None) -> dict:
    """Rockstar out_*.list: header line names columns; positions Mpc/h."""
    with open(path) as f:
        header = f.readline().lstrip("#").split()
    cols = {name.split("(")[0].lower(): i for i, name in enumerate(header)}
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    h = unit_system.h if unit_system is not None else 0.7
    kpc_per_mpch = 1000.0 / h

    def col(name, default=0.0, scale=1.0):
        i = cols.get(name)
        if i is None:
            return np.full(len(data), default)
        return data[:, i] * scale

    return {
        "x": col("x", scale=kpc_per_mpch),
        "y": col("y", scale=kpc_per_mpch),
        "z": col("z", scale=kpc_per_mpch),
        "M_halo": col("mvir", scale=1.0 / h),
        "M_star": np.zeros(len(data)),
        "SFR": np.zeros(len(data)),
        "R_200": col("rvir", scale=1.0 / h),  # kpc/h -> kpc
        "type": np.where(col("pid", default=-1) < 0, 0, 1).astype(np.int64),
        "halo_id": col("id", default=-1).astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Halo-list model (Section 4.G): filtering, sorting, threshold masks
# ---------------------------------------------------------------------------

import re as _re

HALO_COLUMNS = ["halo_id", "type", "M_halo", "M_star", "SFR", "R_200"]
_FILTER_RE = _re.compile(
    r"^\s*(\w+)\s*(>=|<=|>|<|=)\s*([0-9.eE+-]+)\s*$")


def parse_halo_filter(text: str):
    """Parse 'M_halo>1e12' style filters -> (field, op, value) or a
    bare integer halo id -> ('halo_id', '=', id). None when empty or
    unparseable."""
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return ("halo_id", "=", float(text))
    m = _FILTER_RE.match(text)
    if not m:
        return None
    field, op, val = m.group(1), m.group(2), float(m.group(3))
    return (field, op, val) if field in HALO_COLUMNS else None


def apply_halo_filter(cat: dict, parsed):
    """Boolean mask over catalog rows for a parsed filter (all-True
    when parsed is None)."""
    import numpy as np

    n = len(cat["halo_id"])
    if parsed is None:
        return np.ones(n, dtype=bool)
    field, op, val = parsed
    col = np.asarray(cat[field], dtype=np.float64)
    return {">": col > val, ">=": col >= val, "<": col < val,
            "<=": col <= val, "=": col == val}[op]


def mass_threshold_mask(cat: dict, log10_threshold: float):
    """Markers visible only above 10^threshold Msun (Section 4.F)."""
    import numpy as np

    return np.asarray(cat["M_halo"], dtype=np.float64) > 10.0**log10_threshold


def sort_halo_indices(cat: dict, column: str, descending: bool):
    """Row order for the table sort (stable argsort)."""
    import numpy as np

    col = np.asarray(cat[column], dtype=np.float64)
    order = np.argsort(col, kind="stable")
    return order[::-1] if descending else order


def halos_to_csv(path: str, cat: dict, mask=None) -> str:
    import csv

    import numpy as np

    if mask is None:
        mask = np.ones(len(cat["halo_id"]), dtype=bool)
    idx = np.flatnonzero(mask)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HALO_COLUMNS + ["x_kpc", "y_kpc", "z_kpc"])
        for i in idx:
            w.writerow([int(cat["halo_id"][i]), int(cat["type"][i]),
                        f"{cat['M_halo'][i]:.6g}",
                        f"{cat['M_star'][i]:.6g}",
                        f"{cat['SFR'][i]:.6g}", f"{cat['R_200'][i]:.6g}",
                        f"{cat['x'][i]:.6g}", f"{cat['y'][i]:.6g}",
                        f"{cat['z'][i]:.6g}"])
    return path
