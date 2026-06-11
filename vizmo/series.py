"""Snapshot-series discovery and time ordering.

discover_snapshots() scans a directory (and one level of snapdir_*
subdirectories) for Gadget/AREPO-style HDF5 snapshots and returns them
sorted by redshift (high z first, i.e. time-ordered).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from .physics import UnitSystem, SEC_PER_GYR, KPC_CGS, KM_CGS


@dataclass
class SnapshotInfo:
    path: str
    redshift: float
    time_gyr: float       # cosmic time at this snapshot (flat LCDM)
    snap_num: int


def _cosmic_time_gyr(units: UnitSystem) -> float:
    """Age of the universe at the snapshot's scale factor (flat LCDM
    closed form); 0.0 for non-cosmological runs (caller may prefer
    Header/Time directly there)."""
    if not units.cosmological or units.omega_m <= 0 or units.omega_l <= 0:
        return 0.0
    h0_cgs = units.h * 100.0 * KM_CGS / (1000.0 * KPC_CGS)
    pref = 2.0 / (3.0 * h0_cgs * np.sqrt(units.omega_l))
    t = pref * np.arcsinh(
        np.sqrt(units.omega_l / units.omega_m) * units.a**1.5)
    return float(t / SEC_PER_GYR)


def _snap_num_from_name(name: str) -> int:
    """Snapshot number from a filename like snapshot_063.0.hdf5 -> 63.

    Extensions (and the multi-part .N suffix) are stripped first so the
    '5' in '.hdf5' can never be mistaken for the snapshot number.
    """
    base = os.path.basename(name)
    base = re.sub(r"\.(hdf5|h5)$", "", base)
    base = re.sub(r"\.\d+$", "", base)  # multi-part suffix
    m = re.findall(r"(\d+)", base)
    return int(m[-1]) if m else -1


def discover_snapshots(path: str) -> list:
    """Find snapshot HDF5 files under `path` (one snapdir level deep).

    A file qualifies when it contains a Header group and at least one
    PartTypeN group with Coordinates. Multi-part snapshots
    (snapshot_NNN.0.hdf5, .1.hdf5, ...) are collapsed to their .0 file.

    Returns SnapshotInfo list sorted by descending redshift (cosmic
    time order).
    """
    import h5py

    candidates = []
    for root in [path] + sorted(
            os.path.join(path, d) for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))):
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for n in names:
            if n.endswith((".hdf5", ".h5")):
                # Collapse multi-part: keep only ".0.hdf5" or partless.
                if re.search(r"\.(\d+)\.hdf5$", n):
                    if not n.endswith(".0.hdf5"):
                        continue
                candidates.append(os.path.join(root, n))

    out = []
    for fp in candidates:
        try:
            with h5py.File(fp, "r") as f:
                if "Header" not in f:
                    continue
                has_parts = any(
                    k.startswith("PartType") and "Coordinates" in f[k]
                    for k in f.keys())
                if not has_parts:
                    continue
                hdr = dict(f["Header"].attrs)
        except OSError:
            continue
        units = UnitSystem(hdr)
        z = float(hdr.get("Redshift", 1.0 / max(units.a, 1e-12) - 1.0))
        out.append(SnapshotInfo(
            path=fp, redshift=z, time_gyr=_cosmic_time_gyr(units),
            snap_num=_snap_num_from_name(fp)))
    out.sort(key=lambda s: -s.redshift)
    return out


class TimelinePlayer:
    """Playback + geometry model for the timeline scrubber.

    Pure logic (testable headlessly): playhead pixel positions are
    linear in redshift across [z_max, z_min]; tick() advances one
    snapshot every 1/fps seconds of monotonic time while playing.
    """

    def __init__(self, snaps, fps=24.0):
        self.snaps = list(snaps)
        self.index = 0
        self.fps = float(fps)
        self.playing = False
        self._last_advance = None

    # -- geometry -----------------------------------------------------------

    def positions_px(self, track_width):
        """Pixel x of each snapshot on a track of `track_width`,
        linear in redshift (z_max at x=0, z_min at the right edge)."""
        if not self.snaps:
            return []
        zs = [s.redshift for s in self.snaps]
        z0, z1 = max(zs), min(zs)
        span = max(z0 - z1, 1e-12)
        return [(z0 - z) / span * track_width for z in zs]

    def nearest_index(self, x_px, track_width):
        pos = self.positions_px(track_width)
        if not pos:
            return 0
        return int(min(range(len(pos)), key=lambda i: abs(pos[i] - x_px)))

    # -- navigation -----------------------------------------------------------

    def step(self, delta):
        new = max(0, min(len(self.snaps) - 1, self.index + delta))
        changed = new != self.index
        self.index = new
        return changed

    def home(self):
        return self.step(-len(self.snaps))

    def end(self):
        return self.step(len(self.snaps))

    def toggle_play(self):
        self.playing = not self.playing
        self._last_advance = None
        return self.playing

    def tick(self, now):
        """Advance while playing; returns True when the index moved.
        `now` is a monotonic timestamp (injected for testability)."""
        if not self.playing or len(self.snaps) < 2:
            return False
        if self._last_advance is None:
            self._last_advance = now
            return False
        if now - self._last_advance < 1.0 / max(self.fps, 1e-6):
            return False
        self._last_advance = now
        if self.index >= len(self.snaps) - 1:
            self.playing = False
            return False
        return self.step(+1)
