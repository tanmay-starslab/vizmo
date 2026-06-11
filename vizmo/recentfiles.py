"""Recently-opened snapshot list, persisted to the vizmo config dir."""

from __future__ import annotations

import json
import os


class RecentFiles:
    """Most-recent-first list of snapshot paths (max 10), stored at
    ~/.config/vizmo/recents.json (XDG_CONFIG_HOME honored)."""

    max_entries = 10

    def __init__(self):
        from .session import config_dir

        self.path = os.path.join(config_dir(), "recents.json")

    def get(self) -> list:
        try:
            with open(self.path) as f:
                out = json.load(f)
            return [p for p in out if isinstance(p, str)]
        except (OSError, ValueError):
            return []

    def add(self, filepath: str) -> None:
        filepath = os.path.abspath(filepath)
        entries = [p for p in self.get() if p != filepath]
        entries.insert(0, filepath)
        entries = entries[: self.max_entries]
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(entries, f, indent=1)
            os.replace(tmp, self.path)
        except OSError:
            pass
