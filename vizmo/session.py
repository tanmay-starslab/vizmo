"""Per-user persistence: config dir, camera bookmarks, app settings."""

import hashlib
import json
import os


def config_dir():
    base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    d = os.path.join(base, "vizmo")
    os.makedirs(d, exist_ok=True)
    return d


def _bookmark_file(snapshot_path):
    digest = hashlib.sha1(os.path.abspath(snapshot_path).encode()).hexdigest()[:16]
    d = os.path.join(config_dir(), "bookmarks")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{digest}.json")


def load_bookmarks(snapshot_path):
    """Return {slot_str: pose_dict} for this snapshot (empty when none)."""
    try:
        with open(_bookmark_file(snapshot_path)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_bookmarks(snapshot_path, bookmarks):
    path = _bookmark_file(snapshot_path)
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(bookmarks, f, indent=1)
        os.replace(tmp, path)
    except OSError as e:
        print(f"  Bookmark save failed: {e}")


def camera_pose(camera):
    """Serializable pose dict for the current camera state."""
    return {
        "position": [float(v) for v in camera.position],
        "forward": [float(v) for v in camera._forward],
        "up": [float(v) for v in camera._up],
        "speed": float(camera.speed),
        "fov": float(camera.fov),
    }


def apply_camera_pose(camera, pose):
    import numpy as np

    camera.position = np.array(pose["position"], dtype=np.float64)
    camera._forward = np.array(pose["forward"], dtype=np.float32)
    camera._up = np.array(pose["up"], dtype=np.float32)
    camera.speed = float(pose.get("speed", camera.speed))
    camera.fov = float(pose.get("fov", camera.fov))
    camera._dirty = True


def _settings_file():
    return os.path.join(config_dir(), "config.json")


def load_settings():
    """User settings persisted across sessions (empty dict when none)."""
    try:
        with open(_settings_file()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    try:
        tmp = _settings_file() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(settings, f, indent=1)
        os.replace(tmp, _settings_file())
    except OSError as e:
        print(f"  Settings save failed: {e}")
