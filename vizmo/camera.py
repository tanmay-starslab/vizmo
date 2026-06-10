"""6DOF fly-through camera with WASD + mouse controls."""

import numpy as np
import glfw


class Camera:
    def __init__(self, position=None, fov=90.0, aspect=16 / 9, near=1e-6, far=1e6):
        # Stored as float64 so sub-f32-ULP movement at large absolute
        # coordinates (cosmological scenes) isn't quantized at the
        # source. The renderer's splat path packs this into a hi/lo
        # f32 pair before sending to the GPU.
        self.position = np.array(position if position is not None else [0, 0, 0], dtype=np.float64)
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        # Orientation as forward/up vectors (right derived from cross product)
        self._forward = np.array([0, 0, -1], dtype=np.float32)
        self._up = np.array([0, 1, 0], dtype=np.float32)

        # Cached basis vectors (recomputed when _dirty is set)
        self._cached_forward = None
        self._cached_right = None
        self._cached_up = None
        self._dirty = True

        # Movement
        self.speed = 1.0  # units/sec, will be auto-scaled from data
        self.mouse_sensitivity = 0.002
        self.invert_mouse = True
        self.roll_speed = 0.75  # rad/sec

        # Input state
        self._keys = set()
        self._last_cursor = None
        self._mouse_captured = False
        self._moving = False

        # Eased fly-to transition state (None when inactive)
        self._transition = None
        # Orbit autopilot: dict(center, axis, radius, angular_speed) or None
        self.orbit = None

    def _recompute_basis(self):
        """Recompute and cache the orthonormal basis vectors."""
        f = self._forward / np.linalg.norm(self._forward)
        r = np.cross(f, self._up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        # u is already unit length since r and f are orthonormal
        self._cached_forward = f
        self._cached_right = r
        self._cached_up = u
        self._dirty = False

    @property
    def forward(self):
        if self._dirty:
            self._recompute_basis()
        return self._cached_forward

    @property
    def right(self):
        if self._dirty:
            self._recompute_basis()
        return self._cached_right

    @property
    def up(self):
        if self._dirty:
            self._recompute_basis()
        return self._cached_up

    def view_matrix(self):
        """Returns 4x4 view matrix (world -> camera)."""
        f = self.forward
        r = self.right
        u = self.up
        p = self.position

        m = np.eye(4, dtype=np.float32)
        m[0, :3] = r
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -np.dot(r, p)
        m[1, 3] = -np.dot(u, p)
        m[2, 3] = np.dot(f, p)
        return m

    def projection_matrix(self):
        """Returns 4x4 perspective projection matrix."""
        f = 1.0 / np.tan(np.radians(self.fov) / 2)
        n, fa = self.near, self.far
        a = self.aspect

        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / a
        m[1, 1] = f
        m[2, 2] = (fa + n) / (n - fa)
        m[2, 3] = (2 * fa * n) / (n - fa)
        m[3, 2] = -1
        return m

    def update(self, dt):
        """Process input and update position/orientation. Returns True if camera moved or rotated."""
        # Don't reset _moving here -- mouse rotation sets it during poll_events()
        moved = self._moving
        self._moving = False
        velocity = np.zeros(3, dtype=np.float64)

        if glfw.KEY_W in self._keys:
            velocity += self.forward
        if glfw.KEY_S in self._keys:
            velocity -= self.forward
        if glfw.KEY_A in self._keys:
            velocity -= self.right
        if glfw.KEY_D in self._keys:
            velocity += self.right
        if glfw.KEY_Z in self._keys:
            velocity += self.up
        if glfw.KEY_X in self._keys:
            velocity -= self.up

        manual = np.dot(velocity, velocity) > 0
        # Any manual flight cancels autopilot (transition or orbit).
        if manual:
            self._transition = None
            self.orbit = None

        if self._transition is not None and self._advance_transition(dt):
            return True
        if self.orbit is not None and not manual:
            self._advance_orbit(dt)
            return True

        if manual:
            velocity = velocity / np.linalg.norm(velocity) * self.speed * dt
            self.position += velocity
            self._moving = True

        # Roll
        if glfw.KEY_Q in self._keys:
            self._roll(-self.roll_speed * dt)
            self._moving = True
        if glfw.KEY_E in self._keys:
            self._roll(self.roll_speed * dt)
            self._moving = True

        moved = moved or self._moving
        return moved

    @property
    def is_moving(self):
        return self._moving

    def on_key(self, key, action):
        if action == glfw.PRESS:
            self._keys.add(key)
        elif action == glfw.RELEASE:
            self._keys.discard(key)

    def on_mouse_button(self, button, action):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_captured = action == glfw.PRESS

    def on_cursor(self, xpos, ypos):
        if self._last_cursor is None:
            self._last_cursor = (xpos, ypos)
            return

        if not self._mouse_captured:
            self._last_cursor = (xpos, ypos)
            return

        dx = xpos - self._last_cursor[0]
        dy = ypos - self._last_cursor[1]
        self._last_cursor = (xpos, ypos)

        if abs(dx) > 0 or abs(dy) > 0:
            # Mouse-look takes over from any autopilot.
            self._transition = None
            self.orbit = None
            sign = -1.0 if self.invert_mouse else 1.0
            self._yaw(-dx * self.mouse_sensitivity * sign)
            self._pitch(-dy * self.mouse_sensitivity * sign)
            self._moving = True

    # ------------------------------------------------------------------
    # Autopilot: eased fly-to transitions and orbit mode
    # ------------------------------------------------------------------

    def fly_to(self, position=None, look_at=None, forward=None, up=None,
               duration=1.0):
        """Begin a smooth (smoothstep-eased) transition.

        Any of position / orientation may be supplied; omitted parts
        keep their current value. `look_at` wins over `forward`.
        """
        end_pos = (np.asarray(position, dtype=np.float64)
                   if position is not None else self.position.copy())
        if look_at is not None:
            d = np.asarray(look_at, dtype=np.float64) - end_pos
            n = np.linalg.norm(d)
            end_fwd = (d / n).astype(np.float32) if n > 0 else self.forward.copy()
        elif forward is not None:
            end_fwd = np.asarray(forward, dtype=np.float32)
            end_fwd = end_fwd / max(np.linalg.norm(end_fwd), 1e-30)
        else:
            end_fwd = self.forward.copy()
        if up is not None:
            end_up = np.asarray(up, dtype=np.float32)
        else:
            end_up = self._up.copy()
        # Degenerate up (parallel to forward): pick any perpendicular.
        if abs(float(np.dot(end_up / max(np.linalg.norm(end_up), 1e-30),
                            end_fwd))) > 0.999:
            end_up = (np.array([0, 0, 1], dtype=np.float32)
                      if abs(end_fwd[2]) < 0.9
                      else np.array([0, 1, 0], dtype=np.float32))
        self.orbit = None
        self._transition = {
            "t": 0.0,
            "duration": max(float(duration), 1e-3),
            "p0": self.position.copy(),
            "p1": end_pos,
            "f0": self.forward.copy(),
            "f1": end_fwd,
            "u0": self.up.copy(),
            "u1": end_up,
        }

    @property
    def in_transit(self):
        return self._transition is not None

    def _advance_transition(self, dt):
        tr = self._transition
        tr["t"] += dt
        s = min(tr["t"] / tr["duration"], 1.0)
        e = s * s * (3.0 - 2.0 * s)  # smoothstep
        self.position = tr["p0"] * (1.0 - e) + tr["p1"] * e

        def nlerp(v0, v1, w):
            v = v0 * (1.0 - w) + v1 * w
            n = np.linalg.norm(v)
            # Antiparallel midpoint: fall back to the end vector.
            return (v / n) if n > 1e-6 else v1

        self._forward = nlerp(tr["f0"], tr["f1"], e)
        self._up = nlerp(tr["u0"], tr["u1"], e)
        self._dirty = True
        if s >= 1.0:
            self._transition = None
        return True

    def start_orbit(self, center, angular_speed=0.25):
        """Begin orbiting `center` at the current radius and elevation,
        rotating about the current up axis. Returns False when the
        camera sits on the center (no orbit radius)."""
        center = np.asarray(center, dtype=np.float64)
        rel = self.position - center
        if np.linalg.norm(rel) <= 0:
            return False
        self._transition = None
        self.orbit = {
            "center": center,
            "axis": self.up.astype(np.float64).copy(),
            "angular_speed": float(angular_speed),
        }
        return True

    def stop_orbit(self):
        self.orbit = None

    def _advance_orbit(self, dt):
        ob = self.orbit
        axis = ob["axis"]
        theta = ob["angular_speed"] * dt
        c, s = np.cos(theta), np.sin(theta)
        rel = self.position - ob["center"]
        rel_rot = (rel * c + np.cross(axis, rel) * s
                   + axis * np.dot(axis, rel) * (1.0 - c))
        self.position = ob["center"] + rel_rot
        # Keep looking at the center; preserve the orbit axis as up so
        # the horizon stays level through the full revolution.
        d = ob["center"] - self.position
        n = np.linalg.norm(d)
        if n > 0:
            self._forward = (d / n).astype(np.float32)
            self._up = axis.astype(np.float32)
            self._dirty = True

    def on_scroll(self, offset):
        self.speed *= 1.15 ** (offset / 3.0)

    def adjust_fov(self, delta_deg):
        """Change field of view, clamped to a sane perspective range.

        Lower FOV = telephoto zoom-in, higher = wide angle.
        """
        self.fov = float(np.clip(self.fov + delta_deg, 10.0, 140.0))
        return self.fov

    def _yaw(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        u = self.up
        self._forward = (
            self._forward * c + np.cross(u, self._forward) * s + u * np.dot(u, self._forward) * (1 - c)
        )
        self._dirty = True

    def _pitch(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        r = self.right
        self._forward = (
            self._forward * c + np.cross(r, self._forward) * s + r * np.dot(r, self._forward) * (1 - c)
        )
        self._up = (self._up * c + np.cross(r, self._up) * s + r * np.dot(r, self._up) * (1 - c))
        self._dirty = True

    def _roll(self, angle):
        c, s = np.cos(angle), np.sin(angle)
        f = self.forward
        self._up = self._up * c + np.cross(f, self._up) * s + f * np.dot(f, self._up) * (1 - c)
        self._dirty = True

    def auto_scale(self, positions, masses=None, boxsize=None):
        """Set speed and clip planes. Starts above the mass-weighted median
        (computed from a 100x subsample) looking down -z."""
        pmin = positions.min(axis=0)
        pmax = positions.max(axis=0)
        extent = np.linalg.norm(pmax - pmin)

        # Mass-weighted median from a 100x subsample
        n = len(positions)
        step = max(1, n // 100) if n > 100 else 1
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

        # Start at top of data looking down -z
        self.position = np.array([center[0], center[1], pmax[2]], dtype=np.float64)
        self._forward = np.array([0, 0, -1], dtype=np.float32)
        self._up = np.array([0, 1, 0], dtype=np.float32)
        self._dirty = True
        self.speed = extent / 20
        self.near = extent * 1e-6
        self.far = extent * 10
