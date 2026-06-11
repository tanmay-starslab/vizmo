"""Timeline scrubber model tests."""

import pytest

from vizmo.series import SnapshotInfo, TimelinePlayer


def _snaps():
    return [SnapshotInfo(path=f"s{i}", redshift=z, time_gyr=t,
                         snap_num=i)
            for i, (z, t) in enumerate([(3.0, 2.1), (2.0, 3.3),
                                        (1.0, 5.9), (0.5, 8.6),
                                        (0.0, 13.8)])]


def test_playhead_positions_linear_in_z():
    tp = TimelinePlayer(_snaps())
    pos = tp.positions_px(800)
    # z=3 -> 0 px, z=0 -> 800 px; linear interpolation between.
    assert pos[0] == pytest.approx(0.0)
    assert pos[-1] == pytest.approx(800.0)
    assert pos[1] == pytest.approx(800 * (3 - 2) / 3)      # z=2
    assert pos[2] == pytest.approx(800 * (3 - 1) / 3)      # z=1
    assert pos[3] == pytest.approx(800 * (3 - 0.5) / 3)    # z=0.5
    assert tp.nearest_index(670, 800) == 3                  # near z=0.5


def test_play_timer_advances():
    tp = TimelinePlayer(_snaps(), fps=1.0)
    tp.toggle_play()
    t = 100.0
    advanced = 0
    # First tick only arms the timer.
    tp.tick(t)
    for k in range(1, 4):
        if tp.tick(t + k * 1.001):
            advanced += 1
    assert advanced == 3
    assert tp.index == 3


def test_play_stops_at_end():
    tp = TimelinePlayer(_snaps()[:2], fps=10.0)
    tp.index = 1
    tp.toggle_play()
    tp.tick(0.0)
    assert tp.tick(1.0) is False
    assert tp.playing is False


def test_keyboard_navigation():
    tp = TimelinePlayer(_snaps())
    assert tp.step(+1) and tp.index == 1
    assert tp.step(-1) and tp.index == 0
    assert tp.step(-1) is False  # clamped at first
    tp.end()
    assert tp.index == 4
    assert tp.step(+1) is False  # clamped at last
    tp.home()
    assert tp.index == 0
