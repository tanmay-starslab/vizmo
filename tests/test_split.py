"""Split-screen RenderState independence tests (Section 5.F)."""

from vizmo.wgpu_renderer import RenderState, SPLIT_MODES


def test_render_state_defaults_and_modes():
    st = RenderState()
    assert st.field == "Masses"
    assert st.colormap == "magma"
    assert st.log_scale == 1
    assert SPLIT_MODES == [None, "lr", "tb"]


def test_render_state_independence():
    left = RenderState()
    right = RenderState(colormap="viridis")
    # Mutating the left view must never leak into the right view.
    left.field = "Temperature"
    left.colormap = "inferno"
    left.qty_min, left.qty_max = 4.0, 7.0
    assert right.field == "Masses"
    assert right.colormap == "viridis"
    assert right.qty_min == -1.0 and right.qty_max == 3.0
    # And vice versa.
    right.field = "RadialVelocity"
    assert left.field == "Temperature"
