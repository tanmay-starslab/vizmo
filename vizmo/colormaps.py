"""Build RGBA colormap arrays from matplotlib colormaps."""

import numpy as np
import matplotlib.pyplot as plt


AVAILABLE_COLORMAPS = [
    "magma",
    "inferno",
    "viridis",
    "plasma",
    "coolwarm",
    "hot",
    "bone",
    "cividis",
]

# CMasher perceptually-uniform scientific colormaps (van der Velden
# 2020) — registered with matplotlib on import, so they work in
# colormap_to_texture_data and everywhere else by name.
try:
    import cmasher  # noqa: F401

    AVAILABLE_COLORMAPS += [
        "cmr.rainforest",
        "cmr.ember",
        "cmr.cosmic",
        "cmr.arctic",
        "cmr.dusk",
        "cmr.eclipse",
        "cmr.freeze",
        "cmr.sunburst",
    ]
except ImportError:
    pass


def colormap_to_texture_data(name, n=256):
    """Convert a matplotlib colormap to an RGBA uint8 array of shape (n, 4)."""
    cmap = plt.get_cmap(name)
    x = np.linspace(0, 1, n)
    return (cmap(x) * 255).astype(np.uint8)
