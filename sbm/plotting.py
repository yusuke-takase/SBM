"""Plotting helpers."""

from pathlib import Path

import numpy as np


def get_cmap():
    """The colour map of the Planck papers (needs matplotlib)."""
    from matplotlib.colors import ListedColormap

    color_data = np.loadtxt(
        Path(__file__).parent / "datautils" / "Planck_Parchment_RGB.txt"
    )
    cmap = ListedColormap(color_data / 255.0)
    cmap.set_bad("gray")
    cmap.set_under("white")
    return cmap
