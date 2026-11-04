# -*- encoding: utf-8 -*-
"""Spin-based map-making (SBM).

The scan of a detector is described by its orientation functions
(:class:`HMaps`), its data by the spin components of a signal model
(:class:`SignalModel`, built from :mod:`sbm.systematics`), and the binned
map-maker solves for the components of a :class:`Basis` in every pixel
(:func:`map_make`).
"""

from . import beam, systematics
from .basis import Basis
from .harmonics import SphericalHarmonics
from .hmaps import HMaps
from .likelihood import forecast
from .map_making import (
    MapMaker,
    MapMakingResult,
    map_make,
    noise_covariance,
    noise_maps,
    orientation_matrix,
)
from .plotting import get_cmap
from .signal_model import SignalModel, SpinComponent
from .sky import Sky
from .spectra import (
    c2d,
    d2c,
    generate_cmb,
    load_fiducial_cl,
    load_fiducial_cl_lens,
    load_fiducial_cl_tens,
)
from .version import __author__, __version__

__all__ = [
    "__author__",
    "__version__",
    "Basis",
    "HMaps",
    "MapMaker",
    "MapMakingResult",
    "SignalModel",
    "SphericalHarmonics",
    "Sky",
    "SpinComponent",
    "beam",
    "c2d",
    "d2c",
    "forecast",
    "generate_cmb",
    "get_cmap",
    "load_fiducial_cl",
    "load_fiducial_cl_lens",
    "load_fiducial_cl_tens",
    "map_make",
    "noise_covariance",
    "noise_maps",
    "orientation_matrix",
    "systematics",
]
