"""Optics: the beam."""

from __future__ import annotations

import numpy as np

from ..beam import convolve
from ..hmaps import canonical
from ..signal_model import SignalModel
from ._registry import systematic


@systematic(data="any")
def beam(
    sky_alm,
    beam_alm,
    nside: int,
    spin_k=(0, 2, 4),
    *,
    hwp: bool = False,
    mueller=None,
    low_ell: str = "narrow_beam",
) -> SignalModel:
    """Signal of the sky seen through an arbitrary beam (and HWP Mueller matrix).

    This is the whole signal, to be used instead of :func:`sky_signal`; see
    :func:`sbm.beam.convolve` for the arguments. The h-maps must hold the
    couples between the basis and the components, up to spin
    ``max(spin_k) + 2`` (``max(spin_k) + 4`` with the template ``(4, 0)`` or
    ``(4, -4)``).
    """
    maps = convolve(
        sky_alm,
        beam_alm,
        nside,
        spin_k,
        hwp=hwp,
        mueller=mueller,
        low_ell=low_ell,
    )
    components = {}
    for (n, m), value in maps.items():
        key, conj = canonical(n, m)
        if not conj and np.any(value):
            components[key] = value
    if not components:
        components[(0, 0)] = np.zeros(next(iter(maps.values())).shape)
    if (0, 0) in components:
        components[(0, 0)] = components[(0, 0)].real
    return SignalModel(components)
