"""The sky seen by an ideal detector."""

from __future__ import annotations

import numpy as np

from ..basis import POLARIZATION, POLARIZATION_HWP
from ..signal_model import SignalModel
from ..sky import Sky
from ._registry import systematic


@systematic(data="any")
def sky_signal(
    sky: Sky,
    hwp: bool = False,
    pair_difference: bool = False,
    pol_angle_rad: float = 0.0,
) -> SignalModel:
    """Signal of the sky seen by an ideal detector (or pair difference).

    ``I`` at ``(0, 0)`` and ``P / 2`` at :data:`~sbm.conventions.POLARIZATION`, or
    at :data:`~sbm.conventions.POLARIZATION_HWP` with HWP.
    A pair difference has no intensity.

    ``pol_angle_rad`` is the polarization angle ``xi`` of the detector, measured
    from the reference axis of the focal plane in the sense of ``psi`` (the
    ``pol_angle_rad`` of litebird_sim); the h-maps do not include it. Without
    HWP the detector sees the polarization at ``psi + xi``, so that ``P`` is
    multiplied by ``e^{-2i xi}``; behind a HWP at ``2 phi - xi + psi``, so that it
    is multiplied by ``e^{+2i xi}``. In a pair difference T is at ``xi`` and B at
    ``xi + pi/2``. The directions of the pointing offsets are measured from the
    reference axis of the focal plane, not from the polarizer.
    """
    if hwp and pair_difference:
        raise ValueError("pair differences are modelled without HWP")
    pol = POLARIZATION_HWP if hwp else POLARIZATION
    p = sky.p
    if pol_angle_rad:
        p = np.exp((2j if hwp else -2j) * pol_angle_rad) * p
    components = {pol: p / 2.0}
    if not pair_difference:
        components[(0, 0)] = sky.i
    return SignalModel(components)
