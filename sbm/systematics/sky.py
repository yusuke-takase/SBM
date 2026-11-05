"""The sky seen by an ideal detector."""

from __future__ import annotations

from ..basis import POLARIZATION, POLARIZATION_HWP
from ..signal_model import SignalModel
from ..sky import Sky
from ._registry import systematic


@systematic(data="any")
def sky_signal(
    sky: Sky, hwp: bool = False, pair_difference: bool = False
) -> SignalModel:
    """Signal of the sky seen by an ideal detector (or pair difference).

    ``I`` at ``(0, 0)`` and ``P / 2`` at ``(2, 0)``, or at ``(2, -4)`` with HWP.
    A pair difference has no intensity.
    """
    if hwp and pair_difference:
        raise ValueError("pair differences are modelled without HWP")
    pol = POLARIZATION_HWP if hwp else POLARIZATION
    components = {pol: sky.p / 2.0}
    if not pair_difference:
        components[(0, 0)] = sky.i
    return SignalModel(components)
