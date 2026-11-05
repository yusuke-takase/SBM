"""Detectors: gains, bandpasses and polarization angles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..basis import POLARIZATION, POLARIZATION_HWP
from ..signal_model import SignalModel
from ..sky import Sky
from ._registry import systematic


@systematic(data="pair", templates=[(0, 0)])
def diff_gain(sky: Sky, g_t: float, g_b: float) -> SignalModel:
    """Gain errors ``g_t``, ``g_b`` of the two detectors of a pair difference.

    ``d = ((1 + g_t) d_T - (1 + g_b) d_B) / 2`` adds ``(g_t - g_b) I / 2`` at
    ``(0, 0)`` and rescales the polarization by ``(g_t + g_b) / 2``. Template:
    ``(0, 0)``.
    """
    return SignalModel(
        {(0, 0): (g_t - g_b) / 2.0 * sky.i, POLARIZATION: (g_t + g_b) / 4.0 * sky.p}
    )


@systematic(data="pair", templates=[(0, 0)])
def bandpass_mismatch(
    components: Sequence[np.ndarray],
    gamma_t: Sequence[float],
    gamma_b: Sequence[float],
) -> SignalModel:
    """Bandpass mismatch of the two detectors of a pair difference
    (Hoang et al. 2017, JCAP 12, 015, Sect. 3.2).

    Each foreground temperature component ``c_i`` leaks into the pair difference
    as ``(gamma_t_i - gamma_b_i) c_i / 2`` at ``(0, 0)``. Template: ``(0, 0)``.
    """
    if not (len(components) == len(gamma_t) == len(gamma_b)):
        raise ValueError("components, gamma_t and gamma_b must have the same length")
    if not components:
        raise ValueError("no components")
    leak = np.zeros(np.shape(components[0]))
    for comp, gt, gb in zip(components, gamma_t, gamma_b):
        leak = leak + 0.5 * (gt - gb) * np.asarray(comp)
    return SignalModel({(0, 0): leak})


@systematic(data="any")
def polarization_angle(sky: Sky, delta_rad: float, hwp: bool = False) -> SignalModel:
    """Error ``delta_rad`` on the polarization angle of the detector(s).

    ``delta_rad`` is an error on litebird_sim's ``pol_angle_rad``: without HWP
    the polarizer is seen at ``psi' + alpha`` (``psi' = -psi``), so the
    polarization is rotated by ``e^{-2i delta}`` at ``(2, 0)``, for a single
    detector and for a pair difference whose two detectors share the error;
    behind a HWP at ``2 phi - alpha + psi'``, so it is rotated by ``e^{+2i delta}``
    at ``(2, -4)``. Template: none (a rotation of P is not separable from P).
    """
    if hwp:
        return SignalModel(
            {POLARIZATION_HWP: (np.exp(2j * delta_rad) - 1.0) * sky.p / 2.0}
        )
    return SignalModel({POLARIZATION: (np.exp(-2j * delta_rad) - 1.0) * sky.p / 2.0})
