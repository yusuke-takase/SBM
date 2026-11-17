"""Pointing offsets: pair differences, single detectors behind a HWP, HWP wedge.

An offset of size ``rho`` in direction ``chi``, measured in the frame of the
detector, samples the sky at the point displaced in the direction
``alpha = psi + chi`` (for the wedge, whose deflection turns with the HWP, the
angle of the HWP on the sky plus ``chi``); the direction and the expansion to
second order are those of :mod:`sbm.conventions`.
"""

from __future__ import annotations

import numpy as np

from ..conventions import POLARIZATION_HWP, SKY_HWP_ANGLE, canonical, phase_factor
from ..signal_model import SignalModel
from ..sky import Sky
from ._registry import systematic


@systematic(data="pair", templates=[(1, 0), (3, 0)])
def diff_pointing(
    sky: Sky,
    rho_t_rad: float,
    chi_t_rad: float,
    rho_b_rad: float,
    chi_b_rad: float,
    second_order: bool = False,
) -> SignalModel:
    """Pointing offsets of the two detectors of a pair difference.

    The offsets of T and B are ``u_t = rho_t e^{i chi_t}`` and
    ``u_b = i rho_b e^{i chi_b}`` (B is rotated by 90 degrees). The temperature
    gradient enters with ``zeta = u_t - u_b`` and weight 1/4, the polarization
    gradient with ``o_zeta = u_t + u_b`` and weight 1/8. With ``second_order``
    the O(rho^2) terms are included. Templates: ``(1, 0)``, ``(3, 0)``.
    """
    u_t = rho_t_rad * phase_factor(chi_t_rad)
    u_b = 1j * rho_b_rad * phase_factor(chi_b_rad)
    zeta = u_t - u_b
    o_zeta = u_t + u_b
    components = {
        (1, 0): -1.0 / 4.0 * zeta * sky.eth_i
        - 1.0 / 8.0 * np.conj(o_zeta) * sky.eth_bar_p,
        (3, 0): -1.0 / 8.0 * o_zeta * sky.eth_p,
    }
    if second_order:
        so = sky.second_order
        components[(0, 0)] = (abs(u_t) ** 2 - abs(u_b) ** 2) / 8.0 * np.real(
            so["lap_i"]
        ) + 2.0 * np.real(
            (np.conj(u_t) ** 2 + np.conj(u_b) ** 2) / 32.0 * so["eth_bar2_p"]
        )
        components[(2, 0)] = (u_t**2 - u_b**2) / 16.0 * so["eth2_i"] + (
            abs(u_t) ** 2 + abs(u_b) ** 2
        ) / 16.0 * so["lap_p"]
        components[(4, 0)] = (u_t**2 + u_b**2) / 32.0 * so["eth2_p"]
    return SignalModel(components)


def _offset_components(sky, rho, chi, dn, dm, second_order):
    """Offset ``rho`` of a single detector behind a HWP, in the direction
    ``alpha = dn psi + dm phi + chi`` on the sky: ``(dn, dm)`` are the
    coefficients of ``psi`` and ``phi`` in the direction, ``(1, 0)`` for an
    offset fixed in the frame of the detector and
    :data:`~sbm.conventions.SKY_HWP_ANGLE` for the wedge. ``eth I`` sits at
    ``(dn, dm)``, and the gradients of ``P``, which sits at
    ``(pn, pm) =`` :data:`~sbm.conventions.POLARIZATION_HWP`, at
    ``(pn - dn, pm - dm)`` (``eth_bar P``) and ``(pn + dn, pm + dm)``
    (``eth P``)."""
    pn, pm = POLARIZATION_HWP
    components = {
        (dn, dm): -rho / 2.0 * phase_factor(chi) * sky.eth_i,
        (pn - dn, pm - dm): -rho / 4.0 * phase_factor(-chi) * sky.eth_bar_p,
        (pn + dn, pm + dm): -rho / 4.0 * phase_factor(chi) * sky.eth_p,
    }
    if second_order:
        so = sky.second_order
        u = rho * phase_factor(chi)
        extra = {
            (0, 0): rho**2 / 4.0 * np.real(so["lap_i"]),
            (pn, pm): rho**2 / 8.0 * so["lap_p"],
            (2 * dn, 2 * dm): u**2 / 8.0 * so["eth2_i"],
            (pn + 2 * dn, pm + 2 * dm): u**2 / 16.0 * so["eth2_p"],
            (pn - 2 * dn, pm - 2 * dm): np.conj(u) ** 2 / 16.0 * so["eth_bar2_p"],
        }
        for couple, value in extra.items():
            key, conj = canonical(*couple)
            value = np.conj(value) if conj else value
            components[key] = components[key] + value if key in components else value
    return SignalModel(components)


def _offset_templates(dn, dm):
    """The first-order couples of :func:`_offset_components`, ``eth I`` first."""
    pn, pm = POLARIZATION_HWP
    couples = [(dn, dm), (pn + dn, pm + dm), (pn - dn, pm - dm)]
    return [canonical(*c)[0] for c in couples]


@systematic(data="single_hwp", templates=_offset_templates(1, 0))
def abs_pointing(
    sky: Sky, rho_rad: float, chi_rad: float, second_order: bool = False
) -> SignalModel:
    """Pointing offset of a single detector behind a HWP.

    The offset is fixed in the frame of the detector (direction ``psi + chi``).
    """
    return _offset_components(sky, rho_rad, chi_rad, 1, 0, second_order)


@systematic(data="single_hwp", templates=_offset_templates(*SKY_HWP_ANGLE))
def hwp_wedge(
    sky: Sky, rho_rad: float, chi_rad: float, second_order: bool = False
) -> SignalModel:
    """Deflection of the beam by a wedged HWP, for a single detector.

    The deflection turns with the HWP: on the sky it points along the angle of
    the HWP on the sky (:data:`~sbm.conventions.SKY_HWP_ANGLE`) plus ``chi``.
    """
    return _offset_components(sky, rho_rad, chi_rad, *SKY_HWP_ANGLE, second_order)
