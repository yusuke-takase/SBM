"""Pointing offsets: pair differences, single detectors behind a HWP, HWP wedge.

An offset of size ``rho`` in direction ``chi``, measured in the frame of the
detector, samples the sky at the point displaced by ``rho`` along
``-(sin(alpha) e_theta + cos(alpha) e_phi)`` with ``alpha = psi + chi``
(``psi - phi + chi`` for the wedge, whose deflection rotates with the HWP); see
:mod:`sbm.sky` for the expansion to second order.
"""

from __future__ import annotations

import numpy as np

from ..basis import POLARIZATION_HWP
from ..hmaps import canonical
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
    u_t = rho_t_rad * np.exp(1j * chi_t_rad)
    u_b = 1j * rho_b_rad * np.exp(1j * chi_b_rad)
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
    """Offset of a single detector behind a HWP, in direction ``alpha + chi``
    with ``alpha = dn psi + dm phi``."""
    pn, pm = POLARIZATION_HWP
    components = {
        (dn, dm): -rho / 2.0 * np.exp(1j * chi) * sky.eth_i,
        (pn - dn, pm - dm): -rho / 4.0 * np.exp(-1j * chi) * sky.eth_bar_p,
        (pn + dn, pm + dm): -rho / 4.0 * np.exp(1j * chi) * sky.eth_p,
    }
    if second_order:
        so = sky.second_order
        u = rho * np.exp(1j * chi)
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


@systematic(data="single_hwp", templates=[(1, 0), (3, -4), (1, -4)])
def abs_pointing(
    sky: Sky, rho_rad: float, chi_rad: float, second_order: bool = False
) -> SignalModel:
    """Pointing offset of a single detector behind a HWP.

    The offset is fixed in the frame of the detector (direction ``psi + chi``).
    Templates: ``(1, 0)``, ``(3, -4)``, ``(1, -4)``.
    """
    return _offset_components(sky, rho_rad, chi_rad, 1, 0, second_order)


@systematic(data="single_hwp", templates=[(1, -1), (3, -5), (1, -3)])
def hwp_wedge(
    sky: Sky, rho_rad: float, chi_rad: float, second_order: bool = False
) -> SignalModel:
    """Deflection of the beam by a wedged HWP, for a single detector.

    The deflection rotates with the HWP: on the sky it points along
    ``psi - phi + chi``. Templates: ``(1, -1)``, ``(3, -5)``, ``(1, -3)``.
    """
    return _offset_components(sky, rho_rad, chi_rad, 1, -1, second_order)
