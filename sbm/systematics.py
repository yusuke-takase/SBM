"""Signal models of the sky and of the systematics.

Each function returns a :class:`.SignalModel`; the model of a detector is the
sum of the sky signal and of the perturbations of its systematics, e.g.::

    model = sky_signal(sky, pair_difference=True) + diff_gain(sky, 0.01, -0.01)

The spins follow :mod:`sbm.signal_model`. The HWP angle ``phi`` is the angle of
the HWP with respect to the telescope, counted opposite to the crossing angle
``psi``, so that the polarization is modulated as ``cos(4 phi - 2 psi)`` and
sits at spin ``(2, -4)``.

Pair differences ``d = (d_T - d_B) / 2`` are taken between two detectors in the
same position with orthogonal polarizers (B rotated by 90 degrees), without
HWP. The pointing offsets are measured in the frame of the detector: an offset
of size ``rho`` in direction ``chi`` samples the sky at the point displaced by
``rho`` along ``-(sin(alpha) e_theta + cos(alpha) e_phi)`` with
``alpha = psi + chi`` (see :mod:`sbm.sky`).

Each docstring lists the template couples that the map-maker can add to the
basis to absorb the systematic.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .basis import POLARIZATION, POLARIZATION_HWP
from .beam import convolve
from .hmaps import canonical
from .signal_model import SignalModel
from .sky import Sky


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


def diff_gain(sky: Sky, g_t: float, g_b: float) -> SignalModel:
    """Gain errors ``g_t``, ``g_b`` of the two detectors of a pair difference.

    ``d = ((1 + g_t) d_T - (1 + g_b) d_B) / 2`` adds ``(g_t - g_b) I / 2`` at
    ``(0, 0)`` and rescales the polarization by ``(g_t + g_b) / 2``. Template:
    ``(0, 0)``.
    """
    return SignalModel(
        {(0, 0): (g_t - g_b) / 2.0 * sky.i, POLARIZATION: (g_t + g_b) / 4.0 * sky.p}
    )


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


def abs_pointing(
    sky: Sky, rho_rad: float, chi_rad: float, second_order: bool = False
) -> SignalModel:
    """Pointing offset of a single detector behind a HWP.

    The offset is fixed in the frame of the detector (direction ``psi + chi``).
    Templates: ``(1, 0)``, ``(3, -4)``, ``(1, -4)``.
    """
    return _offset_components(sky, rho_rad, chi_rad, 1, 0, second_order)


def hwp_wedge(
    sky: Sky, rho_rad: float, chi_rad: float, second_order: bool = False
) -> SignalModel:
    """Deflection of the beam by a wedged HWP, for a single detector.

    The deflection rotates with the HWP: on the sky it points along
    ``psi - phi + chi``. Templates: ``(1, -1)``, ``(3, -5)``, ``(1, -3)``.
    """
    return _offset_components(sky, rho_rad, chi_rad, 1, -1, second_order)


def hwp_ip(sky: Sky, epsilon: float, phase_rad: float) -> SignalModel:
    """Instrumental polarization of the HWP at 4f, for a single detector.

    ``d = epsilon I cos(4 phi - phase)``: the leakage depends only on the HWP
    angle and sits at spin ``(0, 4)``. Template: ``(0, 4)``.
    """
    return SignalModel({(0, 4): epsilon / 2.0 * np.exp(-1j * phase_rad) * sky.i})


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


def beam(
    sky_alm,
    beam_alm,
    nside: int,
    spin_k=(0, 2, 4),
    *,
    hwp: bool = False,
    mueller=None,
    lowell_hwp: str = "shifted",
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
        lowell_hwp=lowell_hwp,
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
