"""The half-wave plate."""

from __future__ import annotations

import numpy as np

from ..beam import IDEAL_HWP
from ..signal_model import SignalModel
from ..sky import Sky
from ._registry import systematic


@systematic(data="single_hwp", templates=[(0, 4)])
def hwp_ip(sky: Sky, epsilon: float, phase_rad: float) -> SignalModel:
    """Instrumental polarization of the HWP at 4f, for a single detector.

    ``d = epsilon I cos(4 phi + phase)``: the 4f elements
    ``M_QI = epsilon cos(4 alpha + phase)`` and ``M_UI = epsilon sin(4 alpha + phase)``
    of the HWP Mueller matrix in the focal plane, as functions of the HWP angle
    ``alpha`` in the focal plane (Patanchon et al. 2024, with ``phase = phi_QI``,
    ``phi_UI = phi_QI - pi/2`` and equal amplitudes), seen by a detector at
    polarization angle 0. Their chain is that of :func:`hwp_mueller`, with
    ``alpha = phi``. The leakage depends only on the HWP angle and sits at spin
    ``(0, 4)``. Template: ``(0, 4)``.
    """
    return SignalModel({(0, 4): epsilon / 2.0 * np.exp(1j * phase_rad) * sky.i})


#: Mueller matrix of an ideal HWP in its own frame (Stokes I, Q, U, V).


def _stokes_rotation(angle):
    """Rotation of the Stokes parameters by ``2 angle`` (Q' = cos Q + sin U), as in litebird_sim."""
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.array(
        [
            [one, zero, zero, zero],
            [zero, c, s, zero],
            [zero, -s, c, zero],
            [zero, zero, zero, one],
        ]
    )


@systematic(data="single_hwp")
def hwp_mueller(sky: Sky, harmonics: dict, rtol: float = 1e-12) -> SignalModel:
    """Non-ideal HWP Mueller matrix, for a single detector.

    The Mueller matrix in the frame of the HWP depends on the HWP angle ``phi``
    through its harmonics,

        M(phi) = M_0 + sum_{k > 0} (M_k e^{i k phi} + conj(M_k) e^{-i k phi}) ,

    and the data are those of litebird_sim's chain
    ``d = (1, 1, 0, 0) . R(phi)^T M(phi) R(phi) R(-psi) (I, Q, U, 0)``, with the
    angles of SBM (:mod:`sbm.systematics`). The function returns the perturbation
    with respect to the ideal HWP (:data:`IDEAL_HWP`), to be added to
    ``sky_signal(sky, hwp=True)``; the spins are computed exactly from the chain
    sampled on a grid of ``psi`` and ``phi``.

    Args:
        sky: the sky (only I, Q and U are used).
        harmonics: ``{k: M_k}`` for ``k >= 0``; ``M_0`` is the real 4x4 Mueller
            matrix of the HWP, ``M_k`` complex 4x4 matrices.
        rtol: components smaller than ``rtol`` times the largest are dropped.

    :func:`hwp_ip` is the case of the harmonics ``k = 2`` of ``M_QI`` and ``M_UI``.
    The templates depend on the harmonics: the intensity leaks at ``(0, k + 2)``
    and ``(0, k - 2)``.
    """
    delta = {}
    for k, matrix in harmonics.items():
        k = int(k)
        matrix = np.asarray(matrix)
        if k < 0 or matrix.shape != (4, 4):
            raise ValueError("harmonics must map k >= 0 to 4x4 matrices")
        if k == 0:
            if np.iscomplexobj(matrix) and np.any(matrix.imag != 0):
                raise ValueError("M_0 must be real")
            matrix = matrix.real - IDEAL_HWP
        delta[k] = np.asarray(matrix, dtype=complex)
    kmax = max(delta) if delta else 0
    n_psi, n_phi = 8, 4 * (kmax + 4)
    psi = 2 * np.pi * np.arange(n_psi) / n_psi
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    m_phi = np.zeros((n_phi, 4, 4))
    for k, matrix in delta.items():
        if k == 0:
            m_phi += matrix.real
        else:
            m_phi += 2 * np.real(matrix[None] * np.exp(1j * k * phi)[:, None, None])
    r_phi = np.moveaxis(_stokes_rotation(phi), -1, 0)  # (n_phi, 4, 4)
    r_psi = np.moveaxis(_stokes_rotation(-psi), -1, 0)  # (n_psi, 4, 4)
    detector = np.array([1.0, 1.0, 0.0, 0.0])
    # row vector of the chain for every (psi, phi): w[j, l, s]
    chain = np.einsum("i,lji,ljk,lkm->lm", detector, r_phi, m_phi, r_phi)  # (n_phi, 4)
    w = np.einsum("lm,jms->jls", chain, r_psi)  # (n_psi, n_phi, 4)
    coeff = np.fft.fft2(w, axes=(0, 1)) / (
        n_psi * n_phi
    )  # coefficient of e^{i(n psi + m phi)}
    stokes = (sky.i, sky.q, sky.u)
    scale = np.max(np.abs(coeff[..., :3])) if coeff.size else 0.0
    components = {}
    for n in range(0, 3):
        for m in range(-(kmax + 4), kmax + 5):
            if n == 0 and m < 0:
                continue
            c = coeff[n % n_psi, m % n_phi, :3]
            if scale == 0 or np.max(np.abs(c)) <= rtol * scale:
                continue
            value = c[0] * stokes[0] + c[1] * stokes[1] + c[2] * stokes[2]
            components[(n, m)] = value.real if (n, m) == (0, 0) else value
    if not components:
        components[(0, 0)] = np.zeros(sky.npix)
    return SignalModel(components)
