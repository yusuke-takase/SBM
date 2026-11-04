"""Scalar I-input ell=0,1 HWP filters with the beam multipole sum retained.

Assume a spatially uniform Mueller matrix in the beam-centred Ludwig-3 frame.
Its spin-h shape in spherical coordinates is exp(-i*h*phi), h=+/-2. Multiplying
this shape by a spin-minus-h beam and projecting onto scalar harmonics gives
the Gaunt weights below. Other input channels and ell>=2 retain the shifted
model. All filters use SBM's sqrt(4*pi/(2*ell+1)) beam normalization.
"""

from functools import lru_cache

import healpy as hp
import numpy as np


@lru_cache(maxsize=128)
def _coupling_weights(ell, n, h, lmax):
    """Cache geometry shared by all beams, skies and Mueller matrices.

    The spatial spin-h coefficient is 8*pi*N_L*(-1)^L/[L*(L+1)], L>=2.
    Triangle selection leaves L=ell1 at ell=0, or at most three L at ell=1.
    The sum therefore has O(lmax) terms for each output coefficient.
    """
    from sympy.physics.wigner import wigner_3j  # exact 3j symbols, only for this option

    weights = np.zeros(lmax + 1)
    for ell1 in range(max(2, abs(n + h)), lmax + 1):
        for ell2 in range(max(2, abs(ell1 - ell)), ell1 + ell + 1):
            spin = float(wigner_3j(ell1, ell2, ell, h, -h, 0))
            order = float(wigner_3j(ell1, ell2, ell, n + h, -h, -n))
            weights[ell1] += (
                (-1) ** (n + ell2)
                * (2 * ell1 + 1)
                * 2
                * (2 * ell2 + 1)
                / (ell2 * (ell2 + 1))
                * spin
                * order
            )
    weights.setflags(write=False)
    return weights


def _beam_source(beam_alm, n, h):
    """Read only the required signed-m spin beam, without copying all TEB alm."""
    lmax = hp.Alm.getlmax(beam_alm.shape[1])
    order = n + h
    q = abs(order)
    source = np.zeros(lmax + 1, dtype=complex)
    start = max(2, q)
    if start <= lmax:
        first = hp.Alm.getidx(lmax, start, q)
        last = hp.Alm.getidx(lmax, lmax, q) + 1
        # Positive h uses P*, negative h uses P. Negative m interchanges them.
        p_sign = -1 if h == 2 else 1
        if order < 0:
            p_sign = -p_sign
        value = -(beam_alm[1, first:last] + p_sign * 1j * beam_alm[2, first:last])
        if order < 0:
            value = (-1) ** q * value.conj()
        source[start:] = value * np.sqrt(
            4 * np.pi / (2 * np.arange(start, lmax + 1) + 1)
        )
    return source


def prepare_lowell_beam(beam_alm):
    """The three independent scalar 2f beam filters ``{(n, h): f_ell}``, ell = 0, 1.

    The n=0, h=-2 filter follows from reality and is not prepared separately.
    The geometric weights are cached across beams.
    """
    beam_alm = np.asarray(beam_alm)
    lmax = hp.Alm.getlmax(beam_alm.shape[1])
    filters = {}
    for n, h in ((0, 2), (1, 2), (1, -2)):
        source = _beam_source(beam_alm, n, h)
        result = np.zeros(2, dtype=complex)
        for ell in range(n, min(1, lmax) + 1):
            result[ell] = _coupling_weights(ell, n, h, lmax) @ source
        result.setflags(write=False)
        filters[n, h] = result
    return filters
