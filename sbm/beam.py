"""Beam convolution in spin space.

:func:`convolve` returns the spin components ``F_{n,m}`` of the signal of a
detector with an arbitrary beam, ``d = sum_{n,m} F_{n,m} exp(i (n psi + m phi))``
in the convention of :mod:`sbm.signal_model`. Without HWP only ``m = 0``
appears. With a rotating HWP of real Mueller matrix ``M`` the narrow-beam
(shifted-beam) approximation is used: the HWP harmonic is ``m`` and the input
beam order ``n + m``.
"""

from __future__ import annotations

import healpy as hp
import numpy as np

IDEAL_HWP = np.diag([1.0, 1.0, -1.0, -1.0])
_ROOT2 = np.sqrt(2.0)


def _check_alm(alm, name):
    alm = np.asarray(alm)
    if alm.ndim != 2 or alm.shape[0] not in (3, 4):
        raise ValueError(f"{name} must have TEB or TEBV components")
    lmax = hp.Alm.getlmax(alm.shape[1])
    if lmax < 0:
        raise ValueError(f"{name} must use the full HEALPix m-major layout")
    return alm, lmax


def _check_mueller(mueller):
    matrix = np.asarray(mueller)
    if matrix.shape != (4, 4) or not np.isrealobj(matrix):
        raise ValueError("mueller must be a real 4x4 Stokes Mueller matrix")
    matrix = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("mueller must contain only finite values")
    return matrix.copy()


def _check_spins(spin_k):
    spins = np.asarray(spin_k)
    if (
        spins.ndim != 1
        or not np.issubdtype(spins.dtype, np.number)
        or not np.isrealobj(spins)
        or not np.all(np.isfinite(spins))
        or np.any(spins != np.floor(spins))
    ):
        raise ValueError("spin_k must be a sequence of integer spin moments")
    return sorted({abs(int(k)) for k in spins})


def _to_spin(alm):
    """E/B convention of HEALPix to spin harmonics ``-(E +/- iB)``."""
    out = np.zeros(alm.shape, np.complex128)
    out[0] = alm[0]
    out[1] = -(alm[1] + 1j * alm[2])
    out[2] = -(alm[1] - 1j * alm[2])
    if alm.shape[0] == 4:
        out[3] = alm[3]
    return out


def beam_transfer(beam_alm, k: int, spin_m: int = 0) -> np.ndarray:
    """Beam coefficients ``b_{l,k+spin_m}`` of the convolution integrals.

    Args:
        beam_alm: T, E, B (and V) harmonic coefficients of the beam.
        k: non-negative spin moment of the crossing angle; the spin ``-k`` maps
            are the complex conjugates of the spin ``k`` maps.
        spin_m: HWP harmonic; the beam order is ``k + spin_m``.
    """
    beam_alm, lmax = _check_alm(beam_alm, "beam_alm")
    if k < 0:
        raise ValueError("k must be non-negative")
    blk = np.zeros((beam_alm.shape[0], lmax + 1), np.complex128)
    order = k + spin_m
    q = abs(order)
    if q > lmax:
        return blk
    start = hp.Alm.getidx(lmax, q, q)
    stop = hp.Alm.getidx(lmax, lmax, q)
    ell = np.arange(lmax + 1)
    blk[:, q:] = _to_spin(beam_alm)[:, start : stop + 1] * np.sqrt(
        4 * np.pi / (2 * ell[q:] + 1)
    )
    if order < 0:
        # b^P_{l,-q} = (-1)^q conj(b^{P*}_{l,q}), and conversely.
        dual = [0, 2, 1, 3] if blk.shape[0] == 4 else [0, 2, 1]
        blk = (-1) ** q * blk[dual].conj()
    return blk


def _clm(sky_spin, blk):
    """Harmonic coefficients of the spin-k maps, separately for the temperature
    and the polarization of the sky."""
    s0_pk = hp.almxfl(sky_spin[0], blk[0])
    s2_pk = hp.almxfl(sky_spin[1], blk[2]) + hp.almxfl(sky_spin[2], blk[1])
    # (-1)^m conj(c_{l,-m}), needed to synthesize a complex spin map
    s0_mk = hp.almxfl(sky_spin[0], np.conj(blk[0]))
    s2_mk = hp.almxfl(sky_spin[1], np.conj(blk[1])) + hp.almxfl(
        sky_spin[2], np.conj(blk[2])
    )
    if sky_spin.shape[0] == 4 and blk.shape[0] == 4:
        s0_pk += hp.almxfl(sky_spin[3], blk[3])
        s0_mk += hp.almxfl(sky_spin[3], np.conj(blk[3]))
    return (
        [-(s0_pk + s0_mk) / 2, -(s0_pk - s0_mk) / (2j)],
        [-(s2_pk + s2_mk) / 2, -(s2_pk - s2_mk) / (2j)],
    )


def _spin_map(clm, k, nside, lmax):
    """One complex spin-k map from the E/B-like pair returned by :func:`_clm`."""
    if k == 0:
        out = hp.alm2map(-clm[0], nside=nside, lmax=lmax, mmax=lmax).astype(complex)
        # unmodulated k=0 maps are real; an HWP harmonic at k=0 need not be
        if np.any(clm[1]):
            out += 1j * hp.alm2map(-clm[1], nside=nside, lmax=lmax, mmax=lmax)
        return out
    maps = hp.alm2map_spin([clm[0], clm[1]], nside=nside, spin=k, lmax=lmax, mmax=lmax)
    return maps[0] + 1j * maps[1]


def _hwp_transfer(blk, spin_m, coherence, lmax):
    """Apply one HWP harmonic to the beam coefficients.

    ``coherence`` is ``T M T^dagger`` in the (I, P, P*, V) basis. The sqrt(2)
    factors account for the unnormalized P/P* coefficients of :func:`_clm` and
    its polarized contribution being divided by two when the map is assembled.
    """
    beam = np.zeros((4, lmax + 1), dtype=complex)
    beam[: blk.shape[0]] = blk
    beam[1:3, :2] = 0.0  # a spin +/-2 harmonic has no ell = 0, 1 coefficients
    b_i, b_p, b_pstar, b_v = beam
    out = np.zeros_like(beam)
    c = coherence
    if spin_m == 0:
        out[0] = c[0, 0] * b_i + c[3, 0] * b_v
        out[1] = c[2, 2] * b_p
        out[2] = c[1, 1] * b_pstar
        out[3] = c[0, 3] * b_i + c[3, 3] * b_v
    elif spin_m == 2:
        out[0] = c[1, 0] * b_pstar / _ROOT2
        out[1] = _ROOT2 * (c[0, 2] * b_i + c[3, 2] * b_v)
        out[3] = c[1, 3] * b_pstar / _ROOT2
    elif spin_m == -2:
        out[0] = c[2, 0] * b_p / _ROOT2
        out[2] = _ROOT2 * (c[0, 1] * b_i + c[3, 1] * b_v)
        out[3] = c[2, 3] * b_p / _ROOT2
    elif spin_m == 4:
        out[1] = c[1, 2] * b_pstar
    elif spin_m == -4:
        out[2] = c[2, 1] * b_p
    else:
        raise ValueError("spin_m must be 0, +/-2 or +/-4")
    out[1:3, :2] = 0.0  # likewise, the polarized sky starts at ell = 2
    return out


def _coherence(matrix):
    basis = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / _ROOT2, 1j / _ROOT2, 0],
            [0, 1 / _ROOT2, -1j / _ROOT2, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )
    coherence = basis @ matrix @ basis.conj().T
    # Sum the Q/U entries before normalizing, to keep the exact zeros of the
    # identity and of the ideal HWP independently of the BLAS implementation.
    c_pp = (matrix[1, 1] + matrix[2, 2] + 1j * (matrix[2, 1] - matrix[1, 2])) / 2
    c_ppstar = (matrix[1, 1] - matrix[2, 2] + 1j * (matrix[2, 1] + matrix[1, 2])) / 2
    coherence[1:3, 1:3] = [[c_pp, c_ppstar], [c_ppstar.conj(), c_pp.conj()]]
    return coherence


def convolve(
    sky_alm,
    beam_alm,
    nside: int,
    spin_k=(0, 2, 4),
    *,
    hwp: bool = False,
    mueller=None,
    lowell_hwp: str = "shifted",
) -> dict:
    """Spin components of the beam-convolved signal, ``{(n, m): F_{n,m}}``.

    Args:
        sky_alm: T, E, B (and V) harmonic coefficients of the sky.
        beam_alm: T, E, B (and V) harmonic coefficients of the beam, with the
            same ``lmax``. The beam need not be axially symmetric.
        nside: resolution of the maps.
        spin_k: crossing-angle moments ``|n|`` to retain (0 is always retained).
        hwp: rotating HWP (narrow-beam approximation).
        mueller: real 4x4 Mueller matrix of the HWP; an ideal HWP by default.
        lowell_hwp: ``"full_sum"`` keeps the beam multipole sum for the
            ``ell = 0, 1`` 2f terms of the temperature (see :mod:`sbm.hwp_lowell`);
            the default ``"shifted"`` keeps the narrow-beam approximation.

    Both ``(n, m)`` and ``(-n, -m)`` are returned. Without HWP only ``m = 0``
    appears; with HWP the harmonics are 0, +/-2 and +/-4, and those that vanish
    identically are omitted.
    """
    sky_alm, lmax = _check_alm(sky_alm, "sky_alm")
    beam_alm, beam_lmax = _check_alm(beam_alm, "beam_alm")
    if beam_lmax != lmax:
        raise ValueError("sky and beam must have the same lmax")
    spins = sorted(set(_check_spins(spin_k)) | {0})
    if mueller is not None and not hwp:
        raise ValueError("mueller requires hwp=True")
    if lowell_hwp not in ("shifted", "full_sum"):
        raise ValueError("lowell_hwp must be 'shifted' or 'full_sum'")
    if lowell_hwp != "shifted" and not hwp:
        raise ValueError("lowell_hwp='full_sum' requires hwp=True")
    sky_spin = _to_spin(sky_alm)
    npix = hp.nside2npix(nside)
    out = {}
    if not hwp:
        for k in spins:
            clm = _clm(sky_spin, beam_transfer(beam_alm, k))
            f = (
                _spin_map(clm[0], k, nside, lmax)
                + _spin_map(clm[1], k, nside, lmax) / 2.0
            )
            if k == 0:
                out[0, 0] = f.real.astype(complex)
            else:
                out[k, 0] = f
                out[-k, 0] = f.conj()
        return out

    matrix = IDEAL_HWP if mueller is None else _check_mueller(mueller)
    coherence = _coherence(matrix)
    stokes_spins = np.array([0, 2, -2, 0])
    differences = stokes_spins[:, None] - stokes_spins[None, :]
    harmonics = [h for h in (0, 2, -2, 4, -4) if np.any(coherence[differences == h])]
    lowell_beam = None
    lowell_sky = np.any(sky_alm[0, : min(2, lmax + 1)])
    if lmax >= 1:
        lowell_sky |= sky_alm[0, hp.Alm.getidx(lmax, 1, 1)] != 0
    if (
        lowell_hwp == "full_sum"
        and lowell_sky
        and np.any(coherence[1:3, 0])
        and lmax >= 2
    ):
        from .hwp_lowell import prepare_lowell_beam

        lowell_beam = prepare_lowell_beam(beam_alm)
    requested = set(spins)
    moments = requested | {0, 1} if lowell_beam is not None else requested
    for k in sorted(moments):
        for spin_m in harmonics:
            if k == 0 and spin_m < 0:
                continue
            f = np.zeros(npix, dtype=complex)
            if k <= lmax:
                if k in requested:
                    blk = _hwp_transfer(
                        beam_transfer(beam_alm, k, spin_m), spin_m, coherence, lmax
                    )
                else:
                    blk = np.zeros((4, lmax + 1), dtype=complex)
                if lowell_beam is not None and (k, spin_m) in lowell_beam:
                    row = 1 if spin_m == 2 else 2
                    blk[0, :2] = coherence[row, 0] / _ROOT2 * lowell_beam[k, spin_m]
                blk[:, :k] = 0.0
                if np.any(blk):
                    clm = _clm(sky_spin, blk)
                    f = (
                        _spin_map(clm[0], k, nside, lmax)
                        + _spin_map(clm[1], k, nside, lmax) / 2.0
                    )
            if k == 0 and spin_m == 0:
                f = f.real.astype(complex)
            out[k, spin_m] = f
            if k != 0 or spin_m != 0:
                out[-k, -spin_m] = f.conj()
    return out


def elliptical_beam(nside: int, fwhm_rad: float, q: float) -> np.ndarray:
    """I, Q and U maps of an elliptical Gaussian beam at the north pole.

    Args:
        nside: resolution of the maps.
        fwhm_rad: full width at half maximum of the major axis.
        q: ratio of the minor to the major axis.

    The maps integrate to one; ``healpy.map2alm`` gives the ``beam_alm`` of
    :func:`convolve`.
    """
    sigma = fwhm_rad / (2.0 * np.sqrt(2 * np.log(2.0)))
    npix = hp.nside2npix(nside)
    maps = np.zeros((3, npix), np.float64)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    # semi-axes (sigma q, sigma): the unit-integral prefactor is 1 / (2 pi sigma^2 q)
    profile = (
        1.0
        / (2.0 * np.pi * sigma**2 * q)
        * np.exp(
            -(np.cos(phi) ** 2 + q**2 * np.sin(phi) ** 2)
            * (theta**2 / (2 * sigma**2 * q**2))
        )
    )
    maps[0] = profile
    maps[1] = profile * np.cos(2.0 * phi)
    maps[2] = -profile * np.sin(2.0 * phi)
    return maps
