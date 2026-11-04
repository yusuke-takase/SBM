"""Map-making in spin space.

For a detector with orientation functions ``h̃`` (:class:`.HMaps`) and signal
model ``S`` (:class:`.SignalModel`), the binned map-maker that fits the
components of a :class:`.Basis` to the time-ordered data solves, in each pixel,

    A y = b ,   A_ij = w_i w_j h̃_{n_i - n_j, m_i - m_j} ,
                b_i = w_i sum_k h̃_{n_i - n_k, m_i - m_k} S_k ,

with ``w = 1`` for ``(0, 0)`` and ``1/2`` otherwise. ``y`` holds ``I`` and the
amplitudes ``Z_{n,m}`` (``P`` at the polarization couple). With several
detectors, their systems weighted by the hit counts are summed before solving.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .basis import Basis
from .hmaps import HMaps
from .signal_model import SignalModel

#: Pixels whose system has a reciprocal condition number below this value are
#: not solved by default.
RCOND_MIN = 1e-12


def orientation_matrix(hmaps: HMaps, basis: Basis) -> np.ndarray:
    """Matrix ``A`` of the map-making system in every pixel, shape ``(npix, k, k)``."""
    full = basis.full
    w = basis.weights
    missing = sorted(
        {
            (ni - nj, mi - mj)
            for ni, mi in full
            for nj, mj in full
            if not hmaps.has(ni - nj, mi - mj)
        }
    )
    if missing:
        raise KeyError(f"the h-maps lack the couples {missing} needed by {basis}")
    a = np.empty((hmaps.npix, len(full), len(full)), dtype=complex)
    for i, (ni, mi) in enumerate(full):
        for j, (nj, mj) in enumerate(full):
            a[:, i, j] = w[i] * w[j] * hmaps.h(ni - nj, mi - mj)
    return a


def _rhs(hmaps: HMaps, model: SignalModel, basis: Basis) -> np.ndarray:
    needed = model.required_couples(basis.full)
    missing = sorted(c for c in needed if not hmaps.has(*c))
    if missing:
        raise KeyError(
            f"the h-maps lack the couples {missing} needed to project {model} on {basis}"
        )
    w = basis.weights
    b = np.empty((hmaps.npix, len(w)), dtype=complex)
    for i, (n, m) in enumerate(basis.full):
        b[:, i] = w[i] * model.project(hmaps, n, m)
    return b


@dataclass(frozen=True)
class MapMakingResult:
    """Solution of the map-maker.

    Attributes:
        basis: the solved basis.
        solution: ``(k, npix)`` complex amplitudes, in the order of
            ``basis.full``; ``NaN`` in the pixels that were not solved.
        mask: pixels that were solved.
    """

    basis: Basis
    solution: np.ndarray
    mask: np.ndarray

    def component(self, n: int, m: int) -> np.ndarray:
        """Solved amplitude at ``(n, m)`` (``I`` at ``(0, 0)``)."""
        return self.solution[self.basis.full.index((int(n), int(m)))]

    @property
    def iqu(self) -> np.ndarray:
        """``(3, npix)`` maps of I, Q and U. I is zero if the basis has no ``(0, 0)``."""
        if self.basis.polarization is None:
            raise ValueError(f"{self.basis} has no polarization couple")
        p = self.component(*self.basis.polarization)
        if (0, 0) in self.basis.couples:
            i = self.component(0, 0).real
        else:
            i = np.where(self.mask, 0.0, np.nan)
        return np.array([i, p.real, p.imag])


def _as_pairs(data) -> list[tuple[HMaps, SignalModel]]:
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], HMaps):
        return [data]
    pairs = list(data)
    for pair in pairs:
        if not (
            isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], HMaps)
        ):
            raise TypeError("map_make takes (HMaps, SignalModel) or a list of them")
    if not pairs:
        raise ValueError("map_make needs at least one (HMaps, SignalModel)")
    return pairs


def _solve(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.full(b.shape, complex(np.nan, np.nan), dtype=complex)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return x
    try:
        x[idx] = np.linalg.solve(a[idx], b[idx, :, None])[..., 0]
    except np.linalg.LinAlgError:
        for i in idx:
            try:
                x[i] = np.linalg.solve(a[i], b[i])
            except np.linalg.LinAlgError:
                pass
    return x


def solvable(
    a: np.ndarray, hits: np.ndarray, rcond_min: float = RCOND_MIN
) -> np.ndarray:
    """Observed pixels whose system has a reciprocal condition number above ``rcond_min``."""
    ok = hits > 0
    idx = np.flatnonzero(ok)
    if idx.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            rcond = 1.0 / np.linalg.cond(a[idx])
        ok[idx] = np.isfinite(rcond) & (rcond > rcond_min)
    return ok


def map_make(data, basis: Basis, mask=None) -> MapMakingResult:
    """Solve the map-making system.

    Args:
        data: ``(HMaps, SignalModel)``, or a list of them, one per detector.
        basis: the components to solve for.
        mask: pixels to solve. By default the observed pixels whose system is
            well conditioned (:func:`solvable`). To reproduce the selection of
            SBM 0.4 without HWP pass ``hmaps.crosslink_mask(threshold)``.

    Returns:
        :class:`MapMakingResult`; the pixels that are not solved are ``NaN``.
    """
    pairs = _as_pairs(data)
    if len(pairs) == 1:
        hmaps, model = pairs[0]
        a = orientation_matrix(hmaps, basis)
        b = _rhs(hmaps, model, basis)
        hits = hmaps.hits
    else:
        npix = pairs[0][0].npix
        a = np.zeros((npix, basis.dimension, basis.dimension), dtype=complex)
        b = np.zeros((npix, basis.dimension), dtype=complex)
        hits = np.zeros(npix)
        for hmaps, model in pairs:
            weight = hmaps.hits
            a += weight[:, None, None] * orientation_matrix(hmaps, basis)
            b += weight[:, None] * _rhs(hmaps, model, basis)
            hits = hits + hmaps.hits
    if mask is None:
        mask = solvable(a, hits)
    else:
        mask = np.asarray(mask, dtype=bool) & (hits > 0)
    x = _solve(a, b, mask)
    unsolved = mask & np.any(np.isnan(x), axis=1)
    if np.any(unsolved):
        warnings.warn(
            f"{int(unsolved.sum())} pixels of the mask have a singular system"
        )
        mask = mask & ~unsolved
    return MapMakingResult(basis=basis, solution=x.T.copy(), mask=mask)


# ---------------------------------------------------------------------- #
# White noise


def noise_covariance(
    hmaps: HMaps,
    basis: Basis,
    net_ukrts: float,
    sampling_rate_hz: float,
    mask=None,
) -> np.ndarray:
    """Covariance of the solved amplitudes for white noise, shape ``(npix, k, k)``.

    With a noise variance ``sigma^2 = NET^2 f_s`` per sample and ``N`` hits,
    ``Cov(y) = sigma^2 / N A^{-1}``. ``net_ukrts`` is the NET of the data that
    enter the map-maker: for a pair difference ``(d_T - d_B) / 2`` of two
    detectors with the same NET it is ``NET / sqrt(2)``. The covariance is
    ``NaN`` in the pixels outside ``mask`` (by default :func:`solvable`).
    """
    a = orientation_matrix(hmaps, basis)
    mask = (
        solvable(a, hmaps.hits)
        if mask is None
        else np.asarray(mask, bool) & (hmaps.hits > 0)
    )
    cov = np.full(a.shape, complex(np.nan, np.nan), dtype=complex)
    idx = np.flatnonzero(mask)
    sigma2 = net_ukrts**2 * sampling_rate_hz
    cov[idx] = np.linalg.inv(a[idx]) * (sigma2 / hmaps.hits[idx])[:, None, None]
    return cov


def _real_transform(basis: Basis) -> np.ndarray:
    """Matrix T with ``y = T r`` from the real unknowns r = (I, Re Z, Im Z, ...)."""
    k = basis.dimension
    t = np.zeros((k, k), dtype=complex)
    i = 0
    for n, m in basis.couples:
        if (n, m) == (0, 0):
            t[i, i] = 1.0
            i += 1
        else:
            t[i, i], t[i, i + 1] = 1.0, 1j
            t[i + 1, i], t[i + 1, i + 1] = 1.0, -1j
            i += 2
    return t


def noise_maps(
    hmaps: HMaps,
    basis: Basis,
    net_ukrts: float,
    sampling_rate_hz: float,
    rng: np.random.Generator,
    mask=None,
) -> MapMakingResult:
    """A realization of white noise in the solved amplitudes, with the
    correlations of :func:`noise_covariance` (I-P, Q-U and templates)."""
    cov = noise_covariance(hmaps, basis, net_ukrts, sampling_rate_hz, mask)
    mask = ~np.isnan(cov[:, 0, 0])
    t = _real_transform(basis)
    t_inv = np.linalg.inv(t)
    idx = np.flatnonzero(mask)
    cov_r = np.einsum("ij,pjk,lk->pil", t_inv, cov[idx], t_inv.conj()).real
    cov_r = 0.5 * (cov_r + np.swapaxes(cov_r, 1, 2))
    chol = np.linalg.cholesky(cov_r)
    z = rng.standard_normal((idx.size, basis.dimension))
    r = np.einsum("pij,pj->pi", chol, z)
    y = np.full((hmaps.npix, basis.dimension), complex(np.nan, np.nan), dtype=complex)
    y[idx] = r @ t.T
    return MapMakingResult(basis=basis, solution=y.T.copy(), mask=mask)
