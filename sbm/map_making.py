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

import contextlib
import warnings
from dataclasses import dataclass

import numpy as np

from .basis import Basis
from .hmaps import HMaps
from .signal_model import SignalModel

#: Pixels whose system has a reciprocal condition number below this value are
#: not solved by default.
RCOND_MIN = 1e-12

#: Number of pixels whose condition number and inverse are computed at once.
CHUNK = 262144


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
    # model.project for every couple of the basis, with each component and its
    # conjugate computed once
    bt = np.zeros((len(basis.full), hmaps.npix), dtype=complex)
    for comp in model.components():
        for i, (n, m) in enumerate(basis.full):
            bt[i] += hmaps.h(n - comp.n, m - comp.m) * comp.map
    return np.ascontiguousarray((basis.weights[:, None] * bt).T)


@dataclass(frozen=True, eq=False)
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
            with contextlib.suppress(np.linalg.LinAlgError):
                x[i] = np.linalg.solve(a[i], b[i])
    return x


def solvable(
    a: np.ndarray,
    hits: np.ndarray,
    rcond_min: float = RCOND_MIN,
    among=None,
    chunk: int = CHUNK,
) -> np.ndarray:
    """Observed pixels whose system has a reciprocal condition number above
    ``rcond_min`` (among the pixels ``among``, all by default), computed for
    ``chunk`` pixels at a time."""
    ok = hits > 0
    if among is not None:
        ok &= np.asarray(among, dtype=bool)
    idx = np.flatnonzero(ok)
    for start in range(0, idx.size, chunk):
        sel = idx[start : start + chunk]
        with np.errstate(divide="ignore", invalid="ignore"):
            rcond = 1.0 / np.linalg.cond(a[sel])
        ok[sel] = np.isfinite(rcond) & (rcond > rcond_min)
    return ok


def _selection(a: np.ndarray, hits: np.ndarray, mask, chunk: int = CHUNK) -> np.ndarray:
    """Pixels to solve: ``mask`` (by default all) restricted to :func:`solvable`."""
    if mask is None:
        return solvable(a, hits, chunk=chunk)
    mask = np.asarray(mask, dtype=bool) & (hits > 0)
    ok = solvable(a, hits, among=mask, chunk=chunk)
    dropped = int((mask & ~ok).sum())
    if dropped:
        warnings.warn(
            f"{dropped} pixels of the mask have an ill-conditioned system "
            "and are not solved"
        )
    return mask & ok


def map_make(data, basis: Basis, mask=None) -> MapMakingResult:
    """Solve the map-making system.

    Args:
        data: ``(HMaps, SignalModel)``, or a list of them, one per detector.
        basis: the components to solve for.
        mask: pixels to solve, among the observed pixels whose system is well
            conditioned (:func:`solvable`); all of these by default. To reproduce
            the selection of SBM 0.4 without HWP pass
            ``hmaps.crosslink_mask(threshold)``.

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
    mask = _selection(a, hits, mask)
    x = _solve(a, b, mask)
    unsolved = mask & np.any(np.isnan(x), axis=1)
    if np.any(unsolved):
        warnings.warn(
            f"{int(unsolved.sum())} pixels of the mask have a singular system"
        )
        mask = mask & ~unsolved
    return MapMakingResult(basis=basis, solution=x.T.copy(), mask=mask)


class MapMaker:
    """Map-maker for fixed h-maps and basis, reused for many signal models.

    The system matrix depends only on the scan and the basis: it is inverted
    once in the solved pixels, and each model costs one product with its
    right-hand side. The result is that of :func:`map_make` for the same
    arguments, up to rounding errors.

    Args:
        hmaps: the h-maps of the detector.
        basis: the components to solve for.
        mask: pixels to solve, as in :func:`map_make`.
        chunk: number of pixels whose condition number and inverse are computed
            at once; this bounds the working memory of these steps, while the
            matrices of every pixel, ``(npix, k, k)``, and the inverses of the
            solved pixels are held in memory.
    """

    def __init__(self, hmaps: HMaps, basis: Basis, mask=None, chunk: int = CHUNK):
        self.hmaps = hmaps
        self.basis = basis
        a = orientation_matrix(hmaps, basis)
        mask = _selection(a, hmaps.hits, mask, chunk)
        idx = np.flatnonzero(mask)
        inverse = np.empty((idx.size, basis.dimension, basis.dimension), dtype=complex)
        ok = np.ones(idx.size, dtype=bool)
        for start in range(0, idx.size, chunk):
            block = a[idx[start : start + chunk]]
            try:
                inverse[start : start + chunk] = np.linalg.inv(block)
            except np.linalg.LinAlgError:
                for j in range(block.shape[0]):
                    try:
                        inverse[start + j] = np.linalg.inv(block[j])
                    except np.linalg.LinAlgError:
                        ok[start + j] = False
        if not ok.all():
            warnings.warn(
                f"{int((~ok).sum())} pixels of the mask have a singular system"
            )
            mask = mask.copy()
            mask[idx[~ok]] = False
        self.mask = mask
        self._idx = idx[ok]
        self._inverse = inverse[ok]

    def __call__(self, model: SignalModel) -> MapMakingResult:
        b = _rhs(self.hmaps, model, self.basis)[self._idx]
        x = np.full(
            (self.basis.dimension, self.hmaps.npix),
            complex(np.nan, np.nan),
            dtype=complex,
        )
        x[:, self._idx] = np.einsum("pij,pj->ip", self._inverse, b)
        return MapMakingResult(basis=self.basis, solution=x, mask=self.mask)


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
    ``NaN`` in the pixels that are not solved (as in :func:`map_make`).
    """
    a = orientation_matrix(hmaps, basis)
    mask = _selection(a, hmaps.hits, mask)
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


# ---------------------------------------------------------------------- #
# Pointing jitter


def _unit_offsets(offset_model) -> list[SignalModel]:
    """The models of the unit offsets along ``chi = 0`` and ``chi = pi/2`` of
    each independent offset model, checked to be linear in the offset."""
    models = [offset_model] if callable(offset_model) else list(offset_model)
    if not models:
        raise ValueError("the pointing jitter needs at least one offset model")
    units = []
    for model in models:
        along_x, along_y = model(1.0, 0.0), model(1.0, np.pi / 2)
        excess = model(2.0, 0.0) - 2.0 * along_x
        scale = max(
            (float(np.max(np.abs(c.map))) for c in along_x.components()), default=0.0
        )
        if any(
            float(np.max(np.abs(c.map))) > 1e-10 * scale for c in excess.components()
        ):
            raise ValueError(
                "the offset model must be linear in the offset (first order, "
                "without second_order)"
            )
        units += [along_x, along_y]
    return units


def pointing_jitter_covariance(
    hmaps: HMaps,
    basis: Basis,
    offset_model,
    sigma_rad: float,
    mask=None,
) -> np.ndarray:
    """Covariance of the solved amplitudes for a random pointing jitter, shape
    ``(npix, k, k)``.

    The pointing offset of every sample is drawn independently of the other
    samples: its two components, along the directions ``chi = 0`` and
    ``chi = pi/2`` of ``offset_model``, are independent Gaussian variables of
    standard deviation ``sigma_rad``. Since this jitter is isotropic, its
    statistics do not depend on the frame in which ``chi`` is measured.

    ``offset_model(rho_rad, chi_rad)`` is the :class:`.SignalModel` of a fixed
    offset at first order, for instance
    ``functools.partial(sbm.systematics.abs_pointing, sky)`` or, for a pair whose
    two detectors share the offset,
    ``lambda rho, chi: sbm.systematics.diff_pointing(sky, rho, chi, rho, chi)``.
    A sequence of models gives jitters that are independent of each other, such
    as those of the two detectors of a pair. A co-pointed pair behind a HWP has
    the data of a single detector that sees the polarization only: give
    ``abs_pointing`` a :class:`.Sky` without intensity.

    At first order the jitter adds ``x g + y g'`` to every sample, with ``g`` and
    ``g'`` the data of the unit offsets along the two directions: white noise of
    variance ``sigma^2 (g^2 + g'^2)``. The right-hand side of the map-maker then
    has the covariance ``sigma^2 / N_hits A_g``, where ``A_g`` is the matrix of
    :func:`orientation_matrix` with the samples weighted by ``g^2 + g'^2``, and
    the amplitudes have ``A^{-1} (sigma^2 / N_hits A_g) A^{-1}``. The pixels are
    independent. The covariance is ``NaN`` in the pixels that are not solved (as
    in :func:`map_make`).

    The h-maps must hold the couples between the differences of the basis couples
    and the components of ``g^2``, whose couples are the sums of two couples of
    the offset model; ``KeyError`` lists the missing ones. The mean effect of the
    jitter, a smoothing of the sky at second order in ``sigma_rad``, is not
    included.

    Raises:
        ValueError: ``sigma_rad`` negative or not finite, or an offset model that
            is not linear in the offset (``second_order=True``).
    """
    if not (np.isfinite(sigma_rad) and sigma_rad >= 0):
        raise ValueError(f"sigma_rad must be finite and non-negative, got {sigma_rad}")
    squared = sum(unit * unit for unit in _unit_offsets(offset_model))
    full, w = basis.full, basis.weights
    differences = {(ni - nj, mi - mj) for ni, mi in full for nj, mj in full}
    missing = sorted(
        c for c in squared.required_couples(differences) if not hmaps.has(*c)
    )
    if missing:
        raise KeyError(
            f"the h-maps lack the couples {missing} needed by the pointing jitter "
            f"on {basis}"
        )
    a = orientation_matrix(hmaps, basis)
    mask = _selection(a, hmaps.hits, mask)
    idx = np.flatnonzero(mask)
    a_g = np.empty((idx.size, len(full), len(full)), dtype=complex)
    for i, (ni, mi) in enumerate(full):
        for j in range(i, len(full)):
            nj, mj = full[j]
            a_g[:, i, j] = w[i] * w[j] * squared.project(hmaps, ni - nj, mi - mj)[idx]
            a_g[:, j, i] = np.conj(a_g[:, i, j])
    a_inv = np.linalg.inv(a[idx])
    cov = np.full(a.shape, complex(np.nan, np.nan), dtype=complex)
    scale = sigma_rad**2 / hmaps.hits[idx]
    cov[idx] = a_inv @ a_g @ a_inv * scale[:, None, None]
    return cov


def pointing_jitter_maps(
    hmaps: HMaps,
    basis: Basis,
    offset_model,
    sigma_rad: float,
    rng: np.random.Generator,
    mask=None,
) -> MapMakingResult:
    """A realization of the random pointing jitter in the solved amplitudes,
    with the covariance of :func:`pointing_jitter_covariance` (same arguments).

    The result is the change of the map-making solution, to be added to that of
    the signal without jitter, e.g.
    ``map_make((hmaps, sky_signal(sky, hwp=True)), basis).iqu + jitter.iqu``.
    """
    cov = pointing_jitter_covariance(hmaps, basis, offset_model, sigma_rad, mask)
    mask = ~np.isnan(cov[:, 0, 0])
    t = _real_transform(basis)
    t_inv = np.linalg.inv(t)
    idx = np.flatnonzero(mask)
    cov_r = np.einsum("ij,pjk,lk->pil", t_inv, cov[idx], t_inv.conj()).real
    cov_r = 0.5 * (cov_r + np.swapaxes(cov_r, 1, 2))
    # a square root that also holds for a singular covariance (sigma = 0, or a
    # component that the jitter does not reach)
    values, vectors = np.linalg.eigh(cov_r)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))[:, None, :]
    z = rng.standard_normal((idx.size, basis.dimension))
    r = np.einsum("pij,pj->pi", root, z)
    y = np.full((hmaps.npix, basis.dimension), complex(np.nan, np.nan), dtype=complex)
    y[idx] = r @ t.T
    return MapMakingResult(basis=basis, solution=y.T.copy(), mask=mask)
