"""Sky maps and their spin derivatives.

The derivatives follow the convention of SBM: on the local tangent plane with
``x`` along theta and ``y`` along phi, ``eth = d_y - i d_x`` and
``eth_bar = d_y + i d_x``, applied to I and to each of Q and U. With the
derivatives of healpy (``d/dtheta`` and ``d/dphi / sin(theta)``),

    eth I     = I_phi - i I_theta
    eth P     = (Q_phi + U_theta) - i (Q_theta - U_phi)
    eth_bar P = (Q_phi - U_theta) + i (Q_theta + U_phi)

with ``P = Q + iU``.

The second-order maps used by the pointing systematics (``eth^2 I``, ``lap I``,
``eth^2 P``, ``eth_bar^2 P``, ``lap P``) are computed from exact geodesic second
differences of the band-limited sky, evaluated at arbitrary points with ducc0: a
pointing offset of size ``rho`` in direction ``alpha`` samples the sky at the
point displaced along ``-(sin(alpha) e_theta + cos(alpha) e_phi)``, and

    S(p') = S - (rho/2) (e^{i alpha} eth + e^{-i alpha} eth_bar) S
            + (rho^2/8) (e^{2i alpha} eth^2 + e^{-2i alpha} eth_bar^2) S
            + (rho^2/4) lap S + O(rho^3) .
"""

from __future__ import annotations

from functools import cached_property

import healpy as hp
import numpy as np

from .harmonics import SphericalHarmonics


SECOND_ORDER_KEYS = ("eth2_i", "lap_i", "eth2_p", "eth_bar2_p", "lap_p")


class Sky:
    """Temperature and polarization of the sky, with derivatives on demand.

    Args:
        i: temperature map.
        p: polarization map ``Q + iU``.
        eth_i, eth_p, eth_bar_p: derivatives, if already known; otherwise they
            are computed from the maps the first time they are used.
        alm: T, E, B coefficients of the maps (:class:`.SphericalHarmonics` or an
            array), if known; the
            second-order maps are computed from them (otherwise from
            ``healpy.map2alm`` of the maps).
        second_order: the second-order maps, if already known, as a dictionary
            with the keys ``eth2_i``, ``lap_i``, ``eth2_p``, ``eth_bar2_p`` and
            ``lap_p``.
        lmax, iter: passed to ``healpy.map2alm`` for the derivatives.
    """

    def __init__(
        self,
        i,
        p,
        *,
        eth_i=None,
        eth_p=None,
        eth_bar_p=None,
        alm=None,
        second_order: dict | None = None,
        lmax: int | None = None,
        iter: int = 3,
    ):
        self.i = np.asarray(i, dtype=float)
        self.p = np.asarray(p, dtype=complex)
        if self.i.shape != self.p.shape:
            raise ValueError("i and p must have the same shape")
        self.nside = hp.npix2nside(self.i.size)
        self._lmax = lmax
        self._iter = iter
        if eth_i is not None:
            self.__dict__["eth_i"] = np.asarray(eth_i, dtype=complex)
        if eth_p is not None and eth_bar_p is not None:
            self.__dict__["_eth_pol"] = (
                np.asarray(eth_p, dtype=complex),
                np.asarray(eth_bar_p, dtype=complex),
            )
        elif (eth_p is None) != (eth_bar_p is None):
            raise ValueError("give both eth_p and eth_bar_p, or neither")
        self.alm = None if alm is None else SphericalHarmonics.of(alm)
        if second_order is not None:
            missing = set(SECOND_ORDER_KEYS) - set(second_order)
            if missing:
                raise ValueError(f"second_order lacks {sorted(missing)}")
            self.__dict__["second_order"] = {
                key: np.asarray(second_order[key]) for key in SECOND_ORDER_KEYS
            }

    @classmethod
    def from_maps(cls, i, q, u, **kwargs) -> "Sky":
        """Sky from I, Q and U maps."""
        return cls(i, np.asarray(q) + 1j * np.asarray(u), **kwargs)

    @classmethod
    def from_alm(cls, alm, nside: int, **kwargs) -> "Sky":
        """Sky from T, E, B coefficients (:class:`.SphericalHarmonics`, litebird_sim's,
        or an array in the healpy ordering)."""
        alm = SphericalHarmonics.of(alm)
        if alm.nstokes < 3:
            raise ValueError("the sky needs T, E and B coefficients")
        i, q, u = alm.to_maps(nside)
        return cls.from_maps(i, q, u, alm=alm, **kwargs)

    @property
    def npix(self) -> int:
        return self.i.size

    @property
    def q(self) -> np.ndarray:
        return self.p.real

    @property
    def u(self) -> np.ndarray:
        return self.p.imag

    def _der1(self, x: np.ndarray) -> np.ndarray:
        alm = hp.map2alm(x, lmax=self._lmax, iter=self._iter)
        return hp.alm2map_der1(alm, nside=self.nside, lmax=self._lmax)

    @cached_property
    def eth_i(self) -> np.ndarray:
        d = self._der1(self.i)
        return d[2] - 1j * d[1]

    @property
    def eth_bar_i(self) -> np.ndarray:
        return np.conj(self.eth_i)

    @cached_property
    def _eth_pol(self) -> tuple[np.ndarray, np.ndarray]:
        dq = self._der1(self.q)
        du = self._der1(self.u)
        eth_p = dq[2] + du[1] - 1j * (dq[1] - du[2])
        eth_bar_p = dq[2] - du[1] + 1j * (dq[1] + du[2])
        return eth_p, eth_bar_p

    @property
    def eth_p(self) -> np.ndarray:
        return self._eth_pol[0]

    @property
    def eth_bar_p(self) -> np.ndarray:
        return self._eth_pol[1]

    @cached_property
    def second_order(self) -> dict:
        """Second-order maps (see the module documentation), computed with ducc0
        from the harmonic coefficients of the sky with a step of one arcminute."""
        if self.alm is not None:
            alm = self.alm
        else:
            alm = np.array(
                hp.map2alm([self.i, self.q, self.u], lmax=self._lmax, iter=self._iter)
            )
        return second_order_maps(alm, self.nside)


def _displaced(theta, phi, alpha, step):
    """Points displaced by ``step`` along the geodesic in direction ``alpha``,
    along ``-(sin(alpha) e_theta + cos(alpha) e_phi)``."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    n = np.stack([st * cp, st * sp, ct])
    e_t = np.stack([ct * cp, ct * sp, -st])
    e_p = np.stack([-sp, cp, np.zeros_like(sp)])
    t = -(np.sin(alpha) * e_t + np.cos(alpha) * e_p)
    v = np.cos(step) * n + np.sin(step) * t
    return np.arccos(np.clip(v[2], -1.0, 1.0)), np.mod(
        np.arctan2(v[1], v[0]), 2 * np.pi
    )


def _synthesize(alm_t, alm_eb, lmax, theta, phi, nthreads, epsilon):
    from ducc0.sht.experimental import synthesis_general

    loc = np.ascontiguousarray(np.stack([theta, phi], axis=1))
    options = dict(lmax=lmax, loc=loc, epsilon=epsilon, nthreads=nthreads)
    t = synthesis_general(alm=alm_t, spin=0, **options)[0]
    qu = synthesis_general(alm=alm_eb, spin=2, **options)
    return t, qu[0], qu[1]


def second_order_maps(
    alm,
    nside: int,
    step_rad: float | None = None,
    nthreads: int = 4,
    epsilon: float = 1e-12,
) -> dict:
    """Second-order maps ``eth2_i``, ``lap_i``, ``eth2_p``, ``eth_bar2_p``, ``lap_p``.

    Args:
        alm: band-limited T, E, B coefficients (:class:`.SphericalHarmonics` or an
            array in the healpy ordering).
        nside: resolution of the maps, evaluated at the pixel centres.
        step_rad: finite-difference step (default: one arcminute).
        nthreads, epsilon: passed to ``ducc0.sht.experimental.synthesis_general``.
    """
    sh = SphericalHarmonics.of(alm).full()
    alm, lmax = sh.values, sh.lmax
    h = np.deg2rad(1.0 / 60.0) if step_rad is None else float(step_rad)
    alm_t = np.ascontiguousarray(alm[:1])
    alm_eb = np.ascontiguousarray(alm[1:3])
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    s0 = np.array(_synthesize(alm_t, alm_eb, lmax, theta, phi, nthreads, epsilon))

    def second_difference(alpha):
        sp = np.array(
            _synthesize(
                alm_t,
                alm_eb,
                lmax,
                *_displaced(theta, phi, alpha, h),
                nthreads,
                epsilon,
            )
        )
        sm = np.array(
            _synthesize(
                alm_t,
                alm_eb,
                lmax,
                *_displaced(theta, phi, alpha, -h),
                nthreads,
                epsilon,
            )
        )
        return 0.5 * (sp + sm - 2.0 * s0)

    # D(alpha) = (h^2/4) Re(e^{2i alpha} A) + (h^2/4) L for each of I, Q and U
    d0 = second_difference(0.0)
    d45 = second_difference(np.pi / 4)
    d90 = second_difference(np.pi / 2)
    lap = 2.0 * (d0 + d90) / h**2
    re_a = 2.0 * (d0 - d90) / h**2
    im_a = lap - 4.0 * d45 / h**2
    a = re_a + 1j * im_a  # eth^2 of each component
    return {
        "eth2_i": a[0],
        "lap_i": lap[0],
        "eth2_p": a[1] + 1j * a[2],
        "eth_bar2_p": np.conj(a[1]) + 1j * np.conj(a[2]),
        "lap_p": lap[1] + 1j * lap[2],
    }
