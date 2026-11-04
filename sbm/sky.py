"""Sky maps and their spin derivatives.

The derivatives follow the convention of SBM: on the local tangent plane with
``x`` along theta and ``y`` along phi, ``eth = d_y - i d_x`` and
``eth_bar = d_y + i d_x``. For the temperature, with the derivatives of healpy
(``d/dtheta`` and ``d/dphi / sin(theta)``), ``eth I = I_phi - i I_theta``.

The polarization ``P = Q + iU`` is a spin-2 field: its components refer to the
local basis ``(e_theta, e_phi)``, which turns from one point to the next. A
pointing offset displaces the beam rigidly, so the polarization direction of the
detector is carried along the displacement (parallel transport), and the
derivatives of ``P`` are the covariant (spin-weighted) ones: they are computed
from the E and B coefficients with spin transforms,

    eth P     =  i  sum (E + iB)_lm sqrt((l - 2)(l + 3)) 3Y_lm
    eth_bar P = -i  sum (E + iB)_lm sqrt((l + 2)(l - 1)) 1Y_lm

(up to the sign convention of healpy's spin maps), and equal
``eth Q + i eth U`` of the components in a frame transported from the point.
The derivatives of Q and U taken as scalar maps differ from them by the
rotation of the basis, a term of order ``cot(theta) P`` that is largest near
the poles. On scales much smaller than the distance to the poles both agree.

The second-order maps used by the pointing systematics (``eth^2 I``, ``lap I``,
``eth^2 P``, ``eth_bar^2 P``, ``lap P``) are computed from exact geodesic second
differences of the band-limited sky, evaluated at arbitrary points with ducc0,
with P transported back to the frame of the point: a pointing offset of size
``rho`` in direction ``alpha`` samples the sky at the point displaced along
``-(sin(alpha) e_theta + cos(alpha) e_phi)``, and

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
        or an array in the healpy ordering with ``mmax = lmax``)."""
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

    @cached_property
    def _harmonics(self) -> SphericalHarmonics:
        """T, E, B coefficients: the given ones, or those of the maps."""
        if self.alm is not None:
            return self.alm.full()
        alm = hp.map2alm([self.i, self.q, self.u], lmax=self._lmax, iter=self._iter)
        return SphericalHarmonics(np.array(alm), hp.Alm.getlmax(len(alm[0])))

    @cached_property
    def _derivatives(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return spin_derivatives(self._harmonics, self.nside)

    @cached_property
    def eth_i(self) -> np.ndarray:
        return self._derivatives[0]

    @property
    def eth_bar_i(self) -> np.ndarray:
        return np.conj(self.eth_i)

    @cached_property
    def _eth_pol(self) -> tuple[np.ndarray, np.ndarray]:
        return self._derivatives[1], self._derivatives[2]

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
        return second_order_maps(self._harmonics, self.nside)


def spin_derivatives(alm, nside: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``eth I``, ``eth P`` and ``eth_bar P`` (covariant) from T, E, B coefficients
    (:class:`.SphericalHarmonics` or an array), with spin transforms of healpy."""
    sh = SphericalHarmonics.of(alm).full()
    if sh.nstokes < 3:
        raise ValueError("the derivatives need T, E and B coefficients")
    t, e, b = sh.values[:3]
    lmax = sh.lmax
    ell = np.arange(lmax + 1.0)

    def spin_map(g, c, spin):  # -sum (g + i c)_lm sY_lm, as healpy
        q, u = hp.alm2map_spin([g, c], nside, spin, lmax, mmax=lmax)
        return q + 1j * u

    r1 = np.sqrt(ell * (ell + 1.0))
    r3 = np.sqrt(np.clip((ell - 2.0) * (ell + 3.0), 0.0, None))
    r1p = np.sqrt(np.clip((ell + 2.0) * (ell - 1.0), 0.0, None))
    zero = np.zeros_like(t)
    eth_i = -1j * spin_map(hp.almxfl(t, r1), zero, 1)
    eth_p = 1j * spin_map(hp.almxfl(e, r3), hp.almxfl(b, r3), 3)
    eth_bar_p = 1j * spin_map(hp.almxfl(e, r1p), hp.almxfl(b, r1p), 1)
    return eth_i, eth_p, eth_bar_p


def _frame(theta, phi):
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    n = np.stack([st * cp, st * sp, ct])
    e_t = np.stack([ct * cp, ct * sp, -st])
    e_p = np.stack([-sp, cp, np.zeros_like(sp)])
    return n, e_t, e_p


def _displaced(theta, phi, alpha, step):
    """Points displaced by ``step`` along the geodesic in direction ``alpha``,
    along ``-(sin(alpha) e_theta + cos(alpha) e_phi)``, and the angle ``delta``
    of the transported ``e_theta`` in the local basis of the new point: there
    the polarization in the transported frame is ``P exp(-2i delta)``."""
    n, e_t, e_p = _frame(theta, phi)
    t = -(np.sin(alpha) * e_t + np.cos(alpha) * e_p)
    v = np.cos(step) * n + np.sin(step) * t
    t_new = -np.sin(step) * n + np.cos(step) * t
    # parallel transport: the component along the geodesic turns with it
    w = e_t + np.sum(e_t * t, axis=0) * (t_new - t)
    theta_new = np.arccos(np.clip(v[2], -1.0, 1.0))
    phi_new = np.mod(np.arctan2(v[1], v[0]), 2 * np.pi)
    _, e_t_new, e_p_new = _frame(theta_new, phi_new)
    delta = np.arctan2(np.sum(w * e_p_new, axis=0), np.sum(w * e_t_new, axis=0))
    return theta_new, phi_new, delta


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
    transport: bool = True,
) -> dict:
    """Second-order maps ``eth2_i``, ``lap_i``, ``eth2_p``, ``eth_bar2_p``, ``lap_p``.

    Args:
        alm: band-limited T, E, B coefficients (:class:`.SphericalHarmonics` or an
            array in the healpy ordering).
        nside: resolution of the maps, evaluated at the pixel centres.
        step_rad: finite-difference step (default: one arcminute).
        nthreads, epsilon: passed to ``ducc0.sht.experimental.synthesis_general``.
        transport: take Q and U in the frame transported from the point (the
            covariant maps, consistent with the first derivatives of
            :class:`Sky`); ``False`` takes them in the local basis of the
            displaced points, consistent with derivatives of Q and U as scalar
            maps (SBM 0.4).
    """
    sh = SphericalHarmonics.of(alm).full()
    alm, lmax = sh.values, sh.lmax
    h = np.deg2rad(1.0 / 60.0) if step_rad is None else float(step_rad)
    alm_t = np.ascontiguousarray(alm[:1])
    alm_eb = np.ascontiguousarray(alm[1:3])
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    s0 = np.array(_synthesize(alm_t, alm_eb, lmax, theta, phi, nthreads, epsilon))

    def transported(alpha, step):
        """I, Q, U at the displaced points, Q and U in the transported frame."""
        theta_d, phi_d, delta = _displaced(theta, phi, alpha, step)
        i, q, u = _synthesize(alm_t, alm_eb, lmax, theta_d, phi_d, nthreads, epsilon)
        p = q + 1j * u
        if transport:
            p = p * np.exp(-2j * delta)
        return np.array([i, p.real, p.imag])

    def second_difference(alpha):
        return 0.5 * (transported(alpha, h) + transported(alpha, -h) - 2.0 * s0)

    # D(alpha) = (h^2/4) Re(e^{2i alpha} A) + (h^2/4) L for each of I, Q and U
    # (Q and U in the frame transported from the point)
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
