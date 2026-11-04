"""Sky maps and their spin derivatives.

The derivatives follow the convention of SBM: on the local tangent plane with
``x`` along theta and ``y`` along phi, ``eth = d_y - i d_x`` and
``eth_bar = d_y + i d_x``, applied to I and to each of Q and U. With the
derivatives of healpy (``d/dtheta`` and ``d/dphi / sin(theta)``),

    eth I     = I_phi - i I_theta
    eth P     = (Q_phi + U_theta) - i (Q_theta - U_phi)
    eth_bar P = (Q_phi - U_theta) + i (Q_theta + U_phi)

with ``P = Q + iU``.
"""

from __future__ import annotations

from functools import cached_property

import healpy as hp
import numpy as np


class Sky:
    """Temperature and polarization of the sky, with derivatives on demand.

    Args:
        i: temperature map.
        p: polarization map ``Q + iU``.
        eth_i, eth_p, eth_bar_p: derivatives, if already known; otherwise they
            are computed from the maps the first time they are used.
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

    @classmethod
    def from_maps(cls, i, q, u, **kwargs) -> "Sky":
        """Sky from I, Q and U maps."""
        return cls(i, np.asarray(q) + 1j * np.asarray(u), **kwargs)

    @classmethod
    def from_alm(cls, alm, nside: int, **kwargs) -> "Sky":
        """Sky from T, E, B harmonic coefficients (healpy ordering)."""
        i, q, u = hp.alm2map(alm, nside)
        return cls.from_maps(i, q, u, **kwargs)

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
