"""Spherical-harmonic coefficients of a sky or a beam.

:class:`SphericalHarmonics` keeps the coefficients together with ``lmax`` and
``mmax``, with the attribute names of litebird_sim's class of the same name, so
that an object of litebird_sim can be passed wherever SBM takes one (SBM does
not depend on litebird_sim). The coefficients follow healpy: the components are
T, E, B (and V for a beam), the layout is m-major, and Q, U are synthesized
with the polarization convention of HEALPix.
"""

from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np


@dataclass(frozen=True, eq=False)
class SphericalHarmonics:
    """Harmonic coefficients ``values`` of shape ``(nstokes, ncoeff)``.

    Args:
        values: T (``nstokes = 1``), T, E, B (3) or T, E, B, V (4) coefficients in
            the m-major layout of healpy.
        lmax: maximum degree.
        mmax: maximum order (``lmax`` by default).
    """

    values: np.ndarray
    lmax: int
    mmax: int | None = None

    def __post_init__(self):
        values = np.atleast_2d(np.asarray(self.values, dtype=complex))
        mmax = self.lmax if self.mmax is None else int(self.mmax)
        if self.lmax < 0 or not 0 <= mmax <= self.lmax:
            raise ValueError(
                f"need 0 <= mmax <= lmax, got lmax={self.lmax}, mmax={mmax}"
            )
        if values.ndim != 2 or values.shape[0] not in (1, 3, 4):
            raise ValueError("values must have 1 (T), 3 (TEB) or 4 (TEBV) components")
        expected = hp.Alm.getsize(int(self.lmax), mmax)
        if values.shape[1] != expected:
            raise ValueError(
                f"{values.shape[1]} coefficients per component, expected {expected} "
                f"for lmax={self.lmax}, mmax={mmax}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "lmax", int(self.lmax))
        object.__setattr__(self, "mmax", mmax)

    @property
    def nstokes(self) -> int:
        return self.values.shape[0]

    @classmethod
    def of(cls, alm, mmax: int | None = None) -> "SphericalHarmonics":
        """``alm`` as :class:`SphericalHarmonics`: itself, an object with the
        attributes ``values``, ``lmax`` and ``mmax`` (such as litebird_sim's), or an
        array of coefficients with ``mmax = lmax`` unless ``mmax`` is given.

        An array does not tell its ``mmax``: one with ``mmax < lmax`` can have as
        many coefficients as a full layout of a smaller ``lmax`` and would be
        misread, so pass such coefficients as :class:`SphericalHarmonics`."""
        if isinstance(alm, cls):
            return alm
        if hasattr(alm, "values") and hasattr(alm, "lmax"):
            return cls(np.asarray(alm.values), alm.lmax, getattr(alm, "mmax", None))
        values = np.atleast_2d(np.asarray(alm))
        lmax = hp.Alm.getlmax(values.shape[1], mmax)
        if lmax < 0:
            raise ValueError("the number of coefficients matches no lmax")
        return cls(values, lmax, mmax)

    @classmethod
    def from_maps(
        cls, maps, lmax: int | None = None, iter: int = 3
    ) -> "SphericalHarmonics":
        """Coefficients of a T map or of I, Q, U maps (``healpy.map2alm``)."""
        maps = np.atleast_2d(np.asarray(maps, dtype=float))
        if maps.shape[0] not in (1, 3):
            raise ValueError("give a T map or I, Q, U maps")
        alm = hp.map2alm(maps if maps.shape[0] == 3 else maps[0], lmax=lmax, iter=iter)
        alm = np.atleast_2d(alm)
        return cls(alm, hp.Alm.getlmax(alm.shape[1]))

    def to_maps(self, nside: int) -> np.ndarray:
        """T map or I, Q, U maps (V is not synthesized)."""
        values = self.values[:3] if self.nstokes >= 3 else self.values[0]
        return np.atleast_2d(hp.alm2map(values, nside, lmax=self.lmax, mmax=self.mmax))

    def full(self) -> "SphericalHarmonics":
        """The same coefficients with ``mmax = lmax`` (zero above the old ``mmax``)."""
        if self.mmax == self.lmax:
            return self
        out = np.zeros((self.nstokes, hp.Alm.getsize(self.lmax)), dtype=complex)
        ell, m = _lm(self.lmax, self.mmax)
        out[:, hp.Alm.getidx(self.lmax, ell, m)] = self.values
        return SphericalHarmonics(out, self.lmax)


def _lm(lmax, mmax):
    """Degrees and orders of the coefficients of the layout (lmax, mmax)."""
    ell, m = [], []
    for mm in range(mmax + 1):
        ell.extend(range(mm, lmax + 1))
        m.extend([mm] * (lmax + 1 - mm))
    return np.array(ell), np.array(m)
