"""Signal model of a detector as spin components.

The time-ordered data of a detector are written as

    d = sum_{n,m} S_{n,m} exp(i (n psi + m phi)) ,

with the crossing angle ``psi`` and the HWP angle ``phi`` of :mod:`sbm.hmaps`.
The data are real, so ``S_{-n,-m}`` is the complex conjugate of ``S_{n,m}``:
a :class:`SignalModel` stores one component per conjugate pair, and
``S_{0,0}`` is real. For example the sky seen by an ideal detector without HWP
is ``S_{0,0} = I`` and ``S_{2,0} = P / 2`` with ``P = Q + iU``.

Models are added and multiplied like the signals they describe: the model of
a detector with several systematics is the sum of the models of each, and the
product of two models is the model of the product of their signals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .hmaps import Couple, HMaps, canonical
from .spin import SpinWeightError, spin_weight


@dataclass(frozen=True, eq=False)
class SpinComponent:
    """One spin component ``S_{n,m}`` of a signal model."""

    n: int
    m: int
    map: np.ndarray

    def conj(self) -> "SpinComponent":
        return SpinComponent(-self.n, -self.m, np.conj(self.map))


class SignalModel:
    """Spin components of the signal of a detector.

    Args:
        components: maps of ``S_{n,m}`` indexed by ``(n, m)``. A component given
            at ``(n, m)`` stands for itself and its conjugate at ``(-n, -m)``;
            components given at the same couple, or at opposite couples, are
            summed. ``S_{0,0}`` must be real.

    A component made of the maps of a :class:`~sbm.Sky` carries its spin weight
    (:mod:`sbm.spin`), which must equal its ``n``: otherwise
    :class:`~sbm.spin.SpinWeightError` is raised. Plain arrays are not checked.
    """

    # NumPy defers to the methods below: an array times a model is an error
    # (a factor per pixel) instead of an array of models
    __array_ufunc__ = None

    def __init__(self, components: Mapping[Couple, np.ndarray]):
        self._s: dict[Couple, np.ndarray] = {}
        npix = None
        for (n, m), value in components.items():
            spin = spin_weight(value) if np.ndim(value) else None
            if spin is not None and spin != n:
                raise SpinWeightError(
                    f"S_{{{n},{m}}} has spin weight {spin}, but a component at "
                    f"({n}, {m}) must have spin weight {n}: the spin weights of the "
                    "factors on its right-hand side (I 0, P 2, eth I 1, eth P 3, "
                    "eth_bar P 1, ...) must add up to n (see sbm.spin)"
                )
            value = np.asarray(value)
            if npix is None:
                npix = value.shape[-1]
            if value.shape != (npix,):
                raise ValueError(
                    f"S_{{{n},{m}}} has shape {value.shape}, expected ({npix},)"
                )
            key, conj = canonical(n, m)
            if key == (0, 0):
                if np.iscomplexobj(value) and np.any(value.imag != 0):
                    raise ValueError("S_{0,0} must be real")
                value = np.real(value).astype(complex)
            else:
                value = np.conj(value) if conj else value.astype(complex)
            if key in self._s:
                self._s[key] = self._s[key] + value
            else:
                self._s[key] = value.copy()
        if npix is None:
            raise ValueError("a signal model needs at least one component")
        self.npix = npix

    @property
    def couples(self) -> list[Couple]:
        """Canonical couples of the model."""
        return sorted(self._s)

    def component(self, n: int, m: int) -> np.ndarray:
        """Return ``S_{n,m}`` (zero if the model has no such component)."""
        key, conj = canonical(n, m)
        value = self._s.get(key)
        if value is None:
            return np.zeros(self.npix, dtype=complex)
        return np.conj(value) if conj else value

    def components(self) -> list[SpinComponent]:
        """All the components, each conjugate pair written out."""
        out = []
        for (n, m), value in sorted(self._s.items()):
            out.append(SpinComponent(n, m, value))
            if (n, m) != (0, 0):
                out.append(SpinComponent(-n, -m, np.conj(value)))
        return out

    def project(self, hmaps: HMaps, n: int, m: int) -> np.ndarray:
        """Binned data projected on the spin ``(n, m)``:
        ``< d exp(-i (n psi + m phi)) > = sum_k h̃_{n-n_k, m-m_k} S_k``.
        """
        if hmaps.npix != self.npix:
            raise ValueError(
                f"the model has {self.npix} pixels and the h-maps {hmaps.npix}"
            )
        total = np.zeros(self.npix, dtype=complex)
        for comp in self.components():
            total += hmaps.h(n - comp.n, m - comp.m) * comp.map
        return total

    def required_couples(self, basis_couples) -> set[Couple]:
        """Canonical h-map couples needed to project the model on ``basis_couples``."""
        couples = list(self._s) + [(-n, -m) for n, m in self._s if (n, m) != (0, 0)]
        needed = set()
        for n, m in basis_couples:
            for cn, cm in couples:
                key, _ = canonical(n - cn, m - cm)
                if key != (0, 0):
                    needed.add(key)
        return needed

    def __add__(self, other: "SignalModel") -> "SignalModel":
        if not isinstance(other, SignalModel):
            return NotImplemented
        merged = dict(self._s)
        for key, value in other._s.items():
            merged[key] = merged[key] + value if key in merged else value
        return SignalModel(merged)

    def __radd__(self, other):
        if np.ndim(other) == 0 and other == 0:  # sum([...])
            return self
        return NotImplemented

    def __mul__(self, other) -> "SignalModel":
        """A number times the model, or the product of two models.

        The product of the signals ``d_a d_b`` is the convolution of their spin
        components: ``S_{n,m} = sum S^a_{n_a,m_a} S^b_{n-n_a,m-m_a}`` over every
        component of each, conjugates included (for instance the non-linear
        response ``d + beta d**2`` of a detector).
        """
        if isinstance(other, SignalModel):
            if other.npix != self.npix:
                raise ValueError("the models have different numbers of pixels")
            out: dict[Couple, np.ndarray] = {}
            for a in self.components():
                for b in other.components():
                    key = (a.n + b.n, a.m + b.m)
                    if canonical(*key)[1]:
                        continue  # the conjugate couple carries the same information
                    value = a.map * b.map
                    out[key] = out[key] + value if key in out else value
            if (0, 0) in out:
                out[(0, 0)] = out[(0, 0)].real  # real up to rounding errors
            return SignalModel(out)
        if np.ndim(other) != 0:
            raise TypeError(
                "a signal model is multiplied by a real number or by another signal "
                "model; for a factor that depends on the pixel, multiply the maps "
                "of the components"
            )
        factor = float(other)
        return SignalModel({key: factor * value for key, value in self._s.items()})

    def __rmul__(self, factor) -> "SignalModel":
        return self * factor

    def __neg__(self) -> "SignalModel":
        return -1.0 * self

    def __sub__(self, other: "SignalModel") -> "SignalModel":
        return self + (-other)

    def __repr__(self) -> str:
        return f"SignalModel(couples={self.couples}, npix={self.npix})"
