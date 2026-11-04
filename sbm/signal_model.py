"""Signal model of a detector as spin components.

The time-ordered data of a detector are written as

    d = sum_{n,m} S_{n,m} exp(i (n psi + m phi)) ,

with the crossing angle ``psi`` and the HWP angle ``phi`` of :mod:`sbm.hmaps`.
The data are real, so ``S_{-n,-m}`` is the complex conjugate of ``S_{n,m}``:
a :class:`SignalModel` stores one component per conjugate pair, and
``S_{0,0}`` is real. For example the sky seen by an ideal detector without HWP
is ``S_{0,0} = I`` and ``S_{2,0} = P / 2`` with ``P = Q + iU``.

Models are added like the signals they describe: the model of a detector with
several systematics is the sum of the models of each.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .hmaps import Couple, HMaps, canonical


@dataclass(frozen=True)
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
    """

    def __init__(self, components: Mapping[Couple, np.ndarray]):
        self._s: dict[Couple, np.ndarray] = {}
        npix = None
        for (n, m), value in components.items():
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
        needed = set()
        for n, m in basis_couples:
            for comp in self.components():
                key, _ = canonical(n - comp.n, m - comp.m)
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
        if other == 0:  # sum([...])
            return self
        return NotImplemented

    def __mul__(self, factor: float) -> "SignalModel":
        factor = float(factor)
        return SignalModel({key: factor * value for key, value in self._s.items()})

    __rmul__ = __mul__

    def __neg__(self) -> "SignalModel":
        return -1.0 * self

    def __sub__(self, other: "SignalModel") -> "SignalModel":
        return self + (-other)

    def __repr__(self) -> str:
        return f"SignalModel(couples={self.couples}, npix={self.npix})"
