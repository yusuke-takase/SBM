"""Spin components solved by the map-maker."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .hmaps import Couple, canonical

#: Spin of the polarization ``P = Q + iU`` without and with HWP.
POLARIZATION = (2, 0)
POLARIZATION_HWP = (2, -4)


class Basis:
    """Spin components that the map-maker solves for, in each pixel.

    Args:
        couples: canonical couples ``(n, m)``; each one but ``(0, 0)`` brings its
            conjugate ``(-n, -m)``, so the dimension of the system is
            ``2 len(couples) - 1`` with ``(0, 0)``, ``2 len(couples)`` without.
        polarization: couple of the polarization ``P = Q + iU``, used for the
            I, Q, U maps of the result. By default ``(2, 0)`` or ``(2, -4)``,
            whichever is in ``couples``.

    The unknowns are ``I`` at ``(0, 0)`` and, at every other couple, the
    complex amplitude ``Z_{n,m}`` whose spin component is ``Z_{n,m} / 2``
    (``Z_{2,0} = P`` without HWP).
    """

    def __init__(self, couples: Iterable[Couple], polarization: Couple | None = None):
        seen = []
        for n, m in couples:
            key, _ = canonical(n, m)
            if key not in seen:
                seen.append(key)
        if not seen:
            raise ValueError("a basis needs at least one couple")
        # (0, 0) first, then the couples in the given order
        self.couples: tuple[Couple, ...] = tuple(
            sorted(seen, key=lambda c: c != (0, 0))
        )
        if polarization is None:
            candidates = [c for c in (POLARIZATION, POLARIZATION_HWP) if c in seen]
            polarization = candidates[0] if len(candidates) == 1 else None
        else:
            polarization = canonical(*polarization)[0]
            if polarization not in seen:
                raise ValueError(f"polarization {polarization} is not in the basis")
        self.polarization = polarization

    @classmethod
    def iqu(cls, hwp: bool) -> "Basis":
        """I and P, the standard 3x3 system (``mdim = 3``)."""
        return cls([(0, 0), POLARIZATION_HWP if hwp else POLARIZATION])

    @classmethod
    def qu(cls, hwp: bool) -> "Basis":
        """P only, the 2x2 system of the pair difference (``mdim = 2``)."""
        return cls([POLARIZATION_HWP if hwp else POLARIZATION])

    def extend(self, couples: Iterable[Couple]) -> "Basis":
        """A basis with more couples (templates of systematics)."""
        return Basis(list(self.couples) + list(couples), self.polarization)

    @property
    def full(self) -> list[Couple]:
        """Every solved couple, each conjugate pair written out."""
        out = []
        for n, m in self.couples:
            out.append((n, m))
            if (n, m) != (0, 0):
                out.append((-n, -m))
        return out

    @property
    def weights(self) -> np.ndarray:
        """1 for ``(0, 0)``, 1/2 for the others (see the class documentation)."""
        return np.array([1.0 if c == (0, 0) else 0.5 for c in self.full])

    @property
    def dimension(self) -> int:
        return len(self.full)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Basis)
            and self.couples == other.couples
            and self.polarization == other.polarization
        )

    def __hash__(self) -> int:
        return hash((self.couples, self.polarization))

    def __repr__(self) -> str:
        return f"Basis({list(self.couples)}, polarization={self.polarization})"
