"""Spin weights of the sky maps, checked by the signal models.

When the local basis ``(e_theta, e_phi)`` turns by an angle ``alpha``, in the
sense in which the polarization angle is counted, ``P = Q + iU`` is multiplied by
``exp(-2i alpha)``: a field of spin weight ``s`` is multiplied by
``exp(-i s alpha)``. ``I`` has spin weight 0 and ``P`` 2; ``eth`` raises the spin
weight by 1 and ``eth_bar`` lowers it by 1, so that ``eth I`` has 1, ``eth P`` 3
and ``eth_bar P`` 1. The crossing angle ``psi`` of SBM is counted in the opposite
sense (:doc:`conventions`) and becomes ``psi + alpha``, while the HWP angle
``phi`` does not change. The data ``d = sum S_{n,m} exp(i (n psi + m phi))``
are the same in every basis only if ``S_{n,m}`` has spin weight ``n``: **the spin
weights of the factors on the right-hand side of a component add up to n**.
A term that breaks this rule is a wrong couple, a missing conjugate or a
derivative of the wrong kind.

The maps of a :class:`~sbm.Sky` are :class:`SpinMap` arrays, which carry their
spin weight through the arithmetic of a model:

* a product adds the spin weights and a quotient subtracts them;
* the complex conjugate negates it, and ``abs`` gives 0;
* a sum or a difference needs the same spin weight on both sides (adding 0 is
  allowed) and raises :class:`SpinWeightError` otherwise;
* a number or a 0-d array has spin weight 0;
* an array without spin weight (e.g. ``sky.q``, or a map read with healpy) makes
  the spin weight of the result unknown, and so does any other function of a map
  whose spin weight is not 0 (``np.real(sky.p)``, ``np.exp(sky.p)``, ...).

:class:`~sbm.SignalModel` raises :class:`SpinWeightError` for a component at
``(n, m)`` whose spin weight is known and is not ``n``, and stores plain arrays,
so that the map-maker is not affected.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SpinMap", "SpinWeightError", "spin_weight"]


class SpinWeightError(ValueError):
    """Terms of different spin weights added, or a component at the wrong couple."""


class SpinMap(np.ndarray):
    """A map with a spin weight, ``spin`` (``None`` if unknown).

    It is a NumPy array: indexing, copies and ``astype`` keep the spin weight,
    ufuncs follow the rules of :mod:`sbm.spin`, and the other NumPy functions
    return plain arrays.
    """

    def __new__(cls, values, spin: int | None):
        obj = np.asarray(values).view(cls)
        obj.spin = None if spin is None else int(spin)
        return obj

    def __array_finalize__(self, obj):
        self.spin = getattr(obj, "spin", None)

    def __reduce__(self):
        constructor, args, state = super().__reduce__()
        return constructor, args, (state, self.spin)

    def __setstate__(self, state):
        state, spin = state
        super().__setstate__(state)
        self.spin = spin

    def __repr__(self) -> str:
        return f"SpinMap({np.asarray(self)!r}, spin={self.spin})"

    @property
    def real(self):
        values = np.asarray(self).real
        return SpinMap(values, 0) if self.spin == 0 else values

    @real.setter
    def real(self, value):
        np.asarray(self).real = value

    @property
    def imag(self):
        values = np.asarray(self).imag
        return SpinMap(values, 0) if self.spin == 0 else values

    @imag.setter
    def imag(self, value):
        np.asarray(self).imag = value

    def conj(self):
        return np.conjugate(self)

    conjugate = conj

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        plain = [_plain(x) for x in inputs]
        if out is not None:
            kwargs["out"] = tuple(_plain(o) for o in out)
        result = getattr(ufunc, method)(*plain, **kwargs)
        if method != "__call__" or ufunc.nout != 1:
            return result  # reductions, outer products, ...: no spin weight
        spin = _result_spin(ufunc, inputs)
        if out is not None:
            target = out[0]
            if isinstance(target, SpinMap):
                target.spin = spin
            return target
        if (
            spin is None
            or not isinstance(result, np.ndarray)
            or result.ndim == 0
            or result.dtype == bool
        ):
            return result
        return SpinMap(result, spin)

    def __array_function__(self, func, types, args, kwargs):
        if func in (np.real, np.imag) and isinstance(args[0], SpinMap):
            return getattr(args[0], func.__name__)
        return func(*_strip(args), **_strip(kwargs))


def spin_weight(value) -> int | None:
    """Spin weight of a map: that of a :class:`SpinMap`, 0 for a number, ``None``
    for an array without spin weight."""
    if isinstance(value, SpinMap):
        return value.spin
    if np.ndim(value) == 0:
        return 0
    return None


def _plain(x):
    return np.asarray(x) if isinstance(x, SpinMap) else x


def _strip(obj):
    if isinstance(obj, SpinMap):
        return np.asarray(obj)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_strip(x) for x in obj)
    if isinstance(obj, dict):
        return {key: _strip(value) for key, value in obj.items()}
    return obj


def _is_zero_number(x) -> bool:
    return not isinstance(x, SpinMap) and np.ndim(x) == 0 and x == 0


def _result_spin(ufunc, inputs) -> int | None:
    spins = [spin_weight(x) for x in inputs]
    if ufunc in (np.add, np.subtract):
        terms = [s for s, x in zip(spins, inputs) if not _is_zero_number(x)]
        if None in terms:
            return None
        if len(set(terms)) > 1:
            raise SpinWeightError(
                f"cannot add maps of spin weights {terms[0]} and {terms[1]}: the "
                "terms of a spin component must have the same spin weight "
                "(see sbm.spin)"
            )
        return terms[0] if terms else 0
    if None in spins:
        return None
    if ufunc is np.multiply:
        return spins[0] + spins[1]
    if ufunc is np.divide:
        return spins[0] - spins[1]
    if ufunc in (np.negative, np.positive):
        return spins[0]
    if ufunc is np.conjugate:
        return -spins[0]
    if ufunc is np.absolute:
        return 0
    if ufunc is np.square:
        return 2 * spins[0]
    if ufunc is np.power and np.ndim(inputs[1]) == 0 and np.isrealobj(inputs[1]):
        exponent = float(inputs[1])
        if exponent.is_integer():
            return spins[0] * int(exponent)
        return 0 if spins[0] == 0 else None
    # any other function of maps: spin weight 0 only if all its arguments have 0
    return 0 if all(s == 0 for s in spins) else None
