"""Orientation functions (h-maps) of a detector.

SBM describes the scanning strategy through the orientation functions

    h̃_{n,m}(p) = < exp(-i (n psi + m phi)) >_p ,

averaged over the samples falling in pixel ``p``, where ``psi`` is the crossing
angle of the detector and ``phi`` the HWP angle (the angle of the HWP with
respect to the telescope, counted opposite to ``psi``). ``h̃_{-n,-m}`` is the
complex conjugate of ``h̃_{n,m}`` and ``h̃_{0,0} = 1``.

litebird_sim and Falcons.jl write ``h_{n,m} = < exp(+i (n psi' + m chi)) >``
with their orientation angle ``psi' = -psi`` and HWP angle ``chi = phi``, so
``h̃_{n,m} = h_{n,-m}``. :meth:`HMaps.load` and :meth:`HMaps.save` convert
between the two.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

Couple = tuple[int, int]


def canonical(n: int, m: int) -> tuple[Couple, bool]:
    """Return the canonical couple of ``(n, m)`` and whether it is its conjugate.

    Every couple but ``(0, 0)`` is stored once: ``n > 0``, or ``n == 0`` and
    ``m > 0``. The others are the complex conjugates of their opposite.
    """
    n, m = int(n), int(m)
    if n > 0 or (n == 0 and m >= 0):
        return (n, m), False
    return (-n, -m), True


class HMaps:
    """Orientation functions and hit counts of one detector (or of a set of
    detectors with the same signal model).

    Args:
        h: maps of ``h̃_{n,m}`` indexed by ``(n, m)``. A couple and its opposite
            are the same information; only one of them is needed.
        hits: number of samples per pixel.
        duration_s, sampling_rate_hz, coordinate_system, name: metadata of the
            observation, kept when the maps are saved.

    The maps are not modified after construction. Unobserved pixels have
    ``hits == 0`` and ``h̃ == 0``.
    """

    def __init__(
        self,
        h: dict,
        hits,
        *,
        duration_s: float | None = None,
        sampling_rate_hz: float | None = None,
        coordinate_system: str | None = None,
        name: str | None = None,
    ):
        hits = np.array(hits, dtype=float)  # a copy: it is made read-only
        self.nside = hp.npix2nside(hits.size)
        self.hits = hits
        self.hits.setflags(write=False)
        self._h = {}
        for (n, m), value in h.items():
            key, conj = canonical(n, m)
            if key == (0, 0):
                continue
            value = np.asarray(value, dtype=complex)
            if value.shape != hits.shape:
                raise ValueError(
                    f"h_{{{n},{m}}} has shape {value.shape}, expected {hits.shape}"
                )
            value = np.conj(value) if conj else value.copy()
            value.setflags(write=False)
            self._h.setdefault(key, value)
        self.duration_s = duration_s
        self.sampling_rate_hz = sampling_rate_hz
        self.coordinate_system = coordinate_system
        self.name = name

    @property
    def npix(self) -> int:
        return self.hits.size

    @property
    def couples(self) -> list[Couple]:
        """Canonical couples available (``(0, 0)`` is always available)."""
        return sorted(self._h)

    def has(self, n: int, m: int) -> bool:
        key, _ = canonical(n, m)
        return key == (0, 0) or key in self._h

    def h(self, n: int, m: int) -> np.ndarray:
        """Return ``h̃_{n,m}`` (``h̃_{0,0} = 1`` in every pixel)."""
        key, conj = canonical(n, m)
        if key == (0, 0):
            return np.ones(self.npix, dtype=complex)
        try:
            value = self._h[key]
        except KeyError:
            raise KeyError(
                f"h_{{{n},{m}}} is not in these h-maps; available couples "
                f"(and their conjugates): {self.couples}"
            ) from None
        return np.conj(value) if conj else value

    def crosslink_mask(self, threshold: float) -> np.ndarray:
        """Observed pixels with ``|h̃_{2,0}| < threshold``.

        This is the pixel selection of the map-maker of SBM 0.4 without HWP,
        whose ``xlink_threshold`` was 1 by default.
        """
        return (self.hits > 0) & (np.abs(self.h(2, 0)) < threshold)

    # ------------------------------------------------------------------ #
    # Construction

    @classmethod
    def from_samples(
        cls,
        nside: int,
        pixels,
        psi,
        phi=None,
        couples: Iterable[Couple] = (),
        **metadata,
    ) -> "HMaps":
        """Compute the orientation functions of a set of samples.

        Args:
            nside: HEALPix resolution.
            pixels: pixel index of each sample.
            psi: crossing angle of each sample, in the SBM convention (radians).
            phi: HWP angle of each sample (radians); zero if omitted.
            couples: the ``(n, m)`` to compute.
        """
        pixels = np.asarray(pixels)
        psi = np.asarray(psi, dtype=float)
        phi = np.zeros_like(psi) if phi is None else np.asarray(phi, dtype=float)
        npix = hp.nside2npix(nside)
        hits = np.bincount(pixels, minlength=npix).astype(float)
        seen = hits > 0
        h = {}
        for n, m in couples:
            key, _ = canonical(n, m)
            if key == (0, 0) or key in h:
                continue
            phase = key[0] * psi + key[1] * phi
            value = np.zeros(npix, dtype=complex)
            value.real = np.bincount(pixels, weights=np.cos(phase), minlength=npix)
            value.imag = -np.bincount(pixels, weights=np.sin(phase), minlength=npix)
            value[seen] /= hits[seen]
            h[key] = value
        return cls(h, hits, **metadata)

    @staticmethod
    def combine(hmaps: Sequence["HMaps"]) -> "HMaps":
        """Hit-weighted combination of the h-maps of several detectors.

        Map-making with the combination is the same as map-making with each
        detector when they share the same signal model. The h-maps do not hold
        the polarization angle of the detectors, which changes their signal
        model: combine only detectors with the same polarization angle.
        """
        hmaps = list(hmaps)
        if not hmaps:
            raise ValueError("no h-maps to combine")
        couples = set(hmaps[0].couples)
        for other in hmaps[1:]:
            if other.nside != hmaps[0].nside:
                raise ValueError("the h-maps have different nside")
            couples &= set(other.couples)
        hits = np.sum([hm.hits for hm in hmaps], axis=0)
        seen = hits > 0
        h = {}
        for key in sorted(couples):
            value = np.sum([hm.hits * hm.h(*key) for hm in hmaps], axis=0)
            value[seen] /= hits[seen]
            value[~seen] = 0.0
            h[key] = value
        durations = [hm.duration_s for hm in hmaps]
        return HMaps(
            h,
            hits,
            duration_s=max(durations) if None not in durations else None,
            sampling_rate_hz=hmaps[0].sampling_rate_hz,
            coordinate_system=hmaps[0].coordinate_system,
            name=None,
        )

    # ------------------------------------------------------------------ #
    # Files

    @classmethod
    def load(cls, path) -> "HMaps":
        """Read an h-map file of litebird_sim or Falcons.jl (``h_maps_det_<name>.h5``).

        The ``(0, 0)`` group holds the hit counts. The convention is converted
        (``h̃_{n,m} = h_{n,-m}``, see the module documentation).
        """
        with h5py.File(path, "r") as f:
            groups = {}
            for key in f.keys():
                couple = _parse_couple(key)
                if couple is not None:  # other groups or datasets are ignored
                    groups[couple] = f[key]
            if (0, 0) not in groups:
                raise ValueError(f"{path} has no (0, 0) group with the hit counts")
            hits = np.asarray(groups[(0, 0)]["Re"][()], dtype=float)
            h = {}
            for (n, m), group in groups.items():
                if (n, m) == (0, 0):
                    continue
                value = group["Re"][()] + 1j * group["Im"][()]
                key, conj = canonical(n, -m)
                h.setdefault(key, np.conj(value) if conj else value)
            attrs = dict(f.attrs)
        return cls(
            h,
            hits,
            duration_s=_float_or_none(attrs.get("duration_s")),
            sampling_rate_hz=_float_or_none(attrs.get("sampling_rate_Hz")),
            coordinate_system=_str_or_none(attrs.get("coordinate_system")) or None,
            name=_str_or_none(attrs.get("det")) or None,
        )

    @classmethod
    def load_legacy(cls, path) -> "HMaps":
        """Read a cross-link file in the format of SBM 0.4 (``h``, ``hitmap``,
        ``toml/spin_n``, ``toml/spin_m``, ``ss``), as written by Falcons.jl 1.0.

        The values are those that ``ScanFields.get_xlink`` returned: the
        ``n = 0`` couples are the conjugates of the stored ``(0, -m)``.
        """
        with h5py.File(path, "r") as f:
            hits = np.asarray(f["hitmap"][()], dtype=float)
            stored = f["h"][()]
            spins_n = [int(n) for n in f["toml"]["spin_n"][()]]
            spins_m = [int(m) for m in f["toml"]["spin_m"][()]]
            ss = {key: f["ss"][key][()] for key in f["ss"].keys() if key != "quat"}
        stored = np.where(np.isnan(stored), 0.0, stored)
        h = {}
        for i_n, n in enumerate(spins_n):
            for i_m, m in enumerate(spins_m):
                if n > 0:
                    h[(n, m)] = stored[:, i_m, i_n]
                elif n == 0 and -m in spins_m and m > 0:
                    h[(0, m)] = np.conj(stored[:, spins_m.index(-m), i_n])
        name = ss.get("name")
        if isinstance(name, np.ndarray):
            name = name[0] if name.size else None
        return cls(
            h,
            hits,
            duration_s=_float_or_none(ss.get("duration")),
            sampling_rate_hz=_float_or_none(ss.get("sampling_rate")),
            coordinate_system=_str_or_none(ss.get("coord")),
            name=_str_or_none(name),
        )

    def save(self, path) -> Path:
        """Write the maps as an h-map file of litebird_sim and Falcons.jl.

        Every couple is written in their convention, ``h_{n,m} = h̃_{n,-m}``,
        with ``n >= 0``; the ``(0, 0)`` group holds the hit counts. The coordinate
        system must be known, ``"Galactic"`` or ``"Ecliptic"`` (or ``"G"``,
        ``"E"``): set :attr:`coordinate_system` first for maps computed with
        :meth:`from_samples`.
        """
        frame = _LBS_COORDINATES.get(self.coordinate_system, self.coordinate_system)
        if frame not in _LBS_COORDINATES.values():
            raise ValueError(
                f"the coordinate system of the h-maps is {self.coordinate_system!r}: "
                "set hmaps.coordinate_system to 'Galactic' or 'Ecliptic' before "
                "saving (the files of litebird_sim and Falcons.jl need one of them)"
            )
        path = Path(path)
        with h5py.File(path, "w") as f:
            f.attrs["coordinate_system"] = frame
            f.attrs["det"] = self.name or ""
            f.attrs["detector_split"] = "full"
            f.attrs["time_split"] = "full"
            # litebird_sim reads both attributes; NaN stands for unknown
            f.attrs["duration_s"] = _nan_if_none(self.duration_s)
            f.attrs["sampling_rate_Hz"] = _nan_if_none(self.sampling_rate_hz)
            f["0,0/Re"] = self.hits
            f["0,0/Im"] = np.zeros_like(self.hits)
            for n, m in self.couples:
                value = self.h(n, m)
                f[f"{n},{-m}/Re"] = value.real
                f[f"{n},{-m}/Im"] = value.imag
        return path


# names of litebird_sim's CoordinateSystem for the codes of the legacy files
_LBS_COORDINATES = {"G": "Galactic", "E": "Ecliptic"}


def _parse_couple(key: str):
    """``(n, m)`` of a group name ``"n,m"``, ``None`` for any other name."""
    parts = key.split(",")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _float_or_none(value):
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value


def _nan_if_none(value):
    return np.nan if value is None else float(value)


def _str_or_none(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)
