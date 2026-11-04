"""Tools to check a signal model against time-ordered data.

A new systematic is validated by writing its time-ordered data (TOD) in real
space, independently of the spin decomposition, and comparing the per-pixel
least-squares fit of those TOD with :func:`sbm.map_make` of the model::

    from sbm.testing import RealSky, assert_matches_tod, random_samples

    real = RealSky(nside=2, seed=1)              # sky values and derivatives
    samples = random_samples(nside=2, n_per_pixel=40, seed=2)
    pixels, psi, phi = samples
    tod = real.detector(pixels, psi) + ...       # your systematic, in real space
    model = sy.sky_signal(real.sky()) + my_systematic(real.sky(), ...)
    assert_matches_tod(model, tod, samples, sbm.Basis.iqu(hwp=False))

The angles are those of SBM: a detector without HWP sees
``I + Q cos 2 psi - U sin 2 psi``, with HWP ``I + Q cos(4 phi - 2 psi) + U sin(4 phi - 2 psi)``.
"""

from __future__ import annotations

import healpy as hp
import numpy as np


def random_samples(nside: int, n_per_pixel: int, seed: int, hwp: bool = False):
    """Pixels, crossing angles ``psi`` and HWP angles ``phi`` of ``n_per_pixel``
    random samples in every pixel (``phi = 0`` without HWP)."""
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    pixels = np.repeat(np.arange(npix), n_per_pixel)
    psi = rng.uniform(-np.pi, np.pi, pixels.size)
    phi = rng.uniform(-np.pi, np.pi, pixels.size) if hwp else np.zeros(pixels.size)
    return pixels, psi, phi


def tod_from_components(components, pixels, psi, phi) -> np.ndarray:
    """Real TOD of the spin components ``{(n, m): S}``, each with its conjugate:
    ``d = S_00 + sum 2 Re[S_nm exp(i (n psi + m phi))]``."""
    d = np.zeros(pixels.size)
    for (n, m), s in components.items():
        term = s[pixels] * np.exp(1j * (n * psi + m * phi))
        d += term.real if (n, m) == (0, 0) else 2.0 * term.real
    return d


def least_squares(d, pixels, psi, phi, couples, npix: int) -> dict:
    """Per-pixel least-squares fit of ``I`` and of the complex amplitudes ``Z``
    whose spin component is ``Z / 2`` at each couple of ``couples``.

    Returns ``{couple: map}`` (``I`` real, ``Z`` complex, ``NaN`` where unobserved).
    """
    out = {c: np.full(npix, complex(np.nan, np.nan)) for c in couples}
    order = np.argsort(pixels, kind="stable")
    bounds = np.searchsorted(pixels[order], np.arange(npix + 1))
    for p in range(npix):
        sel = order[bounds[p] : bounds[p + 1]]
        if sel.size == 0:
            continue
        columns = []
        for n, m in couples:
            theta = n * psi[sel] + m * phi[sel]
            if (n, m) == (0, 0):
                columns.append(np.ones(theta.size))
            else:  # Z/2 e^{i theta} + c.c. = Re Z cos(theta) - Im Z sin(theta)
                columns.extend([np.cos(theta), -np.sin(theta)])
        x = np.linalg.lstsq(np.array(columns).T, d[sel], rcond=None)[0]
        i = 0
        for c in couples:
            if c == (0, 0):
                out[c][p] = x[i]
                i += 1
            else:
                out[c][p] = x[i] + 1j * x[i + 1]
                i += 2
    return out


def assert_matches_tod(
    model, tod, samples, basis, rtol: float = 1e-9, atol: float = 1e-9
):
    """Check that :func:`sbm.map_make` of ``model`` is the least-squares fit of ``tod``.

    ``samples`` is ``(pixels, psi, phi)`` as from :func:`random_samples`. The
    h-maps are computed from the same samples, with every couple that the model
    and the basis need. ``atol`` is relative to the largest value of ``tod``, so
    that the check does not depend on the units.
    """
    from .hmaps import HMaps
    from .map_making import map_make

    pixels, psi, phi = samples
    npix = model.npix
    nside = hp.npix2nside(npix)
    couples = model.required_couples(basis.full) | {
        (ni - nj, mi - mj) for ni, mi in basis.full for nj, mj in basis.full
    }
    hm = HMaps.from_samples(nside, pixels, psi, phi, couples=couples)
    result = map_make((hm, model), basis)
    expected = least_squares(tod, pixels, psi, phi, basis.couples, npix)
    scale = max(float(np.max(np.abs(tod))), np.finfo(float).tiny)
    for c in basis.couples:
        np.testing.assert_allclose(
            result.component(*c),
            expected[c],
            rtol=rtol,
            atol=atol * scale,
            err_msg=f"component {c} of {basis}",
        )


class RealSky:
    """A random sky given by its values and real derivatives in every pixel.

    ``x[s]``, ``d[s, a]`` and ``h[s, a]`` hold the values, the gradient
    (``a`` in ``"x"``, ``"y"``: along theta and phi) and the Hessian (``"xx"``,
    ``"xy"``, ``"yy"``) of ``s`` in ``"I"``, ``"Q"``, ``"U"``. :meth:`sky` returns
    the :class:`sbm.Sky` with the same derivatives, and :meth:`detector` the TOD of
    a detector that samples the sky at an offset pointing (Taylor expansion), so
    that pointing models can be tested without the spin decomposition.
    """

    def __init__(self, nside: int, seed: int):
        rng = np.random.default_rng(seed)
        self.npix = hp.nside2npix(nside)
        self.x = {s: rng.standard_normal(self.npix) for s in "IQU"}
        self.d = {
            (s, a): rng.standard_normal(self.npix) for s in "IQU" for a in ("x", "y")
        }
        self.h = {
            (s, a): rng.standard_normal(self.npix)
            for s in "IQU"
            for a in ("xx", "xy", "yy")
        }

    def _eth(self, s):
        return self.d[s, "y"] - 1j * self.d[s, "x"]

    def _eth2(self, s):  # (d_y - i d_x)^2
        return self.h[s, "yy"] - self.h[s, "xx"] - 2j * self.h[s, "xy"]

    def _lap(self, s):
        return self.h[s, "xx"] + self.h[s, "yy"]

    def sky(self):
        """The :class:`sbm.Sky` of these values and derivatives."""
        from .sky import Sky

        eth, eth2, lap = self._eth, self._eth2, self._lap
        return Sky.from_maps(
            self.x["I"],
            self.x["Q"],
            self.x["U"],
            eth_i=eth("I"),
            eth_p=eth("Q") + 1j * eth("U"),
            eth_bar_p=np.conj(eth("Q")) + 1j * np.conj(eth("U")),
            second_order={
                "eth2_i": eth2("I"),
                "lap_i": lap("I"),
                "eth2_p": eth2("Q") + 1j * eth2("U"),
                "eth_bar2_p": np.conj(eth2("Q")) + 1j * np.conj(eth2("U")),
                "lap_p": lap("Q") + 1j * lap("U"),
            },
        )

    def sample(self, s, pixels, rho=0.0, alpha=0.0, second_order=False):
        """Value of ``s`` at the offset ``rho`` along ``-(sin(alpha) e_x + cos(alpha) e_y)``."""
        dx, dy = -rho * np.sin(alpha), -rho * np.cos(alpha)
        v = (
            self.x[s][pixels]
            + dx * self.d[s, "x"][pixels]
            + dy * self.d[s, "y"][pixels]
        )
        if second_order:
            v = v + 0.5 * (
                dx**2 * self.h[s, "xx"][pixels]
                + 2 * dx * dy * self.h[s, "xy"][pixels]
                + dy**2 * self.h[s, "yy"][pixels]
            )
        return v

    def detector(self, pixels, psi, phi=None, rho=0.0, alpha=0.0, second_order=False):
        """TOD of a single detector: ``I + Q cos 2psi - U sin 2psi`` without HWP
        (``phi=None``), ``I + Q cos(4phi - 2psi) + U sin(4phi - 2psi)`` with HWP,
        sampling the sky at the offset ``rho`` in direction ``alpha``."""
        i, q, u = (self.sample(s, pixels, rho, alpha, second_order) for s in "IQU")
        angle = 2 * psi if phi is None else 2 * psi - 4 * phi
        return i + q * np.cos(angle) - u * np.sin(angle)
