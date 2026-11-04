"""Time-ordered data and binned least squares, written independently of sbm.

The samples are random crossing angles ``psi`` and HWP angles ``phi`` in every
pixel; a signal model gives the data through

    d = sum_{n,m} S_{n,m} exp(i (n psi + m phi)) .
"""

import healpy as hp
import numpy as np


def random_samples(nside, n_per_pixel, seed, hwp=False):
    """Pixels, psi and phi of ``n_per_pixel`` random samples in every pixel."""
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    pixels = np.repeat(np.arange(npix), n_per_pixel)
    psi = rng.uniform(-np.pi, np.pi, pixels.size)
    phi = rng.uniform(-np.pi, np.pi, pixels.size) if hwp else np.zeros(pixels.size)
    return pixels, psi, phi


def tod_from_components(components, pixels, psi, phi):
    """Real data of the spin components ``{(n, m): S}`` (each with its conjugate)."""
    d = np.zeros(pixels.size)
    for (n, m), s in components.items():
        term = s[pixels] * np.exp(1j * (n * psi + m * phi))
        d += term.real if (n, m) == (0, 0) else 2.0 * term.real
    return d


def least_squares(d, pixels, psi, phi, couples, npix):
    """Per-pixel least-squares fit of ``I`` and of the complex amplitudes ``Z``
    whose spin component is ``Z / 2`` at each couple of ``couples``.

    Returns ``{couple: map}`` (``I`` real, ``Z`` complex).
    """
    out = {c: np.full(npix, np.nan, dtype=complex) for c in couples}
    for p in range(npix):
        sel = pixels == p
        if not np.any(sel):
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
