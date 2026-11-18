"""Fiducial CMB power spectra and CMB realizations.

The fiducial spectra are those of the Planck 2018 best fit used for the
LiteBIRD forecasts (PTEP 2023), shipped with SBM in ``sbm/datautils``: the
lensed scalar spectra (``r = 0``) and the tensor spectra for ``r = 1``. The
rows are TT, EE, BB, TE (healpy ordering).
"""

from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np

DATA_DIR = Path(__file__).parent / "datautils"


def load_fiducial_cl_lens(lmax: int | None = None) -> np.ndarray:
    """Lensed scalar spectra (``r = 0``), up to ``lmax``."""
    cl = hp.read_cl(DATA_DIR / "Cls_Planck2018_for_PTEP_2020_r0.fits")
    return cl if lmax is None else cl[:, : lmax + 1]


def load_fiducial_cl_tens(r: float, lmax: int | None = None) -> np.ndarray:
    """Tensor spectra for the tensor-to-scalar ratio ``r``, up to ``lmax``."""
    cl = hp.read_cl(DATA_DIR / "Cls_Planck2018_for_PTEP_2020_tensor_r1.fits") * r
    return cl if lmax is None else cl[:, : lmax + 1]


def load_fiducial_cl(r: float, lmax: int | None = None) -> np.ndarray:
    """Lensed scalar plus tensor spectra, up to ``lmax``."""
    return load_fiducial_cl_lens(lmax) + load_fiducial_cl_tens(r, lmax)


def generate_cmb(nside: int, r: float = 0.0, cmb_seed: int | None = None) -> np.ndarray:
    """I, Q, U maps of a CMB realization of the fiducial spectra.

    ``cmb_seed`` seeds the global NumPy generator used by ``healpy.synfast``, so
    a seed gives the same realization as in SBM 0.4.
    """
    cl = load_fiducial_cl(r)
    if cmb_seed is not None:
        np.random.seed(cmb_seed)
    return hp.synfast(cl, nside=nside, new=True)


def c2d(cl, ell_start: int = 2) -> np.ndarray:
    """``D_ell = ell (ell + 1) C_ell / 2 pi``; ``cl[0]`` is the multipole ``ell_start``."""
    ell = np.arange(ell_start, len(cl) + ell_start)
    return cl * ell * (ell + 1.0) / (2.0 * np.pi)


def d2c(dl, ell_start: int = 2) -> np.ndarray:
    """``C_ell = 2 pi D_ell / ell (ell + 1)``; ``dl[0]`` is the multipole ``ell_start``."""
    ell = np.arange(ell_start, len(dl) + ell_start)
    return dl * (2.0 * np.pi) / (ell * (ell + 1.0))
