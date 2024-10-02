import os
from pathlib import Path
import numpy as np
import pandas as pd
import litebird_sim as lbs
import healpy as hp
from matplotlib.colors import ListedColormap

def get_cmap():
    """ This function generates color scheme which is often used Planck paper. """
    datautils_dir = Path(__file__).parent / "datautils"
    color_data = np.loadtxt(datautils_dir / "Planck_Parchment_RGB.txt")
    colombi1_cmap = ListedColormap(color_data/255.)
    colombi1_cmap.set_bad("gray")
    colombi1_cmap.set_under("white")
    planck_cmap = colombi1_cmap
    return planck_cmap

def c2d(cl, ell_start=2.):
    """ The function to convert C_ell to D_ell

    Parameters
    ----------
        cl: 1d-array
            Power spectrum
        ell_start:float (default = 2.)
            The multi-pole ell value of first index of the `cl`.

    Return
    ------
        dl: 1d-array
    """
    ell = np.arange(ell_start, len(cl)+ell_start)
    return cl*ell*(ell+1.)/(2.*np.pi)

def d2c(dl, ell_start=2.):
    """ The function to convert D_ell to C_ell

    Parameters
    ----------
        dl: 1d-array
            (Reduced) Power spectrum
        ell_start:float (default = 2.)
            The multi-pole ell value of first index of the `dl`.

    Return
    ------
        cl: 1d-array
    """
    ell = np.arange(ell_start, len(dl)+ell_start)
    return dl*(2.*np.pi)/(ell*(ell+1.))

def load_fiducial_cl(r):
    """ This function reads the power spectrum of the CMB used in the map base simulation of litebird_sim.

    Parameter
    ---------
        r: int

    Return
    ------
        cl: 2d-arrays
    """
    datautils_dir = Path(lbs.__file__).parent / "datautils"
    cl_cmb_scalar = hp.read_cl(datautils_dir / "Cls_Planck2018_for_PTEP_2020_r0.fits")
    cl_cmb_tensor = hp.read_cl(datautils_dir / "Cls_Planck2018_for_PTEP_2020_tensor_r1.fits") * r
    cl_cmb = cl_cmb_scalar + cl_cmb_tensor
    return cl_cmb

def generate_cmb(nside, r=0., smb_seed=None):
    """ This function generates the CMB map used in the map base simulation of litebird_sim.

    Return
    ------
        cmb: 1d-array
    """
    cl_cmb = load_fiducial_cl(r)
    if smb_seed is not None:
        np.random.seed(smb_seed)
    cmb = hp.synfast(cl_cmb, nside=nside, new=True)
    return cmb
