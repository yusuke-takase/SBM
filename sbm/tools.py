import os
from pathlib import Path
import numpy as np
import pandas as pd
import litebird_sim as lbs
from litebird_sim import Imo
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
    """ The function to convert C_ell to D_ell (i.e. ell*(ell+1)*C_ell/(2*pi))

    Args:
        cl: 1d-array
            Power spectrum
        ell_start: float (default = 2.)
            The multi-pole ell value of first index of the `cl`.

    Return:
        dl: 1d-array
    """
    ell = np.arange(ell_start, len(cl)+ell_start)
    return cl*ell*(ell+1.)/(2.*np.pi)

def d2c(dl, ell_start=2.):
    """ The function to convert D_ell to C_ell (i.e. C_ell = D_ell*(2*pi)/(ell*(ell+1)))

    Args:
        dl: 1d-array
            Power spectrum
        ell_start: float (default = 2.)
            The multi-pole ell value of first index of the `dl`.

    Return:
        cl: 1d-array
    """
    ell = np.arange(ell_start, len(dl)+ell_start)
    return dl*(2.*np.pi)/(ell*(ell+1.))

def load_fiducial_cl(r):
    """ This function loads the fiducial CMB power spectrum used in the map base simulation of litebird_sim.

    Args:
        r: float
            The tensor-to-scalar ratio of the CMB.

    Return:
        cl_cmb: 2d-array
    """
    datautils_dir = Path(lbs.__file__).parent / "datautils"
    cl_cmb_scalar = hp.read_cl(datautils_dir / "Cls_Planck2018_for_PTEP_2020_r0.fits")
    cl_cmb_tensor = hp.read_cl(datautils_dir / "Cls_Planck2018_for_PTEP_2020_tensor_r1.fits") * r
    cl_cmb = cl_cmb_scalar + cl_cmb_tensor
    return cl_cmb

def generate_cmb(nside, r=0., cmb_seed=None):
    """ This function generates the CMB map used in the map base simulation of litebird_sim.

    Args:
        nside: int
            The resolution of the map.
        r: float (default = 0.)
            The tensor-to-scalar ratio of the CMB.
        cmb_seed: int (default = None)
            The seed of the random number generator.

    Return:
        cmb (np.ndarray) : The I, Q, U maps of the CMB.
    """
    cl_cmb = load_fiducial_cl(r)
    if cmb_seed is not None:
        np.random.seed(cmb_seed)
    cmb = hp.synfast(cl_cmb, nside=nside, new=True)
    return cmb


def get_instrument_table(imo:Imo, imo_version="v2"):
    """
    This function generates DataFrame which is used for FGBuster as `instrument` from IMo.

    Args:
        imo (Imo): IMo object which contains the instrument information given by the `litebird_sim`

        imo_version (str): version of the IMo. Default is "v2"

    Returns:
        instrument (pd.DataFrame): DataFrame which contains the instrument information
    """
    telescopes     = ["LFT", "MFT", "HFT"]
    channel_list   = []
    freq           = []
    depth_p        = []
    fwhm           = []
    telescope_list = []
    bandwidth      = []
    numOfdets      = []
    net_detector_ukrts = []
    net_channel_ukrts = []

    for i in telescopes:
        inst_info = imo.query("/releases/"+imo_version+"/satellite/"+i+"/instrument_info")
        channel_list.append(inst_info.metadata["channel_names"])
    channel_list = [item for sublist in channel_list for item in sublist]

    for i in channel_list:
        if i[0]   == "L":
            telescope = "LFT"
        elif i[0] == "M":
            telescope = "MFT"
        elif i[0] == "H":
            telescope = "HFT"
        chinfo = lbs.FreqChannelInfo.from_imo(imo,
                  "/releases/{}/satellite/{}/{}/channel_info".format(imo_version, telescope, i))
        freq.append(chinfo.band.bandcenter_ghz)
        depth_p.append(chinfo.pol_sensitivity_channel_ukarcmin)
        fwhm.append(chinfo.fwhm_arcmin)
        telescope_list.append(telescope)
        bandwidth.append(chinfo.bandwidth_ghz)
        net_detector_ukrts.append(chinfo.net_detector_ukrts)
        net_channel_ukrts.append(chinfo.net_channel_ukrts)
        numOfdets.append(len(chinfo.detector_names))

    instrument = pd.DataFrame(data = {
        'channel'    : channel_list,
        'frequency'  : freq,
        'bandwidth'  : bandwidth,
        'depth_p'    : depth_p,
        'fwhm'       : fwhm,
        'net_detector_ukrts' : net_detector_ukrts,
        'net_channel_ukrts'  : net_channel_ukrts,
        'ndet'       : numOfdets,
        'f_sky'      : [1.0                  for i in range(len(channel_list))],
        'status'     : ["forecast"           for i in range(len(channel_list))],
        'reference'  : ["IMo-" + imo_version for i in range(len(channel_list))],
        'type'       : ["satellite"          for i in range(len(channel_list))],
        'experiment' : ["LiteBIRD"           for i in range(len(channel_list))],
        'telescope'  : telescope_list
    })
    return instrument
