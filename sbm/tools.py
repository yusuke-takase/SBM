# -*- encoding: utf-8 -*-

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import litebird_sim as lbs
from litebird_sim import Imo
import healpy as hp
from matplotlib.colors import ListedColormap
from iminuit import Minuit


def get_cmap():
    """This function generates color scheme which is often used Planck paper."""
    datautils_dir = Path(__file__).parent / "datautils"
    color_data = np.loadtxt(datautils_dir / "Planck_Parchment_RGB.txt")
    colombi1_cmap = ListedColormap(color_data / 255.0)
    colombi1_cmap.set_bad("gray")
    colombi1_cmap.set_under("white")
    planck_cmap = colombi1_cmap
    return planck_cmap


def c2d(cl, ell_start=2):
    """The function to convert :math:`C_{l}` to :math:`D_{l}` i.e.

    .. math::
        D_{l} = C_{l}l(l+1)}/2\pi

    Args:
        cl (`np.ndarray`, 1d-array): Power spectrum
        ell_start (`int`, default = 2): The multi-pole ell value of first index of the :math:`C_{l}`.

    Return:
        dl (`np.ndarray`,1d-array)
    """
    ell = np.arange(ell_start, len(cl) + ell_start)
    return cl * ell * (ell + 1.0) / (2.0 * np.pi)


def d2c(dl, ell_start=2):
    """The function to convert :math:`D_{l}` to :math:`C_{l}` i.e.

    .. math::
        C_{l} = D_{l}2\pi/l(l+1)

    Args:
        dl (`np.ndarray`, 1d-array): Power spectrum
        ell_start (`int`, default = 2): The multi-pole ell value of first index of the :math:`D_{l}`.

    Return:
        cl (`np.ndarray`,1d-array)
    """
    ell = np.arange(ell_start, len(dl) + ell_start)
    return dl * (2.0 * np.pi) / (ell * (ell + 1.0))


def load_fiducial_cl_lens(lmax=None):
    """This function loads the fiducial CMB power spectrum used in the map base simulation of litebird_sim.

    Args:
        lmax: ell max

    Return:
        lensing cl_cmb (`np.ndarray`, 2d-array)
    """
    datautils_dir = Path(lbs.__file__).parent / "datautils"
    cl_cmb_scalar = hp.read_cl(datautils_dir / "Cls_Planck2018_for_PTEP_2020_r0.fits")
    
    cl_cmb = cl_cmb_scalar 
    if lmax is not None:
        cl_cmb = cl_cmb[:, : lmax + 1]
    return cl_cmb

def load_fiducial_cl_tens(r, lmax=None):
    """This function loads the fiducial CMB power spectrum used in the map base simulation of litebird_sim.

    Args:
        r (`float`): The tensor-to-scalar ratio of the CMB.
        lmax: ell max

    Return:
        tensor cl_cmb (`np.ndarray`, 2d-array)
    """
    datautils_dir = Path(lbs.__file__).parent / "datautils"
    cl_cmb_tensor = (
        hp.read_cl(datautils_dir / "Cls_Planck2018_for_PTEP_2020_tensor_r1.fits") * r
    )
    cl_cmb = cl_cmb_tensor
    if lmax is not None:
        cl_cmb = cl_cmb[:, : lmax + 1]
    return cl_cmb

def load_fiducial_cl(r, lmax=None):
    """This function loads the fiducial CMB power spectrum used in the map base simulation of litebird_sim.

    Args:
        r (`float`): The tensor-to-scalar ratio of the CMB.
        lmax: ell max

    Return:
        tensor cl_cmb (`np.ndarray`, 2d-array)
    """
    cl_cmb_scalar = load_fiducial_cl_lens(lmax)
    cl_cmb_tensor = load_fiducial_cl_tens(r, lmax)
    cl_cmb = cl_cmb_scalar + cl_cmb_tensor
    if lmax is not None:
        cl_cmb = cl_cmb[:, : lmax + 1]
    return cl_cmb



def generate_cmb(nside, r=0.0, cmb_seed=None):
    """This function generates the CMB map used in the map base simulation of litebird_sim.

    Args:
        nside (`int`): The resolution of the map.

        r (`float`, default = 0): The tensor-to-scalar ratio of the CMB.

        cmb_seed (`int`, default = None): The seed of the random number generator.

    Return:
        cmb (`np.ndarray`) : The I, Q, U maps of the CMB.
    """
    cl_cmb = load_fiducial_cl(r)
    if cmb_seed is not None:
        np.random.seed(cmb_seed)
    cmb = hp.synfast(cl_cmb, nside=nside, new=True)
    return cmb


def get_instrument_table(imo: Imo, imo_version="v2"):
    """
    This function generates DataFrame which is used for FGBuster as `instrument` from IMo.

    Args:
        imo (`lbs.Imo`): IMo object which contains the instrument information given by the `litebird_sim`

        imo_version (`str`): version of the IMo. Default is "v2"

    Returns:
        instrument (`pandas.DataFrame`): DataFrame which contains the instrument information
    """
    telescopes = ["LFT", "MFT", "HFT"]
    channel_list = []
    freq = []
    depth_p = []
    fwhm = []
    telescope_list = []
    bandwidth = []
    numOfdets = []
    net_detector_ukrts = []
    net_channel_ukrts = []

    for i in telescopes:
        inst_info = imo.query(
            "/releases/" + imo_version + "/satellite/" + i + "/instrument_info"
        )
        channel_list.append(inst_info.metadata["channel_names"])
    channel_list = [item for sublist in channel_list for item in sublist]

    for i in channel_list:
        if i[0] == "L":
            telescope = "LFT"
        elif i[0] == "M":
            telescope = "MFT"
        elif i[0] == "H":
            telescope = "HFT"
        chinfo = lbs.FreqChannelInfo.from_imo(
            imo,
            "/releases/{}/satellite/{}/{}/channel_info".format(
                imo_version, telescope, i
            ),
        )
        freq.append(chinfo.band.bandcenter_ghz)
        depth_p.append(chinfo.pol_sensitivity_channel_ukarcmin)
        fwhm.append(chinfo.fwhm_arcmin)
        telescope_list.append(telescope)
        bandwidth.append(chinfo.bandwidth_ghz)
        net_detector_ukrts.append(chinfo.net_detector_ukrts)
        net_channel_ukrts.append(chinfo.net_channel_ukrts)
        numOfdets.append(len(chinfo.detector_names))

    instrument = pd.DataFrame(
        data={
            "channel": channel_list,
            "frequency": freq,
            "bandwidth": bandwidth,
            "depth_p": depth_p,
            "fwhm": fwhm,
            "net_detector_ukrts": net_detector_ukrts,
            "net_channel_ukrts": net_channel_ukrts,
            "ndet": numOfdets,
            "f_sky": [1.0 for i in range(len(channel_list))],
            "status": ["forecast" for i in range(len(channel_list))],
            "reference": ["IMo-" + imo_version for i in range(len(channel_list))],
            "type": ["satellite" for i in range(len(channel_list))],
            "experiment": ["LiteBIRD" for i in range(len(channel_list))],
            "telescope": telescope_list,
        }
    )
    return instrument


def _get_likelihood(x, ell, cl_tens, cl_lens, cl_syst, n_el, fsky):  # x is r
    Cl_hat = (
        cl_syst[ell - 2] + cl_lens[ell - 2] + n_el[ell - 2]
    )  # there should be the noise cl_noise (noise and fg residuals), now assuming noiseless case
    Cl = (
        x * cl_tens[ell - 2] + cl_lens[ell - 2] + n_el[ell - 2]
    )  # there should be the noise cl_noise (noise and fg residuals), now assuming noiseless case
    return -np.sum(
        (-0.5)
        * fsky
        * (2.0 * ell + 1.0)
        * (
            (Cl_hat / Cl)
            + np.log(Cl)
            - ((2.0 * ell - 1.0) / (2.0 * ell + 1.0)) * np.log(Cl_hat)
        )
    )


def _get_likelihood_gradient(x, ell, cl_tens, cl_lens, cl_syst, n_el, fsky):  # x is r
    """Derivative in r of the negative log-likelihood returned by `_get_likelihood`."""
    Cl_hat = cl_syst[ell - 2] + cl_lens[ell - 2] + n_el[ell - 2]
    Cl = x * cl_tens[ell - 2] + cl_lens[ell - 2] + n_el[ell - 2]
    return np.sum(0.5 * fsky * (2.0 * ell + 1.0) * cl_tens[ell - 2] * (Cl - Cl_hat) / Cl**2)


def _maximize_likelihood(ell, cl_tens, cl_lens, cl_syst, n_el, fsky, rmax, rtol=1e-12, maxiter=400):
    """Maximum of the likelihood over 0 <= r <= rmax, found as the root of its derivative.

    The derivative is negative below the maximum and positive above it, so the root is bracketed
    by bisection to a relative precision `rtol`. If the likelihood already decreases at r = 0
    (no positive bias, e.g. a residual at the level of rounding errors) the maximum is r = 0.
    """

    def grad(r):
        return _get_likelihood_gradient(r, ell, cl_tens, cl_lens, cl_syst, n_el, fsky)

    if grad(0.0) >= 0.0:
        return 0.0
    if grad(rmax) <= 0.0:
        warnings.warn(f"likelihood maximum is above rmax={rmax}; returning rmax")
        return float(rmax)
    lo, hi = 0.0, float(rmax)
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        if grad(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= rtol * hi:
            break
    return 0.5 * (lo + hi)


def forecast(
    cl_syst,
    n_el=None,
    fsky=1.0,
    lmax=191,
    r0=0.0,
    tol=1e-8,
    rmin=1e-10,
    rmax=100,
    rresol=1000,
    method="exact",
):
    """
    This function estimates the bias on the tensor-to-scalar ratio due to pointing systematics
    This function based on the paper: https://academic.oup.com/ptep/article/2023/4/042F01/6835420, P88, Sec. (5.3.2)

    Args:
        cl_syst (`np.ndarray`, 1d array): residual B-modes power spectrum

        fsky (`float`): sky fraction

        lmax (`int`): maximum multipole considered in the maximization

        rmin (`float`): lower limit of r for ``method="minuit"`` (``"exact"`` searches
            from r = 0), and the scale of the displayed grid when the bias is zero

        rmax (`float`): maximum value of r considered in the maximization

        rresol (`int`): resolution of the grid in the r range

        method (`str`): how the maximum of the likelihood in r is found.
            ``"exact"`` (default) solves for the root of its derivative by bisection over 0 <= r <= rmax
            to a relative precision of 1e-12, and returns 0 when the likelihood decreases from r = 0.
            ``"minuit"`` is the previous MIGRAD minimisation; its precision is limited to about 1e-9 by
            `tol`, and it returns `rmin` when the maximum lies below it.
    """
    assert (
        cl_syst[0] == cl_syst[1] == 0.0
    ), "First and second elements must be 0 because it is recognized as monopole and dipole"

    # l range, from 2 to lmax
    ell = np.arange(2, lmax + 1)

    # the [2] selects the BB spectra and the [2:lmax+1] excludes multipoles 0 and 1,
    # that are null, and multipole above lmax
    cl_tens = load_fiducial_cl_tens(r=1.0, lmax=lmax)[2][2:]
    cl_lens = load_fiducial_cl_lens(lmax=lmax)[2][2:]
    cl_syst = cl_syst[2 : lmax + 1]
    if n_el is None:
        n_el = np.zeros_like(cl_lens)
    else:
        assert (
            n_el[0] == n_el[1] == 0.0
        ), "First and second elements must be 0 because it is recognized as monopole and dipole"
        n_el = n_el[2 : lmax + 1]

    def wrapped_likelihood(r):
        return _get_likelihood(r, ell, cl_tens, cl_lens, cl_syst, n_el, fsky)

    if method == "exact":
        delta_r = _maximize_likelihood(ell, cl_tens, cl_lens, cl_syst, n_el, fsky, rmax)
    elif method == "minuit":
        m = Minuit(wrapped_likelihood, r0)
        m.limits = (rmin, rmax)
        m.errordef = Minuit.LIKELIHOOD
        m.tol = tol
        m.migrad()
        delta_r = m.values[0]  # delta_r value
    else:
        raise ValueError(f"unknown method {method!r}; use 'exact' or 'minuit'")
    # Calculate likelihood function one last time in the range delta_r*1e-2 < r < delta_r*3
    # Note that delta_r has already been estimated, this likelihood is just used for display.
    # A zero bias uses rmin as the scale, or 1e-10 (the default rmin) if rmin is not positive.
    scale = delta_r if delta_r > 0 else (rmin if rmin > 0 else 1e-10)
    r_grid_display = np.linspace(scale * 1e-2, scale * 3.0, rresol)
    likelihood = np.zeros(rresol)
    for i, r in enumerate(r_grid_display):
        likelihood[i] = wrapped_likelihood(r)
    return {"delta_r": delta_r, "grid_r": r_grid_display, "likelihood": likelihood}
