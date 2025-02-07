# -*- encoding: utf-8 -*-

import litebird_sim as lbs
import numpy as np
from multiprocessing import Pool
import healpy as hp
from tqdm import tqdm
import os
from typing import Any, Dict, List, Union
import fcntl
import random
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo
from .scan_fields import ScanFields, DB_ROOT_PATH, channel_list
from .signal_fields import SignalFields

GREEN = "\033[92m"
RESET = "\033[0m"


class Configlation:
    """Configuration class for the simulation.

    Args:
        imo (`str`): imo instance given by the litebird_sim
        channel (`str`): The name of the channel

    Attributes:
        lbs_base_path (`str`): The base path of the litebird_sim
        imo_version (`str`): The version of the imo
        nside (`int`): The nside of the healpix map
        mdim (`int`): The dimension to perform the map-making
        parallel (`bool`): If `True`, the simulation is performed in thread parallel
        xlink_threshold (`float`): The threshold of the cross-linking.
            The pixel with the value less than this threshold is ignored when
            the map-making is performed.
        only_iqu (`bool`): If `True`, only P, Q, and U are returned after the map-making
    """

    def __init__(self, imo, channel):
        self.imo = imo
        self.channel = channel
        self.lbs_base_path = None
        self.imo_version = "v2"
        self.nside = 128
        self.mdim = 3
        self.parallel = True
        self.xlink_threshold = 0.7
        self.only_iqu = True
        self.use_hwp = None
        self.spin_n_basis = []
        self.spin_m_basis = []


class BandpassParams:
    def __init__(
        self,
        detectors: List[str],
        gamma_T_list: Union[list, None] = None,
        gamma_B_list: Union[list, None] = None,
    ):
        self.detectors = detectors
        self.gamma_T_list = gamma_T_list
        self.gamma_B_list = gamma_B_list


class Systematics:
    """Systematics class for the simulation

    Attributes:
        sigma_gain_T (`float`): The standard deviation of the gain for the top detectors
        sigma_gain_B (`float`): The standard deviation of the gain for the bottom detectors
        sigma_rho_T (`float`): The standard deviation of the pointing for the top detectors
        sigma_rho_B (`float`): The standard deviation of the pointing for the bottom detectors
        sigma_chi_T (`float`): The standard deviation of the polarization angle for the top detectors
        sigma_chi_B (`float`): The standard deviation of the polarization angle for the bottom detectors
        syst_seed (`int`): The seed for the random number generator for the systematics
        noise_seed (`int`): The seed for the random number generator for the noise
    """

    def __init__(self):
        self.sigma_gain_T = None
        self.sigma_gain_B = None
        self.sigma_rho_T = None
        self.sigma_rho_B = None
        self.sigma_chi_T = None
        self.sigma_chi_B = None
        self.syst_seed = None
        self.noise_seed = None
        self.start_seed = None
        self.end_seed = None
        self.bpm = None

    def set_bandpass_mismatch(
        self,
        detectors: List[str],
        gamma_T_list: Union[list, None] = None,
        gamma_B_list: Union[list, None] = None,
    ):
        self.bpm = BandpassParams(detectors, gamma_T_list, gamma_B_list)


def process_gain(args):
    (
        i,
        filename,
        dirpath,
        gain_T,
        gain_B,
        temp_map,
        pol_map,
        mdim,
        only_iqu,
        xlink_threshold,
    ) = args
    sf = ScanFields.load_det(filename, dirpath)
    sf.xlink_threshold = xlink_threshold
    sf.use_hwp = False
    diff_gain_signal = SignalFields.diff_gain_field(
        sf, mdim, gain_T[i], gain_B[i], temp_map, pol_map
    )
    output = sf.map_make(diff_gain_signal, only_iqu)
    result = {
        "hitmap": sf.hitmap,
        "map": output,
        "xlink2": np.abs(sf.get_xlink(2, 0)),
    }
    return result


def process_pointing(args):
    (
        i,
        filename,
        dirpath,
        rho_T,
        rho_B,
        chi_T,
        chi_B,
        pol_map,
        eth_I,
        eth_P,
        o_eth_P,
        mdim,
        only_iqu,
        xlink_threshold,
    ) = args
    sf = ScanFields.load_det(filename, dirpath)
    sf.xlink_threshold = xlink_threshold
    sf.use_hwp = False
    diff_signal = SignalFields.diff_pointing_field(
        sf, mdim, rho_T[i], rho_B[i], chi_T[i], chi_B[i], pol_map, eth_I, eth_P, o_eth_P
    )
    output = sf.map_make(diff_signal, only_iqu)
    result = {
        "hitmap": sf.hitmap,
        "map": output,
        "xlink2": np.abs(sf.get_xlink(2, 0)),
    }
    return result


def generate_maps(mbs, config, lock=True):
    """Generate the maps with the lock file

    Args:
        mbs (`lbs.MbsParameters`): The litebird_sim object
        config (:class:`.Configlation`): The configuration class
        lock (`bool`): If `True`, the lock file is used
    """
    if lock:
        lockfile = "/tmp/sbm_lockfile"
        with open(lockfile, "a") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another instance is running")
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                map_info = mbs.run_all()
                fiducial_map = map_info[0][config.channel]
                input_maps = {
                    "cmb": mbs.generate_cmb()[0],
                    "fg": mbs.generate_fg()[0],
                }
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                os.remove(lockfile)
    else:
        map_info = mbs.run_all()
        fiducial_map = map_info[0][config.channel]
        input_maps = {
            "cmb": mbs.generate_cmb()[0],
            "fg": mbs.generate_fg()[0],
        }
    return fiducial_map, input_maps


def sim_diff_gain_per_ch(
    config: Configlation,
    syst: Systematics,
    mbsparams: lbs.MbsParameters,
):
    """Simulate the differential gain systematics for each channel
    The map-making is performed for each detector in the channel

    Args:
        config (:class:`.Configlation`): The configuration class

        syst (:class:`.Systematics`): The systematics class

        mbsparams (`lbs.MbsParameters`): The parameters for the litebird_sim

    Returns:
        observed_map (`np.ndarray`): The observed map after the map-making

        input_maps (`dict`): The input maps for the simulation
    """
    npix = hp.nside2npix(config.nside)
    telescope = config.channel[0] + "FT"
    sim = lbs.Simulation(base_path=config.lbs_base_path, random_seed=None)
    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            config.imo,
            f"/releases/{config.imo_version}/satellite/{telescope}/instrument_info",
        )
    )
    ch_info = lbs.FreqChannelInfo.from_imo(
        url="/releases/"
        + config.imo_version
        + "/satellite/"
        + telescope
        + "/"
        + config.channel
        + "/channel_info",
        imo=config.imo,
    )
    mbs = lbs.Mbs(simulation=sim, parameters=mbsparams, channel_list=ch_info)
    fiducial_map, input_maps = generate_maps(mbs, config, lock=False)
    temp_map = fiducial_map[0]
    pol_map = fiducial_map[1] + 1j * fiducial_map[2]
    dirpath = os.path.join(DB_ROOT_PATH, config.channel)
    filenames = os.listdir(dirpath)
    filenames = [os.path.splitext(filename)[0] for filename in filenames]

    assert syst.sigma_gain_T is not None
    assert syst.sigma_gain_B is not None
    if syst.syst_seed is not None:
        channel_id = np.where(np.array(channel_list) == config.channel)[0][0]
        np.random.seed(syst.syst_seed + channel_id)
        gain_T = np.random.normal(loc=0.0, scale=syst.sigma_gain_T, size=len(filenames))
        gain_B = np.random.normal(loc=0.0, scale=syst.sigma_gain_B, size=len(filenames))
    else:
        gain_T = np.random.normal(loc=0.0, scale=syst.sigma_gain_T, size=len(filenames))
        gain_B = np.random.normal(loc=0.0, scale=syst.sigma_gain_B, size=len(filenames))

    observed_map = np.zeros([3, npix])
    sky_weight = np.zeros(npix)
    if config.parallel is True:
        file_args = [
            (
                i,
                filename,
                dirpath,
                gain_T,
                gain_B,
                temp_map,
                pol_map,
                config.mdim,
                config.only_iqu,
                config.xlink_threshold,
            )
            for i, filename in enumerate(filenames)
        ]
        with Pool() as pool:
            for i, result in enumerate(
                tqdm(
                    pool.imap(process_gain, file_args),
                    total=len(file_args),
                    desc=f"{GREEN}Processing {config.channel}{RESET}",
                    bar_format="{l_bar}{bar:10}{r_bar}",
                    colour="green",
                )
            ):
                observed_map += result["map"]
                sky_weight[result["xlink2"] < config.xlink_threshold] += 1.0
    else:
        for i, filename in enumerate(
            tqdm(
                filenames,
                desc=f"{GREEN}Processing {config.channel}{RESET}",
                bar_format="{l_bar}{bar:10}{r_bar}",
                colour="green",
            )
        ):
            sf = ScanFields.load_det(filename, base_path=dirpath)
            sf.xlink_threshold = config.xlink_threshold
            sf.use_hwp = config.use_hwp
            diff_gain_signal = SignalFields.diff_gain_field(
                sf, config.mdim, gain_T[i], gain_B[i], temp_map, pol_map
            )
            output = sf.map_make(diff_gain_signal, config.only_iqu)
            observed_map += output
            xlink2 = np.abs(sf.get_xlink(2, 0))
            sky_weight[xlink2 < config.xlink_threshold] += 1.0
    observed_map = np.array(observed_map) / sky_weight
    return observed_map, input_maps


def sim_diff_pointing_per_ch(
    config: Configlation,
    syst: Systematics,
    mbsparams: lbs.MbsParameters,
):
    """Simulate the differential pointing systematics for each channel
    The map-making is performed for each detector in the channel

    Args:
        config (:class:`.Configlation`): The configuration class

        syst (:class:`.Systematics`): The systematics class

        mbsparams (`lbs.MbsParameters`): The parameters for the litebird_sim

    Returns:
        observed_map (`np.ndarray`): The observed map after the map-making

        input_maps (`dict`): The input maps for the simulation
    """
    npix = hp.nside2npix(config.nside)
    telescope = config.channel[0] + "FT"
    sim = lbs.Simulation(base_path=config.lbs_base_path, random_seed=None)
    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            config.imo,
            f"/releases/{config.imo_version}/satellite/{telescope}/instrument_info",
        )
    )
    ch_info = lbs.FreqChannelInfo.from_imo(
        url="/releases/"
        + config.imo_version
        + "/satellite/"
        + telescope
        + "/"
        + config.channel
        + "/channel_info",
        imo=config.imo,
    )
    mbs = lbs.Mbs(simulation=sim, parameters=mbsparams, channel_list=ch_info)
    fiducial_map, input_maps = generate_maps(mbs, config, lock=False)
    pol_map = fiducial_map[1] + 1j * fiducial_map[2]
    dirpath = os.path.join(DB_ROOT_PATH, config.channel)
    filenames = os.listdir(dirpath)
    filenames = [os.path.splitext(filename)[0] for filename in filenames]

    assert syst.sigma_rho_T is not None
    assert syst.sigma_rho_B is not None
    assert syst.sigma_chi_T is not None
    assert syst.sigma_chi_B is not None

    if syst.syst_seed is not None:
        channel_id = np.where(np.array(channel_list) == config.channel)[0][0]
        np.random.seed(syst.syst_seed + channel_id)
        rho_T = np.random.normal(loc=0.0, scale=syst.sigma_rho_T, size=len(filenames))
        rho_B = np.random.normal(loc=0.0, scale=syst.sigma_rho_B, size=len(filenames))
        chi_T = np.random.normal(loc=0.0, scale=syst.sigma_chi_T, size=len(filenames))
        chi_B = np.random.normal(loc=0.0, scale=syst.sigma_chi_B, size=len(filenames))
    else:
        rho_T = np.random.normal(loc=0.0, scale=syst.sigma_rho_T, size=len(filenames))
        rho_B = np.random.normal(loc=0.0, scale=syst.sigma_rho_B, size=len(filenames))
        chi_T = np.random.normal(loc=0.0, scale=syst.sigma_chi_T, size=len(filenames))
        chi_B = np.random.normal(loc=0.0, scale=syst.sigma_chi_B, size=len(filenames))

    dI = hp.alm2map_der1(hp.map2alm(fiducial_map[0]), nside=config.nside)
    dQ = hp.alm2map_der1(hp.map2alm(fiducial_map[1]), nside=config.nside)
    dU = hp.alm2map_der1(hp.map2alm(fiducial_map[2]), nside=config.nside)
    eth_I = dI[2] - 1j * dI[1]
    eth_P = dQ[2] + dU[1] - 1j * (dQ[1] - dU[2])
    o_eth_P = dQ[2] - dU[1] + 1j * (dQ[1] + dU[2])

    observed_maps = np.zeros([3, npix])
    sky_weight = np.zeros(npix)
    if config.parallel is True:
        file_args = [
            (
                i,
                filename,
                dirpath,
                rho_T,
                rho_B,
                chi_T,
                chi_B,
                pol_map,
                eth_I,
                eth_P,
                o_eth_P,
                config.mdim,
                config.only_iqu,
                config.xlink_threshold,
            )
            for i, filename in enumerate(filenames)
        ]
        with Pool() as pool:
            for i, result in enumerate(
                tqdm(
                    pool.imap(process_pointing, file_args),
                    total=len(file_args),
                    desc=f"{GREEN}Processing {config.channel}{RESET}",
                    bar_format="{l_bar}{bar:10}{r_bar}",
                    colour="green",
                )
            ):
                observed_maps += result["map"]
                sky_weight[result["xlink2"] < config.xlink_threshold] += 1.0
    else:
        for i, filename in enumerate(
            tqdm(
                filenames,
                desc=f"{GREEN}Processing {config.channel}{RESET}",
                bar_format="{l_bar}{bar:10}{r_bar}",
                colour="green",
            )
        ):
            sf = ScanFields.load_det(filename, base_path=dirpath)
            sf.xlink_threshold = config.xlink_threshold
            sf.use_hwp = config.use_hwp
            diff_signal = SignalFields.diff_pointing_field(
                sf,
                config.mdim,
                rho_T[i],
                rho_B[i],
                chi_T[i],
                chi_B[i],
                pol_map,
                eth_I,
                eth_P,
                o_eth_P,
            )
            output = sf.map_make(diff_signal, config.only_iqu)
            xlink2 = np.abs(sf.get_xlink(2, 0))
            sky_weight[xlink2 < config.xlink_threshold] += 1.0
            observed_maps += output
    observed_maps = np.array(observed_maps) / sky_weight
    return observed_maps, input_maps


def process_bpm(args):
    (
        idet,
        dirpath,
        gamma_T_list,
        gamma_B_list,
        component_tmaps,
        pol_map,
        mdim,
        only_iqu,
        xlink_threshold,
    ) = args
    sf = ScanFields.load_det(idet, dirpath)
    sf.xlink_threshold = xlink_threshold
    sf.use_hwp = False

    signal_field = SignalFields.bandpass_mismatch_field(
        sf,
        mdim,
        pol_map,
        gamma_T_list,
        gamma_B_list,
        component_tmaps,
    )
    output = sf.map_make(signal_field, only_iqu)
    result = {
        "hitmap": sf.hitmap,
        "map": output,
        "xlink2": np.abs(sf.get_xlink(2, 0)),
    }
    return result


def dBodTth(nu):
    return lbs.hwp_sys.hwp_sys._dBodTth(nu)

def BlackBody(nu, T):
    x = const.h.value * nu * 1e9 / const.k_B.value / T
    ex = np.exp(x)
    exm1 = ex - 1.0e0
    return (
        2
        * const.h.value
        * nu
        * nu
        * nu
        * 1e27
        / const.c.value
        / const.c.value
        / exm1
    )


def sed_dust(nu,betad, nud = 545, Td = 20):
    return ((nu/nud)**betad)*BlackBody(nu,Td)/BlackBody(nud,Td)

# synch spectral index is -3 +2 due to RJ to power conversion
def sed_synch(nu,nus = 0.408,betas = -1):  
    return (nu/nus)**betas

def color_correction_dust(nu, nu0, betad, band):
    return (np.trapz(band*sed_dust(nu,betad)/sed_dust(nu0,betad),nu)/
             np.trapz(band*dBodTth(nu),nu) * dBodTth(nu0)
           )
# nu^(-2) throughput factor already in band definition
def color_correction_synch(nu, nu0, band):
    return (np.trapz(band*sed_synch(nu)/sed_synch(nu0),nu)/
             np.trapz(band*dBodTth(nu),nu) * dBodTth(nu0)
           )

def sim_bandpass_mismatch(
    config: Configlation,
    syst: Systematics,
    mbsparams: lbs.MbsParameters,
    detector_list: Union[list, None] = None,
    base_path: Union[str, None] = None
):
    """Simulate the bandpass mismatch systematics.

    Args:
        config (:class:`.Configlation`): The configuration class

        syst (:class:`.Systematics`): The systematics class

        mbsparams (`lbs.MbsParameters`): The parameters for the litebird_sim

        detector_list (list of `lbs.DetectorInfo`): List of DetectorInfo for each detector considered,
                                                    generated with `lbs.DetectorInfo`. In each DetectorInfo, the 
                                                    `band_weights` are the bandpass for that detector. These
                                                    bandpasses are used to compute the foreground input maps.
                                                    If None, the foreground maps are computed at the central freq
                                                    of the channel.

    Returns:
        observed_map (`np.ndarray`): The observed map after the map-making

        input_maps (`dict`): The input maps for the simulation
    """
    npix = hp.nside2npix(config.nside)
    telescope = config.channel[0] + "FT"
    sim = lbs.Simulation(base_path=config.lbs_base_path, random_seed=None)
    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            config.imo,
            f"/releases/{config.imo_version}/satellite/{telescope}/instrument_info",
        )
    )
    ch_info = lbs.FreqChannelInfo.from_imo(
        url="/releases/"
        + config.imo_version
        + "/satellite/"
        + telescope
        + "/"
        + config.channel
        + "/channel_info",
        imo=config.imo,
    )
    fg_models = mbsparams.fg_models
    mbs = lbs.Mbs(simulation=sim, parameters=mbsparams, channel_list=ch_info)
    map_info = mbs.run_all()[0]
    fgs = mbs.generate_fg()[0]
    fg_tmap_list = [fgs[fg][0][0] for fg in fg_models]
    if not base_path:
        dirpath = os.path.join(DB_ROOT_PATH, config.channel)
    else: 
        dirpath = os.path.join(base_path, config.channel)

    #computing the gamma factors from the bandpasses 
    if detector_list:
        assert(len(detector_list) == len(syst.bpm.detectors))
        mbs_bp = lbs.Mbs(simulation=sim, parameters=mbsparams, detector_list=detector_list)
        map_info_bp = mbs_bp.run_all()[0]
        gamma_T_list = np.zeros((len(syst.bpm.detectors), len(fg_models)))
        gamma_B_list = np.zeros((len(syst.bpm.detectors), len(fg_models)))
        input_maps = np.zeros([3, npix])

        pol_map = {}
        for d in detector_list:
            # index of bpm.detectors with same name as in d
            input_maps_d = map_info_bp[d.name]
            pol_map[d.name] = input_maps_d[1] + 1j * input_maps_d[2]
            ind = np.where(np.isin(syst.bpm.detectors,d.name))
            for ifg,fg in enumerate(fg_models):
                if fg == "pysm_dust_0":
                    g = color_correction_dust(nu = d.band_freqs_ghz, 
                                          nu0 = d.bandcenter_ghz, 
                                          betad = 1.54, band = d.band_weights)
                if fg == "pysm_synch_0":
                    g = color_correction_synch(nu = d.band_freqs_ghz, 
                                          nu0 = d.bandcenter_ghz,
                                          band = d.band_weights)
    
                if d.name[-1] == "T":
                    gamma_T_list[ind,ifg] = g
                if d.name[-1] == "B":
                    gamma_B_list[ind,ifg] = g
    
    #using the values passed to the set_bandpass_mismatch class
    else:
        input_maps = map_info[config.channel]
        pol_map = input_maps[1] + 1j * input_maps[2]
        gamma_T_list = syst.bpm.gamma_T_list
        gamma_B_list = syst.bpm.gamma_B_list
        assert len(syst.bpm.detectors) == len(gamma_T_list)
        assert len(syst.bpm.detectors) == len(gamma_B_list)

    observed_map = np.zeros([3, npix])
    sky_weight = np.zeros(npix)
    if config.parallel is True:
        for i, idet in enumerate(syst.bpm.detectors):
            if detector_list:
                pm = pol_map[idet]
                input_maps += map_info_bp[idet]
            else:
                pm = pol_map
            file_args = [
            (
                idet,
                dirpath,
                gamma_T_list[i],
                gamma_B_list[i],
                fg_tmap_list,
                pm,
                config.mdim,
                config.only_iqu,
                config.xlink_threshold,
            )]

        with Pool() as pool:
            for i, result in enumerate(
                tqdm(
                    pool.imap(process_bpm, file_args),
                    total=len(file_args),
                    desc=f"{GREEN}Processing {config.channel}{RESET}",
                    bar_format="{l_bar}{bar:10}{r_bar}",
                    colour="green",
                )
            ):
                observed_map += result["map"]
                sky_weight[result["xlink2"] < config.xlink_threshold] += 1.0
    else:
        for i, idet in enumerate(
            tqdm(
                syst.bpm.detectors,
                desc=f"{GREEN}Processing {config.channel}{RESET}",
                bar_format="{l_bar}{bar:10}{r_bar}",
                colour="green",
            )
        ):
            sf = ScanFields.load_det(idet, base_path=dirpath)
            sf.xlink_threshold = config.xlink_threshold
            sf.use_hwp = config.use_hwp

            if detector_list:
                pm = pol_map[idet]
                input_maps += map_info_bp[idet]
            else:
                pm = pol_map

            signal_field = SignalFields.bandpass_mismatch_field(
                sf, config.mdim, pm, gamma_T_list[i], gamma_B_list[i], fg_tmap_list
            )
            output = sf.map_make(signal_field, config.only_iqu)
            observed_map += output
            xlink2 = np.abs(sf.get_xlink(2, 0))
            sky_weight[xlink2 < config.xlink_threshold] += 1.0
    observed_map = np.array(observed_map) / sky_weight
    if detector_list:
        input_maps = np.array(input_maps) / sky_weight

    return observed_map, input_maps

def generate_noise_seeds(config: Configlation, syst: Systematics, num_of_dets: int):
    channel_id = np.where(np.array(channel_list) == config.channel)[0][0]
    if syst.noise_seed is not None:
        _max = int(1e5)
        seed_range = list(range(0, int(_max)))
        random.seed(channel_id + syst.noise_seed)
        random.seed(random.randint(0, _max))
        noise_seeds = random.choices(seed_range, k=num_of_dets)
    else:
        noise_seeds = [None] * num_of_dets
    return noise_seeds


def process_noise(args):
    (
        i,
        filename,
        dirpath,
        spin_n_basis,
        spin_m_basis,
        only_iqu,
        xlink_threshold,
        imo,
        use_hwp,
        noise_seed_i,
    ) = args
    sf = ScanFields.load_det(filename, dirpath)
    sf.xlink_threshold = xlink_threshold
    sf.use_hwp = use_hwp
    sf.generate_noise_pdf(imo, scale=2.0)
    output = sf.generate_noise(spin_n_basis, spin_m_basis, seed=noise_seed_i)
    result = {
        "hitmap": sf.hitmap,
        "map": output,
        "xlink2": np.abs(sf.get_xlink(2, 0)),
    }
    return result


def sim_noise_per_ch(
    config: Configlation,
    syst: Systematics,
):
    """Simulate the noise for each channel

    Args:
        config (:class:`.Configlation`): The configuration class

        syst (:class:`.Systematics`): The systematics class
    """
    assert config.use_hwp is not None

    npix = hp.nside2npix(config.nside)
    dirpath = os.path.join(DB_ROOT_PATH, config.channel)
    filenames = os.listdir(dirpath)
    filenames = [os.path.splitext(filename)[0] for filename in filenames]
    noise_map = np.zeros([3, npix])
    sky_weight = np.zeros(npix)

    noise_seeds = generate_noise_seeds(config, syst, len(filenames))
    if config.parallel is True:
        file_args = [
            (
                i,
                filename,
                dirpath,
                config.spin_n_basis,
                config.spin_m_basis,
                config.only_iqu,
                config.xlink_threshold,
                config.imo,
                config.use_hwp,
                noise_seeds[i],
            )
            for i, filename in enumerate(filenames)
        ]
        with Pool() as pool:
            for i, result in enumerate(
                tqdm(
                    pool.imap(process_noise, file_args),
                    total=len(file_args),
                    desc=f"{GREEN}Processing {config.channel}{RESET}",
                    bar_format="{l_bar}{bar:10}{r_bar}",
                    colour="green",
                )
            ):
                noise_map += result["map"]
                sky_weight[result["xlink2"] < config.xlink_threshold] += 1.0
    else:
        for i, filename in enumerate(
            tqdm(
                filenames,
                desc=f"{GREEN}Processing {config.channel}{RESET}",
                bar_format="{l_bar}{bar:10}{r_bar}",
                colour="green",
            )
        ):
            sf = ScanFields.load_det(filename, base_path=dirpath)
            sf.generate_noise_pdf(config.imo, scale=2.0)
            sf.use_hwp = config.use_hwp
            noise_map += sf.generate_noise(
                config.spin_n_basis, config.spin_m_basis, seed=noise_seeds[i]
            )
            xlink2 = np.abs(sf.get_xlink(2, 0))
            sky_weight[xlink2 < config.xlink_threshold] += 1.0
    noise_map /= sky_weight
    return noise_map
