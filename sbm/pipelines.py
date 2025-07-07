# -*- encoding: utf-8 -*-

import litebird_sim as lbs
import numpy as np
from multiprocessing import Pool
import healpy as hp
from tqdm import tqdm
import os
from typing import List, Union
import fcntl
import random
from astropy import constants as const
from .scan_fields import ScanFields, DB_ROOT_PATH, channel_list
from .signal_fields import SignalFields
import pysm3
import pysm3.units as u


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


def dBRJ_dT(nu):
    return (2 * const.k_B.value * nu * nu * 1e18) / const.c.value / const.c.value


def BlackBody(nu, T):
    x = const.h.value * nu * 1e9 / const.k_B.value / T
    ex = np.exp(x)
    exm1 = ex - 1.0e0
    return (
        2 * const.h.value * nu * nu * nu * 1e27 / const.c.value / const.c.value / exm1
    )


def sed_dust(nu, betad, Td=20.0, nud=545.0):
    if isinstance(Td, float) and isinstance(betad, float):
        sed = ((nu / nud) ** betad) * BlackBody(nu, Td) / BlackBody(nud, Td)
    else:
        sed = (
            ((nu / nud) ** betad[..., np.newaxis])
            * BlackBody(nu, Td[..., np.newaxis])
            / BlackBody(nud, Td[..., np.newaxis])
        )
    return sed


# synch spectral index is -3 +2 due to RJ to power conversion
def sed_synch(nu, betas=-1.0, nus=0.408):
    if isinstance(betas, float):
        sed = (nu / nus) ** betas
    else:
        sed = (nu / nus) ** (betas[..., np.newaxis] + 2)
    return sed


def sed_freefree(nu, freq_ref_I=30.0, pl_index=-2.14):
    # spectral index is -2.14 +2 due to RJ to power conversion
    return (nu / freq_ref_I) ** (pl_index + 2)


def color_correction_dust(nu, nu0, betad, band, Td=20.0):
    return (
        np.trapz(band * sed_dust(nu, betad, Td) / sed_dust(nu0, betad, Td), nu)
        / np.trapz(band * dBodTth(nu), nu)
        * dBodTth(nu0)
    )


# nu^(-2) throughput factor already in band definition
def color_correction_synch(nu, nu0, band, betas=-1.0):
    return (
        np.trapz(band * sed_synch(nu, betas) / sed_synch(nu0, betas), nu)
        / np.trapz(band * dBodTth(nu), nu)
        * dBodTth(nu0)
    )


def color_correction_freefree(nu, nu0, band):
    return (
        np.trapz(band * sed_freefree(nu) / sed_freefree(nu0), nu)
        / np.trapz(band * dBodTth(nu), nu)
        * dBodTth(nu0)
    )


def color_correction_co(nu, band, lines, line_frequency, out_units):
    weights = band * dBRJ_dT(nu) / np.trapz(band * dBRJ_dT(nu), nu)
    weights /= np.trapz(weights, nu)

    out = 0
    for line in lines:
        line_freq = line_frequency[line].to_value(u.GHz)
        # check if the line is in the bandpass
        if line_freq >= nu[0] and line_freq <= nu[-1]:
            # interpolate the value of the bandpass at the freq of the CO line
            weight = np.interp(line_freq, nu, weights)
            convert_to_uK_RJ = (1 * u.K_CMB).to_value(
                u.uK_RJ,
                equivalencies=u.cmb_equivalencies(line_freq * u.GHz),
            )
            # sums over the co lines in the bandpass
            out += convert_to_uK_RJ * weight

    # converts to K_CMB/uK_CMB units
    sed = (out << u.uK_RJ) * pysm3.bandpass_unit_conversion(nu * u.GHz, band, out_units)

    return sed.value


def co_map(nu, lines, line_frequency, nside, template):
    out = np.zeros((3, hp.nside2npix(nside)), dtype=np.float64)
    for line in lines:
        line_freq = line_frequency[line].to_value(u.GHz)
        if line_freq >= nu[0] and line_freq <= nu[-1]:
            print(f"CO line {line_freq} in the band")
            I_map = template[line].copy()

            # if self.include_high_galactic_latitude_clouds:
            #    I_map += self.simulate_high_galactic_latitude_CO(line)

            # if self.has_polarization:
            #    out[1:] += (
            #        self.simulate_polarized_emission(I_map).value
            #        * convert_to_uK_RJ
            #        * weight   )

            out[0] += I_map.value
    return out


def init_ame(
    map_I,
    freq_ref_I,
    emissivity,
    freq_peak,
    freq_ref_peak,
    nside,
    max_nside=None,
    unit_I=None,
    map_dist=None,
):
    I_ref = pysm3.read_map(map_I, nside, unit=unit_I)
    I_ref <<= u.uK_RJ
    freq_ref_I = u.Quantity(freq_ref_I).to(u.GHz)
    try:  # input is a number
        freq_peak = u.Quantity(freq_peak).to(u.GHz)
    except TypeError:  # input is a path
        freq_peak = pysm3.read_map(freq_peak, nside, unit=u.GHz)
    freq_ref_peak = u.Quantity(freq_ref_peak).to(u.GHz)
    freq_peak /= freq_ref_peak
    emissivity = pysm3.models.template.read_txt(emissivity, unpack=True)

    return I_ref, freq_ref_I, freq_peak, emissivity


def sed_ame(nu, band, freq_ref_I, freq_peak, emissivity, nside):
    freqs = pysm3.utils.check_freq_input(nu * u.GHz)
    # tau(nu) * dB_RJ(nu)/dT / np.trapz( tau(nu) * dB_RJ(nu)/dT, nu)
    weights = pysm3.utils.normalize_weights(nu, band)

    if len(freqs) > 1:
        sed = np.zeros((hp.nside2npix(nside), len(freqs)))
        for i, (freq, _weight) in enumerate(zip(freqs, weights)):
            scaled_freq = freq / freq_peak
            scaled_ref_freq = freq_ref_I / freq_peak

            sed[:, i] = (
                (freq_ref_I / freq) ** 2
                * np.interp(scaled_freq, emissivity[0], emissivity[1])
                / np.interp(scaled_ref_freq, emissivity[0], emissivity[1])
            )

            sed_tot = np.trapz(sed * weights, freqs)
    else:
        scaled_freq = freqs / freq_peak
        scaled_ref_freq = freq_ref_I / freq_peak

        sed = np.zeros(hp.nside2npix(nside))
        sed = (
            (freq_ref_I / freqs) ** 2
            * np.interp(scaled_freq, emissivity[0], emissivity[1])
            / np.interp(scaled_ref_freq, emissivity[0], emissivity[1])
        )

        sed_tot = sed * weights
    return sed_tot


def sim_bandpass_mismatch(
    config: Configlation,
    syst: Systematics,
    mbsparams: lbs.MbsParameters,
    detector_list: Union[list, None] = None,
    base_path: Union[str, None] = None,
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
        base_path (`str`): The base path for the HDF5 files containing the cross-link maps

    Returns:
        observed_map (`np.ndarray`): The observed map after the map-making

        returned_input_map (`np.ndarray`): The input map for the simulation
    """
    npix = hp.nside2npix(config.nside)
    telescope = config.channel[0] + "FT"
    sim = lbs.Simulation(base_path=config.lbs_base_path, random_seed=None)

    if config.imo_version != "IMo_vReformationPlan_Option1M":
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
    else:
        sim.set_instrument(
            lbs.InstrumentInfo.from_imo(
                config.imo,
                f"/releases/{config.imo_version}/LMHFT/instrument_info",
            )
        )
        ch_info = lbs.FreqChannelInfo.from_imo(
            url="/releases/"
            + config.imo_version
            + "/LMHFT/"
            + config.channel
            + "/channel_info",
            imo=config.imo,
        )
    
    fg_models = mbsparams.fg_models
    # no bandpass integration to compute fg at nu0
    mbsparams.bandpass_int = False
    mbs = lbs.Mbs(simulation=sim, parameters=mbsparams, channel_list=ch_info)
    map_info = mbs.run_all()[0]
    input_map_nu0 = map_info[config.channel]
    fgs = mbs.generate_fg()[0]
    fg_tmap_list = [fgs[fg][0][0] for fg in fg_models]

    if mbsparams.units == "uK_CMB":
        out_units = u.uK_CMB
    if mbsparams.units == "K_CMB":
        out_units = u.K_CMB

    if "pysm_dust_1" in fg_models:
        mbb_T = pysm3.read_map("pysm_2/dust_temp.fits", nside=config.nside)
        mbb_ind = pysm3.read_map("pysm_2/dust_beta.fits", nside=config.nside)

    if "pysm_synch_1" in fg_models:
        mbb_s = pysm3.read_map("pysm_2/synch_beta.fits", nside=config.nside)

    if "pysm_co_1" in fg_models:
        lines = ["10", "21", "32"]
        line_index = {"10": 0, "21": 1, "32": 2}
        line_frequency = {
            "10": 115.271 * u.GHz,
            "21": 230.538 * u.GHz,
            "32": 345.796 * u.GHz,
        }

        # has_polarization = False
        # include_high_galactic_latitude_clouds = False
        template_nside = 512 if config.nside <= 512 else 2048

        planck_templatemap_filename = (
            f"co/HFI_CompMap_CO-Type1_{template_nside}_R2.00_ring.fits"
        )

        remote_data = pysm3.utils.RemoteData()

        map_in = pysm3.models.template.read_map(
            remote_data.get(planck_templatemap_filename),
            field=[line_index[line] for line in lines],
            unit=u.K_CMB,
            nside=template_nside,
        )

        planck_templatemap = pysm3.models.co_lines.build_lines_dict(
            lines,
            hp.ud_grade(np.array(map_in), nside_out=config.nside) << u.K_CMB,
        )

    if "pysm_ame_1" in fg_models:
        freq_ref_I_1 = "22.8 GHz"
        emissivity_1 = "pysm_2/emissivity.txt"
        freq_peak_1 = "pysm_2/ame_nu_peak_0.fits"
        freq_ref_peak_1 = "30 GHz"

        freq_ref_I_2 = "41 GHz"
        emissivity_2 = "pysm_2/emissivity.txt"
        freq_peak_2 = "33.35 GHz"
        freq_ref_peak_2 = "30 GHz"

        I_ref1, freq_ref_I1, freq_peak1, emissivity1 = init_ame(
            "pysm_2/ame_t_new.fits",
            freq_ref_I_1,
            emissivity_1,
            freq_peak_1,
            freq_ref_peak_1,
            config.nside,
            unit_I="uK_RJ",
        )

        I_ref2, freq_ref_I2, freq_peak2, emissivity2 = init_ame(
            "pysm_2/ame2_t_new.fits",
            freq_ref_I_2,
            emissivity_2,
            freq_peak_2,
            freq_ref_peak_2,
            config.nside,
            unit_I="uK_RJ",
        )

    if not base_path:
        dirpath = os.path.join(DB_ROOT_PATH, config.channel)
    else:
        dirpath = os.path.join(base_path, config.channel)

    # computing the gamma factors from the bandpasses
    if detector_list:
        gamma_T_dict = {}
        gamma_B_dict = {}
        assert len(detector_list) == len(syst.bpm.detectors)
        # now bandpass integration
        mbsparams.bandpass_int = True
        mbs_bp = lbs.Mbs(
            simulation=sim, parameters=mbsparams, detector_list=detector_list
        )
        map_info_bp = mbs_bp.run_all()[0]

        returned_input_map = np.zeros([3, npix])

        pol_map = {}

        for d in detector_list:
            if (
                "pysm_dust_1" in fg_models
                or "pysm_synch_1" in fg_models
                or "pysm_co_1" in fg_models
                or "pysm_ame_1" in fg_models
            ):
                if d.name[-1] == "T":
                    gamma_T_dict[d.name] = np.zeros((len(fg_models), npix))
                if d.name[-1] == "B":
                    gamma_B_dict[d.name] = np.zeros((len(fg_models), npix))

            else:
                if d.name[-1] == "T":
                    gamma_T_dict[d.name] = np.zeros(len(fg_models))
                if d.name[-1] == "B":
                    gamma_B_dict[d.name] = np.zeros(len(fg_models))

            # index of bpm.detectors with same name as in d
            input_maps_d = map_info_bp[d.name]

            returned_input_map += map_info_bp[d.name]
            pol_map[d.name] = input_maps_d[1] + 1.0j * input_maps_d[2]

            # ind = np.where(np.isin(syst.bpm.detectors, d.name))
            for ifg, fg in enumerate(fg_models):
                if fg == "pysm_dust_0":
                    g = color_correction_dust(
                        nu=d.band_freqs_ghz,
                        nu0=d.bandcenter_ghz,
                        betad=1.54,
                        band=d.band_weights,
                    )
                if fg == "pysm_synch_0":
                    g = color_correction_synch(
                        nu=d.band_freqs_ghz, nu0=d.bandcenter_ghz, band=d.band_weights
                    )

                if fg == "pysm_dust_1":
                    g = color_correction_dust(
                        nu=d.band_freqs_ghz,
                        nu0=d.bandcenter_ghz,
                        betad=mbb_ind,
                        band=d.band_weights,
                        Td=mbb_T,
                    )
                if fg == "pysm_synch_1":
                    g = color_correction_synch(
                        nu=d.band_freqs_ghz,
                        nu0=d.bandcenter_ghz,
                        band=d.band_weights,
                        betas=mbb_s,
                    )

                if fg == "pysm_freefree_1":
                    g = color_correction_freefree(
                        nu=d.band_freqs_ghz, nu0=d.bandcenter_ghz, band=d.band_weights
                    )

                if fg == "pysm_co_1":
                    # setting the correct CO map, it would be otherwise 0
                    # this is just the template map, to be multiplied with g
                    # to get the full map
                    # in principle they are the same for each det, but we never know
                    # the freq range between one det and the other could be different
                    # and so they could include different numbers of CO lines
                    # not really expected but still possible
                    co_map_d = co_map(
                        nu=d.band_freqs_ghz,
                        lines=lines,
                        line_frequency=line_frequency,
                        nside=config.nside,
                        template=planck_templatemap,
                    )[0]
                    g = (
                        color_correction_co(
                            nu=d.band_freqs_ghz,
                            band=d.band_weights,
                            lines=lines,
                            line_frequency=line_frequency,
                            out_units=out_units,
                        )
                        * co_map_d
                    )

                    # we fix the I(nu0) map to 1, so that we recover the total CO map in g
                    fg_tmap_list[ifg] = np.ones(npix)

                if fg == "pysm_ame_1":
                    sed_ame1 = sed_ame(
                        d.band_freqs_ghz,
                        d.band_weights,
                        freq_ref_I1.value,
                        freq_peak1.value,
                        emissivity1,
                        config.nside,
                    )
                    sed_ame2 = sed_ame(
                        d.band_freqs_ghz,
                        d.band_weights,
                        freq_ref_I2.value,
                        freq_peak2.value,
                        emissivity2,
                        config.nside,
                    )

                    # the gamma factor is the whole map in this case (sed_ame_1*I_1 + sed_ame_2*I_2)
                    g = (
                        sed_ame1 * I_ref1 + sed_ame2 * I_ref2
                    ) * pysm3.bandpass_unit_conversion(
                        d.band_freqs_ghz * u.GHz, d.band_weights, out_units
                    )

                    # we fix the I(nu0) map to 1, so that we recover the total AME map in g
                    fg_tmap_list[ifg] = np.ones(npix)

                if d.name[-1] == "T" and gamma_T_dict[d.name].shape == (
                    len(fg_models),
                    npix,
                ):
                    if hasattr(g, "__len__"):
                        gamma_T_dict[d.name][ifg, :] = g
                    else:
                        gamma_T_dict[d.name][ifg, :] = g * np.ones(npix)

                if d.name[-1] == "T" and gamma_T_dict[d.name].shape == (
                    len(fg_models),
                ):
                    gamma_T_dict[d.name][ifg] = g

                if d.name[-1] == "B" and gamma_B_dict[d.name].shape == (
                    len(fg_models),
                    npix,
                ):
                    if hasattr(g, "__len__"):
                        gamma_B_dict[d.name][ifg, :] = g
                    else:
                        gamma_B_dict[d.name][ifg, :] = g * np.ones(npix)

                if d.name[-1] == "B" and gamma_B_dict[d.name].shape == (
                    len(fg_models),
                ):
                    gamma_B_dict[d.name][ifg] = g

    # using the values passed to the set_bandpass_mismatch class
    else:
        pol_map = input_map_nu0[1] + 1.0j * input_map_nu0[2]
        gamma_T_dict = syst.bpm.gamma_T_dict
        gamma_B_dict = syst.bpm.gamma_B_dict
        assert len(syst.bpm.detectors) // 2 == len(gamma_T_dict.keys())
        assert len(syst.bpm.detectors) // 2 == len(gamma_B_dict.keys())

    observed_map = np.zeros([3, npix])
    sky_weight = np.zeros(npix)

    if config.parallel is True:
        file_args = []

        for i, idet in enumerate(syst.bpm.detectors):
            if idet[-1] == "T":
                tname = idet
                bname = idet[:-1] + "B"

                if detector_list:
                    # averaging over T and B pol input map
                    pm = (pol_map[tname] + pol_map[bname]) / 2.0
                else:
                    pm = pol_map

                file_args.append(
                    (
                        idet,
                        dirpath,
                        gamma_T_dict[tname],
                        gamma_B_dict[bname],
                        fg_tmap_list,
                        pm,
                        config.mdim,
                        config.only_iqu,
                        config.xlink_threshold,
                    )
                )

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
            if idet[-1] == "T":
                tname = idet
                bname = idet[:-1] + "B"

                sf = ScanFields.load_det(idet, base_path=dirpath)
                sf.xlink_threshold = config.xlink_threshold
                sf.use_hwp = config.use_hwp

                if detector_list:
                    # averaging over T and B pol input map
                    pm = (pol_map[tname] + pol_map[bname]) / 2.0
                else:
                    pm = pol_map

                signal_field = SignalFields.bandpass_mismatch_field(
                    sf,
                    config.mdim,
                    pm,
                    gamma_T_dict[tname],
                    gamma_B_dict[bname],
                    fg_tmap_list,
                )
                output = sf.map_make(signal_field, config.only_iqu)
                observed_map += output
                xlink2 = np.abs(sf.get_xlink(2, 0))
                sky_weight[xlink2 < config.xlink_threshold] += 1.0
    observed_map = np.array(observed_map) / sky_weight

    if not detector_list:
        returned_input_map = input_map_nu0
    else:
        returned_input_map /= len(detector_list)

    return observed_map, returned_input_map


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
