import litebird_sim as lbs
import numpy as np
from multiprocessing import Pool
import healpy as hp
from tqdm import tqdm
import os
from .main import ScanFields, DB_ROOT_PATH

GREEN = '\033[92m'
RESET = '\033[0m'

class Configlation:
    def __init__(self, imo, channel):
        self.imo = imo
        self.channel = channel
        self.imo_version = "v2"
        self.nside = 128
        self.mdim = 3
        self.parallel = True
        self.xlink_threshold = 0.7
        self.only_iqu = True

class Systematics:
    def __init__(self):
        self.sigma_gain_t = None
        self.sigma_gain_b = None
        self.sigma_rho_t = None
        self.sigma_rho_b = None
        self.sigma_chi_t = None
        self.sigma_chi_b = None
        self.syst_seed = None
        self.noise_seed = None

def process_gain(args):
    i, filename, dirpath, gain_t, gain_b, I, P, mdim, only_iqu = args
    sf = ScanFields.load_det(filename, dirpath)
    diff_gain_signal = ScanFields.diff_gain_field(gain_t[i], gain_b[i], I, P)
    output = sf.map_make(diff_gain_signal, mdim, only_iqu)
    result = {
        "hitmap": sf.hitmap,
        "map": output,
        "xlink2": np.abs(sf.get_xlink(2)),
        }
    return result

def process_pointing(args):
    i,filename,dirpath,rho_t,rho_b,chi_t,chi_b,P,eth_I,eth_P,o_eth_P, mdim, only_iqu = args
    sf = ScanFields.load_det(filename, dirpath)
    diff_signal = ScanFields.diff_pointing_field(rho_t[i],rho_b[i],chi_t[i],chi_b[i],P,eth_I,eth_P,o_eth_P)
    output = sf.map_make(diff_signal, mdim, only_iqu)
    result = {
        "hitmap": sf.hitmap,
        "map": output,
        "xlink2": np.abs(sf.get_xlink(2)),
        }
    return result

def sim_diff_gain_per_ch(
    config: Configlation,
    syst: Systematics,
    mbsparams: lbs.MbsParameters,
    ):

    npix = hp.nside2npix(config.nside)
    telescope = config.channel[0]+"FT"
    sim = lbs.Simulation(random_seed=None)
    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            config.imo,
            f"/releases/{config.imo_version}/satellite/{telescope}/instrument_info",
        )
    )
    ch_info = lbs.FreqChannelInfo.from_imo(url="/releases/"+config.imo_version+"/satellite/"+telescope+"/"+config.channel+"/channel_info", imo=config.imo)
    mbs = lbs.Mbs(
        simulation=sim,
        parameters=mbsparams,
        channel_list=ch_info
    )
    fiducial_map = mbs.run_all()[0][config.channel]
    cmb = mbs.generate_cmb()[0]
    fg = mbs.generate_fg()[0]
    input_maps = {
        "cmb": cmb,
        "fg": fg,
    }
    I = fiducial_map[0]
    P = fiducial_map[1] + 1j*fiducial_map[2]
    dirpath = os.path.join(DB_ROOT_PATH, config.channel)
    filenames = os.listdir(dirpath)
    filenames = [os.path.splitext(filename)[0] for filename in filenames]

    assert syst.sigma_gain_t is not None
    assert syst.sigma_gain_b is not None
    if syst.syst_seed is not None:
        np.random.seed(syst.syst_seed)
        gain_t = np.random.normal(loc=0.0, scale=syst.sigma_gain_t, size=len(filenames))
        gain_b = np.random.normal(loc=0.0, scale=syst.sigma_gain_b, size=len(filenames))
    else:
        gain_t = np.random.normal(loc=0.0, scale=syst.sigma_gain_t, size=len(filenames))
        gain_b = np.random.normal(loc=0.0, scale=syst.sigma_gain_b, size=len(filenames))

    _sf = ScanFields.load_det(filenames[0], base_path=dirpath)
    _sf.generate_noise_pdf(config.imo, scale=2.0)
    observed_map = np.zeros([config.mdim, npix])
    noise_map = np.zeros([config.mdim, npix])
    sky_weight = np.zeros(npix)
    if config.parallel == True:
        file_args = [(i, filename, dirpath, gain_t, gain_b, I, P, config.mdim, config.only_iqu) for i, filename in enumerate(filenames)]
        with Pool() as pool:
            for i, result in enumerate(tqdm(pool.imap(process_gain, file_args),
                                            total=len(file_args),
                                            desc=f"{GREEN}Processing {config.channel}{RESET}",
                                            bar_format='{l_bar}{bar:10}{r_bar}',
                                            colour='green')):
                observed_map += result["map"]
                noise_map += _sf.generate_noise(config.mdim, seed=syst.noise_seed)
                sky_weight[result["xlink2"] < config.xlink_threshold] += 1.0
    else:
        for i, filename in enumerate(tqdm(filenames,
                                          desc=f"{GREEN}Processing {config.channel}{RESET}",
                                          bar_format='{l_bar}{bar:10}{r_bar}',
                                          colour='green')):
            sf = ScanFields.load_det(filename, base_path=dirpath)
            diff_gain_signal = ScanFields.diff_gain_field(gain_t[i], gain_b[i], I, P)
            output = sf.map_make(diff_gain_signal, config.mdim, config.only_iqu)
            observed_map += output
            noise_map += _sf.generate_noise(config.mdim, seed=syst.noise_seed)
            xlink2 = np.abs(sf.get_xlink(2))
            sky_weight[xlink2 < config.xlink_threshold] += 1.0
    observed_map = np.array(observed_map)/sky_weight
    noise_map = np.array(noise_map)/sky_weight
    return observed_map, noise_map, input_maps

def sim_diff_pointing_per_ch(
    config: Configlation,
    syst: Systematics,
    mbsparams: lbs.MbsParameters,
    ):

    npix = hp.nside2npix(config.nside)
    telescope = config.channel[0]+"FT"
    sim = lbs.Simulation(random_seed=None)
    sim.set_instrument(
        lbs.InstrumentInfo.from_imo(
            config.imo,
            f"/releases/{config.imo_version}/satellite/{telescope}/instrument_info",
        )
    )
    ch_info = lbs.FreqChannelInfo.from_imo(url="/releases/"+config.imo_version+"/satellite/"+telescope+"/"+config.channel+"/channel_info", imo=config.imo)
    mbs = lbs.Mbs(
        simulation=sim,
        parameters=mbsparams,
        channel_list=ch_info
    )
    fiducial_map = mbs.run_all()[0][config.channel]
    cmb = mbs.generate_cmb()[0]
    fg = mbs.generate_fg()[0]
    input_maps = {
        "cmb": cmb,
        "fg": fg,
    }
    I = fiducial_map[0]
    P = fiducial_map[1] + 1j*fiducial_map[2]
    dirpath = os.path.join(DB_ROOT_PATH, config.channel)
    filenames = os.listdir(dirpath)
    filenames = [os.path.splitext(filename)[0] for filename in filenames]

    assert syst.sigma_rho_t is not None
    assert syst.sigma_rho_b is not None
    assert syst.sigma_chi_t is not None
    assert syst.sigma_chi_b is not None
    if syst.syst_seed is not None:
        np.random.seed(syst.syst_seed)
        rho_t = np.random.normal(loc=0.0, scale=syst.sigma_rho_t, size=len(filenames))
        rho_b = np.random.normal(loc=0.0, scale=syst.sigma_rho_b, size=len(filenames))
        chi_t = np.random.normal(loc=0.0, scale=syst.sigma_chi_t, size=len(filenames))
        chi_b = np.random.normal(loc=0.0, scale=syst.sigma_chi_b, size=len(filenames))
    else:
        rho_t = np.random.normal(loc=0.0, scale=syst.sigma_rho_t, size=len(filenames))
        rho_b = np.random.normal(loc=0.0, scale=syst.sigma_rho_b, size=len(filenames))
        chi_t = np.random.normal(loc=0.0, scale=syst.sigma_chi_t, size=len(filenames))
        chi_b = np.random.normal(loc=0.0, scale=syst.sigma_chi_b, size=len(filenames))

    dI = hp.alm2map_der1(hp.map2alm(fiducial_map[0]), nside=config.nside)
    dQ = hp.alm2map_der1(hp.map2alm(fiducial_map[1]), nside=config.nside)
    dU = hp.alm2map_der1(hp.map2alm(fiducial_map[2]), nside=config.nside)
    eth_I = dI[2] - 1j*dI[1]
    eth_P   = dQ[2] + dU[1] - 1j*(dQ[1] - dU[2])
    o_eth_P = dQ[2] - dU[1] + 1j*(dQ[1] + dU[2])

    _sf = ScanFields.load_det(filenames[0], base_path=dirpath)
    _sf.generate_noise_pdf(config.imo, scale=2.0) # scale=2.0 i.e. consider differential detection
    noise_map = np.zeros([config.mdim, npix])
    if config.only_iqu == True:
        observed_maps = np.zeros([3, npix])
    else:
        observed_maps = np.zeros([3, npix])

    sky_weight = np.zeros(npix)
    if config.parallel == True:
        file_args = [
            (
            i,
            filename,
            dirpath,
            rho_t,
            rho_b,
            chi_t,
            chi_b,
            P,
            eth_I,
            eth_P,
            o_eth_P,
            config.mdim,
            config.only_iqu,
            ) for i, filename in enumerate(filenames)
            ]
        with Pool() as pool:
            #for i,result in enumerate(tqdm(pool.imap(process_pointing, file_args), total=len(file_args), desc=f"Processing {config.channel}")):
            for i,result in enumerate(tqdm(pool.imap(process_pointing, file_args), total=len(file_args),
                                           desc=f"{GREEN}Processing {config.channel}{RESET}",
                                           bar_format='{l_bar}{bar:10}{r_bar}',
                                           colour='green')):
                observed_maps += result["map"]
                sky_weight[result["xlink2"] < config.xlink_threshold] += 1.0
                noise_map += _sf.generate_noise(config.mdim, seed=syst.noise_seed)
    else:
        #for i, filename in enumerate(tqdm(filenames, desc=f"Processing {config.channel}")):
        for i, filename in enumerate(tqdm(filenames,
                                          desc=f"{GREEN}Processing {config.channel}{RESET}",
                                          bar_format='{l_bar}{bar:10}{r_bar}', colour='green')):
            sf = ScanFields.load_det(filename, base_path=dirpath)
            diff_signal = ScanFields.diff_pointing_field(rho_t[i],rho_b[i],chi_t[i],chi_b[i],P,eth_I,eth_P,o_eth_P)
            output = sf.map_make(diff_signal, config.mdim, config.only_iqu)
            xlink2 = np.abs(sf.get_xlink(2))
            sky_weight[xlink2 < config.xlink_threshold] += 1.0
            observed_maps += output
            noise_map += _sf.generate_noise(config.mdim, seed=syst.noise_seed)
    observed_maps = np.array(observed_maps)/sky_weight
    noise_map = np.array(noise_map)/sky_weight
    return observed_maps, noise_map, input_maps

def sim_noise_per_ch(
    config: Configlation,
    syst: Systematics,
    ):
    sf = ScanFields.load_channel(config.channel)
    sf.generate_noise_pdf(config.imo, scale=2.0) # scale=2.0 i.e. consider differential detection
    noise_map = sf.generate_noise(config.mdim, seed=syst.noise_seed)
    return noise_map
