import h5py
import numpy as np
import healpy as hp
import os
import copy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
import litebird_sim as lbs
from litebird_sim import Imo
from pathlib import Path
import toml

CONFIG_PATH = Path.home() / ".config" / "sbm_dataset"
CONFIG_FILE_PATH = CONFIG_PATH / "sbm_dataset.toml"

def extract_location_from_toml(file_path):
    with open(file_path, 'r') as file:
        data = toml.load(file)
        loc = data['repositories'][0]['location']
    return loc


if not CONFIG_FILE_PATH.exists():
    DB_ROOT_PATH = None
else:
    DB_ROOT_PATH = extract_location_from_toml(CONFIG_FILE_PATH)

channel_list = [
    'L1-040','L2-050','L1-060','L3-068','L2-068','L4-078','L1-078','L3-089','L2-089','L4-100','L3-119','L4-140',
    'M1-100','M2-119','M1-140','M2-166','M1-195',
    'H1-195','H2-235','H1-280','H2-337','H3-402'
]

fwhms = [70.5,58.5,51.1,41.6,47.1,36.9,43.8,33.0,41.5,30.2,26.3,23.7,37.8,33.6,30.8,28.9,28.0,28.6,24.7,22.5,20.9,17.9]

class Field:
    """ Class to store the field data of detectors """
    def __init__(self, field: np.ndarray, spin_n: int, spin_m: int):
        """ Initialize the class with field and spin data

        Args:
            field (np.ndarray): field data of the detector

            spin_n (int): spin number of the crossing angle

            spin_m (int): spin number of the HWP angle
        """
        if all(isinstance(x, float) for x in field):
            self.field = field + 1j*np.zeros(len(field))
        else:
            self.field = field
        self.spin_n = spin_n
        self.spin_m = spin_m

    def conj(self):
        """ Get the complex conjugate of the field """
        return Field(self.field.conj(), -self.spin_n, -self.spin_m)

    def __add__(self, other):
        """ Add the field of two detectors

        Args:
            other (Field): field of the other detector

        Returns:
            result (Field): sum of the fields of the two detectors
        """
        if not isinstance(other, Field):
            return NotImplemented
        result = copy.deepcopy(self)
        result.field += other.field
        return result

class SignalFields:
    """ Class to store the signal fields data of detectors """
    def __init__(self, *fields: Field):
        """ Initialize the class with field data

        Args:
            fields (Field): field (map) data of the signal
        """
        self.fields = sorted(fields, key=lambda field: (field.spin_n, field.spin_m))
        self.spins_n = np.array([field.spin_n for field in self.fields])
        self.spins_m = np.array([field.spin_m for field in self.fields])

    def get_field(self, spin_n, spin_m):
        """ Get the field of the given spin number

        Args:
            spin_n (int): spin number for which the field is to be obtained

            spin_m (int): spin number for which the field is to be obtained

        Returns:
            field (np.ndarray): field of the detector for the given spin number
        """
        for i, field in enumerate(self.fields):
            if field.spin_n == spin_n and field.spin_m == spin_m:
                return field.field
        return None

    def __add__(self, other):
        """ Add the signal fieldd

        Args:
            other (SignalFields): signal fields

        Returns:
            result (SignalFields): sum of the signal fields
        """
        if not isinstance(other, SignalFields):
            return NotImplemented
        result = copy.deepcopy(self)
        for i in range(len(self.spins_n)): # この足し方でいいのか？
            result.fields[i].field += other.fields[i].field
        return result

class ScanFields:
    """ Class to store the scan fields data of detectors """
    def __init__(self):
        """ Initialize the class with empty data

        ss (dict):  of the scanning strategy parameters
        hitmap (np.ndarray): hitmap of the detector
        h (np.ndarray): cross-link (orientation function) of the detector
        spins_n (np.ndarray): array of spin_n numbers
        spins_m (np.ndarray): array of spin_m number
        compled_fields (np.ndarray): coupled fields between scan fields and signal fields
        use_hwp (bool): whether the observation uses HWP or not
        nside (int): nside of the map
        npix (int): number of pixels in the map
        mdim (int): dimension of the liner system to do the map-making
        ndet (int): number of detectors
        duration (float): duration [s] of the observation
        sampling_rate (float): sampling rate [Hz] of the observation
        channel (str): name of the channel
        net_detector_ukrts (float): net detector noise level [uKrts] of the detector
        net_channel_ukrts (float): net channel noise level [uKrts] of the detectors
        noise_pdf (np.ndarray): probability density function of the noise per sky pixel
        covmat_inv (np.ndarray): inverse of the covariance matrix of the stokes parameters
        """
        self.ss = {}
        self.hitmap = []
        self.h = []
        self.spins_n = []
        self.spins_m = []
        self.compled_fields = None
        self.use_hwp = None
        self.nside = None
        self.npix = None
        self.mdim = None
        self.ndet = None
        self.duration = None
        self.sampling_rate = None
        self.channel = None
        self.net_detector_ukrts = None
        self.net_channel_ukrts = None
        self.noise_pdf = None
        self.covmat_inv = None
        self.xlink_threshold = 0.7

    @classmethod
    def load_det(cls, det_name: str, base_path=DB_ROOT_PATH):
        """ Load the scan fields data of a detector from a .h5 file

        Args:
            filename (str): name of the *.h5 file containing the scan fields data simulated by Falcons.jl

            base_path (str): path to the directory containing the *.h5 file

        Returns:
            instance (ScanFields): instance of the ScanFields class containing
            the scan fields data of the detector
        """
        instance = cls()
        if base_path.split("/")[-1] in channel_list:
            instance.channel = base_path.split("/")[-1]
        instance.ndet = 1
        t2b = None
        if det_name[-1] == "B":
            t2b = True
            det_name = det_name[:-1] + "T"
        filename = det_name + ".h5"

        with h5py.File(os.path.join(base_path, filename), 'r') as f:
            instance.ss = {key: value[()] for key, value in zip(f['ss'].keys(), f['ss'].values()) if key != "quat"}
            instance.hitmap = f['hitmap'][:]
            instance.h = f['h'][:,:,:]
            instance.h[np.isnan(instance.h)] = 1.0
            instance.spins_n = f['toml']['spin_n'][()]
            instance.spins_m = f['toml']['spin_m'][()]
        instance.nside = instance.ss["nside"]
        instance.duration = instance.ss["duration"]
        instance.sampling_rate = instance.ss["sampling_rate"]
        instance.npix = hp.nside2npix(instance.nside)
        if t2b == True:
            instance = instance.t2b()
        return instance

    @classmethod
    def load_channel(cls, channel: str, base_path=DB_ROOT_PATH):
        """ Load the scan fields data of a channel from the directory containing the *.h5 files

        Args:
            base_path (str): path to the directory containing the *.h5 files

            channel (str): name of the channel to load the scan fields data from

        Returns:
            instance (ScanFields): instance of the ScanFields class containing
            the scan fields data of the channel
        """
        dirpath = os.path.join(base_path, channel)
        filenames = os.listdir(dirpath)
        filenames = [os.path.splitext(filename)[0] for filename in filenames]
        first_sf = cls.load_det(filenames[0], base_path=dirpath)
        instance = cls()
        instance.channel = channel
        instance.ndet = len(filenames)
        instance.hitmap = np.zeros_like(first_sf.hitmap)
        instance.h = np.zeros_like(first_sf.h)
        instance.nside = first_sf.nside
        instance.npix = hp.nside2npix(instance.nside)
        instance.duration = first_sf.duration
        instance.sampling_rate = first_sf.sampling_rate
        for filename in filenames:
            sf = cls.load_det(filename, base_path=dirpath)
            instance.hitmap += sf.hitmap
            instance.h += sf.hitmap[:, np.newaxis, np.newaxis] * sf.h
        instance.h /= instance.hitmap[:, np.newaxis, np.newaxis]
        instance.spins_n = first_sf.spins_n
        instance.spins_m = first_sf.spins_m
        return instance

    @classmethod
    def _load_channel_task(cls, args):
        base_path, ch = args
        return cls.load_channel(ch, base_path)

    @classmethod
    def load_full_FPU(cls, channel_list: list, base_path=DB_ROOT_PATH, max_workers=None):
        """ Load the scan fields data of all the channels in the FPU from
        the directory containing the *.h5 files

        Args:
            base_path (str): path to the directory containing the channel's data

            channel_list (list): list of channels to load the scan fields data

            max_workers (int): number of processes to use for loading the scan
                               fields data of the channels. Default is None, which
                               uses the number of CPUs in the system

        Returns:
            instance (ScanFields): instance of the ScanFields class containing the scan
                                   fields data of all the channels in the FPU
        """
        if max_workers is None:
            max_workers = os.cpu_count()
        print(f"Using {max_workers} processes")
        with Pool(processes=max_workers) as pool:
            crosslink_channels = pool.map(cls._load_channel_task, [(base_path, ch) for ch in channel_list])
        instance = cls()
        hitmap = np.zeros_like(crosslink_channels[0].hitmap)
        h = np.zeros_like(crosslink_channels[0].h)
        instance.nside = crosslink_channels[0].nside
        instance.npix = hp.nside2npix(instance.nside)
        instance.duration = crosslink_channels[0].duration
        instance.sampling_rate = crosslink_channels[0].sampling_rate
        ndet = 0
        for sf in crosslink_channels:
            hitmap += sf.hitmap
            h += sf.hitmap[:, np.newaxis, np.newaxis] * sf.h
            ndet += sf.ndet
        instance.ndet = ndet
        instance.hitmap = hitmap
        instance.h = h / hitmap[:, np.newaxis, np.newaxis]
        instance.spins_n = crosslink_channels[0].spins_n
        instance.spins_m = crosslink_channels[0].spins_m
        return instance

    def t2b(self):
        """ Transform Top detector cross-link to Bottom detector cross-link
        It assume top and bottom detector make a orthogonal pair.
        """
        class_copy = copy.deepcopy(self)
        class_copy.h *= np.exp(-1j * self.spins_n * (np.pi / 2))
        return class_copy

    def __add__(self, other):
        """ Add `hitmap` and `h` of two Scanfield instances
        For the `hitmap`, it adds the `hitmap` of the two instances
        For `h`, it adds the cross-link of the two instances weighted by the hitmap
        """
        if not isinstance(other, ScanFields):
            return NotImplemented
        result = copy.deepcopy(self)
        result.hitmap += other.hitmap
        result.h = (self.h*self.hitmap[:,np.newaxis,np.newaxis] + other.h*other.hitmap[:,np.newaxis,np.newaxis])/result.hitmap[:,np.newaxis,np.newaxis]
        return result

    def initialize(self, mdim):
        """ Initialize the scan fields data """
        self.hitmap = np.zeros_like(self.hitmap)
        self.h = np.zeros_like(self.h)
        self.nside = hp.npix2nside(len(self.hitmap))
        self.npix = hp.nside2npix(self.nside)
        self.mdim = mdim
        self.ndet = 0
        self.coupled_fields = np.zeros([self.mdim, self.npix], dtype=np.complex128)

    def get_xlink(self, spin_n, spin_m):
        """ Get the cross-link of the detector for a given spin number

        Args:
            spin_n (int): spin number for which the cross-link is to be obtained

            spin_m (int): spin number for which the cross-link is to be obtained

            If `spin_n` and `spin_m` are 0, the cross-link for the spin number 0 is returned, i.e,
            the map which has 1 in the real part and zero in the imaginary part.

        Returns:
            xlink (1d-np.ndarray): cross-link of the detector for the given spin numbers
        """
        assert abs(spin_n) in self.spins_n, f"spin_n={spin_n} is not in the spins_n={self.spins_n}"
        assert spin_m in self.spins_m, f"spin_m={spin_m} is not in the spins_m={self.spins_m}"
        if spin_n == 0 and spin_m == 0:
            return np.ones_like(self.h[:, 0, 0]) + 1j * np.zeros_like(self.h[:, 0, 0])
        if spin_n > 0:
            idx_n = np.where(self.spins_n == np.abs(spin_n))[0][0]
            idx_m = np.where(self.spins_m == spin_m)[0][0]
            result = self.h[:, idx_m, idx_n]
        else:
            idx_n = np.where(self.spins_n == np.abs(spin_n))[0][0]
            idx_m = np.where(self.spins_m == -spin_m)[0][0]
            result = self.h[:, idx_m, idx_n].conj()
        return result

    def get_covmat(self, mdim, use_hwp):
        """ Get the covariance matrix of the detector in `mdim`x`mdim` matrix form

        Args:
            mdim (int): dimension of the covariance matrix.
        """
        if use_hwp == True:
            if mdim == 2:
                covmat = self.create_covmat([-2,2], [4,-4], use_hwp)
            elif mdim == 3:
                covmat = self.create_covmat([0,-2,2], [0,4,-4], use_hwp)
            elif mdim == 5:
                covmat = self.create_covmat([0,-1,1,-2, 2],
                                            [0, 0,0, 4,-4], use_hwp)
            elif mdim == 9:
                covmat = self.create_covmat([0,-1, 1,-2, 2,-3, 3,-1, 1],
                                            [0, 0, 0, 4,-4, 4,-4, 4,-4], use_hwp)
            else:
                raise ValueError("mdim is 2, 3, 5 and 9 are only supported")
        else:
            if mdim == 2:
                covmat = self.create_covmat([2,-2], [0,0], use_hwp)
            elif mdim == 3:
                covmat = self.create_covmat([0,2,-2], [0,0,0], use_hwp)
            elif mdim == 5:
                covmat = self.create_covmat([0,1,-1,2,-2], [0,0,0,0,0], use_hwp)
            elif mdim == 7:
                covmat = self.create_covmat([0,1,-1,2,-2,3,-3], [0,0,0,0,0,0,0], use_hwp)
            else:
                raise ValueError("mdim is 2, 3, 5 and 7 are only supported")
        return covmat

    def create_covmat(self, base_spin_n: list, base_spin_m: list, use_hwp: bool):
        """ Get the covariance matrix of the detector in `mdim`x`mdim` matrix form

        Args:
            base_spin_n (list): list of spin_n to create the covariance matrix

            base_spin_m (list): list of spin_m to create the covariance matrix
        """
        assert len(base_spin_n) == len(base_spin_m), "base_spin_n and base_spin_m should have the same length"
        base_spin_n = np.array(base_spin_n)
        base_spin_m = np.array(base_spin_m)
        waits = np.array([0.5 if x != 0 else 1.0 for x in base_spin_n])
        spin_n_mat =  base_spin_n[:,np.newaxis] - base_spin_n[np.newaxis,:]
        spin_m_mat =  base_spin_m[:,np.newaxis] - base_spin_m[np.newaxis,:]
        if use_hwp == True:
            spin_n_mat = -spin_n_mat
            spin_m_mat = -spin_m_mat
        wait_mat = np.abs(waits[np.newaxis,:]) * np.abs(waits[:,np.newaxis])
        covmat = np.zeros([len(base_spin_n),len(base_spin_m),self.npix], dtype=complex)
        for i in range(len(base_spin_n)):
            for j in range(len(base_spin_m)):
                covmat[i,j,:] = self.get_xlink(spin_n_mat[i,j], spin_m_mat[i,j])*wait_mat[i,j]
        return covmat

    def get_coupled_field(self, signal_fields: SignalFields, spin_n_out: int, spin_m_out: int):
        """ Multiply the scan fields and signal fields to get the detected fields by
        given cross-linking

        Args:
            signal_fields (SignalFields): signal fields data of the detector

            spin_n_out (int): spin_n of the output field

            spin_m_out (int): spin_m of the output field

        Returns:
            results (np.ndarray): detected fields by the given cross-linking
        """
        results = []
        for i in range(len(signal_fields.spins_n)):
            n = signal_fields.spins_n[i]
            m = signal_fields.spins_m[i]
            delta_n = spin_n_out - n
            delta_m = spin_m_out - m
            hS = self.get_xlink(delta_n,delta_m) * signal_fields.get_field(n,m)
            results.append(hS)
        coupled_field = np.array(results).sum(0)
        return coupled_field

    @staticmethod
    def diff_gain_field(gain_T, gain_B, I, P):
        """" Get the differential gain field of the detector

        Args:
            gain_T (float): gain of the `Top` detector

            gain_B (float): gain of the `Bottom` detector

            I (np.ndarray): temperature map

            P (np.ndarray): polarization map (i.e. Q+iU)

        Returns:
            signal_fields (SignalFields): differential gain field of the detector
        """
        delta_g = gain_T - gain_B
        signal_fields = SignalFields(
            Field(delta_g*I/2.0, spin_n=0, spin_m=0),
            Field((2.0+gain_T+gain_B)*P/4.0, spin_n=2, spin_m=0),
            Field((2.0+gain_T+gain_B)*P.conj()/4.0, spin_n=-2, spin_m=0),
        )
        return signal_fields

    @staticmethod
    def diff_pointing_field(
        rho_T: float,
        rho_B: float,
        chi_T: float,
        chi_B: float,
        P: np.ndarray,
        eth_I: np.ndarray,
        eth_P: np.ndarray,
        o_eth_P: np.ndarray
        ):
        """ Get the differential pointing field of the detector

        Args:
            rho_T (float): magnitude of pointing offset of the `Top` detector in radian

            rho_B (float): magnitude of pointing offset of the `Bottom` detector in radian

            chi_T (float): direction of the pointing offset of the `Top` detector in radian

            chi_B (float): direction of the pointing offset of the `Bottom` detector in radian

            P (np.ndarray): polarization map (i.e. Q+iU)

            eth_I (np.ndarray): spin up gradient of the temperature map

            eth_P (np.ndarray): spin up gradient of the polarization map

            o_eth_P (np.ndarray): spin down gradient of the polarization map
        """
        zeta   = rho_T * np.exp(1j*chi_T) - 1j*rho_B * np.exp(1j*chi_B)
        o_zeta = rho_T * np.exp(1j*chi_T) + 1j*rho_B * np.exp(1j*chi_B) #\overline{\zeta}

        spin_0_field  = Field(np.zeros(len(P)), spin_n=0, spin_m=0)
        spin_1_field  = Field(-1.0/4.0 * (zeta*eth_I + o_zeta.conj()*o_eth_P), spin_n=1, spin_m=0)
        spin_m1_field = spin_1_field.conj()
        spin_2_field  = Field(P/2.0, spin_n=2, spin_m=0)
        spin_m2_field = spin_2_field.conj()
        spin_3_field  = Field(-1.0/4.0 * o_zeta * eth_P, spin_n=3, spin_m=0)
        spin_m3_field = spin_3_field.conj()

        diff_pointing_field = SignalFields(
            spin_0_field,
            spin_1_field,
            spin_m1_field,
            spin_2_field,
            spin_m2_field,
            spin_3_field,
            spin_m3_field,
        )
        return diff_pointing_field

    @staticmethod
    def hwp_ip_field(epsilon, phi_qi, I):
        """ Get the HWP instrumental polarization field of the detector

        Args:
            epsilon (float): Amplitude of the HWP Meuller matrix element varying at 4f

            phi_qi (float): Phase shift in the HWP

            I (np.ndarray): temperature map
        """
        signal_fields = SignalFields(
            Field(epsilon/2.0 * np.exp(-1j*phi_qi)*I, spin_n=4, spin_m=-4),
            Field(epsilon/2.0 * np.exp(1j*phi_qi)*I, spin_n=-4, spin_m=4),
        )
        return signal_fields

    @staticmethod
    def abs_pointing_field(
        rho: float,
        chi: float,
        I: np.ndarray,
        P: np.ndarray,
        eth_I: np.ndarray,
        eth_P: np.ndarray,
        o_eth_P: np.ndarray
        ):
        """ Get the absolute pointing field of the detector

        Args:
            rho (float): magnitude of pointing offset in radian

            chi (float): direction of the pointing offset in radian

            I (np.ndarray): temperature map

            P (np.ndarray): polarization map (i.e. Q+iU)

            eth_I (np.ndarray): spin up gradient of the temperature map

            eth_P (np.ndarray): spin up gradient of the polarization map

            o_eth_P (np.ndarray): spin down gradient of the polarization map
        """
        spin_00_field   = Field(I, spin_n=0, spin_m=0)
        spin_p2m4_field = Field(P/2.0, spin_n=2, spin_m=-4)
        spin_m2p4_field = spin_p2m4_field.conj()
        spin_p10_field  = Field(-rho/2.0*np.exp(1j*chi)*eth_I, spin_n=1, spin_m=0)
        spin_m10_field  = spin_p10_field.conj()
        spin_p1m4_field = Field(-rho/4.0*np.exp(-1j*chi)*o_eth_P, spin_n=1, spin_m=-4)
        spin_m1p4_field = spin_p1m4_field.conj()
        spin_p3m4_field = Field(-rho/4.0*np.exp(1j*chi)*eth_P, spin_n=3, spin_m=-4)
        spin_m3p4_field = spin_p3m4_field.conj()
        abs_pointing_field = SignalFields(
            spin_00_field,
            spin_p10_field,
            spin_m10_field,
            spin_p2m4_field,
            spin_m2p4_field,
            spin_p3m4_field,
            spin_m3p4_field,
            spin_p1m4_field,
            spin_m1p4_field,
        )
        return abs_pointing_field

    def couple(self, signal_fields: SignalFields, mdim):
        """ Get the coupled fields which is obtained by multiplication between cross-link
        and signal fields. The function save the coupled fields in the class instance.
        This method must be called before the 'ScanFields.map_make()' and 'ScanFields.solve()'method.

        Args:
            signal_fields (SignalFields): signal fields data of the detector

            mdim (int): dimension of the system (here, the map)
        """
        self.mdim = mdim

        if np.all(signal_fields.spins_m == 0):
            self.use_hwp = False
        else:
            self.use_hwp = True

        if self.use_hwp == True:
            s_00 = self.get_coupled_field(signal_fields, spin_n_out=0, spin_m_out=0)
            sp2m4 = self.get_coupled_field(signal_fields, spin_n_out=2, spin_m_out=-4)
            if self.mdim==3:
                coupled_fields = np.array([s_00, sp2m4/2.0, sp2m4.conj()/2.0])
            elif self.mdim==5:
                # for pointing offset
                sp10 = self.get_coupled_field(signal_fields, spin_n_out=1, spin_m_out=0)
                coupled_fields = np.array([s_00, sp10/2.0, sp10.conj()/2.0, sp2m4/2.0, sp2m4.conj()/2.0])
            elif self.mdim==9:
                # for pointing offset
                sp10 = self.get_coupled_field(signal_fields, spin_n_out=1, spin_m_out=0)
                sp3m4 = self.get_coupled_field(signal_fields, spin_n_out=3, spin_m_out=-4)
                sp1m4 = self.get_coupled_field(signal_fields, spin_n_out=1, spin_m_out=-4)
                coupled_fields = np.array([
                    s_00,
                    sp10/2.0,
                    sp10.conj()/2.0,
                    sp2m4/2.0,
                    sp2m4.conj()/2.0,
                    sp3m4/2.0,
                    sp3m4.conj()/2.0,
                    sp1m4/2.0,
                    sp1m4.conj()/2.0,
                    ])
            else:
                raise ValueError("mdim is 2, 3, 5 and 7 only supported")
        else:
            sp2 = self.get_coupled_field(signal_fields, spin_n_out=2, spin_m_out=0)
            if self.mdim==2:
                coupled_fields = np.array([sp2/2.0, sp2.conj()/2.0])
            elif self.mdim==3:
                s_0 = self.get_coupled_field(signal_fields, spin_n_out=0, spin_m_out=0)
                coupled_fields = np.array([s_0, sp2/2.0, sp2.conj()/2.0])
            elif self.mdim==5:
                s_0 = self.get_coupled_field(signal_fields, spin_n_out=0, spin_m_out=0)
                sp1 = self.get_coupled_field(signal_fields, spin_n_out=1, spin_m_out=0)
                coupled_fields = np.array([s_0, sp1/2.0, sp1.conj()/2.0, sp2/2.0, sp2.conj()/2.0])
            elif self.mdim==7:
                s_0 = self.get_coupled_field(signal_fields, spin_n_out=0, spin_m_out=0)
                sp1 = self.get_coupled_field(signal_fields, spin_n_out=1, spin_m_out=0)
                sp3 = self.get_coupled_field(signal_fields, spin_n_out=3, spin_m_out=0)
                coupled_fields = np.array([s_0, sp1/2.0, sp1.conj()/2.0, sp2/2.0, sp2.conj()/2.0, sp3/2.0, sp3.conj()/2.0])
            else:
                raise ValueError("mdim is 2, 3, 5 and 7 only supported")
        self.coupled_fields = coupled_fields

    def map_make(self, signal_fields, mdim, only_iqu=True):
        """ Get the output map by solving the linear equation Ax=b
        This operation gives us an equivalent result of the simple binning map-making aproach

        Args:
            signal_fields (SignalFields): signal fields data of the detector

            mdim (int): dimension of the liner system

            only_iqu (bool): if True, return only I, Q, U map

        Returns:
            output_map (np.ndarray, [`mdim`, `npix`])
        """
        self.mdim = mdim
        self.couple(signal_fields, mdim=self.mdim)
        b = self.coupled_fields
        x = np.zeros_like(b)

        if self.use_hwp == True:
            A = self.get_covmat(self.mdim, use_hwp=True)
            for i in range(self.npix):
                if self.hitmap[i] != 0:
                    x[:,i] = np.linalg.solve(A[:,:,i], b[:,i])
        else:
            A = self.get_covmat(self.mdim, use_hwp=False)
            xlink2 = np.abs(self.get_xlink(2,0))
            for i in range(self.npix):
                if xlink2[i] < self.xlink_threshold:
                    x[:,i] = np.linalg.solve(A[:,:,i], b[:,i])

        if mdim == 2:
            # output_map =        [Fake I                  , Q        , U        ]
            output_map = np.array([np.zeros_like(x[0].real), x[0].real, x[0].imag])
        elif mdim == 3:
            # output_map =        [I        , Q        , U        ]
            output_map = np.array([x[0].real, x[1].real, x[1].imag])
        elif mdim == 5:
            # output_map =        [I        , Z1^Q     , Z1^U     , Q        , U        ]
            output_map = np.array([x[0].real, x[1].real, x[1].imag, x[3].real, x[3].imag])
        elif mdim == 7:
            # output_map =        [I        , Z1^Q     , Z1^U     , Q        , U        ,Z3^Q      , Z3^U     ]
            output_map = np.array([x[0].real, x[1].real, x[1].imag, x[3].real, x[3].imag, x[5].real, x[5].imag])
        elif mdim == 9:
            # output_map =        [I        , Z1^Q     , Z1^U     , Q        , U        ,Z3^Q      , Z3^U     ,Z1^Q^4    , Z1^U^4   ]
            output_map = np.array([x[0].real, x[1].real, x[1].imag, x[3].real, x[3].imag, x[5].real, x[5].imag, x[7].real, x[7].imag])
        elif only_iqu == True:
            if mdim > 3:
                output_map = np.array([output_map[0], output_map[3], output_map[4]])
        return output_map

    def solve(self):
        """ Get the output map by solving the linear equation Ax=b
        This operation gives us an equivalent result of the simple binning map-making aproach
        """
        assert self.coupled_fields is not None, "Couple the fields first by `ScanFields.couple()` method."
        b = self.coupled_fields
        x = np.zeros_like(b)

        if self.use_hwp == True:
            A = self.get_covmat(self.mdim, use_hwp=True)
            for i in range(b.shape[1]):
                if self.hitmap[i] != 0:
                    x[:,i] = np.linalg.solve(A[:,:,i], b[:,i])
        else:
            A = self.get_covmat(self.mdim, use_hwp=True)
            xlink2 = np.abs(self.get_xlink(2,0))
            for i in range(b.shape[1]):
                if xlink2[i] < self.xlink_threshold:
                    x[:,i] = np.linalg.solve(A[:,:,i], b[:,i])

        if self.mdim == 2:
            # output_map =        [Fake I                  , Q        , U        ]
            output_map = np.array([np.zeros_like(x[0].real), x[0].real, x[0].imag])
        elif self.mdim == 3:
            # output_map =        [I        , Q        , U        ]
            output_map = np.array([x[0].real, x[1].real, x[1].imag])
        elif self.mdim == 5:
            # output_map =        [I        , Z1^Q     , Z1^U     , Q        , U        ]
            output_map = np.array([x[0].real, x[1].real, x[1].imag, x[3].real, x[3].imag])
        elif self.mdim == 7:
            # output_map =        [I        , Z1^Q     , Z1^U     , Q        , U        ,Z3^Q      , Z3^U     ]
            output_map = np.array([x[0].real, x[1].real, x[1].imag, x[3].real, x[3].imag, x[5].real, x[5].imag])
        elif mdim == 9:
            # output_map =        [I        , Z1^Q     , Z1^U     , Q        , U        ,Z3^Q      , Z3^U     ,Z1^Q^4    , Z1^U^4   ]
            output_map = np.array([x[0].real, x[1].real, x[1].imag, x[3].real, x[3].imag, x[5].real, x[5].imag, x[7].real, x[7].imag])
        return output_map

    @classmethod
    def sim_diff_gain_channel(
        cls,
        channel: str,
        mdim: int,
        input_map: np.ndarray,
        gain_T: np.ndarray,
        gain_B: np.ndarray,
        base_path=DB_ROOT_PATH,
        ):
        """ Simulate the differential gain channel

        Args:
            channel (str): name of the channel to simulate

            mdim (int): dimension of the system (here, the map)

            input_map (np.ndarray): input map of the simulation

            gain_T (np.ndarray): gain of the `Top` detector

            gain_B (np.ndarray): gain of the `Bottom` detector

            base_path (str): path to the directory containing the channel's data.
                             Default is `DB_ROOT_PATH`
        """
        dirpath = os.path.join(base_path, channel)
        filenames = os.listdir(dirpath)
        filenames = [os.path.splitext(filename)[0] for filename in filenames]
        assert len(filenames) == len(gain_T) == len(gain_B)
        total_sf = cls.load_det(filenames[0], base_path=dirpath)
        total_sf.initialize(mdim)
        total_sf.ndet = len(filenames)
        total_sf.use_hwp = False
        assert input_map.shape == (3,len(total_sf.hitmap))
        I = input_map[0]
        P = input_map[1] + 1j*input_map[2]
        for i,filename in enumerate(filenames):
            sf = cls.load_det(filename, base_path=dirpath)
            signal_fields = ScanFields.diff_gain_field(gain_T[i], gain_B[i], I, P)
            sf.couple(signal_fields, mdim)
            total_sf.hitmap += sf.hitmap
            total_sf.h += sf.h * sf.hitmap[:, np.newaxis, np.newaxis]
            total_sf.coupled_fields += sf.coupled_fields * sf.hitmap
        total_sf.coupled_fields /= total_sf.hitmap
        total_sf.h /= total_sf.hitmap[:, np.newaxis, np.newaxis]
        return total_sf

    @classmethod
    def sim_diff_pointing_channel(
        cls,
        channel: str,
        mdim: int,
        input_map: np.ndarray,
        rho_T: np.ndarray, # Pointing offset magnitude
        rho_B: np.ndarray,
        chi_T: np.ndarray, # Pointing offset direction
        chi_B: np.ndarray,
        base_path=DB_ROOT_PATH,
        ):
        """ Simulate the differential pointing channel

        Args:
            channel (str): name of the channel to simulate

            mdim (int): dimension of the system (here, the map)

            input_map (np.ndarray): input map of the simulation

            rho_T (np.ndarray): magnitude of pointing offset of the `Top` detector in radian

            rho_B (np.ndarray): magnitude of pointing offset of the `Bottom` detector in radian

            chi_T (np.ndarray): direction of the pointing offset of the `Top` detector in radian

            chi_B (np.ndarray): direction of the pointing offset of the `Bottom` detector in radian

            base_path (str): path to the directory containing the channel's data.
                             Default is `DB_ROOT_PATH`
        """
        dirpath = os.path.join(base_path, channel)
        filenames = os.listdir(dirpath)
        filenames = [os.path.splitext(filename)[0] for filename in filenames]
        assert len(filenames) == len(rho_T) == len(chi_T) == len(rho_B) == len(chi_B)
        total_sf = cls.load_det(filenames[0], base_path=dirpath)
        total_sf.initialize(mdim)
        total_sf.ndet = len(filenames)
        total_sf.use_hwp = False
        assert input_map.shape == (3,len(total_sf.hitmap))

        I = input_map[0]
        P = input_map[1] + 1j*input_map[2]
        nside = hp.npix2nside(len(I))
        dI = hp.alm2map_der1(hp.map2alm(input_map[0]), nside=nside)
        dQ = hp.alm2map_der1(hp.map2alm(input_map[1]), nside=nside)
        dU = hp.alm2map_der1(hp.map2alm(input_map[2]), nside=nside)

        eth_I = dI[2] - 1j*dI[1]
        eth_P   = dQ[2] + dU[1] - 1j*(dQ[1] - dU[2])
        o_eth_P = dQ[2] - dU[1] + 1j*(dQ[1] + dU[2])

        for i,filename in enumerate(filenames):
            sf = cls.load_det(filename, base_path=dirpath)
            signal_fields = ScanFields.diff_pointing_field(rho_T[i], rho_B[i], chi_T[i], chi_B[i], P, eth_I, eth_P, o_eth_P)
            sf.couple(signal_fields, mdim)
            total_sf.hitmap += sf.hitmap
            total_sf.h += sf.h * sf.hitmap[:, np.newaxis, np.newaxis]
            total_sf.coupled_fields += sf.coupled_fields * sf.hitmap
        total_sf.coupled_fields /= total_sf.hitmap
        total_sf.h /= total_sf.hitmap[:, np.newaxis, np.newaxis]
        return total_sf

    def generate_noise_pdf(
        self,
        imo=None,
        net_ukrts=None,
        return_pdf=False,
        scale=1.0,
        ):
        """ Generate probability density function (PDF) of the noise.
        The function store the noise PDF in the `self.noise_pdf` attribute.

        Args:
            imo (Imo): IMo object which contains the instrument information given by the `litebird_sim`

            net_ukrts (float): net sensitivity of the detector in uK√s

            return_pdf (bool): if True, return the noise PDF

            scale (float): scale factor to adjust the noise PDF.
                           When the defferential detection is performed, it should be scale = 2.0.
        """
        channel = self.channel
        hitmap_tmp = self.hitmap.copy()
        hitmap_tmp[hitmap_tmp == 0] = 1 # avoid zero division
        if channel:
            assert imo is not None, "imo is required when channel is given"
            inst = get_instrument_table(imo, imo_version="v2")
            net_detector_ukrts = inst.loc[inst["channel"] == channel, "net_detector_ukrts"].values[0]
            net_channel_ukrts = inst.loc[inst["channel"] == channel, "net_channel_ukrts"].values[0]

            sigma_i = net_detector_ukrts * np.sqrt(self.sampling_rate) / np.sqrt(scale*hitmap_tmp)
            sigma_i *= np.sign(self.hitmap)
            sigma_p = sigma_i/np.sqrt(2.0)
            self.net_channel_ukrts = net_channel_ukrts
        else:
            assert net_ukrts is not None, "net_ukrts is required when channel is not given"
            net_detector_ukrts = net_ukrts
            sigma_i = net_detector_ukrts * np.sqrt(self.sampling_rate) / np.sqrt(scale*hitmap_tmp)
            sigma_i *= np.sign(self.hitmap)
            sigma_p = sigma_i/np.sqrt(2.0)
        self.net_detector_ukrts = net_detector_ukrts
        self.noise_pdf = np.array([sigma_i, sigma_p])
        if return_pdf:
            return self.noise_pdf

    def generate_noise(self, mdim, use_hwp, seed=None):
        """ Generate observed noise map with the noise PDF.

        Args:
            mdim (int): dimension of the linear system in the map-making equation

            seed (int): seed for the random number generator

        Returns:
            noise (np.ndarray) [3,npix]: noise map
        """
        assert self.noise_pdf is not None, "Generate noise pdf first by `ScanField.generate_noise_pdf()` method."
        xlink2 = np.abs(self.get_xlink(2,0))

        cov = self.get_covmat(mdim, use_hwp)
        covmat_inv = np.empty_like(cov)
        if use_hwp == True:
            for i in range(self.npix):
                if self.hitmap[i] != 0:
                    covmat_inv[:,:,i] = np.linalg.inv(cov[:,:,i])
                else:
                    covmat_inv[:,:,i] = np.zeros_like(cov[:,:,i])
        else:
            for i in range(self.npix):
                if xlink2[i] < self.xlink_threshold:
                    covmat_inv[:,:,i] = np.linalg.inv(cov[:,:,i])
                else:
                    covmat_inv[:,:,i] = np.zeros_like(cov[:,:,i])
        self.covmat_inv = np.sqrt(covmat_inv)

        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()
        n_i = np.random.normal(loc=0., scale=self.noise_pdf[0], size=[self.npix])
        n_q = np.random.normal(loc=0., scale=self.noise_pdf[1], size=[self.npix])
        n_u = np.random.normal(loc=0., scale=self.noise_pdf[1], size=[self.npix])

        if use_hwp == True:
            n_i[self.hitmap == 0] = 0.0
            n_q[self.hitmap == 0] = 0.0
            n_u[self.hitmap == 0] = 0.0
        else:
            n_i[xlink2 > self.xlink_threshold] = 0.0
            n_q[xlink2 > self.xlink_threshold] = 0.0
            n_u[xlink2 > self.xlink_threshold] = 0.0

        if mdim == 2:
            noise = np.array([
                np.zeros_like(n_i),
                n_q * self.covmat_inv[0,0,:].real,
                n_u * self.covmat_inv[0,0,:].real,
                ])
        elif mdim == 3:
            noise = np.array([
                n_i * self.covmat_inv[0,0,:].real,
                n_q * self.covmat_inv[1,1,:].real,
                n_u * self.covmat_inv[1,1,:].real,
            ])
        elif mdim > 3:
            noise = np.array([
                    n_i * self.covmat_inv[0,0,:].real,
                    n_q * self.covmat_inv[3,3,:].real,
                    n_u * self.covmat_inv[3,3,:].real,
                ])
        return noise


def plot_maps(mdim, input_map, output_map, residual, cmap="viridis"):
    """ Plot the input, output and residual maps

    Args:
        mdim (int): dimension of the linear system in the map-making equation

        input_map (np.ndarray): input map of the simulation

        output_map (np.ndarray): output map of the simulation

        residual (np.ndarray): residual map of the simulation

        cmap (str): colormap to use
    """
    titles = ["Input", "Output", "Residual"]
    maps = [input_map, output_map, residual]
    if mdim == 2:
        units = ["I", "Q", "U"]
        for i, title in enumerate(titles[:2]):
            plt.figure(figsize=(10, 5))
            for j in range(1,3):
                hp.mollview(maps[i][j], sub=(1, 2, j), title=f"{title} ${units[j]}$", unit="$\mu K_{CMB}$", cmap=cmap)

        plt.figure(figsize=(10, 5))
        for j in range(1, 3):
            hp.mollview(residual[j], sub=(1, 2, j), title=f"Residual $\Delta {units[j]}$", unit="$\mu K_{CMB}$", cmap=cmap)

    elif mdim == 3:
        units = ["I", "Q", "U"]
        for i, title in enumerate(titles[:2]):
            plt.figure(figsize=(15, 5))
            for j in range(3):
                hp.mollview(maps[i][j], sub=(1, 3, j + 1), title=f"{title} ${units[j]}$", unit="$\mu K_{CMB}$", cmap=cmap)

        plt.figure(figsize=(15, 5))
        for j in range(1, 3):
            hp.mollview(residual[j], sub=(1, 2, j), title=f"Residual $\Delta {units[j]}$", unit="$\mu K_{CMB}$", cmap=cmap)

    elif mdim == 5:
        titles = ["Input", "Output", "Residual"]
        units = ["I", "{}_{1}Z^Q", "{}_{1}Z^U", "Q", "U"]
        for i, title in enumerate(titles):
            plt.figure(figsize=(15, 5))
            for j in range(maps[i].shape[0]):
                hp.mollview(maps[i][j], sub=(1, len(units), j + 1), title=f"{title} ${units[j]}$", unit="$\mu K_{CMB}$", cmap=cmap)

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
