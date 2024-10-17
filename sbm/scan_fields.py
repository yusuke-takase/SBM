import h5py
import numpy as np
import healpy as hp
import os
import copy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
import litebird_sim as lbs
from pathlib import Path
import toml
from .signal_fields import Field, SignalFields

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

    def create_covmat(self, spin_n_basis: list, spin_m_basis: list):
        """ Get the covariance matrix of the detector in `mdim`x`mdim` matrix form

        Args:
            base_spin_n (list): list of spin_n to create the covariance matrix

            base_spin_m (list): list of spin_m to create the covariance matrix
        """
        base_spin_n = np.array(spin_n_basis)
        base_spin_m = np.array(spin_m_basis)
        if np.all(base_spin_m == 0):
            self.use_hwp = False
        else:
            self.use_hwp = True
        waits = np.array([0.5 if x != 0 else 1.0 for x in base_spin_n])
        spin_n_mat =  base_spin_n[:,np.newaxis] - base_spin_n[np.newaxis,:]
        spin_m_mat =  base_spin_m[:,np.newaxis] - base_spin_m[np.newaxis,:]
        if self.use_hwp == True:
            spin_n_mat = -spin_n_mat
            spin_m_mat = -spin_m_mat
        wait_mat = np.abs(waits[np.newaxis,:]) * np.abs(waits[:,np.newaxis])
        covmat = np.zeros([len(base_spin_n),len(base_spin_m),self.npix], dtype=complex)
        for i in range(len(base_spin_n)):
            for j in range(len(base_spin_m)):
                covmat[i,j,:] = self.get_xlink(spin_n_mat[i,j], spin_m_mat[i,j])*wait_mat[i,j]
        return covmat

    def map_make(
        self,
        signal_fields: SignalFields,
        only_iqu=True
        ):
        """ Get the output map by solving the linear equation Ax=b
        This operation gives us an equivalent result of the simple binning map-making aproach

        Args:
            signal_fields (SignalFields): signal fields data of the detector

            only_iqu (bool): if True, return only I, Q, U map

        Returns:
            if only_iqu == True:
                output_map (np.ndarray, [3, `npix`])
            if only_iqu == False:
                output_map (np.ndarray, [len(signal_fields.spin_n_basis), `npix`])
        """
        assert signal_fields.coupled_fields is not None, "No coupled field in the SignalFields.coupled_fields"
        if np.all(signal_fields.spins_m == 0):
            self.use_hwp = False
        else:
            self.use_hwp = True
        spin_n_basis = np.array(signal_fields.spin_n_basis)
        spin_m_basis = np.array(signal_fields.spin_m_basis)
        b = signal_fields.coupled_fields
        x = np.zeros_like(b)
        if self.use_hwp == True:
            pol_idx = np.where((spin_n_basis == 2) & (spin_m_basis == -4))[0][0]
            A = self.create_covmat(
                spin_n_basis,
                spin_m_basis,
                )
            for i in range(self.npix):
                if self.hitmap[i] != 0:
                    x[:,i] = np.linalg.solve(A[:,:,i], b[:,i])
        else:
            pol_idx = np.where((spin_n_basis == 2) & (spin_m_basis == 0))[0][0]
            A = self.create_covmat(
                spin_n_basis,
                spin_m_basis,
                )
            xlink2 = np.abs(self.get_xlink(2,0))
            for i in range(self.npix):
                if xlink2[i] < self.xlink_threshold:
                    x[:,i] = np.linalg.solve(A[:,:,i], b[:,i])
        if only_iqu == True:
            output_map = [np.zeros_like(x[pol_idx].real), x[pol_idx].real, x[pol_idx].imag]
        else:
            output_map = x
        return output_map


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

    def generate_noise(self, spin_n_basis: list, spin_m_basis: list, seed=None):
        """ Generate observed noise map with the noise PDF.

        Args:
            mdim (int): dimension of the linear system in the map-making equation

            seed (int): seed for the random number generator

        Returns:
            noise (np.ndarray) [3,npix]: noise map
        """
        assert self.noise_pdf is not None, "Generate noise pdf first by `ScanField.generate_noise_pdf()` method."
        spin_n_basis = np.array(spin_n_basis)
        spin_m_basis = np.array(spin_m_basis)
        xlink2 = np.abs(self.get_xlink(2,0))
        if np.all(spin_m_basis == 0):
            self.use_hwp = False
        else:
            self.use_hwp = True

        cov = self.create_covmat(
            spin_n_basis,
            spin_m_basis,
            )
        covmat_inv = np.empty_like(cov)

        if self.use_hwp == True:
            pol_idx = np.where((spin_n_basis == 2) & (spin_m_basis == -4))[0][0]
            for i in range(self.npix):
                if self.hitmap[i] != 0:
                    covmat_inv[:,:,i] = np.linalg.inv(cov[:,:,i])
                else:
                    covmat_inv[:,:,i] = np.zeros_like(cov[:,:,i])
        else:
            pol_idx = np.where((spin_n_basis == 2) & (spin_m_basis == 0))[0][0]
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

        if self.use_hwp == True:
            n_i[self.hitmap == 0] = 0.0
            n_q[self.hitmap == 0] = 0.0
            n_u[self.hitmap == 0] = 0.0
        else:
            n_i[xlink2 > self.xlink_threshold] = 0.0
            n_q[xlink2 > self.xlink_threshold] = 0.0
            n_u[xlink2 > self.xlink_threshold] = 0.0
        noise = np.array([
                np.zeros_like(n_i),
                n_q * self.covmat_inv[pol_idx,pol_idx,:].real,
                n_u * self.covmat_inv[pol_idx,pol_idx,:].real,
                ])
        return noise
