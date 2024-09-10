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
class Field:
    """ Class to store the field data of detectors """
    def __init__(self, field: np.ndarray, spin: int):
        """Initialize the class with field and spin data

        Args:
            field (np.ndarray): field data of the detector

            spin (int): spin number of the detector
        """
        if all(isinstance(x, float) for x in field):
            self.field = field + 1j*np.zeros(len(field))
        else:
            self.field = field
        self.spin = spin

    def conj(self):
        """Get the complex conjugate of the field"""
        return Field(self.field.conj(), -self.spin)

    def __add__(self, other):
        """Add the field of two detectors

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
        self.fields = sorted(fields, key=lambda field: field.spin)
        self.spins = np.array([field.spin for field in self.fields])

    def __add__(self, other):
        """Add the signal fieldd

        Args:
            other (SignalFields): signal fields

        Returns:
            result (SignalFields): sum of the signal fields
        """
        if not isinstance(other, SignalFields):
            return NotImplemented
        result = copy.deepcopy(self)
        for i in range(len(self.spins)):
            result.fields[i].field += other.fields[i].field
        return result

class ScanFields:
    """ Class to store the scan fields data of detectors """
    def __init__(self):
        """ Initialize the class with empty data

        ss (dict):  of the scanning strategy parameters

        hitmap (np.ndarray): hitmap of the detector

        h (np.ndarray): cross-link (orientation function) of the detector

        spins (np.ndarray): array of spin numbers

        std (np.ndarray): standard deviation of the hitmap and h

        mean (np.ndarray): mean of the hitmap and h

        all_channels (list): list of all the channels in the LiteBIRD
        """
        self.ss = {}
        self.hitmap = []
        self.h = []
        self.spins = []
        self.compled_fields = None
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
        self.all_channels = [
            'L1-040','L2-050','L1-060','L3-068','L2-068','L4-078','L1-078','L3-089','L2-089','L4-100','L3-119','L4-140',
            'M1-100','M2-119','M1-140','M2-166','M1-195',
            'H1-195','H2-235','H1-280','H2-337','H3-402'
        ]
        self.fwhms = [70.5,58.5,51.1,41.6,47.1,36.9,43.8,33.0,41.5,30.2,26.3,23.7,37.8,33.6,30.8,28.9,28.0,28.6,24.7,22.5,20.9,17.9]

    @classmethod
    def load_det(cls, base_path: str, det_name: str):
        """ Load the scan fields data of a detector from a .h5 file

        Args:
            base_path (str): path to the directory containing the .h5 file

            filename (str): name of the .h5 file containing the scan fields data simulated by Falcons.jl
            The fileformat requires cross-link_2407-dataset's format.
            The file should contain the following groups:
                - ss: scanning strategy parameters
                - hitmap: hitmap of the detector
                - h: cross-link (orientation function) of the detector
                - quantify: group containing the following datasets
                    - n: number of spins
                    - mean: mean of the hitmap and h
                    - std: standard deviation of the hitmap and h
        Returns:
            instance (ScanFields): instance of the ScanFields class containing the scan fields data of the detector
        """
        instance = cls()
        if base_path.split("/")[-1] in instance.all_channels:
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
            instance.h = f['h'][:, 0, :]
            instance.h[np.isnan(instance.h)] = 1.0
            instance.spins = f['quantify']['n'][()]
        instance.nside = instance.ss["nside"]
        instance.duration = instance.ss["duration"]
        instance.sampling_rate = instance.ss["sampling_rate"]
        instance.npix = hp.nside2npix(instance.nside)
        if t2b == True:
            instance = instance.t2b()
        return instance

    @classmethod
    def load_channel(cls, base_path: str, channel: str):
        """Load the scan fields data of a channel from the directory containing the .h5 files

        Args:
            base_path (str): path to the directory containing the .h5 files

            channel (str): name of the channel to load the scan fields data from

        Returns:
            instance (ScanFields): instance of the ScanFields class containing the scan fields data of the channel
        """
        dirpath = os.path.join(base_path, channel)
        filenames = os.listdir(dirpath)
        filenames = [os.path.splitext(filename)[0] for filename in filenames]
        first_sf = cls.load_det(dirpath, filenames[0])
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
            sf = cls.load_det(dirpath, filename)
            instance.hitmap += sf.hitmap
            instance.h += sf.hitmap[:, np.newaxis] * sf.h
        instance.h /= instance.hitmap[:, np.newaxis]
        instance.spins = first_sf.spins
        return instance

    @classmethod
    def _load_channel_task(cls, args):
        base_path, ch = args
        return cls.load_channel(base_path, ch)

    @classmethod
    def load_full_FPU(cls, base_path: str, channel_list: list, max_workers=None):
        """ Load the scan fields data of all the channels in the FPU from the directory containing the .h5 files

        Args:
            base_path (str): path to the directory containing the .h5 files

            channel_list (list): list of channels to load the scan fields data from

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
        instance.nside = crosslink_channels[0].ss["nside"]
        instance.npix = hp.nside2npix(instance.nside)
        instance.duration = crosslink_channels[0].ss["duration"]
        instance.sampling_rate = crosslink_channels[0].ss["sampling_rate"]
        ndet = 0
        for sf in crosslink_channels:
            hitmap += sf.hitmap
            h += sf.hitmap[:, np.newaxis] * sf.h
            ndet += sf.ndet
        instance.ndet = ndet
        instance.hitmap = hitmap
        instance.h = h / hitmap[:, np.newaxis]
        instance.spins = crosslink_channels[0].spins
        return instance

    def initialize(self, mdim):
        self.hitmap = np.zeros_like(self.hitmap)
        self.h = np.zeros_like(self.h)
        self.nside = hp.npix2nside(len(self.hitmap))
        self.npix = hp.nside2npix(self.nside)
        self.spins = np.zeros_like(self.spins)
        self.mdim = mdim
        self.ndet = 0
        self.coupled_fields = np.zeros([self.mdim, len(self.hitmap)], dtype=np.complex128)

    def get_xlink(self, spin_n: int):
        """Get the cross-link of the detector for a given spin number

        Args:
            spin_n (int): spin number for which the cross-link is to be obtained

            If s`pin_n` is 0, the cross-link for the spin number 0 is returned, i.e,
            the map which has 1 in the real part and zero in the imaginary part.

        Returns:
            xlink (1d-np.ndarray): cross-link of the detector for the given spin number
        """

        if spin_n == 0:
            return np.ones_like(self.h[:, 0]) + 1j * np.zeros_like(self.h[:, 0])
        if spin_n < 0:
            return self.h[:, np.abs(spin_n) - 1].conj()
        else:
            return self.h[:, spin_n - 1]

    def get_covmat(self, mdim):
        """Get the covariance matrix of the detector in `mdim`x`mdim` matrix form

        Args:
            mdim (int): dimension of the covariance matrix.
        """
        if mdim == 2:
            covmat = self.create_covmat(base_spin=[2,-2])
        elif mdim == 3:
            covmat = self.create_covmat(base_spin=[0,2,-2])
        elif mdim == 5:
            covmat = self.create_covmat(base_spin=[0,1,-1,2,-2])
        elif mdim == 7:
            covmat = self.create_covmat(base_spin=[0,1,-1,2,-2,3,-3])
        else:
            raise ValueError("mdim is 2, 3, 5 and 7 are only supported")
        return covmat

    def create_covmat(self, base_spin: list):
        """Get the covariance matrix of the detector in `mdim`x`mdim` matrix form

        Args:
            base_spin (list): list of spin to create the covariance matrix
        """
        base_spin = np.array(base_spin)
        waits = np.array([0.5 if x != 0 else 1.0 for x in base_spin])
        spin_mat =  base_spin[:,np.newaxis] - base_spin[np.newaxis,:]
        #print(spin_mat)
        wait_mat = np.abs(waits[np.newaxis,:]) * np.abs(waits[:,np.newaxis])
        covmat = np.zeros([len(base_spin),len(base_spin),self.npix], dtype=complex)
        for i in range(len(base_spin)):
            for j in range(len(base_spin)):
                covmat[i,j,:] = self.get_xlink(spin_mat[i,j])*wait_mat[i,j]
        return covmat

    def t2b(self):
        """Transform Top detector cross-link to Bottom detector cross-link
        It assume top and bottom detector make a orthogonal pair.
        """
        class_copy = copy.deepcopy(self)
        class_copy.h *= np.exp(-1j * self.spins * (np.pi / 2))
        return class_copy

    def __add__(self, other):
        """Add `hitmap` and `h` of two Scanfield instances
        For the `hitmap`, it adds the `hitmap` of the two instances
        For `h`, it adds the cross-link of the two instances weighted by the hitmap
        """
        if not isinstance(other, ScanFields):
            return NotImplemented
        result = copy.deepcopy(self)
        result.hitmap += other.hitmap
        result.h = (self.h*self.hitmap[:, np.newaxis] + other.h*other.hitmap[:, np.newaxis])/result.hitmap[:, np.newaxis]
        return result

    def get_coupled_field(self, signal_fields: SignalFields, spin_out: int):
        """ Multiply the scan fields and signal fields to get the detected fields by
        given cross-linking

        Args:
            scan_fields (ScanFields): scan fields data of the detector

            signal_fields (SignalFields): signal fields data of the detector

            spin_out (int): spin number of the output field

        Returns:
            results (np.ndarray): detected fields by the given cross-linking
        """
        results = []
        for i in range(len(signal_fields.spins)):
            n = spin_out - signal_fields.spins[i]
            #print(f"n-n': {n}, n: {signal_fields.spins[i]}")
            results.append(self.get_xlink(n) * signal_fields.fields[i].field)
        return np.array(results).sum(0)

    @staticmethod
    def diff_gain_field(gain_a, gain_b, I, P):
        delta_g = gain_a - gain_b
        signal_fields = SignalFields(
            Field(delta_g*I/2.0, spin=0),
            Field((2.0+gain_a+gain_b)*P/4.0, spin=2),
            Field((2.0+gain_a+gain_b)*P.conj()/4.0, spin=-2),
        )
        return signal_fields

    @classmethod
    def sim_diff_gain_channel(
        cls,
        base_path: str,
        channel: str,
        mdim: int,
        input_map: np.ndarray,
        gain_a: np.ndarray,
        gain_b: np.ndarray
        ):
        dirpath = os.path.join(base_path, channel)
        filenames = os.listdir(dirpath)
        filenames = [os.path.splitext(filename)[0] for filename in filenames]
        assert len(filenames) == len(gain_a) == len(gain_b)
        total_sf = cls.load_det(dirpath, filenames[0])
        total_sf.initialize(mdim)
        total_sf.ndet = len(filenames)
        assert input_map.shape == (3,len(total_sf.hitmap))
        I = input_map[0]
        P = input_map[1] + 1j*input_map[2]
        for i,filename in enumerate(filenames):
            sf = cls.load_det(dirpath, filename)
            signal_fields = ScanFields.diff_gain_field(gain_a[i], gain_b[i], I, P)
            sf.couple(signal_fields, mdim)
            total_sf.hitmap += sf.hitmap
            total_sf.h += sf.h * sf.hitmap[:, np.newaxis]
            total_sf.coupled_fields += sf.coupled_fields * sf.hitmap
        total_sf.coupled_fields /= total_sf.hitmap
        total_sf.h /= total_sf.hitmap[:, np.newaxis]
        return total_sf

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
        zeta   = rho_T * np.exp(1j*chi_T) - 1j*rho_B * np.exp(1j*chi_B)
        o_zeta = rho_T * np.exp(1j*chi_T) + 1j*rho_B * np.exp(1j*chi_B) #\overline{\zeta}

        spin_0_field  = Field(np.zeros(len(P)), spin=0)
        spin_1_field  = Field(-1.0/4.0 * (zeta*eth_I + o_zeta.conj()*o_eth_P), spin=1)
        spin_m1_field = spin_1_field.conj()
        spin_2_field  = Field(P/2.0, spin=2)
        spin_m2_field = spin_2_field.conj()
        spin_3_field  = Field(-1.0/4.0 * o_zeta * eth_P, spin=3)
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

    @classmethod
    def sim_diff_pointing_channel(
        cls,
        base_path: str,
        channel: str,
        mdim: int,
        input_map: np.ndarray,
        rho_T: np.ndarray, # Pointing offset magnitude
        rho_B: np.ndarray,
        chi_T: np.ndarray,  # Pointing offset direction
        chi_B: np.ndarray,
        ):

        dirpath = os.path.join(base_path, channel)
        filenames = os.listdir(dirpath)
        filenames = [os.path.splitext(filename)[0] for filename in filenames]
        assert len(filenames) == len(rho_T) == len(chi_T) == len(rho_B) == len(chi_B)
        total_sf = cls.load_det(dirpath, filenames[0])
        total_sf.initialize(mdim)
        total_sf.ndet = len(filenames)
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
            sf = cls.load_det(dirpath, filename)
            signal_fields = ScanFields.diff_pointing_field(rho_T[i], rho_B[i], chi_T[i], chi_B[i], P, eth_I, eth_P, o_eth_P)
            sf.couple(signal_fields, mdim)
            total_sf.hitmap += sf.hitmap
            total_sf.h += sf.h * sf.hitmap[:, np.newaxis]
            total_sf.coupled_fields += sf.coupled_fields * sf.hitmap
        total_sf.coupled_fields /= total_sf.hitmap
        total_sf.h /= total_sf.hitmap[:, np.newaxis]
        return total_sf

    def couple(self, signal_fields, mdim):
        """Get the coupled fields which is obtained by multiplication between cross-link
        and signal fields

        Args:
            signal_fields (SignalFields): signal fields data of the detector

            mdim (int): dimension of the system (here, the map)

        Returns:
            compled_fields (np.ndarray)
        """
        self.mdim = mdim

        sp2 = self.get_coupled_field(signal_fields, spin_out=2)
        sm2 = self.get_coupled_field(signal_fields, spin_out=-2)
        if self.mdim==2:
            coupled_fields = np.array([sp2/2.0, sm2/2.0])
        elif self.mdim==3:
            s_0 = self.get_coupled_field(signal_fields, spin_out=0)
            coupled_fields = np.array([s_0, sp2/2.0, sm2/2.0])
        elif self.mdim==5:
            s_0 = self.get_coupled_field(signal_fields, spin_out=0)
            sp1 = self.get_coupled_field(signal_fields, spin_out=1)
            sm1 = self.get_coupled_field(signal_fields, spin_out=-1)
            coupled_fields = np.array([s_0, sp1/2.0, sm1/2.0, sp2/2.0, sm2/2.0])
        elif self.mdim==7:
            s_0 = self.get_coupled_field(signal_fields, spin_out=0)
            sp1 = self.get_coupled_field(signal_fields, spin_out=1)
            sm1 = self.get_coupled_field(signal_fields, spin_out=-1)
            sp3 = self.get_coupled_field(signal_fields, spin_out=3)
            sm3 = self.get_coupled_field(signal_fields, spin_out=-3)
            coupled_fields = np.array([s_0, sp1/2.0, sm1/2.0, sp2/2.0, sm2/2.0, sp3/2.0, sm3/2.0])
        else:
            raise ValueError("mdim is 2, 3, 5 and 7 only supported")
        self.coupled_fields = coupled_fields

    def map_make(self, signal_fields, mdim):
        """Get the output map by solving the linear equation Ax=b
        This operation gives us an equivalent result of the simple binning map-making aproach

        Args:
            signal_fields (SignalFields): signal fields data of the detector

            mdim (int): dimension of the liner system

        Returns:
            output_map (np.ndarray, [`mdim`, `npix`])
        """
        self.couple(signal_fields, mdim=mdim)
        #if seed:
        #    np.random.seed(seed)
        #    noise = self.generate_noise(seed)
        b = self.coupled_fields# + noise
        A = self.get_covmat(mdim)
        x = np.empty_like(b)
        for i in range(b.shape[1]):
            x[:,i] = np.linalg.solve(A[:,:,i], b[:,i])
        if mdim == 2:
            # Note that:
            # x[0] = Q + iU
            # x[1] = Q - iU
            output_map = np.array([np.zeros_like(x[0].real), x[0].real, x[0].imag])
        if mdim == 3:
            # Note that:
            # x[1] = Q + iU
            # x[2] = Q - iU
            output_map = np.array([x[0].real, x[1].real, x[1].imag])
        if mdim == 5:
            output_map = np.array([x[0].real, x[1].real, x[1].imag, x[3].real, x[3].imag])
        if mdim == 7:
            output_map = np.array([x[0].real, x[1].real, x[1].imag, x[2].real, x[2].imag, x[3].real, x[3].imag])
        return output_map

    def solve(self):
        """Get the output map by solving the linear equation Ax=b
        This operation gives us an equivalent result of the simple binning map-making aproach
        """
        assert self.coupled_fields is not None, "Couple the fields first"
        b = self.coupled_fields
        A = self.get_covmat(self.mdim)
        x = np.empty_like(b)
        for i in range(b.shape[1]):
            x[:,i] = np.linalg.solve(A[:,:,i], b[:,i])
        if self.mdim == 2:
            output_map = np.array([np.zeros_like(x[0].real), x[0].real, x[0].imag])
        if self.mdim == 3:
            output_map = np.array([x[0].real, x[1].real, x[1].imag])
        if self.mdim == 5:
            output_map = np.array([x[0].real, x[1].real, x[1].imag, x[3].real, x[3].imag])
        if self.mdim == 7:
            output_map = np.array([x[0].real, x[1].real, x[1].imag, x[3].real, x[3].imag, x[5].real, x[5].imag])
        return output_map

    def generate_noise_pdf(
        self,
        imo=None,
        net_ukrts=None,
        return_pdf=False,
        scale=1.0,
        ):
        """ Generate probability density function of the noise.
        The function store the noise PDF in the `self.noise_pdf` attribute.

        Args:
            imo (Imo): IMo object which contains the instrument information given by the `litebird_sim`

            net_ukrts (float): net sensitivity of the detector in uK√s

            return_pdf (bool): if True, return the noise PDF

            scale (float): scale factor to adjust the noise PDF.
                           When the defferential detection is performed, it should be scale = 2.0.
        """
        channel = self.channel
        if channel:
            assert imo is not None, "imo is required when channel is given"
            inst = get_instrument_table(imo, imo_version="v2")
            net_detector_ukrts = inst.loc[inst["channel"] == channel, "net_detector_ukrts"].values[0]
            net_channel_ukrts = inst.loc[inst["channel"] == channel, "net_channel_ukrts"].values[0]
            sigma_i = net_detector_ukrts * np.sqrt(self.sampling_rate) / np.sqrt(scale*self.hitmap)
            sigma_p = sigma_i/np.sqrt(2.0)
            self.net_channel_ukrts = net_channel_ukrts
        else:
            assert net_ukrts is not None, "net_ukrts is required when channel is not given"
            net_detector_ukrts = net_ukrts
            sigma_i = net_detector_ukrts * np.sqrt(self.sampling_rate) / np.sqrt(scale*self.hitmap)
            sigma_p = sigma_i/np.sqrt(2.0)
        self.net_detector_ukrts = net_detector_ukrts
        self.noise_pdf = np.array([sigma_i, sigma_p])
        if return_pdf:
            return self.noise_pdf

    def generate_noise(self, mdim, seed=None):
        """ Generate observed noise map with the noise PDF.

        Args:
            mdim (int): dimension of the linear system in the map-making equation

            seed (int): seed for the random number generator

        Returns:
            noise (np.ndarray): noise map
        """
        assert self.noise_pdf is not None, "Generate noise pdf first by `ScanField.generate_noise_pdf()` method."
        if self.covmat_inv is None or self.covmat_inv.shape[0] != mdim:
            cov = self.get_covmat(mdim)
            covmat_inv = np.empty_like(cov)
            for i in range(self.npix):
                covmat_inv[:,:,i] = np.linalg.inv(cov[:,:,i])
            self.covmat_inv = np.sqrt(covmat_inv)
        if seed:
            np.random.seed(seed)
        n_i = np.random.normal(loc=0., scale=self.noise_pdf[0], size=[self.npix])
        n_q = np.random.normal(loc=0., scale=self.noise_pdf[1], size=[self.npix])
        n_u = np.random.normal(loc=0., scale=self.noise_pdf[1], size=[self.npix])
        if mdim == 2:
            noise = np.array([
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
