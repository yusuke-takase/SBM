import h5py
import numpy as np
import healpy as hp
import os
import copy
import matplotlib.pyplot as plt
from multiprocessing import Pool


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

class SignalFields:
    """ Class to store the signal fields data of detectors """
    def __init__(self, *fields: Field):
        """ Initialize the class with field data

        Args:
            fields (Field): field (map) data of the signal
        """
        self.fields = sorted(fields, key=lambda field: field.spin)
        self.spins = np.array([field.spin for field in self.fields])

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
        self.all_channels = [
            'L1-040','L2-050','L1-060','L3-068','L2-068','L4-078','L1-078','L3-089','L2-089','L4-100','L3-119','L4-140',
            'M1-100','M2-119','M1-140','M2-166','M1-195',
            'H1-195','H2-235','H1-280','H2-337','H3-402'
        ]
        self.fwhms = [70.5,58.5,51.1,41.6,47.1,36.9,43.8,33.0,41.5,30.2,26.3,23.7,37.8,33.6,30.8,28.9,28.0,28.6,24.7,22.5,20.9,17.9]

    @classmethod
    def load_det(cls, base_path: str, filename: str):
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
        assert filename.endswith('.h5'), "Invalid file path. Must be a .h5 file"
        instance = cls()
        instance.ndet = 1
        with h5py.File(os.path.join(base_path, filename), 'r') as f:
            instance.ss = {key: value[()] for key, value in zip(f['ss'].keys(), f['ss'].values()) if key != "quat"}
            instance.hitmap = f['hitmap'][:]
            instance.h = f['h'][:, 0, :]
            instance.h[np.isnan(instance.h)] = 1.0
            instance.spins = f['quantify']['n'][()]
        instance.nside = hp.npix2nside(len(instance.hitmap))
        instance.npix = hp.nside2npix(instance.nside)
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
        first_sf = cls.load_det(dirpath, filenames[0])
        instance = cls()
        instance.ndet = len(filenames)
        instance.hitmap = np.zeros_like(first_sf.hitmap)
        instance.h = np.zeros_like(first_sf.h)
        instance.nside = hp.npix2nside(len(instance.hitmap))
        instance.npix = hp.nside2npix(instance.nside)
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
        instance.nside = hp.npix2nside(len(instance.hitmap))
        instance.npix = hp.nside2npix(instance.nside)
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
            covmat = np.array([
                [self.get_xlink(0)/4.0 , self.get_xlink(4)/4.0],
                [self.get_xlink(-4)/4.0, self.get_xlink(0)/4.0]
            ])
        elif mdim==3:
            covmat = np.array([
                [self.get_xlink(0)     , self.get_xlink(-2)/2.0, self.get_xlink(2)/2.0],
                [self.get_xlink(2)/2.0 , self.get_xlink(0)/4.0 , self.get_xlink(4)/4.0],
                [self.get_xlink(-2)/2.0, self.get_xlink(-4)/4.0, self.get_xlink(0)/4.0]
                ])
        else:
            raise ValueError("mdim is 2 or 3 only supported")
        return covmat

    def create_covmat(self, base_spin: list):
        """Get the covariance matrix of the detector in `mdim`x`mdim` matrix form

        Args:
            base_spin (list): list of spin to create the covariance matrix
        """
        base_spin = np.array(base_spin)
        waits = np.array([0.5 if x != 0 else 1.0 for x in base_spin])
        spin_mat = base_spin[np.newaxis,:] - base_spin[:,np.newaxis]
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
        assert len(filenames) == len(gain_a) == len(gain_b)
        total_sf = cls.load_det(dirpath, filenames[0])
        total_sf.initialize(mdim)
        total_sf.ndet = len(filenames)
        assert input_map.shape == (3,len(total_sf.hitmap))
        I = input_map[0]
        P = input_map[1] + 1j*input_map[2]
        for i,filename in enumerate(filenames):
            sf = cls.load_det(dirpath, filename)
            delta_g = gain_a[i] - gain_b[i]
            signal_fields = SignalFields(
                Field(delta_g*I/2.0, spin=0),
                Field((2.0+gain_a[i]+gain_b[i])*P/4.0, spin=2),
                Field((2.0+gain_a[i]+gain_b[i])*P.conj()/4.0, spin=-2),
            )
            sf.couple(signal_fields, mdim)
            total_sf.hitmap += sf.hitmap
            total_sf.h += sf.h * sf.hitmap[:, np.newaxis]
            total_sf.coupled_fields += sf.coupled_fields * sf.hitmap
        total_sf.coupled_fields /= total_sf.hitmap
        total_sf.h /= total_sf.hitmap[:, np.newaxis]
        return total_sf

    @staticmethod
    def _diff_pointing_field(
        rho: float,
        chi: float,
        I: np.ndarray,
        P: np.ndarray,
        eth_I: np.ndarray,
        eth_P: np.ndarray
        ):
        spin_0_field  = Field(I, spin=0)
        spin_1_field  = Field(-rho/2*np.exp(1j*chi)*eth_I, spin=1)
        spin_m1_field = spin_1_field.conj()
        spin_2_field  = Field(P/2.0, spin=2)
        spin_m2_field = spin_2_field.conj()
        spin_3_field  = Field(-rho/4*np.exp(1j*chi)*eth_P, spin=3)
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
        rho: np.ndarray, # Pointing offset magnitude
        chi: np.ndarray  # Pointing offset direction
        ):

        dirpath = os.path.join(base_path, channel)
        filenames = os.listdir(dirpath)
        assert len(filenames) == len(rho) == len(chi)
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

        eth_I = dI[2] - dI[1]*1j
        eth_P = dQ[2] + dU[1] - (dQ[1] - dU[2])*1j

        for i,filename in enumerate(filenames):
            sf = cls.load_det(dirpath, filename)
            signal_fields = ScanFields._diff_pointing_field(rho[i], chi[i], I, P, eth_I, eth_P)
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
        s_0 = self.get_coupled_field(signal_fields, spin_out=0)
        sp2 = self.get_coupled_field(signal_fields, spin_out=2)
        sm2 = self.get_coupled_field(signal_fields, spin_out=-2)
        if self.mdim==2:
            coupled_fields = np.array([sp2/2.0, sm2/2.0])
        elif self.mdim==3:
            coupled_fields = np.array([s_0, sp2/2.0, sm2/2.0])
        else:
            raise ValueError("mdim is 2 or 3 only supported")
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
        b = self.coupled_fields
        A = self.get_covmat(mdim)
        x = np.empty_like(b)
        for i in range(b.shape[1]):
            x[:,i] = np.linalg.solve(A[:,:,i], b[:,i])
        if mdim == 2:
            # None that:
            # x[0] = Q + iU
            # x[1] = Q - iU
            output_map = np.array([np.zeros_like(x[0].real), x[0].real, x[0].imag])
        if mdim == 3:
            # None that:
            # x[1] = Q + iU
            # x[2] = Q - iU
            output_map = np.array([x[0].real, x[1].real, x[1].imag])
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
            # None that:
            # x[0] = Q + iU
            # x[1] = Q - iU
            output_map = np.array([np.zeros_like(x[0].real), x[0].real, x[0].imag])
        if self.mdim == 3:
            # None that:
            # x[1] = Q + iU
            # x[2] = Q - iU
            output_map = np.array([x[0].real, x[1].real, x[1].imag])
        return output_map


def plot_maps(mdim, input_map, output_map, residual):
    if mdim == 2:
        plt.figure(figsize=(10,5))
        hp.mollview(input_map[1], sub=(1,2,1), title="Input $Q$", unit="$\mu K_{CMB}$")
        hp.mollview(input_map[2], sub=(1,2,2), title="Input $U$", unit="$\mu K_{CMB}$")

        plt.figure(figsize=(10,5))
        hp.mollview(output_map[1], sub=(1,2,1), title="Output $Q$", unit="$\mu K_{CMB}$")
        hp.mollview(output_map[2], sub=(1,2,2), title="Output $U$", unit="$\mu K_{CMB}$")

        plt.figure(figsize=(10,5))
        hp.mollview(residual[1], sub=(1,2,1), title="Residual $\Delta Q$", unit="$\mu K_{CMB}$")
        hp.mollview(residual[2], sub=(1,2,2), title="Residual $\Delta U$", unit="$\mu K_{CMB}$")
    elif mdim == 3:
        plt.figure(figsize=(15,5))
        hp.mollview(input_map[0], sub=(1,3,1), title="Input $T$", unit="$\mu K_{CMB}$")
        hp.mollview(input_map[1], sub=(1,3,2), title="Input $Q$", unit="$\mu K_{CMB}$")
        hp.mollview(input_map[2], sub=(1,3,3), title="Input $U$", unit="$\mu K_{CMB}$")

        plt.figure(figsize=(15,5))
        hp.mollview(output_map[0], sub=(1,3,1), title="Output $T$", unit="$\mu K_{CMB}$")
        hp.mollview(output_map[1], sub=(1,3,2), title="Output $Q$", unit="$\mu K_{CMB}$")
        hp.mollview(output_map[2], sub=(1,3,3), title="Output $U$", unit="$\mu K_{CMB}$")

        plt.figure(figsize=(15,5))
        hp.mollview(residual[1], sub=(1,2,1), title="Residual $\Delta Q$", unit="$\mu K_{CMB}$")
        hp.mollview(residual[2], sub=(1,2,2), title="Residual $\Delta U$", unit="$\mu K_{CMB}$")
