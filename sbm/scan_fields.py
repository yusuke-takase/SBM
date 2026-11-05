# -*- encoding: utf-8 -*-

import h5py
import numpy as np
import healpy as hp
import os
import copy
import warnings
from multiprocessing import Pool
from pathlib import Path
import sympy as sp
import toml
from rich import print
from IPython.display import display, Math
from .signal_fields import SignalFields, Field
from .tools import get_instrument_table

CONFIG_PATH = Path.home() / ".config" / "sbm_dataset"
CONFIG_FILE_PATH = CONFIG_PATH / "sbm_dataset.toml"

#: Pixels whose map-making system has a reciprocal condition number (2-norm) at or
#: below this value are not solved by ScanFields.map_make and ScanFields.generate_noise:
#: they are 0, like the unobserved pixels.
RCOND_MIN = 1e-12


def extract_location_from_toml(file_path):
    with open(file_path, "r") as file:
        data = toml.load(file)
        location = data["repositories"][0]["location"]
    return location


if CONFIG_FILE_PATH.exists():
    try:
        DB_ROOT_PATH = extract_location_from_toml(CONFIG_FILE_PATH)
    except Exception as e:
        print(
            f"[red]Error:[/red] Extracting database location from TOML was fail: {e}\nYou should check '~/.config/sbm_dataset/sbm_dataset.toml'\nDB_ROOT_PATH set as `None`"
        )
        DB_ROOT_PATH = None
else:
    DB_ROOT_PATH = None

channel_list = [
    "L1-040",
    "L2-050",
    "L1-060",
    "L3-068",
    "L2-068",
    "L4-078",
    "L1-078",
    "L3-089",
    "L2-089",
    "L4-100",
    "L3-119",
    "L4-140",
    "M1-100",
    "M2-119",
    "M1-140",
    "M2-166",
    "M1-195",
    "H1-195",
    "H2-235",
    "H1-280",
    "H2-337",
    "H3-402",
]

fwhms = [
    70.5,
    58.5,
    51.1,
    41.6,
    47.1,
    36.9,
    43.8,
    33.0,
    41.5,
    30.2,
    26.3,
    23.7,
    37.8,
    33.6,
    30.8,
    28.9,
    28.0,
    28.6,
    24.7,
    22.5,
    20.9,
    17.9,
]


def read_scanfiled(file_path: str) -> "ScanFields":
    """Read the scan fields data of a detector from a HDF5 file

    Args:
        file_path (`str`): file path of the HDF5 file containing the scan fields data simulated by Falcons.jl

    Returns:
        instance (:class:`.ScanFields`): scan fields instance
    """
    return ScanFields.load_hdf5(file_path)


def _solvable_pixels(covmat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pixels of `mask` whose system `covmat[:, :, i]` is well conditioned.

    A pixel is solved if the reciprocal condition number (2-norm) of its system is
    above ``RCOND_MIN``; a pixel seen with fewer distinct angles than the basis needs
    is not. The pixels of the mask that are dropped are reported with a warning.
    """
    mask = np.asarray(mask, dtype=bool)
    ok = mask.copy()
    idx = np.flatnonzero(mask)
    if idx.size:
        a = np.moveaxis(covmat[:, :, idx], -1, 0)
        finite = np.all(np.isfinite(a), axis=(1, 2))
        rcond = np.zeros(idx.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            rcond[finite] = 1.0 / np.linalg.cond(a[finite])
        ok[idx] = np.isfinite(rcond) & (rcond > RCOND_MIN)
    dropped = int(np.count_nonzero(mask & ~ok))
    if dropped:
        warnings.warn(
            f"{dropped} of the pixels to solve have an ill-conditioned system and "
            "are not solved (set to 0)"
        )
    return ok


def _attr_to_str(value) -> str:
    """String value of an HDF5 attribute.

    Fixed-length string attributes, such as those written by Falcons.jl, are read by
    h5py as ``bytes`` and would otherwise become ``"b'...'"``.
    """
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


class ScanFields:
    """Class to store the scan fields data of detectors

    Attributes:
        ss (`dict`): scanning strategy parameters

        hitmap (`np.ndarray`): hitmap of the detector

        h (`np.ndarray`): cross-link (orientation function) of the detector

        spins_n (`np.ndarray`): array of spin_n numbers

        spins_m (`np.ndarray`): array of spin_m number

        compled_fields (`np.ndarray`): coupled fields between scan fields and signal fields

        use_hwp (`bool`): whether the observation uses HWP or not

        nside (`int`): nside of the map

        npix (`int`): number of pixels in the map

        mdim (`int`): dimension of the liner system to do the map-making

        ndet (`int`): number of detectors

        duration (`float`): duration [s] of the observation

        sampling_rate (`float`): sampling rate [Hz] of the observation

        channel (`str`): name of the channel

        net_detector_ukrts (`float`): net detector noise level [uKrts] of the detector

        net_channel_ukrts (`float`): net channel noise level [uKrts] of the detectors

        noise_pdf (`np.ndarray`): probability density function of the noise per sky pixel

        covmat_inv (`np.ndarray`): inverse of the covariance matrix of the stokes parameters

        xlink_threshold (`float`): threshold value to decide whether stokes parameter estimation is
        performed or not in the map-making. The sky pixel with the cross-link value larger than this
        threshold is ignored in the map-making. Default is 1.0 i.e. all the sky pixels stokes paramters are estimated.
    """

    def __init__(self):
        self.ss = {}
        self.hitmap = []
        self.h = []
        self.spins_n = []
        self.spins_m = []
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
        self.xlink_threshold = 1.0

    @classmethod
    def load_hdf5(cls, file_path: str) -> "ScanFields":
        """Load the scan fields data of a detector from a HDF5 file

        Args:
            file_path (`str`): file path of the HDF5 file containing the scan fields data simulated by Falcons.jl

        Returns:
            instance (:class:`.ScanFields`): instance of scan fields
            the scan fields data of the detector
        """
        assert os.path.exists(file_path), f"File not found: {file_path}"
        instance = cls()

        with h5py.File(os.path.join(file_path), "r") as f:
            instance.ss = {
                key: value[()]
                for key, value in zip(f["ss"].keys(), f["ss"].values())
                if key != "quat"
            }
            instance.hitmap = f["hitmap"][:]
            instance.h = f["h"][:, :, :]
            instance.h[np.isnan(instance.h)] = 1.0
            instance.spins_n = f["toml"]["spin_n"][()]
            instance.spins_m = f["toml"]["spin_m"][()]
        instance.nside = instance.ss["nside"]
        instance.duration = instance.ss["duration"]
        instance.sampling_rate = instance.ss["sampling_rate"]
        instance.npix = hp.nside2npix(instance.nside)
        return instance

    @classmethod
    def load_lbs_h_maps(cls, file_paths) -> "ScanFields":
        """Load h-maps written by litebird_sim's or Falcons.jl's `save_h_maps`

        The files follow litebird_sim's layout (``h_maps_det_<name>.h5``): one
        group ``"n,m"`` per couple holding the real and imaginary parts
        ``Re`` and ``Im`` of

        .. math:: h^{\\rm lbs}_{n,m} = \\langle e^{+i(n\\psi + m\\chi)} \\rangle,

        where :math:`\\psi` is litebird_sim's orientation angle and
        :math:`\\chi` the HWP angle, and the ``(0, 0)`` couple holds the hit
        counts. SBM's crossing angle is the opposite of litebird_sim's
        orientation angle, :math:`\\psi_{\\rm SBM} = -\\psi`: with this choice
        SBM's signal model reproduces litebird_sim's TOD model, including a
        rotating HWP. The cross-links are therefore stored as

        .. math:: \\tilde h_{n,m} = \\langle e^{-i(n\\psi_{\\rm SBM} + m\\chi)} \\rangle
                  = h^{\\rm lbs}_{n,-m}.

        The HWP angle of these scan fields is litebird_sim's :math:`\\chi`, which
        is the ``"telescope"`` HWP-angle convention of
        :meth:`.SignalFields.circular_pointing_field` (HWP wedge) and
        :meth:`.SignalFields.hwp_ip_field`: pass ``hwp_angle_frame="telescope"``
        to them (their default ``"legacy"`` only reproduces earlier results).

        The h-maps of litebird_sim do not include the detector polarisation
        angle: the cross-links of a file are those of a detector with
        polarisation angle 0. Unobserved pixels get cross-links equal to 1, as
        with :meth:`.load_hdf5`.

        Args:
            file_paths (`str` or `list` of `str`): one file, or several files,
                which are combined by hit-weighted averaging as in
                :meth:`.load_channel`. Combine only detectors with the same
                polarisation angle. The files do not hold it, and it changes the
                cross-links of each detector (without HWP, a polarisation angle
                :math:`\\alpha` shifts the phase of
                :math:`\\langle e^{-in\\psi}\\rangle` by :math:`n\\alpha`), so the
                average over detectors with different polarisation angles (e.g.
                all the detectors of a channel) does not describe their scan.

        Returns:
            instance (:class:`.ScanFields`): scan fields with ``spins_n`` the
            non-negative spin orders found in the files and ``spins_m`` the HWP orders
        """
        if isinstance(file_paths, (str, os.PathLike)):
            file_paths = [file_paths]
        assert len(file_paths) > 0, "No file given"
        instance = None
        for path in file_paths:
            current = cls._load_single_lbs_h_map(path)
            if instance is None:
                instance = current
                continue
            assert np.array_equal(instance.spins_n, current.spins_n) and np.array_equal(
                instance.spins_m, current.spins_m
            ), f"{path} has different spin orders"
            # valid only for detectors with the same polarisation angle (see the docstring)
            seen_a, seen_b = instance.hitmap > 0, current.hitmap > 0
            total = instance.hitmap + current.hitmap
            weighted = (
                np.where(seen_a, instance.hitmap, 0)[:, None, None] * instance.h
                + np.where(seen_b, current.hitmap, 0)[:, None, None] * current.h
            )
            instance.h = np.where(
                (total > 0)[:, None, None], weighted / np.maximum(total, 1)[:, None, None], 1.0
            )
            instance.hitmap = total
            instance.duration = max(instance.duration, current.duration)
        instance.ndet = len(file_paths)
        return instance

    @classmethod
    def _load_single_lbs_h_map(cls, file_path) -> "ScanFields":
        assert os.path.exists(file_path), f"File not found: {file_path}"
        couples = {}
        with h5py.File(file_path, "r") as f:
            attrs = dict(f.attrs)
            for key in f.keys():
                parts = key.split(",")
                if len(parts) != 2:
                    continue
                n, m = int(parts[0]), int(parts[1])
                couples[(n, m)] = f[key]["Re"][()] + 1j * f[key]["Im"][()]
        if (0, 0) not in couples:
            raise ValueError(f"{file_path} has no (0, 0) couple, which holds the hit counts")
        hitmap = np.rint(couples[(0, 0)].real).astype(np.int64)

        # h_{n,m} in SBM's convention, from either (n, -m) or the conjugate of (-n, m)
        def sbm_xlink(n, m):
            if (n, -m) in couples:
                return couples[(n, -m)]
            if (-n, m) in couples:
                return couples[(-n, m)].conj()
            return None

        spins_n = sorted({abs(n) for n, _ in couples})
        spins_m = sorted({-m if n >= 0 else m for n, m in couples if n != 0} | {0})
        npix = len(hitmap)
        h = np.empty((npix, len(spins_m), len(spins_n)), dtype=np.complex128)
        missing = []
        for j, m in enumerate(spins_m):
            for i, n in enumerate(spins_n):
                if n == 0 and m == 0:
                    h[:, j, i] = 1.0
                    continue
                value = sbm_xlink(n, m)
                if value is None:
                    missing.append((n, -m))
                    continue
                h[:, j, i] = value
        if missing:
            raise ValueError(
                f"{file_path} lacks the (n, m) couples {missing} (litebird_sim orders) needed for the full grid"
            )
        h[hitmap == 0] = 1.0

        instance = cls()
        instance.hitmap = hitmap
        instance.h = h
        instance.spins_n = np.array(spins_n)
        instance.spins_m = np.array(spins_m)
        instance.npix = npix
        instance.nside = hp.npix2nside(npix)
        instance.duration = float(attrs.get("duration_s", 0.0))
        instance.sampling_rate = float(attrs.get("sampling_rate_Hz", 0.0))
        instance.ss = {
            "nside": instance.nside,
            "duration": instance.duration,
            "sampling_rate": instance.sampling_rate,
            "coord": _attr_to_str(attrs.get("coordinate_system", "")),
            "name": _attr_to_str(attrs.get("det", "")),
        }
        instance.ndet = 1
        return instance

    @classmethod
    def load_det(cls, det_name: str, base_path=DB_ROOT_PATH) -> "ScanFields":
        """Load the scan fields data of a detector from a HDF5 file

        Args:
            filename (`str`): name of the HDF5 file containing the scan fields data simulated by Falcons.jl

            base_path (`str`): path to the directory containing the HDF5 file

        Returns:
            instance (:class:`.ScanFields`): instance of the ScanFields class containing
            the scan fields data of the detector
        """
        assert os.path.exists(
            base_path
        ), f"Directory not found: {base_path}\nPlease check the path to the database: {DB_ROOT_PATH}"
        instance = cls()
        if base_path.split("/")[-1] in channel_list:
            instance.channel = base_path.split("/")[-1]
        instance.ndet = 1
        t2b = None
        pa45 = None

        #changing pol angle
        if not det_name == "nside_32_boresight_hwp" and det_name.split("_")[3] != "XX":
            pa45 = True
            # bring back the det name to XX for polang
            dspl = det_name.split("_")
            det_name = dspl[0] +"_"+ dspl[1] + "_" + dspl[2] + "_" + "XX" + "_" + dspl[4] + "_" + dspl[5]
            print(f"{det_name}, 45 angle")
        
        if det_name[-1] == "B":
            t2b = True
            det_name = det_name[:-1] + "T"
            print(f"{det_name}, +90 angle")
        

        filename = det_name + ".h5"
        hdf5_file_path = os.path.join(base_path, filename)

        assert os.path.exists(
            hdf5_file_path
        ), f"File not found: {hdf5_file_path}\nPlease check the path to the database: {DB_ROOT_PATH}"
        with h5py.File(hdf5_file_path, "r") as f:
            instance.ss = {
                key: value[()]
                for key, value in zip(f["ss"].keys(), f["ss"].values())
                if key != "quat"
            }
            instance.hitmap = f["hitmap"][:]
            instance.h = f["h"][:, :, :]
            instance.h[np.isnan(instance.h)] = 1.0
            instance.spins_n = f["toml"]["spin_n"][()]
            instance.spins_m = f["toml"]["spin_m"][()]
        instance.nside = instance.ss["nside"]
        instance.duration = instance.ss["duration"]
        instance.sampling_rate = instance.ss["sampling_rate"]
        instance.npix = hp.nside2npix(instance.nside)
        if pa45 is True:
            instance = instance.pa45()
        if t2b is True:
            instance = instance.t2b()
        return instance

    @classmethod
    def load_channel(
        cls, channel: str, detector_list=None, base_path=DB_ROOT_PATH
    ) -> "ScanFields":
        """Load the scan fields data of a channel from the directory containing the HDF5 files

        Args:
            base_path (`str`): path to the directory containing the HDF5 files

            channel (`str`): name of the channel to load the scan fields data from

        Returns:
            instance (:class:`.ScanFields`): instance of the ScanFields class containing
            the scan fields data of the channel
        """
        dirpath = os.path.join(base_path, channel)
        if detector_list is None:
            filenames = os.listdir(dirpath)
            filenames = [os.path.splitext(filename)[0] for filename in filenames]
        else:
            filenames = detector_list
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
    def load_full_FPU(
        cls, channel_list: list, base_path=DB_ROOT_PATH, max_workers=None
    ) -> "ScanFields":
        """Load the scan fields data of all the channels in the FPU from
        the directory containing the HDF5 files

        Args:
            base_path (`str`): path to the directory containing the channel's data

            channel_list (`list`): list of channels to load the scan fields data

            max_workers (`int`): number of processes to use for loading the scan
                               fields data of the channels. Default is None, which
                               uses the number of CPUs in the system

        Returns:
            instance (:class:`.ScanFields`): instance of the ScanFields class containing the scan
                                   fields data of all the channels in the FPU
        """
        if max_workers is None:
            max_workers = os.cpu_count()
        print(f"Using {max_workers} processes")
        with Pool(processes=max_workers) as pool:
            crosslink_channels = pool.map(
                cls._load_channel_task, [(base_path, ch) for ch in channel_list]
            )
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

    def pa45(self) -> "ScanFields":
        """Change pol angle to 45 degree compared to 0 degree
        It assume top and bottom detector make a orthogonal pair.
        """
        class_copy = copy.deepcopy(self)
        class_copy.h *= np.exp(-1j * self.spins_n * (np.pi / 4))
        return class_copy

    def t2b(self) -> "ScanFields":
        """Transform Top detector cross-link to Bottom detector cross-link
        It assume top and bottom detector make a orthogonal pair.
        """
        class_copy = copy.deepcopy(self)
        class_copy.h *= np.exp(-1j * self.spins_n * (np.pi / 2))
        return class_copy

    def __add__(self, other: "ScanFields") -> "ScanFields":
        """Add `hitmap` and `h` of two Scanfield instances
        For the `hitmap`, it adds the `hitmap` of the two instances
        For `h`, it adds the cross-link of the two instances weighted by the hitmap
        """
        if not isinstance(other, ScanFields):
            return NotImplemented
        result = copy.deepcopy(self)
        result.hitmap += other.hitmap
        result.h = (
            self.h * self.hitmap[:, np.newaxis, np.newaxis]
            + other.h * other.hitmap[:, np.newaxis, np.newaxis]
        ) / result.hitmap[:, np.newaxis, np.newaxis]
        return result

    def initialize(self, mdim: int):
        """Initialize the scan fields data"""
        self.hitmap = np.zeros_like(self.hitmap)
        self.h = np.zeros_like(self.h)
        self.nside = hp.npix2nside(len(self.hitmap))
        self.npix = hp.nside2npix(self.nside)
        self.mdim = mdim
        self.ndet = 0
        self.coupled_fields = np.zeros([self.mdim, self.npix], dtype=np.complex128)

    def get_xlink(self, spin_n: int, spin_m: int) -> np.ndarray:
        """Get the cross-link of the detector for a given spin number

        Args:
            spin_n (`int`): spin number for which the cross-link is to be obtained

            spin_m (`int`): spin number for which the cross-link is to be obtained

        If `spin_n` and `spin_m` are 0, the cross-link for the spin number 0 is returned, i.e,
        the map which has 1 in the real part and zero in the imaginary part.

        Returns:
            xlink (`np.ndarray`): cross-link of the detector for the given spin numbers
        """
        if spin_n == 0 and spin_m == 0:
            return np.ones_like(self.h[:, 0, 0]) + 1j * np.zeros_like(self.h[:, 0, 0])
        else:
            assert (
                abs(spin_n) in self.spins_n
            ), f"spin_n={spin_n} is not in the spins_n={self.spins_n}"
            assert (
                spin_m in self.spins_m
            ), f"spin_m={spin_m} is not in the spins_m={self.spins_m}"
        if spin_n > 0:
            idx_n = np.where(self.spins_n == np.abs(spin_n))[0][0]
            idx_m = np.where(self.spins_m == spin_m)[0][0]
            result = self.h[:, idx_m, idx_n]
        else:
            idx_n = np.where(self.spins_n == np.abs(spin_n))[0][0]
            idx_m = np.where(self.spins_m == -spin_m)[0][0]
            result = self.h[:, idx_m, idx_n].conj()
        return result

    def create_covmat(self, spin_n_basis: list, spin_m_basis: list) -> np.ndarray:
        """Get the covariance matrix of the detector in `mdim`x`mdim` matrix form

        Args:
            base_spin_n (`list`): list of spin_n to create the covariance matrix

            base_spin_m (`list`): list of spin_m to create the covariance matrix
        """
        base_spin_n = np.array(spin_n_basis)
        base_spin_m = np.array(spin_m_basis)
        if np.all(base_spin_m == 0):
            self.use_hwp = False
        else:
            self.use_hwp = True
        # weight 1 only for the real (0,0) component; every other component comes in a
        # complex-conjugate pair and is halved in SignalFields.build_linear_system.
        # (Checking spin_n alone mis-weighted n=0, m!=0 components such as an HWP-IP
        # template at (0,+-4).)
        waits = np.array(
            [1.0 if (n == 0 and m == 0) else 0.5 for n, m in zip(base_spin_n, base_spin_m)]
        )
        spin_n_mat = base_spin_n[:, np.newaxis] - base_spin_n[np.newaxis, :]
        spin_m_mat = base_spin_m[:, np.newaxis] - base_spin_m[np.newaxis, :]
        if self.use_hwp is True:
            spin_n_mat = -spin_n_mat
            spin_m_mat = -spin_m_mat
        wait_mat = np.abs(waits[np.newaxis, :]) * np.abs(waits[:, np.newaxis])
        covmat = np.zeros(
            [len(base_spin_n), len(base_spin_m), self.npix], dtype=complex
        )
        for i in range(len(base_spin_n)):
            for j in range(len(base_spin_m)):
                covmat[i, j, :] = (
                    self.get_xlink(spin_n_mat[i, j], spin_m_mat[i, j]) * wait_mat[i, j]
                )
        model_covmat = sp.zeros(len(base_spin_n), len(base_spin_m))
        for i in range(len(base_spin_n)):
            for j in range(len(base_spin_m)):
                w = sp.Rational(wait_mat[i, j])
                n = spin_n_mat[i, j]
                m = spin_m_mat[i, j]
                if n == 0 and m == 0:
                    h = sp.symbols("1")
                else:
                    h = sp.symbols(rf"{{}}_{{{n}\,{m}}}\tilde{{h}}")
                model_covmat[i, j] = w * h
        self.model_covmat = model_covmat
        return covmat

    def map_make(self, signal_fields: SignalFields, only_iqu=True, show_eq=False):
        """Get the output map by solving the linear equation :math:`Ax=b`
        This operation gives us an equivalent result of the simple binning map-making approach.

        Args:
            signal_fields (:class:`.SignalFields`): signal fields data of the detector

            only_iqu (`bool`): if `True`, return only I, Q and U map

            show_eq (`bool`): if `True`, display the equation

        If `only_iqu` is `True`, the output map has [3, `npix`] shape.
        If `only_iqu` is `False`, the output map has [len(`signal_fields.spin_n_basis`), `npix`] shape.

        The pixels to solve are the observed pixels with HWP, and the pixels whose
        cross-link :math:`|{}_{2}\\tilde{h}|` is below `xlink_threshold` without HWP. Among
        them, a pixel whose system is ill-conditioned (reciprocal condition number at or
        below ``RCOND_MIN`` = 1e-12, e.g. a pixel seen with too few distinct angles) is not
        solved, with a warning. The pixels that are not solved are 0.

        Returns:
            output (`np.ndarray` or :class:`.SignalFields`): numpy array of maps[3,`npix`] or signal fields
        """
        assert (
            signal_fields.coupled_fields is not None
        ), "No coupled field in the SignalFields.coupled_fields"
        if np.all(np.asarray(signal_fields.spin_m_basis) == 0):
            self.use_hwp = False
        else:
            self.use_hwp = True
        spin_n_basis = np.array(signal_fields.spin_n_basis)
        spin_m_basis = np.array(signal_fields.spin_m_basis)
        b = signal_fields.coupled_fields
        x = np.zeros_like(b)
        if self.use_hwp is True:
            pol_idx = np.where((spin_n_basis == -2) & (spin_m_basis == 4))[0][0]
            A = self.create_covmat(
                spin_n_basis,
                spin_m_basis,
            )
            mask = self.hitmap != 0
        else:
            pol_idx = np.where((spin_n_basis == 2) & (spin_m_basis == 0))[0][0]
            A = self.create_covmat(
                spin_n_basis,
                spin_m_basis,
            )
            xlink2 = np.abs(self.get_xlink(2, 0))
            mask = xlink2 < self.xlink_threshold
        singular = 0
        for i in np.flatnonzero(_solvable_pixels(A, mask)):
            try:
                x[:, i] = np.linalg.solve(A[:, :, i], b[:, i])
            except np.linalg.LinAlgError:
                singular += 1
        if singular:
            warnings.warn(
                f"{singular} of the pixels to solve have a singular system and are "
                "not solved (set to 0)"
            )
        if only_iqu:
            temp_idx = next(
                (
                    i
                    for i, (n, m) in enumerate(zip(spin_n_basis, spin_m_basis))
                    if n == 0 and m == 0
                ),
                None,
            )
            output = np.array(
                [
                    x[temp_idx].real
                    if temp_idx is not None
                    else np.zeros_like(x[pol_idx].real),
                    x[pol_idx].real,
                    x[pol_idx].imag,
                ]
            )
        else:
            fields = [
                Field(
                    x[i],
                    spin_n=signal_fields.spin_n_basis[i],
                    spin_m=signal_fields.spin_m_basis[i],
                )
                for i in range(len(signal_fields.spin_n_basis))
            ]
            output = SignalFields(fields)
            output.spin_n_basis = signal_fields.spin_n_basis
            output.spin_m_basis = signal_fields.spin_m_basis
            output.field_name = signal_fields.field_name
        map_maker = rf"{sp.latex(signal_fields.solved_vector)} = {sp.latex(self.model_covmat)}^{{-1}} {sp.latex(signal_fields.model_vector)}"

        self.map_maker = map_maker
        if show_eq is True:
            display(Math(map_maker))
        return output

    def generate_noise_pdf(
        self,
        imo=None,
        net_ukrts=None,
        return_pdf=False,
        scale=1.0,
    ):
        """Generate probability density function (PDF) of the noise.
        The function store the noise PDF in the `self.noise_pdf` attribute.

        Args:
            imo (`Imo`): IMo object which contains the instrument information given by the `litebird_sim`

            net_ukrts (`float`): net sensitivity of the detector in uK√s

            return_pdf (`bool`): if True, return the noise PDF

            scale (`float`): scale factor to adjust the noise PDF.
            When the defferential detection is performed, it should be scale = 2.0.
        """
        channel = self.channel
        hitmap_tmp = self.hitmap.copy()
        hitmap_tmp[hitmap_tmp == 0] = 1  # avoid zero division
        if channel:
            assert imo is not None, "imo is required when channel is given"
            inst = get_instrument_table(imo, imo_version="v2")
            net_detector_ukrts = inst.loc[
                inst["channel"] == channel, "net_detector_ukrts"
            ].values[0]
            net_channel_ukrts = inst.loc[
                inst["channel"] == channel, "net_channel_ukrts"
            ].values[0]

            sigma_i = (
                net_detector_ukrts
                * np.sqrt(self.sampling_rate)
                / np.sqrt(scale * hitmap_tmp)
            )
            sigma_i *= np.sign(self.hitmap)
            sigma_p = sigma_i * np.sqrt(2.0)
            self.net_channel_ukrts = net_channel_ukrts
        else:
            assert (
                net_ukrts is not None
            ), "net_ukrts is required when channel is not given"
            net_detector_ukrts = net_ukrts
            sigma_i = (
                net_detector_ukrts
                * np.sqrt(self.sampling_rate)
                / np.sqrt(scale * hitmap_tmp)
            )
            sigma_i *= np.sign(self.hitmap)
            sigma_p = sigma_i * np.sqrt(2.0)
        self.net_detector_ukrts = net_detector_ukrts
        self.noise_pdf = np.array([sigma_i, sigma_p])
        if return_pdf:
            return self.noise_pdf

    def generate_noise(
        self, spin_n_basis: list, spin_m_basis: list, seed=None
    ) -> np.ndarray:
        """Generate observed noise map with the noise PDF.
        Before using this method, you should generate the noise PDF by :meth:`.ScanFields.generate_noise_pdf` method.
        The noise is 0 in the pixels that :meth:`.ScanFields.map_make` does not solve.

        Args:
            spin_n_basis (`list`): list of spin_n to create the covariance matrix

            spin_m_basis (`list`): list of spin_m to create the covariance matrix

            seed (`int`): seed for the random number generator

        Returns:
            noise (`np.ndarray` [3, `npix`]): noise map
        """
        assert (
            self.noise_pdf is not None
        ), "Generate noise pdf first by `ScanFields.generate_noise_pdf()` method."
        spin_n_basis = np.array(spin_n_basis)
        spin_m_basis = np.array(spin_m_basis)
        xlink2 = np.abs(self.get_xlink(2, 0))
        if np.all(spin_m_basis == 0):
            self.use_hwp = False
        else:
            self.use_hwp = True

        cov = self.create_covmat(
            spin_n_basis,
            spin_m_basis,
        )
        covmat_inv = np.empty_like(cov)
        try:
            temp_idx = np.where((spin_n_basis == 0) & (spin_m_basis == 0))[0][0]
        except IndexError:
            temp_idx = None
        if self.use_hwp is True:
            pol_idx = np.where((spin_n_basis == -2) & (spin_m_basis == 4))[0][0]
            mask = self.hitmap != 0
        else:
            pol_idx = np.where((spin_n_basis == 2) & (spin_m_basis == 0))[0][0]
            mask = xlink2 < self.xlink_threshold
        # the pixels that map_make solves; the noise is 0 in the others
        solved = _solvable_pixels(cov, mask)
        for i in range(self.npix):
            if solved[i]:
                covmat_inv[:, :, i] = np.linalg.inv(cov[:, :, i])
            else:
                covmat_inv[:, :, i] = np.zeros_like(cov[:, :, i])
        self.covmat_inv = np.sqrt(covmat_inv)

        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()
        n_i = np.random.normal(loc=0.0, scale=self.noise_pdf[0], size=[self.npix])
        n_q = np.random.normal(loc=0.0, scale=self.noise_pdf[1], size=[self.npix])
        n_u = np.random.normal(loc=0.0, scale=self.noise_pdf[1], size=[self.npix])

        if self.use_hwp is True:
            n_i[self.hitmap == 0] = 0.0
            n_q[self.hitmap == 0] = 0.0
            n_u[self.hitmap == 0] = 0.0
        else:
            n_i[xlink2 > self.xlink_threshold] = 0.0
            n_q[xlink2 > self.xlink_threshold] = 0.0
            n_u[xlink2 > self.xlink_threshold] = 0.0

        if temp_idx is None:
            noise = np.array(
                [
                    np.zeros_like(n_i),
                    n_q * self.covmat_inv[pol_idx, pol_idx, :].real,
                    n_u * self.covmat_inv[pol_idx, pol_idx, :].real,
                ]
            )
        else:
            noise = np.array(
                [
                    n_i * self.covmat_inv[temp_idx, temp_idx, :].real,
                    n_q * self.covmat_inv[pol_idx, pol_idx, :].real,
                    n_u * self.covmat_inv[pol_idx, pol_idx, :].real,
                ]
            )
        return noise
