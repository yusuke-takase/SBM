# -*- encoding: utf-8 -*-
import numpy as np
import healpy as hp


class Convolver:
    """Convolver class for spin map convolution

    Attributes:
        alm (`np.ndarray`): Spherical harmonic expansion coefficients for sky or beam.
        Needs to be in the :math:`E` and :math:`B` convention of HEALPix.

        nside (`int`): Nside of the input and output map.

        spin_k (`float`): List of spin moments of the crossing angle.

        use_hwp (`bool`): Whether the observation uses HWP or not.

        mueller (`np.ndarray` or `None`): Real 4x4 Stokes matrix of the rotating
        HWP. Requires ``use_hwp=True``. None selects an ideal HWP.
    """

    def __init__(
        self, alm: np.ndarray, nside: int, spin_k: float, use_hwp=False, mueller=None
    ):
        self.alm = np.asarray(alm)
        if self.alm.ndim != 2 or self.alm.shape[0] not in (3, 4):
            raise ValueError("alm must have TEB or TEBV components")
        self.lmax = hp.Alm.getlmax(self.alm.shape[1])
        if self.lmax < 0:
            raise ValueError("alm must use the full HEALPix m-major layout")
        spins = np.asarray(spin_k)
        if (
            spins.ndim != 1
            or not np.issubdtype(spins.dtype, np.number)
            or not np.isrealobj(spins)
            or not np.all(np.isfinite(spins))
            or np.any(spins != np.floor(spins))
        ):
            raise ValueError("spin_k must be a sequence of integer spin moments")
        self.spin_k = spins.astype(int)
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.use_hwp = use_hwp
        if mueller is not None and not use_hwp:
            raise ValueError("mueller requires use_hwp=True")
        self.mueller = None if mueller is None else self._validate_mueller(mueller)

    def to_spin(self):
        """Convert :math:`E` and :math:`B` convention to spin convention.

        Returns:
            Spherical harmonic expansion coefficients in spin form.
        """
        new_xlm = np.zeros(self.alm.shape, np.complex128)
        new_xlm[0, :] = self.alm[0, :]
        new_xlm[1, :] = -(self.alm[1, :] + 1j * self.alm[2, :])
        new_xlm[2, :] = -(self.alm[1, :] - 1j * self.alm[2, :])
        if self.alm.shape[0] == 4:
            new_xlm[3, :] = self.alm[3, :]
        return new_xlm

    def get_blk(self, k: float, spin_m=0):
        """Calculate :math:`b_{lk}` values for convolution integrals.

        Args:
            k (`float`): Spin moment of the crossing angle. Must be non-negative,
                the spin :math:`-k` maps are the complex conjugate of the spin :math:`k` maps.

            spin_m (`int`): HWP harmonic. The beam order is ``k + spin_m``.
                Defaults to zero for the unmodulated convolution.

        Returns:
            blk (`np.ndarray`): Transfer function of the beam corresponding to spin `k`.
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        blk = np.zeros((self.alm.shape[0], self.lmax + 1), np.complex128)
        beam_order = k + spin_m
        q = abs(beam_order)
        if q > self.lmax:
            return blk
        idx_start = hp.Alm.getidx(self.lmax, q, q)
        idx_stop = hp.Alm.getidx(self.lmax, self.lmax, q)
        ell = np.arange(self.lmax + 1)
        blm_spin = self.to_spin()
        sqrt_factor = np.sqrt(4 * np.pi / (2 * ell[q:] + 1))
        blk[:, q:] = blm_spin[:, idx_start : idx_stop + 1] * sqrt_factor
        if beam_order < 0:
            # b^P_{l,-q}=(-1)^q conj(b^{P*}_{l,q}), and conversely.
            dual = [0, 2, 1, 3] if blk.shape[0] == 4 else [0, 2, 1]
            blk = (-1) ** q * blk[dual].conj()
        return blk

    def get_clm(self, blk: np.ndarray, k: float):
        """Calculate spherical harmonic expansion coefficients for spin `k` maps.

        Args:
            k (`float`): Spin moment of the crossing angle.

        Returns:
            clm (`np.ndarray`): Transfer function of the beam corresponding to spin `k`.
        """
        alm_spin = self.to_spin()
        # The same contraction is used after get_hwp_blk for each HWP harmonic.
        s0_pk_clm = hp.almxfl(alm_spin[0, :], blk[0, :])
        s2_pk_clm = hp.almxfl(alm_spin[1, :], blk[2, :]) + hp.almxfl(
            alm_spin[2, :], blk[1, :]
        )
        # (-1)^m conj(c_{l,-m}), needed to synthesize a complex spin map.
        s0_mk_clm = hp.almxfl(alm_spin[0, :], np.conj(blk[0, :]))
        s2_mk_clm = hp.almxfl(alm_spin[1, :], np.conj(blk[1, :])) + hp.almxfl(
            alm_spin[2, :], np.conj(blk[2, :])
        )
        if alm_spin.shape[0] == 4 and blk.shape[0] == 4:
            s0_pk_clm += hp.almxfl(alm_spin[3], blk[3])
            s0_mk_clm += hp.almxfl(alm_spin[3], np.conj(blk[3]))
        return (
            [-(s0_pk_clm + s0_mk_clm) / 2, -(s0_pk_clm - s0_mk_clm) / (2j)],
            [-(s2_pk_clm + s2_mk_clm) / 2, -(s2_pk_clm - s2_mk_clm) / (2j)],
        )

    @staticmethod
    def _validate_mueller(mueller):
        matrix = np.asarray(mueller)
        if matrix.shape != (4, 4) or not np.isrealobj(matrix):
            raise ValueError("mueller must be a real 4x4 Stokes Mueller matrix")
        matrix = np.asarray(matrix, dtype=float)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("mueller must contain only finite values")
        return matrix.copy()

    def get_hwp_blk(self, blk: np.ndarray, spin_m: int, C: np.ndarray):
        """Apply one HWP harmonic to the beam filters before get_clm.

        ``blk`` comes from ``beam.get_blk(k, spin_m)`` and already contains
        the beam order k+spin_m and the spherical-harmonic normalization.
        C is T M T^dagger, without an additional conjugation. Its indices are
        ordered (I, P, P*, V). These are the five kernels in the SBM formula.
        The sqrt(2) factors account for get_clm's unnormalized P/P* coefficients
        and its polarized contribution being divided by two at map assembly.
        """
        beam = np.zeros((4, self.lmax + 1), dtype=complex)
        beam[: blk.shape[0]] = blk
        # A spin +/-2 harmonic has no ell=0 or ell=1 coefficients.
        beam[1:3, :2] = 0.0
        b_I, b_P, b_Pstar, b_V = beam
        blk_hwp = np.zeros_like(beam)
        sqrt2 = np.sqrt(2.0)
        if spin_m == 0:
            blk_hwp[0] = C[0, 0] * b_I + C[3, 0] * b_V
            blk_hwp[1] = C[2, 2] * b_P
            blk_hwp[2] = C[1, 1] * b_Pstar
            blk_hwp[3] = C[0, 3] * b_I + C[3, 3] * b_V
        elif spin_m == 2:
            blk_hwp[0] = C[1, 0] * b_Pstar / sqrt2
            blk_hwp[1] = sqrt2 * (C[0, 2] * b_I + C[3, 2] * b_V)
            blk_hwp[3] = C[1, 3] * b_Pstar / sqrt2
        elif spin_m == -2:
            blk_hwp[0] = C[2, 0] * b_P / sqrt2
            blk_hwp[2] = sqrt2 * (C[0, 1] * b_I + C[3, 1] * b_V)
            blk_hwp[3] = C[2, 3] * b_P / sqrt2
        elif spin_m == 4:
            blk_hwp[1] = C[1, 2] * b_Pstar
        elif spin_m == -4:
            blk_hwp[2] = C[2, 1] * b_P
        else:
            raise ValueError("spin_m must be 0, +/-2 or +/-4")
        # Likewise, the polarized sky in get_clm starts at ell=2.
        blk_hwp[1:3, :2] = 0.0
        return blk_hwp

    def get_map(self, clm, k: int):
        """Synthesize one complex spin map from the E/B pair returned by get_clm."""
        if k == 0:
            sk_map = hp.alm2map(
                -clm[0], nside=self.nside, lmax=self.lmax, mmax=self.lmax
            ).astype(complex)
            # Unmodulated k=0 maps are real; an HWP harmonic at k=0 need not be.
            if np.any(clm[1]):
                sk_map += 1j * hp.alm2map(
                    -clm[1], nside=self.nside, lmax=self.lmax, mmax=self.lmax
                )
        else:
            maps = hp.alm2map_spin(
                [clm[0], clm[1]], nside=self.nside, spin=abs(k),
                lmax=self.lmax, mmax=self.lmax
            )
            sk_map = maps[0] + maps[1] * 1j
            if k < 0:
                sk_map = sk_map.conj()
        return sk_map

    def convolve_hwp(self, other, mueller=None):
        """Return ``{(n, h): F_nh}`` for a rotating, fixed Mueller matrix.

        The SBM signal is ``sum(F_nh * exp(1j*(n*psi + h*alpha)))``.
        This implements the inherited narrow-beam / fundamental-spatial-mode
        approximation: h = sigma_X - sigma_Y and the input beam order is n+h.
        C = T M T^dagger is used without an additional conjugation. Both signed
        crossing moments are returned for the absolute moments in spin_k.
        The HWP harmonics are 0, +/-2 and +/-4; identically absent harmonics are
        omitted. Missing V components are zero. The default is an ideal HWP.
        """
        if self.lmax != other.lmax:
            raise ValueError("sky and beam must have the same lmax")
        supplied = [m for m in (mueller, self.mueller, other.mueller) if m is not None]
        if supplied:
            matrix = self._validate_mueller(supplied[0])
            if any(not np.array_equal(matrix, m) for m in supplied[1:]):
                raise ValueError("conflicting Mueller matrices on sky and beam")
        else:
            matrix = np.diag([1.0, 1.0, -1.0, -1.0])
        root2 = np.sqrt(2.0)
        basis = np.array(
            [[1, 0, 0, 0], [0, 1/root2, 1j/root2, 0],
             [0, 1/root2, -1j/root2, 0], [0, 0, 0, 1]], dtype=complex
        )
        coherence = basis @ matrix @ basis.conj().T
        # Sum Q/U entries before normalization to preserve exact symmetry zeros
        # (identity / ideal HWP) independently of the BLAS implementation.
        c_pp = (matrix[1, 1] + matrix[2, 2]
                + 1j * (matrix[2, 1] - matrix[1, 2])) / 2
        c_ppstar = (matrix[1, 1] - matrix[2, 2]
                    + 1j * (matrix[2, 1] + matrix[1, 2])) / 2
        coherence[1:3, 1:3] = [[c_pp, c_ppstar], [c_ppstar.conj(), c_pp.conj()]]
        spins = np.array([0, 2, -2, 0])
        differences = spins[:, None] - spins[None, :]
        harmonics = [h for h in (0, 2, -2, 4, -4) if np.any(coherence[differences == h])]
        all_maps = {}
        for k in sorted({abs(int(k)) for k in self.spin_k}):
            for spin_m in harmonics:
                if k == 0 and spin_m < 0:
                    continue
                sk_map = np.zeros(self.npix, dtype=complex)
                if k <= self.lmax:
                    blk = other.get_blk(k, spin_m)
                    blk = self.get_hwp_blk(blk, spin_m, coherence)
                    blk[:, :k] = 0.0
                    if np.any(blk):
                        clm = self.get_clm(blk, k)
                        s0_sk_map = self.get_map(clm[0], k)
                        s2_sk_map = self.get_map(clm[1], k)
                        sk_map = s0_sk_map + s2_sk_map / 2.0
                if k == 0 and spin_m == 0:
                    sk_map = sk_map.real.astype(complex)
                all_maps[k, spin_m] = sk_map
                if k != 0 or spin_m != 0:
                    all_maps[-k, -spin_m] = sk_map.conj()
        return all_maps

    def __mul__(self, other):
        """Implement map-based convolution for spins given by spin `k`.

        Returns:
            all_maps (`np.ndarray`): Convolved maps corresponding to spin `k`.
            ``all_maps[i, 0]`` is the contribution of the temperature beam and
            ``all_maps[i, 1]`` is the contribution of the polarized beam.
            With ``use_hwp=True``, returns the dictionary of complete signal
            maps documented by :meth:`convolve_hwp` instead.
        """
        if self.use_hwp:
            return self.convolve_hwp(other)
        if not self.use_hwp:
            all_maps = np.zeros((len(self.spin_k), 2, self.npix), np.complex128)
            assert (
                self.alm[0, :].size == other.alm[0, :].size
            ), "The array sizes of alm and blm are different."
            for k_idx, k in enumerate(self.spin_k):
                blk = other.get_blk(abs(k))
                clm = self.get_clm(blk, abs(k))
                all_maps[k_idx, 0, :] = self.get_map(clm[0], k)
                all_maps[k_idx, 1, :] = self.get_map(clm[1], k)
            return all_maps


def elliptical_beam(nside: int, fwhm: float, q: float):
    """Calculate elliptical beam profile.

    Args:
        nside (`int`): Nside for the beam profile.

        fwhm (`float`): Base FWHM to make elliptical beam profile.

        q (`float`): Beam ellipticity.

    Returns:
        maps: Elliptical I, Q, and U beam profile maps.
    """
    sigma = fwhm / (2.0 * np.sqrt(2 * np.log(2.0)))
    npix = hp.nside2npix(nside)
    maps = np.zeros((3, npix), np.float64)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    # semi-axes are (sigma*q, sigma), so the unit-integral prefactor is 1/(2 pi sigma^2 q)
    result = (
        1.0
        / (2.0 * np.pi * sigma**2 * q)
        * np.exp(
            -(np.cos(phi) ** 2 + q**2 * np.sin(phi) ** 2)
            * (theta**2 / (2 * (sigma**2) * (q**2)))
        )
    )
    maps[0] = result
    maps[1] = result * np.cos(2.0 * phi)
    maps[2] = -result * np.sin(2.0 * phi)
    return maps
