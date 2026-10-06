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
    """

    def __init__(self, alm: np.ndarray, nside: int, spin_k: float, use_hwp=False):
        self.alm = alm
        self.lmax = hp.Alm.getlmax(alm.shape[1])
        self.spin_k = spin_k
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.use_hwp = use_hwp

    def to_spin(self):
        """Convert :math:`E` and :math:`B` convention to spin convention.

        Returns:
            Spherical harmonic expansion coefficients in spin form.
        """
        new_xlm = np.zeros(self.alm.shape, np.complex128)
        new_xlm[0, :] = self.alm[0, :]
        new_xlm[1, :] = -(self.alm[1, :] + 1j * self.alm[2, :])
        new_xlm[2, :] = -(self.alm[1, :] - 1j * self.alm[2, :])
        return new_xlm

    def get_blk(self, k: float):
        """Calculate :math:`b_{lk}` values for convolution integrals.

        Args:
            k (`float`): Spin moment of the crossing angle. Must be non-negative,
            the spin :math:`-k` maps are the complex conjugate of the spin :math:`k` maps.

        Returns:
            blk (`np.ndarray`): Transfer function of the beam corresponding to spin `k`.
        """
        if self.use_hwp:
            raise NotImplementedError("HWP usage is not implemented.")
        if k < 0:
            raise ValueError("k must be non-negative")
        blk = np.zeros((3, self.lmax + 1), np.complex128)
        if k > self.lmax:
            return blk
        idx_start = hp.Alm.getidx(self.lmax, k, k)
        idx_stop = hp.Alm.getidx(self.lmax, self.lmax, k)
        ell = np.arange(self.lmax + 1)
        blm_spin = self.to_spin()
        sqrt_factor = np.sqrt(4 * np.pi / (2 * ell[k:] + 1))
        blk[:, k:] = blm_spin[:, idx_start : idx_stop + 1] * sqrt_factor
        return blk

    def get_clm(self, blk: np.ndarray, k: float):
        """Calculate spherical harmonic expansion coefficients for spin `k` maps.

        Args:
            k (`float`): Spin moment of the crossing angle.

        Returns:
            clm (`np.ndarray`): Transfer function of the beam corresponding to spin `k`.
        """
        alm_spin = self.to_spin()
        if not self.use_hwp:
            # c_lm: coefficients of the spin-k map for all m
            s0_pk_clm = hp.almxfl(alm_spin[0, :], blk[0, :])
            s2_pk_clm = hp.almxfl(alm_spin[1, :], blk[2, :]) + hp.almxfl(
                alm_spin[2, :], blk[1, :]
            )
            # (-1)^m conj(c_{l,-m}): needed for the m<0 half of the spin-k map
            s0_mk_clm = hp.almxfl(alm_spin[0, :], np.conj(blk[0, :]))
            s2_mk_clm = hp.almxfl(alm_spin[1, :], np.conj(blk[1, :])) + hp.almxfl(
                alm_spin[2, :], np.conj(blk[2, :])
            )
            clm = (
                [-(s0_pk_clm + s0_mk_clm) / 2, -(s0_pk_clm - s0_mk_clm) / (2j)],
                [-(s2_pk_clm + s2_mk_clm) / 2, -(s2_pk_clm - s2_mk_clm) / (2j)],
            )
            return clm
        else:
            raise NotImplementedError("HWP usage is not implemented.")

    def __mul__(self, other):
        """Implement map-based convolution for spins given by spin `k`.

        Returns:
            all_maps (`np.ndarray`): Convolved maps corresponding to spin `k`.
            ``all_maps[i, 0]`` is the contribution of the temperature beam and
            ``all_maps[i, 1]`` is the contribution of the polarized beam.
        """
        if not self.use_hwp:
            all_maps = np.zeros((len(self.spin_k), 2, self.npix), np.complex128)
            assert (
                self.alm[0, :].size == other.alm[0, :].size
            ), "The array sizes of alm and blm are different."
            for k_idx, k in enumerate(self.spin_k):
                blk = other.get_blk(abs(k))
                clm = self.get_clm(blk, abs(k))
                if k == 0:
                    s0_sk_map = hp.alm2map(
                        -clm[0][0], nside=self.nside, lmax=self.lmax, mmax=self.lmax
                    )
                    s2_sk_map = hp.alm2map(
                        -clm[1][0], nside=self.nside, lmax=self.lmax, mmax=self.lmax
                    )
                    all_maps[k_idx, 0, :] = s0_sk_map
                    all_maps[k_idx, 1, :] = s2_sk_map
                else:
                    s0_sk_map = hp.alm2map_spin(
                        [clm[0][0], clm[0][1]],
                        nside=self.nside,
                        spin=abs(k),
                        lmax=self.lmax,
                        mmax=self.lmax,
                    )
                    s2_sk_map = hp.alm2map_spin(
                        [clm[1][0], clm[1][1]],
                        nside=self.nside,
                        spin=abs(k),
                        lmax=self.lmax,
                        mmax=self.lmax,
                    )
                    all_maps[k_idx, 0, :] = s0_sk_map[0] + s0_sk_map[1] * 1j
                    all_maps[k_idx, 1, :] = s2_sk_map[0] + s2_sk_map[1] * 1j
                    if k < 0:
                        # spin -k map is the complex conjugate of the spin k map
                        all_maps[k_idx] = all_maps[k_idx].conj()
            return all_maps
        else:
            raise NotImplementedError("HWP usage is not implemented.")


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
