# -*- encoding: utf-8 -*-
import numpy as np
import healpy as hp


class Convolver:
    """Class to for the beam convolution"""

    def __init__(self, alm: np.ndarray, nside: int, spin_k: float, use_hwp=False):
        """Initialize the class with alm or blm and nside spin data

        Args:
            alm (np.ndarray): Spherical harmonic expansion coefficients for sky or beam

            nside (int): Nside of the in and out map

            spin_k (float): list of spin number of the crossing angle

            use_hwp (bool): whether the observation uses HWP or not
        """
        self.alm = alm
        self.lmax = hp.Alm.getlmax(alm[0, :].size)
        self.spin_k = spin_k
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        self.use_hwp = use_hwp

    def eb2spin(self):
        """Converte from alm EB format to spin format

        Returns:
            Spherical harmonic expansion coefficients in spin form
        """
        new_xlm = np.zeros(self.alm.shape, np.complex128)
        new_xlm[0, :] = self.alm[0, :]
        new_xlm[1, :] = -(self.alm[1, :] + 1j * self.alm[2, :])
        new_xlm[2, :] = -(self.alm[1, :] - 1j * self.alm[2, :])
        return new_xlm

    def make_blk(self, k):
        """Calculate blk values for convolution integrals

        Args:
            k (float): spin number of the crossing angle

        Returns:
            transfer function of the beam corresponding to spin k
        """
        idx_start = hp.Alm.getidx(self.lmax, k, k)
        idx_stop = hp.Alm.getidx(self.lmax, self.lmax, k)
        ell = np.linspace(0, self.lmax, self.lmax + 1)
        blm_spin = self.eb2spin()
        if self.use_hwp is False:
            blk = np.zeros((3, self.lmax + 1), np.complex64)
            blk[0, k:] = blm_spin[0, idx_start : idx_stop + 1] * np.sqrt(
                4 * np.pi / (2 * ell[k:] + 1)
            )
            blk[1, k:] = blm_spin[1, idx_start : idx_stop + 1] * np.sqrt(
                4 * np.pi / (2 * ell[k:] + 1)
            )
            blk[2, k:] = blm_spin[2, idx_start : idx_stop + 1] * np.sqrt(
                4 * np.pi / (2 * ell[k:] + 1)
            )
            return blk
        else:
            return NotImplemented

    def make_clm(self, blk, k):
        """Calculate spherical harmonic expansion coefficients for Spin k maps

        Args:
            k (float): spin number of the crossing angle

        Returns:
            transfer function of the beam corresponding to spin k
        """
        alm_spin = self.eb2spin()
        if self.use_hwp is False:
            s0_pk_clm = hp.almxfl(alm_spin[0, :], blk[0, :])
            s2_pk_clm = hp.almxfl(alm_spin[1, :], blk[2, :]) + hp.almxfl(
                alm_spin[2, :], blk[1, :]
            )
            s0_mk_clm = (hp.almxfl(alm_spin[0, :], blk[0, :])) * (-1) ** k
            s2_mk_clm = (
                hp.almxfl(alm_spin[1, :], np.conj(blk[1, :]))
                + hp.almxfl(alm_spin[2, :], np.conj(blk[2, :]))
            ) * (-1) ** k
            return [-(s0_pk_clm + s0_mk_clm) / 2, -(s0_pk_clm - s0_mk_clm) / (2j)], [
                -(s2_pk_clm + s2_mk_clm) / 2,
                -(s2_pk_clm - s2_mk_clm) / (2j),
            ]
        else:
            return NotImplemented

    def __mul__(self, other):
        """Implement map base convolution for spins given by spin_k

        Returns:
            Convolution map corresponding to spin k
        """
        if self.use_hwp is False:
            all_maps = np.zeros((len(self.spin_k), 2, self.npix), np.complex128)
            assert (
                self.alm[0, :].size == other.alm[0, :].size
            ), "The array sizes of alm and blm are different."
            for k_idx, k in enumerate(self.spin_k):
                blk = other.make_blk(k)
                clm = self.make_clm(blk, k)
                if k == 0:
                    s0_sk_map = hp.alm2map(
                        -clm[0][0], nside=self.nside, lmax=self.lmax, mmax=self.lmax
                    )
                    s2_sk_map = hp.alm2map(
                        -clm[1][0], nside=self.nside, lmax=self.lmax, mmax=self.lmax
                    )
                    all_maps[k_idx, 0, :] = s0_sk_map
                    all_maps[k_idx, 1, :] = s2_sk_map
                elif k > 0:
                    s0_sk_map = hp.alm2map_spin(
                        [clm[0][0], clm[0][1]],
                        nside=self.nside,
                        spin=k,
                        lmax=self.lmax,
                        mmax=self.lmax,
                    )
                    s2_sk_map = hp.alm2map_spin(
                        [clm[1][0], clm[1][1]],
                        nside=self.nside,
                        spin=k,
                        lmax=self.lmax,
                        mmax=self.lmax,
                    )
                    all_maps[k_idx, 0, :] = s0_sk_map[0] + s0_sk_map[1] * 1j
                    all_maps[k_idx, 1, :] = s2_sk_map[0] + s2_sk_map[1] * 1j
                elif k < 0:
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
                    all_maps[k_idx, 0, :] = s0_sk_map[0] - s0_sk_map[1] * 1j
                    all_maps[k_idx, 1, :] = s2_sk_map[0] - s2_sk_map[1] * 1j
            return all_maps
        else:
            return NotImplemented


def elliptical_beam(nside: int, fwhm: float, q: float):
    """Calculate elliptical beam map

    Args:
        nside (int): nside for the beam map

        fwhm (float): base fwhm to make ellptical beam

        q (float): beam ellipticity

    Returns:
        elliptical TQU beam map
    """
    sigma = fwhm / (2.0 * np.sqrt(2 * np.log(2.0)))
    npix = hp.nside2npix(nside)
    maps = np.zeros((3, npix), np.float64)
    pixs = np.linspace(0, npix - 1, npix)
    theta, phi = hp.pix2ang(nside, np.int32(pixs))
    result = (
        1.0
        / (2.0 * np.pi * sigma**2 * q**2)
        * np.exp(
            -(np.cos(phi) ** 2 + q**2 * np.sin(phi) ** 2)
            * (theta**2 / (2 * (sigma**2) * (q**2)))
        )
    )
    maps[0] = result
    maps[1] = result * np.cos(2.0 * phi)
    maps[2] = -result * np.sin(2.0 * phi)
    return maps
