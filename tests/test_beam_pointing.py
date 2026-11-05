"""A Gaussian beam centred off the boresight is the smoothed sky with a pointing offset.

The beam is rotated rigidly on the sphere (``healpy.rotate_alm``), so its centre
moves by ``rho`` and its polarization direction is carried along: this is the
pointing offset of the pointing models, whose first-order terms use the
covariant derivatives of :class:`sbm.Sky` and whose second-order terms transport
the polarization frame. The beam convolution is exact without HWP; with a HWP it
uses the narrow-beam approximation, whose error on the offset terms falls with
the width of the beam.
"""

import healpy as hp
import numpy as np
import pytest

from sbm import beam as sbm_beam
from sbm import systematics as sy
from sbm.harmonics import SphericalHarmonics
from sbm.sky import Sky

PHI0 = 1.0  # azimuth of the beam centre in the beam frame
CHI = -PHI0 - np.pi / 2  # the same offset in the convention of the pointing models


def random_alm(lmax, seed, temperature=True):
    rng = np.random.default_rng(seed)
    ell, m = hp.Alm.getlm(lmax)
    alm = rng.standard_normal((3, ell.size)) + 1j * rng.standard_normal((3, ell.size))
    alm /= 1.0 + ell
    alm[:, m == 0] = alm[:, m == 0].real
    alm[1:, ell < 2] = 0
    if not temperature:
        alm[0] = 0
    return alm


def shifted_gaussian(fwhm, lmax, rho, detector=(1.0, 1.0, 0.0, 0.0)):
    values = sbm_beam.gaussian(fwhm, lmax, detector).values
    teb = [values[0].copy(), values[1].copy(), values[2].copy()]
    hp.rotate_alm(teb, -PHI0, rho, PHI0)  # centre at theta = rho, phi = PHI0
    return SphericalHarmonics(np.array(teb), lmax)


def largest(model):
    return max(np.max(np.abs(model.component(*c))) for c in model.couples)


def test_the_rotated_beam_is_centred_at_the_offset():
    lmax, rho = 47, np.deg2rad(20.0)
    t = hp.alm2map(shifted_gaussian(np.deg2rad(8.0), lmax, rho).values[0], 64)
    theta, phi = hp.pix2ang(64, np.argmax(t))
    assert abs(theta - rho) < hp.nside2resol(64) and abs(phi - PHI0) < 0.05


@pytest.mark.parametrize("second_order", [False, True])
def test_shifted_beam_is_smoothing_and_offset_of_a_pair(second_order):
    nside, lmax, fwhm = 16, 47, np.deg2rad(12.0)
    alm = random_alm(lmax, 5)
    smoothed = hp.smoothalm(alm, fwhm=fwhm, pol=True, inplace=False)
    sky = Sky.from_alm(SphericalHarmonics(smoothed, lmax), nside)
    b_detector = sy.beam(
        alm, sbm_beam.gaussian(fwhm, lmax, (1.0, -1.0, 0.0, 0.0)), nside, range(7)
    )
    residuals = []
    for rho in np.deg2rad([1.0, 0.5]):
        t_detector = sy.beam(alm, shifted_gaussian(fwhm, lmax, rho), nside, range(7))
        model = sy.sky_signal(sky, pair_difference=True) + sy.diff_pointing(
            sky, rho, CHI, 0.0, 0.0, second_order=second_order
        )
        residuals.append(largest(0.5 * (t_detector - b_detector) - model))
    # O(rho^2) at first order, O(rho^3) at second order
    assert residuals[0] / residuals[1] > (7.0 if second_order else 3.5), residuals


def test_shifted_beam_with_hwp_is_the_offset_within_the_narrow_beam_approximation():
    rho_over_fwhm = 0.05
    errors = []
    for fwhm_deg, nside in ((12.0, 16), (6.0, 32)):
        lmax, fwhm = 3 * nside - 1, np.deg2rad(fwhm_deg)
        alm = random_alm(lmax, 5, temperature=False)
        smoothed = hp.smoothalm(alm, fwhm=fwhm, pol=True, inplace=False)
        sky = Sky.from_alm(SphericalHarmonics(smoothed, lmax), nside)
        rho = rho_over_fwhm * fwhm
        fields = sbm_beam.convolve(
            alm, shifted_gaussian(fwhm, lmax, rho), nside, range(5), hwp=True
        )
        offset = sy.abs_pointing(sky, rho, CHI)
        errors.append(
            max(
                np.max(np.abs(fields[c] - offset.component(*c)))
                / np.max(np.abs(offset.component(*c)))
                for c in ((1, -4), (3, -4))
            )
        )
    # about 0.8 % per degree of FWHM: 10 % at 12 degrees, halved at 6 degrees
    assert errors[0] < 0.15 and errors[1] < 0.6 * errors[0], errors
