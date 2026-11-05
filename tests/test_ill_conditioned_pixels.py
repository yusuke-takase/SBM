"""Pixels whose map-making system is ill-conditioned are not solved.

ScanFields.map_make solves the observed pixels with HWP, and the pixels whose
cross-link |h_2| is below xlink_threshold without HWP. A pixel seen with only two
distinct angles has a singular I, Q, U system although |h_2| < 1, and numpy either
raised LinAlgError or returned meaningless values. Such pixels are now 0, like the
unobserved pixels, with a warning; generate_noise does the same.
"""

import warnings

import healpy as hp
import numpy as np
import pytest

from sbm import Field, ScanFields, SignalFields

NSIDE = 2
NPIX = hp.nside2npix(NSIDE)
NSAMPLES = 40
SPINS_N = np.arange(0, 7)
SPINS_M = np.arange(-8, 9)
BAD = 5


def scan_from_samples(psi, phi):
    scan = ScanFields()
    scan.nside, scan.npix = NSIDE, NPIX
    scan.hitmap = np.full(NPIX, float(psi.shape[1]))
    scan.spins_n, scan.spins_m = SPINS_N, SPINS_M
    scan.sampling_rate = 1.0
    scan.h = np.array(
        [
            [np.mean(np.exp(-1j * (n * psi + m * phi)), axis=1) for n in SPINS_N]
            for m in SPINS_M
        ]
    ).transpose(2, 0, 1)
    return scan


def angles(hwp, seed=0, two_angles=True):
    """Random angles; the pixel BAD is seen with two distinct angles only."""
    rng = np.random.default_rng(seed)
    psi = rng.uniform(-np.pi, np.pi, (NPIX, NSAMPLES))
    phi = rng.uniform(-np.pi, np.pi, (NPIX, NSAMPLES)) if hwp else np.zeros_like(psi)
    if two_angles:
        psi[BAD] = np.where(np.arange(NSAMPLES) % 2 == 0, 0.3, 1.1)
        phi[BAD] = 0.0
    return psi, phi


def iqu_signal(scan, temp, pol, hwp):
    spin_m = -4 if hwp else 0
    pol_field = Field(pol / 2, spin_n=2, spin_m=spin_m)
    signal = SignalFields(Field(temp, spin_n=0, spin_m=0), pol_field, pol_field.conj())
    s_00 = signal.get_coupled_field(scan, spin_n_out=0, spin_m_out=0)
    s_pol = signal.get_coupled_field(scan, spin_n_out=2, spin_m_out=spin_m)
    signal.build_linear_system([s_00, s_pol, s_pol.conj()])
    return signal


def least_squares(psi, phi, temp, pol, pixels):
    out = np.zeros((3, NPIX))
    for p in pixels:
        theta = 2 * psi[p] - 4 * phi[p]
        tod = temp[p] + np.real(pol[p] * np.exp(1j * theta))
        design = np.array([np.ones(NSAMPLES), np.cos(theta), -np.sin(theta)]).T
        out[:, p] = np.linalg.lstsq(design, tod, rcond=None)[0]
    return out


@pytest.mark.parametrize("hwp", [False, True])
def test_ill_conditioned_pixel_is_not_solved(hwp):
    psi, phi = angles(hwp)
    scan = scan_from_samples(psi, phi)
    assert np.abs(scan.get_xlink(2, 0))[BAD] < scan.xlink_threshold  # in the mask
    rng = np.random.default_rng(1)
    temp = rng.standard_normal(NPIX)
    pol = rng.standard_normal(NPIX) + 1j * rng.standard_normal(NPIX)
    with pytest.warns(UserWarning, match="1 of the pixels to solve have an ill-cond"):
        actual = np.array(scan.map_make(iqu_signal(scan, temp, pol, hwp)))
    np.testing.assert_array_equal(actual[:, BAD], 0.0)
    good = np.delete(np.arange(NPIX), BAD)
    expected = least_squares(psi, phi, temp, pol, good)
    np.testing.assert_allclose(actual[:, good], expected[:, good], rtol=0, atol=1e-10)


@pytest.mark.parametrize("hwp", [False, True])
def test_noise_is_zero_in_the_pixels_that_are_not_solved(hwp):
    scan = scan_from_samples(*angles(hwp))
    scan.generate_noise_pdf(net_ukrts=1.0)
    spin_m = [0, -4, 4] if hwp else [0, 0, 0]
    with pytest.warns(UserWarning, match="ill-conditioned"):
        noise = scan.generate_noise([0, 2, -2], spin_m, seed=2)
    np.testing.assert_array_equal(noise[:, BAD], 0.0)
    assert np.all(np.isfinite(noise)) and np.all(np.delete(noise, BAD, axis=1) != 0)


@pytest.mark.parametrize("hwp", [False, True])
def test_well_conditioned_scan_raises_no_warning(hwp):
    psi, phi = angles(hwp, two_angles=False)
    scan = scan_from_samples(psi, phi)
    rng = np.random.default_rng(3)
    temp = rng.standard_normal(NPIX)
    pol = rng.standard_normal(NPIX) + 1j * rng.standard_normal(NPIX)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = np.array(scan.map_make(iqu_signal(scan, temp, pol, hwp)))
    expected = least_squares(psi, phi, temp, pol, range(NPIX))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
