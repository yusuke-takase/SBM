"""Weights of ScanFields.create_covmat for basis components with n == 0 and m != 0.

Every basis component except (0,0) comes with its complex conjugate and is halved in
SignalFields.build_linear_system; the covariance matrix must use the same weights. The
signal here is a temperature leakage modulated by the HWP angle alone,
``epsilon T cos(4 phi - phase)``, at spin (0,+-4).
"""

from pathlib import Path

import healpy as hp
import numpy as np
import pytest

import sbm
from sbm import Field, ScanFields, SignalFields

NSIDE = 2
NSAMPLES = 200
SPINS_N = np.arange(0, 7)
SPINS_M = np.arange(-8, 9)
EPSILON, PHASE = 1e-2, 0.3


def hwp_synchronous_leakage(scan_field, temp, pol, mdim):
    """I + P at their usual spins plus epsilon T cos(4 phi - PHASE) at (0,+-4)."""
    leak = Field(EPSILON / 2 * np.exp(-1j * PHASE) * temp, spin_n=0, spin_m=4)
    pol_field = Field(pol / 2, spin_n=2, spin_m=-4)
    signal = SignalFields(Field(temp, spin_n=0, spin_m=0), pol_field, pol_field.conj(), leak, leak.conj())
    s_00 = signal.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=0)
    sp2m4 = signal.get_coupled_field(scan_field, spin_n_out=2, spin_m_out=-4)
    fields = [s_00, sp2m4, sp2m4.conj()]
    if mdim == 5:
        s_04 = signal.get_coupled_field(scan_field, spin_n_out=0, spin_m_out=4)
        fields += [s_04, s_04.conj()]
    signal.build_linear_system(fields)
    return signal


def test_weights_of_the_covariance_matrix():
    rng = np.random.default_rng(3)
    scan = ScanFields()
    scan.nside, scan.npix = NSIDE, hp.nside2npix(NSIDE)
    scan.hitmap = np.ones(scan.npix)
    scan.spins_n, scan.spins_m = SPINS_N, SPINS_M
    scan.h = rng.standard_normal((scan.npix, SPINS_M.size, SPINS_N.size)) + 0j
    cov = scan.create_covmat([0, 2, -2, 0, 0], [0, -4, 4, 4, -4])
    # diagonal: 1 for (0,0), 1/4 for every component of a conjugate pair
    np.testing.assert_array_equal(np.diagonal(cov[:, :, 0]).real, [1, 0.25, 0.25, 0.25, 0.25])


@pytest.mark.parametrize("mdim", [3, 5])
def test_map_making_matches_tod_least_squares(mdim):
    rng = np.random.default_rng(4)
    npix = hp.nside2npix(NSIDE)
    psi = rng.uniform(-np.pi, np.pi, (npix, NSAMPLES))
    phi = rng.uniform(-np.pi, np.pi, (npix, NSAMPLES))
    scan = ScanFields()
    scan.nside, scan.npix = NSIDE, npix
    scan.hitmap = np.full(npix, NSAMPLES)
    scan.spins_n, scan.spins_m = SPINS_N, SPINS_M
    scan.h = np.array(
        [[np.mean(np.exp(-1j * (n * psi + m * phi)), axis=1) for n in SPINS_N] for m in SPINS_M]
    ).transpose(2, 0, 1)
    temp = rng.standard_normal(npix)
    pol = rng.standard_normal(npix) + 1j * rng.standard_normal(npix)
    actual = np.array(scan.map_make(hwp_synchronous_leakage(scan, temp, pol, mdim)))

    expected = np.zeros_like(actual)
    for p in range(npix):
        theta = 2 * psi[p] - 4 * phi[p]
        tod = temp[p] + np.real(pol[p] * np.exp(1j * theta)) + EPSILON * temp[p] * np.cos(4 * phi[p] - PHASE)
        design = [np.ones(NSAMPLES), np.cos(theta), -np.sin(theta)]
        if mdim == 5:
            design += [np.cos(4 * phi[p]), np.sin(4 * phi[p])]
        expected[:, p] = np.linalg.lstsq(np.array(design).T, tod, rcond=None)[0][:3]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    if mdim == 5:
        # the (0,+-4) template absorbs the leakage completely
        np.testing.assert_allclose(actual, [temp, pol.real, pol.imag], rtol=0, atol=1e-12)


def test_template_removes_the_leakage_on_a_real_scan():
    sf = sbm.read_scanfiled(str(Path(__file__).parent / "nside_32_boresight_hwp.h5"))
    npix = hp.nside2npix(sf.nside)
    temp = np.random.default_rng(1).standard_normal(npix)
    zero = np.zeros(npix, dtype=complex)
    leak3 = np.array(sf.map_make(hwp_synchronous_leakage(sf, temp, zero, 3)))
    leak5 = np.array(sf.map_make(hwp_synchronous_leakage(sf, temp, zero, 5)))
    seen = sf.hitmap > 0
    p3 = np.sqrt(np.mean(leak3[1, seen] ** 2 + leak3[2, seen] ** 2))
    p5 = np.sqrt(np.mean(leak5[1, seen] ** 2 + leak5[2, seen] ** 2))
    assert p3 > 0
    assert p5 < 1e-10 * p3
