"""Standard deviations of the noise maps of ScanFields.generate_noise.

For TOD d = I + Q cos(theta) - U sin(theta) (+ templates) with white noise of
variance sigma^2 = NET^2 f_s per sample, the least-squares fit of the N samples of a
pixel has the covariance sigma^2 (X^T X)^{-1} in the real unknowns, which is the
covariance sigma^2/N A^{-1} of the map-maker. Each of I, Q and U must have the
standard deviation of its diagonal (the correlations are not drawn). Q and U used to
have twice this standard deviation for an ideal scan.
"""

import healpy as hp
import numpy as np
import pytest

from sbm import ScanFields

NSIDE = 4
NPIX = hp.nside2npix(NSIDE)
SPINS_N = np.arange(0, 7)
SPINS_M = np.arange(-8, 9)
NET, FS = 10.0, 5.0
NDRAWS = 300

# (spin_n_basis, spin_m_basis, HWP); the polarization is at (2,0) without HWP and at
# (-2,4) with HWP, the other conjugate pairs are templates
BASES = {
    "iqu": ([0, 2, -2], [0, 0, 0], False),
    "iqu+1": ([0, 1, -1, 2, -2], [0, 0, 0, 0, 0], False),
    "qu+1": ([1, -1, 2, -2], [0, 0, 0, 0], False),
    "iqu_hwp": ([0, 2, -2], [0, -4, 4], True),
    "iqu_hwp+4f": ([0, 2, -2, 0, 0], [0, -4, 4, 4, -4], True),
}


def scan_from_samples(psi, phi):
    scan = ScanFields()
    scan.nside, scan.npix = NSIDE, NPIX
    scan.hitmap = np.full(NPIX, float(psi.shape[1]))
    scan.spins_n, scan.spins_m = SPINS_N, SPINS_M
    scan.sampling_rate = FS
    scan.h = np.array(
        [
            [np.mean(np.exp(-1j * (n * psi + m * phi)), axis=1) for n in SPINS_N]
            for m in SPINS_M
        ]
    ).transpose(2, 0, 1)
    return scan


def least_squares_std(psi, phi, spin_n, spin_m, hwp):
    """Standard deviations of I, Q and U of the least-squares fit, per pixel."""
    sigma2 = NET**2 * FS
    std = np.zeros((3, NPIX))
    sign = (
        -1 if hwp else 1
    )  # phase of the component (n, m): exp(sign i (n psi + m phi))
    pol = (-2, 4) if hwp else (2, 0)
    pairs = [pol]  # one component of each conjugate pair, the polarization first
    for n, m in zip(spin_n, spin_m):
        if (n, m) != (0, 0) and (n, m) not in pairs and (-n, -m) not in pairs:
            pairs.append((n, m))
    has_i = (0, 0) in zip(spin_n, spin_m)
    for p in range(NPIX):
        cols = [np.ones(psi.shape[1])] if has_i else []
        for n, m in pairs:
            theta = sign * (n * psi[p] + m * phi[p])
            cols.extend([np.cos(theta), -np.sin(theta)])
        x = np.array(cols).T
        var = np.diag(sigma2 * np.linalg.inv(x.T @ x))
        std[:, p] = np.sqrt(var[:3]) if has_i else [0.0, *np.sqrt(var[:2])]
    return std


def draws(scan, spin_n, spin_m):
    scan.generate_noise_pdf(net_ukrts=NET)
    return np.array(
        [scan.generate_noise(spin_n, spin_m, seed=s) for s in range(NDRAWS)]
    )


@pytest.mark.parametrize("name", BASES)
def test_noise_has_the_variance_of_the_least_squares_fit(name):
    spin_n, spin_m, hwp = BASES[name]
    rng = np.random.default_rng(1)
    psi = rng.uniform(-np.pi, np.pi, (NPIX, 30))
    phi = rng.uniform(-np.pi, np.pi, (NPIX, 30)) if hwp else np.zeros_like(psi)
    expected = least_squares_std(psi, phi, spin_n, spin_m, hwp)
    noise = draws(scan_from_samples(psi, phi), spin_n, spin_m)
    for k in range(3) if (0, 0) in zip(spin_n, spin_m) else (1, 2):
        z = noise[:, k] / expected[k]
        # pooled over the pixels (statistical error 0.2%) and per pixel (4%)
        assert abs(np.std(z) - 1.0) < 0.03, ("IQU"[k], np.std(z))
        np.testing.assert_allclose(np.std(z, axis=0), 1.0, atol=0.25)


@pytest.mark.parametrize("hwp", [False, True])
def test_ideal_scan_has_sqrt2_sigma_in_q_and_u(hwp):
    # uniform angles: A = diag(1, 1/4, 1/4), sigma_Q = sigma_U = sqrt(2) sigma_I
    grid = np.arange(8) * 2 * np.pi / 8
    psi, phi = np.meshgrid(grid, grid if hwp else [0.0])
    psi = np.tile(psi.ravel(), (NPIX, 1))
    phi = np.tile(phi.ravel(), (NPIX, 1))
    scan = scan_from_samples(psi, phi)
    spin_n, spin_m, _ = BASES["iqu_hwp" if hwp else "iqu"]
    noise = draws(scan, spin_n, spin_m)
    sigma_i = NET * np.sqrt(FS / psi.shape[1])
    np.testing.assert_allclose(
        scan.noise_pdf, [[sigma_i] * NPIX, [np.sqrt(2) * sigma_i] * NPIX]
    )
    std = np.std(noise, axis=(0, 2)) / sigma_i
    np.testing.assert_allclose(std, [1.0, np.sqrt(2), np.sqrt(2)], rtol=0.03)
