"""Pointing models against the sky sampled at the offset pointing.

The TOD sample a band-limited sky (ducc0 at arbitrary points) at the point
displaced along the geodesic, with Q and U taken in the local frame of that
point and the crossing angle of the undisplaced pointing, as a TOD simulation
does. The first derivatives of I, Q and U are central differences of the same
synthesis (Q and U are not band-limited as scalar fields, so their healpy
derivatives would add an error unrelated to the models); the second-order maps
are those that Sky computes from the alm. The residual must fall as rho (sky
only), rho^2 (first order) and rho^3 (second order): a wrong direction, sign or
weight of the offset breaks the hierarchy.
"""

import healpy as hp
import numpy as np
import pytest

from sbm import systematics as sy
from sbm.sky import Sky

ducc0 = pytest.importorskip("ducc0")

NSIDE = 16
LMAX = 12
RHO = 2e-3


def band_limited_alm(seed):
    rng = np.random.default_rng(seed)
    size = hp.Alm.getsize(LMAX)
    alm = rng.standard_normal((3, size)) + 1j * rng.standard_normal((3, size))
    ell, m = hp.Alm.getlm(LMAX)
    alm[:, m == 0] = alm[:, m == 0].real
    alm /= (ell + 1.0) ** 1.5
    alm[1:, ell < 2] = 0
    return alm


def displaced(theta, phi, alpha, rho):
    """Geodesic step rho along -(sin(alpha) e_theta + cos(alpha) e_phi)."""
    n = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    e_theta = np.array(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
    )
    e_phi = np.array([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    v = np.cos(rho) * n - np.sin(rho) * (
        np.sin(alpha) * e_theta + np.cos(alpha) * e_phi
    )
    return np.arccos(np.clip(v[2], -1, 1)), np.mod(np.arctan2(v[1], v[0]), 2 * np.pi)


def iqu_at(alm, theta, phi):
    from ducc0.sht.experimental import synthesis_general

    loc = np.ascontiguousarray(np.stack([theta, phi], axis=1))
    i = synthesis_general(alm=alm[:1], spin=0, lmax=LMAX, loc=loc, epsilon=1e-13)[0]
    q, u = synthesis_general(
        alm=np.ascontiguousarray(alm[1:]), spin=2, lmax=LMAX, loc=loc, epsilon=1e-13
    )
    return i, q, u


def exact_sky(alm, h=1e-5):
    """Sky with the first derivatives of I, Q, U from central differences."""
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    i, q, u = iqu_at(alm, theta, phi)
    d_theta = (
        np.array(iqu_at(alm, theta + h, phi)) - np.array(iqu_at(alm, theta - h, phi))
    ) / (2 * h)
    d_phi = (
        np.array(iqu_at(alm, theta, phi + h)) - np.array(iqu_at(alm, theta, phi - h))
    ) / (2 * h * np.sin(theta))
    x, y = d_theta, d_phi  # x along theta, y along phi
    eth = y - 1j * x  # of each of I, Q, U
    return Sky.from_maps(
        i,
        q,
        u,
        eth_i=eth[0],
        eth_p=eth[1] + 1j * eth[2],
        eth_bar_p=np.conj(eth[1]) + 1j * np.conj(eth[2]),
        alm=alm,
    )


def evaluate(model, pixels, psi, phi):
    d = np.zeros(pixels.size)
    for comp in model.components():
        d += (comp.map[pixels] * np.exp(1j * (comp.n * psi + comp.m * phi))).real
    return d


def samples(seed, n=8):
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(NSIDE)
    pixels = np.repeat(np.arange(npix), n)
    theta, phi_sky = hp.pix2ang(NSIDE, pixels)
    psi, phi = rng.uniform(-np.pi, np.pi, (2, pixels.size))
    return pixels, theta, phi_sky, psi, phi


def response(iqu, angle):
    i, q, u = iqu
    return i + q * np.cos(angle) - u * np.sin(angle)


def test_synthesis_matches_healpy():
    alm = band_limited_alm(1)
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    np.testing.assert_allclose(
        iqu_at(alm, theta, phi), hp.alm2map(alm, NSIDE, lmax=LMAX), atol=1e-12
    )


def hierarchy(true, models, pixels, psi, phi):
    residuals = [np.max(np.abs(true - evaluate(m, pixels, psi, phi))) for m in models]
    # each order gains about rho * ell ~ 0.03; ask for a factor of 5 at least
    assert residuals[1] < 0.2 * residuals[0], residuals
    assert residuals[2] < 0.2 * residuals[1], residuals
    return residuals


@pytest.mark.parametrize("kind", ["abs_pointing", "hwp_wedge"])
def test_single_detector_offsets(kind):
    alm = band_limited_alm(2)
    sky = exact_sky(alm)
    pixels, theta, phi_sky, psi, phi = samples(3)
    chi = 0.6
    alpha = psi + chi if kind == "abs_pointing" else psi - phi + chi
    true = response(
        iqu_at(alm, *displaced(theta, phi_sky, alpha, RHO)), 2 * psi - 4 * phi
    )
    base = sy.sky_signal(sky, hwp=True)
    fn = getattr(sy, kind)
    hierarchy(
        true,
        [base, base + fn(sky, RHO, chi), base + fn(sky, RHO, chi, second_order=True)],
        pixels,
        psi,
        phi,
    )


def test_pair_difference_offsets():
    alm = band_limited_alm(4)
    sky = exact_sky(alm)
    pixels, theta, phi_sky, psi, phi = samples(5)
    rho_t, chi_t, rho_b, chi_b = RHO, 0.3, 0.7 * RHO, -1.2
    psi_b = psi + np.pi / 2
    d_t = response(iqu_at(alm, *displaced(theta, phi_sky, psi + chi_t, rho_t)), 2 * psi)
    d_b = response(
        iqu_at(alm, *displaced(theta, phi_sky, psi_b + chi_b, rho_b)), 2 * psi_b
    )
    base = sy.sky_signal(sky, pair_difference=True)
    models = [
        base,
        base + sy.diff_pointing(sky, rho_t, chi_t, rho_b, chi_b),
        base + sy.diff_pointing(sky, rho_t, chi_t, rho_b, chi_b, second_order=True),
    ]
    hierarchy((d_t - d_b) / 2, models, pixels, psi, np.zeros_like(psi))
