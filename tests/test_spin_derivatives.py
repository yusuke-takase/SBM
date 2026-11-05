"""The covariant derivatives of spin_derivatives, used by the pointing pipelines.

A pointing offset displaces the beam rigidly: the polarization direction of the
detector is carried along the displacement, and the first-order term of the offset
involves the covariant derivatives of P. They are checked against an E-mode with a
known derivative and against central differences of the sky along geodesics, with
the polarization in the frame transported along the displacement. The derivatives of
Q and U as scalar maps, which the pipelines used before, miss the rotation of the
local basis.
"""

import healpy as hp
import numpy as np
import pytest

import sbm
from sbm import spin_derivatives

NSIDE = 16
LMAX = 12


def scalar_derivatives(maps, lmax):
    """eth I, eth P, eth_bar P with I, Q and U differentiated as scalar maps, as the
    pipelines did before (alm2map_der1 of map2alm of each map)."""
    nside = hp.npix2nside(len(maps[0]))
    d_i, d_q, d_u = (
        hp.alm2map_der1(hp.map2alm(m, lmax=lmax), nside, lmax=lmax) for m in maps
    )
    eth_i = d_i[2] - 1j * d_i[1]
    eth_p = d_q[2] + d_u[1] - 1j * (d_q[1] - d_u[2])
    eth_bar_p = d_q[2] - d_u[1] + 1j * (d_q[1] + d_u[2])
    return eth_i, eth_p, eth_bar_p


def test_is_exported():
    assert "spin_derivatives" in sbm.__all__


def test_derivatives_of_an_l2_e_mode():
    # E_{2,0} = 1: P = -2Y_20 = -c sin^2(theta). Its covariant derivatives are
    # eth P = 0 (no spin-3 harmonic at l = 2) and eth_bar P = -4i c sin cos, while
    # the derivatives of Q and U as scalar maps give +-2i c sin cos.
    lmax, nside = 4, 32
    alm = np.zeros((3, hp.Alm.getsize(lmax)), complex)
    alm[1, hp.Alm.getidx(lmax, 2, 0)] = 1.0
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    c = 0.25 * np.sqrt(15 / (2 * np.pi))
    i, q, u = hp.alm2map(alm, nside, lmax=lmax)
    np.testing.assert_allclose(q + 1j * u, -c * np.sin(theta) ** 2, atol=1e-14)
    eth_i, eth_p, eth_bar_p = spin_derivatives(alm, nside)
    np.testing.assert_allclose(eth_i, 0.0, atol=1e-14)
    np.testing.assert_allclose(eth_p, 0.0, atol=1e-13)
    sin_cos = np.sin(theta) * np.cos(theta)
    np.testing.assert_allclose(eth_bar_p, -4j * c * sin_cos, atol=1e-13)
    _, scalar_eth_p, scalar_eth_bar_p = scalar_derivatives([i, q, u], lmax)
    np.testing.assert_allclose(scalar_eth_p, 2j * c * sin_cos, atol=1e-6)
    np.testing.assert_allclose(scalar_eth_bar_p, -2j * c * sin_cos, atol=1e-6)


def test_derivatives_of_a_dipole():
    # I = cos(theta): d/dtheta = -sin(theta), so eth I = i sin(theta) = -eth_bar I
    nside, lmax = 32, 4
    alm = np.zeros((3, hp.Alm.getsize(lmax)), complex)
    alm[0, hp.Alm.getidx(lmax, 1, 0)] = np.sqrt(4 * np.pi / 3)
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    eth_i, eth_p, eth_bar_p = spin_derivatives(alm, nside)
    np.testing.assert_allclose(eth_i, 1j * np.sin(theta), atol=1e-13)
    assert not np.any(eth_p) and not np.any(eth_bar_p)


def test_temperature_derivative_equals_that_of_alm2map_der1():
    alm = band_limited_alm(2)
    eth_i, _, _ = spin_derivatives(alm, NSIDE)
    d_i = hp.alm2map_der1(alm[0], NSIDE, lmax=LMAX)
    np.testing.assert_allclose(eth_i, d_i[2] - 1j * d_i[1], rtol=0, atol=1e-12)


def test_spins_above_lmax_vanish():
    # lmax = 2 < 3: eth P has no harmonic (healpy would stop the process)
    lmax = 2
    alm = np.zeros((3, hp.Alm.getsize(lmax)), complex)
    alm[0, hp.Alm.getidx(lmax, 1, 0)] = 1.0
    alm[1, hp.Alm.getidx(lmax, 2, 0)] = 1.0
    eth_i, eth_p, eth_bar_p = spin_derivatives(alm, NSIDE)
    assert not np.any(eth_p)
    assert np.max(np.abs(eth_i)) > 0.1 and np.max(np.abs(eth_bar_p)) > 0.1


def test_rejects_alm_without_polarization():
    with pytest.raises(ValueError, match="T, E and B"):
        spin_derivatives(np.zeros((1, hp.Alm.getsize(4)), complex), NSIDE)


# ---- against central differences along geodesics, with parallel transport -------


def band_limited_alm(seed):
    rng = np.random.default_rng(seed)
    size = hp.Alm.getsize(LMAX)
    alm = rng.standard_normal((3, size)) + 1j * rng.standard_normal((3, size))
    ell, m = hp.Alm.getlm(LMAX)
    alm[:, m == 0] = alm[:, m == 0].real
    alm /= (ell + 1.0) ** 1.5
    alm[1:, ell < 2] = 0
    return alm


def local_basis(theta, phi):
    n = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    e_theta = np.array(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
    )
    e_phi = np.array([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    return n, e_theta, e_phi


def displaced(theta, phi, alpha, rho):
    """Geodesic step rho along -(sin(alpha) e_theta + cos(alpha) e_phi), and the
    angle of the transported e_theta in the local basis of the new point."""
    n, e_theta, e_phi = local_basis(theta, phi)
    t = -(np.sin(alpha) * e_theta + np.cos(alpha) * e_phi)
    v = np.cos(rho) * n + np.sin(rho) * t
    axis = np.cross(n, t, axis=0)  # the rotation that moves n to v, applied to e_theta
    w = (
        np.cos(rho) * e_theta
        + np.sin(rho) * np.cross(axis, e_theta, axis=0)
        + (1 - np.cos(rho)) * np.sum(axis * e_theta, axis=0) * axis
    )
    theta_new = np.arccos(np.clip(v[2], -1, 1))
    phi_new = np.mod(np.arctan2(v[1], v[0]), 2 * np.pi)
    _, e_theta_new, e_phi_new = local_basis(theta_new, phi_new)
    delta = np.arctan2(np.sum(w * e_phi_new, axis=0), np.sum(w * e_theta_new, axis=0))
    return theta_new, phi_new, delta


def iqu_at(alm, theta, phi):
    from ducc0.sht.experimental import synthesis_general

    loc = np.ascontiguousarray(np.stack([theta, phi], axis=1))
    i = synthesis_general(alm=alm[:1], spin=0, lmax=LMAX, loc=loc, epsilon=1e-13)[0]
    q, u = synthesis_general(
        alm=np.ascontiguousarray(alm[1:]), spin=2, lmax=LMAX, loc=loc, epsilon=1e-13
    )
    return i, q, u


def iqu_displaced(alm, theta, phi, alpha, rho):
    """I, Q, U at the displaced point, Q and U in the transported frame."""
    theta_new, phi_new, delta = displaced(theta, phi, alpha, rho)
    i, q, u = iqu_at(alm, theta_new, phi_new)
    p = (q + 1j * u) * np.exp(-2j * delta)
    return i, p.real, p.imag


def test_derivatives_are_transported_central_differences():
    pytest.importorskip("ducc0")
    alm = band_limited_alm(6)
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    h = 1e-5

    def derivative(alpha):  # along -(sin(alpha) e_theta + cos(alpha) e_phi)
        plus = np.array(iqu_displaced(alm, theta, phi, alpha, h))
        minus = np.array(iqu_displaced(alm, theta, phi, alpha, -h))
        return (plus - minus) / (2 * h)

    x, y = -derivative(np.pi / 2), -derivative(0.0)  # along theta and phi
    eth = y - 1j * x  # of each of I, Q, U (Q, U in the transported frame)
    expected = (
        eth[0],
        eth[1] + 1j * eth[2],
        np.conj(eth[1]) + 1j * np.conj(eth[2]),
    )
    actual = spin_derivatives(alm, NSIDE)
    for name, a, e in zip(("eth_i", "eth_p", "eth_bar_p"), actual, expected):
        np.testing.assert_allclose(a, e, atol=1e-7, err_msg=name)
    # from the maps (healpy.map2alm of I, Q, U), at a resolution where it is exact
    from_maps = spin_derivatives(
        hp.map2alm(hp.alm2map(alm, NSIDE, lmax=LMAX), lmax=LMAX), NSIDE
    )
    for name, a, e in zip(("eth_i", "eth_p", "eth_bar_p"), from_maps, expected):
        np.testing.assert_allclose(a, e, atol=1e-6, err_msg=name)
    # the derivatives of Q and U as scalar maps miss the rotation of the local basis
    scalar = scalar_derivatives(hp.alm2map(alm, NSIDE, lmax=LMAX), 3 * NSIDE - 1)
    for name, a, e in zip(("eth_p", "eth_bar_p"), scalar[1:], expected[1:]):
        assert np.max(np.abs(a - e)) > 1e-2 * np.max(np.abs(e)), name
