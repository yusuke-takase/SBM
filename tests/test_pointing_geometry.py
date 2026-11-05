"""Pointing models against the sky sampled at the offset pointing.

The TOD sample a band-limited sky (ducc0 at arbitrary points) at the point
displaced along the geodesic, with the polarization in the frame transported
along the displacement (the detector is displaced rigidly) and the crossing angle
of the undisplaced pointing. The first derivatives are central differences of
the same synthesis along geodesics, independent of the spin transforms of Sky;
the second-order maps are those that Sky computes from the alm. The residual
must fall as rho (sky only), rho^2 (first order) and rho^3 (second order): a
wrong direction, sign or weight of the offset, or a missing transport of the
polarization frame, breaks the hierarchy.
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
    rotation (angle of the transported e_theta in the new local basis)."""
    n, e_theta, e_phi = local_basis(theta, phi)
    t = -(np.sin(alpha) * e_theta + np.cos(alpha) * e_phi)
    v = np.cos(rho) * n + np.sin(rho) * t
    # the rotation about n x t that moves n to v, applied to e_theta
    axis = np.cross(n, t, axis=0)
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


def iqu_displaced(alm, theta, phi, alpha, rho):
    """I, Q, U at the displaced point, Q and U in the transported frame."""
    theta_new, phi_new, delta = displaced(theta, phi, alpha, rho)
    i, q, u = iqu_at(alm, theta_new, phi_new)
    p = (q + 1j * u) * np.exp(-2j * delta)
    return i, p.real, p.imag


def iqu_at(alm, theta, phi):
    from ducc0.sht.experimental import synthesis_general

    loc = np.ascontiguousarray(np.stack([theta, phi], axis=1))
    i = synthesis_general(alm=alm[:1], spin=0, lmax=LMAX, loc=loc, epsilon=1e-13)[0]
    q, u = synthesis_general(
        alm=np.ascontiguousarray(alm[1:]), spin=2, lmax=LMAX, loc=loc, epsilon=1e-13
    )
    return i, q, u


def exact_sky(alm, h=1e-5):
    """Sky with the first derivatives from central differences along geodesics."""
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    i, q, u = iqu_at(alm, theta, phi)

    def derivative(alpha):  # along -(sin(alpha) e_theta + cos(alpha) e_phi)
        plus = np.array(iqu_displaced(alm, theta, phi, alpha, h))
        minus = np.array(iqu_displaced(alm, theta, phi, alpha, -h))
        return (plus - minus) / (2 * h)

    x, y = -derivative(np.pi / 2), -derivative(0.0)  # along theta and phi
    eth = y - 1j * x  # of each of I, Q, U (Q, U in the transported frame)
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
    true = response(iqu_displaced(alm, theta, phi_sky, alpha, RHO), 2 * psi - 4 * phi)
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
    d_t = response(iqu_displaced(alm, theta, phi_sky, psi + chi_t, rho_t), 2 * psi)
    d_b = response(iqu_displaced(alm, theta, phi_sky, psi_b + chi_b, rho_b), 2 * psi_b)
    base = sy.sky_signal(sky, pair_difference=True)
    models = [
        base,
        base + sy.diff_pointing(sky, rho_t, chi_t, rho_b, chi_b),
        base + sy.diff_pointing(sky, rho_t, chi_t, rho_b, chi_b, second_order=True),
    ]
    hierarchy((d_t - d_b) / 2, models, pixels, psi, np.zeros_like(psi))


def test_sky_derivatives_are_the_covariant_ones():
    alm = band_limited_alm(6)
    exact = exact_sky(alm)
    sky = Sky.from_alm(alm, NSIDE)
    for name in ("eth_i", "eth_p", "eth_bar_p"):
        np.testing.assert_allclose(
            getattr(sky, name), getattr(exact, name), atol=1e-7, err_msg=name
        )
    # from the maps alone (healpy.map2alm of I, Q, U), at a resolution where it is exact
    from_maps = Sky.from_maps(*hp.alm2map(alm, NSIDE, lmax=LMAX), lmax=LMAX)
    np.testing.assert_allclose(from_maps.eth_p, exact.eth_p, atol=1e-6)


def test_second_order_maps_without_transport_are_the_local_frame():
    # transport=False: Q and U of the displaced points in their own local basis
    from sbm.sky import second_order_maps

    alm = band_limited_alm(7)
    transported = second_order_maps(alm, NSIDE)
    local = second_order_maps(alm, NSIDE, transport=False)
    scale = np.max(np.abs(local["lap_i"]))
    np.testing.assert_allclose(transported["lap_i"], local["lap_i"], atol=1e-4 * scale)
    assert np.max(np.abs(transported["lap_p"] - local["lap_p"])) > 1e-3 * np.max(
        np.abs(local["lap_p"])
    )


def test_second_order_maps_are_transported_second_differences():
    # independent of the spin transforms of Sky: second differences of the exact
    # sky along geodesics, with the polarization in the transported frame
    alm = band_limited_alm(8)
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    h = 1e-3
    s0 = np.array(iqu_at(alm, theta, phi))

    def second_difference(alpha):
        plus = np.array(iqu_displaced(alm, theta, phi, alpha, h))
        minus = np.array(iqu_displaced(alm, theta, phi, alpha, -h))
        return (plus + minus - 2 * s0) / 2

    d0, d45, d90 = (second_difference(a) for a in (0.0, np.pi / 4, np.pi / 2))
    lap = 2 * (d0 + d90) / h**2
    a = 2 * (d0 - d90) / h**2 + 1j * (lap - 4 * d45 / h**2)
    expected = {
        "eth2_i": a[0],
        "lap_i": lap[0],
        "eth2_p": a[1] + 1j * a[2],
        "eth_bar2_p": np.conj(a[1]) + 1j * np.conj(a[2]),
        "lap_p": lap[1] + 1j * lap[2],
    }
    so = Sky.from_alm(alm, NSIDE).second_order
    for key, value in expected.items():
        np.testing.assert_allclose(
            so[key], value, atol=1e-4 * np.max(np.abs(value)), err_msg=key
        )


def test_derivatives_of_a_sky_below_the_spin_of_the_transforms():
    # lmax = 2 < 3: the spin-3 and spin-4 maps vanish (healpy would stop)
    lmax = 2
    alm = np.zeros((3, hp.Alm.getsize(lmax)), complex)
    alm[0, hp.Alm.getidx(lmax, 1, 0)] = 1.0
    alm[1, hp.Alm.getidx(lmax, 2, 0)] = 1.0
    sky = Sky.from_alm(alm, NSIDE)
    np.testing.assert_allclose(sky.eth_p, 0.0, atol=1e-14)
    assert np.max(np.abs(sky.eth_i)) > 0.1 and np.max(np.abs(sky.eth_bar_p)) > 0.1
    np.testing.assert_allclose(sky.second_order["eth2_p"], 0.0, atol=1e-14)
    one = Sky.from_maps(*hp.alm2map(alm, 1, lmax=lmax), lmax=lmax)  # nside 1
    assert np.all(np.isfinite(one.eth_i)) and np.all(np.isfinite(one.eth_bar_p))
