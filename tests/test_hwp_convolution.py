"""Independent signal and Wigner-sum checks of the SBM HWP convention."""

import healpy as hp
import numpy as np
import pytest
from scipy.linalg import expm

from sbm import Convolver, ScanFields, SignalFields


NSIDE = 2
LMAX = 4
SPINS = np.array([0, 2, -2, 0])
DUAL = np.array([0, 2, 1, 3])
ROOT2 = np.sqrt(2.0)
T = np.array(
    [[1, 0, 0, 0], [0, 1/ROOT2, 1j/ROOT2, 0],
     [0, 1/ROOT2, -1j/ROOT2, 0], [0, 0, 0, 1]], dtype=complex
)


def rotation(angle):
    co, si = np.cos(2*angle), np.sin(2*angle)
    return np.array([[1, 0, 0, 0], [0, co, si, 0],
                     [0, -si, co, 0], [0, 0, 0, 1]])


def random_alms(seed, components=4):
    rng = np.random.default_rng(seed)
    result = rng.normal(size=(components, hp.Alm.getsize(LMAX))) + 1j*rng.normal(
        size=(components, hp.Alm.getsize(LMAX))
    )
    ell, m = hp.Alm.getlm(LMAX)
    result[:, m == 0] = result[:, m == 0].real
    # Exclude the low-ell regime of the inherited shifted-beam approximation.
    result[:, ell < 2] = 0
    return result


def pencil_beam(detector=(1.0, 1.0, 0.0, 0.0), components=4):
    """Harmonics of a unit pencil response at the north pole."""
    i, q, u, v = detector
    beam = np.zeros((components, hp.Alm.getsize(LMAX)), dtype=complex)
    for ell in range(LMAX + 1):
        scale = np.sqrt((2*ell + 1)/(4*np.pi))
        beam[0, hp.Alm.getidx(LMAX, ell, 0)] = i*scale
        if components == 4:
            beam[3, hp.Alm.getidx(LMAX, ell, 0)] = v*scale
        if ell >= 2:
            idx = hp.Alm.getidx(LMAX, ell, 2)
            beam[1, idx] = -(q - 1j*u)*scale/2
            beam[2, idx] = -1j*(q - 1j*u)*scale/2
    return beam


def sky_maps(alm):
    maps = list(hp.alm2map(alm[:3], NSIDE, lmax=LMAX))
    maps.append(hp.alm2map(alm[3], NSIDE, lmax=LMAX) if len(alm) == 4
                else np.zeros(hp.nside2npix(NSIDE)))
    return np.array(maps)


def reconstruct(fields, psi, alpha, pixels=None):
    result = np.zeros_like(psi, dtype=complex)
    for (n, h), field in fields.items():
        values = field if pixels is None else field[pixels]
        result += values*np.exp(1j*(n*psi + h*alpha))
    np.testing.assert_allclose(result.imag, 0, atol=2e-12)
    return result.real


def convolve(sky, beam, matrix=None, spins=range(LMAX + 1)):
    return Convolver(sky, NSIDE, spins, use_hwp=True, mueller=matrix) * Convolver(
        beam, NSIDE, spins, use_hwp=True
    )


def point_response(maps, detector, matrix, psi, alpha):
    return np.array([
        detector @ rotation(-a) @ matrix @ rotation(a) @ rotation(-p) @ stokes
        for stokes, p, a in zip(maps.T, psi, alpha)
    ])


def test_ideal_hwp_has_existing_sbm_polarization_sign():
    sky = random_alms(11, components=3)
    fields = convolve(sky, pencil_beam(components=3))
    maps = sky_maps(sky)
    p = maps[1] + 1j*maps[2]
    np.testing.assert_allclose(fields[2, -4], p/2, atol=2e-13)
    np.testing.assert_allclose(fields[-2, 4], p.conj()/2, atol=2e-13)
    np.testing.assert_allclose(fields[0, 0], maps[0], atol=2e-13)
    for pair, value in fields.items():
        if pair not in [(0, 0), (2, -4), (-2, 4)]:
            np.testing.assert_allclose(value, 0, atol=2e-13)


@pytest.mark.parametrize("row,col", [(r, c) for r in range(4) for c in range(4)])
def test_each_real_mueller_coupling_against_stokes_measurement(row, col):
    sky = random_alms(12)
    detector = np.array([1.0, .3, -.2, .1])
    matrix = np.zeros((4, 4))
    matrix[row, col] = 1.0
    fields = convolve(sky, pencil_beam(detector), matrix)
    rng = np.random.default_rng(13)
    psi, alpha = rng.uniform(-np.pi, np.pi, (2, hp.nside2npix(NSIDE)))
    actual = reconstruct(fields, psi, alpha)
    expected = point_response(sky_maps(sky), detector, matrix, psi, alpha)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("components", [3, 4])
def test_identity_mueller_reproduces_unmodulated_convolution(components):
    sky, beam = random_alms(15, components), random_alms(16, components)
    spins = np.arange(LMAX + 1)
    old = Convolver(sky, NSIDE, spins) * Convolver(beam, NSIDE, spins)
    fields = convolve(sky, beam, np.eye(4))
    assert all(h == 0 for n, h in fields)
    for idx, n in enumerate(spins):
        np.testing.assert_allclose(fields[n, 0], old[idx, 0] + old[idx, 1]/2,
                                   rtol=2e-12, atol=2e-12)


def test_missing_v_matches_explicit_zero_v():
    sky, beam = random_alms(17, 3), random_alms(18, 3)
    matrix = np.random.default_rng(19).normal(size=(4, 4))
    implicit = convolve(sky, beam, matrix)
    explicit = convolve(np.vstack([sky, np.zeros(sky.shape[1])]),
                        np.vstack([beam, np.zeros(beam.shape[1])]), matrix)
    assert implicit.keys() == explicit.keys()
    for pair in implicit:
        np.testing.assert_array_equal(implicit[pair], explicit[pair])


def full_components(alm, ell):
    """Independent conversion of packed real TEBV to all signed m."""
    out = np.zeros((4, 2*ell + 1), dtype=complex)
    for m in range(ell + 1):
        value = T @ alm[:, hp.Alm.getidx(LMAX, ell, m)]
        # Inputs are TEBV, whereas T acts on IQUV. Spin harmonics are -(E +/- iB).
        value[1:3] *= -1
        out[:, ell + m] = value
        if m:
            out[:, ell - m] = (-1)**m * value[DUAL].conj()
    return out


def wigner_small_d(ell, theta):
    # Rotate the angular-momentum representation directly, without spin maps.
    raising = np.zeros((2*ell + 1, 2*ell + 1))
    for idx, m in enumerate(range(-ell, ell)):
        raising[idx + 1, idx] = np.sqrt((ell - m)*(ell + m + 1))
    jy = (raising - raising.T)/(2j)
    return expm(-1j*theta*jy).real


def direct_wigner_sum(sky, beam, matrix, theta, phi, psi, alpha):
    ca = T @ rotation(-alpha) @ matrix @ rotation(alpha) @ T.conj().T
    value = 0j
    for ell in range(2, LMAX + 1):
        a, b = full_components(sky, ell), full_components(beam, ell)
        effective = np.zeros_like(b)
        for k in range(-ell, ell + 1):
            for y in range(4):
                for x in range(4):
                    source = k + SPINS[y] - SPINS[x]
                    if abs(source) <= ell:
                        effective[y, k + ell] += ca[x, y].conjugate()*b[x, source + ell]
        d = wigner_small_d(ell, theta)
        orders = np.arange(-ell, ell + 1)
        # Conventional Wigner angle is -psi_SBM.
        dstar = np.exp(1j*orders[:, None]*phi)*d*np.exp(-1j*orders[None, :]*psi)
        value += np.sum((a.T @ effective.conj())*dstar)
    assert abs(value.imag) < 2e-11
    return value.real


@pytest.mark.parametrize("identity", [False, True])
def test_asymmetric_beam_against_direct_wigner_sum(identity):
    sky, beam = random_alms(20), random_alms(21)
    matrix = np.eye(4) if identity else np.random.default_rng(22).normal(size=(4, 4))
    fields = convolve(sky, beam, matrix)
    rng = np.random.default_rng(23)
    pixels = np.array([0, 3, 9, 18, 25, 37, 44, 47])
    theta, phi = hp.pix2ang(NSIDE, pixels)
    psi, alpha = rng.uniform(-np.pi, np.pi, (2, len(pixels)))
    actual = reconstruct(fields, psi, alpha, pixels)
    expected = [direct_wigner_sum(sky, beam, matrix, t, f, p, a)
                for t, f, p, a in zip(theta, phi, psi, alpha)]
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    for (n, h), field in fields.items():
        np.testing.assert_array_equal(field.conj(), fields[-n, -h])
    # This fixture also exercises complex n=0 maps and negative odd beam orders.
    if not identity:
        assert np.max(np.abs(fields[0, 2].imag)) > 1e-3
        assert np.max(np.abs(fields[1, -4])) > 1e-3


def scan_fields(psi, alpha):
    scan = ScanFields()
    scan.nside = NSIDE
    scan.npix = hp.nside2npix(NSIDE)
    scan.spins_n = np.arange(9)
    scan.spins_m = np.arange(-8, 9, 2)
    scan.hitmap = np.full(scan.npix, psi.shape[1])
    scan.h = np.array([
        [np.mean(np.exp(-1j*(n*psi + h*alpha)), axis=1) for n in scan.spins_n]
        for h in scan.spins_m
    ]).transpose(2, 0, 1)
    return scan


@pytest.mark.parametrize("mdim", [2, 3, 5])
@pytest.mark.parametrize("kind", ["ideal", "identity", "general"])
def test_mapmaking_against_sampled_tod_least_squares(mdim, kind):
    sky = random_alms(30)
    detector = np.array([1., 1., 0., 0.])
    matrices = {
        "ideal": np.diag([1., 1., -1., -1.]),
        "identity": np.eye(4),
        "general": np.array([[.9, .04, -.03, .02], [.06, .8, -.07, .03],
                             [-.02, .05, -.7, -.04], [.01, -.03, .02, -.8]])
    }
    matrix = matrices[kind]
    rng = np.random.default_rng(31)
    psi, alpha = rng.uniform(-np.pi, np.pi, (2, hp.nside2npix(NSIDE), 64))
    scan = scan_fields(psi, alpha)
    signal = SignalFields.beam_field(
        scan, mdim, sky, pencil_beam(detector), use_hwp=True, mueller=matrix,
        spin_k=range(LMAX + 1)
    )
    actual = scan.map_make(signal)
    expected = np.zeros_like(actual)
    for pixel, stokes in enumerate(sky_maps(sky).T):
        phase = 4*alpha[pixel] - 2*psi[pixel]
        design = [np.cos(phase), np.sin(phase)]
        if mdim != 2:
            design.insert(0, np.ones_like(phase))
        if mdim == 5:
            extra = 4*alpha[pixel] - 4*psi[pixel]
            design.extend([np.cos(extra), np.sin(extra)])
        tod = point_response(np.tile(stokes[:, None], (1, 64)), detector,
                             matrix, psi[pixel], alpha[pixel])
        fit = np.linalg.lstsq(np.array(design).T, tod, rcond=None)[0]
        if mdim == 2:
            expected[1:, pixel] = fit
        else:
            expected[:, pixel] = fit[:3]
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("matrix", [np.eye(3), np.eye(4)*1j, np.full((4, 4), np.nan)])
def test_rejects_invalid_mueller_matrix(matrix):
    with pytest.raises(ValueError, match="mueller"):
        Convolver(random_alms(40), NSIDE, [0, 2], use_hwp=True, mueller=matrix)


def test_mueller_requires_hwp_mode():
    with pytest.raises(ValueError, match="use_hwp"):
        Convolver(random_alms(41), NSIDE, [0, 2], mueller=np.eye(4))
