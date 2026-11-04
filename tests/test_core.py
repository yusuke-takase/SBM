"""HMaps, SignalModel, Basis, Sky and the map-maker, against TOD binned by least squares."""

from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import pytest

from sbm.basis import Basis
from sbm.hmaps import HMaps
from sbm.map_making import map_make, noise_covariance, noise_maps, orientation_matrix
from sbm.signal_model import SignalModel
from sbm.sky import Sky

from .tod import least_squares, random_samples, tod_from_components

NSIDE = 2
NPIX = hp.nside2npix(NSIDE)
COUPLES = [(n, m) for n in range(0, 9) for m in range(-12, 13, 4)]


def random_map(rng, complex_=True):
    x = rng.standard_normal(NPIX)
    return x + 1j * rng.standard_normal(NPIX) if complex_ else x


# --------------------------------------------------------------------- #
# HMaps


def test_from_samples_is_the_mean_of_the_phases():
    pixels, psi, phi = random_samples(NSIDE, 16, seed=1, hwp=True)
    hm = HMaps.from_samples(NSIDE, pixels, psi, phi, couples=[(2, -4), (0, 4)])
    for n, m in [(2, -4), (-2, 4), (0, 4), (0, -4)]:
        expected = [
            np.mean(np.exp(-1j * (n * psi + m * phi))[pixels == p]) for p in range(NPIX)
        ]
        np.testing.assert_allclose(hm.h(n, m), expected, atol=1e-14)
    np.testing.assert_array_equal(hm.h(0, 0), 1.0)
    np.testing.assert_array_equal(hm.hits, 16)
    with pytest.raises(KeyError, match="not in these h-maps"):
        hm.h(1, 0)


def test_save_and_load_follow_the_litebird_sim_convention(tmp_path):
    # litebird_sim: h_{n,m} = <exp(+i(n psi' + m chi))> with psi' = -psi, chi = phi
    pixels, psi, phi = random_samples(NSIDE, 16, seed=2, hwp=True)
    psi_lbs = -psi
    path = tmp_path / "h_maps_det_a.h5"
    with h5py.File(path, "w") as f:
        f.attrs.update(
            coordinate_system="Galactic",
            det="a",
            detector_split="full",
            time_split="full",
            duration_s=10.0,
            sampling_rate_Hz=19.0,
        )
        f["0,0/Re"] = np.bincount(pixels, minlength=NPIX).astype(float)
        f["0,0/Im"] = np.zeros(NPIX)
        for n, m in [(2, 4), (4, 0), (-1, 4)]:
            v = [
                np.mean(np.exp(1j * (n * psi_lbs + m * phi))[pixels == p])
                for p in range(NPIX)
            ]
            f[f"{n},{m}/Re"] = np.real(v)
            f[f"{n},{m}/Im"] = np.imag(v)
    hm = HMaps.load(path)
    expected = HMaps.from_samples(
        NSIDE, pixels, psi, phi, couples=[(2, -4), (4, 0), (1, 4)]
    )
    assert hm.couples == expected.couples
    for c in expected.couples:
        np.testing.assert_allclose(hm.h(*c), expected.h(*c), atol=1e-14)
    assert (hm.name, hm.duration_s, hm.sampling_rate_hz) == ("a", 10.0, 19.0)

    again = HMaps.load(hm.save(tmp_path / "copy.h5"))
    for c in hm.couples:
        np.testing.assert_array_equal(again.h(*c), hm.h(*c))
    np.testing.assert_array_equal(again.hits, hm.hits)


def test_load_legacy_reads_the_couples_of_get_xlink():
    path = Path(__file__).parent / "nside_32_boresight_hwp.h5"
    hm = HMaps.load_legacy(path)
    with h5py.File(path) as f:
        stored = np.nan_to_num(f["h"][()], nan=0.0)
        spins_n = list(f["toml/spin_n"][()])
        spins_m = list(f["toml/spin_m"][()])
        hits = f["hitmap"][()]
    np.testing.assert_array_equal(hm.hits, hits)
    for n in spins_n[1:]:
        for m in spins_m:
            np.testing.assert_array_equal(
                hm.h(n, m), stored[:, spins_m.index(m), spins_n.index(n)]
            )
            np.testing.assert_array_equal(
                hm.h(-n, -m), np.conj(stored[:, spins_m.index(m), spins_n.index(n)])
            )
    for m in spins_m:
        if m != 0:
            np.testing.assert_array_equal(
                hm.h(0, m), np.conj(stored[:, spins_m.index(-m), 0])
            )


# --------------------------------------------------------------------- #
# SignalModel and Basis


def test_signal_model_merges_conjugate_couples():
    rng = np.random.default_rng(3)
    s = random_map(rng)
    model = SignalModel({(2, 0): s}) + SignalModel({(-2, 0): np.conj(s)})
    np.testing.assert_array_equal(model.component(2, 0), 2 * s)
    np.testing.assert_array_equal(model.component(-2, 0), 2 * np.conj(s))
    assert model.couples == [(2, 0)]
    with pytest.raises(ValueError, match="real"):
        SignalModel({(0, 0): s})
    assert sum([model, model]).couples == [(2, 0)]


def test_basis_layout():
    basis = Basis.iqu(hwp=True).extend([(-1, 0)])
    assert basis.couples == ((0, 0), (2, -4), (1, 0))
    assert basis.full == [(0, 0), (2, -4), (-2, 4), (1, 0), (-1, 0)]
    np.testing.assert_array_equal(basis.weights, [1, 0.5, 0.5, 0.5, 0.5])
    assert basis.polarization == (2, -4)
    assert Basis.qu(hwp=False).dimension == 2


# --------------------------------------------------------------------- #
# Map-making against TOD


def _random_model(rng, couples):
    comps = {c: random_map(rng, complex_=c != (0, 0)) for c in couples}
    return comps, SignalModel(comps)


CASES = [
    # (hwp, model couples, basis)
    (False, [(0, 0), (2, 0)], Basis.iqu(hwp=False)),
    (False, [(0, 0), (1, 0), (2, 0), (3, 0)], Basis.qu(hwp=False)),
    (False, [(0, 0), (1, 0), (2, 0), (3, 0)], Basis.iqu(hwp=False).extend([(1, 0)])),
    (True, [(0, 0), (2, -4), (1, 0), (0, 4)], Basis.iqu(hwp=True)),
    (
        True,
        [(0, 0), (2, -4), (1, -4), (3, -4), (1, 0)],
        Basis.iqu(hwp=True).extend([(1, 0), (0, 4)]),
    ),
]


@pytest.mark.parametrize("hwp,couples,basis", CASES)
def test_map_make_is_the_least_squares_fit_of_the_tod(hwp, couples, basis):
    rng = np.random.default_rng(4)
    pixels, psi, phi = random_samples(NSIDE, 40, seed=5, hwp=hwp)
    comps, model = _random_model(rng, couples)
    d = tod_from_components(comps, pixels, psi, phi)
    expected = least_squares(d, pixels, psi, phi, basis.couples, NPIX)

    hm = HMaps.from_samples(NSIDE, pixels, psi, phi, couples=COUPLES)
    result = map_make((hm, model), basis)
    assert result.mask.all()
    for c in basis.couples:
        np.testing.assert_allclose(
            result.component(*c), expected[c], rtol=1e-10, atol=1e-10
        )
    iqu = result.iqu
    p = expected[basis.polarization]
    np.testing.assert_allclose(iqu[1] + 1j * iqu[2], p, rtol=1e-10, atol=1e-10)


def test_several_detectors_sum_their_normal_equations():
    rng = np.random.default_rng(6)
    basis = Basis.iqu(hwp=False)
    samples = [random_samples(NSIDE, n, seed=s) for n, s in ((24, 7), (40, 8))]
    pairs, tods = [], []
    for pixels, psi, phi in samples:
        comps, model = _random_model(rng, [(0, 0), (1, 0), (2, 0)])
        tods.append(tod_from_components(comps, pixels, psi, phi))
        pairs.append(
            (HMaps.from_samples(NSIDE, pixels, psi, phi, couples=COUPLES), model)
        )
    pixels, psi, phi = (np.concatenate(x) for x in zip(*samples))
    expected = least_squares(
        np.concatenate(tods), pixels, psi, phi, basis.couples, NPIX
    )
    result = map_make(pairs, basis)
    for c in basis.couples:
        np.testing.assert_allclose(
            result.component(*c), expected[c], rtol=1e-10, atol=1e-10
        )

    # with the same model, the combined h-maps give the same maps
    same = [(hm, pairs[0][1]) for hm, _ in pairs]
    combined = map_make((HMaps.combine([hm for hm, _ in pairs]), pairs[0][1]), basis)
    np.testing.assert_allclose(
        combined.solution, map_make(same, basis).solution, rtol=1e-12, atol=1e-12
    )


def test_unobserved_and_singular_pixels_are_not_solved():
    rng = np.random.default_rng(9)
    pixels, psi, phi = random_samples(NSIDE, 8, seed=10)
    keep = pixels != 3
    psi = np.where(pixels == 5, 0.7, psi)  # one direction only in pixel 5
    hm = HMaps.from_samples(NSIDE, pixels[keep], psi[keep], phi[keep], couples=COUPLES)
    _, model = _random_model(rng, [(0, 0), (2, 0)])
    result = map_make((hm, model), Basis.iqu(hwp=False))
    assert not result.mask[3] and not result.mask[5] and result.mask.sum() == NPIX - 2
    assert np.all(np.isnan(result.iqu[:, [3, 5]]))
    with pytest.raises(KeyError, match="lack"):
        map_make(
            (HMaps.from_samples(NSIDE, pixels, psi, phi, couples=[(2, 0)]), model),
            Basis.iqu(hwp=False),
        )


# --------------------------------------------------------------------- #
# White noise


def test_noise_covariance_is_the_covariance_of_the_least_squares_fit():
    pixels, psi, phi = random_samples(NSIDE, 30, seed=11, hwp=True)
    hm = HMaps.from_samples(NSIDE, pixels, psi, phi, couples=COUPLES)
    basis = Basis.iqu(hwp=True).extend([(0, 4)])
    net, fs = 30.0, 19.0
    cov = noise_covariance(hm, basis, net, fs)
    # exact covariance of the least-squares estimator: sigma^2 (X^T X)^{-1} in the
    # real unknowns (I, Re P, Im P, Re Z, Im Z), mapped to y = (I, P, P*, Z, Z*)
    sigma2 = net**2 * fs
    for p in (0, 17):
        sel = pixels == p
        cols = [np.ones(sel.sum())]
        for n, m in basis.couples[1:]:
            theta = n * psi[sel] + m * phi[sel]
            cols.extend([np.cos(theta), -np.sin(theta)])
        x = np.array(cols).T
        cov_r = sigma2 * np.linalg.inv(x.T @ x)
        t = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 1j, 0, 0],
                [0, 1, -1j, 0, 0],
                [0, 0, 0, 1, 1j],
                [0, 0, 0, 1, -1j],
            ]
        )
        np.testing.assert_allclose(
            cov[p], t @ cov_r @ t.conj().T, rtol=1e-10, atol=1e-8
        )


def test_noise_maps_have_the_noise_covariance():
    pixels, psi, phi = random_samples(NSIDE, 30, seed=12)
    hm = HMaps.from_samples(NSIDE, pixels, psi, phi, couples=COUPLES)
    basis = Basis.iqu(hwp=False)
    cov = noise_covariance(hm, basis, 10.0, 5.0)
    rng = np.random.default_rng(13)
    draws = np.array([noise_maps(hm, basis, 10.0, 5.0, rng).iqu for _ in range(4000)])
    # empirical covariance of (I, Q, U) in one pixel
    p = 7
    emp = np.cov(draws[:, :, p].T)
    t = np.array([[1, 0, 0], [0, 1, 1j], [0, 1, -1j]])
    t_inv = np.linalg.inv(t)
    expected = (t_inv @ cov[p] @ t_inv.conj().T).real
    np.testing.assert_allclose(
        emp, expected, rtol=0.1, atol=0.1 * np.abs(expected).max()
    )


def test_orientation_matrix_of_an_ideal_scan():
    # uniform psi: h_{n,0} = 0 for n != 0, so A = diag(1, 1/4, 1/4)
    n = 16
    psi = np.tile(np.arange(n) * 2 * np.pi / n, NPIX)
    pixels = np.repeat(np.arange(NPIX), n)
    hm = HMaps.from_samples(NSIDE, pixels, psi, couples=COUPLES)
    a = orientation_matrix(hm, Basis.iqu(hwp=False))
    np.testing.assert_allclose(
        a, np.broadcast_to(np.diag([1, 0.25, 0.25]), a.shape), atol=1e-14
    )


# --------------------------------------------------------------------- #
# Sky


def test_sky_derivatives_of_a_dipole():
    # checks the signs of the convention; lmax=4 keeps the aliasing of map2alm small
    nside = 64
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    i = np.cos(theta)  # d/dtheta = -sin(theta), d/dphi = 0
    sky = Sky.from_maps(i, np.zeros_like(i), i, lmax=4)  # Q = 0, U = cos(theta)
    np.testing.assert_allclose(sky.eth_i, 1j * np.sin(theta), atol=1e-3)
    np.testing.assert_allclose(sky.eth_bar_i, -1j * np.sin(theta), atol=1e-3)
    # eth P = (Q_phi + U_theta) - i (Q_theta - U_phi) = U_theta
    # eth_bar P = (Q_phi - U_theta) + i (Q_theta + U_phi) = -U_theta
    np.testing.assert_allclose(sky.eth_p, -np.sin(theta), atol=1e-3)
    np.testing.assert_allclose(sky.eth_bar_p, np.sin(theta), atol=1e-3)


def test_sky_accepts_known_derivatives():
    rng = np.random.default_rng(14)
    i, q, u = rng.standard_normal((3, NPIX))
    e = rng.standard_normal((3, NPIX)) + 1j * rng.standard_normal((3, NPIX))
    sky = Sky.from_maps(i, q, u, eth_i=e[0], eth_p=e[1], eth_bar_p=e[2])
    np.testing.assert_array_equal([sky.eth_i, sky.eth_p, sky.eth_bar_p], e)
    with pytest.raises(ValueError, match="both"):
        Sky.from_maps(i, q, u, eth_p=e[1])
