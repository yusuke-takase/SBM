"""HMaps, SignalModel, Basis, Sky and the map-maker, against TOD binned by least squares."""

from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import pytest

from sbm.basis import Basis
from sbm.hmaps import HMaps
from sbm.map_making import (
    MapMaker,
    _rhs,
    map_make,
    noise_covariance,
    noise_maps,
    orientation_matrix,
    solvable,
)
from sbm.signal_model import SignalModel
from sbm.sky import Sky

from sbm.testing import least_squares, random_samples, tod_from_components

NSIDE = 2
NPIX = hp.nside2npix(NSIDE)
COUPLES = [(n, m) for n in range(0, 9) for m in range(-12, 13, 4)]


def random_map(rng, complex_=True):
    x = rng.standard_normal(NPIX)
    return x + 1j * rng.standard_normal(NPIX) if complex_ else x


# --------------------------------------------------------------------- #
# HMaps


def test_from_samples_is_the_mean_of_the_phases():
    pixels, psi, phi = random_samples(NSIDE, 16, seed=1)
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
    pixels, psi, phi = random_samples(NSIDE, 16, seed=2)
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


def test_signal_models_are_scaled_by_numbers_only():
    rng = np.random.default_rng(5)
    model = SignalModel({(0, 0): random_map(rng, False), (2, 0): random_map(rng)})
    np.testing.assert_array_equal(
        (np.float64(2.0) * model).component(2, 0), 2 * model.component(2, 0)
    )
    with pytest.raises(TypeError, match="factor that depends on the pixel"):
        np.ones(NPIX) * model
    with pytest.raises(TypeError, match="factor that depends on the pixel"):
        model * np.ones(NPIX)
    with pytest.raises(TypeError):
        np.ones(NPIX) + model


def test_results_compare_by_identity_and_bases_are_hashable():
    hm = HMaps(
        {(2, 0): np.zeros(NPIX, complex), (4, 0): np.zeros(NPIX, complex)},
        np.ones(NPIX),
    )
    model = SignalModel({(0, 0): np.ones(NPIX), (2, 0): np.ones(NPIX, complex)})
    basis = Basis.iqu(hwp=False)
    a, b = map_make((hm, model), basis), map_make((hm, model), basis)
    assert a == a and a != b
    assert model.components()[0] != model.components()[0]  # no comparison of maps
    assert {basis: 1}[Basis.iqu(hwp=False)] == 1


def test_right_hand_side_is_the_projection_on_every_couple():
    rng = np.random.default_rng(6)
    hm = HMaps({c: random_map(rng) for c in COUPLES if c != (0, 0)}, np.ones(NPIX))
    model = SignalModel(
        {
            (0, 0): random_map(rng, False),
            (2, -4): random_map(rng),
            (1, 4): random_map(rng),
        }
    )
    basis = Basis.iqu(hwp=True).extend([(1, 0), (3, -4)])
    expected = np.stack(
        [w * model.project(hm, n, m) for w, (n, m) in zip(basis.weights, basis.full)],
        axis=1,
    )
    np.testing.assert_array_equal(_rhs(hm, model, basis), expected)


def test_solvable_does_not_depend_on_the_chunk():
    rng = np.random.default_rng(7)
    hm = HMaps(
        {c: 0.3 * random_map(rng) for c in COUPLES if c != (0, 0)}, np.ones(NPIX)
    )
    a = orientation_matrix(hm, Basis.iqu(hwp=True).extend([(1, 0), (3, -4)]))
    np.testing.assert_array_equal(solvable(a, hm.hits, chunk=5), solvable(a, hm.hits))


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
    pixels, psi, phi = random_samples(NSIDE, 40, seed=5)
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


def test_a_given_mask_excludes_the_singular_pixels():
    rng = np.random.default_rng(9)
    pixels, psi, phi = random_samples(NSIDE, 8, seed=10)
    psi = np.where(pixels == 5, 0.7, psi)  # one direction only in pixel 5
    hm = HMaps.from_samples(NSIDE, pixels, psi, phi, couples=COUPLES)
    _, model = _random_model(rng, [(0, 0), (2, 0)])
    basis = Basis.iqu(hwp=False)
    everywhere = np.ones(NPIX, bool)
    with pytest.warns(UserWarning, match="ill-conditioned"):
        result = map_make((hm, model), basis, mask=everywhere)
    assert not result.mask[5] and np.all(np.isnan(result.iqu[:, 5]))
    with pytest.warns(UserWarning, match="ill-conditioned"):
        reused = MapMaker(hm, basis, mask=everywhere)(model)
    np.testing.assert_array_equal(reused.mask, result.mask)
    with pytest.warns(UserWarning, match="ill-conditioned"):
        noise = noise_maps(hm, basis, 1.0, 1.0, rng, mask=everywhere)
    assert not noise.mask[5] and noise.mask.sum() == NPIX - 1


def test_hmaps_copy_the_hits_and_write_files_that_litebird_sim_reads(tmp_path):
    hits = np.ones(NPIX)
    hm = HMaps({(2, 0): np.zeros(NPIX, complex)}, hits, coordinate_system="G")
    hits[0] = 3.0  # the caller's array stays writable and independent
    assert hm.hits[0] == 1.0
    path = hm.save(tmp_path / "h.h5")
    with h5py.File(path) as f:
        assert f.attrs["coordinate_system"] == "Galactic"
        assert np.isnan(f.attrs["duration_s"]) and np.isnan(f.attrs["sampling_rate_Hz"])
    back = HMaps.load(path)
    assert back.duration_s is None and back.sampling_rate_hz is None


def test_saving_needs_a_known_coordinate_system(tmp_path):
    pixels, psi, phi = random_samples(NSIDE, 16, seed=4)
    hm = HMaps.from_samples(NSIDE, pixels, psi, phi, couples=[(2, -4)])
    with pytest.raises(ValueError, match="coordinate system"):
        hm.save(tmp_path / "h.h5")
    hm.coordinate_system = "E"
    back = HMaps.load(hm.save(tmp_path / "h.h5"))
    assert (back.coordinate_system, back.name) == ("Ecliptic", None)


def test_load_ignores_other_groups(tmp_path):
    hm = HMaps({(2, 0): np.full(NPIX, 0.5j)}, np.ones(NPIX), coordinate_system="G")
    path = hm.save(tmp_path / "h.h5")
    with h5py.File(path, "a") as f:
        f["provenance/version"] = 1
        f["mask"] = np.ones(NPIX)
    np.testing.assert_array_equal(HMaps.load(path).h(2, 0), hm.h(2, 0))


# --------------------------------------------------------------------- #
# White noise


def test_noise_covariance_is_the_covariance_of_the_least_squares_fit():
    pixels, psi, phi = random_samples(NSIDE, 30, seed=11)
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
    sky = Sky.from_maps(i, np.zeros_like(i), np.zeros_like(i), lmax=4)
    np.testing.assert_allclose(sky.eth_i, 1j * np.sin(theta), atol=1e-3)
    np.testing.assert_allclose(sky.eth_bar_i, -1j * np.sin(theta), atol=1e-3)


def test_polarization_derivatives_are_covariant():
    # E_{2,0} = 1: P = -2Y_20 = -c sin^2(theta). Its covariant derivatives are
    # eth P = 0 (no spin-3 harmonic at l = 2) and eth_bar P = -4i c sin cos, while
    # the derivatives of Q and U as scalars would give +-2i c sin cos.
    lmax, nside = 4, 32
    alm = np.zeros((3, hp.Alm.getsize(lmax)), complex)
    alm[1, hp.Alm.getidx(lmax, 2, 0)] = 1.0
    sky = Sky.from_alm(alm, nside)
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    c = 0.25 * np.sqrt(15 / (2 * np.pi))
    np.testing.assert_allclose(sky.p, -c * np.sin(theta) ** 2, atol=1e-14)
    np.testing.assert_allclose(sky.eth_p, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        sky.eth_bar_p, -4j * c * np.sin(theta) * np.cos(theta), atol=1e-13
    )


def test_sky_accepts_known_derivatives():
    rng = np.random.default_rng(14)
    i, q, u = rng.standard_normal((3, NPIX))
    e = rng.standard_normal((3, NPIX)) + 1j * rng.standard_normal((3, NPIX))
    sky = Sky.from_maps(i, q, u, eth_i=e[0], eth_p=e[1], eth_bar_p=e[2])
    np.testing.assert_array_equal([sky.eth_i, sky.eth_p, sky.eth_bar_p], e)
    with pytest.raises(ValueError, match="both"):
        Sky.from_maps(i, q, u, eth_p=e[1])


def test_map_maker_reuses_the_inverse_for_many_models():
    rng = np.random.default_rng(15)
    pixels, psi, phi = random_samples(NSIDE, 30, seed=16)
    keep = pixels != 4
    hm = HMaps.from_samples(NSIDE, pixels[keep], psi[keep], phi[keep], couples=COUPLES)
    basis = Basis.iqu(hwp=True).extend([(1, 0)])
    maker = MapMaker(hm, basis)
    assert not maker.mask[4] and maker.mask.sum() == NPIX - 1
    for seed in range(3):
        _, model = _random_model(
            np.random.default_rng(seed), [(0, 0), (2, -4), (1, 0), (0, 4)]
        )
        expected = map_make((hm, model), basis)
        actual = maker(model)
        np.testing.assert_array_equal(actual.mask, expected.mask)
        np.testing.assert_allclose(
            actual.solution, expected.solution, rtol=1e-12, atol=1e-12
        )
    del rng
