"""The random pointing jitter against the least-squares fit of TOD in real space.

The jitter draws an offset for every sample, with independent Gaussian
components of standard deviation sigma along two orthogonal directions. The
least-squares estimator of a pixel is linear in its TOD, so the covariance of
the fit is the sum over the samples of the variance of the jitter in the TOD of
that sample times the outer product of the fit of a TOD that is 1 at that sample
only. The TOD of the unit offsets are written in real space (RealSky, offsets
along ``cos(alpha) e_x + sin(alpha) e_y``), without the spin decomposition.
"""

import functools

import numpy as np
import pytest

from sbm import (
    Basis,
    HMaps,
    map_make,
    noise_covariance,
    pointing_jitter_covariance,
    pointing_jitter_maps,
)
from sbm import systematics as sy
from sbm.testing import RealSky, least_squares, random_samples

NSIDE = 1
NPIX = 12
NPER = 40
SIGMA = 2e-3
# behind a HWP the squared data reach the sums of two couples of the offset
# models, and the matrix the differences of two couples of the basis
COUPLES = [(n, m) for n in range(13) for m in range(-20, 21)]


def _samples(seed=3):
    pixels, psi, phi = random_samples(NSIDE, NPER, seed=seed)
    hm = HMaps.from_samples(NSIDE, pixels, psi, phi, couples=COUPLES)
    return hm, (pixels, psi, phi)


def _exact_covariance(samples, basis, perturbations, sigma=SIGMA):
    """Covariance of the least-squares fit for the jitter: ``perturbations``
    gives, for every independent offset, the TOD of the unit offsets along the
    two directions (``chi`` = 0 and pi/2)."""
    pixels, psi, phi = samples
    var = sigma**2 * sum(g**2 for g in perturbations)
    full = basis.full
    cov = np.zeros((NPIX, len(full), len(full)), dtype=complex)
    for k in range(NPER):  # the k-th sample of every pixel at once
        sel = np.arange(NPIX) * NPER + k
        assert np.all(pixels[sel] == np.arange(NPIX))
        unit = np.zeros(pixels.size)
        unit[sel] = 1.0
        fit = least_squares(unit, pixels, psi, phi, basis.couples, NPIX)
        y = np.array(
            [
                fit[c] if c in basis.couples else np.conj(fit[(-c[0], -c[1])])
                for c in full
            ]
        )
        cov += var[sel][:, None, None] * np.einsum("ip,jp->pij", y, y.conj())
    return cov


def _unit(tod):
    """The TOD of the unit offsets along chi = 0 and pi/2 of ``tod(rho, chi)``."""
    return [tod(1.0, chi) - tod(0.0, 0.0) for chi in (0.0, np.pi / 2)]


def _real_sky(intensity=True):
    real = RealSky(NSIDE, seed=5)
    if not intensity:  # a co-pointed pair behind a HWP sees the polarization only
        real.x["I"] = np.zeros(NPIX)
        for table in (real.d, real.h):
            for key in table:
                if key[0] == "I":
                    table[key] = np.zeros(NPIX)
    return real


def _single_hwp(real, samples, direction):
    pixels, psi, phi = samples
    return lambda rho, chi: real.detector(
        pixels, psi, phi, rho=rho, alpha=direction(psi, phi) + chi
    )


def _pair(real, samples, offset_t, offset_b):
    """``(d_T - d_B) / 2`` without HWP, B at 90 degrees, both offsets measured
    from the same reference axis (``offset = (rho, chi)``)."""
    pixels, psi, _ = samples
    (rt, ct), (rb, cb) = offset_t, offset_b
    t = real.detector(pixels, psi, rho=rt, alpha=psi + ct)
    b = real.detector(pixels, psi + np.pi / 2, rho=rb, alpha=psi + cb)
    return (t - b) / 2


def cases():
    hm, samples = _samples()
    out = []

    real = _real_sky()
    sky = real.sky()
    for fn, direction in (
        (sy.abs_pointing, lambda ps, ph: ps),
        (sy.hwp_wedge, lambda ps, ph: ps + ph),
    ):
        tod = _single_hwp(real, samples, direction)
        for basis in (Basis.iqu(hwp=True), Basis.iqu(hwp=True).extend(fn.templates)):
            out.append(
                (
                    f"{fn.__name__}-{basis.dimension}",
                    basis,
                    functools.partial(fn, sky),
                    _unit(tod),
                )
            )

    real_pol = _real_sky(intensity=False)
    tod = _single_hwp(real_pol, samples, lambda ps, ph: ps)
    out.append(
        (
            "co-pointed pair behind a HWP",
            Basis.iqu(hwp=True),
            functools.partial(sy.abs_pointing, real_pol.sky()),
            _unit(tod),
        )
    )

    pair_basis = Basis.qu(hwp=False).extend(sy.diff_pointing.templates)
    out.append(
        (
            "pair, shared offset",
            pair_basis,
            lambda rho, chi: sy.diff_pointing(sky, rho, chi, rho, chi),
            _unit(lambda rho, chi: _pair(real, samples, (rho, chi), (rho, chi))),
        )
    )
    out.append(
        (
            "pair, independent offsets",
            Basis.qu(hwp=False),
            [
                lambda rho, chi: sy.diff_pointing(sky, rho, chi, 0.0, 0.0),
                lambda rho, chi: sy.diff_pointing(sky, 0.0, 0.0, rho, chi),
            ],
            _unit(lambda rho, chi: _pair(real, samples, (rho, chi), (0.0, 0.0)))
            + _unit(lambda rho, chi: _pair(real, samples, (0.0, 0.0), (rho, chi))),
        )
    )
    return hm, samples, out


HM, SAMPLES, CASES = cases()


@pytest.mark.parametrize(
    ("basis", "model", "perturbations"),
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_covariance_is_that_of_the_least_squares_fit(basis, model, perturbations):
    cov = pointing_jitter_covariance(HM, basis, model, SIGMA)
    expected = _exact_covariance(SAMPLES, basis, perturbations)
    np.testing.assert_allclose(
        cov, expected, rtol=1e-9, atol=1e-9 * np.abs(expected).max()
    )


def test_a_flat_squared_gradient_is_white_noise():
    # an offset model whose unit offsets have the data 1 and 0 at every sample
    # (an intensity gradient along x only, sampled without HWP): the jitter is
    # white noise of variance sigma^2
    one = np.ones(NPIX)

    def model(rho, chi):
        return sy.sky_signal(_constant_sky(rho * np.cos(chi) * one)) - sy.sky_signal(
            _constant_sky(0 * one)
        )

    basis = Basis.iqu(hwp=False)
    np.testing.assert_allclose(
        pointing_jitter_covariance(HM, basis, model, SIGMA),
        noise_covariance(HM, basis, SIGMA, 1.0),
        rtol=1e-12,
        atol=1e-20,
    )


def _constant_sky(i):
    from sbm import Sky

    zero = np.zeros(NPIX, dtype=complex)
    return Sky(i, zero, eth_i=zero, eth_p=zero, eth_bar_p=zero)


def test_the_maps_have_the_covariance():
    _, basis, model, _ = CASES[0]
    cov = pointing_jitter_covariance(HM, basis, model, SIGMA)
    rng = np.random.default_rng(13)
    draws = np.array(
        [pointing_jitter_maps(HM, basis, model, SIGMA, rng).iqu for _ in range(4000)]
    )
    p = 7
    emp = np.cov(draws[:, :, p].T)
    t = np.array([[1, 0, 0], [0, 1, 1j], [0, 1, -1j]])
    t_inv = np.linalg.inv(t)
    expected = (t_inv @ cov[p] @ t_inv.conj().T).real
    np.testing.assert_allclose(
        emp, expected, rtol=0.1, atol=0.1 * np.abs(expected).max()
    )


def test_the_maps_follow_the_generator():
    _, basis, model, _ = CASES[0]
    first = pointing_jitter_maps(HM, basis, model, SIGMA, np.random.default_rng(1))
    again = pointing_jitter_maps(HM, basis, model, SIGMA, np.random.default_rng(1))
    other = pointing_jitter_maps(HM, basis, model, SIGMA, np.random.default_rng(2))
    np.testing.assert_array_equal(first.solution, again.solution)
    assert not np.array_equal(first.solution, other.solution)


def test_no_jitter_and_unsolved_pixels():
    _, basis, model, _ = CASES[0]
    mask = np.ones(NPIX, bool)
    mask[3] = False
    zero = pointing_jitter_maps(HM, basis, model, 0.0, np.random.default_rng(1), mask)
    assert np.all(zero.solution[:, mask] == 0) and np.all(np.isnan(zero.solution[:, 3]))
    cov = pointing_jitter_covariance(HM, basis, model, SIGMA, mask)
    assert np.all(np.isnan(cov[3])) and np.all(np.isfinite(cov[mask]))
    # the jitter is added to the solution of the signal without jitter
    ideal = map_make((HM, sy.sky_signal(_real_sky().sky(), hwp=True)), basis, mask)
    np.testing.assert_array_equal(ideal.mask, zero.mask)


@pytest.mark.parametrize("sigma", [-1e-3, np.nan, np.inf])
def test_invalid_sigma(sigma):
    _, basis, model, _ = CASES[0]
    with pytest.raises(ValueError, match="sigma_rad"):
        pointing_jitter_covariance(HM, basis, model, sigma)


def test_offset_models_must_be_linear():
    sky = _real_sky().sky()
    model = functools.partial(sy.abs_pointing, sky, second_order=True)
    with pytest.raises(ValueError, match="linear"):
        pointing_jitter_covariance(HM, Basis.iqu(hwp=True), model, SIGMA)
    with pytest.raises(ValueError, match="at least one"):
        pointing_jitter_covariance(HM, Basis.iqu(hwp=True), [], SIGMA)


def test_missing_couples_are_listed():
    pixels, psi, phi = SAMPLES
    few = HMaps.from_samples(
        NSIDE,
        pixels,
        psi,
        phi,
        couples=[(n, m) for n in range(5) for m in range(-8, 9)],
    )
    _, basis, model, _ = CASES[0]
    with pytest.raises(KeyError, match="pointing jitter"):
        pointing_jitter_covariance(few, basis, model, SIGMA)


def test_truncation_falls_as_sigma_squared():
    """Layer 2: with the second-order TOD the variance of a sample gains
    ``sigma^4 / 2 tr(H^2)``, ``H`` the Hessian of its data in the offset, so that
    the first-order covariance differs from the exact one by O(sigma^2)."""
    real = _real_sky()
    pixels, psi, phi = SAMPLES
    _, basis, model, unit = CASES[0]

    def tod(dx, dy):  # data at the offset (dx, dy) in the frame of the detector
        rho, chi = np.hypot(dx, dy), np.arctan2(dy, dx)
        return real.detector(
            pixels, psi, phi, rho=rho, alpha=psi + chi, second_order=True
        )

    d0 = tod(0.0, 0.0)
    hxx = tod(1.0, 0.0) + tod(-1.0, 0.0) - 2 * d0
    hyy = tod(0.0, 1.0) + tod(0.0, -1.0) - 2 * d0
    hxy = tod(1.0, 1.0) - tod(1.0, 0.0) - tod(0.0, 1.0) + d0
    gradient = sum(g**2 for g in unit)
    residuals = []
    for sigma in (1e-3, 2e-3):
        extra = 0.5 * sigma**2 * (hxx**2 + 2 * hxy**2 + hyy**2)
        exact = _exact_covariance(SAMPLES, basis, [np.sqrt(gradient + extra)], sigma)
        first = pointing_jitter_covariance(HM, basis, model, sigma)
        residuals.append(np.abs(first - exact).max() / np.abs(exact).max())
    assert residuals[0] > 1e-9
    np.testing.assert_allclose(residuals[1] / residuals[0], 4.0, rtol=0.01)
