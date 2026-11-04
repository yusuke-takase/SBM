"""Signal models of the systematics against TOD simulated in real space.

The sky is given by its values and real derivatives (gradient and Hessian along
x = theta and y = phi) in every pixel. The TOD sample it at the offset pointing
by a Taylor expansion and apply the response of the detector, without the
spin decomposition; the least-squares fit of the TOD is compared with the
map-maker.
"""

import healpy as hp
import numpy as np
import pytest

from sbm import systematics as sy
from sbm.basis import Basis
from sbm.hmaps import HMaps
from sbm.map_making import map_make
from sbm.sky import Sky

from .tod import least_squares, random_samples

NSIDE = 2
NPIX = hp.nside2npix(NSIDE)
COUPLES = [(n, m) for n in range(0, 13) for m in range(-16, 17)]


class RealSky:
    """I, Q, U with their gradients and Hessians, and the equivalent Sky."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.x = {s: rng.standard_normal(NPIX) for s in "IQU"}
        self.d = {(s, a): rng.standard_normal(NPIX) for s in "IQU" for a in ("x", "y")}
        self.h = {
            (s, a): rng.standard_normal(NPIX) for s in "IQU" for a in ("xx", "xy", "yy")
        }

    def eth(self, s):
        return self.d[s, "y"] - 1j * self.d[s, "x"]

    def eth2(self, s):  # (d_y - i d_x)^2
        return self.h[s, "yy"] - self.h[s, "xx"] - 2j * self.h[s, "xy"]

    def lap(self, s):
        return self.h[s, "xx"] + self.h[s, "yy"]

    def sky(self):
        x, eth, eth2, lap = self.x, self.eth, self.eth2, self.lap
        eth_bar = {s: np.conj(eth(s)) for s in "IQU"}
        return Sky.from_maps(
            x["I"],
            x["Q"],
            x["U"],
            eth_i=eth("I"),
            eth_p=eth("Q") + 1j * eth("U"),
            eth_bar_p=eth_bar["Q"] + 1j * eth_bar["U"],
            second_order={
                "eth2_i": eth2("I"),
                "lap_i": lap("I"),
                "eth2_p": eth2("Q") + 1j * eth2("U"),
                "eth_bar2_p": np.conj(eth2("Q")) + 1j * np.conj(eth2("U")),
                "lap_p": lap("Q") + 1j * lap("U"),
            },
        )

    def sample(self, s, pixels, rho, alpha, second_order):
        """Value of s at the offset rho along -(sin(alpha) e_x + cos(alpha) e_y)."""
        dx, dy = -rho * np.sin(alpha), -rho * np.cos(alpha)
        v = (
            self.x[s][pixels]
            + dx * self.d[s, "x"][pixels]
            + dy * self.d[s, "y"][pixels]
        )
        if second_order:
            v = v + 0.5 * (
                dx**2 * self.h[s, "xx"][pixels]
                + 2 * dx * dy * self.h[s, "xy"][pixels]
                + dy**2 * self.h[s, "yy"][pixels]
            )
        return v

    def detector(self, pixels, psi, phi=None, rho=0.0, alpha=0.0, second_order=False):
        """Single detector: I + Re[P e^{2i psi}] without HWP, I + Re[P e^{i(2 psi - 4 phi)}] with HWP."""
        i, q, u = (self.sample(s, pixels, rho, alpha, second_order) for s in "IQU")
        angle = 2 * psi if phi is None else 2 * psi - 4 * phi
        return i + q * np.cos(angle) - u * np.sin(angle)


def check(tod, samples, model, basis):
    pixels, psi, phi = samples
    hm = HMaps.from_samples(NSIDE, pixels, psi, phi, couples=COUPLES)
    result = map_make((hm, model), basis)
    expected = least_squares(tod, pixels, psi, phi, basis.couples, NPIX)
    assert result.mask.all()
    for c in basis.couples:
        np.testing.assert_allclose(
            result.component(*c), expected[c], rtol=1e-9, atol=1e-9
        )


NO_HWP_BASES = [
    Basis.qu(hwp=False),
    Basis.iqu(hwp=False),
    Basis.qu(hwp=False).extend([(1, 0), (3, 0)]),
]
HWP_BASES = [
    Basis.iqu(hwp=True),
    Basis.iqu(hwp=True).extend([(1, 0), (3, -4), (1, -4)]),
    Basis.iqu(hwp=True).extend([(1, -1), (3, -5), (1, -3), (0, 4)]),
]


@pytest.mark.parametrize("basis", NO_HWP_BASES)
def test_sky_signal_of_a_detector_and_a_pair(basis):
    real = RealSky(1)
    samples = random_samples(NSIDE, 40, seed=2)
    pixels, psi, _ = samples
    single = real.detector(pixels, psi)
    check(single, samples, sy.sky_signal(real.sky()), basis)
    pair = (single - real.detector(pixels, psi + np.pi / 2)) / 2
    check(pair, samples, sy.sky_signal(real.sky(), pair_difference=True), basis)


@pytest.mark.parametrize("basis", NO_HWP_BASES)
def test_diff_gain(basis):
    real = RealSky(3)
    samples = random_samples(NSIDE, 40, seed=4)
    pixels, psi, _ = samples
    g_t, g_b = 0.013, -0.021
    tod = (
        (1 + g_t) * real.detector(pixels, psi)
        - (1 + g_b) * real.detector(pixels, psi + np.pi / 2)
    ) / 2
    model = sy.sky_signal(real.sky(), pair_difference=True) + sy.diff_gain(
        real.sky(), g_t, g_b
    )
    check(tod, samples, model, basis)


@pytest.mark.parametrize("second_order", [False, True])
@pytest.mark.parametrize("basis", NO_HWP_BASES)
def test_diff_pointing(basis, second_order):
    real = RealSky(5)
    samples = random_samples(NSIDE, 40, seed=6)
    pixels, psi, _ = samples
    rho_t, chi_t, rho_b, chi_b = 0.03, 0.4, 0.02, -1.3
    d_t = real.detector(
        pixels, psi, rho=rho_t, alpha=psi + chi_t, second_order=second_order
    )
    psi_b = psi + np.pi / 2
    d_b = real.detector(
        pixels, psi_b, rho=rho_b, alpha=psi_b + chi_b, second_order=second_order
    )
    sky = real.sky()
    model = sy.sky_signal(sky, pair_difference=True) + sy.diff_pointing(
        sky, rho_t, chi_t, rho_b, chi_b, second_order=second_order
    )
    check((d_t - d_b) / 2, samples, model, basis)


@pytest.mark.parametrize("basis", NO_HWP_BASES)
def test_bandpass_mismatch(basis):
    real = RealSky(7)
    samples = random_samples(NSIDE, 40, seed=8)
    pixels, psi, _ = samples
    rng = np.random.default_rng(9)
    comps = [rng.standard_normal(NPIX) for _ in range(2)]
    gamma_t, gamma_b = [0.1, -0.03], [0.04, 0.02]
    leak_t = sum(g * c[pixels] for g, c in zip(gamma_t, comps))
    leak_b = sum(g * c[pixels] for g, c in zip(gamma_b, comps))
    tod = (
        real.detector(pixels, psi)
        + leak_t
        - real.detector(pixels, psi + np.pi / 2)
        - leak_b
    ) / 2
    model = sy.sky_signal(real.sky(), pair_difference=True) + sy.bandpass_mismatch(
        comps, gamma_t, gamma_b
    )
    check(tod, samples, model, basis)


@pytest.mark.parametrize("second_order", [False, True])
@pytest.mark.parametrize("basis", HWP_BASES)
def test_abs_pointing(basis, second_order):
    real = RealSky(10)
    samples = random_samples(NSIDE, 60, seed=11, hwp=True)
    pixels, psi, phi = samples
    rho, chi = 0.03, 0.7
    tod = real.detector(
        pixels, psi, phi, rho=rho, alpha=psi + chi, second_order=second_order
    )
    sky = real.sky()
    model = sy.sky_signal(sky, hwp=True) + sy.abs_pointing(
        sky, rho, chi, second_order=second_order
    )
    check(tod, samples, model, basis)


@pytest.mark.parametrize("second_order", [False, True])
@pytest.mark.parametrize("basis", HWP_BASES)
def test_hwp_wedge(basis, second_order):
    real = RealSky(12)
    samples = random_samples(NSIDE, 60, seed=13, hwp=True)
    pixels, psi, phi = samples
    rho, chi = 0.03, -0.5
    tod = real.detector(
        pixels, psi, phi, rho=rho, alpha=psi - phi + chi, second_order=second_order
    )
    sky = real.sky()
    model = sy.sky_signal(sky, hwp=True) + sy.hwp_wedge(
        sky, rho, chi, second_order=second_order
    )
    check(tod, samples, model, basis)


@pytest.mark.parametrize("basis", HWP_BASES)
def test_hwp_ip(basis):
    real = RealSky(14)
    samples = random_samples(NSIDE, 60, seed=15, hwp=True)
    pixels, psi, phi = samples
    epsilon, phase = 2e-3, 0.8
    tod = real.detector(pixels, psi, phi) + epsilon * real.x["I"][pixels] * np.cos(
        4 * phi - phase
    )
    sky = real.sky()
    check(
        tod,
        samples,
        sy.sky_signal(sky, hwp=True) + sy.hwp_ip(sky, epsilon, phase),
        basis,
    )


def test_hwp_ip_template_removes_the_leakage():
    real = RealSky(16)
    samples = random_samples(NSIDE, 60, seed=17, hwp=True)
    pixels, psi, phi = samples
    hm = HMaps.from_samples(NSIDE, pixels, psi, phi, couples=COUPLES)
    leak = sy.hwp_ip(real.sky(), 1e-2, 0.3)
    p3 = map_make((hm, leak), Basis.iqu(hwp=True)).iqu[1:]
    p5 = map_make((hm, leak), Basis.iqu(hwp=True).extend([(0, 4)])).iqu[1:]
    assert np.abs(p3).max() > 1e-4
    assert np.abs(p5).max() < 1e-12 * np.abs(p3).max()


def test_hwp_spins_are_consistent():
    # The HWP-IP, the polarization and the temperature term of the wedge are
    # functions of the same angles: IP = 2 pol + 4 wedge_T between spins.
    sky = RealSky(18).sky()
    (ip,) = [c for c in sy.hwp_ip(sky, 1.0, 0.0).couples]
    wedge_t = (1, -1)
    assert sy.hwp_wedge(sky, 1.0, 0.0).component(*wedge_t).any()
    pol = Basis.iqu(hwp=True).polarization
    combination = (2 * pol[0] - 4 * wedge_t[0], 2 * pol[1] - 4 * wedge_t[1])
    assert ip in (combination, (-combination[0], -combination[1]))


def test_models_add_up():
    sky = RealSky(19).sky()
    both = sy.diff_gain(sky, 0.01, 0.0) + sy.diff_gain(sky, 0.0, -0.01)
    np.testing.assert_allclose(
        both.component(0, 0), sy.diff_gain(sky, 0.01, -0.01).component(0, 0)
    )
    with pytest.raises(ValueError, match="without HWP"):
        sy.sky_signal(sky, hwp=True, pair_difference=True)
