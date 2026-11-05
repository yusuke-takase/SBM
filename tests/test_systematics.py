"""Every signal model of sbm.systematics against TOD simulated in real space.

Each function of ``sbm.systematics.ALL`` (the sky signal and every systematic) has
one or more cases in ``CASES``:
the full signal model and the TOD of the same signal written without the spin
decomposition (a Taylor expansion of the sky at the offset pointing, the gains,
litebird_sim's Mueller chain, ...). The tests check, for every case, that the
map-maker of the model is the least-squares fit of the TOD in several bases,
that the templates of the systematic absorb its first-order perturbation, that
no function of ``ALL`` lacks a case, and that the documentation lists all of them.

To add a systematic, add its cases here (see "Adding a systematic" in the docs).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from sbm import systematics as sy
from sbm.basis import Basis
from sbm.hmaps import HMaps
from sbm.map_making import map_make
from sbm.testing import RealSky, assert_matches_tod, random_samples, tod_from_components

from .test_hwp_convolution import (
    LMAX,
    pencil_beam,
    point_response,
    random_alms,
    sky_maps,
)

NSIDE = 2
NO_HWP_BASES = [
    Basis.qu(hwp=False),
    Basis.iqu(hwp=False),
    Basis.qu(hwp=False).extend([(1, 0), (3, 0)]),
]
HWP_BASES = [
    Basis.iqu(hwp=True),
    Basis.iqu(hwp=True).extend([(1, 0), (3, -4), (1, -4)]),
    Basis.iqu(hwp=True).extend([(1, -1), (3, -5), (1, -3), (0, 4), (0, 2)]),
]


# --------------------------------------------------------------------- #
# Reference TOD written independently of SBM's spin components


def lbs_rotation(angle):
    """litebird_sim's rotation of the Stokes parameters by 2 angle."""
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]])


def lbs_mueller_tod(real, pixels, psi, phi, harmonics):
    """litebird_sim's chain det R(chi)^T M(chi) R(chi) R(psi') S with psi' = -psi, chi = phi."""
    out = np.empty(pixels.size)
    for j, (p, a, b) in enumerate(zip(pixels, psi, phi)):
        mueller = np.zeros((4, 4))
        for k, m_k in harmonics.items():
            mueller += np.real(m_k) if k == 0 else 2 * np.real(m_k * np.exp(1j * k * b))
        stokes = np.array([real.x["I"][p], real.x["Q"][p], real.x["U"][p], 0.0])
        chain = lbs_rotation(b).T @ mueller @ lbs_rotation(b) @ lbs_rotation(-a)
        out[j] = np.array([1.0, 1.0, 0.0, 0.0]) @ chain @ stokes
    return out


def lbs_polarizer_tod(real, pixels, angle):
    """Detector seeing the polarization at litebird_sim's angle ``angle``."""
    i, q, u = (real.x[s][pixels] for s in "IQU")
    return i + q * np.cos(2 * angle) + u * np.sin(2 * angle)


# --------------------------------------------------------------------- #
# Cases: (systematic, the full model, its TOD)


@dataclass
class Case:
    systematic: Callable
    name: str
    hwp: bool
    build: Callable  # () -> (full SignalModel, tod(pixels, psi, phi))
    absorbed: Callable | None = (
        None  # () -> (base, base + perturbation the templates absorb)
    )

    def __str__(self):
        return self.name


def _pair(real, pixels, psi, gain_t=0.0, gain_b=0.0, **offsets):
    rho_t, chi_t = offsets.get("rho_t", 0.0), offsets.get("chi_t", 0.0)
    rho_b, chi_b = offsets.get("rho_b", 0.0), offsets.get("chi_b", 0.0)
    so = offsets.get("second_order", False)
    psi_b = psi + np.pi / 2
    d_t = real.detector(pixels, psi, rho=rho_t, alpha=psi + chi_t, second_order=so)
    d_b = real.detector(pixels, psi_b, rho=rho_b, alpha=psi_b + chi_b, second_order=so)
    return ((1 + gain_t) * d_t - (1 + gain_b) * d_b) / 2


def cases():
    real = RealSky(NSIDE, seed=1)
    sky = real.sky()
    pair = sy.sky_signal(sky, pair_difference=True)
    single = sy.sky_signal(sky)
    single_hwp = sy.sky_signal(sky, hwp=True)
    out = []

    def add(fn, name, hwp, model, tod, absorbed=None):
        out.append(Case(fn, name, hwp, lambda: (model, tod), absorbed))

    add(
        sy.sky_signal,
        "sky_signal-single",
        False,
        single,
        lambda px, ps, ph: real.detector(px, ps),
    )
    add(
        sy.sky_signal,
        "sky_signal-pair",
        False,
        pair,
        lambda px, ps, ph: _pair(real, px, ps),
    )
    add(
        sy.sky_signal,
        "sky_signal-hwp",
        True,
        single_hwp,
        lambda px, ps, ph: real.detector(px, ps, ph),
    )

    g_t, g_b = 0.013, -0.021
    add(
        sy.diff_gain,
        "diff_gain",
        False,
        pair + sy.diff_gain(sky, g_t, g_b),
        lambda px, ps, ph: _pair(real, px, ps, g_t, g_b),
        lambda: (pair, pair + sy.diff_gain(sky, 0.01, -0.01)),
    )

    comps = [np.random.default_rng(2).standard_normal(real.npix) for _ in range(2)]
    gamma_t, gamma_b = [0.1, -0.03], [0.04, 0.02]

    def bandpass_tod(px, ps, ph):
        leak = sum((gt - gb) * c[px] for gt, gb, c in zip(gamma_t, gamma_b, comps)) / 2
        return _pair(real, px, ps) + leak

    bandpass = pair + sy.bandpass_mismatch(comps, gamma_t, gamma_b)
    add(
        sy.bandpass_mismatch,
        "bandpass_mismatch",
        False,
        bandpass,
        bandpass_tod,
        lambda: (pair, bandpass),
    )

    offsets = dict(rho_t=0.03, chi_t=0.4, rho_b=0.02, chi_b=-1.3)
    for so in (False, True):
        model = pair + sy.diff_pointing(sky, 0.03, 0.4, 0.02, -1.3, second_order=so)
        add(
            sy.diff_pointing,
            f"diff_pointing-so{int(so)}",
            False,
            model,
            lambda px, ps, ph, so=so: _pair(real, px, ps, second_order=so, **offsets),
            None if so else (lambda model=model: (pair, model)),
        )

    for fn, direction in (
        (sy.abs_pointing, lambda ps, ph: ps),
        (sy.hwp_wedge, lambda ps, ph: ps - ph),
    ):
        for so in (False, True):
            model = single_hwp + fn(sky, 0.03, 0.7, second_order=so)
            add(
                fn,
                f"{fn.__name__}-so{int(so)}",
                True,
                model,
                lambda px, ps, ph, so=so, d=direction: real.detector(
                    px, ps, ph, rho=0.03, alpha=d(ps, ph) + 0.7, second_order=so
                ),
                None if so else (lambda model=model: (single_hwp, model)),
            )

    ip = single_hwp + sy.hwp_ip(sky, 2e-3, 0.8)
    add(
        sy.hwp_ip,
        "hwp_ip",
        True,
        ip,
        lambda px, ps, ph: real.detector(px, ps, ph)
        + 2e-3 * real.x["I"][px] * np.cos(4 * ph + 0.8),
        lambda: (single_hwp, ip),
    )

    rng = np.random.default_rng(3)
    harmonics = {
        0: sy.IDEAL_HWP + 0.02 * rng.standard_normal((4, 4)),
        2: 0.01 * (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))),
        4: 0.01 * (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))),
    }
    add(
        sy.hwp_mueller,
        "hwp_mueller",
        True,
        single_hwp + sy.hwp_mueller(sky, harmonics),
        lambda px, ps, ph: lbs_mueller_tod(real, px, ps, ph, harmonics),
    )

    delta = 0.05
    add(
        sy.polarization_angle,
        "polarization_angle-single",
        False,
        single + sy.polarization_angle(sky, delta),
        lambda px, ps, ph: lbs_polarizer_tod(real, px, -ps + delta),
    )
    add(
        sy.polarization_angle,
        "polarization_angle-pair",
        False,
        pair + sy.polarization_angle(sky, delta),
        lambda px, ps, ph: (
            lbs_polarizer_tod(real, px, -ps + delta)
            - lbs_polarizer_tod(real, px, -(ps + np.pi / 2) + delta)
        )
        / 2,
    )
    add(
        sy.polarization_angle,
        "polarization_angle-hwp",
        True,
        single_hwp + sy.polarization_angle(sky, delta, hwp=True),
        lambda px, ps, ph: lbs_polarizer_tod(real, px, 2 * ph - delta - ps),
    )

    # beam: a pencil beam behind a HWP of general Mueller matrix (see test_hwp_convolution)
    alm = random_alms(30)
    matrix = np.array(
        [
            [0.9, 0.04, -0.03, 0.02],
            [0.06, 0.8, -0.07, 0.03],
            [-0.02, 0.05, -0.7, -0.04],
            [0.01, -0.03, 0.02, -0.8],
        ]
    )
    detector = np.array([1.0, 1.0, 0.0, 0.0])
    maps = sky_maps(alm)
    beam_model = sy.beam(
        alm, pencil_beam(detector), NSIDE, range(LMAX + 1), hwp=True, mueller=matrix
    )
    add(
        sy.beam,
        "beam-hwp",
        True,
        beam_model,
        lambda px, ps, ph: np.array(
            [
                point_response(maps[:, p : p + 1], detector, matrix, [a], [b])[0]
                for p, a, b in zip(px, ps, ph)
            ]
        ),
    )
    return out


CASES = cases()


@pytest.mark.parametrize("case", CASES, ids=str)
def test_model_matches_tod(case):
    model, tod = case.build()
    samples = random_samples(NSIDE, 60 if case.hwp else 40, seed=7)
    data = tod(*samples)
    for basis in HWP_BASES if case.hwp else NO_HWP_BASES:
        assert_matches_tod(model, data, samples, basis)


@pytest.mark.parametrize("case", [c for c in CASES if not c.hwp], ids=str)
def test_models_without_hwp_have_no_hwp_harmonic(case):
    model, _ = case.build()
    assert all(m == 0 for _, m in model.couples), model.couples


@pytest.mark.parametrize("case", [c for c in CASES if c.absorbed is not None], ids=str)
def test_templates_absorb_the_first_order_perturbation(case):
    base, full = case.absorbed()
    templates = case.systematic.templates
    assert templates, f"{case.systematic.__name__} declares no templates"
    samples = random_samples(NSIDE, 60 if case.hwp else 40, seed=8)
    basis = (Basis.iqu(hwp=True) if case.hwp else Basis.qu(hwp=False)).extend(templates)
    couples = (
        full.required_couples(basis.full)
        | base.required_couples(basis.full)
        | {(a[0] - b[0], a[1] - b[1]) for a in basis.full for b in basis.full}
    )
    hm = HMaps.from_samples(NSIDE, *samples, couples=couples)
    with_systematic = map_make((hm, full), basis)
    without = map_make((hm, base), basis)
    np.testing.assert_allclose(with_systematic.iqu[1:], without.iqu[1:], atol=1e-12)


def test_every_exported_function_is_a_registered_systematic():
    exported = {
        getattr(sy, name)
        for name in sy.__all__
        if callable(getattr(sy, name)) and name != "systematic"
    }
    assert exported == set(sy.ALL)
    assert len(sy.ALL) == len(set(sy.ALL))


def test_every_systematic_has_a_case():
    covered = {c.systematic for c in CASES}
    missing = [f.__name__ for f in sy.ALL if f not in covered]
    assert not missing, f"add cases for {missing} to tests/test_systematics.py"
    absorbed = {c.systematic for c in CASES if c.absorbed is not None}
    lacking = [f.__name__ for f in sy.ALL if f.templates and f not in absorbed]
    assert not lacking, f"add a case with `absorbed` for the templates of {lacking}"


def test_every_systematic_is_documented():
    text = (
        Path(__file__).parents[1] / "docs" / "source" / "systematics.rst"
    ).read_text()
    missing = [
        f.__name__ for f in sy.ALL if f"sbm.systematics.{f.__name__}" not in text
    ]
    assert not missing, f"list {missing} in docs/source/systematics.rst"


def test_declaring_a_systematic_again_replaces_it():
    # e.g. when its module is reloaded: same module and qualified name
    before = list(sy.ALL)

    def declare():
        @sy.systematic(data="any")
        def probe(sky):
            return sky

        return probe

    try:
        first, second = declare(), declare()
        assert second in sy.ALL and first not in sy.ALL
        assert len(sy.ALL) == len(before) + 1
    finally:
        sy.ALL[:] = before


def test_declarations():
    for f in sy.ALL:
        assert f.data in sy.DATA
        assert all(isinstance(t, tuple) and len(t) == 2 for t in f.templates)
    assert sy.abs_pointing.templates == ((1, 0), (3, -4), (1, -4))
    with pytest.raises(ValueError, match="data"):
        sy.systematic(data="unknown")
    with pytest.raises(ValueError, match="couples"):
        sy.systematic(data="any", templates=(0, 4))  # one couple, not a list of couples


# --------------------------------------------------------------------- #
# Relations between the models


def test_hwp_ip_is_a_case_of_hwp_mueller():
    # The 4f IP in the focal plane, M_QI = eps cos(4 a + phase) and
    # M_UI = eps sin(4 a + phase) (Patanchon et al. 2024), taken to the frame of
    # the HWP, M_hwp(a) = R(a) M_fp(a) R(a)^T, and expanded in harmonics of a.
    sky = RealSky(NSIDE, seed=9).sky()
    eps, phase = 3e-3, 0.7
    alpha = 2 * np.pi * np.arange(64) / 64
    m_fp = np.zeros((alpha.size, 4, 4))
    m_fp[:, 1, 0] = eps * np.cos(4 * alpha + phase)
    m_fp[:, 2, 0] = eps * np.sin(4 * alpha + phase)
    c, s = np.cos(2 * alpha), np.sin(2 * alpha)
    rot = np.zeros((alpha.size, 4, 4))
    rot[:, 0, 0] = rot[:, 3, 3] = 1.0
    rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = c, s, -s, c
    m_hwp = rot @ m_fp @ np.swapaxes(rot, 1, 2)
    harmonics = np.einsum("a,aij->ij", np.exp(-2j * alpha), m_hwp) / alpha.size
    assert np.allclose(np.einsum("a,aij->ij", np.exp(-6j * alpha), m_hwp), 0)
    general, ip = sy.hwp_mueller(sky, {2: harmonics}), sy.hwp_ip(sky, eps, phase)
    assert general.couples == ip.couples == [(0, 4)]
    np.testing.assert_allclose(general.component(0, 4), ip.component(0, 4), atol=1e-17)
    assert sy.hwp_mueller(sky, {0: sy.IDEAL_HWP}).couples == [(0, 0)]
    assert not np.any(sy.hwp_mueller(sky, {0: sy.IDEAL_HWP}).component(0, 0))


def test_hwp_spins_are_consistent():
    # The HWP-IP, the polarization and the temperature term of the wedge are
    # functions of the same angles: IP = 2 pol + 4 wedge_T between spins.
    sky = RealSky(NSIDE, seed=10).sky()
    (ip,) = sy.hwp_ip(sky, 1.0, 0.0).couples
    wedge_t = (1, -1)
    assert sy.hwp_wedge(sky, 1.0, 0.0).component(*wedge_t).any()
    pol = Basis.iqu(hwp=True).polarization
    combination = (2 * pol[0] - 4 * wedge_t[0], 2 * pol[1] - 4 * wedge_t[1])
    assert ip in (combination, (-combination[0], -combination[1]))


def test_product_of_models_is_the_product_of_signals():
    real = RealSky(NSIDE, seed=11)
    sky = real.sky()
    a = sy.sky_signal(sky, hwp=True) + sy.hwp_ip(sky, 0.1, 0.3)
    b = sy.sky_signal(sky) + sy.diff_pointing(sky, 0.1, 0.2, 0.0, 0.0)
    pixels, psi, phi = random_samples(NSIDE, 5, seed=12)

    def tod(model):
        comps = {
            (c.n, c.m): c.map for c in model.components() if (c.n, c.m) in model.couples
        }
        return tod_from_components(comps, pixels, psi, phi)

    np.testing.assert_allclose(tod(a * b), tod(a) * tod(b), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tod(a * a), tod(a) ** 2, rtol=1e-12, atol=1e-12)


def test_models_add_up():
    sky = RealSky(NSIDE, seed=13).sky()
    both = sy.diff_gain(sky, 0.01, 0.0) + sy.diff_gain(sky, 0.0, -0.01)
    np.testing.assert_allclose(
        both.component(0, 0), sy.diff_gain(sky, 0.01, -0.01).component(0, 0)
    )
    with pytest.raises(ValueError, match="without HWP"):
        sy.sky_signal(sky, hwp=True, pair_difference=True)
