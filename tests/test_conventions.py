"""The conventions of SBM, written out here independently of the package.

The anchor tests state the convention literally, without sbm.conventions nor
sbm.testing: the spin expansion, the angles of litebird_sim, the h-map files, the
HWP-IP, the rotation of the basis and the derivatives of the sky. A change of
sbm.conventions that the package does not follow, or a change of the package
that breaks the convention, fails them. The pointing offsets (their direction,
the B detector of a pair and the wedge) are checked against the sky displaced
by a rotation in tests/test_pointing_geometry.py, also written out.

The other tests keep the conventions in one place: the crossing angle enters the
package only through sbm.conventions, the tables of the documentation are built
from the package, and the examples of sbm.conventions run.
"""

import ast
import doctest
import json
import re
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import pytest

from sbm import HMaps, Sky
from sbm import conventions as cv
from sbm import systematics as sy

ROOT = Path(__file__).parents[1]
NSIDE = 4
LMAX = 8

# --------------------------------------------------------------------- #
# The convention, written out (that of SBM 0.5)
#
#   d = sum_{n,m} S_{n,m} exp(-i (n psi + m phi)),   h_{n,m} = < exp(+i (n psi + m phi)) >
#
# with the angles of litebird_sim, psi (crossing angle) and chi = phi (HWP
# angle), and the standard eth = -(d_theta + i d_phi / sin(theta)).


def synthesize(model, pixels, psi, phi):
    d = np.zeros(pixels.size)
    for c in model.components():
        d += (c.map[pixels] * np.exp(-1j * (c.n * psi + c.m * phi))).real
    return d


def projection(pixels, psi, phi, n, m, npix):
    hits = np.bincount(pixels, minlength=npix)
    total = np.bincount(pixels, weights=np.exp(1j * (n * psi + m * phi)).real, minlength=npix)
    total = total + 1j * np.bincount(
        pixels, weights=np.exp(1j * (n * psi + m * phi)).imag, minlength=npix
    )
    return total / hits


def lbs_psi(psi):
    return psi


def lbs_rotation(angle):
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]])


# --------------------------------------------------------------------- #


def random_alm(seed):
    rng = np.random.default_rng(seed)
    size = hp.Alm.getsize(LMAX)
    alm = rng.standard_normal((3, size)) + 1j * rng.standard_normal((3, size))
    ell, m = hp.Alm.getlm(LMAX)
    alm[:, m == 0] = alm[:, m == 0].real
    alm[1:, ell < 2] = 0
    return alm


def random_angles(seed, n=12):
    rng = np.random.default_rng(seed)
    pixels = np.repeat(np.arange(hp.nside2npix(NSIDE)), n)
    psi, phi = rng.uniform(-np.pi, np.pi, (2, pixels.size))
    return pixels, psi, phi


def stokes(sky, pixels):
    return sky.i[pixels], sky.p.real[pixels], sky.p.imag[pixels]


def test_ideal_detector_without_hwp():
    sky = Sky.from_alm(random_alm(1), NSIDE)
    pixels, psi, phi = random_angles(2)
    i, q, u = stokes(sky, pixels)
    a = lbs_psi(psi)
    np.testing.assert_allclose(
        synthesize(sy.sky_signal(sky), pixels, psi, phi),
        i + q * np.cos(2 * a) + u * np.sin(2 * a),
        atol=1e-12,
    )
    # pair difference (d_T - d_B) / 2, B at 90 degrees
    d_b = i + q * np.cos(2 * (a + np.pi / 2)) + u * np.sin(2 * (a + np.pi / 2))
    np.testing.assert_allclose(
        synthesize(sy.sky_signal(sky, pair_difference=True), pixels, psi, phi),
        (i + q * np.cos(2 * a) + u * np.sin(2 * a) - d_b) / 2,
        atol=1e-12,
    )


def test_ideal_detector_behind_a_hwp_is_the_chain_of_litebird_sim():
    sky = Sky.from_alm(random_alm(3), NSIDE)
    pixels, psi, phi = random_angles(4)
    i, q, u = stokes(sky, pixels)
    hwp = np.diag([1.0, 1.0, -1.0, -1.0])
    detector = np.array([1.0, 1.0, 0.0, 0.0])
    tod = [
        detector
        @ lbs_rotation(chi).T
        @ hwp
        @ lbs_rotation(chi)
        @ lbs_rotation(lbs_psi(p))
        @ np.array([ii, qq, uu, 0.0])
        for ii, qq, uu, p, chi in zip(i, q, u, psi, phi, strict=True)
    ]
    np.testing.assert_allclose(
        synthesize(sy.sky_signal(sky, hwp=True), pixels, psi, phi), tod, atol=1e-12
    )


def test_polarization_angle_of_the_detector():
    # litebird_sim's det = (1, cos 2 xi, sin 2 xi, 0), xi from the reference axis
    sky = Sky.from_alm(random_alm(11), NSIDE)
    pixels, psi, phi = random_angles(12)
    i, q, u = stokes(sky, pixels)
    xi = 0.45
    angle = 2 * (lbs_psi(psi) + xi)
    np.testing.assert_allclose(
        synthesize(sy.sky_signal(sky, pol_angle_rad=xi), pixels, psi, phi),
        i + q * np.cos(angle) + u * np.sin(angle),
        atol=1e-12,
    )
    hwp = np.diag([1.0, 1.0, -1.0, -1.0])
    detector = np.array([1.0, np.cos(2 * xi), np.sin(2 * xi), 0.0])
    tod = [
        detector
        @ lbs_rotation(chi).T
        @ hwp
        @ lbs_rotation(chi)
        @ lbs_rotation(lbs_psi(p))
        @ np.array([ii, qq, uu, 0.0])
        for ii, qq, uu, p, chi in zip(i, q, u, psi, phi, strict=True)
    ]
    np.testing.assert_allclose(
        synthesize(sy.sky_signal(sky, hwp=True, pol_angle_rad=xi), pixels, psi, phi),
        tod,
        atol=1e-12,
    )


def test_hwp_ip_is_epsilon_i_cos_4_phi_plus_phi_qi():
    sky = Sky.from_alm(random_alm(5), NSIDE)
    pixels, psi, phi = random_angles(6)
    epsilon, phase = 1e-2, 0.7
    np.testing.assert_allclose(
        synthesize(sy.hwp_ip(sky, epsilon, phase), pixels, psi, phi),
        epsilon * sky.i[pixels] * np.cos(4 * phi + phase),
        atol=1e-14,
    )


def write_lbs_file(path, pixels, psi, phi, npix):
    """An h-map file in the layout of litebird_sim, h_{n,m} = <exp(+i (n psi' + m chi))>."""
    hits = np.bincount(pixels, minlength=npix).astype(float)
    with h5py.File(path, "w") as f:
        f.attrs["coordinate_system"] = "Galactic"
        f["0,0/Re"] = hits
        f["0,0/Im"] = np.zeros(npix)
        for n in range(4):
            for m in range(-6, 7):
                if (n, m) == (0, 0):
                    continue
                w = np.exp(1j * (n * lbs_psi(psi) + m * phi))
                f[f"{n},{m}/Re"] = np.bincount(pixels, weights=w.real, minlength=npix) / hits
                f[f"{n},{m}/Im"] = np.bincount(pixels, weights=w.imag, minlength=npix) / hits
    return path


COUPLES = [(1, 0), (2, 0), (2, -4), (2, 4), (1, -4), (3, -4), (0, 4), (1, 1), (1, -1)]


def test_h_maps_of_litebird_sim_and_of_samples(tmp_path):
    pixels, psi, phi = random_angles(7)
    npix = hp.nside2npix(NSIDE)
    loaded = HMaps.load(write_lbs_file(tmp_path / "h.h5", pixels, psi, phi, npix))
    computed = HMaps.from_samples(NSIDE, pixels, psi, phi, couples=COUPLES)
    for n, m in COUPLES:
        expected = projection(pixels, psi, phi, n, m, npix)
        np.testing.assert_allclose(loaded.h(n, m), expected, atol=1e-13, err_msg=f"{n, m}")
        np.testing.assert_allclose(computed.h(n, m), expected, atol=1e-13, err_msg=f"{n, m}")
    # and written back in the layout of litebird_sim
    computed.coordinate_system = "Galactic"
    computed.save(tmp_path / "saved.h5")
    with h5py.File(tmp_path / "saved.h5", "r") as f:
        for key in f:
            n, m = (int(x) for x in key.split(","))
            if (n, m) == (0, 0):
                continue
            w = np.exp(1j * (n * lbs_psi(psi) + m * phi))
            expected = (
                np.bincount(pixels, weights=w.real, minlength=npix)
                + 1j * np.bincount(pixels, weights=w.imag, minlength=npix)
            ) / np.bincount(pixels, minlength=npix)
            stored = f[key]["Re"][()] + 1j * f[key]["Im"][()]
            np.testing.assert_allclose(stored, expected, atol=1e-13, err_msg=key)


def rotated(sky, alpha):
    """The sky in a local basis turned by alpha in the sense of the polarization
    angle: a field of spin weight s is multiplied by exp(-i s alpha)."""

    def turn(x, s):
        return np.asarray(x) * np.exp(-1j * s * alpha)

    return Sky(
        sky.i,
        turn(sky.p, 2),
        eth_i=turn(sky.eth_i, 1),
        eth_p=turn(sky.eth_p, 3),
        eth_bar_p=turn(sky.eth_bar_p, 1),
    )


def test_turning_the_basis_changes_the_crossing_angle():
    # psi becomes psi - alpha, and the HWP angle is unchanged
    sky = Sky.from_alm(random_alm(8), NSIDE)
    pixels, psi, phi = random_angles(9)
    alpha = 0.4
    turned = rotated(sky, alpha)
    for build, hwp in [
        (lambda s: sy.sky_signal(s), False),
        (lambda s: sy.sky_signal(s, hwp=True) + sy.abs_pointing(s, 1e-2, 0.3), True),
        (lambda s: sy.sky_signal(s, hwp=True) + sy.hwp_wedge(s, 1e-2, -0.8), True),
    ]:
        np.testing.assert_allclose(
            synthesize(build(turned), pixels, psi - alpha, phi),
            synthesize(build(sky), pixels, psi, phi),
            atol=1e-12,
            err_msg=f"hwp={hwp}",
        )


def test_eth_of_the_temperature():
    alm = random_alm(10)
    sky = Sky.from_alm(alm, NSIDE)
    _, d_theta, d_phi = hp.alm2map_der1(alm[0], NSIDE, lmax=LMAX)
    np.testing.assert_allclose(sky.eth_i, -(d_theta + 1j * d_phi), atol=1e-10)
    np.testing.assert_allclose(sky.eth_bar_i, -(d_theta - 1j * d_phi), atol=1e-10)


# --------------------------------------------------------------------- #
# One place for the conventions


def test_the_examples_of_the_conventions_run():
    failures, tried = doctest.testmod(cv, optionflags=doctest.ELLIPSIS)
    assert tried > 0 and failures == 0


TRIGONOMETRIC = {"cos", "sin", "tan", "exp", "arctan2", "angle"}


def _mentions_psi(node):
    return any(
        isinstance(n, ast.Name) and n.id.startswith("psi") for n in ast.walk(node)
    )


def _is_psi(node):
    return isinstance(node, ast.Name) and node.id.startswith("psi")


def _name(func):
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def test_the_crossing_angle_enters_only_through_the_conventions():
    # Its direction is a convention: outside sbm.conventions (and the TOD of
    # sbm.testing, written out on purpose) a crossing angle is only passed on.
    # The HWP angle is not checked: it enters physical formulas (the harmonics
    # of a Mueller matrix), and ``phi`` is also the longitude of the sky.
    offenders = []
    for path in sorted((ROOT / "sbm").rglob("*.py")):
        if path.name in ("conventions.py", "testing.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.BinOp):
                hit = _is_psi(node.left) or _is_psi(node.right)
            elif isinstance(node, ast.UnaryOp):
                hit = _is_psi(node.operand)
            elif isinstance(node, ast.Call) and _name(node.func) in TRIGONOMETRIC:
                hit = any(_mentions_psi(a) for a in node.args)
            else:
                continue
            if hit:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, f"use sbm.conventions for the crossing angle at {offenders}"


def test_the_tables_of_the_documentation_are_built():
    docs = ROOT / "docs" / "source"
    conventions = (docs / "conventions.rst").read_text()
    assert ".. automodule:: sbm.conventions" in conventions
    for table in ("components", "relations"):
        assert f".. sbm-table:: {table}" in conventions
    systematics = (docs / "systematics.rst").read_text()
    assert ".. sbm-table:: systematics" in systematics
    assert ".. list-table::" not in systematics


# --------------------------------------------------------------------- #
# Until 0.5.0 is released: the ways of writing the convention of SBM 0.4.
# They are checked once sbm.conventions is that of 0.5 (SYNTHESIS_SIGN = -1);
# a line that refers to SBM 0.4, the thesis or the literature on purpose is
# skipped.

FORMER = [
    r"\(2, ?-4\)",
    r"\(1, ?-4\)",
    r"\(3, ?-4\)",
    r"\(1, ?-1\)",
    r"\(1, ?-3\)",
    r"\(3, ?-5\)",
    r"\(4, ?-4\)",
    r"\(4, ?-6\)",
    r"\(4, ?-8\)",
    r"\(2, ?-2\)",
    # the synthesis with e^{+i}; e^{+i} is the projection in 0.5
    r"S[^=]{0,40}exp\(i ?\(n ?psi",
    r"S[^=]{0,40}e\^\{i ?\(n ?\\?psi",
    r"S[^=]{0,40}e\^\{i ?n ?\\psi\}",
    r"counted opposite",
    r"- ?U ?\\?sin ?\(?2 ?\\?psi",
    r"4 ?\\?phi ?- ?2 ?\\?psi",
    r"psi ?- ?\\?phi ?\+ ?\\?chi",
    r"d_y ?- ?i ?d_x",
    r"partial_y ?- ?i ?\\?partial_x",
    r"h_\{n, ?-m\}",
    r"psi' ?= ?-",
]
DELIBERATE = re.compile(r"SBM 0\.4|0\.4\.\d|thesis|literature|Takase|Patanchon", re.I)


def _sources():
    for pattern in ("sbm/**/*.py", "tests/*.py", "docs/source/*.rst", "*.md",
                    "agent_workspace/*.md"):
        for path in sorted(ROOT.glob(pattern)):
            # these write the convention of SBM 0.4 on purpose: sbm.conventions (its
            # relations), the history, the migration and the page of the thesis
            if path.name in (
                "test_conventions.py",
                "conventions.py",
                "CHANGELOG.md",
                "migration.rst",
                "thesis_systematics.rst",
            ):
                continue
            yield path, path.read_text().splitlines()
    for path in sorted(ROOT.glob("notebooks/*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        yield path, [line for c in cells for line in "".join(c["source"]).splitlines()]


@pytest.mark.skipif(cv.SYNTHESIS_SIGN > 0, reason="the convention of SBM 0.4 is current")
def test_no_former_convention_is_left():
    found = []
    for path, lines in _sources():
        for k, line in enumerate(lines, 1):
            if DELIBERATE.search(line):
                continue
            for pattern in FORMER:
                if re.search(pattern, line):
                    found.append(f"{path.relative_to(ROOT)}:{k}: {line.strip()[:80]}")
                    break
    assert not found, "\n".join(found)
