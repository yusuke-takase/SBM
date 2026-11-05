"""SphericalHarmonics and the beams of sbm.beam."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest

from sbm import beam as sbm_beam
from sbm.beam import convolve
from sbm.harmonics import SphericalHarmonics
from sbm.sky import Sky


def random_teb(lmax, seed):
    rng = np.random.default_rng(seed)
    size = hp.Alm.getsize(lmax)
    alm = rng.standard_normal((3, size)) + 1j * rng.standard_normal((3, size))
    ell, m = hp.Alm.getlm(lmax)
    alm[:, m == 0] = alm[:, m == 0].real
    alm[1:, ell < 2] = 0
    return alm


def test_layout_and_conversions():
    alm = random_teb(8, 1)
    sh = SphericalHarmonics(alm, lmax=8)
    assert (sh.nstokes, sh.lmax, sh.mmax) == (3, 8, 8)
    assert SphericalHarmonics.of(alm).lmax == 8
    assert SphericalHarmonics.of(sh) is sh
    # an object with the attributes of litebird_sim's SphericalHarmonics
    lbs_like = SimpleNamespace(values=alm, lmax=8, mmax=8, units=None)
    np.testing.assert_array_equal(SphericalHarmonics.of(lbs_like).values, alm)
    with pytest.raises(ValueError, match="coefficients"):
        SphericalHarmonics(alm[:, :-1], lmax=8)
    with pytest.raises(ValueError, match="components"):
        SphericalHarmonics(np.zeros((2, hp.Alm.getsize(8))), lmax=8)


def test_full_layout_of_a_truncated_mmax():
    lmax, mmax = 10, 3
    rng = np.random.default_rng(2)
    size = hp.Alm.getsize(lmax, mmax)
    values = rng.standard_normal((3, size)) + 1j * rng.standard_normal((3, size))
    values[:, : lmax + 1] = values[:, : lmax + 1].real
    sh = SphericalHarmonics(values, lmax, mmax)
    nside = 8
    expected = hp.alm2map(values, nside, lmax=lmax, mmax=mmax)
    np.testing.assert_allclose(sh.full().to_maps(nside), expected, atol=1e-12)
    np.testing.assert_allclose(sh.to_maps(nside), expected, atol=1e-12)


def test_maps_round_trip():
    lmax, nside = 8, 16
    alm = random_teb(lmax, 3)
    maps = hp.alm2map(alm, nside, lmax=lmax)
    sh = SphericalHarmonics.from_maps(maps, lmax=lmax)
    np.testing.assert_allclose(sh.values, alm, atol=1e-6)
    sky = Sky.from_alm(sh, nside)
    assert sky.alm is sh
    np.testing.assert_allclose(sky.q, sh.to_maps(nside)[1], atol=1e-14)


@pytest.mark.parametrize("hwp", [False, True])
def test_gaussian_beam_is_healpy_smoothing(hwp):
    lmax, nside, fwhm = 16, 8, np.deg2rad(10.0)
    alm = random_teb(lmax, 4)
    fields = convolve(alm, sbm_beam.gaussian(fwhm, lmax), nside, range(5), hwp=hwp)
    smoothed = hp.alm2map(hp.smoothalm(alm, fwhm=fwhm, pol=True, inplace=False), nside)
    pol = (2, -4) if hwp else (2, 0)
    np.testing.assert_allclose(fields[0, 0].real, smoothed[0], atol=1e-12)
    np.testing.assert_allclose(
        fields[pol], (smoothed[1] + 1j * smoothed[2]) / 2, atol=1e-12
    )
    for couple, value in fields.items():
        if couple not in [(0, 0), pol, (-pol[0], -pol[1])]:
            np.testing.assert_allclose(value, 0, atol=1e-12)


def test_round_elliptical_beam_is_the_gaussian_beam():
    lmax, fwhm = 32, np.deg2rad(5.0)
    round_ = sbm_beam.elliptical(fwhm, 1.0, lmax, nside=256)
    gauss = sbm_beam.gaussian(fwhm, lmax)
    # the map-based beam differs by the pixelization of its maps (8e-4 at nside 256,
    # falling with nside in T, E and B alike)
    assert np.max(np.abs(round_.values - gauss.values)) < 1e-3 * np.max(
        np.abs(gauss.values)
    )


def test_pencil_beam_components():
    assert sbm_beam.pencil(4).nstokes == 3
    assert sbm_beam.pencil(4, (1.0, 1.0, 0.0, 0.2)).nstokes == 4


def test_rejects_an_invalid_mmax_and_compares_by_identity():
    values = np.zeros((3, hp.Alm.getsize(8)), complex)
    for mmax in (-1, 9):
        with pytest.raises(ValueError, match="mmax"):
            SphericalHarmonics(values, 8, mmax)
    sh = SphericalHarmonics(values, 8)
    assert sh == sh and sh != SphericalHarmonics(values, 8)
    assert isinstance(hash(sh), int)


@pytest.mark.parametrize("hwp", [False, True])
def test_convolve_spins_above_lmax_vanish(hwp):
    """A spin-k map has no multipole below k: the moments k > lmax are zero maps.

    healpy aborts the process ("bad spin") for a spin above lmax, so the convolution
    runs in a subprocess.
    """
    code = f"""
import healpy as hp
import numpy as np
from sbm import beam
lmax, nside = 3, 4
alm = np.random.default_rng(0).standard_normal((3, hp.Alm.getsize(lmax))) + 0j
alm[1:, :2] = 0
gaussian = beam.gaussian(0.1, lmax)
full = beam.convolve(alm, gaussian, nside, (0, 2, 4), hwp={hwp})
low = beam.convolve(alm, gaussian, nside, (0, 2), hwp={hwp})
assert {{k for k in full if abs(k[0]) <= lmax}} == set(low)
for key, value in full.items():
    if abs(key[0]) > lmax:
        assert not np.any(value), key
    else:
        np.testing.assert_array_equal(value, low[key])
print("ok")
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
