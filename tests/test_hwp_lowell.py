"""Independent real-space checks of the optional I-input low-ell HWP path."""

import ducc0
import healpy as hp
import numpy as np
import pytest

pytest.importorskip("sympy", reason="lowell_hwp='full_sum' needs sympy")

from sbm import systematics as sy
from sbm.basis import Basis
from sbm.beam import beam_transfer, convolve
from sbm.hwp_lowell import _beam_source, prepare_lowell_beam
from sbm.map_making import map_make
from .test_hwp_convolution import (
    LMAX, NSIDE, hmaps_of, random_alms, reconstruct, rotation,
)


def lowell_sky():
    monopole, dipole = .7, np.array([.31, -.22, .17])
    alm = np.zeros((3, hp.Alm.getsize(LMAX)), complex)
    alm[0, 0] = monopole*np.sqrt(4*np.pi)
    alm[0, hp.Alm.getidx(LMAX, 1, 0)] = dipole[2]*np.sqrt(4*np.pi/3)
    alm[0, hp.Alm.getidx(LMAX, 1, 1)] = -(dipole[0]-1j*dipole[1])*np.sqrt(2*np.pi/3)
    return alm, monopole, dipole


def beam_fixture():
    # Signed basis combinations span asymmetric and cross-polarized responses.
    beam = random_alms(60, components=3)
    beam[0, 0] = 1/np.sqrt(4*np.pi)
    beam[0, hp.Alm.getidx(LMAX, 1, 0)] = .07
    beam[0, hp.Alm.getidx(LMAX, 1, 1)] = .03+.02j
    return beam


def realspace_tod(beam, monopole, dipole, theta, phi, psi, alpha, matrix):
    """Integrate the actual finite-band IQU beam with real Stokes rotations.

    Neither the shifted/Gaunt filters nor the complex Mueller matrix are used.
    The same spatial model is explicit here: M is uniform in Ludwig-3 coordinates.
    """
    ntheta, nphi = 16, 24
    x, w = np.polynomial.legendre.leggauss(ntheta)
    th = np.arccos(x[::-1])[:, None]
    ph = np.arange(nphi)[None, :]*(2*np.pi/nphi)
    weight = np.broadcast_to(w[::-1, None]*(2*np.pi/nphi), (ntheta, nphi))
    xyz = np.array(np.broadcast_arrays(
        np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)
    ))
    options = dict(lmax=LMAX, ntheta=ntheta, nphi=nphi, geometry="GL", nthreads=1)
    intensity = ducc0.sht.experimental.synthesis_2d(alm=beam[:1], spin=0, **options)[0]
    q, u = ducc0.sht.experimental.synthesis_2d(alm=beam[1:3], spin=2, **options)
    p_ludwig = (q+1j*u)*np.exp(2j*ph)
    maps = np.array([intensity, p_ludwig.real, p_ludwig.imag])
    mean = np.einsum("cij,ij->c", maps, weight)
    moment = np.einsum("cij,kij,ij->ck", maps, xyz, weight)
    vx, vy, vz = dipole
    dx = np.cos(theta)*(np.cos(phi)*vx+np.sin(phi)*vy)-np.sin(theta)*vz
    dy = -np.sin(phi)*vx+np.cos(phi)*vy
    dz = np.sin(theta)*(np.cos(phi)*vx+np.sin(phi)*vy)+np.cos(theta)*vz
    local = np.array([np.cos(psi)*dx-np.sin(psi)*dy,
                      np.sin(psi)*dx+np.cos(psi)*dy, dz])
    integrals = monopole*mean[:, None]+moment @ local
    return np.array([
        response @ (rotation(-a) @ matrix @ rotation(a))[:3, 0]
        for response, a in zip(integrals.T, alpha)
    ])


def test_full_sum_against_realspace_tod_and_map():
    sky, monopole, dipole = lowell_sky()
    beam = beam_fixture()
    matrix = np.zeros((4, 4))
    matrix[0, 0], matrix[1, 0], matrix[2, 0] = .9, .013, -.008
    npix, hits = hp.nside2npix(NSIDE), 32
    psi, alpha = np.random.default_rng(61).uniform(-np.pi, np.pi, (2, npix, hits))
    pixels = np.repeat(np.arange(npix), hits)
    theta, phi = hp.pix2ang(NSIDE, pixels)
    expected = realspace_tod(beam, monopole, dipole, theta, phi,
                             psi.ravel(), alpha.ravel(), matrix)
    fields = convolve(sky, beam, NSIDE, range(5), hwp=True, mueller=matrix, lowell_hwp="full_sum")
    actual = reconstruct(fields, psi.ravel(), alpha.ravel(), pixels)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    for (n, h), field in fields.items():
        np.testing.assert_array_equal(field.conj(), fields[-n, -h])
    model = sy.beam(sky, beam, NSIDE, range(5), hwp=True, mueller=matrix, lowell_hwp="full_sum")
    expected_map = np.empty((3, npix))
    for pixel in range(npix):
        phase = 4*alpha[pixel]-2*psi[pixel]
        design = np.array([np.ones(hits), np.cos(phase), np.sin(phase)]).T
        expected_map[:, pixel] = np.linalg.lstsq(
            design, expected.reshape(npix, hits)[pixel], rcond=None
        )[0]
    actual_map = map_make((hmaps_of(psi, alpha), model), Basis.iqu(hwp=True)).iqu
    np.testing.assert_allclose(actual_map, expected_map, rtol=2e-11, atol=2e-12)


def test_full_sum_restores_odd_crossing_correction_without_adding_other_terms():
    sky, monopole, dipole = lowell_sky()
    beam = beam_fixture()
    matrix = np.zeros((4, 4))
    matrix[1, 0], matrix[2, 0] = .013, -.008
    fields = convolve(sky, beam, NSIDE, (0, 2, 4), hwp=True, mueller=matrix, lowell_hwp="full_sum")
    assert np.max(abs(fields[1, -2])) > 1e-4
    pixels = np.arange(hp.nside2npix(NSIDE))
    theta, phi = hp.pix2ang(NSIDE, pixels)
    psi, alpha = np.random.default_rng(62).uniform(-np.pi, np.pi, (2, pixels.size))
    expected = realspace_tod(beam, monopole, dipole, theta, phi, psi, alpha, matrix)
    np.testing.assert_allclose(reconstruct(fields, psi, alpha), expected,
                               rtol=2e-11, atol=2e-12)


def test_beam_source_is_the_spin_beam_transfer():
    beam = beam_fixture()
    for n, h in ((0, 2), (1, 2), (1, -2)):
        component = 2 if h == 2 else 1
        expected = beam_transfer(beam, n, h)[component]
        expected[:2] = 0
        np.testing.assert_array_equal(_beam_source(beam, n, h), expected)
    first = prepare_lowell_beam(beam)
    beam[1, hp.Alm.getidx(LMAX, 2, 2)] += .1
    assert not np.array_equal(first[0, 2], prepare_lowell_beam(beam)[0, 2])


def test_lowell_mode_validation_and_unchanged_high_ell_and_identity():
    sky, beam = random_alms(63, components=3), beam_fixture()
    matrix = np.random.default_rng(64).normal(size=(4, 4))
    for input_sky, mueller in ((sky, matrix), (lowell_sky()[0], np.eye(4))):
        baseline = convolve(input_sky, beam, NSIDE, (0, 2, 4), hwp=True, mueller=mueller)
        actual = convolve(input_sky, beam, NSIDE, (0, 2, 4), hwp=True, mueller=mueller,
                          lowell_hwp="full_sum")
        assert actual.keys() == baseline.keys()
        for pair in baseline:
            np.testing.assert_array_equal(actual[pair], baseline[pair])
    with pytest.raises(ValueError, match="lowell_hwp"):
        convolve(sky, beam, NSIDE, (0, 2), hwp=True, lowell_hwp="ell1_2")
    with pytest.raises(ValueError, match="hwp"):
        convolve(sky, beam, NSIDE, (0, 2), lowell_hwp="full_sum")
