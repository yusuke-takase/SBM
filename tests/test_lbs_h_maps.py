"""Reading litebird_sim / Falcons.jl h-maps and the crossing-angle convention.

litebird_sim's TOD model for a detector behind a HWP with Mueller matrix M is

    d = det · R(χ)ᵀ · M · R(χ) · R(ψ) · S,

with R(x) the Stokes rotation by 2x (Q' = cos 2x Q + sin 2x U) and ψ the
orientation angle measured from East towards North. SBM's model is the same
with R(-ψ_SBM) in place of R(ψ), so ψ_SBM = -ψ. These tests check that the
h-maps read by `HMaps.load` reproduce litebird_sim's TOD binning, and that
feeding litebird_sim's angles to SBM unchanged would mirror the polarisation.
"""

import h5py
import healpy as hp
import numpy as np
import pytest

from sbm import systematics as sy
from sbm.basis import Basis
from sbm.hmaps import HMaps
from sbm.map_making import map_make
from sbm.sky import Sky

from .test_hwp_convolution import LMAX, NSIDE, pencil_beam, random_alms, sky_maps

SPINS_N = np.arange(9)
SPINS_M = np.arange(-8, 9, 2)
NSAMPLES = 64
NPIX = hp.nside2npix(NSIDE)


def lbs_rotation(angle):
    """litebird_sim's Stokes rotation, written independently of SBM's tests."""
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]])


def lbs_tod(stokes, detector, mueller, psi, chi):
    return np.array(
        [
            detector
            @ lbs_rotation(c).T
            @ mueller
            @ lbs_rotation(c)
            @ lbs_rotation(p)
            @ stokes
            for p, c in zip(psi, chi)
        ]
    )


def write_lbs_h_maps(
    path, psi, chi, hits=None, couples=None, name="det", fixed_length_strings=False
):
    """Write h-maps of the samples (npix, nsamples) in litebird_sim's layout.

    litebird_sim writes variable-length string attributes (read back as ``str``),
    Falcons.jl fixed-length ones (read back by h5py as ``bytes``).
    """
    npix = psi.shape[0]
    hits = np.full(npix, psi.shape[1]) if hits is None else hits
    if couples is None:
        couples = [(n, m) for n in SPINS_N for m in SPINS_M]
    text = np.bytes_ if fixed_length_strings else str
    with h5py.File(path, "w") as f:
        f.attrs["coordinate_system"] = text("Galactic")
        f.attrs["det"] = text(name)
        f.attrs["detector_split"] = text("full")
        f.attrs["time_split"] = text("full")
        f.attrs["duration_s"] = 600.0
        f.attrs["sampling_rate_Hz"] = 19.0
        seen = hits > 0
        for n, m in couples:
            value = np.zeros(npix, dtype=complex)
            if (n, m) == (0, 0):
                value[:] = hits
            else:
                value[seen] = np.mean(
                    np.exp(1j * (n * psi[seen] + m * chi[seen])), axis=1
                )
            f[f"{n},{m}/Re"] = value.real
            f[f"{n},{m}/Im"] = value.imag
    return path


def samples(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, (2, NPIX, NSAMPLES))


def sbm_hmaps(psi_sbm, chi):
    """h-maps of the samples computed by SBM in its own convention."""
    pixels = np.repeat(np.arange(psi_sbm.shape[0]), psi_sbm.shape[1])
    couples = [(int(n), int(m)) for n in SPINS_N for m in SPINS_M]
    return HMaps.from_samples(
        NSIDE, pixels, psi_sbm.ravel(), chi.ravel(), couples=couples
    )


def test_load_uses_minus_psi(tmp_path):
    psi, chi = samples(51)
    hits = np.full(NPIX, NSAMPLES)
    hits[[0, 5, 11]] = 0
    hm = HMaps.load(write_lbs_h_maps(tmp_path / "h_maps_det_det.h5", psi, chi, hits))
    np.testing.assert_array_equal(hm.hits, hits)
    seen = hits > 0
    expected = sbm_hmaps(-psi, chi)
    for n in SPINS_N:
        for m in SPINS_M:
            np.testing.assert_allclose(
                hm.h(n, m)[seen], expected.h(n, m)[seen], atol=1e-14
            )
            if (n, m) != (0, 0):
                assert np.all(hm.h(n, m)[~seen] == 0)
    assert (hm.nside, hm.duration_s, hm.sampling_rate_hz) == (NSIDE, 600.0, 19.0)
    assert (hm.coordinate_system, hm.name) == ("Galactic", "det")


def test_fixed_length_string_attributes_are_decoded(tmp_path):
    psi, chi = samples(56)
    path = write_lbs_h_maps(
        tmp_path / "h.h5", psi, chi, name="000_001_A", fixed_length_strings=True
    )
    with h5py.File(path, "r") as f:
        assert isinstance(f.attrs["det"], bytes)
    hm = HMaps.load(path)
    assert (hm.coordinate_system, hm.name) == ("Galactic", "000_001_A")


@pytest.mark.parametrize(
    "hwp,basis",
    [
        (False, Basis.iqu(hwp=False)),
        (True, Basis.iqu(hwp=True)),
        (True, Basis.iqu(hwp=True).extend([(4, -4)])),
    ],
)
def test_map_making_matches_litebird_sim_tod_binning(tmp_path, hwp, basis):
    psi, chi = samples(52)
    if not hwp:
        chi = np.zeros_like(chi)
    sky = random_alms(53)
    detector = np.array([1.0, 1.0, 0.0, 0.0])
    mueller = (
        np.array(
            [
                [0.9, 0.04, -0.03, 0.02],
                [0.06, 0.8, -0.07, 0.03],
                [-0.02, 0.05, -0.7, -0.04],
                [0.01, -0.03, 0.02, -0.8],
            ]
        )
        if hwp
        else np.eye(4)
    )
    hm = HMaps.load(write_lbs_h_maps(tmp_path / "h.h5", psi, chi))
    kwargs = dict(hwp=True, mueller=mueller) if hwp else {}
    model = sy.beam(sky, pencil_beam(detector), NSIDE, range(LMAX + 1), **kwargs)
    actual = map_make((hm, model), basis).iqu

    expected = np.zeros_like(actual)
    for pixel, stokes in enumerate(sky_maps(sky).T):
        phase = 4 * chi[pixel] + 2 * psi[pixel]
        design = [np.ones_like(phase), np.cos(phase), np.sin(phase)]
        if len(basis.couples) == 3:
            extra = 4 * chi[pixel] + 4 * psi[pixel]
            design.extend([np.cos(extra), np.sin(extra)])
        tod = lbs_tod(stokes, detector, mueller, psi[pixel], chi[pixel])
        expected[:, pixel] = np.linalg.lstsq(np.array(design).T, tod, rcond=None)[0][:3]
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)

    if not hwp:
        # Without systematics the binned map of an ideal detector is the input
        # sky whatever the orientation convention: nothing more to compare.
        return
    # Feeding litebird_sim's angles to SBM unchanged (ψ_SBM = ψ) describes the
    # mirrored TOD and does not reproduce the binning.
    mirrored = map_make((sbm_hmaps(psi, chi), model), basis).iqu
    assert np.max(np.abs(mirrored[1:] - expected[1:])) > 1e-3


def test_several_detectors_are_hit_weighted(tmp_path):
    psi, chi = samples(54)
    half = NSAMPLES // 2
    hits_a = np.full(NPIX, half)
    hits_a[3] = 0
    a = write_lbs_h_maps(
        tmp_path / "a.h5", psi[:, :half], chi[:, :half], hits_a, name="a"
    )
    b = write_lbs_h_maps(tmp_path / "b.h5", psi[:, half:], chi[:, half:], name="b")
    both = HMaps.combine([HMaps.load(a), HMaps.load(b)])
    np.testing.assert_array_equal(both.hits, hits_a + half)
    expected = sbm_hmaps(-psi, chi)
    keep = np.arange(NPIX) != 3
    only_b = sbm_hmaps(-psi[3:4, half:], chi[3:4, half:])
    for n, m in [(2, 0), (2, -4), (4, 8), (0, 4)]:
        np.testing.assert_allclose(
            both.h(n, m)[keep], expected.h(n, m)[keep], atol=1e-14
        )
        # pixel 3 is seen by b only
        np.testing.assert_allclose(both.h(n, m)[3], only_b.h(n, m)[0], atol=1e-14)


def test_incomplete_files_are_rejected(tmp_path):
    psi, chi = samples(55)
    no_hits = write_lbs_h_maps(tmp_path / "a.h5", psi, chi, couples=[(2, 0), (4, 0)])
    with pytest.raises(ValueError, match="hit counts"):
        HMaps.load(no_hits)
    gap = HMaps.load(
        write_lbs_h_maps(
            tmp_path / "b.h5", psi, chi, couples=[(0, 0), (2, 0), (2, 4), (4, 0)]
        )
    )
    model = sy.sky_signal(Sky.from_maps(*np.ones((3, NPIX))), hwp=True)
    with pytest.raises(KeyError, match="lack"):
        map_make((gap, model), Basis.iqu(hwp=True))
