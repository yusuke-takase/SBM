"""Reading litebird_sim / Falcons.jl h-maps and the crossing-angle convention.

litebird_sim's TOD model for a detector behind a HWP with Mueller matrix M is

    d = det · R(χ)ᵀ · M · R(χ) · R(ψ) · S,

with R(x) the Stokes rotation by 2x (Q' = cos 2x Q + sin 2x U) and ψ the
orientation angle measured from East towards North. SBM's model is the same
with R(-ψ_SBM) in place of R(ψ), so ψ_SBM = -ψ. These tests check that the
h-maps read by `ScanFields.load_lbs_h_maps` reproduce litebird_sim's TOD
binning, and that feeding litebird_sim's angles to SBM unchanged would mirror
the polarisation.
"""

import h5py
import healpy as hp
import numpy as np
import pytest

from sbm import ScanFields, SignalFields

from .test_hwp_convolution import LMAX, NSIDE, pencil_beam, random_alms, sky_maps

SPINS_N = np.arange(9)
SPINS_M = np.arange(-8, 9, 2)
NSAMPLES = 64


def lbs_rotation(angle):
    """litebird_sim's Stokes rotation, written independently of SBM's tests."""
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]])


def lbs_tod(stokes, detector, mueller, psi, chi):
    return np.array([
        detector @ lbs_rotation(c).T @ mueller @ lbs_rotation(c) @ lbs_rotation(p) @ stokes
        for p, c in zip(psi, chi)
    ])


def write_lbs_h_maps(path, psi, chi, hits=None, couples=None, name="det", fixed_length_strings=False):
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
                value[seen] = np.mean(np.exp(1j * (n * psi[seen] + m * chi[seen])), axis=1)
            f[f"{n},{m}/Re"] = value.real
            f[f"{n},{m}/Im"] = value.imag
    return path


def sbm_h(psi_sbm, alpha):
    """SBM's cross-links of the same samples, as in test_hwp_convolution."""
    return np.array([
        [np.mean(np.exp(-1j * (n * psi_sbm + m * alpha)), axis=1) for n in SPINS_N]
        for m in SPINS_M
    ]).transpose(2, 0, 1)


def samples(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, (2, hp.nside2npix(NSIDE), NSAMPLES))


def test_cross_links_use_minus_psi(tmp_path):
    psi, chi = samples(51)
    hits = np.full(hp.nside2npix(NSIDE), NSAMPLES)
    hits[[0, 5, 11]] = 0
    scan = ScanFields.load_lbs_h_maps(str(write_lbs_h_maps(tmp_path / "h_maps_det_det.h5", psi, chi, hits)))
    np.testing.assert_array_equal(scan.spins_n, SPINS_N)
    np.testing.assert_array_equal(scan.spins_m, SPINS_M)
    np.testing.assert_array_equal(scan.hitmap, hits)
    seen = hits > 0
    np.testing.assert_allclose(scan.h[seen], sbm_h(-psi, chi)[seen], atol=1e-14)
    assert np.all(scan.h[~seen] == 1.0)
    assert (scan.nside, scan.duration, scan.sampling_rate) == (NSIDE, 600.0, 19.0)
    assert scan.ss["coord"] == "Galactic" and scan.ss["name"] == "det"


def test_fixed_length_string_attributes_are_decoded(tmp_path):
    psi, chi = samples(56)
    path = write_lbs_h_maps(tmp_path / "h_maps_det_det.h5", psi, chi, name="000_001_A", fixed_length_strings=True)
    with h5py.File(path, "r") as f:
        assert isinstance(f.attrs["det"], bytes)
    scan = ScanFields.load_lbs_h_maps(str(path))
    assert scan.ss["coord"] == "Galactic" and scan.ss["name"] == "000_001_A"


@pytest.mark.parametrize("use_hwp,mdim", [(False, 3), (True, 3), (True, 5)])
def test_map_making_matches_litebird_sim_tod_binning(tmp_path, use_hwp, mdim):
    psi, chi = samples(52)
    if not use_hwp:
        chi = np.zeros_like(chi)
    sky = random_alms(53)
    detector = np.array([1.0, 1.0, 0.0, 0.0])
    mueller = np.array([[.9, .04, -.03, .02], [.06, .8, -.07, .03],
                        [-.02, .05, -.7, -.04], [.01, -.03, .02, -.8]]) if use_hwp else np.eye(4)
    scan = ScanFields.load_lbs_h_maps(str(write_lbs_h_maps(tmp_path / "h.h5", psi, chi)))
    kwargs = dict(use_hwp=True, mueller=mueller) if use_hwp else {}
    signal = SignalFields.beam_field(scan, mdim, sky, pencil_beam(detector), spin_k=range(LMAX + 1), **kwargs)
    actual = scan.map_make(signal)

    expected = np.zeros_like(actual)
    for pixel, stokes in enumerate(sky_maps(sky).T):
        phase = 4 * chi[pixel] + 2 * psi[pixel]
        design = [np.ones_like(phase), np.cos(phase), np.sin(phase)]
        if mdim == 5:
            extra = 4 * chi[pixel] + 4 * psi[pixel]
            design.extend([np.cos(extra), np.sin(extra)])
        tod = lbs_tod(stokes, detector, mueller, psi[pixel], chi[pixel])
        expected[:, pixel] = np.linalg.lstsq(np.array(design).T, tod, rcond=None)[0][:3]
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)

    if not use_hwp:
        # Without systematics the binned map of an ideal detector is the input
        # sky whatever the orientation convention: nothing more to compare.
        return
    # Feeding litebird_sim's angles to SBM unchanged (ψ_SBM = ψ) describes the
    # mirrored TOD and does not reproduce the binning.
    mirrored = ScanFields.load_lbs_h_maps(str(tmp_path / "h.h5"))
    mirrored.h = sbm_h(psi, chi)
    signal = SignalFields.beam_field(mirrored, mdim, sky, pencil_beam(detector), spin_k=range(LMAX + 1), **kwargs)
    assert np.max(np.abs(mirrored.map_make(signal)[1:] - expected[1:])) > 1e-3


def test_several_files_are_hit_weighted(tmp_path):
    psi, chi = samples(54)
    hits_a = np.full(hp.nside2npix(NSIDE), NSAMPLES // 2)
    hits_a[3] = 0
    a = write_lbs_h_maps(tmp_path / "a.h5", psi[:, :NSAMPLES // 2], chi[:, :NSAMPLES // 2], hits_a, name="a")
    b = write_lbs_h_maps(tmp_path / "b.h5", psi[:, NSAMPLES // 2:], chi[:, NSAMPLES // 2:], name="b")
    both = ScanFields.load_lbs_h_maps([str(a), str(b)])
    assert both.ndet == 2
    np.testing.assert_array_equal(both.hitmap, hits_a + NSAMPLES // 2)
    expected = sbm_h(-psi, chi)
    keep = np.arange(hp.nside2npix(NSIDE)) != 3
    np.testing.assert_allclose(both.h[keep], expected[keep], atol=1e-14)
    np.testing.assert_allclose(both.h[3], sbm_h(-psi[3:4, NSAMPLES // 2:], chi[3:4, NSAMPLES // 2:])[0], atol=1e-14)


def test_incomplete_files_are_rejected(tmp_path):
    psi, chi = samples(55)
    no_hits = write_lbs_h_maps(tmp_path / "a.h5", psi, chi, couples=[(2, 0), (4, 0)])
    with pytest.raises(ValueError, match="hit counts"):
        ScanFields.load_lbs_h_maps(str(no_hits))
    gap = write_lbs_h_maps(tmp_path / "b.h5", psi, chi, couples=[(0, 0), (2, 0), (2, 4), (4, 0)])
    with pytest.raises(ValueError, match="lacks"):
        ScanFields.load_lbs_h_maps(str(gap))
