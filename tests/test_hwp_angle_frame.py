"""Consistency of the HWP angle conventions for the wedge and the HWP-IP (hwp_angle_frame).

Physical model, with the crossing angle psi_0 and the focal-plane HWP angle a both
counted like SBM's crossing angle without HWP: the polarization is modulated as
P exp(i(2 psi_0 + 4a)), the HWP-fixed wedge deflects the beam along psi_0 + a + chi,
and the 4f HWP-IP is epsilon cos(4 phi_fp + phi_qi) (Patanchon et al. 2024) with
phi_fp = -a the focal-plane HWP angle counted in the standard sense. The scan fields of
the "telescope" frame store (psi, phi) = (psi_0, -a), those of the "sky" frame
(-psi_0, -(psi_0 + a)).
"""

import healpy as hp
import numpy as np
import pytest

import sbm
from sbm import ScanFields, SignalFields
from sbm.signal_fields import (
    HWP_ANGLE_FRAMES,
    hwp_ip_route_spins,
    wedge_offset_direction_spin,
)

POL = (2, -4)


def _consistent(frame, wedge=None):
    """IP = 2 pol - 4 wedge_T between spin vectors, IP the route carrying exp(-i phi_qi).

    4a = 2 (2 psi_0 + 4a) - 4 (psi_0 + a): the relation holds for the physical angles,
    hence for any linear choice of the angle variables of the scan fields. The route
    exp(-i phi_qi) exp(4ia) gives epsilon cos(4 phi_fp + phi_qi) with phi_fp = -a.
    """
    w = np.array(wedge_offset_direction_spin(frame) if wedge is None else wedge)
    expected = tuple(int(x) for x in 2 * np.array(POL) - 4 * w)
    return expected == hwp_ip_route_spins(frame)[0]


@pytest.mark.parametrize("frame", ["sky", "telescope"])
def test_frames_are_self_consistent(frame):
    assert _consistent(frame)


def test_legacy_frame_is_inconsistent():
    # kept only to reproduce earlier results
    assert not _consistent("legacy")


@pytest.mark.parametrize("frame", ["sky", "telescope"])
def test_consistency_check_detects_a_flipped_wedge(frame):
    dn, dm = wedge_offset_direction_spin(frame)
    assert not _consistent(frame, wedge=(-dn, -dm))


def test_spins_per_frame():
    assert wedge_offset_direction_spin("legacy") == (1, 1)
    assert wedge_offset_direction_spin("sky") == (0, -1)
    assert wedge_offset_direction_spin("telescope") == (1, -1)
    assert hwp_ip_route_spins("legacy") == ((4, -4), (-4, 4))
    assert hwp_ip_route_spins("sky") == ((4, -4), (-4, 4))
    assert hwp_ip_route_spins("telescope") == ((0, -4), (0, 4))


def test_routes_are_conjugate_pairs():
    for frame in HWP_ANGLE_FRAMES:
        (na, ma), (nb, mb) = hwp_ip_route_spins(frame)
        assert (na, ma) == (-nb, -mb)


def test_invalid_frame_raises():
    with pytest.raises(ValueError):
        wedge_offset_direction_spin("instrument")
    with pytest.raises(ValueError):
        hwp_ip_route_spins("instrument")


# ---- wedge against a TOD simulated with the physical angles -------------------------

NSIDE_TOD, NSAMPLES = 2, 400
SPINS_N_TOD, SPINS_M_TOD = np.arange(0, 9), np.arange(-10, 11)
RHO, CHI = 1e-3, 0.7


def frame_angles(frame, psi_0, a):
    """Angles (psi, phi) of the scan fields of a frame (see the module docstring)."""
    return {"telescope": (psi_0, -a), "sky": (-psi_0, -(psi_0 + a))}[frame]


def scan_of_samples(psi, phi):
    """Scan fields h_{n,m} = <exp(-i(n psi + m phi))> of the samples (npix, nsamples)."""
    scan = ScanFields()
    scan.nside, scan.npix = NSIDE_TOD, psi.shape[0]
    scan.hitmap = np.full(scan.npix, psi.shape[1])
    scan.spins_n, scan.spins_m = SPINS_N_TOD, SPINS_M_TOD
    scan.h = np.array(
        [[np.mean(np.exp(-1j * (n * psi + m * phi)), axis=1) for n in SPINS_N_TOD] for m in SPINS_M_TOD]
    ).transpose(2, 0, 1)
    return scan


def offset(f, f_bar, beta):
    """First-order change of a field for a pointing offset RHO along beta.

    beta is counted like SBM's crossing angle without HWP, as for the absolute pointing
    offset; f and f_bar are the spin-raising and spin-lowering derivatives.
    """
    return -(RHO / 2) * (np.exp(1j * beta) * f[:, None] + np.exp(-1j * beta) * f_bar[:, None])


@pytest.mark.parametrize("mdim", [3, 5, 9])
@pytest.mark.parametrize("frame", ["sky", "telescope"])
def test_wedge_matches_tod(frame, mdim):
    rng = np.random.default_rng(11)
    npix = hp.nside2npix(NSIDE_TOD)
    psi_0, a = rng.uniform(-np.pi, np.pi, (2, npix, NSAMPLES))
    temp = rng.standard_normal(npix)
    pol, eth_temp, eth_pol, o_eth_pol = rng.standard_normal((4, npix)) + 1j * rng.standard_normal((4, npix))

    theta = 2 * psi_0 + 4 * a  # polarization modulation
    gamma = psi_0 + a  # wedge offset direction without chi
    beta = gamma + CHI
    tod = (
        temp[:, None]
        + offset(eth_temp, eth_temp.conj(), beta).real
        + np.real((pol[:, None] + offset(eth_pol, o_eth_pol, beta)) * np.exp(1j * theta))
    )
    expected = np.zeros((3, npix))
    for p in range(npix):
        design = [np.ones(NSAMPLES), np.cos(theta[p]), -np.sin(theta[p])]
        if mdim >= 5:
            design += [np.cos(gamma[p]), np.sin(gamma[p])]
        if mdim == 9:
            design += [np.cos(theta[p] + gamma[p]), np.sin(theta[p] + gamma[p])]
            design += [np.cos(theta[p] - gamma[p]), np.sin(theta[p] - gamma[p])]
        expected[:, p] = np.linalg.lstsq(np.array(design).T, tod[p], rcond=None)[0][:3]

    scan = scan_of_samples(*frame_angles(frame, psi_0, a))
    signal = SignalFields.circular_pointing_field(
        scan, mdim, RHO, CHI, temp, pol, eth_temp, eth_pol, o_eth_pol, hwp_angle_frame=frame
    )
    actual = np.array(scan.map_make(signal))
    ideal = np.array([temp, pol.real, pol.imag])
    residual = np.max(np.abs(expected - ideal))
    if mdim == 9:
        # every first-order term has a template: the input comes back
        assert residual < 1e-12
        np.testing.assert_allclose(actual, ideal, rtol=0, atol=1e-12)
    else:
        assert residual > 1e-5
        assert np.max(np.abs(actual - expected)) < 1e-8 * residual


@pytest.mark.parametrize("mdim", [3, 5])
@pytest.mark.parametrize("frame", ["sky", "telescope", "legacy"])
def test_hwp_ip_matches_tod(frame, mdim):
    """epsilon T cos(4 phi_fp + phi_qi), with the same phi_qi in every frame."""
    eps, phi_qi = 1e-2, 0.3
    rng = np.random.default_rng(12)
    npix = hp.nside2npix(NSIDE_TOD)
    psi_0, a = rng.uniform(-np.pi, np.pi, (2, npix, NSAMPLES))
    temp = rng.standard_normal(npix)
    pol = rng.standard_normal(npix) + 1j * rng.standard_normal(npix)

    theta = 2 * psi_0 + 4 * a
    phi_fp = -a
    leakage = eps * temp[:, None] * np.cos(4 * phi_fp + phi_qi)
    tod = temp[:, None] + np.real(pol[:, None] * np.exp(1j * theta)) + leakage
    ideal = np.array([temp, pol.real, pol.imag])
    expected = np.zeros((3, npix))
    for p in range(npix):
        design = [np.ones(NSAMPLES), np.cos(theta[p]), -np.sin(theta[p])]
        if mdim == 5:
            design += [np.cos(4 * a[p]), np.sin(4 * a[p])]
        expected[:, p] = np.linalg.lstsq(np.array(design).T, tod[p], rcond=None)[0][:3]

    # the HWP-IP of "legacy" is that of "sky", with the same angles
    scan = scan_of_samples(*frame_angles("telescope" if frame == "telescope" else "sky", psi_0, a))
    # hwp_ip_field models the leakage alone
    actual = np.array(scan.map_make(SignalFields.hwp_ip_field(scan, mdim, eps, phi_qi, temp, hwp_angle_frame=frame)))
    residual = np.max(np.abs(expected - ideal))
    if mdim == 5:
        assert residual < 1e-12
        np.testing.assert_allclose(actual, 0.0, rtol=0, atol=1e-12)
    else:
        assert residual > 1e-4
        assert np.max(np.abs(actual - (expected - ideal))) < 1e-8 * residual


def test_hwp_ip_field_telescope_frame_runs():
    """mdim=5 HWP-IP with the telescope frame builds and solves on the HWP test scan."""
    from pathlib import Path

    sf = sbm.read_scanfiled(str(Path(__file__).parent / "nside_32_boresight_hwp.h5"))
    npix = hp.nside2npix(sf.nside)
    rng = np.random.default_rng(0)
    temp = rng.standard_normal(npix)
    # the test scan stores n <= 6, |m| <= 8: legacy mdim=5 needs h_{8,-8}, telescope mdim=5 only h_{0,8}
    for frame, mdim in (("legacy", 3), ("telescope", 3), ("telescope", 5)):
        signal = sbm.SignalFields.hwp_ip_field(sf, mdim, 1e-5, 0.3, temp, hwp_angle_frame=frame)
        out = sf.map_make(signal)
        assert np.all(np.isfinite(out[1])) and np.all(np.isfinite(out[2]))


def test_hwp_ip_telescope_mdim5_removes_leakage():
    """The (0,+-4) HWP-IP template of the telescope frame is complete for mdim=5."""
    from pathlib import Path

    sf = sbm.read_scanfiled(str(Path(__file__).parent / "nside_32_boresight_hwp.h5"))
    npix = hp.nside2npix(sf.nside)
    temp = np.random.default_rng(1).standard_normal(npix)
    leak3 = np.array(sf.map_make(SignalFields.hwp_ip_field(sf, 3, 1e-2, 0.3, temp, hwp_angle_frame="telescope")))
    leak5 = np.array(sf.map_make(SignalFields.hwp_ip_field(sf, 5, 1e-2, 0.3, temp, hwp_angle_frame="telescope")))
    seen = sf.hitmap > 0
    p3 = np.sqrt(np.mean(leak3[1, seen] ** 2 + leak3[2, seen] ** 2))
    p5 = np.sqrt(np.mean(leak5[1, seen] ** 2 + leak5[2, seen] ** 2))
    assert p3 > 0
    assert p5 < 1e-10 * p3
