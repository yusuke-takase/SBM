"""Regression and independent TOD checks for SignalFields.random_pointing_field."""

import json
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pytest

from sbm import ScanFields, SignalFields


SPINS = np.array([(3, -4), (-1, 4), (1, 0), (5, -8), (-3, 8)])
REFERENCE = Path(__file__).parent / 'reference' / 'random_pointing.npz'


def make_scan(hitmap, spins, harmonics):
    scan = ScanFields()
    scan.hitmap = np.asarray(hitmap).copy()
    scan.npix = scan.hitmap.size
    scan.nside = hp.npix2nside(scan.npix)
    scan.spins_n = np.arange(11)
    scan.spins_m = np.arange(-16, 17)
    scan.h = np.zeros((scan.npix, 33, 11), complex)
    for (n, m), values in zip(spins, harmonics):
        scan.h[:, m + 16, n] = values
        if n == 0:
            scan.h[:, -m + 16, n] = values.conj()
    return scan


@pytest.fixture
def reference():
    with np.load(REFERENCE, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def test_original_cartesian_implementation(reference):
    d = reference
    scan = make_scan(d['hitmap'], d['spins'], d['harmonics'])
    fields = SignalFields.random_pointing_field(
        scan, 3, float(d['sigma_rad']), d['P'], d['eth_P'], d['o_eth_P'],
        seed=int(d['seed']),
    )
    assert isinstance(fields, SignalFields)
    assert fields.field_name == 'random_pointing_field'
    np.testing.assert_array_equal(fields.spin_n_basis, [0, -2, 2])
    np.testing.assert_array_equal(fields.spin_m_basis, [0, 4, -4])
    np.testing.assert_allclose(scan.map_make(fields), d['output_iqu'], rtol=1e-12, atol=1e-12)
    reconstructed = scan.map_make(fields, only_iqu=False)
    np.testing.assert_allclose(reconstructed.extract_iqu(), d['output_iqu'], atol=1e-12)


def test_exact_tod_projection():
    rng = np.random.default_rng(29)
    count = 97
    psi, varphi = rng.uniform(-np.pi, np.pi, (2, count))
    x, y = rng.normal(size=(2, count))
    bar_zeta = 2 * (y + 1j * x)
    eth_p, o_eth_p = -0.3 + 0.5j, 0.2 - 0.4j
    pol = 1.7 - 0.6j
    phase = 2 * psi - 4 * varphi
    direct = -0.125 * (
        bar_zeta * np.exp(1j * (3 * psi - 4 * varphi)) * eth_p
        + bar_zeta.conj() * np.exp(1j * (psi - 4 * varphi)) * o_eth_p
        + bar_zeta * np.exp(1j * (-psi + 4 * varphi)) * np.conj(o_eth_p)
        + bar_zeta.conj() * np.exp(1j * (-3 * psi + 4 * varphi)) * np.conj(eth_p)
    )
    direct += np.real(pol * np.exp(1j * phase))
    design = np.column_stack((np.ones(count), np.cos(phase), -np.sin(phase)))
    expected = np.linalg.solve(design.T @ design, design.T @ direct.real)
    spins = np.array([(n, m) for n in range(11) for m in range(-16, 17)])
    harmonics = np.array([np.full(12, np.mean(np.exp(-1j * (n * psi + m * varphi))))
                          for n, m in spins])
    scan = make_scan(np.full(12, count), spins, harmonics)
    weights = np.exp(-1j * (SPINS @ np.array([psi, varphi])))
    moments = (np.tile(weights @ x, (12, 1)), np.tile(weights @ y, (12, 1)))
    fields = SignalFields.random_pointing_field(
        scan, 3, 1.0, np.full(12, pol), np.full(12, eth_p),
        np.full(12, o_eth_p), moments=moments,
    )
    np.testing.assert_allclose(scan.map_make(fields), np.tile(expected[:, None], (1, 12)), atol=2e-14)


def test_zero_width_and_unobserved_pixels(reference):
    d = reference
    scan = make_scan(d['hitmap'], d['spins'], d['harmonics'])
    scan.hitmap[0] = 0
    scan.h[0] = np.nan
    fields = SignalFields.random_pointing_field(scan, 3, 0, d['P'], d['eth_P'], d['o_eth_P'])
    expected = np.array([np.zeros(12), d['P'].real, d['P'].imag])
    expected[:, 0] = 0
    np.testing.assert_allclose(scan.map_make(fields), expected, atol=1e-12)


def test_seed_reproducibility_and_global_rng(reference):
    d = reference
    scan = make_scan(d['hitmap'], d['spins'], d['harmonics'])
    state = np.random.get_state()
    args = (scan, 3, float(d['sigma_rad']), d['P'], d['eth_P'], d['o_eth_P'])
    first = SignalFields.random_pointing_field(*args, seed=1).coupled_fields
    np.testing.assert_array_equal(first, SignalFields.random_pointing_field(*args, seed=1).coupled_fields)
    assert not np.array_equal(first, SignalFields.random_pointing_field(*args, seed=2).coupled_fields)
    after = np.random.get_state()
    for left, right in zip(state, after):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('mdim,sigma', [(5, 1e-4), (3, -1), (3, np.nan), (3, np.inf)])
def test_invalid_parameters(reference, mdim, sigma):
    d = reference
    scan = make_scan(d['hitmap'], d['spins'], d['harmonics'])
    with pytest.raises(ValueError):
        SignalFields.random_pointing_field(scan, mdim, sigma, d['P'], d['eth_P'], d['o_eth_P'])


def test_invalid_moments_and_missing_harmonics(reference):
    d = reference
    scan = make_scan(d['hitmap'], d['spins'], d['harmonics'])
    args = (scan, 3, float(d['sigma_rad']), d['P'], d['eth_P'], d['o_eth_P'])
    with pytest.raises(ValueError):
        SignalFields.random_pointing_field(*args, moments=(np.zeros((12, 4)), np.zeros((12, 5))))
    with pytest.raises(ValueError):
        SignalFields.random_pointing_field(*args, seed=1, moments=(np.zeros((12, 5)), np.zeros((12, 5))))
    scan.spins_n = scan.spins_n[:5]
    with pytest.raises(AssertionError, match='spin_n'):
        SignalFields.random_pointing_field(*args, seed=1)


def test_saved_nside256_bb_spectrum():
    """Replay seed 23000001 against a pre-existing computed BB spectrum.

    Set SBM_RANDOM_POINTING_REFERENCE_ROOT to the original study directory.
    The large scan/sky inputs are external; the expected spectrum is bundled.
    """
    root_value = os.environ.get('SBM_RANDOM_POINTING_REFERENCE_ROOT')
    if root_value is None:
        pytest.skip('set SBM_RANDOM_POINTING_REFERENCE_ROOT for the full nside256 replay')
    root = Path(root_value)
    base = root / 'results/cache/litebird_reformation_2026'
    with np.load(base / 'hwp_copointed_pair_nside256_fwhm0p5/scan_h2m4_h4m8_h6m12_h8m16.npz') as data:
        hitmap = data['hitmap']
        harmonics = {(n, m): data[name] / hitmap for n, m, name in (
            (2, -4, 'H2m4_sum'), (4, -8, 'H4m8_sum'),
            (6, -12, 'H6m12_sum'), (8, -16, 'H8m16_sum'))}
    with np.load(base / 'hwp_copointed_pair_explicit_xy_nside256/scan_xy_pseudocovariance_harmonics.npz') as data:
        np.testing.assert_array_equal(hitmap, data['hitmap'])
        harmonics.update({tuple(spin): values / hitmap
                          for spin, values in zip(data['spins'], data['harmonic_sums'])})

    # Sparse cache adapter retains the existing ScanFields map maker, without
    # allocating every unused (n,m) combination in a dense multi-GB scan cube.
    class CachedScanFields(ScanFields):
        def get_xlink(self, n, m):
            if (n, m) == (0, 0):
                return np.ones(self.npix, complex)
            if (n, m) in harmonics:
                return harmonics[(n, m)]
            return harmonics[(-n, -m)].conj()

    scan = CachedScanFields()
    scan.hitmap = hitmap
    scan.npix = len(hitmap)
    scan.nside = hp.npix2nside(scan.npix)
    sky = np.load(root / 'results/cache/no_hwp_nside256_sbm_fwhm0p5/input_cmb_synfast_fwhm0p5deg_r0_seed20260924.npy')
    lmax = 767
    dq = hp.alm2map_der1(hp.map2alm(sky[1], lmax=lmax), scan.nside, lmax=lmax)
    du = hp.alm2map_der1(hp.map2alm(sky[2], lmax=lmax), scan.nside, lmax=lmax)
    p = sky[1] + 1j * sky[2]
    eth = dq[2] + du[1] - 1j * (dq[1] - du[2])
    o_eth = dq[2] - du[1] + 1j * (dq[1] + du[2])
    with np.load(REFERENCE.parent / 'random_pointing_saved_bb.npz') as expected:
        fields = SignalFields.random_pointing_field(
            scan, 3, float(expected['sigma_rad']), p, eth, o_eth, seed=int(expected['seed']))
        residual = np.array([np.zeros(scan.npix), sky[1], sky[2]]) - scan.map_make(fields)
        actual_bb = hp.anafast(residual, lmax=lmax, pol=True)[2]
        np.testing.assert_allclose(actual_bb, expected['bb'], rtol=1e-8, atol=1e-20)
