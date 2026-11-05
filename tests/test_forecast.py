import numpy as np

import sbm
from sbm.tools import (
    _get_likelihood_gradient,
    load_fiducial_cl_lens,
    load_fiducial_cl_tens,
)

LMAX = 191


def _spectra():
    ell = np.arange(2, LMAX + 1)
    cl_tens = load_fiducial_cl_tens(r=1.0, lmax=LMAX)[2][2:]
    cl_lens = load_fiducial_cl_lens(lmax=LMAX)[2][2:]
    return ell, cl_tens, cl_lens


def _full(cl):
    out = np.zeros(LMAX + 1)
    out[2:] = cl
    return out


def test_forecast_exact_zero_residual():
    assert sbm.forecast(np.zeros(LMAX + 1))["delta_r"] == 0.0


def test_forecast_display_grid_of_a_zero_bias():
    # the grid is scaled by rmin when the bias is zero, and must not collapse for rmin = 0
    for rmin in (1e-10, 0.0):
        out = sbm.forecast(np.zeros(LMAX + 1), rmin=rmin, rresol=50)
        assert out["delta_r"] == 0.0
        assert out["grid_r"][0] > 0.0 and np.all(np.diff(out["grid_r"]) > 0.0)
        assert np.all(np.isfinite(out["likelihood"]))


def test_forecast_exact_matches_linearised_bias():
    # residual much smaller than lensing: delta_r = sum w C^t dC / C_lens^2 / sum w (C^t)^2 / C_lens^2
    # (the linearisation error is of order r C^t / C_lens at low ell, ~1e-5 here)
    ell, cl_tens, cl_lens = _spectra()
    dcl = 1e-6 * cl_lens
    w = 2.0 * ell + 1.0
    lin = np.sum(w * cl_tens * dcl / cl_lens**2) / np.sum(w * cl_tens**2 / cl_lens**2)
    exact = sbm.forecast(_full(dcl))["delta_r"]
    assert abs(exact / lin - 1.0) < 1e-4


def test_forecast_exact_is_stationary_and_matches_minuit():
    ell, cl_tens, cl_lens = _spectra()
    dcl = 0.3 * cl_lens
    exact = sbm.forecast(_full(dcl))["delta_r"]
    zero = np.zeros_like(cl_lens)
    grad = _get_likelihood_gradient(exact, ell, cl_tens, cl_lens, dcl, zero, 1.0)
    scale = _get_likelihood_gradient(0.0, ell, cl_tens, cl_lens, dcl, zero, 1.0)
    assert abs(grad) < 1e-9 * abs(scale)
    minuit = sbm.forecast(_full(dcl), method="minuit")["delta_r"]
    assert abs(minuit / exact - 1.0) < 1e-3


def test_forecast_exact_resolves_small_bias():
    # MIGRAD with tol=1e-8 cannot resolve delta_r ~ 1e-9; the exact solver tracks the linear scaling
    ell, cl_tens, cl_lens = _spectra()
    base = 1e-6 * cl_lens
    d1 = sbm.forecast(_full(base))["delta_r"]
    d2 = sbm.forecast(_full(2.0 * base))["delta_r"]
    assert d1 > 0.0
    assert abs(d2 / d1 - 2.0) < 1e-4


def test_forecast_recovers_injected_tensor_bias():
    # a systematic equal to r_in times the tensor spectrum makes C_hat equal to the model at r_in,
    # so the likelihood maximum is exactly r_in, with or without noise
    ell, cl_tens, cl_lens = _spectra()
    noise = np.full_like(cl_lens, np.deg2rad(2.0 / 60.0) ** 2)  # 2 uK-arcmin white noise
    for r_in in (1e-2, 1e-5, 1e-9):
        assert abs(sbm.forecast(_full(r_in * cl_tens))["delta_r"] / r_in - 1.0) < 1e-9
        got = sbm.forecast(_full(r_in * cl_tens), n_el=_full(noise))["delta_r"]
        assert abs(got / r_in - 1.0) < 1e-9
