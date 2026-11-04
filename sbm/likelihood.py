"""Bias on the tensor-to-scalar ratio from a residual B-mode spectrum.

The likelihood is that of the LiteBIRD forecasts (PTEP 2023, Sect. 5.3.2): the
bias is the maximum in r of the likelihood of the residual (plus lensing and
noise) given the lensing, the noise and r times the tensor spectrum.
"""

import warnings

import numpy as np

from .spectra import load_fiducial_cl_lens, load_fiducial_cl_tens


def _get_likelihood(x, ell, cl_tens, cl_lens, cl_syst, n_el, fsky):  # x is r
    Cl_hat = (
        cl_syst[ell - 2] + cl_lens[ell - 2] + n_el[ell - 2]
    )  # there should be the noise cl_noise (noise and fg residuals), now assuming noiseless case
    Cl = (
        x * cl_tens[ell - 2] + cl_lens[ell - 2] + n_el[ell - 2]
    )  # there should be the noise cl_noise (noise and fg residuals), now assuming noiseless case
    return -np.sum(
        (-0.5)
        * fsky
        * (2.0 * ell + 1.0)
        * (
            (Cl_hat / Cl)
            + np.log(Cl)
            - ((2.0 * ell - 1.0) / (2.0 * ell + 1.0)) * np.log(Cl_hat)
        )
    )


def _get_likelihood_gradient(x, ell, cl_tens, cl_lens, cl_syst, n_el, fsky):  # x is r
    """Derivative in r of the negative log-likelihood returned by `_get_likelihood`."""
    Cl_hat = cl_syst[ell - 2] + cl_lens[ell - 2] + n_el[ell - 2]
    Cl = x * cl_tens[ell - 2] + cl_lens[ell - 2] + n_el[ell - 2]
    return np.sum(
        0.5 * fsky * (2.0 * ell + 1.0) * cl_tens[ell - 2] * (Cl - Cl_hat) / Cl**2
    )


def _maximize_likelihood(
    ell, cl_tens, cl_lens, cl_syst, n_el, fsky, rmax, rtol=1e-12, maxiter=400
):
    """Maximum of the likelihood over 0 <= r <= rmax, found as the root of its derivative.

    The derivative is negative below the maximum and positive above it, so the root is bracketed
    by bisection to a relative precision `rtol`. If the likelihood already decreases at r = 0
    (no positive bias, e.g. a residual at the level of rounding errors) the maximum is r = 0.
    """

    def grad(r):
        return _get_likelihood_gradient(r, ell, cl_tens, cl_lens, cl_syst, n_el, fsky)

    if grad(0.0) >= 0.0:
        return 0.0
    if grad(rmax) <= 0.0:
        warnings.warn(f"likelihood maximum is above rmax={rmax}; returning rmax")
        return float(rmax)
    lo, hi = 0.0, float(rmax)
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        if grad(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= rtol * hi:
            break
    return 0.5 * (lo + hi)


def forecast(
    cl_syst,
    n_el=None,
    fsky=1.0,
    lmax=191,
    rmin=1e-10,
    rmax=100,
    rresol=1000,
):
    """Bias on r from the residual B-mode spectrum ``cl_syst`` (from ell = 0).

    Args:
        cl_syst (`np.ndarray`, 1d array): residual B-modes power spectrum

        n_el (`np.ndarray`, 1d array): noise spectrum (from ell = 0), zero by default

        fsky (`float`): sky fraction

        lmax (`int`): maximum multipole considered in the maximization

        rmin (`float`): scale of the displayed grid when the bias is zero

        rmax (`float`): maximum value of r considered in the maximization

        rresol (`int`): resolution of the displayed grid in r

    The maximum of the likelihood in r is the root of its derivative, found by
    bisection over 0 <= r <= rmax to a relative precision of 1e-12; the bias is
    0 when the likelihood decreases from r = 0.

    Returns:
        ``{"delta_r": bias, "grid_r": r grid, "likelihood": likelihood on the grid}``
    """
    assert cl_syst[0] == cl_syst[1] == 0.0, (
        "First and second elements must be 0 because it is recognized as monopole and dipole"
    )

    # l range, from 2 to lmax
    ell = np.arange(2, lmax + 1)

    # the [2] selects the BB spectra and the [2:lmax+1] excludes multipoles 0 and 1,
    # that are null, and multipole above lmax
    cl_tens = load_fiducial_cl_tens(r=1.0, lmax=lmax)[2][2:]
    cl_lens = load_fiducial_cl_lens(lmax=lmax)[2][2:]
    cl_syst = cl_syst[2 : lmax + 1]
    if n_el is None:
        n_el = np.zeros_like(cl_lens)
    else:
        assert n_el[0] == n_el[1] == 0.0, (
            "First and second elements must be 0 because it is recognized as monopole and dipole"
        )
        n_el = n_el[2 : lmax + 1]

    def wrapped_likelihood(r):
        return _get_likelihood(r, ell, cl_tens, cl_lens, cl_syst, n_el, fsky)

    delta_r = _maximize_likelihood(ell, cl_tens, cl_lens, cl_syst, n_el, fsky, rmax)
    # Calculate likelihood function one last time in the range delta_r*1e-2 < r < delta_r*3
    # Note that delta_r has already been estimated, this likelihood is just used for display
    scale = delta_r if delta_r > 0 else rmin
    r_grid_display = np.linspace(scale * 1e-2, scale * 3.0, rresol)
    likelihood = np.zeros(rresol)
    for i, r in enumerate(r_grid_display):
        likelihood[i] = wrapped_likelihood(r)
    return {"delta_r": delta_r, "grid_r": r_grid_display, "likelihood": likelihood}
