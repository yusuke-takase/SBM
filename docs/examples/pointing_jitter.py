"""Random pointing jitter of a co-pointed pair behind a HWP (docs/source/systematics.rst)."""

# [scan]
import functools

import healpy as hp
import numpy as np

import sbm
from sbm import systematics as sy
from sbm.testing import random_samples

nside = 8
pixels, psi, phi = random_samples(nside, 64, seed=1)
couples = [(n, m) for n in range(11) for m in range(-16, 17)]
hmaps = sbm.HMaps.from_samples(nside, pixels, psi, phi, couples=couples)
# [sky]
# The two detectors of the pair share the offset: their pair difference is the
# data of a single detector behind the HWP that sees the polarization only.
alm = hp.synalm(sbm.load_fiducial_cl_lens(), lmax=3 * nside - 1, new=True)
alm[0] = 0.0
sky = sbm.Sky.from_alm(alm, nside)
# [jitter]
basis = sbm.Basis.iqu(hwp=True)
model = functools.partial(sy.abs_pointing, sky)  # first order in the offset
sigma = np.radians(1.0 / 60.0)  # 1 arcmin per axis and per sample
ideal = sbm.map_make((hmaps, sy.sky_signal(sky, hwp=True)), basis)
jitter = sbm.pointing_jitter_maps(hmaps, basis, model, sigma, np.random.default_rng(0))
iqu = ideal.iqu + jitter.iqu
cov = sbm.pointing_jitter_covariance(hmaps, basis, model, sigma)
# [end]
assert np.all(np.isfinite(iqu)) and cov.shape == (hmaps.npix, 3, 3)
