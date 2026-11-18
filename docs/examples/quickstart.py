"""The code of the page "Quickstart", which includes it; tests/test_docs_examples.py runs it."""

# [h-maps]
import numpy as np

import sbm

nside = 8
rng = np.random.default_rng(0)
npix = 12 * nside**2
pixels = np.repeat(np.arange(npix), 50)
psi, phi = rng.uniform(-np.pi, np.pi, (2, pixels.size))
couples = [(n, m) for n in range(9) for m in range(-12, 13, 4)]
hmaps = sbm.HMaps.from_samples(nside, pixels, psi, phi, couples=couples)
# [model]
from sbm import systematics as sy

sky = sbm.Sky.from_maps(*sbm.generate_cmb(nside, cmb_seed=1))
model = sy.sky_signal(sky, hwp=True) + sy.hwp_ip(sky, epsilon=1e-4, phase_rad=0.3)
# [map-making]
iqu = sbm.map_make((hmaps, model), sbm.Basis.iqu(hwp=True)).iqu
cleaned = sbm.map_make(
    (hmaps, model), sbm.Basis.iqu(hwp=True).extend(sy.hwp_ip.templates)
).iqu
# [noise]
rng = np.random.default_rng(2)
noise = sbm.noise_maps(
    hmaps, sbm.Basis.iqu(hwp=True), net_ukrts=50.0, sampling_rate_hz=19.0, rng=rng
).iqu
# [end]

# the template of the HWP-IP absorbs it: the polarization is that of the sky
assert np.allclose(cleaned[1] + 1j * cleaned[2], sky.p, atol=1e-10)
