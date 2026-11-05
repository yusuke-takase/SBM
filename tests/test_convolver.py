"""Convolver with spin moments above lmax.

A spin-k map has no multipole below k, so the moments k > lmax are zero maps. The
path with HWP already returned them; without HWP, Convolver called healpy's
alm2map_spin with a spin above lmax and libsharp aborted the process ("bad spin"),
so the convolution runs in a subprocess.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CODE = """
import healpy as hp
import numpy as np
from sbm import Convolver

lmax, nside = 3, 4
rng = np.random.default_rng(0)
alm, blm = rng.standard_normal((2, 3, hp.Alm.getsize(lmax))) + 0j
alm[1:, :2] = 0
blm[1:, :2] = 0


def convolve(spins):
    return Convolver(alm, nside, spins, use_hwp={hwp}) * Convolver(blm, nside, spins)


full = convolve([0, 2, 4])
low = convolve([0, 2])
if {hwp}:
    assert {{k for k in full if abs(k[0]) <= lmax}} == set(low)
    for key, value in full.items():
        if abs(key[0]) > lmax:
            assert not np.any(value), key
        else:
            np.testing.assert_array_equal(value, low[key])
else:
    assert full.shape == (3, 2, hp.nside2npix(nside))
    np.testing.assert_array_equal(full[:2], low)
    assert not np.any(full[2])
    assert np.any(full[1])
print("ok")
"""


@pytest.mark.parametrize("hwp", [False, True])
def test_spins_above_lmax_are_zero_maps(hwp):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", CODE.format(hwp=hwp)],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"
