# Random pointing references

`random_pointing.npz` is a small fixed regression fixture evaluated with the
original `src/map_based_time_dependent_systematics/random_pointing.py` in the
map_based_time_dependent_systematics study, without calling the new SBM method.
It uses explicit scan angles (NumPy RNG seed 42, 12 pixels, 512 hits/pixel),
Cartesian width 1 arcmin and realization seed 23000001.

`random_pointing_saved_bb.npz` contains the **previously computed** first
`explicit_xy_bb` spectrum and seed from:

```
results/cache/litebird_reformation_2026/hwp_copointed_pair_explicit_xy_nside256/sbm_comparison_smoke_n10.npz
```

Source SHA256: `3cffe8d85a6fbc9d2c22f7a72586a4e5d393c63f709f1adf3cbd5868e063e980`.
This expected spectrum was copied directly, without recalculation. Its metadata
is retained in the fixture (nside=256, lmax=767, HWP=42 rpm, per-axis sigma=1 arcmin).
The replay test reads the large input sky/scan from the original study, selected
with `SBM_RANDOM_POINTING_REFERENCE_ROOT`, and runs the existing SBM map maker.

```bash
SBM_RANDOM_POINTING_REFERENCE_ROOT=/path/to/map_based_time_dependent_systematics \
OMP_NUM_THREADS=1 python -m pytest tests -q
```
