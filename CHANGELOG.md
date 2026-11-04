# Head
- fix `SignalFields.diff_pointing_field` (pair difference without HWP): the polarization-gradient terms were twice too large. In `d = (d_T - d_B)/2` the polarization of each detector enters with weight 1/2, so the weight of the `o_zeta` terms of the spin-1 and spin-3 fields is `-1/8`, not `-1/4`; the temperature-gradient term `-1/4 zeta eth I` was already correct. Checked against a direct TOD simulation of the offset pointings (the residual of `mdim=4` agrees with slope 0.997, 0.498 before). The reference maps `tests/reference/diff_pointing_output_map_mdim_{2,4,6}.fits` are regenerated.
- add `ScanFields.load_lbs_h_maps(file_paths)`, which reads the h-maps written by litebird_sim (`make_h_maps` / `save_h_maps`) and by Falcons.jl (`save_h_maps`) in litebird_sim's layout `h_maps_det_<name>.h5`, one file or several files combined by hit-weighted averaging. litebird_sim defines `h_{n,m} = <exp(+i(n psi + m chi))>`; SBM's signal model equals litebird_sim's TOD model, rotating HWP and general Mueller matrix included, when SBM's crossing angle is `-psi`, so the reader stores `h~_{n,m} = h^lbs_{n,-m}`. The h-maps do not include the detector polarization angle, so combine only detectors with the same polarization angle. String attributes are decoded whether they are variable-length (litebird_sim) or fixed-length (Falcons.jl, read by h5py as `bytes`).
- expose general asymmetric beam convolution as `SignalFields.beam_field()`; retain `elliptical_beam_field()` as a compatibility alias.
- add exception in the DB_ROOT_PATH setting in `scan_fields.py` to avoid to be down after import sbm.
- fix the map-based beam convolution: normalization of `elliptical_beam()`, the missing factor 1/2 of the polarized beam term in the spin-0 field of `SignalFields.elliptical_beam_field()`, and odd/negative spin `k` in `Convolver`. `elliptical_beam_field()` takes `spin_k` to include higher spin moments of the beam.

# Version 0.4.1

# Version 0.4.0
- Visualization function for map-making equation is implemented.
- `read_scanfield()` is added.
- add `convolver.py`, main beam convolution function, and its notebook [#4](https://github.com/yusuke-takase/SBM/pull/4).


# Version 0.3.0
- Sphinx theme is changed to `pydata-pshinx-theme`.
