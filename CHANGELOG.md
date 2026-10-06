# Head
- add exception in the DB_ROOT_PATH setting in `scan_fields.py` to avoid to be down after import sbm.
- fix the map-based beam convolution: normalization of `elliptical_beam()`, the missing factor 1/2 of the polarized beam term in the spin-0 field of `SignalFields.elliptical_beam_field()`, and odd/negative spin `k` in `Convolver`. `elliptical_beam_field()` takes `spin_k` to include higher spin moments of the beam.

# Version 0.4.1

# Version 0.4.0
- Visualization function for map-making equation is implemented.
- `read_scanfield()` is added.
- add `convolver.py`, main beam convolution function, and its notebook [#4](https://github.com/yusuke-takase/SBM/pull/4).


# Version 0.3.0
- Sphinx theme is changed to `pydata-pshinx-theme`.
