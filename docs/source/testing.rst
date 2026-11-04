Testing
=======

Run the tests with ``pytest`` (from any directory). They use no reference
file but ``tests/nside_32_boresight_hwp.h5``, a scan in the cross-link format
of SBM 0.4, and take about ten seconds.

Every test compares SBM with an independent computation:

* ``tests/tod.py`` draws random crossing and HWP angles in every pixel of a
  small map, writes the TOD of spin components and fits them per pixel by
  least squares. ``test_core.py`` checks the map-maker, several detectors and
  the white-noise covariance against it.
* ``test_systematics.py`` writes the TOD of every systematic in real space: a
  Taylor expansion of the sky at the offset pointing, the gains, the HWP
  leakage, the response of the detector, with no spin decomposition.
* ``test_pointing_geometry.py`` samples a band-limited sky with ducc0 at the
  offset pointing and checks that the residual of the pointing models falls as
  :math:`\rho`, :math:`\rho^2` and :math:`\rho^3` at zeroth, first and second
  order.
* ``test_hwp_convolution.py`` and ``test_beam_low_ell.py`` compare the beam
  convolution with Stokes measurements, direct Wigner sums and real-space beam
  integrals, and the map-maker with sampled TOD.
* ``test_lbs_h_maps.py`` writes h-map files in the layout of litebird_sim and
  checks the binning of litebird_sim's TOD model.
* ``test_forecast.py`` checks the likelihood maximum (zero, linear and
  injected biases).

The tests that need the optional sympy and iminuit are skipped without them.
A new systematic comes with a real-space TOD test (:doc:`adding_systematics`);
changing a weight of a model makes these tests fail.
