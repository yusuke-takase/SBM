API reference
=============

Scan, signal and map-making
---------------------------

.. autosummary::
   :toctree: generated/

   sbm.HMaps
   sbm.SignalModel
   sbm.SpinComponent
   sbm.Basis
   sbm.Sky
   sbm.SphericalHarmonics
   sbm.MapMakingResult
   sbm.map_make
   sbm.MapMaker
   sbm.orientation_matrix
   sbm.noise_covariance
   sbm.noise_maps

Systematics
-----------

.. autosummary::
   :toctree: generated/

   sbm.systematics.sky_signal
   sbm.systematics.diff_gain
   sbm.systematics.diff_pointing
   sbm.systematics.bandpass_mismatch
   sbm.systematics.abs_pointing
   sbm.systematics.hwp_wedge
   sbm.systematics.hwp_ip
   sbm.systematics.hwp_mueller
   sbm.systematics.polarization_angle
   sbm.systematics.beam
   sbm.systematics.systematic

Beam convolution
----------------

.. autosummary::
   :toctree: generated/

   sbm.beam.convolve
   sbm.beam.beam_transfer
   sbm.beam.pencil
   sbm.beam.gaussian
   sbm.beam.elliptical
   sbm.sky.second_order_maps
   sbm.sky.spin_derivatives

Spectra and forecasts
---------------------

.. autosummary::
   :toctree: generated/

   sbm.generate_cmb
   sbm.load_fiducial_cl
   sbm.load_fiducial_cl_lens
   sbm.load_fiducial_cl_tens
   sbm.c2d
   sbm.d2c
   sbm.forecast
   sbm.get_cmap

Testing a model
---------------

.. autosummary::
   :toctree: generated/

   sbm.testing.random_samples
   sbm.testing.tod_from_components
   sbm.testing.least_squares
   sbm.testing.assert_matches_tod
   sbm.testing.RealSky
