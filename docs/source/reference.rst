API Reference
=============

Main functions
--------------

.. toctree::
.. autosummary::
   :toctree: generated/

   sbm.c2d
   sbm.d2c
   sbm.elliptical_beam
   sbm.forecast
   sbm.generate_cmb
   sbm.generate_maps
   sbm.get_cmap
   sbm.get_instrument_table
   sbm.load_fiducial_cl
   sbm.read_scanfiled
   sbm.sim_bandpass_mismatch
   sbm.sim_diff_gain_per_ch
   sbm.sim_diff_pointing_per_ch
   sbm.sim_noise_per_ch

Classes
-------

.. toctree::
   :maxdepth: 2

.. autosummary::
   :toctree: generated/

   sbm.Configlation
   sbm.Convolver
   sbm.Field
   sbm.ScanFields
   sbm.SignalFields
   sbm.Systematics

Methods
-------

.. toctree::
   :maxdepth: 2

.. autosummary::
   :toctree: generated/
   :recursive:

   sbm.Convolver.get_blk
   sbm.Convolver.get_clm
   sbm.Convolver.to_spin
   sbm.Field.conj
   sbm.ScanFields.create_covmat
   sbm.ScanFields.generate_noise
   sbm.ScanFields.generate_noise_pdf
   sbm.ScanFields.get_xlink
   sbm.ScanFields.initialize
   sbm.ScanFields.load_channel
   sbm.ScanFields.load_det
   sbm.ScanFields.load_full_FPU
   sbm.ScanFields.load_hdf5
   sbm.ScanFields.map_make
   sbm.ScanFields.t2b
   sbm.SignalFields.abs_pointing_field
   sbm.SignalFields.bandpass_mismatch_field
   sbm.SignalFields.build_linear_system
   sbm.SignalFields.circular_pointing_field
   sbm.SignalFields.diff_gain_field
   sbm.SignalFields.diff_pointing_field
   sbm.SignalFields.elliptical_beam_field
   sbm.SignalFields.extract_iqu
   sbm.SignalFields.get_coupled_field
   sbm.SignalFields.get_field
   sbm.SignalFields.hwp_ip_field
   sbm.Systematics.set_bandpass_mismatch

