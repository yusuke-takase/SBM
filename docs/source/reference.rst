API Reference
=============

Main functions
--------------

.. toctree::
.. autosummary::
   :toctree: generated/

   sbm.c2d
   sbm.d2c
   sbm.forecast
   sbm.generate_cmb
   sbm.generate_maps
   sbm.get_cmap
   sbm.get_instrument_table
   sbm.load_fiducial_cl
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

   sbm.Field.conj
   sbm.ScanFields.create_covmat
   sbm.ScanFields.generate_noise
   sbm.ScanFields.generate_noise_pdf
   sbm.ScanFields.get_xlink
   sbm.ScanFields.initialize
   sbm.ScanFields.map_make
   sbm.ScanFields.t2b
   sbm.SignalFields.abs_pointing_field
   sbm.SignalFields.build_linear_system
   sbm.SignalFields.diff_gain_field
   sbm.SignalFields.diff_pointing_field
   sbm.SignalFields.get_coupled_field
   sbm.SignalFields.get_field
   sbm.SignalFields.hwp_ip_field

