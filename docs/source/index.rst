SBM User's Manual
=================

SBM (spin-based map-making) forecasts the systematic effects of a CMB
polarization experiment in spin space. The scan of a detector is reduced to
its orientation functions :math:`{}_{n,m}\tilde h`, the data to the spin
components :math:`{}_{n,m}\tilde S` of a signal model, and the binned
map-maker becomes a small linear system in every pixel, so that maps with
systematics are computed without time-ordered data.

.. code-block:: python

   import sbm
   from sbm import systematics as sy

   hmaps = sbm.HMaps.load("h_maps_det_A.h5")  # litebird_sim / Falcons.jl
   sky = sbm.Sky.from_maps(*sbm.generate_cmb(nside=hmaps.nside, cmb_seed=1))
   model = sy.sky_signal(sky, hwp=True) + sy.abs_pointing(sky, rho_rad=1e-4, chi_rad=0.3)
   iqu = sbm.map_make((hmaps, model), sbm.Basis.iqu(hwp=True)).iqu

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation.rst
   quickstart.rst
   formalism.rst
   conventions.rst
   systematics.rst
   adding_systematics.rst
   testing.rst
   hwp_beam_convolution.rst
   migration.rst
   reference.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
