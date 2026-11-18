Quickstart
==========

The four objects of SBM are the orientation functions of the scan
(:class:`sbm.HMaps`), the sky (:class:`sbm.Sky`), the signal model of a
detector (:class:`sbm.SignalModel`, from :mod:`sbm.systematics`) and the
components that the map-maker solves for (:class:`sbm.Basis`).

Orientation functions
---------------------

:meth:`sbm.HMaps.load` reads the h-map files that litebird_sim
(``make_h_maps``) and Falcons.jl (``save_h_maps``) write, one file per
detector. :meth:`sbm.HMaps.load_legacy` reads the cross-link files of SBM 0.4.
For tests and toy scans, :meth:`sbm.HMaps.from_samples` computes them from
the pixel, crossing angle and HWP angle of every sample:

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # [h-maps]
   :end-before: # [model]

``hmaps.h(n, m)`` returns :math:`{}_{n,m}\tilde h` (the conjugate couple is
computed), ``hmaps.hits`` the hit counts.

Signal models
-------------

A model is the sum of the sky signal and of the perturbations of the
systematics:

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # [model]
   :end-before: # [map-making]

Map-making
----------

The basis fixes the components solved in every pixel; templates of
systematics are added with :meth:`sbm.Basis.extend`:

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # [map-making]
   :end-before: # [noise]

The pixels that cannot be solved are ``NaN``. With several detectors, pass a
list of ``(HMaps, SignalModel)``: their systems, weighted by the hit counts,
are summed before solving.

Noise and forecasts
-------------------

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-after: # [noise]
   :end-before: # [end]

:func:`sbm.forecast` turns a residual B-mode spectrum into a bias on the
tensor-to-scalar ratio with the likelihood of the LiteBIRD forecasts.
