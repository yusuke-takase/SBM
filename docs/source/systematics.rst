Systematics
===========

Every function of :mod:`sbm.systematics` takes the :class:`.Sky` (or maps)
and physical parameters and returns a :class:`.SignalModel`. The model of a
detector is :func:`~sbm.systematics.sky_signal` plus the perturbations of its
systematics; :func:`~sbm.systematics.beam` is the whole signal through a beam
and replaces :func:`~sbm.systematics.sky_signal`. The templates are the
couples to add to the basis (:meth:`.Basis.extend`) to absorb the systematic;
each function carries them as ``.templates`` (and the kind of data as
``.data``), so that ``Basis.iqu(hwp=True).extend(sy.abs_pointing.templates)``
works. :data:`sbm.systematics.ALL` lists the signal models of the package:
:func:`~sbm.systematics.sky_signal` and every systematic.

The modules follow the path of the signal through the instrument:
``sky`` (the ideal detector), ``pointing``, ``optics`` (the beam), ``hwp`` and
``detector``.

.. sbm-table:: systematics

The table is built from :data:`sbm.systematics.ALL`; the components of each
model are given on the page of its function. The second-order terms of the
pointing need :attr:`.Sky.second_order`, the covariant second derivatives
computed exactly with healpy's spin transforms of the harmonic coefficients of
the sky (:func:`sbm.sky.second_order_maps`; only its ``transport=False`` maps,
the former finite differences, use ducc0).

The h-maps must hold every couple between the basis and the components of the
model; :func:`.map_make` names the missing ones. For instance the second-order
diff pointing with the templates :math:`(1,0), (3,0)` needs :math:`n` up to 7.

Random pointing jitter
----------------------

A pointing offset drawn independently for every sample is not a signal model:
like white noise, it is a random contribution to the solution, whose
covariance :func:`.pointing_jitter_covariance` gives and that
:func:`.pointing_jitter_maps` draws. Both take the first-order model of a fixed
offset, as a function of ``rho_rad`` and ``chi_rad``, and the standard
deviation of each of the two components of the offset; a list of models gives
jitters that are independent of each other, such as those of the two detectors
of a pair. A co-pointed pair behind a HWP, whose two detectors share the
offset, has the data of a single detector that sees the polarization only:

.. literalinclude:: ../examples/pointing_jitter.py
   :language: python
   :start-after: # [sky]
   :end-before: # [end]

The variance of the jitter is quadratic in the data of the offset, so that the
h-maps must hold more couples than for a fixed offset: the sums of two couples
of the model combined with the differences of two couples of the basis
(:math:`n` up to 10 and :math:`|m|` up to 16 for this basis). The jitter is
first order in the offset; its mean effect, a smoothing of the sky at second
order, is not included.
