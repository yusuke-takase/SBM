Conventions
===========

The conventions are written once, in :mod:`sbm.conventions`: the package, the
tables of this documentation and the tests take them from there.

.. automodule:: sbm.conventions
   :members:
   :member-order: bysource

Components of the ideal detectors
---------------------------------

.. sbm-table:: components

A pair difference :math:`(d_T - d_B)/2` is taken between two detectors in the
same position with orthogonal polarizers; it has no intensity. Every
:class:`.SignalModel` and :class:`.HMaps` stores one couple per conjugate pair
(:func:`~sbm.conventions.canonical`); ``h(n, m)`` and ``component(n, m)``
return the conjugate for the others.

Couples in other conventions
----------------------------

.. sbm-table:: relations

:meth:`.HMaps.load` and :meth:`.HMaps.save` convert the couples of the h-map
files of litebird_sim and Falcons.jl, and :meth:`.HMaps.load_legacy` those of
the cross-link files of SBM 0.4. The h-maps of litebird_sim do not include the
polarization angle of the detector; SBM puts it in the signal
(``pol_angle_rad`` of :func:`~sbm.systematics.sky_signal` and
:func:`~sbm.systematics.hwp_mueller`).

Spin weights
------------

The component :math:`{}_{n,m}\tilde S` has spin weight :math:`n`: the spin
weights of the factors on its right-hand side add up to :math:`n`.

.. list-table::
   :header-rows: 1
   :widths: 40 20

   * - map of :class:`.Sky`
     - spin weight
   * - :math:`I` (``i``), :math:`\nabla^2 I` (``lap_i``), :math:`\bar\eth^2 P` (``eth_bar2_p``)
     - 0
   * - :math:`P = Q + iU` (``p``), :math:`\eth^2 I` (``eth2_i``), :math:`\nabla^2 P` (``lap_p``)
     - 2
   * - :math:`\eth I` (``eth_i``), :math:`\bar\eth P` (``eth_bar_p``)
     - 1
   * - :math:`\bar\eth I` (``eth_bar_i``)
     - -1
   * - :math:`\eth P` (``eth_p``)
     - 3
   * - :math:`\eth^2 P` (``eth2_p``)
     - 4

Numbers (:math:`\rho`, the phase factors, gains) have spin weight 0, the
complex conjugate negates the spin weight, and the HWP angle enters only
:math:`m`. For example the first-order pointing offset behind a HWP gives
:math:`\eth I` at :math:`(1,0)` and :math:`\eth P`, :math:`\bar\eth P` at the
couple of the polarization plus and minus :math:`(1,0)`. The maps of a
:class:`.Sky` carry their spin weight (:class:`~sbm.spin.SpinMap`), and
:class:`.SignalModel` raises :class:`~sbm.spin.SpinWeightError` for a component
whose spin weight is not its :math:`n`, or for a sum of terms of different spin
weights (:mod:`sbm.spin`). Arrays without spin weight, such as ``sky.q`` and
``sky.u`` taken separately, are not checked.

Covariant derivatives
---------------------

On the polarization :math:`P = Q + iU` the derivatives are the covariant
(spin-weighted) ones: the beam is displaced rigidly, so the polarization
direction of the detector is carried along the displacement, and
:math:`\eth P` is :math:`\eth Q + i\,\eth U` in a frame transported from the
point. :class:`.Sky` computes them with spin transforms of the T, E and B
coefficients (:func:`sbm.sky.spin_derivatives`), and the covariant
second-order maps exactly in the same way (:func:`sbm.sky.second_order_maps`).
The derivatives of :math:`Q` and :math:`U` taken as scalar maps (healpy's
``alm2map_der1`` of each map, as in SBM 0.4) omit the rotation of the local
basis: they differ from the covariant ones by a term of order
:math:`\cot\theta\, P`, small on scales much smaller than the distance to the
poles but not near them, and :math:`Q`, :math:`U` are not band-limited as
scalar maps. Known derivatives can be passed to :class:`.Sky` directly, in the
convention of :data:`~sbm.conventions.ETH_PHASE`.

Units
-----

Angles are in radians (arguments ending in ``_rad``), the NET in
:math:`\mu K\sqrt{s}` and the sampling rate in Hz; the noise maps have the
units of the NET times :math:`\sqrt{\rm Hz}`, i.e. :math:`\mu K`.
