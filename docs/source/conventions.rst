Conventions
===========

Angles and spins
----------------

The data of a detector are written as

.. math::

   d = \sum_{n,m} {}_{n,m}\tilde S \, e^{i(n\psi + m\phi)},
   \qquad {}_{-n,-m}\tilde S = {}_{n,m}\tilde S^*,

with the crossing angle :math:`\psi` and the HWP angle :math:`\phi`, and the
orientation functions are
:math:`{}_{n,m}\tilde h = \langle e^{-i(n\psi + m\phi)} \rangle`.

* An ideal detector without HWP sees
  :math:`d = I + \mathrm{Re}[P e^{2i\psi}] = I + Q\cos 2\psi - U\sin 2\psi`, with
  :math:`P = Q + iU`: :math:`I` sits at :math:`(0,0)` and :math:`P/2` at
  :math:`(2,0)`.
* The HWP angle :math:`\phi` is the angle of the HWP with respect to the
  telescope, on which it is mounted. It is counted opposite to :math:`\psi`, so
  that an ideal HWP modulates the polarization as :math:`\cos(4\phi - 2\psi)`:
  :math:`P/2` sits at :math:`(2,-4)`.
* A pair difference :math:`(d_T - d_B)/2` is taken between two detectors in
  the same position with orthogonal polarizers,
  :math:`\psi_B = \psi_T + \pi/2`; it has no intensity.

Every :class:`.SignalModel` and :class:`.HMaps` stores one couple per conjugate
pair (:math:`n > 0`, or :math:`n = 0` and :math:`m > 0`); ``h(n, m)`` and
``component(n, m)`` return the conjugate for the others.

Spin weights
------------

When the local basis turns by an angle :math:`\alpha`, in the sense in which the
polarization angle is counted, a field of spin weight :math:`s` is multiplied by
:math:`e^{-is\alpha}` and :math:`\psi` becomes :math:`\psi + \alpha`; the
HWP angle does not change. The data are the same in every basis, so
**the component** :math:`{}_{n,m}\tilde S` **has spin weight** :math:`n`: the
spin weights of the factors on its right-hand side add up to :math:`n`.

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

Numbers (:math:`\rho`, :math:`e^{i\chi}`, gains) have spin weight 0, the
complex conjugate negates the spin weight, and the HWP angle enters only
:math:`m`. For example the first-order pointing offset behind a HWP gives
:math:`\eth I` at :math:`(1,0)`, :math:`\bar\eth P` at :math:`(1,-4)` and
:math:`\eth P` at :math:`(3,-4)`. The maps of a :class:`.Sky` carry their spin
weight (:class:`~sbm.spin.SpinMap`), and :class:`.SignalModel` raises
:class:`~sbm.spin.SpinWeightError` for a component whose spin weight is not its
:math:`n`, or for a sum of terms of different spin weights
(:mod:`sbm.spin`). Arrays without spin weight, such as ``sky.q`` and
``sky.u`` taken separately, are not checked.

Derivatives and pointing offsets
--------------------------------

On the local tangent plane, with :math:`x` along :math:`\theta` and :math:`y`
along :math:`\phi`, :math:`\eth = \partial_y - i\partial_x` and
:math:`\bar\eth = \partial_y + i\partial_x` (:class:`.Sky`). On the
polarization :math:`P = Q + iU` they are the covariant (spin-weighted)
derivatives: the beam is displaced rigidly, so the polarization direction of
the detector is carried along the displacement, and :math:`\eth P` is
:math:`\eth Q + i\,\eth U` in a frame transported from the point. A pointing offset of size
:math:`\rho` in direction :math:`\chi`, measured in the frame of the detector,
samples the sky at the point displaced by :math:`\rho` along
:math:`-(\sin\alpha\, e_\theta + \cos\alpha\, e_\phi)` with
:math:`\alpha = \psi + \chi`, so that

.. math::

   S(p') = S - \frac{\rho}{2}\left(e^{i\alpha}\eth + e^{-i\alpha}\bar\eth\right) S
   + \frac{\rho^2}{8}\left(e^{2i\alpha}\eth^2 + e^{-2i\alpha}\bar\eth^2\right) S
   + \frac{\rho^2}{4}\nabla^2 S + O(\rho^3).

The deflection of a wedged HWP rotates with the HWP:
:math:`\alpha = \psi - \phi + \chi`.

:class:`.Sky` computes the derivatives with spin transforms of the T, E and B
coefficients (:func:`sbm.sky.spin_derivatives`), and the covariant second-order
maps exactly in the same way (:func:`sbm.sky.second_order_maps`). The
derivatives of :math:`Q` and :math:`U` taken as scalar maps (healpy's
``alm2map_der1`` of each map, as in SBM 0.4) omit the rotation of the local
basis: they differ from the covariant ones by a term of order
:math:`\cot\theta\, P`, small on scales much smaller than the distance to the
poles but not near them, and :math:`Q`, :math:`U` are not band-limited as
scalar maps. Known derivatives can be passed to :class:`.Sky` directly.

litebird_sim and Falcons.jl
---------------------------

litebird_sim and Falcons.jl compute
:math:`h_{n,m} = \langle e^{+i(n\psi' + m\chi)} \rangle` with the orientation
angle :math:`\psi'` measured from East towards North and the HWP angle
:math:`\chi`; their TOD model for a detector behind a HWP of Mueller matrix
:math:`M` is
:math:`d = \mathrm{det}\cdot R(\chi)^T M R(\chi) R(\psi') S`, with
:math:`R(x)` the rotation of the Stokes parameters by :math:`2x`. It is SBM's
model with :math:`\psi = -\psi'` and :math:`\phi = \chi`, so

.. math::

   {}_{n,m}\tilde h = h_{n,-m}.

:meth:`.HMaps.load` and :meth:`.HMaps.save` apply this relation. Using their
angles unchanged would describe the mirrored polarization (:math:`U \to -U`).
The h-maps of litebird_sim do not include the polarization angle of the
detector; SBM models detectors with a zero polarization angle.

Units
-----

Angles are in radians (arguments ending in ``_rad``), the NET in
:math:`\mu K\sqrt{s}` and the sampling rate in Hz; the noise maps have the
units of the NET times :math:`\sqrt{\rm Hz}`, i.e. :math:`\mu K`.
