From the thesis to the code
===========================

The five systematics modelled in chapter 6 of the thesis of Y. Takase,
*Design of the full-sky scanning strategy and systematic effect control in a
cosmic microwave background probe* (Okayama University, 2025,
`arXiv:2503.03176 <https://arxiv.org/abs/2503.03176>`_), are functions of
:mod:`sbm.systematics`. This page takes each of them from its equations in the
thesis, numbered as in the arXiv version, to the code: what the parameters and
the spins are, where the thesis has to be corrected, and how the components
are written with the maps of a :class:`.Sky`. It is a worked example of
:doc:`adding_systematics`.

Reading the thesis
------------------

**Convention.** The thesis writes the binned signal as
:math:`S^d = \sum_{n,m} {}_{n,m}\tilde S^d e^{i(n\psi + m\phi)}` with
:math:`{}_{n,m}\tilde S^d = \sum_{n',m'} {}_{n-n',m-m'}\tilde h \;
{}_{n',m'}\tilde S` and
:math:`{}_{n,m}\tilde h = \langle e^{-i(n\psi + m\phi)} \rangle`,
Eqs. (4.18)–(4.20), with the crossing angle :math:`\psi` counted opposite to the
polarization angle and the HWP angle :math:`\phi` with respect to the telescope:
the convention of SBM 0.4. The code counts :math:`\psi` in the sense of the
polarization angle, that of litebird_sim (:doc:`conventions`); the HWP angle is
the same. A component or an orientation function at :math:`(n, m)` in the thesis
is at :func:`sbm.conventions.from_0_4` :math:`(n, m) = (n, -m)` in the code.
This page quotes the equations of the thesis in its convention and gives the
couples of the code. The coupling equations of each section (e.g.
Eqs. (6.34)–(6.35)) are computed by the map-maker, so a systematic gives only
:math:`{}_{n,m}\tilde S`. The thesis lists both members of each conjugate pair,
:math:`{}_{-n,-m}\tilde S = {}_{n,m}\tilde S^*`; a :class:`.SignalModel` takes
one of them.

**Angles.** In the thesis the ideal detector without HWP is
:math:`I + \frac12 P e^{2i\psi} + \frac12 P^* e^{-2i\psi}
= I + Q\cos 2\psi - U\sin 2\psi`, Eq. (4.9). The real form printed in Eq. (4.9),
with :math:`+U\sin 2\psi`, has the wrong sign of :math:`U` in its convention (it
is the real form in the convention of the code); the complex form is the one
used throughout. Behind a HWP the thesis puts :math:`P/2` at :math:`(2,-4)`,
Eq. (4.21), the couple |pol_hwp| of the code.

**Derivatives.** The thesis uses :math:`\eth = \partial_y - i\partial_x` and
:math:`\bar\eth = \partial_y + i\partial_x` (below Eq. (6.17)), with :math:`x`
along :math:`\theta` and :math:`y` along :math:`\phi`: :math:`i` times the
standard operators of the code. (The thesis says that :math:`x` points east and
:math:`y` north; with these axes :math:`\eth` would be multiplied by :math:`-i`,
i.e. :math:`\chi` shifted by 90 degrees. Read with :math:`x` along
:math:`\theta`, the formulas of the thesis are those of SBM 0.4.) The thesis
differentiates :math:`Q` and :math:`U` as scalar maps, whereas :class:`.Sky`
computes the covariant derivatives of :math:`P` (:doc:`conventions`);
derivatives given to :class:`.Sky` are used as they are, in the convention of
the code.

**Pointing offsets.** An offset of size :math:`\rho` in direction
:math:`\psi + \chi`, fixed in the frame of the detector, perturbs the sampled
field as :math:`[1 - (\partial_x\Delta x + \partial_y\Delta y)] S`,
Eqs. (6.16) and (6.27), with

.. math::

   \partial_x\Delta x + \partial_y\Delta y
   = \frac{\rho}{2}\left[e^{i(\psi+\chi)}\eth + e^{-i(\psi+\chi)}\bar\eth\right]
   \qquad \text{Eq. (6.17)},

at first order. The code writes the same offset with the expansion of
:mod:`sbm.conventions`, in which :math:`\chi` is the direction of the offset
itself, measured from the reference axis of the focal plane in the sense of
:math:`\psi`: the direction :math:`\chi` of the thesis is
:math:`-\chi - \pi/2` in the code. The products of a phase and a derivative,
and hence the values of the components, are the same.

**From a formula to a function.**

#. Write the perturbed signal and collect the coefficient of every
   :func:`~sbm.conventions.synthesis_phase` :math:`(n, m)`.
#. Keep the perturbation (the ideal signal is
   :func:`~sbm.systematics.sky_signal`) and one couple of each conjugate pair.
#. Write each coefficient with ``sky.i``, ``sky.p``, ``sky.eth_i``,
   ``sky.eth_bar_i``, ``sky.eth_p``, ``sky.eth_bar_p`` and numbers
   (:math:`\rho`, the phase :func:`~sbm.conventions.phase_factor` of an angle,
   gains).
#. Check that the spin weights of the factors add up to :math:`n`
   (:math:`I` 0, :math:`P` 2, :math:`\eth I` 1, :math:`\bar\eth P` 1,
   :math:`\eth P` 3, conjugation negates, see :doc:`conventions`).
   :class:`.SignalModel` checks it when the model is built; it catches the
   misplaced :math:`\eth P` of the wedge, Eq. (6.41), and the HWP-IP at
   :math:`n = 4`, Eq. (6.47), below.
#. Declare the function with ``@systematic(data=..., templates=...)``.
   ``data`` is the data the model describes: ``"pair"`` for the pair difference
   :math:`(d_T - d_B)/2` of two orthogonal detectors without HWP (Sect. 6.1 of the
   thesis), ``"single"`` for the data of one detector, not differenced, without
   HWP, ``"single_hwp"`` for one detector behind a HWP (Sect. 6.2), and
   ``"any"`` when the arguments choose (as in
   :func:`~sbm.systematics.sky_signal`).
#. The templates are the couples of the first-order perturbation that the basis
   of the ideal detector lacks: they are what the extended map-makers of the
   thesis add, Eqs. (6.58), (6.65), (6.71), (6.75) and (6.77).

Differential gain
-----------------

*Thesis, Sect. 6.1.1.* A pair of orthogonal detectors T and B without HWP
(:math:`\psi_B = \psi_T + \pi/2`), with gain errors :math:`g_T` and :math:`g_B`:
:math:`S_{T,g} = (1 + g_T) S_T`, :math:`S_{B,g} = (1 + g_B) S_B`, and the pair
difference :math:`D_g = \frac12 (S_{T,g} - S_{B,g})`, Eqs. (6.3)–(6.6). Its spin
components are, Eqs. (6.7)–(6.9),

.. math::

   {}_{0}\tilde D_g = \frac12 \Delta g\, I, \qquad
   {}_{2}\tilde D_g = {}_{-2}\tilde D_g^* = \frac14 (2 + g_T + g_B) P,
   \qquad \Delta g = g_T - g_B .

*Reading.* No HWP, so :math:`m = 0`. :math:`I` (spin weight 0) at :math:`n = 0`
and :math:`P` (2) at :math:`n = 2`. The ideal pair difference is
:math:`\frac12 P` at :math:`(2, 0)`
(``sky_signal(sky, pair_difference=True)``), so the perturbation is
:math:`\frac12 \Delta g\, I` at :math:`(0, 0)` and
:math:`\frac14 (g_T + g_B) P` at :math:`(2, 0)`.

*Code.* ``g_t`` and ``g_b`` are :math:`g_T` and :math:`g_B`:

.. literalinclude:: ../../sbm/systematics/detector.py
   :pyobject: diff_gain

*Template.* :math:`(0, 0)`, the spin-0 template of the 3×3 map-maker,
Eqs. (6.58)–(6.59). It absorbs the leakage of :math:`I`; the mean gain
:math:`(g_T + g_B)/2` rescales :math:`P` and is not separated from it,
Eq. (6.60).

.. code-block:: python

   model = sy.sky_signal(sky, pair_difference=True) + sy.diff_gain(sky, g_t, g_b)
   basis = sbm.Basis.qu(hwp=False).extend(sy.diff_gain.templates)

Differential pointing
---------------------

*Thesis, Sect. 6.1.2.* The same pair, each detector with a pointing offset
:math:`(\rho_T, \chi_T)` and :math:`(\rho_B, \chi_B)`, Eqs. (6.16)–(6.19). Because
:math:`\psi_B = \psi_T + \pi/2`, the offset of B enters with a factor
:math:`i`, and the pair difference depends on, Eqs. (6.21)–(6.22),

.. math::

   \zeta = \rho_T e^{i\chi_T} - i\rho_B e^{i\chi_B}, \qquad
   \bar\zeta = \rho_T e^{i\chi_T} + i\rho_B e^{i\chi_B} .

For an offset common to both detectors :math:`\zeta = 0` and
:math:`\bar\zeta = 2\rho e^{i\chi}`: :math:`\zeta` is the differential offset,
which leaks the temperature, and :math:`\bar\zeta` twice the common one.

*Correction.* Eqs. (6.20), (6.23) and (6.25) give the polarization gradients
the coefficient :math:`-\frac14 \bar\zeta`. From the detectors, Eqs. (6.18)–(6.19),
it is :math:`-\frac18 \bar\zeta`: :math:`\frac12` of the pair difference,
:math:`\frac12` of :math:`\frac12 P` and :math:`\frac{\rho}{2}` of the offset.
For a common offset the pair difference must be the response of one detector,
:math:`-\frac{\rho}{4} e^{i\chi}\eth P`, which :math:`-\frac18 \bar\zeta` gives
and :math:`-\frac14 \bar\zeta` doubles. The temperature coefficient
:math:`-\frac14 \zeta` is right. The spin components are

.. math::

   {}_{1}\tilde D_p = -\frac14 \zeta\, \eth I - \frac18 \bar\zeta^*\, \bar\eth P,
   \qquad
   {}_{3}\tilde D_p = -\frac18 \bar\zeta\, \eth P .

*Reading.* :math:`\eth I` and :math:`\bar\eth P` (spin weight 1) at
:math:`n = 1`, :math:`\eth P` (3) at :math:`n = 3`, and :math:`m = 0`.

*Code.* The code measures both directions from the reference axis of the focal
plane, that of T, as offsets themselves: :math:`\chi_T` and :math:`\chi_B` of the
thesis are :math:`-\chi_T - \pi/2` and :math:`-\chi_B - \pi` in the code (the
latter in the frame of B, rotated by 90 degrees). ``u_t`` and ``u_b`` are
:math:`\rho_T` and :math:`\rho_B` times :func:`~sbm.conventions.phase_factor` of
these directions; ``zeta`` and ``o_zeta`` are :math:`i\zeta` and
:math:`i\bar\zeta`, and the standard derivatives :math:`-i` times those of the
thesis, so that the components are the same (the second-order terms follow in
the source):

.. literalinclude:: ../../sbm/systematics/pointing.py
   :pyobject: diff_pointing
   :end-before: if second_order:

*Templates.* :math:`(1, 0)` and :math:`(3, 0)`, the 6×6 map-maker,
Eqs. (6.65)–(6.66); :math:`(1, 0)` alone is the 4×4 one, Eqs. (6.62)–(6.63).
Without HWP the couples of the thesis and of the code are the same.

Absolute pointing offset
------------------------

*Thesis, Sect. 6.2.1.* A single detector behind a HWP, with an offset
:math:`(\rho, \chi)` fixed in its frame, Eqs. (6.27)–(6.28). The spin components,
Eqs. (6.29)–(6.33), are right in the convention of the thesis:

.. math::

   {}_{1,0}\tilde S = -\frac{\rho}{2} e^{i\chi}\eth I, \qquad
   {}_{1,-4}\tilde S = -\frac{\rho}{4} e^{-i\chi}\bar\eth P, \qquad
   {}_{3,-4}\tilde S = -\frac{\rho}{4} e^{i\chi}\eth P .

In the code they are at :math:`(1, 0)`, :math:`(1, 4)` and :math:`(3, 4)`, with
the phase and the derivatives of the code.

*Reading,* in the convention of the code. Write the direction of the offset as
:math:`\alpha = \psi + \chi`. By :mod:`sbm.conventions` the offset applies
:math:`-\frac{\rho}{2}\left(e^{-i\alpha}\eth + e^{i\alpha}\bar\eth\right)` to
each term of the ideal signal :math:`I + \frac12 P e^{-i(2\psi + 4\phi)} + {\rm
c.c.}`. Each product of exponentials is one couple: the coefficients of
:math:`-\psi` and :math:`-\phi` in its exponent are :math:`n` and :math:`m`, and
the constant :math:`e^{\mp i\chi}` stays in the coefficient.

.. list-table::
   :header-rows: 1
   :widths: 40 22 16 22

   * - term of the perturbation
     - :math:`\psi`, :math:`\phi` in the exponent
     - :math:`(n, m)`
     - spin weight of the field
   * - :math:`-\frac{\rho}{2} e^{-i\alpha}\, \eth I`
     - :math:`-\psi`
     - :math:`(1, 0)`
     - 1
   * - :math:`-\frac{\rho}{2} e^{i\alpha}\, \bar\eth I`
     - :math:`\psi`
     - :math:`(-1, 0)`, the conjugate of :math:`(1, 0)`
     - -1
   * - :math:`-\frac{\rho}{4} e^{i\alpha}\, \bar\eth P\, e^{-i(2\psi + 4\phi)}`
     - :math:`-(2\psi + 4\phi) + \psi`
     - :math:`(1, 4)`
     - 1
   * - :math:`-\frac{\rho}{4} e^{-i\alpha}\, \eth P\, e^{-i(2\psi + 4\phi)}`
     - :math:`-(2\psi + 4\phi) - \psi`
     - :math:`(3, 4)`
     - 3
   * - the same with :math:`P^*`
     -
     - :math:`(-1, -4)`, :math:`(-3, -4)`: the conjugates
     -

In every row :math:`n` equals the spin weight of the field.

*Code.* The table holds for any direction
:math:`\alpha = dn\,\psi + dm\,\phi + \chi`: :math:`(dn, dm)` are the
coefficients of :math:`\psi` and :math:`\phi` in the direction of the offset,
:math:`(1, 0)` here and :data:`~sbm.conventions.SKY_HWP_ANGLE` for the wedge
below. The temperature gradient sits at :math:`(dn, dm)`. The gradients of
:math:`P` start from the couple of :math:`P`, ``POLARIZATION_HWP`` =
``(pn, pm)``: :math:`\bar\eth P` at :math:`(pn - dn, pm - dm)` and
:math:`\eth P` at :math:`(pn + dn, pm + dm)`. ``_offset_components`` writes these
three components, and ``abs_pointing`` calls it with :math:`(dn, dm) = (1, 0)`:

.. literalinclude:: ../../sbm/systematics/pointing.py
   :pyobject: _offset_components
   :end-before: if second_order:

.. literalinclude:: ../../sbm/systematics/pointing.py
   :pyobject: abs_pointing

*Templates.* Those of Eqs. (6.71)–(6.72), the 9×9 map-maker, at the couples of the
code; :math:`(1, 0)` alone is the 5×5 one, Eqs. (6.69)–(6.70).

Pointing disturbance by the HWP wedge
-------------------------------------

*Thesis, Sect. 6.2.2.* A wedged HWP deflects the beam by an angle
:math:`\rho_w = (n_{\rm ref} - 1) w`, Eq. (6.36) (written :math:`\xi` in the
thesis), in a direction that rotates with the HWP. The thesis obtains the
signal, Eq. (6.37), by replacing :math:`\chi \to \phi + \chi` in Eq. (6.28),
i.e. a deflection along :math:`\psi + \phi + \chi`, with the spin components
Eqs. (6.38)–(6.42) at :math:`(1, 1)`, :math:`(-3, 3)` and :math:`(1, -5)`.

*Corrections,* in the convention of the thesis.

* The sense of the HWP angle. The deflection is fixed in the frame of the HWP,
  which turns by :math:`\phi` with respect to the telescope, and the thesis
  counts :math:`\phi` opposite to :math:`\psi`, Eq. (4.21). The direction on the
  sky is the angle of the telescope, :math:`\psi`, plus the angle of the HWP
  counted in the same sense, :math:`-\phi`: in the thesis the deflection points
  along :math:`\psi - \phi + \chi`. (In the code, which counts :math:`\psi` and
  :math:`\phi` in the same sense, it points along :math:`\psi + \phi + \chi`.)
* The indices of Eqs. (6.40)–(6.42). The :math:`\eth P` term of Eq. (6.37) is at
  :math:`(3, -3)`, not :math:`(-3, 3)`: the spin weight of :math:`\eth P` is 3,
  and the spin check fails at :math:`n = -3`.

With both corrections the components of the thesis are at :math:`(1, -1)`,
:math:`(1, -3)` and :math:`(3, -5)`, and in the code at :math:`(1, 1)`,
:math:`(1, 3)` and :math:`(3, 5)`.

*Reading,* in the convention of the code. The table of the absolute pointing
offset with :math:`\alpha = \psi + \phi + \chi` instead of :math:`\psi + \chi`:
every :math:`e^{\mp i\alpha}` now also carries :math:`e^{\mp i\phi}`, so
:math:`m` changes by :math:`\pm 1` together with :math:`n`.

.. list-table::
   :header-rows: 1
   :widths: 40 22 16 22

   * - term of the perturbation
     - :math:`\psi`, :math:`\phi` in the exponent
     - :math:`(n, m)`
     - spin weight of the field
   * - :math:`-\frac{\rho_w}{2} e^{-i\alpha}\, \eth I`
     - :math:`-(\psi + \phi)`
     - :math:`(1, 1)`
     - 1
   * - :math:`-\frac{\rho_w}{4} e^{i\alpha}\, \bar\eth P\, e^{-i(2\psi + 4\phi)}`
     - :math:`-(2\psi + 4\phi) + (\psi + \phi)`
     - :math:`(1, 3)`
     - 1
   * - :math:`-\frac{\rho_w}{4} e^{-i\alpha}\, \eth P\, e^{-i(2\psi + 4\phi)}`
     - :math:`-(2\psi + 4\phi) - (\psi + \phi)`
     - :math:`(3, 5)`
     - 3

:math:`\chi` is the direction of the deflection at :math:`\phi = 0`, measured
like that of the absolute offset (:math:`-\chi - \pi/2` of the thesis).

*Code.* :math:`(dn, dm)` = :data:`~sbm.conventions.SKY_HWP_ANGLE`; ``rho_rad`` is
:math:`\rho_w` and ``chi_rad`` is :math:`\chi`:

.. literalinclude:: ../../sbm/systematics/pointing.py
   :pyobject: hwp_wedge

*Templates.* The three couples, the 9×9 map-maker of Eqs. (6.75)–(6.76) with
them; the temperature term alone is the 5×5 one, Eqs. (6.73)–(6.74).

Instrumental polarization of the HWP
------------------------------------

*Thesis, Sect. 6.2.3.* The elements :math:`M_{QI}` and :math:`M_{UI}` of the HWP
Mueller matrix at :math:`4 f_\phi`, with amplitudes :math:`\epsilon_1`,
:math:`\epsilon_2` and phases :math:`\phi_{QI}`, :math:`\phi_{UI}`, leak the
temperature into the polarization (Patanchon et al. 2024, JCAP 04, 074).
Eq. (6.45) writes the leakage seen by a detector at polarization angle
:math:`\psi_0` as

.. math::

   \Delta S_{\rm ip} = \left[\epsilon_1 \cos(4\phi - 4\psi + \phi_{QI}) \cos 2\psi_0
   + \epsilon_2 \cos(4\phi - 4\psi + \phi_{UI}) \sin 2\psi_0\right] I ,

and, with :math:`\epsilon_1 = \epsilon_2`, :math:`\phi_{QI} = \phi_{UI} + \pi/2`
and :math:`\psi_0 = 0`, Eqs. (6.46)–(6.47),
:math:`{}_{4,-4}\Delta\tilde S_{\rm ip} = \frac{\epsilon_1}{2} e^{-i\phi_{QI}} I`.

*Correction.* The leakage arises in the instrument: it depends on the angle of
the HWP with respect to the telescope and on the polarizer of the detector, not
on the orientation of the telescope on the sky. With the HWP angle :math:`\phi`
of Eq. (4.21), the angle of the h-maps, it does not involve :math:`\psi`; the
:math:`-4\psi` of Eq. (6.45) treats :math:`\phi` as an angle on the sky
(:math:`(4, -4)` is the couple of this leakage in an expansion in the HWP angle
on the sky, :func:`~sbm.conventions.to_sky_hwp_angle`). The spin check shows it:
:math:`I` has spin weight 0 and cannot sit at :math:`n = 4`. With the HWP angle
:math:`\alpha = \phi` in the focal plane, Patanchon et al. (2024) give, for
:math:`\psi_0 = 0`,

.. math::

   \Delta d = \epsilon_1 I \cos(4\phi + \phi_{QI}),

the component :math:`\frac{\epsilon_1}{2}` :func:`~sbm.conventions.phase_factor`
:math:`(\phi_{QI})\, I` at :math:`(0, 4)` (in the thesis's convention the phase
would be :math:`e^{+i\phi_{QI}}`). (The second exponent of Eq. (6.46) should also
read :math:`-2\psi_0`, as in the first, for the leakage to be real; at
:math:`\psi_0 = 0` both agree.)

*Reading.* :math:`I` (spin weight 0) at :math:`n = 0`; the HWP harmonic
:math:`4 f_\phi` gives :math:`m = 4`. The model is the perturbation alone, as in
the thesis ("the deviation signal"); the thesis feeds it with the temperature
and the solar dipole.

*Code.* ``epsilon`` is :math:`\epsilon_1` and ``phase_rad`` is
:math:`\phi_{QI}`. The general Mueller matrix of the HWP is
:func:`~sbm.systematics.hwp_mueller`, in the chain of litebird_sim; the test
``test_hwp_ip_is_a_case_of_hwp_mueller`` builds :math:`M_{QI}` and
:math:`M_{UI}` of Patanchon et al. (2024) in that chain and finds this model:

.. literalinclude:: ../../sbm/systematics/hwp.py
   :pyobject: hwp_ip

*Template.* :math:`(0, 4)`, the 5×5 map-maker of Eqs. (6.77)–(6.78) with this
couple instead of :math:`(\pm 4, \mp 4)`.

Summary
-------

The couples of the thesis, in its convention, as printed and corrected; the
couples and the templates of the code are those of :doc:`systematics`, at
:func:`~sbm.conventions.from_0_4` of the corrected ones.

.. list-table::
   :header-rows: 1
   :widths: 24 14 20 42

   * - thesis
     - equations
     - function
     - couples of the thesis
   * - differential gain, Sect. 6.1.1
     - (6.3)–(6.9)
     - :func:`~sbm.systematics.diff_gain`
     - :math:`(0,0)`, :math:`(2,0)`
   * - differential pointing, Sect. 6.1.2
     - (6.16)–(6.25)
     - :func:`~sbm.systematics.diff_pointing`
     - :math:`(1,0)`, :math:`(3,0)`; :math:`-\frac18\bar\zeta` for :math:`P`
   * - absolute pointing offset, Sect. 6.2.1
     - (6.27)–(6.33)
     - :func:`~sbm.systematics.abs_pointing`
     - :math:`(1,0)`, :math:`(1,-4)`, :math:`(3,-4)`
   * - HWP wedge, Sect. 6.2.2
     - (6.36)–(6.42)
     - :func:`~sbm.systematics.hwp_wedge`
     - :math:`(1,-1)`, :math:`(1,-3)`, :math:`(3,-5)`, not :math:`(1,1)`,
       :math:`(-3,3)`, :math:`(1,-5)`
   * - HWP instrumental polarization, Sect. 6.2.3
     - (6.45)–(6.47)
     - :func:`~sbm.systematics.hwp_ip`
     - :math:`(0,4)`, not :math:`(4,-4)`
