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
works. :data:`sbm.systematics.ALL` lists every systematic.

The modules follow the path of the signal through the instrument:
``sky`` (the ideal detector), ``pointing``, ``optics`` (the beam), ``hwp`` and
``detector``.

.. list-table::
   :header-rows: 1
   :widths: 22 18 34 26

   * - Function
     - Data
     - Perturbation
     - Templates
   * - :func:`~sbm.systematics.sky_signal`
     - any
     - :math:`(0,0)\colon I`, :math:`(2,0)\colon P/2` or :math:`(2,-4)\colon P/2`
     -
   * - :func:`~sbm.systematics.diff_gain`
     - pair difference
     - :math:`(0,0)\colon \frac{g_t-g_b}{2} I`, :math:`(2,0)\colon \frac{g_t+g_b}{4} P`
     - :math:`(0,0)`
   * - :func:`~sbm.systematics.diff_pointing`
     - pair difference
     - :math:`(1,0)\colon -\frac{\zeta}{4}\eth I - \frac{\bar\zeta^*}{8}\bar\eth P`,
       :math:`(3,0)\colon -\frac{\bar\zeta}{8}\eth P`; second order at
       :math:`(0,0), (2,0), (4,0)`
     - :math:`(1,0), (3,0)`
   * - :func:`~sbm.systematics.bandpass_mismatch`
     - pair difference
     - :math:`(0,0)\colon \sum_i \frac{\gamma_{t,i}-\gamma_{b,i}}{2} c_i`
     - :math:`(0,0)`
   * - :func:`~sbm.systematics.abs_pointing`
     - single detector, HWP
     - :math:`(1,0), (1,-4), (3,-4)`; second order at
       :math:`(0,0), (2,-4), (2,0), (4,-4), (0,4)`
     - :math:`(1,0), (3,-4), (1,-4)`
   * - :func:`~sbm.systematics.hwp_wedge`
     - single detector, HWP
     - :math:`(1,-1), (1,-3), (3,-5)`; second order at
       :math:`(0,0), (2,-4), (2,-2), (4,-6), (0,2)`
     - :math:`(1,-1), (3,-5), (1,-3)`
   * - :func:`~sbm.systematics.hwp_ip`
     - single detector, HWP
     - :math:`(0,4)\colon \frac{\epsilon}{2} e^{i\varphi_{QI}} I`
     - :math:`(0,4)`
   * - :func:`~sbm.systematics.hwp_mueller`
     - single detector, HWP
     - non-ideal Mueller matrix of the HWP, with its harmonics in the HWP angle
       (perturbation with respect to the ideal HWP)
     - depend on the harmonics: :math:`(0, k\pm2)` for the intensity
   * - :func:`~sbm.systematics.polarization_angle`
     - any
     - :math:`(2,0)\colon (e^{-2i\delta}-1) P/2` without HWP,
       :math:`(2,-4)\colon (e^{2i\delta}-1) P/2` with HWP
     -
   * - :func:`~sbm.systematics.beam`
     - any
     - whole signal through the beam and the HWP Mueller matrix
     - none declared; the :math:`4\psi` term of the beam is fitted by
       ``extend([(4, 0)])`` without HWP or ``extend([(4, -4)])`` with HWP

In the pair difference the offsets of T and B are
:math:`u_t = \rho_t e^{i\chi_t}` and :math:`u_b = i\rho_b e^{i\chi_b}` (B is
rotated by 90 degrees), :math:`\zeta = u_t - u_b` and
:math:`\bar\zeta = u_t + u_b`. The second-order terms of the pointing need
:attr:`.Sky.second_order`, computed with ducc0 from the harmonic coefficients
of the sky.

The h-maps must hold every couple between the basis and the components of the
model; :func:`.map_make` names the missing ones. For instance the second-order
diff pointing with the templates :math:`(1,0), (3,0)` needs :math:`n` up to 7.
