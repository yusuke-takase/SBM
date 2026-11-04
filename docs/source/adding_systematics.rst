Adding a systematic
===================

A systematic is a function that writes the perturbation of the data of a
detector as spin components. It does not know the scan nor the map-maker, so
adding one touches three places: the function, its test, and the table of
:doc:`systematics`. The tutorial ``notebooks/adding_a_systematic.ipynb`` goes
through the same steps with the example below.

Where things are
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Contents
   * - ``sbm/systematics/sky.py``
     - ``sky_signal``: the ideal detector, to which the perturbations are added
   * - ``sbm/systematics/pointing.py``
     - pointing offsets (``diff_pointing``, ``abs_pointing``, ``hwp_wedge``)
   * - ``sbm/systematics/optics.py``
     - the beam
   * - ``sbm/systematics/hwp.py``
     - the half-wave plate (``hwp_mueller``, ``hwp_ip``)
   * - ``sbm/systematics/detector.py``
     - gains, bandpasses, polarization angles
   * - ``sbm/systematics/_registry.py``
     - the ``@systematic`` decorator and the list ``ALL``
   * - ``sbm/signal_model.py``
     - :class:`.SignalModel`: spin components, sums and products
   * - ``sbm/sky.py``
     - :class:`.Sky`: ``i``, ``p``, ``eth_i``, ``eth_bar_i``, ``eth_p``,
       ``eth_bar_p`` and ``second_order``
   * - ``sbm/testing.py``
     - TOD in real space and their least-squares fit (:mod:`sbm.testing`)
   * - ``tests/test_systematics.py``
     - one or more cases per systematic, checked automatically

Step 1: the model
-----------------

Write the TOD with the systematic in the conventions of :doc:`conventions`
(``d = sum S_{n,m} exp(i (n psi + m phi))``, ``I`` at ``(0,0)``, ``P/2`` at
``(2,0)`` or ``(2,-4)``) and keep the difference with the ideal detector. For
an error ``delta`` on the polarization angle of a detector without HWP,
``d = I + Re[P e^{2i psi} e^{-2i delta}]``, so the perturbation is
``(e^{-2i delta} - 1) P / 2`` at ``(2, 0)``.

Step 2: the function
--------------------

Add it to the module of the stage where the systematic acts, declared with
``@systematic``:

.. code-block:: python

   @systematic(data="any")
   def polarization_angle(sky: Sky, delta_rad: float, hwp: bool = False) -> SignalModel:
       """Error delta_rad on the polarization angle of the detector(s). ..."""
       if hwp:
           return SignalModel({(2, -4): (np.exp(2j * delta_rad) - 1.0) * sky.p / 2.0})
       return SignalModel({(2, 0): (np.exp(-2j * delta_rad) - 1.0) * sky.p / 2.0})

and import it in ``sbm/systematics/__init__.py``.

* ``data`` is the kind of data the model describes: ``"pair"`` (pair
  difference without HWP), ``"single"``, ``"single_hwp"`` or ``"any"``.
* ``templates`` are the couples whose amplitudes the map-maker can fit to
  absorb the first-order perturbation (``Basis.extend(f.templates)``); leave it
  empty if there is none.
* A component given at ``(n, m)`` stands for itself and its conjugate at
  ``(-n, -m)``; components at the same or opposite couples are summed, and
  ``(0, 0)`` must be real.
* Angles end in ``_rad``; the docstring states the TOD model, the convention of
  the parameters and the templates.

The decorator only attaches ``data`` and ``templates`` to the function and
appends it to :data:`sbm.systematics.ALL`; the function is called as usual.

Step 3: the test
----------------

Write the TOD of the same signal in real space, without spin components, and
add a case to ``cases()`` in ``tests/test_systematics.py``:

.. code-block:: python

   delta = 0.05
   add(sy.polarization_angle, "polarization_angle-single", False,
       single + sy.polarization_angle(sky, delta),
       lambda px, ps, ph: lbs_polarizer_tod(real, px, -ps + delta))

``real`` is a :class:`sbm.testing.RealSky` (random values and derivatives in
every pixel, with a detector that samples the sky at an offset pointing) and
the TOD here use litebird_sim's angles. The tests then check that
:func:`.map_make` of the model is the least-squares fit of the TOD in several
bases. If the systematic has templates, add ``absorbed=lambda: (base, base +
perturbation)`` with parameters for which the templates absorb it exactly; the
tests check that the polarization does not change. A systematic without a case,
or with templates but no ``absorbed`` case, fails ``test_every_systematic_has_a_case``.

Step 4: the documentation
-------------------------

Add a row to the table of :doc:`systematics` (``test_every_systematic_is_documented``
checks it) and the function to :doc:`reference`.

Checklist
---------

#. Perturbation written as spin components, in the module of its stage.
#. ``@systematic(data=..., templates=...)`` and the import in ``__init__.py``.
#. A real-space TOD case (and an ``absorbed`` case if it has templates).
#. A row in :doc:`systematics`, an entry in :doc:`reference` and in ``CHANGELOG.md``.
#. ``pytest`` passes.

Other kinds of systematics
--------------------------

* **Signals that do not depend on the sky**, such as the differential emission
  of the optics seen by the two detectors of a pair, or the emission of the
  HWP: constant maps at ``(0, 0)`` (pair difference) or at ``(0, m)`` (HWP
  harmonics ``m``), e.g. ``SignalModel({(0, 2): a2 * np.ones(npix)})``.
* **Non-linear responses**: the product of two models is the model of the
  product of their signals, so ``d + beta d**2`` is ``d + beta * (d * d)`` for
  a model ``d`` that includes the sky and the other systematics.
* **The HWP**: non-idealities of its Mueller matrix, constant or varying with the
  HWP angle, are harmonics of :func:`~sbm.systematics.hwp_mueller`.
