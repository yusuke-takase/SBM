Adding a systematic
===================

A systematic is a function that writes the perturbation of the data of a
detector as spin components; it does not know the scan nor the map-maker.

1. Write the perturbation of the TOD in the form
   :math:`\delta d = \sum_{n,m} {}_{n,m}\tilde S\, e^{i(n\psi + m\phi)}` with
   the conventions of :doc:`conventions`, and read off one component per
   conjugate pair.
2. Add a function to ``sbm/systematics.py`` that returns them:

   .. code-block:: python

      def polarization_angle(sky: Sky, delta_rad: float) -> SignalModel:
          """Polarization angle error of a single detector without HWP.

          d = I + Re[P e^{2i(psi - delta)}]: the polarization is rotated by
          e^{-2i delta}. Template: none.
          """
          return SignalModel({(2, 0): (np.exp(-2j * delta_rad) - 1) * sky.p / 2})

   A component given at :math:`(n,m)` stands for itself and its conjugate at
   :math:`(-n,-m)`; components at the same or opposite couples are summed, and
   :math:`(0,0)` must be real. List the template couples in the docstring.
3. Test it against TOD written in real space, without the spin decomposition
   (see :doc:`testing`): simulate the detector response in ``tests/``, fit it
   per pixel with ``least_squares`` of ``tests/tod.py`` and compare with
   :func:`.map_make` for a few bases.

Nothing else has to change: the map-maker builds the system from the model,
the basis and the h-maps, and reports the couples that the h-maps lack.
