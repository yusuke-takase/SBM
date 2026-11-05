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

.. code-block:: python

   import numpy as np
   import sbm

   nside = 8
   rng = np.random.default_rng(0)
   npix = 12 * nside**2
   pixels = np.repeat(np.arange(npix), 50)
   psi, phi = rng.uniform(-np.pi, np.pi, (2, pixels.size))
   couples = [(n, m) for n in range(9) for m in range(-12, 13, 4)]
   hmaps = sbm.HMaps.from_samples(nside, pixels, psi, phi, couples=couples)

``hmaps.h(n, m)`` returns :math:`{}_{n,m}\tilde h` (the conjugate couple is
computed), ``hmaps.hits`` the hit counts.

Signal models
-------------

A model is the sum of the sky signal and of the perturbations of the
systematics:

.. code-block:: python

   from sbm import systematics as sy

   sky = sbm.Sky.from_maps(*sbm.generate_cmb(nside, cmb_seed=1))
   model = sy.sky_signal(sky, hwp=True) + sy.hwp_ip(sky, epsilon=1e-4, phase_rad=0.3)

Map-making
----------

The basis fixes the components solved in every pixel; templates of
systematics are added with :meth:`sbm.Basis.extend`:

.. code-block:: python

   iqu = sbm.map_make((hmaps, model), sbm.Basis.iqu(hwp=True)).iqu
   cleaned = sbm.map_make((hmaps, model), sbm.Basis.iqu(hwp=True).extend([(0, 4)])).iqu

The pixels that cannot be solved are ``NaN``. With several detectors, pass a
list of ``(HMaps, SignalModel)``: their systems, weighted by the hit counts,
are summed before solving.

Noise and forecasts
-------------------

.. code-block:: python

   rng = np.random.default_rng(2)
   noise = sbm.noise_maps(hmaps, sbm.Basis.iqu(hwp=True), net_ukrts=50.0,
                          sampling_rate_hz=19.0, rng=rng).iqu

:func:`sbm.forecast` turns a residual B-mode spectrum into a bias on the
tensor-to-scalar ratio with the likelihood of the LiteBIRD forecasts.
