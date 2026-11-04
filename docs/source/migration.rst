Migration from the former API
=============================

The former classes and functions are removed, without aliases.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Former
     - Now
   * - ``ScanFields.load_hdf5(path)``, ``read_scanfiled(path)``
     - ``HMaps.load_legacy(path)``
   * - ``ScanFields.load_lbs_h_maps(paths)``
     - ``HMaps.load(path)``; ``HMaps.combine([...])`` for several files
   * - ``ScanFields.load_det``, ``load_channel``, ``load_full_FPU``, ``install_db``
     - removed: pass the files of the detectors explicitly
   * - ``scan.get_xlink(n, m)``
     - ``hmaps.h(n, m)``
   * - ``scan.hitmap``, ``scan.nside``, ``scan.npix``
     - ``hmaps.hits``, ``hmaps.nside``, ``hmaps.npix``
   * - ``scan.create_covmat(spin_n_basis, spin_m_basis)`` (shape ``(k, k, npix)``)
     - ``orientation_matrix(hmaps, basis)`` (shape ``(npix, k, k)``)
   * - ``SignalFields.xxx_field(scan, mdim, ...)`` and ``scan.map_make(signal)``
     - ``map_make((hmaps, model), basis).iqu`` with ``model`` from
       :mod:`sbm.systematics` and ``basis`` from :class:`.Basis`
   * - ``mdim``
     - :class:`.Basis`: ``Basis.qu(hwp)`` (2), ``Basis.iqu(hwp)`` (3),
       ``.extend([...])`` for the templates (see :doc:`systematics`)
   * - ``scan.xlink_threshold`` (no HWP)
     - ``mask=hmaps.crosslink_mask(threshold)``; by default the observed,
       well-conditioned pixels are solved
   * - unsolved pixels set to 0
     - unsolved pixels are ``NaN`` (``result.mask`` tells which)
   * - ``diff_gain_field(scan, mdim, g_t, g_b, I, P)``
     - ``sky_signal(sky, pair_difference=True) + diff_gain(sky, g_t, g_b)``
   * - ``diff_pointing_field(scan, mdim, rho_T, rho_B, chi_T, chi_B, ...)``
     - ``sky_signal(sky, pair_difference=True) + diff_pointing(sky, rho_t, chi_t, rho_b, chi_b)``
       (note the order of the arguments)
   * - ``abs_pointing_field(scan, mdim, rho, chi, ...)``
     - ``sky_signal(sky, hwp=True) + abs_pointing(sky, rho, chi)``
   * - ``circular_pointing_field(...)``, ``hwp_ip_field(...)``
     - ``hwp_wedge(sky, rho, chi)``, ``hwp_ip(sky, epsilon, phase)``, in the
       ``"telescope"`` convention only; the ``"legacy"`` and ``"sky"`` HWP
       angles are removed. ``hwp_ip`` returns the leakage only, as before
   * - ``bandpass_mismatch_field(scan, mdim, P, gamma_T, gamma_B, comps)``
     - ``sky_signal(sky, pair_difference=True) + bandpass_mismatch(comps, gamma_t, gamma_b)``
   * - ``beam_field(scan, mdim, alm, blm, use_hwp, spin_k, mueller, lowell_hwp)``
     - ``beam(alm, blm, nside, spin_k, hwp=..., mueller=..., low_ell=...)``; ``lowell_hwp="shifted"`` / ``"full_sum"`` are ``low_ell="narrow_beam"`` / ``"beam_sum"``
   * - ``Convolver(sky, ...) * Convolver(beam, ...)``
     - ``sbm.beam.convolve(sky_alm, beam_alm, nside, spin_k, hwp=...)``, which
       always returns ``{(n, m): map}``
   * - ``Field``, ``SignalFields(...)``, ``get_coupled_field``,
       ``build_linear_system``
     - ``SignalModel({(n, m): map})``, ``model.project(hmaps, n, m)``; the
       system is built by :func:`.map_make`
   * - eth maps passed to every builder
     - :class:`.Sky` (``Sky.from_maps``, ``Sky.from_alm``), which computes them
       with the same healpy calls, or takes them
   * - ``pointing_second_order_maps(...)`` and ``second_order=dict``
     - ``second_order=True``; the maps are :attr:`.Sky.second_order`
   * - ``generate_noise_pdf`` and ``generate_noise(spin_n, spin_m, seed)``
     - ``noise_maps(hmaps, basis, net_ukrts, sampling_rate_hz, rng)``, with the
       correlations between the components; the former maps had independent
       components with twice the standard deviation of Q and U of an ideal scan
   * - ``pa45``, ``t2b``
     - removed (zero polarization angle)
   * - ``Configlation``, ``Systematics``, ``sim_*``, ``generate_maps``,
       ``get_instrument_table``
     - removed (they depended on litebird_sim)
   * - ``sbm.tools.*``
     - ``sbm.spectra`` (fiducial spectra, ``generate_cmb``, ``c2d``, ``d2c``),
       ``sbm.likelihood`` (``forecast``), ``sbm.plotting`` (``get_cmap``); the
       top-level names are unchanged

``generate_cmb(nside, r, cmb_seed)`` and ``forecast`` give the same numbers as
before; the fiducial spectra are shipped with SBM.
