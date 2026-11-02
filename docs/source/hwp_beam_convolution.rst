Beam convolution with a rotating HWP
========================================

The HWP path uses SBM's signal convention

.. math::

   d(\Omega,\psi,\alpha)=\sum_{n,h}F_{n,h}(\Omega)e^{i(n\psi+h\alpha)}.

``Convolver`` accepts a real 4 by 4 Mueller matrix in the Stokes IQUV basis.
Its default is the ideal HWP, ``diag(1, 1, -1, -1)``. Harmonic inputs use the
full HEALPix m-major layout with TEB or TEBV components; missing V is zero.
For example::

    sky = sbm.Convolver(alm, nside, spin_k=(0, 2, 4),
                        use_hwp=True, mueller=M)
    beam = sbm.Convolver(blm, nside, spin_k=(0, 2, 4), use_hwp=True)
    fields = sky * beam
    # fields[(n, h)] is the complete complex signal map F_{n,h}.

Both signed crossing moments are included. The possible HWP harmonics are
0, +/-2 and +/-4; absent harmonics are omitted. The reality relation is
``fields[(-n, -h)] == fields[(n, h)].conj()``. In particular, the two n=0 maps
at h=+2 and h=-2 can be complex conjugates rather than individually real.

To couple these fields to a scan and perform map-making::

    signal = sbm.SignalFields.beam_field(
        scan, mdim=3, alm=alm, blm=blm,
        use_hwp=True, mueller=M, spin_k=(0, 2, 4),
    )
    output_iqu = scan.map_make(signal)

``beam_field`` accepts general asymmetric beams. The original name
``elliptical_beam_field`` remains available as a compatibility alias.

For an ideal HWP and an axisymmetric co-polar beam the polarization is at
``(n, h) = (2, -4)`` and its conjugate. ``mdim=2`` fits polarization only,
``mdim=3`` fits IQU, and ``mdim=5`` additionally fits the pair ``(4, -4)`` and
``(-4, 4)``. All scan cross-links required by the differences between output
and signal pairs must be present. A general Mueller matrix can require HWP
cross-link moments +/-2, +/-4, +/-6 and +/-8; a database containing only
multiples of four is sufficient only when the active signal has no 2f terms.

The implementation uses ``C = T M T^dagger`` without an extra conjugation.
For the entry ``C[X,Y]``, its HWP moment is ``h = sigma[X] - sigma[Y]``, with
``sigma = (0, 2, -2, 0)``. The input beam is the dual component of X at order
``n+h``. The crossing angle has the opposite sign to lbsim/Condor; the HWP
angle has the same sign.

The calculation follows the existing unmodulated convolution: ``get_blk``
extracts the beam coefficients, ``get_clm`` contracts them with the sky,
and ``get_map`` synthesizes the spin map. For each HWP harmonic ``spin_m``,
the same functions are used with one additional beam-filter step::

    blk = beam.get_blk(k, spin_m)  # beam order k + spin_m
    blk = sky.get_hwp_blk(blk, spin_m, C)
    blk[:, :k] = 0.0  # a spin-k map starts at ell=k
    clm = sky.get_clm(blk, k)
    field = sky.get_map(clm[0], k) + sky.get_map(clm[1], k) / 2.0

``get_hwp_blk`` contains the five Mueller kernels for h=0, +/-2 and +/-4.
The sky contraction and map synthesis are shared with ``use_hwp=False``.

This is the shifted-beam narrow-beam approximation with the fundamental
spatial Mueller modes, as in the derivation supplied with the HWP project.
It is not an exact general spatial Mueller-field convolution: its accuracy
at low multipoles and for a broad beam is inherited from that approximation.
An identity Mueller matrix reproduces the unmodulated convolution, and
``use_hwp=False`` retains the existing array return format and behavior.

Optional low-multipole intensity response
----------------------------------------

``lowell_hwp="full_sum"`` adds the I-input monopole and dipole 2f response::

    signal = sbm.SignalFields.beam_field(
        scan, mdim=3, alm=alm, blm=blm, use_hwp=True, mueller=M,
        spin_k=(0, 2, 4), lowell_hwp="full_sum",
    )
    output_iqu = scan.map_make(signal)

The same option can be passed to the sky ``Convolver``. The default
``lowell_hwp="shifted"`` retains the previous results and LBS comparison model.
Only the I-input ell=0,1 terms proportional to M_QI and M_UI are replaced;
ell>=2, polarized inputs and V-input terms retain the shifted approximation.
Spin-2 beam harmonics at ell<2 remain zero: the extension constructs a scalar
effective beam by combining polarized beam harmonics with ell1>=2.

The spatial assumption is a Mueller matrix uniform in the beam-centred
Ludwig-3 frame. In spherical coordinates its spin-h shape is
``exp(-i*h*phi)`` for h=+/-2. The corresponding Gaunt sum is retained rather
than replacing all input beam multipoles ell1 by the output ell.
Triangle selection leaves ell2=ell1 for ell=0, and at most three ell2 values
for ell=1. The sum terminates at ell2<=beam_lmax+1 and costs O(beam_lmax)
per output coefficient; there is no extra sum per TOD sample.

Geometry is cached independently of beam, sky and Mueller matrix. The beam
object also caches its prepared filters. Changes to the relevant coefficients
of its mutable ``alm`` array invalidate that beam cache. Symbolic Wigner 3j
evaluation is used only when a geometry cache entry is first constructed.

The correction includes its n=0,+/-1 fields even if these moments were not
requested in ``spin_k``. Other terms still follow ``spin_k``. For an asymmetric
beam, the scan must therefore provide the odd crossing-angle cross-links
needed by these fields (through absolute n=3 for IQU fitting, or n=5 for
``mdim=5``), together with the required HWP moments. A database containing
only even crossing moments cannot model the general dipole correction.

The full sum is exact for this spatial Mueller model and the supplied
finite-band beam in the added channels. A finite-band pencil beam is not an
exact delta distribution, and its low-ell response need not equal the ideal
point-detector formula. General beam accuracy at ell>=2 is a separate question.
The retained coupling follows the real-space product before the narrow-beam
step in `Duivenvoorden's implementation notes, equations 51--59
<https://github.com/AdriJD/beamconv/files/8568824/beamconv_hwp.pdf>`_.
