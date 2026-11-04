Beam convolution with a rotating HWP
====================================

The HWP path uses SBM's signal convention

.. math::

   d(\Omega,\psi,\alpha)=\sum_{n,h}F_{n,h}(\Omega)e^{i(n\psi+h\alpha)}.

:func:`sbm.beam.convolve` accepts a real 4 by 4 Mueller matrix in the Stokes
IQUV basis. Its default is the ideal HWP, ``diag(1, 1, -1, -1)``. Harmonic
inputs use the full HEALPix m-major layout with TEB or TEBV components; missing
V is zero. For example::

    fields = sbm.beam.convolve(alm, blm, nside, spin_k=range(5), hwp=True, mueller=M)
    # fields[(n, h)] is the complete complex signal map F_{n,h}.

Without HWP (``hwp=False``) the same function returns the maps at ``h = 0``.

Both signed crossing moments are included. The possible HWP harmonics are
0, +/-2 and +/-4; absent harmonics are omitted. The reality relation is
``fields[(-n, -h)] == fields[(n, h)].conj()``. In particular, the two n=0 maps
at h=+2 and h=-2 can be complex conjugates rather than individually real.

To couple these fields to a scan and perform map-making::

    model = sbm.systematics.beam(alm, blm, nside, spin_k=range(5), hwp=True, mueller=M)
    output_iqu = sbm.map_make((hmaps, model), sbm.Basis.iqu(hwp=True)).iqu

``beam`` accepts general asymmetric beams; :func:`sbm.beam.elliptical_beam`
gives the maps of an elliptical Gaussian beam. These examples retain both odd
and even crossing moments through absolute n=4. Increase ``spin_k`` if the
beam requires additional moments.

For an ideal HWP and an axisymmetric co-polar beam the polarization is at
``(n, h) = (2, -4)`` and its conjugate. ``Basis.qu(hwp=True)`` fits
polarization only, ``Basis.iqu(hwp=True)`` fits IQU, and
``Basis.iqu(hwp=True).extend([(4, -4)])`` additionally fits the pair
``(4, -4)`` and ``(-4, 4)``. All the orientation functions required by the
differences between the basis and the signal couples must be present. A
general Mueller matrix can require HWP moments +/-2, +/-4, +/-6 and +/-8;
h-maps with only multiples of four are sufficient only when the signal has no
2f terms.

The implementation uses ``C = T M T^dagger`` without an extra conjugation.
For the entry ``C[X,Y]``, its HWP moment is ``h = sigma[X] - sigma[Y]``, with
``sigma = (0, 2, -2, 0)``. The input beam is the dual component of X at order
``n+h``. The crossing angle has the opposite sign to lbsim/Condor; the HWP
angle has the same sign.

The calculation follows the unmodulated convolution: ``beam_transfer``
extracts the beam coefficients, ``_clm`` contracts them with the sky, and
``_spin_map`` synthesizes the spin map. For each HWP harmonic ``spin_m`` the
same functions are used with one additional beam-filter step::

    blk = beam_transfer(blm, k, spin_m)  # beam order k + spin_m
    blk = _hwp_transfer(blk, spin_m, C, lmax)
    blk[:, :k] = 0.0  # a spin-k map starts at ell=k
    clm = _clm(sky_spin, blk)
    field = _spin_map(clm[0], k, nside, lmax) + _spin_map(clm[1], k, nside, lmax) / 2.0

``_hwp_transfer`` contains the five Mueller kernels for h=0, +/-2 and +/-4.
The sky contraction and map synthesis are shared with ``hwp=False``.

This is the shifted-beam narrow-beam approximation with the fundamental
spatial Mueller modes, as in the derivation supplied with the HWP project.
It is not an exact general spatial Mueller-field convolution: its accuracy
at low multipoles and for a broad beam is inherited from that approximation.
An identity Mueller matrix reproduces the unmodulated convolution.

Optional low-multipole intensity response
-----------------------------------------

``low_ell="beam_sum"`` adds the I-input monopole and dipole
2f response::

    model = sbm.systematics.beam(alm, blm, nside, spin_k=range(5), hwp=True,
                                 mueller=M, low_ell="beam_sum")
    output_iqu = sbm.map_make((hmaps, model), sbm.Basis.iqu(hwp=True)).iqu

The same option can be passed to :func:`sbm.beam.convolve`. The default
``low_ell="narrow_beam"`` retains the previous results and LBS comparison model.
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

The geometry is cached independently of beam, sky and Mueller matrix.
The Wigner 3j symbols (from ducc0) are evaluated only when a geometry cache
entry is first constructed.

The correction includes its n=0,+/-1 fields even if these moments were not
requested in ``spin_k``. Other terms still follow ``spin_k``: with the default
``spin_k=(0, 2, 4)``, an asymmetric beam's M_II response to the intensity dipole
at n=+/-1 remains omitted, even though the low-ell 2f correction is included.
The examples therefore use ``spin_k=range(5)`` to retain both odd and even
signal moments through absolute n=4.

For these examples, the h-maps must provide crossing-angle moments through
absolute n=6 for IQU fitting, or n=8 with the template ``(4, -4)``, including
odd moments, together with the required HWP moments. H-maps with only even
crossing moments cannot model the general asymmetric-beam dipole response.

The full sum is exact for this spatial Mueller model and the supplied
finite-band beam in the added channels. A finite-band pencil beam is not an
exact delta distribution, and its low-ell response need not equal the ideal
point-detector formula. General beam accuracy at ell>=2 is a separate question.
The retained coupling follows the real-space product before the narrow-beam
step in `Duivenvoorden's implementation notes, equations 51--59
<https://github.com/AdriJD/beamconv/files/8568824/beamconv_hwp.pdf>`_.
