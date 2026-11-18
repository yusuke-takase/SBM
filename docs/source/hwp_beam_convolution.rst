Beam convolution with a rotating HWP
====================================

The maps ``fields[(n, m)]`` are the spin components of the signal in the
convention of :mod:`sbm.conventions`, with the crossing angle :math:`\psi` and
the HWP angle :math:`\alpha`. Internally the HWP harmonic is :math:`h`, the
factor :math:`e^{ih\alpha}` of the signal, at the component
:math:`m = \sigma h` with :math:`\sigma` = :data:`~sbm.conventions.SYNTHESIS_SIGN`.

:func:`sbm.beam.convolve` accepts a real 4 by 4 Mueller matrix in the Stokes
IQUV basis. Its default is the ideal HWP, ``diag(1, 1, -1, -1)``. Harmonic
inputs use the full HEALPix m-major layout with TEB or TEBV components; missing
V is zero. For example::

    fields = sbm.beam.convolve(alm, blm, nside, spin_k=range(5), hwp=True, mueller=M)
    # fields[(n, m)] is the complete complex signal map F_{n,m}.

Without HWP (``hwp=False``) the same function returns the maps at ``m = 0``.

Both signed crossing moments are included. The possible HWP harmonics are
0, +/-2 and +/-4; absent harmonics are omitted. The reality relation is
``fields[(-n, -m)] == fields[(n, m)].conj()``. In particular, the two n=0 maps
at m=+2 and m=-2 can be complex conjugates rather than individually real.

To couple these fields to a scan and perform map-making::

    model = sbm.systematics.beam(alm, blm, nside, spin_k=range(5), hwp=True, mueller=M)
    output_iqu = sbm.map_make((hmaps, model), sbm.Basis.iqu(hwp=True)).iqu

``beam`` accepts general asymmetric beams, as :class:`sbm.SphericalHarmonics`:
:func:`sbm.beam.pencil`, :func:`sbm.beam.gaussian`, :func:`sbm.beam.elliptical`
or :meth:`sbm.SphericalHarmonics.from_maps` of beam maps. These examples retain both odd
and even crossing moments through absolute n=4. Increase ``spin_k`` if the
beam requires additional moments.

For an ideal HWP and an axisymmetric co-polar beam the polarization is at
|pol_hwp| and its conjugate. ``Basis.qu(hwp=True)`` fits polarization only,
``Basis.iqu(hwp=True)`` fits IQU, and extending it with |beam_4psi_hwp|
additionally fits the :math:`4\psi` term of the beam and its conjugate. All the
orientation functions required by the
differences between the basis and the signal couples must be present. A
general Mueller matrix can require HWP moments +/-2, +/-4, +/-6 and +/-8;
h-maps with only multiples of four are sufficient only when the signal has no
2f terms.

The implementation uses ``C = T M T^dagger`` without an extra conjugation.
For the entry ``C[X,Y]``, its HWP moment is ``h = sigma[X] - sigma[Y]``, with
``sigma = (0, 2, -2, 0)``. The input beam is the dual component of X at order
``n+h``, and the field is the component ``m = SYNTHESIS_SIGN h``. The crossing
and HWP angles are those of litebird_sim.

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
An identity Mueller matrix reproduces the unmodulated convolution. For a beam
whose centre is off the boresight, the polarization terms of the offset (the
templates of :func:`~sbm.systematics.abs_pointing` but :math:`(1,0)`) differ
from those of the exact rigid offset by an
amount that falls linearly with the width of the beam: in the tests, a Gaussian
beam offset by 5 % of its FWHM is off by about 0.8 % of these terms per degree
of FWHM (``tests/test_beam_pointing.py``). Without HWP the convolution is exact.

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
absolute n=6 for IQU fitting, or n=8 with the template |beam_4psi_hwp|, including
odd moments, together with the required HWP moments. H-maps with only even
crossing moments cannot model the general asymmetric-beam dipole response.

The full sum is exact for this spatial Mueller model and the supplied
finite-band beam in the added channels. A finite-band pencil beam is not an
exact delta distribution, and its low-ell response need not equal the ideal
point-detector formula. General beam accuracy at ell>=2 is a separate question.
The retained coupling follows the real-space product before the narrow-beam
step in `Duivenvoorden's implementation notes, equations 51--59
<https://github.com/AdriJD/beamconv/files/8568824/beamconv_hwp.pdf>`_.
