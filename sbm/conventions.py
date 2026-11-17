r"""The conventions of SBM: the spin expansion of the data, the angles, and their
relations to litebird_sim, Falcons.jl, the files of SBM 0.4 and the literature.

This module is the only place where they are written. The rest of the package
takes them from here, the documentation shows this text and the tables built
from these values, and the tests check the examples below. The physical TOD of
the tests (:class:`sbm.testing.RealSky` and the anchor tests) are written out
independently, so that a change here that is not carried through the package
makes them fail.

.. rubric:: Angles

* :math:`\psi` (``psi``) is the crossing angle of the detector: the orientation
  of the reference axis of the focal plane on the sky, measured from the axis of
  :math:`Q`, and counted opposite to the polarization angle of :math:`Q` and
  :math:`U` (the angle of litebird_sim is :math:`\psi' = -\psi`,
  :func:`litebird_sim_angle`).
* :math:`\phi` (``phi``) is the HWP angle with respect to the telescope, on
  which the HWP is mounted, counted as the polarization angle (the angle
  :math:`\chi` of litebird_sim). On the sky the HWP is at :math:`\psi' + \phi
  = -\psi + \phi` (:data:`SKY_HWP_ANGLE`).

.. rubric:: Spin expansion

The data of a detector are

.. math::

   d(\psi, \phi) = \sum_{n,m} {}_{n,m}\tilde S \, e^{i(n\psi + m\phi)},
   \qquad {}_{-n,-m}\tilde S = {}_{n,m}\tilde S^*,

(:func:`synthesis_phase`; :data:`SYNTHESIS_SIGN` is the sign of the exponent),
the orientation functions of a pixel are the averages over its samples

.. math::

   {}_{n,m}\tilde h = \langle e^{-i(n\psi + m\phi)} \rangle,
   \qquad {}_{-n,-m}\tilde h = {}_{n,m}\tilde h^*,

(:func:`projection_parts`), and the binned data of the pixel are
:math:`\langle d\, e^{-i(n\psi + m\phi)} \rangle = \sum_{n',m'}
{}_{n-n',m-m'}\tilde h \; {}_{n',m'}\tilde S`. A model and the h-maps store
one couple per conjugate pair (:func:`canonical`). When the local basis turns by
an angle :math:`\alpha` in the sense of the polarization angle, a field of spin
weight :math:`s` is multiplied by :math:`e^{-is\alpha}` and :math:`\psi`
becomes :math:`\psi + \alpha`, so the component :math:`{}_{n,m}\tilde S` has
spin weight :math:`n`.

With :math:`P = Q + iU`, an ideal detector without HWP sees
:math:`I + Q\cos 2\psi - U\sin 2\psi`: :math:`I` at :math:`(0,0)` and
:math:`P/2` at :data:`POLARIZATION`. Behind an ideal HWP it sees
:math:`I + Q\cos(4\phi - 2\psi) + U\sin(4\phi - 2\psi)`: :math:`P/2` at
:data:`POLARIZATION_HWP`. A term :math:`A\cos(n\psi + m\phi + a)` is the
component :math:`\frac{A}{2}` :func:`phase_factor` :math:`(a)` at :math:`(n,m)`.

.. rubric:: Derivatives and offsets

:math:`\eth` and :math:`\bar\eth` are :data:`ETH_PHASE` times the standard
spin-raising and spin-lowering operators,
:math:`\eth_{\rm std} = -(\partial_\theta + i\partial_\phi/\sin\theta)`: on the
local tangent plane with :math:`x` along :math:`\theta` and :math:`y` along
:math:`\phi`, :math:`\eth = \partial_y - i\partial_x` and
:math:`\bar\eth = \partial_y + i\partial_x`. An offset of size :math:`\rho` in
direction :math:`\alpha` samples the sky at the point displaced by :math:`\rho`
along :math:`-(\sin\alpha\, e_\theta + \cos\alpha\, e_\phi)`, and

.. math::

   S(p') = S - \frac{\rho}{2}\left(e^{i\alpha}\eth + e^{-i\alpha}\bar\eth\right) S
   + \frac{\rho^2}{8}\left(e^{2i\alpha}\eth^2 + e^{-2i\alpha}\bar\eth^2\right) S
   + \frac{\rho^2}{4}\nabla^2 S + O(\rho^3).

A pointing offset fixed in the frame of the detector has
:math:`\alpha = \psi + \chi`, the deflection of a wedged HWP
:math:`\alpha = \psi - \phi + \chi` (it turns with the HWP on the sky,
:data:`SKY_HWP_ANGLE`).

.. rubric:: Other codes and files

* litebird_sim and Falcons.jl write
  :math:`h_{n,m} = \langle e^{+i(n\psi' + m\chi)} \rangle`; their couple of
  :math:`{}_{n,m}\tilde h` is :func:`litebird_sim_couple`, here
  :math:`h_{n,-m}`. Their TOD model behind a HWP of Mueller matrix :math:`M` is
  :math:`d = (1, 1, 0, 0) \cdot R(\chi)^T M R(\chi) R(\psi') S`, with
  :func:`stokes_rotation`.
* The cross-link files of SBM 0.4 (``HMaps.load_legacy``) and the
  couples of SBM 0.4 map to the couples here with :func:`from_0_4` (the
  identity: this is the convention of SBM 0.4).
* The literature that expands the data in the HWP angle on the sky,
  :math:`\rho = \psi' + \phi`, has the conjugate pair :func:`to_sky_hwp_angle`.

.. rubric:: Examples

>>> import numpy as np
>>> from sbm import conventions as cv
>>> cv.POLARIZATION, cv.POLARIZATION_HWP
((2, 0), (2, -4))
>>> psi, phi, i, q, u = 0.3, 1.1, 1.0, 0.2, -0.5
>>> p = q + 1j * u
>>> def tod(n, m, s):  # a component and its conjugate
...     return 2 * (s * cv.synthesis_phase(n, m, psi, phi)).real
>>> ideal = i + tod(*cv.POLARIZATION_HWP, p / 2)
>>> bool(np.isclose(ideal, i + q * np.cos(4 * phi - 2 * psi) + u * np.sin(4 * phi - 2 * psi)))
True
>>> without = i + tod(*cv.POLARIZATION, p / 2)
>>> bool(np.isclose(without, i + q * np.cos(2 * psi) - u * np.sin(2 * psi)))
True
>>> bool(np.isclose(tod(0, 4, 0.5 * cv.phase_factor(0.7)), np.cos(4 * phi + 0.7)))
True
>>> cv.litebird_sim_couple(2, -4), cv.from_0_4(2, -4), cv.to_sky_hwp_angle(2, -4)
((2, 4), (2, -4), (2, -4))
"""

from __future__ import annotations

import numpy as np

Couple = tuple[int, int]

#: Sign of the exponent of the spin expansion,
#: ``d = sum S_{n,m} exp(SYNTHESIS_SIGN i (n psi + m phi))``.
SYNTHESIS_SIGN = 1

#: Phase of the derivatives of SBM with respect to the standard spin-raising
#: operator, ``eth = ETH_PHASE eth_std`` and ``eth_bar = conj(ETH_PHASE) eth_bar_std``.
ETH_PHASE = 1j

#: Couple of ``P / 2`` (``P = Q + iU``) for an ideal detector without HWP.
POLARIZATION = (2, 0)

#: Couple of ``P / 2`` for an ideal detector behind a HWP.
POLARIZATION_HWP = (2, -4 * SYNTHESIS_SIGN)

#: Coefficients of ``psi`` and ``phi`` in the angle of the HWP on the sky,
#: counted as ``psi``; the deflection of a wedged HWP turns with it.
SKY_HWP_ANGLE = (1, -SYNTHESIS_SIGN)


def canonical(n: int, m: int) -> tuple[Couple, bool]:
    """Return the canonical couple of ``(n, m)`` and whether it is its conjugate.

    Every couple but ``(0, 0)`` is stored once: ``n > 0``, or ``n == 0`` and
    ``m > 0``. The others are the complex conjugates of their opposite.
    """
    n, m = int(n), int(m)
    if n > 0 or (n == 0 and m >= 0):
        return (n, m), False
    return (-n, -m), True


def synthesis_phase(n: int, m: int, psi, phi):
    """The factor of the component ``(n, m)`` in the data at the angles
    ``psi``, ``phi``: ``exp(SYNTHESIS_SIGN i (n psi + m phi))``."""
    return np.exp(SYNTHESIS_SIGN * 1j * (n * psi + m * phi))


def projection_parts(n: int, m: int, psi, phi):
    """Real and imaginary parts of the factor of the orientation function and of
    the binning, the conjugate of :func:`synthesis_phase`:
    ``cos(n psi + m phi)`` and ``-SYNTHESIS_SIGN sin(n psi + m phi)``.

    The orientation function is the average of ``re + i im``, and a component
    ``Z / 2`` at ``(n, m)`` plus its conjugate is ``Re(Z) re + Im(Z) im``.
    """
    theta = n * psi + m * phi
    return np.cos(theta), -SYNTHESIS_SIGN * np.sin(theta)


def phase_factor(angle_rad):
    """The factor ``exp(SYNTHESIS_SIGN i angle)``: a term
    ``A cos(n psi + m phi + angle)`` is the component
    ``A / 2 * phase_factor(angle)`` at ``(n, m)``, and an offset in direction
    ``psi + chi`` enters with ``phase_factor(chi)``."""
    return np.exp(SYNTHESIS_SIGN * 1j * angle_rad)


def litebird_sim_angle(psi):
    """The crossing angle of litebird_sim and Falcons.jl, ``psi'``, of SBM's ``psi``."""
    return -SYNTHESIS_SIGN * psi


def litebird_sim_couple(n: int, m: int) -> Couple:
    """The couple of the h-maps of litebird_sim and Falcons.jl that holds the
    orientation function ``(n, m)`` of SBM, and conversely (the relation is its
    own inverse)."""
    return int(n), -SYNTHESIS_SIGN * int(m)


def from_0_4(n: int, m: int) -> Couple:
    """The couple of a component or an orientation function ``(n, m)`` of SBM 0.4
    (and of its cross-link files), and conversely (the relation is
    its own inverse)."""
    return int(n), SYNTHESIS_SIGN * int(m)


def to_sky_hwp_angle(n: int, m: int) -> Couple:
    """The couple ``(n', m')`` of the component ``(n, m)`` in the literature that
    expands the data in ``psi'`` and the HWP angle on the sky
    ``rho = psi' + phi``, as the canonical member of the conjugate pair
    (the sign of the exponent differs between papers)."""
    return canonical(n + SYNTHESIS_SIGN * m, -SYNTHESIS_SIGN * m)[0]


def stokes_rotation(angle):
    """Rotation of the Stokes parameters ``(I, Q, U, V)`` by ``2 angle``
    (``Q' = cos Q + sin U``), as in litebird_sim; ``angle`` may be an array,
    with the matrix indices first."""
    c, s = np.cos(2 * angle), np.sin(2 * angle)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.array(
        [
            [one, zero, zero, zero],
            [zero, c, s, zero],
            [zero, -s, c, zero],
            [zero, zero, zero, one],
        ]
    )


def fourier_coefficients(values, axes=(0, 1)):
    """Spin components of a function sampled on uniform grids
    ``psi_j = 2 pi j / N`` and ``phi_l = 2 pi l / L`` along ``axes``: the
    coefficients ``c[n, m]`` (indices modulo ``N`` and ``L``) of
    ``values = sum c[n, m] synthesis_phase(n, m, psi, phi)``."""
    values = np.asarray(values)
    size = values.shape[axes[0]] * values.shape[axes[1]]
    if SYNTHESIS_SIGN > 0:
        return np.fft.fft2(values, axes=axes) / size
    return np.fft.ifft2(values, axes=axes)
