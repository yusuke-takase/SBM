"""Signal models of the sky and of the systematics.

Each function returns a :class:`sbm.SignalModel`; the model of a detector is
the sum of the sky signal and of the perturbations of its systematics::

    model = sky_signal(sky, pair_difference=True) + diff_gain(sky, 0.01, -0.01)
    basis = Basis.qu(hwp=False).extend(diff_gain.templates)

The spins follow :mod:`sbm.signal_model`: ``d = sum S_{n,m} exp(i (n psi + m phi))``
with the crossing angle ``psi`` and the HWP angle ``phi``, the angle of the HWP
with respect to the telescope, counted opposite to ``psi``, so that the
polarization is modulated as ``cos(4 phi - 2 psi)`` and sits at ``(2, -4)``.
Pair differences ``(d_T - d_B) / 2`` are taken between two detectors in the
same position with orthogonal polarizers (B rotated by 90 degrees), without HWP.

The modules follow the path of the signal through the instrument:

* :mod:`~sbm.systematics.sky`: :func:`sky_signal`, the ideal detector;
* :mod:`~sbm.systematics.pointing`: :func:`diff_pointing`, :func:`abs_pointing`,
  :func:`hwp_wedge`;
* :mod:`~sbm.systematics.optics`: :func:`beam`;
* :mod:`~sbm.systematics.hwp`: :func:`hwp_mueller`, :func:`hwp_ip`;
* :mod:`~sbm.systematics.detector`: :func:`diff_gain`, :func:`bandpass_mismatch`,
  :func:`polarization_angle`.

To add a systematic, write its perturbation as spin components in the module of
the stage where it acts, declare it with :func:`systematic` (the kind of data
and the template couples), and register its real-space TOD in
``tests/test_systematics.py``; see the documentation page "Adding a systematic"
and ``notebooks/adding_a_systematic.ipynb``. :data:`ALL` lists every systematic.
"""

from ._registry import ALL, DATA, systematic
from .sky import sky_signal
from .pointing import abs_pointing, diff_pointing, hwp_wedge
from .optics import beam
from .hwp import IDEAL_HWP, hwp_ip, hwp_mueller
from .detector import bandpass_mismatch, diff_gain, polarization_angle

__all__ = [
    "ALL",
    "DATA",
    "IDEAL_HWP",
    "abs_pointing",
    "bandpass_mismatch",
    "beam",
    "diff_gain",
    "diff_pointing",
    "hwp_ip",
    "hwp_mueller",
    "hwp_wedge",
    "polarization_angle",
    "sky_signal",
    "systematic",
]
