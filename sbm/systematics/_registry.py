"""The ``@systematic`` decorator and the list of the signal models of :mod:`sbm.systematics`.

A decorator is a function applied to another one where it is defined: writing
``@systematic(data="pair", templates=[(0, 0)])`` above ``def diff_gain(...)``
is the same as ``diff_gain = systematic(data="pair", templates=[(0, 0)])(diff_gain)``.
Here it only attaches two attributes to the function and appends it to
:data:`ALL` (or replaces the function of the same name and module, when a module
is reloaded); the function is called as usual.
"""

from __future__ import annotations

#: Kinds of data a systematic describes:
#: ``"pair"`` the pair difference ``(d_T - d_B) / 2`` of two orthogonal detectors without HWP,
#: ``"single"`` a single detector without HWP,
#: ``"single_hwp"`` a single detector behind a HWP,
#: ``"any"`` several of these (chosen by the arguments of the function).
DATA = ("pair", "single", "single_hwp", "any")

#: Every function declared with :func:`systematic`, in the order of definition: the signal
#: models of the package, i.e. the sky signal of an ideal detector (``sky_signal``) and the
#: systematics.
ALL: list = []


def systematic(data: str, templates=()):
    """Declare a signal model: a systematic, or the sky signal of an ideal detector.

    Args:
        data: the kind of data the model describes (see :data:`DATA`).
        templates: couples ``(n, m)`` to add to the basis (:meth:`sbm.Basis.extend`)
            to absorb the first-order perturbation, e.g. ``[(0, 4)]``; empty if
            there is none or if they depend on the arguments.

    The decorated function gets the attributes ``data`` and ``templates``, so
    that ``Basis.iqu(hwp=True).extend(abs_pointing.templates)`` works, and is
    added to :data:`ALL`, whose every member the tests check against TOD.
    """
    if data not in DATA:
        raise ValueError(f"data must be one of {DATA}, got {data!r}")
    try:
        templates = tuple((int(n), int(m)) for n, m in templates)
    except (TypeError, ValueError):
        raise ValueError(
            f"templates must be a list of (n, m) couples, e.g. [(0, 4)], got {templates!r}"
        ) from None

    def register(function):
        function.data = data
        function.templates = templates
        key = (function.__module__, function.__qualname__)
        for i, other in enumerate(ALL):
            if (other.__module__, other.__qualname__) == key:
                ALL[i] = function  # declared again, e.g. after reloading its module
                break
        else:
            ALL.append(function)
        return function

    return register
