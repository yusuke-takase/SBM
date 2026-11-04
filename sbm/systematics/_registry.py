"""The ``@systematic`` decorator and the list of all the systematics.

A decorator is a function applied to another one where it is defined: writing
``@systematic(data="pair", templates=[(0, 0)])`` above ``def diff_gain(...)``
is the same as ``diff_gain = systematic(data="pair", templates=[(0, 0)])(diff_gain)``.
Here it only attaches two attributes to the function and appends it to
:data:`ALL`; the function is called as usual.
"""

from __future__ import annotations

#: Kinds of data a systematic describes:
#: ``"pair"`` the pair difference ``(d_T - d_B) / 2`` of two orthogonal detectors without HWP,
#: ``"single"`` a single detector without HWP,
#: ``"single_hwp"`` a single detector behind a HWP,
#: ``"any"`` several of these (chosen by the arguments of the function).
DATA = ("pair", "single", "single_hwp", "any")

#: Every function declared with :func:`systematic`, in the order of definition.
ALL: list = []


def systematic(data: str, templates=()):
    """Declare a systematic.

    Args:
        data: the kind of data the model describes (see :data:`DATA`).
        templates: couples to add to the basis (:meth:`sbm.Basis.extend`) to
            absorb the first-order perturbation; empty if there is none or if
            they depend on the arguments.

    The decorated function gets the attributes ``data`` and ``templates``, so
    that ``Basis.iqu(hwp=True).extend(abs_pointing.templates)`` works, and is
    added to :data:`ALL`, whose every member the tests check against TOD.
    """
    if data not in DATA:
        raise ValueError(f"data must be one of {DATA}, got {data!r}")
    templates = tuple((int(n), int(m)) for n, m in templates)

    def register(function):
        function.data = data
        function.templates = templates
        ALL.append(function)
        return function

    return register
