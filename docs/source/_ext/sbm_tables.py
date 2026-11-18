"""Tables of the documentation built from the package when the documentation is
built, so that they follow :mod:`sbm.conventions` and the registry of
:mod:`sbm.systematics` instead of copying them.

``.. sbm-table:: components`` lists the components of the ideal detectors,
``.. sbm-table:: relations`` the couples of a few components in the other
conventions, and ``.. sbm-table:: systematics`` the signal models of
:data:`sbm.systematics.ALL` with their data and templates.
"""

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

DATA_LABELS = {
    "pair": "pair difference",
    "single": "single detector",
    "single_hwp": "single detector, HWP",
    "any": "any",
}


def couple(c):
    return f":math:`({c[0]},{c[1]})`"


def list_table(header, rows, widths):
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        f"   :widths: {' '.join(str(w) for w in widths)}",
        "",
    ]
    for row in [header, *rows]:
        lines.append(f"   * - {row[0]}")
        lines.extend(f"     - {cell}" for cell in row[1:])
    return lines


def components_table():
    from sbm import conventions as cv

    rows = [
        ["single detector", couple((0, 0)) + ": :math:`I`",
         couple(cv.POLARIZATION) + ": :math:`P/2`"],
        ["single detector, HWP", couple((0, 0)) + ": :math:`I`",
         couple(cv.POLARIZATION_HWP) + ": :math:`P/2`"],
        ["pair difference", "none", couple(cv.POLARIZATION) + ": :math:`P/2`"],
    ]
    return list_table(["data", "intensity", "polarization"], rows, (30, 30, 40))


def relations_table():
    from sbm import conventions as cv

    examples = [
        ("polarization, HWP", cv.POLARIZATION_HWP),
        ("polarization, no HWP", cv.POLARIZATION),
        ("HWP-IP", (0, 4)),
        ("intensity term of the wedge", cv.SKY_HWP_ANGLE),
    ]
    rows = [
        [
            label,
            couple(c),
            couple(cv.litebird_sim_couple(*c)),
            couple(cv.from_0_4(*c)),
            couple(cv.to_sky_hwp_angle(*c)),
        ]
        for label, c in examples
    ]
    header = [
        "component",
        "SBM",
        "litebird_sim, Falcons.jl (h-maps)",
        "SBM 0.4",
        "HWP angle on the sky (pair)",
    ]
    return list_table(header, rows, (28, 12, 22, 18, 20))


def systematics_table():
    from sbm import systematics as sy

    rows = []
    for f in sy.ALL:
        summary = (f.__doc__ or "").strip().splitlines()[0] if f.__doc__ else ""
        templates = ", ".join(couple(t) for t in f.templates) or "none"
        rows.append(
            [
                f":func:`~sbm.systematics.{f.__name__}`",
                DATA_LABELS[f.data],
                summary,
                templates,
            ]
        )
    return list_table(["Function", "Data", "Model", "Templates"], rows, (22, 18, 40, 20))


TABLES = {
    "components": components_table,
    "relations": relations_table,
    "systematics": systematics_table,
}


class SbmTable(SphinxDirective):
    required_arguments = 1

    def run(self):
        name = self.arguments[0]
        if name not in TABLES:
            raise self.error(f"unknown table {name!r}, expected one of {sorted(TABLES)}")
        container = nodes.container()
        self.state.nested_parse(
            StringList(TABLES[name](), source=f"<sbm-table {name}>"),
            self.content_offset,
            container,
        )
        return container.children


def setup(app):
    app.add_directive("sbm-table", SbmTable)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
