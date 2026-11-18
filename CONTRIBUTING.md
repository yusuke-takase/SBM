# Contributing to SBM

## Adding a systematic

A systematic is a function that returns the perturbation of the data of a
detector as spin components (a `SignalModel`). It knows neither the scan nor
the map-maker. The full guide is the documentation page
[Adding a systematic](docs/source/adding_systematics.rst), and
[`notebooks/adding_a_systematic.ipynb`](notebooks/adding_a_systematic.ipynb)
goes through it with an example. In short:

1. Write the perturbation in the conventions of `sbm/conventions.py` (shown on
   the documentation page [Conventions](docs/source/conventions.rst)); take the
   couples of the polarization from there instead of writing them as numbers.
2. Add the function to the module of `sbm/systematics/` where the systematic
   acts (`sky`, `pointing`, `optics`, `hwp`, `detector`), declared with
   `@systematic(data=..., templates=[...])`, and import it in
   `sbm/systematics/__init__.py`.
3. Add a case to `tests/test_systematics.py`: the same signal as TOD written in
   real space, without spin components (`sbm.testing.RealSky` helps). The tests
   compare the map-maker with the least-squares fit of these TOD, check the
   templates, and fail if a systematic has no case.
4. Add a row to [docs/source/systematics.rst](docs/source/systematics.rst), the
   function to [docs/source/reference.rst](docs/source/reference.rst), and an
   entry to `CHANGELOG.md`.

## Tests and style

From the repository root, with [uv](https://docs.astral.sh/uv/) (or
`pip install -e ".[test,plot,docs,lint,notebooks]"` in any virtual environment):

```
uv venv
uv pip install --python .venv/bin/python -e ".[test,plot,docs,lint,notebooks]"
.venv/bin/python -m pytest tests
.venv/bin/ruff check sbm tests docs/examples
.venv/bin/python -m sphinx -b html -W docs/source docs/build/html
```

The tests run from any directory and use no reference maps; with the extra
`notebooks` they also run the notebooks. The version and the rules of Ruff are
fixed in `pyproject.toml`, and the check reports no diagnostics. Code, comments
and documentation are in English.

## Conventions

The conventions (the spin expansion, the angles, the couples of the
polarization, the relations to litebird_sim and to SBM 0.4) are written once, in
`sbm/conventions.py`. The code takes them from there, the tables of the
documentation are built from the package (`docs/source/_ext`), `@systematic`
writes the templates into the docstrings, and the code shown by the
documentation is included from `docs/examples` and the tests. To change a
convention, change `sbm/conventions.py`; `tests/test_conventions.py` (anchor
tests that write the convention out, and the checks that keep it in one place)
and `tests/test_pointing_geometry.py` then list what does not follow.
