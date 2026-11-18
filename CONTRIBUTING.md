# Contributing to SBM

## Adding a systematic

A systematic is a function that returns the perturbation of the data of a
detector as spin components (a `SignalModel`). It knows neither the scan nor
the map-maker. The full guide is the documentation page
[Adding a systematic](docs/source/adding_systematics.rst), and
[`notebooks/adding_a_systematic.ipynb`](notebooks/adding_a_systematic.ipynb)
goes through it with an example. In short:

1. Write the perturbation in the conventions of
   [docs/source/conventions.rst](docs/source/conventions.rst):
   `d = sum S_{n,m} exp(i (n psi + m phi))`, `I` at `(0,0)`, `P/2` at `(2,0)`
   (no HWP) or `(2,-4)` (HWP).
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
`pip install -e ".[test,plot,docs,lint]"` in any virtual environment):

```
uv venv
uv pip install --python .venv/bin/python -e ".[test,plot,docs,lint]"
.venv/bin/python -m pytest tests
.venv/bin/ruff check sbm tests
.venv/bin/python -m sphinx -b html -W docs/source docs/build/html
```

The tests run from any directory and use no reference maps. The version and the
rules of Ruff are fixed in `pyproject.toml`, and the check reports no
diagnostics. Code, comments and documentation are in English.
