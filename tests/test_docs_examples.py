"""The code of the documentation (docs/examples), which the pages include, runs."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parents[1] / "docs" / "examples").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_runs(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(path), run_name="__main__")
