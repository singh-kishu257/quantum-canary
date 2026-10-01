"""Execute every README code block tagged exactly ```python (offline, on BasicSimulator)."""

import pathlib
import re

import pytest

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"^```python\n(.*?)^```", README.read_text(encoding="utf-8"), flags=re.S | re.M)


def test_readme_has_python_blocks():
    assert len(BLOCKS) >= 2


@pytest.mark.parametrize("code", BLOCKS)
def test_readme_block_runs(code):
    exec(compile(code, str(README), "exec"), {"__name__": "__readme__"})
