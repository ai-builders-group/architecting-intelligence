import os
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"


def test_publication_ci_scope_and_pins() -> None:
    # These assertions cover the original checks job; execution has its own job.
    text = WORKFLOW.read_text().split("\n  run-listings:", 1)[0]
    assert "COMPANION_ROOT: '.'" in text
    assert "ruff check companion_tooling\n" in text
    assert "mypy companion_tooling\n" in text
    assert "python -m companion_tooling l0" in text
    assert "python -m companion_tooling check_pointers" in text
    assert "PYTHONPATH: tools" in text and "working-directory: tools" in text
    assert text.count('--repo "$COMPANION_ROOT"') == 2
    for unwanted in (
        "--excerpts", " generate ", "pytest", "ruff check .", "explicit-package-bases"
    ):
        assert unwanted not in text
    actions = re.findall(r"uses: (\S+)", text)
    assert len(actions) == 2
    assert all(re.fullmatch(r"actions/[\w-]+@[0-9a-f]{40}", action) for action in actions)


def test_run_listings_ci_job() -> None:
    text = WORKFLOW.read_text().split("\n  run-listings:\n", 1)[1]
    assert "python-version: '3.12'" in text
    assert "-r requirements/all.txt" in text
    assert "download.pytorch.org/whl/cpu" in text
    assert "python tools/run_listings.py" in text
    assert "working-directory:" not in text
    actions = re.findall(r"uses: (\S+)", text)
    assert len(actions) == 2
    assert all(re.fullmatch(r"actions/[\w-]+@[0-9a-f]{40}", action) for action in actions)


@pytest.mark.parametrize("broken", [False, True])
def test_ci_chapter_compile_uses_toy_files(toy_repo: Path, broken: bool) -> None:
    # Execute the workflow's actual heredoc; type errors alone must compile.
    text = WORKFLOW.read_text()
    code = text.split("python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    (toy_repo / "chapters/ch04/one.py").write_text(
        "def broken(" if broken else "value: int = 'toy'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", dedent(code)],
        env={**os.environ, "COMPANION_ROOT": str(toy_repo)},
        capture_output=True, text=True, check=False,
    )
    assert (result.returncode != 0) == broken
    if not broken:
        assert len(list(toy_repo.glob("chapters/**/__pycache__/*.pyc"))) == 8
