import subprocess
import sys
from pathlib import Path

import pytest

from companion_tooling.slice import check_slice, excerpt_lines


@pytest.mark.parametrize(
    "excerpt",
    [
        "```python\n# excerpt of lst-4.1\nx = 1  \n# ...\n\ny = 2\n```",
        "```python {#lst-4.1}\nx = 1\n# ...\ny = 2\n```",
        "# excerpt of lst-4.1\nx = 1\n    # ...\ny = 2\n",
    ],
)
def test_fence_contract(excerpt: str) -> None:
    assert excerpt_lines(excerpt)[0] == "lst-4.1"
    assert check_slice(excerpt, "x = 1\nignored = 0\ny = 2   \n") == "aligned"


@pytest.mark.parametrize(
    ("excerpt", "full", "expected"),
    [
        ("# excerpt of lst-4.1\nx\ny", "y\nx", "missing"),
        ("# excerpt of lst-4.1\nx", "x\nx", "ambiguous"),
        ("# excerpt of lst-4.1\nx\nx", "x\nx", "aligned"),
        ("# excerpt of lst-4.1\nx\nx", "x\nx\nx", "ambiguous"),
        ("# excerpt of lst-4.1\nx", " x", "aligned"),
        ("# excerpt of lst-4.1\n# ...", "x", "missing"),
        ("``` {#lst-4.1}\n# excerpt of lst-4.2\nx\n```", "x", "missing"),
        ("```python\n# excerpt of lst-4.1\nx", "x", "missing"),
        ("x", "x", "missing"),
    ],
)
def test_alignment(excerpt: str, full: str, expected: str) -> None:
    assert check_slice(excerpt, full) == expected


@pytest.mark.parametrize(
    ("code", "full", "expected"),
    [
        ("x = 1  # (1)  \ny = 2\t# (12)", "x = 1\ny = 2\n", "aligned"),
        ("x = 1\ny = 2", "x = 1  # (1)  \ny = 2\t# (12)\n", "aligned"),
        ("x = 1  # (12)", "x = 1  # (1)\n", "aligned"),
        ("x = 1", "x = 1  # (123)\n", "missing"),
        ("x = 1  # (123)", "x = 1  # (123)\n", "aligned"),
        ("x = 2  # (1)", "x = 1\n", "missing"),
        ("x = 1", "# --8<-- [start:toy]\nx = 1\n# --8<-- [end:toy]\n", "aligned"),
        ("# --8<-- [start:toy]", "# --8<-- [start:toy]\n", "missing"),
        ("x = 1  # (123)", "x = 1\n", "missing"),
        ("x = 1  # ()", "x = 1\n", "missing"),
        ("x = 1  # (1) extra", "x = 1\n", "missing"),
    ],
)
def test_alignment_addendum(code: str, full: str, expected: str) -> None:
    assert check_slice(f"# excerpt of lst-4.1\n{code}\n", full) == expected


@pytest.mark.parametrize("marker", ["...", "# ...", "# continues below", "# continues"])
def test_elisions_and_whitespace(marker: str) -> None:
    excerpt = f"# excerpt of lst-4.1\n    x = 1 \t\n  {marker}\t\n\ty = 2  \n"
    assert check_slice(excerpt, "\tx = 1\nignored = 0\n    y = 2\t\n") == "aligned"


@pytest.mark.parametrize(
    "header", ["# chapters/ch04/toy.py", "# chapters/ch04/toy.py:1-20", "# toy.py:L2-L9"]
)
def test_first_path_header_on_both_sides(header: str) -> None:
    for excerpt in (
        f"{header}\n# excerpt of lst-4.1\nx\n",
        f"# excerpt of lst-4.1\n{header}\nx\n",
        f"```python {{#lst-4.1}}\n{header}\nx\n```",
    ):
        assert check_slice(excerpt, "# elsewhere/toy.py:4-8\nx\n") == "aligned"


def test_only_first_path_comment_is_ignored() -> None:
    assert check_slice("# excerpt of lst-4.1\nx\n# toy.py\n", "x\n") == "missing"
    assert check_slice("# excerpt of lst-4.1\n# explanation\nx\n", "x\n") == "missing"
    assert check_slice("# excerpt of lst-4.1\nx...\n", "x\n") == "missing"


def test_cli_misaligned_and_missing(tmp_path: Path) -> None:
    excerpt, full = tmp_path / "excerpt.txt", tmp_path / "full.py"
    excerpt.write_text("# excerpt of lst-4.1\nx\ny\n")
    full.write_text("y\nx\n")
    result = subprocess.run(
        [sys.executable, "-m", "companion_tooling", "check_slice", str(excerpt), str(full)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode, result.stdout) == (1, "missing\n")
    full.unlink()
    result = subprocess.run(
        [sys.executable, "-m", "companion_tooling", "check_slice", str(excerpt), str(full)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode, result.stdout) == (1, "missing\n")
