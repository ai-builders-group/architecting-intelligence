from pathlib import Path

import pytest

from companion_tooling.generate import generate
from companion_tooling.l0 import main


def run(repo: Path) -> int:
    return main(["--inventory", str(repo / "inventory.tsv"), "--excerpts", str(repo / "excerpts")])


def test_l0_clean(toy_repo: Path) -> None:
    generate(toy_repo / "inventory.tsv", toy_repo)
    assert run(toy_repo) == 0


@pytest.mark.parametrize(
    ("code", "full", "expected"),
    [
        ("x = 1  # (1)\ny = 2  # (12)", "x = 1\ny = 2\n", 0),
        ("x = 1\ny = 2", "x = 1  # (1)  \ny = 2\t# (12)\n", 0),
        ("x = 1  # (12)", "x = 1  # (1)\n", 0),
        ("x = 1", "x = 1  # (123)\n", 1),
        ("x = 1  # (123)", "x = 1  # (123)\n", 0),
        ("x = 2  # (1)", "x = 1\n", 1),
        ("x = 1", "# --8<-- [start:toy]\nx = 1\n# --8<-- [end:toy]\n", 0),
        ("# --8<-- [start:toy]", "# --8<-- [start:toy]\n", 1),
    ],
)
def test_l0_alignment_addendum(toy_repo: Path, code: str, full: str, expected: int) -> None:
    (toy_repo / "excerpts/lst-4.1.txt").write_text(f"# excerpt of lst-4.1\n{code}\n")
    (toy_repo / "chapters/ch04/one.py").write_text(full)
    assert run(toy_repo) == expected


@pytest.mark.parametrize(
    "fault",
    [
        "missing-file",
        "misaligned",
        "duplicate",
        "bad-id",
        "missing-excerpt",
        "wrong-header",
        "cut-excerpt",
    ],
)
def test_l0_failures(toy_repo: Path, fault: str) -> None:
    generate(toy_repo / "inventory.tsv", toy_repo)
    full = toy_repo / "chapters/ch04/one.py"
    excerpt = toy_repo / "excerpts/lst-4.1.txt"
    inventory = toy_repo / "inventory.tsv"
    if fault == "missing-file":
        full.unlink()
    elif fault == "misaligned":
        excerpt.write_text("# excerpt of lst-4.1\nnever_in_full_file()\n")
    elif fault in {"duplicate", "bad-id"}:
        inventory.write_text(
            inventory.read_text().replace(
                "lst-4.2\t04", "lst-4.1\t04" if fault == "duplicate" else "bad\t04"
            )
        )
    elif fault == "missing-excerpt":
        excerpt.unlink()
    elif fault == "wrong-header":
        excerpt.write_text(excerpt.read_text().replace("lst-4.1", "lst-4.9"))
    elif fault == "cut-excerpt":
        (toy_repo / "excerpts/lst-4.4.txt").write_text("# excerpt of lst-4.4\nx\n")
    assert run(toy_repo) == 1


def test_optional_excerpts(toy_repo: Path) -> None:
    assert main(["--inventory", str(toy_repo / "inventory.tsv")]) == 0


def test_ambiguous_passes_with_info(toy_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    full = toy_repo / "chapters/ch04/one.py"
    full.write_text(full.read_text() * 2)
    assert run(toy_repo) == 0
    assert capsys.readouterr().out == "L0: INFO ambiguous alignment: lst-4.1\nL0: PASS\n"
