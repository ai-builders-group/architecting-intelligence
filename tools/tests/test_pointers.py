from pathlib import Path

import pytest

from companion_tooling.generate import generate
from companion_tooling.pointers import main


def run(repo: Path) -> int:
    return main(
        ["--inventory", str(repo / "inventory.tsv"), "--pointers", str(repo / "pointers.txt")]
    )


def test_all_pointers(toy_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    generate(toy_repo / "inventory.tsv", toy_repo)
    assert run(toy_repo) == 0
    assert capsys.readouterr().out == "8/8\n"


@pytest.mark.parametrize("fault", ["file", "landing", "row", "duplicate", "empty"])
def test_pointer_misses(toy_repo: Path, capsys: pytest.CaptureFixture[str], fault: str) -> None:
    generate(toy_repo / "inventory.tsv", toy_repo)
    if fault == "file":
        (toy_repo / "chapters/ch04/one.py").unlink()
    elif fault == "landing":
        (toy_repo / "lst/lst-4.1.md").unlink()
    elif fault == "row":
        (toy_repo / "pointers.txt").write_text("lst-99.1\n")
    elif fault == "duplicate":
        (toy_repo / "pointers.txt").write_text("lst-4.1\nlst-4.1\n")
    else:
        (toy_repo / "pointers.txt").write_text("")
    assert run(toy_repo) == 1
    expected = {"row": "0/1", "duplicate": "1/2", "empty": "0/0"}.get(fault, "7/8")
    assert capsys.readouterr().out.strip() == expected


@pytest.mark.parametrize(
    "link",
    [
        "`[full](../chapters/ch04/one.py)`",
        "`../chapters/ch04/one.py`",
        "```markdown\n[full](../chapters/ch04/one.py)\n```",
        "    [full](../chapters/ch04/one.py)",
        "<!-- [full](../chapters/ch04/one.py) -->",
        "![full](../chapters/ch04/one.py)",
        "[full](../chapters/ch04/missing.py)",
        "[full](https://example.invalid/one.py)",
    ],
)
def test_non_clickable_or_wrong_link(toy_repo: Path, link: str) -> None:
    generate(toy_repo / "inventory.tsv", toy_repo)
    (toy_repo / "lst/lst-4.1.md").write_text(link)
    assert run(toy_repo) == 1
