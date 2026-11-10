from pathlib import Path

import pytest

from companion_tooling.generate import generate, main
from companion_tooling.inventory import load_inventory


def test_deterministic_generation(toy_repo: Path) -> None:
    assert generate(toy_repo / "inventory.tsv", toy_repo) == 8
    pages = [toy_repo / "LISTINGS.md", *sorted((toy_repo / "lst").glob("*.md"))]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in pages]
    assert generate(toy_repo / "inventory.tsv", toy_repo) == 8
    assert before == [(p.read_bytes(), p.stat().st_mtime_ns) for p in pages]
    assert "[chapters/ch04/one.py](../chapters/ch04/one.py)" in pages[1].read_text()
    assert "Chapter 04" in pages[1].read_text()
    assert "Needs earlier listing lst-4.1" in (toy_repo / "lst/lst-4.2.md").read_text()
    assert "Needs service toy-api" in (toy_repo / "lst/lst-4.3.md").read_text()
    assert len(pages) == 9


def test_cut_has_no_excerpt(toy_repo: Path) -> None:
    assert not (toy_repo / "excerpts/lst-4.4.txt").exists()
    assert main(["--inventory", str(toy_repo / "inventory.tsv")]) == 0
    assert (toy_repo / "lst/lst-4.4.md").is_file()


def test_missing_full_file(toy_repo: Path) -> None:
    (toy_repo / "chapters/ch04/one.py").unlink()
    assert main(["--inventory", str(toy_repo / "inventory.tsv")]) == 1
    assert not (toy_repo / "LISTINGS.md").exists()


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("lst-4.2\t04", "lst-4.1\t04"),
        ("lst-4.1\t04", "lst-04.1\t04"),
        ("lst-4.1\t04", "lst-4.1\t05"),
        ("chapters/ch04/one.py", "chapters/ch04/../../one.py"),
        ("chapters/ch04/one.py", "/chapters/ch04/one.py"),
        ("needs-earlier:lst-4.1", "needs-earlier:lst-5.1"),
        ("needs-service:toy-api", "unknown"),
    ],
)
def test_bad_inventory(toy_repo: Path, before: str, after: str) -> None:
    inventory = toy_repo / "inventory.tsv"
    inventory.write_text(inventory.read_text().replace(before, after))
    with pytest.raises(ValueError):
        load_inventory(inventory)


def test_symlink_escape(toy_repo: Path, tmp_path: Path) -> None:
    (toy_repo / "lst").symlink_to(tmp_path, target_is_directory=True)
    assert main(["--inventory", str(toy_repo / "inventory.tsv")]) == 1
