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


def test_titles_run_command_and_empty_chapters(toy_repo: Path) -> None:
    meta = toy_repo / "listings_meta.tsv"
    rows = [f"lst-{c}.{n}\tToy {c}.{n}\tShows toy {c}.{n}." for c in (4, 5) for n in (1, 2, 3, 4)]
    meta.write_text("id\ttitle\twhat\n" + "\n".join(rows) + "\n")
    chapters = toy_repo / "chapters.tsv"
    chapters.write_text("chapter\ttitle\n03\tEarlier\n04\tToys\n05\tMore toys\n")
    assert generate(toy_repo / "inventory.tsv", toy_repo, meta, chapters) == 8
    page = (toy_repo / "lst/lst-4.1.md").read_text()
    assert page.startswith("# Toy 4.1 (lst-4.1)\n\nShows toy 4.1.\n")
    assert "- **Chapter:** 04 Toys" in page and "- **Run:** `python chapters/ch04/one.py`" in page
    assert "**Run:**" not in (toy_repo / "lst/lst-4.4.md").read_text()  # a sketch has no run command
    listing = (toy_repo / "LISTINGS.md").read_text()
    assert "| 03 Earlier | No code listings in this chapter | | |" in listing
    assert listing.index("03 Earlier") < listing.index("04 Toys") < listing.index("05 More toys")
    meta.write_text("id\ttitle\twhat\n" + "\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError):
        generate(toy_repo / "inventory.tsv", toy_repo, meta, chapters)


def test_numbers_only_chapters(toy_repo: Path) -> None:
    meta = toy_repo / "listings_meta.tsv"
    rows = [f"lst-{c}.{n}\tToy {c}.{n}\tShows toy {c}.{n}." for c in (4, 5) for n in (1, 2, 3, 4)]
    meta.write_text("id\ttitle\twhat\n" + "\n".join(rows) + "\n")
    chapters = toy_repo / "chapters.tsv"
    chapters.write_text("chapter\ttitle\n03\t\n04\t\n05\t\n")
    generate(toy_repo / "inventory.tsv", toy_repo, meta, chapters)
    assert "| 03 | No code listings in this chapter | | |" in (toy_repo / "LISTINGS.md").read_text()
    assert "- **Chapter:** 04\n" in (toy_repo / "lst/lst-4.1.md").read_text()


def test_needs_line_and_run_rules(toy_repo: Path) -> None:
    (toy_repo / "requirements").mkdir()
    (toy_repo / "requirements/core.txt").write_text("pydantic>=2.7\n")
    (toy_repo / "requirements/ml.txt").write_text("-r core.txt\nscikit-learn>=1.5\n")
    (toy_repo / "chapters/ch04/one.py").write_text("import json\nfrom sklearn import svm\n")
    meta = toy_repo / "listings_meta.tsv"
    rows = [f"lst-{c}.{n}\tToy {c}.{n}\tShows toy {c}.{n}." for c in (4, 5) for n in (1, 2, 3, 4)]
    meta.write_text("id\ttitle\twhat\n" + "\n".join(rows) + "\n")
    generate(toy_repo / "inventory.tsv", toy_repo, meta)
    one = (toy_repo / "lst/lst-4.1.md").read_text()
    assert "`scikit-learn` (`pip install -r requirements/ml.txt`)" in one
    two = (toy_repo / "lst/lst-4.2.md").read_text()
    assert "the definitions from [lst-4.1](lst-4.1.md)" in two and "**Run:**" not in two
    assert "toy-api (see [SERVICES](../docs/SERVICES.md))" in (toy_repo / "lst/lst-4.3.md").read_text()


def test_table_breaking_title_rejected(toy_repo: Path) -> None:
    meta = toy_repo / "listings_meta.tsv"
    rows = [f"lst-{c}.{n}\tToy | {c}.{n}\tShows." for c in (4, 5) for n in (1, 2, 3, 4)]
    meta.write_text("id\ttitle\twhat\n" + "\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        generate(toy_repo / "inventory.tsv", toy_repo, meta)


def test_known_issue_line(toy_repo: Path) -> None:
    meta = toy_repo / "listings_meta.tsv"
    rows = [f"lst-{c}.{n}\tToy {c}.{n}\tShows toy {c}.{n}." for c in (4, 5) for n in (1, 2, 3, 4)]
    meta.write_text("id\ttitle\twhat\n" + "\n".join(rows) + "\n")
    issues = toy_repo / "known_issues.tsv"
    issues.write_text("id\tnote\nlst-4.1\tfails on toy input; a fix is planned.\n")
    generate(toy_repo / "inventory.tsv", toy_repo, meta, None, issues)
    page = (toy_repo / "lst/lst-4.1.md").read_text()
    assert "- **Class:** Runs on its own\n- **Known issue:** Fails on toy input; a fix is planned.\n" in page
    assert "Known issue" not in (toy_repo / "lst/lst-4.2.md").read_text()
    issues.write_text("id\tnote\nlst-9.9\tno such id.\n")
    with pytest.raises(ValueError):
        generate(toy_repo / "inventory.tsv", toy_repo, meta, None, issues)
