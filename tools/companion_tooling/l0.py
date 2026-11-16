"""Validate inventory, optional excerpt fixtures and portable listing output."""

import argparse
from pathlib import Path

from companion_tooling.inventory import load_inventory, local_path
from companion_tooling.patterns import denylist_terms, host_pattern, local_paths
from companion_tooling.slice import check_slice, excerpt_lines


def check_l0(
    inventory: Path, repo: Path, excerpts: Path | None = None, denylist: Path | None = None
) -> tuple[bool, list[str]]:
    rows = load_inventory(inventory)
    ambiguous: list[str] = []
    terms = denylist_terms(denylist) if denylist is not None else []
    portable = host_pattern()
    for row in rows:
        full = local_path(repo, row.full_path)
        if not full.is_file():
            return False, ambiguous
        texts = [row.full_path, full.read_text(encoding="utf-8")]
        if excerpts is not None:
            excerpt = local_path(excerpts, f"{row.id}.txt")
            if row.kind == "excerpt":
                text = excerpt.read_text(encoding="utf-8")
                alignment = check_slice(text, texts[1])
                if excerpt_lines(text)[0] != row.id or alignment == "missing":
                    return False, ambiguous
                if alignment == "ambiguous":
                    ambiguous.append(row.id)
                texts.append(text)
            elif excerpt.exists():
                return False, ambiguous
        page = local_path(repo, f"lst/{row.id}.md")
        if page.exists():
            texts.append(page.read_text(encoding="utf-8"))
        if any(any(local_paths(text, terms)) or portable.search(text) for text in texts):
            return False, ambiguous
    index = local_path(repo, "LISTINGS.md")
    if index.exists():
        text = index.read_text(encoding="utf-8")
        if any(local_paths(text, terms)) or portable.search(text):
            return False, ambiguous
    return True, ambiguous


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--excerpts", type=Path)
    parser.add_argument("--denylist", type=Path)
    args = parser.parse_args(argv)
    ambiguous: list[str] = []
    try:
        passed, ambiguous = check_l0(
            args.inventory, args.repo or args.inventory.parent, args.excerpts, args.denylist
        )
    except (ValueError, OSError, UnicodeError):
        passed = False
    for listing_id in ambiguous:
        print(f"L0: INFO ambiguous alignment: {listing_id}")
    print("L0: PASS" if passed else "L0: FAIL")
    return 0 if passed else 1
