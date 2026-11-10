"""Require each printed pointer to resolve through a clickable landing-page link."""

import argparse
import re
from pathlib import Path

from companion_tooling.inventory import load_inventory, local_path
from companion_tooling.slice import ID


def has_full_link(page: str, full_path: str) -> bool:
    page = re.sub(r"<!--.*?-->", "", page, flags=re.DOTALL)
    fence: str | None = None
    for line in page.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence or line.startswith(("    ", "\t")):
            continue
        line = re.sub(r"(`+).*?\1", "", line)
        for link in re.finditer(r"(?<!!)(?<!\\)\[[^\]\n]+\]\(([^)\s]+)\)", line):
            if link[1] == f"../{full_path}":
                return True
    return False


def check_pointers(pointers: Path, inventory: Path, repo: Path) -> tuple[int, int]:
    ids = [
        line.strip() for line in pointers.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = {row.id: row for row in load_inventory(inventory)}
    passed = 0
    seen: set[str] = set()
    for listing_id in ids:
        try:
            if listing_id in seen or not re.fullmatch(ID, listing_id):
                continue
            seen.add(listing_id)
            row = rows.get(listing_id)
            if row is None or not local_path(repo, row.full_path).is_file():
                continue
            page = local_path(repo, f"lst/{listing_id}.md")
            if page.is_file() and has_full_link(page.read_text(encoding="utf-8"), row.full_path):
                passed += 1
        except (ValueError, OSError, UnicodeError):
            continue
    return passed, len(ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointers", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args(argv)
    try:
        passed, total = check_pointers(
            args.pointers, args.inventory, args.repo or args.inventory.parent
        )
    except (ValueError, OSError, UnicodeError):
        try:
            total = sum(bool(line.strip()) for line in args.pointers.read_text().splitlines())
        except (OSError, UnicodeError):
            total = 0
        print(f"0/{total}")
        return 1
    print(f"{passed}/{total}")
    return 0 if passed == total and total > 0 else 1
