"""Normalize first-line path comments to inventory repository-relative paths."""

import argparse
from pathlib import Path

from companion_tooling.inventory import load_inventory, local_path
from companion_tooling.slice import is_path_header


def normalize_headers(inventory: Path, repo: Path) -> list[str]:
    rows = load_inventory(inventory)
    changes: list[tuple[str, Path, bytes]] = []
    for relative in dict.fromkeys(row.full_path for row in rows):
        path = local_path(repo, relative)
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines(keepends=True)
        if not lines or not is_path_header(lines[0]):
            continue
        first = lines[0].rstrip("\r\n")
        replacement = f"# {relative}" + lines[0][len(first) :]
        updated = (replacement + "".join(lines[1:])).encode("utf-8")
        if updated != raw:
            changes.append((relative, path, updated))
    # Validate all inventory inputs before the first rewrite.
    for _, path, updated in changes:
        path.write_bytes(updated)
    return [relative for relative, _, _ in changes]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args(argv)
    try:
        changed = normalize_headers(args.inventory, args.repo or args.inventory.parent)
    except (ValueError, OSError, UnicodeError):
        print("normalize_headers: invalid inventory, unsafe path, or unreadable file")
        return 1
    print(f"normalize_headers: changed {len(changed)} files")
    for relative in changed:
        print(relative)
    return 0
