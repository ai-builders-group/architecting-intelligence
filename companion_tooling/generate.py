"""Generate deterministic listing landing pages and the chapter index."""

import argparse
from pathlib import Path

from companion_tooling.inventory import class_words, load_inventory, local_path


def generate(inventory: Path, repo: Path) -> int:
    rows = load_inventory(inventory)
    outputs: dict[str, str] = {}
    table = ["# Listings", "", "| Chapter | Listing | File |", "| --- | --- | --- |"]
    for row in rows:
        if not local_path(repo, row.full_path).is_file():
            raise ValueError("missing full file")
        outputs[f"lst/{row.id}.md"] = (
            f"# {row.id}\n\nChapter {row.chapter}\n\n{class_words(row.classification)}\n\n"
            f"[{row.full_path}](../{row.full_path})\n"
        )
        table.append(
            f"| {row.chapter} | [{row.id}](lst/{row.id}.md) | [{row.full_path}]({row.full_path}) |"
        )
    outputs["LISTINGS.md"] = "\n".join(table) + "\n"
    # Resolve every output before writing, including existing symlink targets.
    paths = [(local_path(repo, name), text) for name, text in outputs.items()]
    for path, text in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != text.encode("utf-8"):
            path.write_text(text, encoding="utf-8")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args(argv)
    try:
        count = generate(args.inventory, args.repo or args.inventory.parent)
    except (ValueError, OSError, UnicodeError):
        print("generate: invalid inventory, unsafe path, or missing file")
        return 1
    print(f"generated {count} landing pages and LISTINGS.md")
    return 0
