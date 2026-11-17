"""Generate deterministic listing landing pages and the chapter index."""

import argparse
import ast
import csv
import sys
from pathlib import Path

from companion_tooling.inventory import Listing, class_words, load_inventory, local_path


def load_tsv(path: Path, header: list[str], blank_ok: bool = False) -> dict[str, list[str]]:
    """A small keyed TSV (first column is the key); the header must match exactly.
    `blank_ok` lets non-key cells be empty (chapters whose titles are not public yet)."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if next(reader, None) != header:
            raise ValueError("invalid header")
        out: dict[str, list[str]] = {}
        for line in reader:
            filled = bool(line[0].strip()) and (blank_ok or all(cell.strip() for cell in line))
            if any(bad in cell for cell in line for bad in "|\r\n"):
                raise ValueError("table-breaking character")
            if len(line) != len(header) or line[0] in out or not filled:
                raise ValueError("invalid row")
            out[line[0]] = line[1:]
    return out


PIP_NAMES = {
    "PIL": "pillow",
    "faiss": "faiss-cpu",
    "langchain_text_splitters": "langchain-text-splitters",
    "sentence_transformers": "sentence-transformers",
    "sklearn": "scikit-learn",
}


SERVICE_WORDS = {
    "http-metrics": "HTTP metrics endpoints",
    "huggingface-download": "a Hugging Face model download",
    "llm-api-key": "an LLM API key",
    "neo4j-gds": "Neo4j with Graph Data Science",
    "postgresql": "PostgreSQL",
    "postgresql-pgvector": "PostgreSQL with pgvector",
    "redis": "Redis",
    "slack-webhook": "a Slack webhook",
    "smtp-server": "an SMTP server",
    "temporal": "a Temporal server",
}


def run_command(row: Listing) -> str:
    """Only a Python file that stands alone or needs only a service gets a run line."""
    runnable = row.classification == "runs" or row.classification.startswith("needs-service:")
    return f"python {row.full_path}" if runnable and row.full_path.endswith(".py") else ""


def packages(path: Path) -> list[str]:
    """Third-party pip packages a Python listing imports (absolute imports only)."""
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    third = {n for n in names if n not in sys.stdlib_module_names and n != "__future__"}
    return sorted(PIP_NAMES.get(n, n) for n in third)


def requirement_sets(repo: Path) -> dict[str, set[str]]:
    """requirements/NAME.txt -> package names, following `-r` includes."""
    folder = repo / "requirements"
    raw = {f.name: f.read_text(encoding="utf-8").split() for f in sorted(folder.glob("*.txt"))}

    def resolve(name: str, seen: frozenset[str]) -> set[str]:
        out: set[str] = set()
        words = raw.get(name, [])
        for i, word in enumerate(words):
            if word == "-r" and i + 1 < len(words) and words[i + 1] not in seen:
                out |= resolve(words[i + 1], seen | {words[i + 1]})
            elif not word.startswith("-") and (i == 0 or words[i - 1] != "-r"):
                out.add(word.split(">")[0].split("=")[0].split("<")[0].strip().lower())
        return out

    return {name: resolve(name, frozenset({name})) for name in raw}


def needs_line(row: Listing, path: Path, sets: dict[str, set[str]]) -> str:
    pkgs = packages(path)
    parts = ["Python 3.12+"]
    if pkgs:
        cover = [n for n, s in sets.items() if {p.lower() for p in pkgs} <= s]
        best = min(cover, key=lambda n: (len(sets[n]), n)) if cover else ""
        install = f" (`pip install -r requirements/{best}`)" if best else ""
        parts.append(", ".join(f"`{p}`" for p in pkgs) + install)
    kind, _, target = row.classification.partition(":")
    if kind == "needs-service":
        parts.append(SERVICE_WORDS.get(target, target) + " (see [SERVICES](../docs/SERVICES.md))")
    elif kind == "needs-earlier":
        parts.append(f"the definitions from [{target}]({target}.md)")
    return "; ".join(parts)


def generate(
    inventory: Path,
    repo: Path,
    meta: Path | None = None,
    chapters: Path | None = None,
    issues: Path | None = None,
) -> int:
    rows = load_inventory(inventory)
    info = load_tsv(meta, ["id", "title", "what"]) if meta else {}
    names = load_tsv(chapters, ["chapter", "title"], blank_ok=True) if chapters else {}
    known = load_tsv(issues, ["id", "note"]) if issues else {}
    if set(known) - {row.id for row in rows}:
        raise ValueError("known issue for an id not in the inventory")
    if info and set(info) != {row.id for row in rows}:
        raise ValueError("listing metadata does not cover the inventory exactly")
    outputs: dict[str, str] = {}
    if info:
        table = ["# Listings", "", "| Chapter | Listing | What it is | File |"]
        table.append("| --- | --- | --- | --- |")
    else:
        table = ["# Listings", "", "| Chapter | Listing | File |", "| --- | --- | --- |"]
    sets = requirement_sets(repo)
    listed = {row.chapter for row in rows}
    for chapter in sorted(set(names) - listed, key=int):
        empty = f"| {f'{chapter} {names[chapter][0]}'.strip()} | No code listings in this chapter |"
        table.append(empty + (" |" if info else "") + " |")
    for row in rows:
        if not local_path(repo, row.full_path).is_file():
            raise ValueError("missing full file")
        named = row.chapter in names
        chapter = f"{row.chapter} {names[row.chapter][0]}".strip() if named else row.chapter
        if info:
            title, what = info[row.id]
            run = run_command(row)
            outputs[f"lst/{row.id}.md"] = (
                f"# {title} ({row.id})\n\n{what}\n\n"
                f"- **Chapter:** {chapter}\n"
                f"- **Class:** {class_words(row.classification)}\n"
                + (f"- **Known issue:** {known[row.id][0][:1].upper()}{known[row.id][0][1:]}\n"
                   if row.id in known else "")
                + f"- **Needs:** {needs_line(row, local_path(repo, row.full_path), sets)}\n"
                + (f"- **Run:** `{run}`\n" if run else "")
                + f"- **File:** [{row.full_path}](../{row.full_path})\n"
            )
            link = f"[{row.full_path}]({row.full_path})"
            table.append(f"| {chapter} | [{row.id}](lst/{row.id}.md) | {title} | {link} |")
        else:
            outputs[f"lst/{row.id}.md"] = (
                f"# {row.id}\n\nChapter {row.chapter}\n\n{class_words(row.classification)}\n\n"
                f"[{row.full_path}](../{row.full_path})\n"
            )
            link = f"[{row.full_path}]({row.full_path})"
            table.append(f"| {row.chapter} | [{row.id}](lst/{row.id}.md) | {link} |")
    head, body = table[:4], table[4:]
    body.sort(key=lambda line: int(line.split("|")[1].split()[0]))
    outputs["LISTINGS.md"] = "\n".join(head + body) + "\n"
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
    parser.add_argument("--meta", type=Path, help="TSV id, title, what: one row per listing")
    parser.add_argument("--chapters", type=Path, help="TSV chapter, title: public chapter titles")
    parser.add_argument("--issues", type=Path, help="TSV id, note: known issues shown on pages")
    args = parser.parse_args(argv)
    try:
        repo = args.repo or args.inventory.parent
        count = generate(args.inventory, repo, args.meta, args.chapters, args.issues)
    except (ValueError, OSError, UnicodeError):
        print("generate: invalid inventory, unsafe path, or missing file")
        return 1
    print(f"generated {count} landing pages and LISTINGS.md")
    return 0
