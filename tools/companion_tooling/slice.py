"""Match an excerpt as an ordered subsequence, counting ambiguity up to two."""

import argparse
import re
from pathlib import Path

ID = r"lst-[1-9][0-9]*\.[1-9][0-9]*"
HEADER = re.compile(rf"# excerpt of ({ID})")
ANCHOR = re.compile(rf"\{{#({ID})\}}")
CALLOUT = re.compile(r"# \([0-9]{1,2}\)$")
PATH_HEADER = re.compile(
    r"# (?P<path>[^\s]+?)(?::L?[0-9]+(?:[-:]L?[0-9]+)?)?"
)


def is_path_header(line: str) -> bool:
    match = PATH_HEADER.fullmatch(line.strip())
    if match is None:
        return False
    path = match["path"]
    return "/" in path or "\\" in path or bool(re.search(r"\w\.[A-Za-z0-9]+$", path))


def without_path_header(lines: list[str]) -> list[str]:
    return lines[1:] if lines and is_path_header(lines[0]) else lines


def is_elision(line: str) -> bool:
    return line in {"...", "# ..."} or line.startswith("# continues")


def excerpt_lines(text: str) -> tuple[str, list[str]]:
    lines = text.strip("\n").splitlines()
    anchor: str | None = None
    if lines and (fence := re.match(r"^(`{3,}|~{3,})(.*)$", lines[0])):
        marker, info = fence.groups()
        found = ANCHOR.search(info)
        anchor = found[1] if found else None
        if len(lines) < 3 or not re.fullmatch(
            re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*", lines[-1]
        ):
            raise ValueError("unclosed excerpt fence")
        lines = lines[1:-1]
    lines = without_path_header(lines)
    header = HEADER.fullmatch(lines[0].strip()) if lines else None
    if header:
        listing_id = header[1]
        if anchor is not None and anchor != listing_id:
            raise ValueError("conflicting excerpt ids")
        lines = lines[1:]
    elif anchor is not None:
        listing_id = anchor
    else:
        raise ValueError("excerpt header required")
    lines = without_path_header(lines)
    code = [CALLOUT.sub("", line.strip()).strip() for line in lines]
    code = [line for line in code if line and not is_elision(line)]
    if not code:
        raise ValueError("empty excerpt")
    return listing_id, code


def check_slice(excerpt: str, full: str) -> str:
    try:
        _, wanted = excerpt_lines(excerpt)
    except ValueError:
        return "missing"
    counts = [1] + [0] * len(wanted)
    for line in without_path_header(full.splitlines()):
        line = CALLOUT.sub("", line.strip()).strip()
        if line.startswith("# --8<--") or is_elision(line):
            continue
        for index in range(len(wanted) - 1, -1, -1):
            if line == wanted[index]:
                counts[index + 1] = min(2, counts[index + 1] + counts[index])
    return ("missing", "aligned", "ambiguous")[counts[-1]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("excerpt", type=Path)
    parser.add_argument("full_file", type=Path)
    args = parser.parse_args(argv)
    try:
        result = check_slice(args.excerpt.read_text(), args.full_file.read_text())
    except (OSError, UnicodeError):
        result = "missing"
    print(result)
    return 0 if result == "aligned" else 1
