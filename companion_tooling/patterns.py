"""Shared host-path, denylist and hostname checks."""

import re
import socket
from collections.abc import Iterator
from pathlib import Path

# Rooted application/system paths are portable; these prefixes identify a host.
ABSOLUTE_PATH = re.compile(
    r"(?<![\w:/.])/(?:home/[^/\s\"'<>]+/|Users/[^/\s\"'<>]+/|root/"
    r"|tmp/claude[^\s\"'<>]*|mnt/[^/\s\"'<>]+/)[^\s\"'<>]*"
    r"|(?i:(?<![\w:/\\])C:\\Users\\[^\s\"'<>]*)|(?i:file:/{2}[^\s\"'<>]+)"
)
PATH = re.compile(
    r"(?<![\w:/.])/(?:[\w.~+-]+/?)+|(?<!\w)[A-Za-z]:[\\/][^\s\"'<>]+"
    r"|\\\\[^\s\\]+\\[^\s]+|(?<![\w:/\\])(?:[\w.~+-]+/)+[\w.~+/-]+"
)


def denylist_terms(path: Path) -> list[str]:
    terms = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not terms:
        raise ValueError("nonempty denylist required")
    return terms


def local_paths(text: str, terms: list[str]) -> Iterator[str]:
    yield from (match[0] for match in ABSOLUTE_PATH.finditer(text))
    for match in PATH.finditer(text):
        if any(term.casefold() in match[0].casefold() for term in terms):
            yield match[0]


def host_pattern() -> re.Pattern[str]:
    hostname = socket.gethostname()
    return re.compile(
        rf"(?i)\b(?:localhost|{re.escape(hostname)}|[\w-]+\.(?:local|internal|lan))\b"
    )
