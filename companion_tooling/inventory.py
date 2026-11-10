"""Strict inventory parsing and repository-relative paths shared by the tools."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from companion_tooling.slice import ID


@dataclass(frozen=True)
class Listing:
    id: str
    chapter: str
    kind: str
    full_path: str
    classification: str


def local_path(repo: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(
            part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
            for part in relative.split("/")
        )
    ):
        raise ValueError("unsafe repository path")
    target = repo / relative
    if not target.resolve().is_relative_to(repo.resolve()):
        raise ValueError("path escapes repository")
    return target


def load_inventory(path: Path) -> list[Listing]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["id", "chapter", "kind", "full_path", "class"]:
            raise ValueError("invalid inventory columns")
        rows: list[Listing] = []
        seen: set[str] = set()
        for data in reader:
            if None in data or any(value is None for value in data.values()):
                raise ValueError("invalid inventory row")
            row = Listing(
                data["id"], data["chapter"], data["kind"], data["full_path"], data["class"]
            )
            if not re.fullmatch(ID, row.id) or row.id in seen:
                raise ValueError("invalid or duplicate listing id")
            if not re.fullmatch(r"[0-9]{2}", row.chapter) or int(row.chapter) != int(
                row.id.removeprefix("lst-").split(".")[0]
            ):
                raise ValueError("chapter does not match listing id")
            if row.kind not in {"excerpt", "cut"}:
                raise ValueError("invalid listing kind")
            local_path(Path("."), row.full_path)
            if not row.full_path.startswith(f"chapters/ch{row.chapter}/"):
                raise ValueError("file is outside its chapter")
            if not re.fullmatch(
                rf"runs|sketch|needs-earlier:{ID}|needs-service:[A-Za-z0-9_-]+", row.classification
            ):
                raise ValueError("invalid class")
            rows.append(row)
            seen.add(row.id)
    if not rows:
        raise ValueError("empty inventory")
    rows.sort(key=lambda row: tuple(int(n) for n in row.id[4:].split(".")))
    earlier: set[str] = set()
    for row in rows:
        if row.classification.startswith("needs-earlier:"):
            if row.classification.split(":", 1)[1] not in earlier:
                raise ValueError("earlier dependency is absent or not earlier")
        earlier.add(row.id)
    return rows


def class_words(value: str) -> str:
    if value == "runs":
        return "Runs on its own"
    if value == "sketch":
        return "Sketch"
    kind, target = value.split(":", 1)
    return (
        f"Needs earlier listing {target}" if kind == "needs-earlier" else f"Needs service {target}"
    )
