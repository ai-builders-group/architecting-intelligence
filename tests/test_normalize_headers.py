import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from companion_tooling.normalize_headers import main, normalize_headers


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b""])
@pytest.mark.parametrize("header", [b"# elsewhere/toy.py", b"# toy.py:1-20"])
def test_rewrite_preserves_body_and_is_idempotent(
    toy_repo: Path, ending: bytes, header: bytes
) -> None:
    relative = "chapters/ch04/one.py"
    full = toy_repo / relative
    body = b"\tx = 1\r\n# elsewhere/second.py\n" if ending else b""
    full.write_bytes(header + ending + body)
    before = hashlib.sha256(body).hexdigest()
    assert normalize_headers(toy_repo / "inventory.tsv", toy_repo) == [relative]
    assert full.read_bytes() == f"# {relative}".encode() + ending + body
    observed_body = full.read_bytes().split(ending, 1)[1] if ending else b""
    assert hashlib.sha256(observed_body).hexdigest() == before
    snapshot = (full.read_bytes(), full.stat().st_mtime_ns)
    assert normalize_headers(toy_repo / "inventory.tsv", toy_repo) == []
    assert (full.read_bytes(), full.stat().st_mtime_ns) == snapshot


@pytest.mark.parametrize("content", [b"", b"# explanation\nx = 1\n", b"x = 1\n# toy.py\n"])
def test_non_headers_are_untouched(toy_repo: Path, content: bytes) -> None:
    full = toy_repo / "chapters/ch04/one.py"
    full.write_bytes(content)
    snapshot = (full.read_bytes(), full.stat().st_mtime_ns)
    assert normalize_headers(toy_repo / "inventory.tsv", toy_repo) == []
    assert (full.read_bytes(), full.stat().st_mtime_ns) == snapshot


def test_preflight_failure_does_not_rewrite(toy_repo: Path) -> None:
    full = toy_repo / "chapters/ch04/one.py"
    full.write_bytes(b"# elsewhere/toy.py\nx = 1\n")
    (toy_repo / "chapters/ch05/four.py").unlink()
    assert main(["--inventory", str(toy_repo / "inventory.tsv")]) == 1
    assert full.read_bytes() == b"# elsewhere/toy.py\nx = 1\n"


def test_cli_reports_changed_files(toy_repo: Path) -> None:
    full = toy_repo / "chapters/ch04/one.py"
    full.write_text("# elsewhere/toy.py:2-8\nx = 1\n")
    result = subprocess.run(
        [sys.executable, "-m", "companion_tooling", "normalize_headers", "--inventory",
         str(toy_repo / "inventory.tsv")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert result.stdout == "normalize_headers: changed 1 files\nchapters/ch04/one.py\n"
