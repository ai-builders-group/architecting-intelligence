# Maintaining the companion code

## What is checked

The pointer check confirms that all 148 pointers resolve to a file and a landing page. Every Python file in `chapters/` compiles with Python 3.12. The listing pages are generated from `tools/inventory.tsv` and `tools/listings_meta.tsv`.

Before release, the files were also checked for syntax, and every excerpt shown in the book was checked against its full listing: the excerpt's code lines appear in the file, in order.

These checks confirm that the files are present and well formed. They do not exercise the code against live services. CI runs L0 without excerpt inputs; the pre-release excerpt check is separate.

## Executing standalone listings

The `run-listings` CI job uses Python 3.12, installs `requirements/all.txt` (with
CPU-only PyTorch), and executes every row with `class` equal to `runs` in
`tools/inventory.tsv`, including Python code stored in `.txt` files. Listings execute as separate Python processes from the repository root, with a
60-second timeout and no live services required.

Run `python tools/run_listings.py` locally after installing the same dependencies.
Use `--only lst-1.3 lst-1.8`, `--timeout 30`, or `--report build/custom.json` to
select IDs, change the per-listing timeout, or choose the report path. Relative
report paths resolve from the repository root. The default JSON report is
`build/run_listings.json`; it records pass/fail, exit codes, timeouts, the last
20 stderr lines, and known-issue verdicts. A summary table is also printed.

`tools/known_issues.tsv` has tab-separated `id` and `note` columns. A failing
listing in that file is reported but does not fail the job; an unlisted failure
or a listed listing that now passes fails the job. Remove stale entries after
confirming the fix. Only executed `runs` rows are checked for stale entries,
including when `--only` selects a subset. Do not exempt new failures without
investigating them.

## Files and generated pages

`tools/companion_tooling/` holds the offline tooling, and `tools/tests/` holds its tests and fixtures. `tools/inventory.tsv` records listing paths and classes. `tools/pointers.txt` records the book's pointers. `tools/listings_meta.tsv` supplies the titles and descriptions for listing pages and the index.

`tools/chapters.tsv` holds blank titles until chapter titles are public. Use chapter numbers in public text for now.

From the repository root, regenerate the listing pages and `LISTINGS.md` with:

```sh
PYTHONPATH=tools python -m companion_tooling generate --inventory tools/inventory.tsv --repo . --meta tools/listings_meta.tsv --chapters tools/chapters.tsv --issues tools/known_issues.tsv
```

`tools/known_issues.tsv` lists listings with a known problem; the note appears on that listing's page.

The `# --8<--` comment lines in listing files are excerpt markers used by the excerpt check. They are harmless Python comments and are kept on purpose.

## Local checks

Use Python 3.12. These steps mirror `.github/workflows/ci.yml`; run them from the repository root:

```sh
python -m pip install ruff==0.17.3 mypy==1.19.0
export COMPANION_ROOT='.'
export PYTHONPATH=tools
python -m companion_tooling l0 --inventory "$COMPANION_ROOT/tools/inventory.tsv" --repo "$COMPANION_ROOT"
python -m companion_tooling check_pointers --inventory "$COMPANION_ROOT/tools/inventory.tsv" --pointers "$COMPANION_ROOT/tools/pointers.txt" --repo "$COMPANION_ROOT"
(cd tools && ruff check companion_tooling && mypy companion_tooling)
python - <<'PY'
import os
import subprocess
import sys
from pathlib import Path

files = sorted(Path(os.environ["COMPANION_ROOT"]).glob("chapters/**/*.py"))
if not files:
    raise SystemExit("No chapter Python files to check")
subprocess.run([sys.executable, "-m", "py_compile", *map(str, files)], check=True)
PY
```

For the tooling tests, install `pytest` and run `python -m pytest` from `tools/`. See [TOOLING.md](TOOLING.md) for excerpt alignment rules and fixture examples.
