"""Execute every runs-class listing and reject unexpected failures or stale issues."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from companion_tooling.generate import load_tsv
from companion_tooling.inventory import load_inventory, local_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_listings(
    repo: Path, only: list[str] | None = None, timeout: float = 60
) -> list[dict]:
    rows = load_inventory(repo / "tools/inventory.tsv")
    known = load_tsv(repo / "tools/known_issues.tsv", ["id", "note"])
    if set(known) - {row.id for row in rows}:
        raise ValueError("known issues contain IDs absent from the inventory")
    runnable = [row for row in rows if row.classification == "runs"]
    if only is not None:
        invalid = set(only) - {row.id for row in runnable}
        if invalid:
            raise ValueError(f"not runs-class listing IDs: {', '.join(sorted(invalid))}")
        runnable = [row for row in runnable if row.id in only]

    results = []
    for row in runnable:
        timed_out = False
        try:
            completed = subprocess.run(
                [sys.executable, str(local_path(repo, row.full_path).resolve())],
                cwd=repo,
                # Running checks must not create bytecode inside listing directories.
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            exit_code = completed.returncode
            stderr = completed.stderr
        except subprocess.TimeoutExpired as error:
            exit_code = None
            timed_out = True
            # TimeoutExpired may carry bytes even when text=True.
            stderr = error.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")

        passed = exit_code == 0 and not timed_out
        known_issue = row.id in known
        if passed:
            verdict = "stale-known-issue" if known_issue else "pass"
        else:
            verdict = "known-issue" if known_issue else "unexpected-failure"
        results.append({
            "id": row.id,
            "full_path": row.full_path,
            "status": "pass" if passed else "fail",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stderr_tail": stderr.splitlines()[-20:],
            "known_issue": known_issue,
            "verdict": verdict,
        })
    return results


def positive_timeout(value: str) -> float:
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite positive number")
    return timeout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these runs-class IDs")
    parser.add_argument("--report", type=Path, default=Path("build/run_listings.json"))
    parser.add_argument("--timeout", type=positive_timeout, default=60, metavar="N")
    args = parser.parse_args(argv)
    try:
        results = run_listings(REPO_ROOT, args.only, args.timeout)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    summary = {
        "total": len(results),
        "passed": sum(row["status"] == "pass" for row in results),
        "failed": sum(row["status"] == "fail" for row in results),
        "known_failures": sum(row["verdict"] == "known-issue" for row in results),
        "unexpected_failures": sum(row["verdict"] == "unexpected-failure" for row in results),
        "stale_known_issues": sum(row["verdict"] == "stale-known-issue" for row in results),
    }
    report = REPO_ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {"timeout_seconds": args.timeout, "summary": summary, "results": results}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{'ID':<16} {'RESULT':<6} {'EXIT':<8} VERDICT", flush=True)
    for row in results:
        exit_text = "timeout" if row["timed_out"] else str(row["exit_code"])
        print(f"{row['id']:<16} {row['status']:<6} {exit_text:<8} {row['verdict']}")
    print(
        f"{summary['total']} listings: {summary['passed']} pass, {summary['failed']} fail; "
        f"{summary['known_failures']} known, {summary['unexpected_failures']} unexpected, "
        f"{summary['stale_known_issues']} stale. Report: {args.report}"
    )
    return int(bool(summary["unexpected_failures"] or summary["stale_known_issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
