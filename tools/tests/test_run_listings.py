import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import run_listings


@pytest.fixture
def execution_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "chapters/ch01").mkdir(parents=True)
    (repo / "fixture.marker").touch()
    scripts = {
        "pass.txt": (
            "import sys\nfrom pathlib import Path\n"
            "assert Path('fixture.marker').is_file()\n"
            "assert Path(__file__).is_absolute()\n"
            f"assert sys.executable == {sys.executable!r}\n"
        ),
        "fail.py": (
            "import sys\n"
            "for i in range(25): print(f'error {i}', file=sys.stderr)\n"
            "sys.exit(7)\n"
        ),
        "timeout.py": (
            "import sys, time\n"
            "print('partial timeout diagnostic', file=sys.stderr, flush=True)\n"
            "time.sleep(30)\n"
        ),
        "known.py": "raise RuntimeError('known toy failure')\n",
        "stale.py": "pass\n",
        "sketch.py": "raise AssertionError('must not run sketch')\n",
        "service.py": "raise AssertionError('must not run service')\n",
    }
    inventory = ["id\tchapter\tkind\tfull_path\tclass"]
    for number, (name, source) in enumerate(scripts.items(), 1):
        classification = {6: "sketch", 7: "needs-service:toy-api"}.get(number, "runs")
        relative = f"chapters/ch01/{name}"
        (repo / relative).write_text(source, encoding="utf-8")
        inventory.append(f"lst-1.{number}\t01\texcerpt\t{relative}\t{classification}")
    (repo / "tools/inventory.tsv").write_text("\n".join(inventory) + "\n")
    (repo / "tools/known_issues.tsv").write_text(
        "id\tnote\nlst-1.4\tKnown toy failure.\nlst-1.5\tStale toy failure.\n"
    )
    monkeypatch.setattr(run_listings, "REPO_ROOT", repo)
    return repo


@pytest.mark.parametrize(
    ("listing_id", "expected_status", "expected_verdict", "expected_exit"),
    [
        ("lst-1.1", "pass", "pass", 0),
        ("lst-1.2", "fail", "unexpected-failure", 1),
        ("lst-1.3", "fail", "unexpected-failure", 1),
        ("lst-1.4", "fail", "known-issue", 0),
        ("lst-1.5", "pass", "stale-known-issue", 1),
    ],
)
def test_execution_verdicts(
    execution_repo: Path,
    capsys: pytest.CaptureFixture[str],
    listing_id: str,
    expected_status: str,
    expected_verdict: str,
    expected_exit: int,
) -> None:
    code = run_listings.main(["--only", listing_id, "--timeout", "5"])
    assert code == expected_exit
    report = json.loads((execution_repo / "build/run_listings.json").read_text())
    assert report["summary"]["total"] == 1
    row, = report["results"]
    assert row["id"] == listing_id
    assert row["status"] == expected_status
    assert row["verdict"] == expected_verdict
    assert row["known_issue"] == (listing_id in {"lst-1.4", "lst-1.5"})
    if listing_id == "lst-1.2":
        assert row["exit_code"] == 7
        assert row["stderr_tail"] == [f"error {i}" for i in range(5, 25)]
    if listing_id == "lst-1.3":
        assert row["timed_out"] is True
        assert row["exit_code"] is None
        assert row["stderr_tail"] == ["partial timeout diagnostic"]
    output = capsys.readouterr().out
    assert "ID" in output and "RESULT" in output and "VERDICT" in output
    assert listing_id in output and expected_verdict in output


def test_all_runs_continue_after_failure_and_timeout(execution_repo: Path) -> None:
    assert run_listings.main(["--timeout", "5", "--report", "build/all.json"]) == 1
    report = json.loads((execution_repo / "build/all.json").read_text())
    assert [row["id"] for row in report["results"]] == [f"lst-1.{i}" for i in range(1, 6)]
    assert report["summary"] == {
        "total": 5,
        "passed": 2,
        "failed": 3,
        "known_failures": 1,
        "unexpected_failures": 2,
        "stale_known_issues": 1,
    }


def test_only_checks_selected_issues_and_absolute_report(execution_repo: Path) -> None:
    report_path = execution_repo / "custom/report.json"
    assert run_listings.main([
        "--only", "lst-1.4", "lst-1.1", "--report", str(report_path)
    ]) == 0
    report = json.loads(report_path.read_text())
    assert [row["id"] for row in report["results"]] == ["lst-1.1", "lst-1.4"]
    assert report["timeout_seconds"] == 60
    assert report["summary"]["stale_known_issues"] == 0


@pytest.mark.parametrize("listing_id", ["lst-99.1", "lst-1.6", "lst-1.7"])
def test_only_rejects_unknown_or_non_runs(execution_repo: Path, listing_id: str) -> None:
    with pytest.raises(SystemExit) as error:
        run_listings.main(["--only", listing_id])
    assert error.value.code == 2
    assert not (execution_repo / "build/run_listings.json").exists()


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "bad"])
def test_invalid_timeout(execution_repo: Path, timeout: str) -> None:
    with pytest.raises(SystemExit) as error:
        run_listings.main(["--timeout", timeout])
    assert error.value.code == 2


def test_missing_listing_is_reported_as_failure(execution_repo: Path) -> None:
    (execution_repo / "chapters/ch01/pass.txt").unlink()
    assert run_listings.main(["--only", "lst-1.1"]) == 1
    report = json.loads((execution_repo / "build/run_listings.json").read_text())
    assert report["results"][0]["status"] == "fail"
    assert report["results"][0]["stderr_tail"]


def test_cli_uses_script_repo_from_another_cwd(execution_repo: Path) -> None:
    source_tools = Path(run_listings.__file__).parent
    shutil.copy2(source_tools / "run_listings.py", execution_repo / "tools/run_listings.py")
    shutil.copytree(source_tools / "companion_tooling", execution_repo / "tools/companion_tooling")
    completed = subprocess.run(
        [sys.executable, str(execution_repo / "tools/run_listings.py"), "--only", "lst-1.1"],
        cwd=execution_repo.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((execution_repo / "build/run_listings.json").read_text())
    assert report["summary"]["passed"] == 1
