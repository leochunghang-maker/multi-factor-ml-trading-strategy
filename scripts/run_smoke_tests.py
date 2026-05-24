"""Run a lightweight end-to-end smoke test for the local platform.

This script is intentionally conservative:
- It does not download market data.
- It does not place trades.
- It does not require API keys.
- It only checks that the project is wired together well enough for review.
"""

from __future__ import annotations

import py_compile
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    """Small record for one smoke-test check."""

    name: str
    status: str
    detail: str


class SmokeTestRunner:
    """Collects checks and prints a readable summary for beginners."""

    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def pass_(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name, "PASS", detail))
        print(f"PASS    {name}: {detail}")

    def warning(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name, "WARNING", detail))
        print(f"WARNING {name}: {detail}")

    def fail(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name, "FAIL", detail))
        print(f"FAIL    {name}: {detail}")

    def check_paths_exist(self, name: str, paths: list[Path], kind: str) -> None:
        """Check that required folders or files are present."""
        missing = [path for path in paths if not path.exists()]
        if missing:
            formatted = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
            self.fail(name, f"missing {kind}: {formatted}")
            return
        self.pass_(name, f"all required {kind} found")

    def check_no_tracked_env(self) -> None:
        """Make sure secrets are not accidentally tracked by Git."""
        git = shutil.which("git")
        if git is None:
            self.warning("Tracked .env files", "git is not available; skipped Git tracking check")
            return

        completed = subprocess.run(
            [git, "ls-files", ".env", ".env.*"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            self.warning("Tracked .env files", completed.stderr.strip() or "git ls-files failed")
            return

        tracked = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip() and line.strip() != ".env.example"
        ]
        if tracked:
            self.fail("Tracked .env files", f"secret-like files tracked by Git: {', '.join(tracked)}")
            return
        self.pass_("Tracked .env files", "no .env secret files are tracked")

    def compile_python_files(self, files: list[Path]) -> None:
        """Compile key scripts without executing their trading or data-download logic."""
        failed: list[str] = []
        for path in files:
            try:
                py_compile.compile(str(path), doraise=True)
            except Exception as exc:  # pragma: no cover - diagnostic path
                failed.append(f"{path.relative_to(PROJECT_ROOT)} ({exc})")

        if failed:
            self.fail("Python compile checks", "; ".join(failed))
            return
        self.pass_("Python compile checks", "key scripts compiled successfully")

    def check_latest_signals(self) -> None:
        """Validate the live-signal artifact only if this local run has generated outputs."""
        live_dir = PROJECT_ROOT / "results" / "live"
        signal_file = live_dir / "latest_signals.csv"

        if signal_file.exists():
            self.pass_("Latest signals artifact", "results/live/latest_signals.csv exists")
            return
        if live_dir.exists():
            self.warning(
                "Latest signals artifact",
                "results/live exists, but latest_signals.csv has not been generated",
            )
            return
        self.warning(
            "Latest signals artifact",
            "no local live-signal output found; run signal generation when market data is available",
        )

    def run_pytest_if_available(self) -> None:
        """Run pytest when installed; skip cleanly when unavailable."""
        pytest = shutil.which("pytest")
        if pytest is None:
            self.warning("Pytest", "pytest is not installed; skipped automated tests")
            return

        completed = subprocess.run(
            [pytest, "-q"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (completed.stdout + completed.stderr).strip()
        if completed.returncode == 0:
            last_line = output.splitlines()[-1] if output else "pytest passed"
            self.pass_("Pytest", last_line)
            return
        self.fail("Pytest", output or f"pytest exited with code {completed.returncode}")

    def exit_code(self) -> int:
        """Return a shell-friendly exit code: any FAIL means non-zero."""
        return 1 if any(result.status == "FAIL" for result in self.results) else 0

    def print_summary(self) -> None:
        counts = {
            "PASS": sum(result.status == "PASS" for result in self.results),
            "WARNING": sum(result.status == "WARNING" for result in self.results),
            "FAIL": sum(result.status == "FAIL" for result in self.results),
        }
        print()
        print("Smoke test summary")
        print(f"PASS: {counts['PASS']}")
        print(f"WARNING: {counts['WARNING']}")
        print(f"FAIL: {counts['FAIL']}")


def main() -> int:
    runner = SmokeTestRunner()
    print("Platform smoke tests")
    print("This run uses local files only. It does not download data or place trades.")
    print()

    runner.check_paths_exist(
        "Required folders",
        [
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "scripts",
            PROJECT_ROOT / "config",
            PROJECT_ROOT / "dashboard",
            PROJECT_ROOT / "reports",
            PROJECT_ROOT / "tests",
        ],
        "folders",
    )
    runner.check_paths_exist(
        "Required config files",
        [PROJECT_ROOT / "config" / "platform_config.json"],
        "config files",
    )
    runner.check_paths_exist("README", [PROJECT_ROOT / "README.md"], "files")
    runner.check_no_tracked_env()
    runner.compile_python_files(
        [
            PROJECT_ROOT / "scripts" / "run_daily_paper_trading.py",
            PROJECT_ROOT / "src" / "live" / "generate_live_signals.py",
            PROJECT_ROOT / "src" / "generate_report.py",
            PROJECT_ROOT / "src" / "simulation" / "run_multi_day_simulation.py",
            PROJECT_ROOT / "dashboard" / "app.py",
        ]
    )
    runner.check_latest_signals()
    runner.check_paths_exist("Reports folder", [PROJECT_ROOT / "reports"], "folders")
    runner.check_paths_exist("Dashboard app", [PROJECT_ROOT / "dashboard" / "app.py"], "files")
    runner.run_pytest_if_available()
    runner.print_summary()
    return runner.exit_code()


if __name__ == "__main__":
    sys.exit(main())
