#!/usr/bin/env python3
"""Serial pytest runner with per-file timeout.

Runs each test file under the `tests/` directory one by one, applying a timeout
to each invocation to avoid hangs. Summarizes pass/fail/timeout results.

Environment variables honored:
- QUARK_SKIP_MODEL_COPY_FOR_TESTS=1 to skip heavy model copies in tests.

Usage:
  python tools_utilities/scripts/serial_pytest_runner.py --timeout 45
  python tools_utilities/scripts/serial_pytest_runner.py --pattern tests/resource_manager/*.py

Integration: Not simulator-integrated; repository tooling for indexing, validation, or CI.
Rationale: Executed by developers/CI to maintain repo health; not part of runtime simulator loop.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def discover_test_files(pattern: str | None) -> List[Path]:
    if pattern:
        paths = [Path(p) for p in shlex.split(pattern)]
        files: List[Path] = []
        for p in paths:
            if p.is_dir():
                files.extend(sorted(p.rglob("test_*.py")))
            elif p.is_file() and p.suffix == ".py":
                files.append(p)
        return files
    root = Path("tests")
    return sorted(root.rglob("test_*.py"))


def run_with_timeout(test_file: Path, timeout: int) -> Tuple[str, int, str]:
    env = os.environ.copy()
    env.setdefault("QUARK_SKIP_MODEL_COPY_FOR_TESTS", "1")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        str(test_file),
        "--maxfail=1",
        "--disable-warnings",
    ]
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Pytest exit code 5 means "No tests collected" – treat as non-fatal/skipped
        if proc.returncode in (0, 5):
            return ("ok", proc.returncode, proc.stdout + proc.stderr)
        return ("fail", proc.returncode, proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired as e:
        def _to_text(x: str | bytes | None) -> str:
            if x is None:
                return ""
            return x.decode(errors="ignore") if isinstance(x, (bytes, bytearray)) else x
        return ("timeout", -1, _to_text(e.stdout) + _to_text(e.stderr))


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial pytest runner with per-file timeout")
    parser.add_argument("--timeout", type=int, default=45, help="Timeout seconds per file (default: 45)")
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional shell-like pattern or paths to limit files (e.g., 'tests/resource_manager/*.py')",
    )
    args = parser.parse_args()

    files = discover_test_files(args.pattern)
    if not files:
        print("No test files found.")
        return 0

    print(f"Discovered {len(files)} test files. Running with timeout={args.timeout}s per file.\n")
    summary: Dict[str, int] = {"ok": 0, "fail": 0, "timeout": 0}
    failures: List[Tuple[Path, str]] = []

    for idx, tf in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] pytest {tf}")
        status, code, output = run_with_timeout(tf, args.timeout)
        summary[status] = summary.get(status, 0) + 1
        if status != "ok":
            failures.append((tf, status))
        print(f"  -> {status} (code={code})")
        sys.stdout.flush()

    print("\n=== Serial runner summary ===")
    print(f"ok={summary['ok']} fail={summary['fail']} timeout={summary['timeout']}")
    if failures:
        print("\nFailures/Timeouts:")
        for tf, st in failures:
            print(f" - {st}: {tf}")

    return 0 if summary["fail"] == 0 and summary["timeout"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

