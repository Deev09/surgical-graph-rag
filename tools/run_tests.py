"""Canonical test command for this repo.

  python3 tools/run_tests.py            # run every tests/**/test_*.py
  python3 tools/run_tests.py -k mask    # only files whose path contains "mask"
  python3 tools/run_tests.py --list     # list test files without running

Every test file in this repo is script-style: it runs standalone, prints
PASS/FAIL per test, and exits 0/1. This runner executes each file in its OWN
subprocess — deliberately, because `unittest discover` collides on duplicate
module names across test packages, and per-process isolation is also what the
files were written for. Dataset-guarded tests self-skip in-body (they print a
skip note and exit 0), so a machine without the Replica dataset still gets a
green run over everything runnable.

Exit code: 0 iff every executed file exited 0.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_files(keyword: str | None = None) -> list[Path]:
    files = sorted(REPO_ROOT.glob("tests/**/test_*.py"))
    if keyword:
        files = [f for f in files if keyword in str(f.relative_to(REPO_ROOT))]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-k", dest="keyword", default=None,
                    help="only run files whose repo-relative path contains this substring")
    ap.add_argument("--list", action="store_true", help="list matching files and exit")
    args = ap.parse_args()

    files = test_files(args.keyword)
    if not files:
        print(f"no test files match {args.keyword!r}")
        return 1
    if args.list:
        for f in files:
            print(f.relative_to(REPO_ROOT))
        return 0

    failures: list[Path] = []
    t0 = time.time()
    for f in files:
        rel = f.relative_to(REPO_ROOT)
        r = subprocess.run([sys.executable, str(f)], cwd=REPO_ROOT,
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"PASS {rel}")
        else:
            failures.append(rel)
            print(f"FAIL {rel}")
            sys.stdout.write(r.stdout)
            sys.stderr.write(r.stderr)

    dt = time.time() - t0
    print(f"\n{len(files) - len(failures)}/{len(files)} test files passed "
          f"in {dt:.1f}s")
    if failures:
        print("failed:")
        for rel in failures:
            print(f"  {rel}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
