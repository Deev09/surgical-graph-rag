"""Phase 1 exit gate (blocking).

Asserts the conditions that define Phase 1 success per phase0_design.md
§11.1 + the P1.12 batch instructions:

  1. P1.08 compat reproduction gate: diff against the legacy 5,414-edge
     Replica artifact is empty.
  2. P1.09 sparse density gate: logical_edges / entity_count <= 14 with
     symmetric edges and inverse pairs counted once.
  3. Schema round-trips pass (P1.02).
  4. Benchmark v0.2 tests pass (P1.11).
  5. Replica sparse reasoner smoke tests pass (subset of P1.10).
  6. No cross-stage `_internal` imports.
  7. Full test suite passes.

The bathroom wrapper is a regression smoke test (its test suite runs
inside the full sweep) but it is NOT a blocking exit condition per the
batch direction.

Exit code 0 on full pass, 1 otherwise.

Run: python tools/phase1_exit_gate.py
"""
from __future__ import annotations

import ast
import importlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.phase1_gates import (
    run_compat_reproduction_gate, run_sparse_density_gate,
)


STAGE_PACKAGES = (
    "adapters", "representations", "extractors", "graph", "reasoner",
    "eval", "benchmark", "common",
)

TEST_SUITES = (
    "tests.schema.test_round_trip",
    "tests.oracle_replica.test_oracle_pipeline",
    "tests.relations.test_directional_proximity",
    "tests.graph.test_builder",
    "tests.benchmark.test_runner_and_schema",
    "tests.eval.test_bundle_correspondence",
    "tests.gates.test_phase1_gates",
    "tests.reasoner.test_reasoner_pipeline",
    "tests.fixtures.test_bathroom_wrapper",
)


def _run_suite(module_path: str) -> tuple[bool, str]:
    """Import the test module and call its main(). Returns (passed, captured_tail).
    The full suite output is captured so we don't drown the gate log; only
    the last line ('N/N passed') is shown unless the suite fails."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            mod = importlib.import_module(module_path)
            # Reload not needed; first import runs the file.
            rc = mod.main()
    except Exception as exc:
        return False, f"exception: {exc}"
    output = buf.getvalue().strip()
    tail = output.splitlines()[-1] if output else ""
    return (rc == 0), tail


def check_no_internal_imports() -> list[str]:
    """Walk every .py file in the stage packages and flag any
    `from sibling import _name` or `from sibling._submodule import ...`
    cross-stage private import.

    Tests and tools may import private names freely. The rule applies
    to the production stage packages only.
    """
    violations: list[str] = []
    stage_set = set(STAGE_PACKAGES)
    for pkg in STAGE_PACKAGES:
        pkg_dir = REPO_ROOT / pkg
        if not pkg_dir.exists():
            continue
        for py_file in pkg_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module is None:
                        continue
                    parts = node.module.split(".")
                    sibling = parts[0]
                    # Only check imports that cross stage-package boundaries.
                    if sibling not in stage_set or sibling == pkg:
                        continue
                    # Flag _submodule paths (e.g. `from sibling._internal import ...`)
                    if any(p.startswith("_") for p in parts[1:]):
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)}: imports from private "
                            f"submodule {node.module!r}"
                        )
                    # Flag `from sibling import _name`
                    for alias in node.names:
                        if alias.name.startswith("_") and alias.name != "_":
                            violations.append(
                                f"{py_file.relative_to(REPO_ROOT)}: imports private "
                                f"name {alias.name!r} from {node.module!r}"
                            )
    return violations


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    print("=== Phase 1 Exit Gate ===\n")

    # 1. Compat gate
    compat_pass = run_compat_reproduction_gate()
    results.append((
        "P1.08 compat reproduction (diff empty)",
        compat_pass,
        f"-> scenes/replica_room_0/eval/oracle_adapter_repro_diff.json",
    ))

    # 2. Sparse density gate
    sparse_pass = run_sparse_density_gate()
    results.append((
        "P1.09 sparse density (ratio <= 14)",
        sparse_pass,
        f"-> scenes/replica_room_0/eval/sparse_density_report.json",
    ))

    # 6. Cross-stage internal-import lint (run before suites for fast feedback)
    violations = check_no_internal_imports()
    results.append((
        "No cross-stage `_internal` imports",
        not violations,
        f"{len(violations)} violation(s)" if violations else "clean",
    ))
    for v in violations:
        print(f"  VIOLATION: {v}")

    # 3-5, 7. Run every test suite (the named ones the batch calls out, plus
    # the full sweep — this list IS the full sweep).
    print("\n--- Test suites ---")
    for suite in TEST_SUITES:
        passed, tail = _run_suite(suite)
        marker = "PASS" if passed else "FAIL"
        print(f"  {marker}  {suite}: {tail}")
        results.append((f"tests: {suite}", passed, tail))

    print("\n--- Summary ---")
    all_pass = all(r[1] for r in results)
    for name, passed, detail in results:
        marker = "PASS" if passed else "FAIL"
        print(f"{marker}  {name}")
        if detail and not passed:
            print(f"      {detail}")

    print()
    if all_pass:
        print("=== PHASE 1 EXIT GATE: ALL PASSING ===")
        return 0
    print("=== PHASE 1 EXIT GATE: FAILED ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
