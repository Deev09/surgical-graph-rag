"""P2.01 tests: raw Replica input verifier.

Run: python tests/tools/test_verify_replica_inputs.py
"""
from __future__ import annotations

import io
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.verify_replica_inputs import EXPECTED_FILES, init_lock, verify


def _write_inputs(root: Path) -> None:
    habitat = root / "habitat"
    habitat.mkdir(parents=True)
    (habitat / "info_semantic.json").write_text('{"objects": []}\n', encoding="utf-8")
    (habitat / "mesh_semantic.ply").write_bytes(b"ply\n")


def _capture(callable_) -> tuple[int, str]:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = callable_()
    return exit_code, stdout.getvalue()


def test_init_and_verify_with_external_root() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "external-replica"
        lock = Path(td) / "replica.lock.json"
        _write_inputs(root)
        init_exit, _ = _capture(lambda: init_lock(root, lock, EXPECTED_FILES))
        verify_exit, output = _capture(lambda: verify(root, lock))
    if init_exit != 0 or verify_exit != 0:
        raise AssertionError(f"expected init=0 and verify=0, got {init_exit=} {verify_exit=}")
    if "All pinned files match." not in output:
        raise AssertionError(f"success output missing confirmation: {output!r}")


def test_verify_detects_drift() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "replica"
        lock = Path(td) / "replica.lock.json"
        _write_inputs(root)
        init_lock(root, lock, EXPECTED_FILES)
        (root / "habitat" / "mesh_semantic.ply").write_bytes(b"ply\ndrift\n")
        exit_code, output = _capture(lambda: verify(root, lock))
    if exit_code != 1:
        raise AssertionError(f"drift should fail verification, got {exit_code}")
    if "size mismatch" not in output:
        raise AssertionError(f"drift failure should identify size mismatch: {output!r}")


def test_init_rejects_missing_inputs() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "replica"
        lock = Path(td) / "replica.lock.json"
        exit_code, output = _capture(lambda: init_lock(root, lock, EXPECTED_FILES))
    if exit_code != 1:
        raise AssertionError(f"missing inputs should fail init, got {exit_code}")
    if "missing or non-file inputs" not in output:
        raise AssertionError(f"failure output should explain unavailable inputs: {output!r}")


def test_init_rejects_directory_inputs_cleanly() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "replica"
        lock = Path(td) / "replica.lock.json"
        for relpath in EXPECTED_FILES:
            (root / relpath).mkdir(parents=True)
        exit_code, output = _capture(lambda: init_lock(root, lock, EXPECTED_FILES))
    if exit_code != 1:
        raise AssertionError(f"directory inputs should fail init, got {exit_code}")
    if "missing or non-file inputs" not in output:
        raise AssertionError(f"failure output should explain non-file inputs: {output!r}")


def test_verify_rejects_directory_input_cleanly() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "replica"
        lock = Path(td) / "replica.lock.json"
        _write_inputs(root)
        init_lock(root, lock, EXPECTED_FILES)
        mesh = root / "habitat" / "mesh_semantic.ply"
        mesh.unlink()
        mesh.mkdir()
        exit_code, output = _capture(lambda: verify(root, lock))
    if exit_code != 1:
        raise AssertionError(f"directory input should fail verification, got {exit_code}")
    if "not a regular file" not in output:
        raise AssertionError(f"failure output should identify non-file input: {output!r}")


TESTS = [
    test_init_and_verify_with_external_root,
    test_verify_detects_drift,
    test_init_rejects_missing_inputs,
    test_init_rejects_directory_inputs_cleanly,
    test_verify_rejects_directory_input_cleanly,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
