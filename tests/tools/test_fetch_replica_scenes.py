"""Phase 8 E1 tests: scene-corpus fetcher + lock round-trip (no network).

Run: python tests/tools/test_fetch_replica_scenes.py
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

from tools.fetch_replica_scenes import (
    PHASE8_SCENES, download_command, main, part_suffixes, wanted_members,
)


def _capture(callable_) -> tuple[int, str]:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = callable_()
    return rc, stdout.getvalue()


def _write_fake_scenes(root: Path) -> None:
    for scene in PHASE8_SCENES:
        habitat = root / scene / "habitat"
        habitat.mkdir(parents=True)
        (habitat / "info_semantic.json").write_text(
            f'{{"scene": "{scene}", "objects": []}}\n', encoding="utf-8"
        )
        (habitat / "mesh_semantic.ply").write_bytes(b"ply-instance-" + scene.encode())
        (root / scene / "mesh.ply").write_bytes(b"ply-raw-" + scene.encode())


def test_wanted_members_layout():
    members = wanted_members()
    if len(members) != 3 * len(PHASE8_SCENES):
        raise AssertionError(f"expected {3 * len(PHASE8_SCENES)} members, got {members}")
    for i, scene in enumerate(PHASE8_SCENES):
        expected = (
            f"{scene}/habitat/info_semantic.json",
            f"{scene}/habitat/mesh_semantic.ply",
            f"{scene}/mesh.ply",
        )
        if tuple(members[3 * i:3 * i + 3]) != expected:
            raise AssertionError(f"unexpected members for {scene}: {members[3*i:3*i+3]}")


def test_part_suffixes():
    suffixes = part_suffixes(17)
    if suffixes[:3] != ("aa", "ab", "ac") or suffixes[-1] != "aq":
        raise AssertionError(f"unexpected suffixes: {suffixes}")
    if len(set(suffixes)) != 17:
        raise AssertionError("suffixes are not unique")


def test_download_command_shape():
    cmd = download_command(data_root=Path("/tmp/x"))
    for fragment in ("curl -sL", "gunzip -c", "tar -xq -f - -C '/tmp/x'",
                     ".partaa", ".partaq",
                     "'room_1/habitat/info_semantic.json'",
                     "'room_1/habitat/mesh_semantic.ply'",
                     "'room_1/mesh.ply'"):
        if fragment not in cmd:
            raise AssertionError(f"missing {fragment!r} in: {cmd}")


def test_init_and_verify_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "replica"
        lock = Path(td) / "scenes.lock.json"
        _write_fake_scenes(root)
        rc, out = _capture(lambda: main([
            "--init", "--data-root", str(root), "--lock", str(lock)]))
        if rc != 0:
            raise AssertionError(f"--init failed: {out}")
        rc, out = _capture(lambda: main([
            "--data-root", str(root), "--lock", str(lock)]))
        if rc != 0:
            raise AssertionError(f"verify failed on unchanged inputs: {out}")


def test_verify_detects_drift():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "replica"
        lock = Path(td) / "scenes.lock.json"
        _write_fake_scenes(root)
        rc, _ = _capture(lambda: main([
            "--init", "--data-root", str(root), "--lock", str(lock)]))
        if rc != 0:
            raise AssertionError("--init failed")
        target = root / PHASE8_SCENES[0] / "habitat" / "info_semantic.json"
        target.write_text('{"tampered": true}\n', encoding="utf-8")
        rc, out = _capture(lambda: main([
            "--data-root", str(root), "--lock", str(lock)]))
        if rc == 0:
            raise AssertionError("verify should fail after content drift")
        if "mismatch" not in out:
            raise AssertionError(f"drift output should name a mismatch: {out!r}")


TESTS = [
    test_wanted_members_layout,
    test_part_suffixes,
    test_download_command_shape,
    test_init_and_verify_roundtrip,
    test_verify_detects_drift,
]


def main_runner() -> int:
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
    sys.exit(main_runner())
