"""Phase 2 P2.01 — raw Replica input verifier.

Phase 2 reads from raw Replica data that is NOT checked into the repo:

    data/replica/room_0/habitat/info_semantic.json
    data/replica/room_0/habitat/mesh_semantic.ply

The repo's Phase 1 path uses pre-imported artifacts under
scenes/replica_room_0/ instead. Phase 2 needs the raw inputs so the
importer can emit OBBs + structural surfaces (P2.02, P2.03) without
silently reinterpreting Phase 1 outputs.

This script does two things:

  --init      : (one-time) record sha256 + sizes of the present raw
                inputs into tools/replica_inputs.lock.json. Use this
                AFTER you have placed the canonical Replica data on
                disk and confirmed it matches the dataset release you
                intend to track. The lock file is checked into the
                repo; the data itself is gitignored.

  (default)   : verify the on-disk inputs against the pinned lock file.
                Exit 0 on full match. Exit 1 on any mismatch (missing
                file, wrong size, wrong sha256).

The lock file is the canonical Phase 2 provenance record. docs/data_inventory.md
explains where the data should come from and how the lock file was created.

Per Q1: no silent canonical fallback. If verification fails, Phase 2 is
gated until the user places matching inputs on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "replica" / "room_0"
LOCK_PATH = REPO_ROOT / "tools" / "replica_inputs.lock.json"

# Canonical relative paths under data/replica/room_0/. These mirror the
# habitat-distributed Replica layout. --root may relocate the data root, but
# the relative layout underneath that root remains fixed and auditable.
EXPECTED_FILES: tuple[str, ...] = (
    "habitat/info_semantic.json",
    "habitat/mesh_semantic.ply",
)


@dataclass(frozen=True)
class FileFact:
    relpath: str
    size: int
    sha256: str


def _sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unavailable_files(root: Path, rel: tuple[str, ...]) -> list[str]:
    """Return paths that are missing or are not regular files."""
    return [r for r in rel if not (root / r).is_file()]


def _fact_for(root: Path, relpath: str) -> FileFact:
    p = root / relpath
    return FileFact(relpath=relpath, size=p.stat().st_size, sha256=_sha256_of(p))


def _show(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def init_lock(root: Path, lock_path: Path, expected: tuple[str, ...]) -> int:
    unavailable = _unavailable_files(root, expected)
    if unavailable:
        print("[--init] FAILED: missing or non-file inputs under", _show(root))
        for relpath in unavailable:
            print("   -", relpath)
        print("Place the raw Replica room_0 inputs on disk first.")
        return 1
    facts = [_fact_for(root, r) for r in expected]
    payload = {
        "schema_version": 1,
        "data_root_relative_to_repo": _show(root),
        "files": [{"relpath": f.relpath, "size": f.size, "sha256": f.sha256} for f in facts],
    }
    lock_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[--init] wrote {_show(lock_path)} with {len(facts)} entries.")
    for f in facts:
        print(f"   {f.relpath}  size={f.size}  sha256={f.sha256[:16]}...")
    return 0


def verify(root: Path, lock_path: Path) -> int:
    if not lock_path.exists():
        print(f"FAILED: lock file not found at {_show(lock_path)}.")
        print("Run with --init after placing the raw Replica inputs on disk.")
        return 1
    pinned = json.loads(lock_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    print(f"Verifying {len(pinned['files'])} pinned file(s) under {_show(root)}/ ...")
    for entry in pinned["files"]:
        relpath = entry["relpath"]
        p = root / relpath
        if not p.exists():
            failures.append(f"missing: {relpath}")
            continue
        if not p.is_file():
            failures.append(f"not a regular file: {relpath}")
            continue
        actual_size = p.stat().st_size
        if actual_size != entry["size"]:
            failures.append(
                f"size mismatch: {relpath} expected={entry['size']} actual={actual_size}"
            )
            continue
        actual_sha = _sha256_of(p)
        if actual_sha != entry["sha256"]:
            failures.append(
                f"sha256 mismatch: {relpath} "
                f"expected={entry['sha256'][:16]}... actual={actual_sha[:16]}..."
            )
            continue
        print(f"  OK  {relpath}  size={actual_size}  sha256={actual_sha[:16]}...")
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("All pinned files match.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--init",
        action="store_true",
        help="Record sha256 + sizes of currently-present inputs into the lock file.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DATA_ROOT,
        help=f"Override the raw data root (default: {DATA_ROOT.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=LOCK_PATH,
        help=f"Override the lock-file path (default: {LOCK_PATH.relative_to(REPO_ROOT)}).",
    )
    args = parser.parse_args(argv)

    if args.init:
        return init_lock(args.root, args.lock, EXPECTED_FILES)
    return verify(args.root, args.lock)


if __name__ == "__main__":
    sys.exit(main())
