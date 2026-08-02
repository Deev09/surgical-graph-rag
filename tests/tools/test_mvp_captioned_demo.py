"""Tests for the presentation-only MVP captioned-demo generator."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MINIMAL_VIEWER = """<!doctype html>
<div id="side"></div><div id="variantBtns"></div>
<script>
function setScene(x){}
function renderPick(x){}
setScene(DATA.scene_order[0]);
</script>
"""


def test_captioned_demo_is_deterministic_and_offline():
    from tools.mvp_captioned_demo import MARKER, build_captioned_demo

    sha = "a" * 64
    first = build_captioned_demo(
        MINIMAL_VIEWER, source_sha256=sha, autoplay_delay_ms=250)
    second = build_captioned_demo(
        MINIMAL_VIEWER, source_sha256=sha, autoplay_delay_ms=250)
    if first != second:
        raise AssertionError("captioned demo generation must be deterministic")
    if not first.startswith(MINIMAL_VIEWER) or MARKER not in first:
        raise AssertionError("accepted viewer must remain an unchanged prefix")
    if "aaaaaaaaaaaaaaaa" not in first or "setTimeout(play,250)" not in first:
        raise AssertionError("source pin or autoplay delay missing")
    if first.count("title:") != 11:
        raise AssertionError("guided demo must retain the declared 11 steps")
    for pattern in ("src=\"http", "src='http", "href=\"http", "href='http",
                    "fetch(", "XMLHttpRequest", "import("):
        if pattern in first:
            raise AssertionError(f"external/request pattern found: {pattern}")


def test_captioned_demo_rejects_invalid_inputs():
    from tools.mvp_captioned_demo import build_captioned_demo

    try:
        build_captioned_demo("<html></html>", source_sha256="b" * 64)
    except ValueError as exc:
        if "missing required contracts" not in str(exc):
            raise
    else:
        raise AssertionError("invalid viewer must be rejected")

    built = build_captioned_demo(MINIMAL_VIEWER, source_sha256="b" * 64)
    try:
        build_captioned_demo(built, source_sha256="b" * 64)
    except ValueError as exc:
        if "already contains" not in str(exc):
            raise
    else:
        raise AssertionError("double injection must be rejected")


TESTS = [
    test_captioned_demo_is_deterministic_and_offline,
    test_captioned_demo_rejects_invalid_inputs,
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
    raise SystemExit(main())
