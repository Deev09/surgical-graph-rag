"""Publication figures for the results narrative (deterministic SVG).

  python3 tools/results_figures.py    # -> docs/assets/fig_*.svg

Reads the frozen MVP-v0 aggregate (runs/mvp_v0/aggregate.json) — figures
render committed numbers only, never recompute. Palette = the validated
4-slot categorical set (validated via the dataviz six-checks script:
adjacent CVD dE 9.1, normal 22.9, all PASS; aqua/yellow sit below 3:1 on
the light surface, so every bar carries a visible value label — the
relief rule). "Not run" is rendered as ABSENT with a note, never as 0;
office_0's C1/C2 zeros are true zeros and are labeled 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "assets"

SERIES = ["A", "B", "C1", "C2"]
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C1": "#1baf7a", "C2": "#eda100"}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e8e8e6"
SURFACE = "#fcfcfb"
FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"


def load_headline() -> dict:
    agg = json.loads((REPO_ROOT / "runs" / "mvp_v0" / "aggregate.json").read_text())
    out: dict[str, dict] = {}
    for r in agg["headline"]:
        out.setdefault(r["scene_id"], {})[r["variant"]] = r["micro_recall"]
    return out


def bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """Rounded top data-end, square baseline anchor."""
    if h <= r:
        r = max(h / 2, 0.5)
    return (f"M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} "
            f"Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} "
            f"{x + w:.1f},{y + r:.1f} L{x + w:.1f},{y + h:.1f} Z")


def fig_ladder(headline: dict) -> str:
    scenes = ["replica_room_0", "replica_room_1", "replica_room_2",
              "replica_office_0"]
    W, H = 760, 380
    L, R, T, B = 56, 16, 64, 46
    pw, ph = W - L - R, H - T - B
    ymax = 0.65
    bw, gap = 30, 2
    gw = 4 * bw + 3 * gap
    step = pw / len(scenes)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         f'<text x="{L}" y="24" font-size="15" font-weight="600" '
         f'fill="{INK}">UID micro-recall vs human-verified keys</text>',
         f'<text x="{L}" y="41" font-size="11.5" fill="{INK2}">Same frozen '
         f'graph + Router; only the input variant changes. C1/C2 rows are '
         f'evaluation-only ladder stages.</text>']
    # legend
    lx = W - R - 4 * 64
    for i, v in enumerate(SERIES):
        x = lx + i * 64
        s.append(f'<rect x="{x}" y="16" width="10" height="10" rx="2" '
                 f'fill="{COLOR[v]}"/>')
        s.append(f'<text x="{x + 14}" y="25" font-size="11.5" '
                 f'fill="{INK}">{v}</text>')
    # grid + y labels
    for val in (0.0, 0.2, 0.4, 0.6):
        y = T + ph - val / ymax * ph
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L - 8}" y="{y + 3.5:.1f}" font-size="10.5" '
                 f'fill="{INK2}" text-anchor="end">{val:.1f}</text>')
    for si, sid in enumerate(scenes):
        gx = L + si * step + (step - gw) / 2
        for vi, v in enumerate(SERIES):
            x = gx + vi * (bw + gap)
            val = headline[sid].get(v, "absent")
            if val is None and v in headline[sid]:
                val = headline[sid][v]
            if v not in headline[sid]:
                s.append(f'<text x="{x + bw / 2:.1f}" y="{T + ph - 6:.1f}" '
                         f'font-size="9" fill="{INK2}" text-anchor="middle" '
                         f'transform="rotate(-90 {x + bw / 2:.1f} '
                         f'{T + ph - 6:.1f})" font-style="italic">not run</text>')
                continue
            if val == 0:
                s.append(f'<text x="{x + bw / 2:.1f}" y="{T + ph - 4:.1f}" '
                         f'font-size="10" fill="{INK2}" '
                         f'text-anchor="middle">0</text>')
                continue
            h = val / ymax * ph
            y = T + ph - h
            s.append(f'<path d="{bar_path(x, y, bw, h)}" fill="{COLOR[v]}"/>')
            s.append(f'<text x="{x + bw / 2:.1f}" y="{y - 4:.1f}" '
                     f'font-size="10" fill="{INK2}" '
                     f'text-anchor="middle">{val:.2f}</text>')
        s.append(f'<text x="{gx + gw / 2:.1f}" y="{T + ph + 18}" '
                 f'font-size="11.5" fill="{INK}" text-anchor="middle">'
                 f'{sid.replace("replica_", "")}</text>')
    s.append(f'<line x1="{L}" y1="{T + ph}" x2="{W - R}" y2="{T + ph}" '
             f'stroke="{INK2}" stroke-width="1"/>')
    s.append(f'<text x="{L}" y="{H - 10}" font-size="10" fill="{INK2}">'
             f'room_0: C1/C2 not run (no GPU spent on it — absent, not zero). '
             f'office_0: C1/C2 are TRUE zeros (coverage collapse).</text>')
    s.append("</svg>")
    return "\n".join(s)


def fig_room2_attribution(headline: dict) -> str:
    vals = [("A", headline["replica_room_2"]["A"]),
            ("B", headline["replica_room_2"]["B"]),
            ("C1", headline["replica_room_2"]["C1"]),
            ("C2", headline["replica_room_2"]["C2"])]
    deltas = ["box source ±0.000", "instance extraction −0.163",
              "labels −0.041"]
    W, H = 640, 360
    L, R, T, B = 56, 16, 64, 60
    pw, ph = W - L - R, H - T - B
    ymax = 0.5
    bw = 64
    step = pw / 4

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         f'<text x="{L}" y="24" font-size="15" font-weight="600" '
         f'fill="{INK}">room_2 — recall attribution down the ladder</text>',
         f'<text x="{L}" y="41" font-size="11.5" fill="{INK2}">Each step '
         f'changes exactly one stage, so each drop is attributable.</text>']
    for val in (0.0, 0.2, 0.4):
        y = T + ph - val / ymax * ph
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L - 8}" y="{y + 3.5:.1f}" font-size="10.5" '
                 f'fill="{INK2}" text-anchor="end">{val:.1f}</text>')
    xs = []
    for i, (v, val) in enumerate(vals):
        x = L + i * step + (step - bw) / 2
        xs.append(x)
        h = val / ymax * ph
        y = T + ph - h
        s.append(f'<path d="{bar_path(x, y, bw, h)}" fill="{COLOR[v]}"/>')
        s.append(f'<text x="{x + bw / 2:.1f}" y="{y - 5:.1f}" font-size="11" '
                 f'fill="{INK}" text-anchor="middle">{val:.3f}</text>')
        s.append(f'<text x="{x + bw / 2:.1f}" y="{T + ph + 18}" '
                 f'font-size="12" fill="{INK}" '
                 f'text-anchor="middle">{v}</text>')
    for i, d in enumerate(deltas):
        cx = (xs[i] + bw + xs[i + 1]) / 2
        s.append(f'<text x="{cx:.1f}" y="{T + ph + 38}" font-size="10" '
                 f'fill="{INK2}" text-anchor="middle">→ {d}</text>')
    s.append(f'<line x1="{L}" y1="{T + ph}" x2="{W - R}" y2="{T + ph}" '
             f'stroke="{INK2}" stroke-width="1"/>')
    s.append("</svg>")
    return "\n".join(s)


def main() -> int:
    headline = load_headline()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig_ladder_recall.svg").write_text(fig_ladder(headline),
                                               encoding="utf-8")
    (OUT / "fig_room2_attribution.svg").write_text(
        fig_room2_attribution(headline), encoding="utf-8")
    print(f"wrote {OUT / 'fig_ladder_recall.svg'}")
    print(f"wrote {OUT / 'fig_room2_attribution.svg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
