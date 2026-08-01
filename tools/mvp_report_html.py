"""MVP-v0 HTML renderer — deterministic JSON -> one self-contained file.

  python3 tools/mvp_report_html.py [--out-dir runs/mvp_v0]

Pure rendering: every number comes from the mvp_demo JSON reports (a
number appearing here but not there is a bug, per the spec). Review PNGs
(question sheets + UID index) are embedded as data URIs so the file makes
zero external requests. Output is deterministic given the same inputs.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CSS = """
body{font:15px/1.5 -apple-system,'Segoe UI',sans-serif;color:#1a1a1a;
     background:#fafafa;margin:0;padding:2rem;max-width:1100px;
     margin-inline:auto}
h1{font-size:1.6rem} h2{font-size:1.25rem;margin-top:2.2rem}
h3{font-size:1.05rem}
table{border-collapse:collapse;width:100%;background:#fff;font-size:.92rem}
th,td{border:1px solid #ddd;padding:.35rem .6rem;text-align:left}
th{background:#f0f0f0}
.hit{color:#1a7a3a;font-weight:600}
.violation{color:#b3261e;font-weight:700;text-decoration:underline}
.extra{color:#b3261e}
.missed{color:#8a6d00}
.anon{border:1px dashed #999;padding:0 .25rem;color:#555;font-style:italic}
.tag{display:inline-block;background:#eee;border-radius:4px;
     padding:0 .45rem;font-size:.8rem;margin-left:.4rem}
details{background:#fff;border:1px solid #ddd;border-radius:6px;
        margin:.5rem 0;padding:.4rem .8rem}
summary{cursor:pointer;font-weight:600}
.variant{margin:.6rem 0 .2rem;font-weight:600}
.disclosure{background:#fff8e6;border:1px solid #e6d9a8;border-radius:6px;
            padding:.8rem 1rem;margin:.6rem 0}
pre{background:#f4f4f4;border:1px solid #ddd;border-radius:6px;
    padding:.8rem;overflow-x:auto;font-size:.8rem}
img{max-width:100%;border:1px solid #ddd;border-radius:6px}
.small{color:#666;font-size:.85rem}
"""

DISCLOSURES = [
    ("Answer keys measure reality, not the system",
     "Keys are human-verified (answer_key_type: human_verified); a variant "
     "failing a question is the evaluation working. Key semantics rulings: "
     "'on furniture' means physical resting/gravity contact even where the "
     "imported boxes disagree; doors and structural pillars are not 'on the "
     "floor'."),
    ("'empty' does not mean 'no such object'",
     "Under the oracle completeness profile an empty answer means the graph "
     "holds no such relation — it is not proof of absence in the scene."),
    ("Near-wall is membership-only",
     "The 0.5 m near-wall question is deliberately non-exhaustive; it is "
     "excluded from micro precision/recall."),
    ("Attachment recall is a downstream-semantics finding",
     "Even variant A (oracle boxes) recovers ~1/14 of the human attached-to "
     "answers on room_2: the keys ruled windows/blinds attached, which the "
     "2 cm ATTACHED_TO contact semantics cannot see. No segmentation or "
     "selection change can fix this; it is a representation finding."),
    ("Some support answers are representationally unreachable",
     "On room_2, 15 of 20 human support answers (mostly shelf contents) are "
     "missed even by variant A with perfect boxes — whole-object AABBs and "
     "the support-class allowlist cap them. C1 reaching A's own support "
     "level means segmentation is not the binding constraint there."),
    ("Review sheets are aids, not physical ground truth",
     "Question and UID PNGs project semantic boxes and can overlap or hide "
     "objects (office_0's room-sized rug is the clearest example). Physical "
     "human judgments are made from the raw RGB 3D mesh; obj_N is the "
     "Replica face object_id used only to key that judgment."),
    ("C1 isolation",
     "C1 labels and structural surfaces are INJECTED from the oracle via "
     "exact vertex correspondence; only instance boundaries are learned. C1 "
     "citations that could not be matched to an oracle object appear as "
     "unlabeled segments. C1 is CLOSED with Mask3D @0.2 as the frozen "
     "reference (docs/c1_closeout.md)."),
]


def _esc(s) -> str:
    return html.escape(str(s))


def _img_tag(path: Path, alt: str) -> str:
    if not path.exists():
        return f"<p class='small'>({_esc(alt)}: image not available)</p>"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f"<details><summary>{_esc(alt)}</summary>"
            f"<img alt='{_esc(alt)}' src='data:image/png;base64,{b64}'>"
            f"</details>")


def _fmt(x):
    return "—" if x is None else x


def _question_card(qid: str, per_variant: dict, key_q: dict) -> str:
    q = key_q["question"]
    labels = key_q.get("candidate_labels", {})
    human = ", ".join(f"{_esc(u)} <span class='small'>{_esc(labels.get(u, ''))}</span>"
                      for u in key_q["expected_must_contain"]) or "<i>empty</i>"
    parts = [f"<details><summary>{_esc(qid)} — {_esc(q)}"
             f"<span class='tag'>{_esc(key_q['expected_outcome'])}"
             f"{', exhaustive' if key_q.get('exhaustive') else ''}</span>"
             f"</summary>",
             f"<p><b>Human answer:</b> {human}</p>"]
    if key_q["expected_must_not_contain"]:
        parts.append("<p><b>Must NOT contain:</b> "
                     + ", ".join(_esc(u) for u in key_q["expected_must_not_contain"])
                     + "</p>")
    for variant in ("A", "B", "C1", "C2"):
        row = per_variant.get(variant)
        if row is None:
            continue
        cites = []
        for c in row["cited"]:
            cls = c["status"]
            uid = _esc(c["uid"])
            if c.get("unlabeled_segment"):
                uid = f"<span class='anon'>{uid}</span>"
            iou = (f" <span class='small'>iou {c['matched_iou']}</span>"
                   if "matched_iou" in c else "")
            cites.append(f"<span class='{cls}'>{uid}</span> "
                         f"<span class='small'>{_esc(c['label'])}</span>{iou}")
        pr = ""
        if row["precision"] is not None or row["recall"] is not None:
            pr = (f"<span class='tag'>P {_fmt(row['precision'])} / "
                  f"R {_fmt(row['recall'])}</span>")
        parts.append(
            f"<div class='variant'>{variant}"
            f"<span class='tag'>{_esc(row['actual_outcome'])}</span>{pr}</div>"
            f"<div>{'; '.join(cites) if cites else '<i>no citations</i>'}</div>")
        if row["missed"]:
            parts.append("<div class='missed'>missed: "
                         + ", ".join(_esc(u) for u in row["missed"]) + "</div>")
        parts.append(f"<div class='small'>“{_esc(row['verbalized'])}”</div>")
    parts.append("</details>")
    return "".join(parts)


def build_html(out_dir: Path) -> Path:
    agg = json.loads((out_dir / "aggregate.json").read_text())
    scene_files = sorted(out_dir.glob("*_mvp.json"))
    scenes = [json.loads(p.read_text()) for p in scene_files]

    h = [f"<style>{CSS}</style>", "<title>MVP-v0 — A/B/C1/C2 vs human keys</title>",
         "<h1>MVP-v0 — one pipeline, four input variants, human-keyed</h1>",
         "<p>A modular, queryable spatial-graph reasoner that exposes "
         "uncertainty and isolates failures under imperfect 3D instance "
         "extraction. Variant A = oracle boxes, B = mesh-derived boxes, "
         "C1 = learned instances (frozen Mask3D @0.2) with oracle labels "
         "injected; C2 replaces labels on matched instances with frozen "
         "zero-shot predictions. Same frozen graph + Router throughout; scores are "
         "against human-verified answer keys.</p>",
         f"<p class='small'>{_esc(agg['comparability'])} "
         f"Git {_esc(agg['git_commit'][:12])}; "
         f"reference check: {_esc(agg['reference_check'])}.</p>",
         "<h2>Headline</h2>",
         "<table><tr><th>scene</th><th>variant</th><th>uid micro-P</th>"
         "<th>uid micro-R</th><th>semantic citation</th>"
         "<th>support hits</th><th>edges</th>"
         "<th>entities@0.5</th></tr>"]
    for r in agg["headline"]:
        h.append(f"<tr><td>{_esc(r['scene_id'])}</td><td>{_esc(r['variant'])}</td>"
                 f"<td>{_fmt(r['micro_precision'])}</td>"
                 f"<td>{_fmt(r['micro_recall'])}</td>"
                 f"<td>{_fmt(r.get('semantic_citation'))}</td>"
                 f"<td>{_fmt(r['support_hits'])}</td>"
                 f"<td>{_fmt(r['n_graph_edges'])}</td>"
                 f"<td>{_fmt(r.get('entity_matches_at_05'))}</td></tr>")
    h.append("</table>")
    h.append("<p class='small'>uid micro-P/R score UID/structural "
             "MEMBERSHIP against the key (the key cites uids, not names). "
             "Semantic citation scores whether uid-correct citations also "
             "carry the canonical label — it is where wrong learned labels "
             "show even when membership is unchanged. room_0 has no C1/C2 "
             "row: Mask3D was never run on it and MVP-v0 spends no GPU — "
             "'not run' is not 0. C2 rows are EVALUATION-ONLY: labels come "
             "from the committed C2.0 prediction sidecars "
             "(docs/c2_matched_labels_protocol.md).</p>")

    for sc in scenes:
        sid = sc["scene_id"]
        key_qs = sc["key_questions"]
        h.append(f"<h2>{_esc(sid)}</h2>")
        h.append(f"<p class='small'>C1 status: {_esc(sc['c1_status'])}</p>")
        h.append(_img_tag(REPO_ROOT / "demo" / f"{sid}_questions.png",
                          f"{sid} question sheet"))
        h.append(_img_tag(REPO_ROOT / "demo" / f"{sid}_uid_index.png",
                          f"{sid} UID index"))
        per_q: dict[str, dict] = {}
        for v, vr in sc["variants"].items():
            for q in vr["questions"]:
                per_q.setdefault(q["question_id"], {})[v] = q
        for qid in sorted(per_q):
            h.append(_question_card(qid, per_q[qid], key_qs[qid]))

    h.append("<h2>Disclosures</h2>")
    for title, body in DISCLOSURES:
        h.append(f"<div class='disclosure'><b>{_esc(title)}.</b> "
                 f"{_esc(body)}</div>")

    h.append("<h2>Provenance appendix</h2>")
    for sc in scenes:
        h.append(f"<h3>{_esc(sc['scene_id'])}</h3>")
        h.append(f"<pre>{_esc(json.dumps(sc['provenance'], indent=1, sort_keys=True))}</pre>")

    out = out_dir / "report.html"
    out.write_text("".join(h), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "runs" / "mvp_v0")
    args = ap.parse_args(argv)
    out = build_html(args.out_dir)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
