"""Tests for geometry-only query-scoped raw-proposal selection."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import SceneFrame
from extractors.base import EntityArtifact, EntityIdentity
from extractors.entity_surfaces import derive_entity_top_surfaces
from segmenter.query_expansion import (
    materialize_query_scoped_assignment,
    raw_proposal_entities,
    select_support_region_candidates,
)


FRAME = SceneFrame(
    gravity=(0.0, 0.0, -1.0),
    canonical_forward=None,
    canonical_right=None,
    units="meters",
    notes="test",
)


def _entity(uid, label, lo, hi):
    return EntityArtifact(
        identity=EntityIdentity(
            object_uid=uid,
            display_label=label,
            source_instance_ref=f"test:{uid}",
        ),
        bbox_aabb=(lo, hi),
        bbox_obb=None,
        centroid=tuple((lo[i] + hi[i]) / 2.0 for i in range(3)),
        geometry_handle=None,
    )


def test_only_contacting_local_candidate_is_selected():
    table = _entity(
        "obj_table", "table",
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
    )
    on_table = _entity(
        "obj_1", "segment_1",
        (0.2, 0.2, 1.0), (0.4, 0.4, 1.2),
    )
    beside_table = _entity(
        "obj_2", "segment_2",
        (1.2, 0.2, 1.0), (1.4, 0.4, 1.2),
    )
    anchors = derive_entity_top_surfaces([table], frame=FRAME)
    decisions = select_support_region_candidates(
        [on_table, beside_table],
        candidate_uids={"obj_1", "obj_2"},
        anchors=anchors,
        gravity=FRAME.gravity,
    )
    by_uid = {decision.candidate_uid: decision for decision in decisions}
    if not by_uid["obj_1"].selected:
        raise AssertionError(f"contacting candidate must activate: {by_uid}")
    if by_uid["obj_1"].matching_anchor_uids != ["obj_table"]:
        raise AssertionError(f"wrong anchor match: {by_uid['obj_1']}")
    if by_uid["obj_2"].selected:
        raise AssertionError(f"outside candidate must stay inactive: {by_uid}")
    if "footprint_ok" not in by_uid["obj_2"].checks[0].failed_clauses:
        raise AssertionError("outside candidate must expose footprint rejection")


def test_missing_candidate_uid_fails_loudly():
    try:
        select_support_region_candidates(
            [],
            candidate_uids={"obj_missing"},
            anchors=[],
            gravity=FRAME.gravity,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("missing candidate geometry must raise")


def test_raw_proposals_are_measured_before_composition():
    xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    masks = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=bool)
    entities = raw_proposal_entities(xyz, masks, proposal_ids={1})
    if len(entities) != 1:
        raise AssertionError(f"expected one raw entity: {entities}")
    entity = entities[0]
    if entity.identity.object_uid != "obj_1":
        raise AssertionError(f"raw proposal id must be stable: {entity}")
    if entity.bbox_aabb != ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)):
        raise AssertionError(f"wrong raw proposal bounds: {entity.bbox_aabb}")


def test_query_local_recomposition_can_reclaim_merge_but_protects_anchor():
    hard = np.array([0, 0, 0, 1, 1, 1, 1, -1, -1], dtype=np.int64)
    masks = np.zeros((3, 9), dtype=bool)
    masks[0, 0:3] = True           # protected support anchor
    masks[1, 3:7] = True           # accepted merged segment
    masks[2, [0, 4, 5, 7, 8]] = True  # provisional; overlaps anchor + merge
    scores = np.array([0.9, 0.8, 0.1])
    out = materialize_query_scoped_assignment(
        hard,
        masks,
        scores,
        selected_ids={2},
        protected_ids={0},
        min_vertices=3,
    )
    if out.tolist() != [0, 0, 0, -1, 2, 2, -1, 2, 2]:
        raise AssertionError(
            "provisional mask must reclaim non-anchor merge vertices, keep "
            f"the anchor, and drop the too-small merge leftover: {out}"
        )


TESTS = [
    test_only_contacting_local_candidate_is_selected,
    test_missing_candidate_uid_fails_loudly,
    test_raw_proposals_are_measured_before_composition,
    test_query_local_recomposition_can_reclaim_merge_but_protects_anchor,
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
