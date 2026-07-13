"""C1 raw-mesh segmentation front end (Phase 8 A/B/C ladder).

See docs/mesh_pipeline_contract.md. The segmenter side of the contract:
a MeshSegmenter consumes ONLY mesh.ply and writes an immutable
SegmentationOutput bundle (dense vertex instance assignment + metadata);
segmenter.candidate turns that bundle into an ANONYMOUS EntityArtifacts
candidate (no oracle labels, no oracle surfaces). Oracle enrichment happens
in a separate evaluation-only step, never here.
"""
