# MVP demo walkthrough (recording script, ~4 minutes)

Everything shown comes from committed, hash-pinned artifacts. Regenerate
first if needed: `python3 tools/mvp_demo.py && python3 tools/mvp_viewer.py`,
then open `runs/mvp_v1/viewer.html`. Numbers quoted below are the frozen
values from `runs/mvp_v0/aggregate.json`; nothing is computed live.

## Beat 1 — the scene is real (0:00–0:30)

Open the viewer (office_0 loads first, raw RGB). Orbit and zoom.

> "This is a real captured scene — the raw Replica office mesh, every
> vertex, no boxes, no annotations. The system's job: answer structural
> questions about it, and tell us honestly when it can't."

## Beat 2 — what the machine sees (0:30–1:00)

Toggle **oracle instances**, then **C1 instances**.

> "Left of my toggle: the ground-truth objects. Right: what a learned
> 3D segmenter actually extracted from the raw mesh — the frozen Mask3D
> reference. Grey is unassigned. You can already see objects merging
> and disappearing. Every claim downstream inherits this."

## Beat 3 — click-to-inspect provenance (1:00–1:30)

Click the sofa, then the purple floor object.

> "Every point is attributable: Replica object id, class, whether C1
> recovered it and at what IoU, and what the zero-shot labeler called
> it. And this purple object? 'No oracle object at this vertex' — the
> dataset itself has no idea what it is, and the viewer says so instead
> of inventing an answer."

## Beat 4 — the ladder isolates failure (1:30–2:20)

Question: **"what is on the table?"** (office_0). Walk A → B → C1.

> "Same graph, same reasoner, three inputs. A — perfect oracle boxes:
> it answers, precision 1.0. B — boxes re-derived from the mesh: still
> answers. C1 — learned instances: the answer collapses to nothing, and
> the scoreboard says so: uid-recall 0.00 on this scene. Because every
> stage is isolated, we KNOW this loss is instance extraction — not the
> graph, not the reasoner, not the labels."

## Beat 5 — room_2, the label experiment (2:20–3:20)

Switch scene to room_2. Question **Q07 "what is on the shelf?"**,
variant **C1**, then **C2**.

> "Room 2, and now the last ladder step: replace oracle labels with
> learned ones on exactly the same instances. Under C1 the shelf is the
> blue anchor and the system finds items on it. Switch to C2 — watch
> the blue anchor vanish: the zero-shot labeler called the shelf a
> 'vent', and that single label error erased every support answer in
> the room. uid-recall 0.245 → 0.204; semantic citation — right object,
> right name — drops to 0.62. One number in a table; in 3D you can see
> exactly which object did it."

## Beat 6 — honesty as a feature (3:20–4:00)

Pick a question where the outcome badge is **empty** or **defer**; show
the **Human key** source; scroll the disclosures.

> "The system has four honest outcomes — answer, empty, defer, unknown
> — and 'empty' explicitly does not claim the object doesn't exist.
> Every number is scored against human-verified keys that record
> reality, not the system's own output; the keys are allowed to fail
> the system, and they do. The headline of this project isn't that
> raw-scene QA is solved — it's that every failure is measured, isolated
> to a stage, and visible. Including the negative results: two backends,
> a selection-repair rule family, and a label pipeline, all run under
> predeclared gates, all committed — pass or fail."

## Numbers cheat-sheet (frozen; for voiceover accuracy)

| scene | A (uid P/R) | B | C1 | C2 | C2 semantic citation |
|---|---|---|---|---|---|
| room_0 | 0.85 / 0.35 | 0.67 / 0.29 | not run | not run | — |
| room_1 | 0.83 / 0.29 | 0.90 / 0.26 | 0.57 / 0.11 | 0.57 / 0.11 | 0.50 |
| room_2 | 0.95 / 0.41 | 0.95 / 0.41 | 1.00 / 0.24 | 1.00 / 0.20 | 0.62 |
| office_0 | 1.00 / 0.375 | 1.00 / 0.625 | — / 0.00 | — / 0.00 | 0.31 |

Key facts: C1 entity recall@0.5 spans 0.25–0.38; Mask3D's ceiling is
proposal coverage (~32% viable), its selection near-optimal; Segment3D
and three oracle-free selection-repair rules failed predeclared gates;
C2 support-owner labels were 9/10 on room_2 but 2/7 on office_0
(generalization failure, preserved); attachment recall ~1/14 even for
variant A (representation finding).
