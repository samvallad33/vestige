# Per-question diff vs the combined Step 31–36 tip

Cells are the `c2_bench.py` per-question vectors.

- LME: S@1, S@5, S@10, T@1, T@5, T@10, Turn All@5
- LoCoMo: S@1, S@5, T@1, T@5, T@10
- BEAM: T@1, T@5, T@10, Turn All@5, S@1, S@5, Session All@5

## lme

- S@1: +2 -0
  - gains: 5d3d2817, d6233ab6
- S@5: +0 -0
- S@10: +0 -0
- T@1: +0 -0
- T@5: +0 -0
- T@10: +0 -0
- TA@5: +0 -0

## locomo

- S@1: +5 -0
  - gains: 2:54, 3:88, 4:110, 4:177, 8:94
- S@5: +0 -0
- T@1: +0 -0
- T@5: +0 -0
- T@10: +0 -0

## beam

- T@1: +0 -0
- T@5: +0 -0
- T@10: +0 -0
- All@5: +0 -0
- S@1: +7 -0
  - gains: 2:information_extraction:1, 7:information_extraction:0, 7:knowledge_update:1, 7:summarization:1, 8:knowledge_update:1, 10:knowledge_update:0, 12:information_extraction:1
- S@5: +0 -0
- SAll@5: +0 -0
