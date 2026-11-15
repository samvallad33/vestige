# Step 37 per-question diff vs the Step 31–35 tip

Baseline `fd883f580d91cc7bb6e62ac5b4d8804d4e30f7a2`, binary sha256 `5cc9d0f9e0e43247e64cf46c17e0a230b855686ab28418ff949c4d4b64ca3b69`.
After binary sha256 `1d7363188e0576173a5d6af36c4af5aa7db7b9863b019d40467979cd4e717b9f`.

Vectors are the `c2_bench.py` bits.

- LME: `[S@1, S@5, S@10, T@1, T@5, T@10, TAll@5]`
- LoCoMo: `[S@1, S@5, T@1, T@5, T@10]`
- BEAM: `[T@1, T@5, T@10, All@5, S@1, S@5, SAll@5]`

LoCoMo: 0 questions differ (1536 compared).
BEAM: 0 questions differ (355 compared).
LME: 1 question differs, 0 losses (470 compared).

| question | before | after |
|---|---|---|
| LME `gpt4_fa19884d` | `[1,1,1,0,1,1,1]` | `[1,1,1,1,1,1,1]` |

Slot 1 before is `answer_ff201787_2#8`. Slot 1 after is `answer_ff201787_2#10` (gold). The following user line was already at rank 3; top-5 membership is unchanged. Session bits stay 1.
