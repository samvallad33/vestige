# Per-question diff, Step 39 vs combined tip `f8570b9a`

Bit order: LoCoMo `[S@1, S@5, T@1, T@5, T@10]`. LME `[S@1, S@5, S@10, T@1, T@5, T@10, All@5]`. BEAM `[T@1, T@5, T@10, All@5, S@1, S@5, SAll@5]`.

## Gains

| Bench | Question | Before | After | Flip |
| --- | --- | --- | --- | --- |
| LoCoMo | `6:15` | `[1,1,0,0,1]` | `[1,1,0,1,1]` | Turn@5 |
| LoCoMo | `8:109` | `[0,1,0,0,1]` | `[0,1,0,1,1]` | Turn@5 |

## Losses

None. LME 0/470 changed. BEAM 0/355 changed. LoCoMo 1534/1536 unchanged on the published bits.

## Published-bit-neutral reorder

LoCoMo `0:13`: `D1:11` moves from rank 9 to rank 4, `D4:10` moves the other way. Both evidence ids are then inside the top 5. Slot 0 was already the other evidence id, so Turn@1 and Turn@5 stay 1. The stored vector does not change.
