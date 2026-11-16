# Handles

A handle is exact. Recall accepts a full memory id, a unique id prefix of 8 or more characters, or an exact tag. The session capsule wraps those handles in a `vst:` envelope so a summary can quote them without cutting one.

The envelope is the characters `vst:` plus one of:

| Form | Meaning |
|---|---|
| `vst:mem-` + 16 hex digits | A memory id |
| `vst:eff-` + 16 hex digits | An effect receipt. `receipt(action="get", receipt_id="eff-…")` resolves it. The `vst:` prefix is not part of the receipt id. |
| `vst:anchor:` + the anchor id | A code anchor |
| `vst:intention:` + the intention id | An intention |
| `vst:head:` + 16 hex seq + `:` + 64 hex frame hash + `:` + 64 hex segment hash or `unsealed` + `:` + 64 hex checkpoint hash or `none` | The log head at a capsule or a connect |
| `vst:rulebook:1` | Default Rulebook version 1 |

A token counts only between backticks and only when it matches one of those forms byte for byte. A shortened or paraphrased token resolves to nothing.
