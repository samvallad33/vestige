# v3 store fixtures for the v4.0.0 release gate

These directories are real v3.x SQLite stores produced by real v3 binaries
(schema 31, 36, and 38). The release matrix copies them and never writes
back into this tree.

| Directory | What it is |
| --- | --- |
| `fresh-v38-ckpt` | Schema 38 checkpoint, memories, one `narrative` link, FSRS columns on `knowledge_nodes`. |
| `fresh-v38-wal` | Same generation with a non-empty WAL sidecar. Migration must refuse it unless `--accept-wal-snapshot`. |
| `vnc-v36` | Schema 36 with `semantic` links. |
| `backfill-v31` | Schema 31 with a `backfill_candidate` link. |
| `demo-v38` | Schema 38, no links. |
| `probe-v38-ckpt` | Schema 38 checkpoint, no links. |

`narrative`, `semantic`, and `backfill_candidate` are legacy data. After
import they are `derived_from` edges with `legacy_inferred = true`. They are
not causal edges.

CI uses these files. `mcpdrv.py` is the stdio driver for regenerating a
fixture from a v3.1.1 `vestige-mcp` binary outside CI:

```sh
# Download the published v3.1.1 release asset for this OS, unpack it, then:
VESTIGE_MCP=/path/to/v3.1.1/vestige-mcp python3 fixtures/mcpdrv.py
```

Do not point that binary at this directory. Copy a fixture out first.
