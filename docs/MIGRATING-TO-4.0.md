# Migrating to Vestige 4.0 (STRATA)

4.0 runs on STRATA: a signed, append-only causal log. The v3 SQLite engine is
**a read-only migration source, not a live store**. This guide covers what
that means for an existing v3 store and how to migrate it.

Source of truth: `crates/vestige-core/src/storage/v3_guard.rs` (the guard),
`crates/strata-migrate/` (the migrator), `crates/vestige-mcp/src/bin/cli.rs`
(the `migrate-to-strata` subcommand).

## 4.0 refuses any SQLite file

Every read-write storage entry point calls the v3 guard first
(`detect_v3` → `ensure_not_v3`):

- Any file that carries the SQLite magic (`SQLite format 3\0` in its first 16
  bytes) is refused — any schema version, because 4.0 itself writes no
  SQLite. Detection reads at most the 100-byte header; no write handle is
  opened, nothing is chmod'ed.
- When the schema version can be read, it is read strictly through a
  read-only `mode=ro&immutable=1` URI — a connection that physically cannot
  write, journal, or create `-wal`/`-shm` siblings.
- The refusal (`V3StoreNeedsMigration`) carries the path, the schema version,
  and the next step:

  ```
  run: vestige migrate-to-strata --from <path>  (the original is never modified)
  ```

## 4.0 never creates a SQLite store

A guard-armed 4.0 build refuses to create a SQLite store either — a fresh
install must not write a file the very next start would refuse. Creation
attempts fail with:

```
SQLite store creation is disabled: SQLite is a read-only migration source
in 4.0. hint: the Strata backend owns new stores (run: vestige
migrate-to-strata --from <path> for existing v3 stores)
```

Opening an **existing** non-SQLite-candidate path is unaffected; the guard
only fires on the SQLite magic or on creation.

## Run the migration

```bash
vestige migrate-to-strata --from <path>            # migrate
vestige migrate-to-strata --from <path> --dry-run  # verify + counts, write nothing
```

- `<path>` may be the SQLite database file, the data directory that contains
  `vestige.db`, or a portable archive (`vestige portable-export` JSON,
  format `vestige.portable.v1`).
- `--to <dir>` chooses the destination STRATA directory. Default:
  `<data-dir>/strata` next to the old store.

## The original is never modified

- The source is read strictly read-only; the migrator hashes it (BLAKE3 over
  db → `-wal` → `-shm`) **before** any SQL read and again **after** the
  replay, and refuses to seal if a single byte changed
  (`SourceTampered`). The before/after hashes ride in the signed
  `MIGRATION_RECEIPT` as `source_blake3_before` / `source_blake3_after`
  (equal in any sealed receipt).
- The CLI says it outright: `Source (never modified): <path> (keep this
  file; it is your pre-migration record)`. **Keep the file** — it is your
  backup of everything that existed before 4.0.

### A non-empty `-wal` needs `--accept-wal-snapshot`

A non-empty `-wal` sibling means the last commits may still sit in the WAL.
The migration refuses (`WalPresent`) unless you pass `--accept-wal-snapshot`,
which copies db + sidecars to a scratch directory and migrates from that
consistent snapshot. The original files are still never touched.

## Destination policy: refuse or no-op, never extend

- **Empty destination** → proceed.
- **Non-empty destination that already holds a MIGRATION_RECEIPT for this
  exact source** (same source BLAKE3) → **idempotent no-op**: the existing
  receipt is verified and echoed (`idempotent_reuse: true`), nothing is
  written. Re-running against the same destination is safe.
- **Any other non-empty destination** → refused (`DestinationNotEmpty`). A
  killed run must be cleared by hand; it is never extended.
- If the source changes mid-run, the run aborts before sealing and the
  destination is poisoned — further runs refuse rather than trust an
  incomplete log.

## What lands in the STRATA log

| SQLite source | STRATA record |
|---|---|
| — | `GENESIS` (0x1F): source format, vestige version, schema version |
| — | `PARAMS v4-migrate/1` (0x26): schema version, source BLAKE3, envelope chain head |
| `knowledge_nodes` rows | `NODE` (0x20): dense `kernel_id` + original UUID (`legacy_id`) + content/tags + **every other source column, verbatim** |
| V40 `walk_receipts` rows | reference `NODE` frames (tagged `migrated_from_v4`) |
| `knowledge_nodes.superseded_by` | `SUPERSESSION` (0x24) lineage pointers (both ids kept) |
| `memory_connections` rows | `EDGE` (0x21): 8-type vocabulary; legacy `link_type` outside the vocabulary becomes `derived_from` with `legacy_inferred = 1` and the original type kept as `legacy_link_type` |
| `fsrs_cards` rows | `FSRS_REVIEW` (0x22): kernel `ReviewEvent` series (see below) |
| `sync_tombstones` / `deletion_tombstones` | `TOMBSTONE` (0x23) |
| (fold state) | `CHECKPOINT` (0x25): sealed kernel `Checkpoint` |
| — | `MIGRATION_RECEIPT` (kind 46): signed seal, final frame |
| every other non-empty table | counted in `skipped_tables` (reported, not mapped) |

Notes:

- **Vectors are dropped and counted.** `node_embeddings` values are never
  read — only `COUNT(*)`, reported as `dropped_vectors` in the report and the
  receipt. Embeddings do not survive into STRATA (H1: no vectors anywhere).
- **Nothing else is silently dropped.** All remaining columns ride in each
  record's `legacy` list verbatim; anything that did *not* ride along is
  named in the receipt's `dropped_columns` — the receipt never presents
  itself as a fuller copy than it is.
- **FSRS reviews are synthesized deterministically.** SQLite kept only final
  FSRS state, so migration emits `reps - lapses` rating-3 events followed by
  `lapses` rating-1 events per card. The kernel fold reproduces
  `review_count` and `lapse_count` exactly; stability/difficulty are
  recomputed by the deterministic fold and become the new truth. Every
  timestamp in the log comes from the source rows — the migration never reads
  the wall clock into hashed state.
- Legacy inferred links become `derived_from{legacy_inferred=1}`.
  <!-- TODO(comment): the spec says these edges are excluded from walks by
  default; no walk code at this HEAD reads `legacy_inferred`, so that
  exclusion is not yet verifiable. Do not rely on it until a walk lands that
  filters it. -->

## The receipt and how to check it

The final frame is the `MIGRATION_RECEIPT` (kind 46): BLAKE3 before/after of
the source, per-table counts, `dropped_vectors`, `dropped_columns`, a blake3
checksum over the body (tamper-evidence), and an **ed25519 signature**
(authorship) made by a key stored **outside the log** (`receipt-signing.key`,
mode 0600, next to the destination log — the log carries only the verifying
key). Field-by-field: [RECEIPTS.md](RECEIPTS.md).

The CLI prints the receipt digest and `signature verified`, plus
`Replay verification` (kernel replay + log tail verification must both pass,
or the command exits nonzero). To re-verify a migrated directory any time
later:

```bash
strata-verify <destination-dir>
```

The verifier re-checks the chain, the receipt checksum and signature, and
replays the frames against the receipt's per-table counts.

## After the migration

- The migrated directory is a STRATA log; open it through the 4.0 store
  surface. Ids of migrated nodes resolve both ways: the dense `kernel_id` is
  the STRATA identity, the original UUID is kept as `legacy_id`, and new
  memories mint `mem-<seq>` ids.
- Keep the original SQLite file as your pre-migration record. Deleting it is
  your choice, not a requirement — 4.0 never asks for it and never touches it.
- Re-running `vestige migrate-to-strata` against the same destination is a
  no-op (the existing receipt is echoed). To migrate the same source into a
  second destination, pass a different `--to`.
