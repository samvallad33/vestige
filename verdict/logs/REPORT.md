# Verdict for PR #297 at e5d04a3

Commit: `e5d04a332aee768d5b122b313a28e0f97fd04a96` (`strata/fix-00a-v3-guard`).
Toolchain: rustc 1.98.1 (48a229cea 2026-09-01).
Shipped binary: `cargo build -p vestige-mcp --bins` default features (legacy-sqlite on, creation guard on).

Do not merge #297 into `v4` on this run. Checks 4 and 6 fail, the SQLite-linkage confirmation fails, and `cargo fmt --check` fails. Check 3 could not be run on the attached 10-store corpus. Lane E (check 7) fails as expected and does not block.

The attached `uploads/v3-real-stores.tar.gz` was not on this machine (no `uploads/` directory anywhere on the filesystem). Substitute stores, labeled as such in the logs: the checked-in synthetic fixture, that fixture with `walk_receipts` dropped, and a store created by the official v3.1.1 release binary (`vestige 3.1.1`, schema 38, 52 `knowledge_nodes` columns, no `walk_receipts` table) plus a WAL sibling.

| Check | Result | Evidence |
|---|---|---|
| 1. Empty data dir, start twice | PASS | `vestige-mcp` and `vestige stats` both exit 1 twice with `SQLite store creation is disabled`; find shows zero `.db`/`.sqlite` files. Second start is identical to the first (no self-created store). Log: `verdict/logs/01-mcp.txt`. |
| 2. Existing v3 store | PASS | Schema-38 v3.1.1 store refused: `v3 SQLite store at …/vestige.db (schema version 38) cannot be opened by 4.0`. sha256 `675f9fdfe0…` unchanged, mode stays 600, no `-wal`/`-shm`. Log: `verdict/logs/02-v311.txt`. |
| 3. Import every real store | BLOCKED | `uploads/v3-real-stores.tar.gz` is not on this VM (`verdict/logs/03-uploads.txt`). Substitute v3.1.1-binary store did import: 3 nodes, 2 edges, 7 FSRS events, 52 node columns in `legacy`, `semantic` and `similarity` became `derived_from` + `legacy_inferred=true`, missing `walk_receipts` did not error, db/wal/shm sha256 unchanged on a clean run, WAL-only row kept with `--accept-wal-snapshot`. |
| 4. strata-verify on good log and live store | FAIL | Good import prints migration `OK` (`checksum_ok`, `signature_ok`, `counts_match`) then falls through and exits 1 on missing `kernel.log`. Live `StrataStore` dir exits 1 the same way, and verify created a new `strata.key` and `.seg` beside `store.meta`. Logs: `04-imported-log.txt`, `04-live-store.txt`. |
| 5. Tamper and forged key | PASS | Flipped sealed byte: `StrataLog` open halts `payload blake3 mismatch (segment 0, offset 58, last_acked_seq 20)`. Replaced key and blake3(source) key: `trailer signature invalid`. Re-signed receipt: `payload blake3 mismatch`. All three `strata-verify` exits are 1. Log: `05-tamper.txt`. |
| 6. SIGKILL resume and non-empty `--to` | FAIL | SIGKILL after a 406-byte segment; rerun exits 1 `destination …/partial is not empty; a killed run must be removed by hand` (2 frames, no receipt, does not complete). Non-empty `--to` also exits 1, but `StrataLog::open` first wrote `strata.key` and a `.seg` next to `keep.txt`. Log: `06-kill.txt`. |
| 7. Lane E strata-store hardening (non-blocking) | FAIL | `head.state` deleted + damaged segment: reopen ok, truncated 468→58, `SILENT_TRUNCATE`. Missing key on unsealed log: `SILENT_REGEN`. Missing `store.meta`: `MISSING_META_ACCEPTED`. Log: `07-lane-e.txt`. |
| 8. test + clippy + fmt | FAIL | `cargo test --workspace`: 2315 passed, 0 failed, 60 ignored. Six strata crates (excluded from the workspace): 111 passed, 0 failed, 1 ignored. `cargo clippy --workspace -- -D warnings`: exit 0. `cargo clippy -p` equivalent on excluded `strata-migrate`: 2 errors. `cargo fmt --all -- --check`: 96 files. |
| SQLite not linked in default `vestige` | FAIL | `ldd` shows no `libsqlite3.so`. `cargo tree -p vestige-mcp -i rusqlite` shows rusqlite via `vestige-core` and `strata-migrate`. `nm` contains `libsqlite3_sys` and `sqlite3_step`. Static link. Log: `09-sqlite-linkage.txt`. |

## Fix tasks

1. **strata-verify must exit 0 on a good migrated log and on a live strata-store directory.** `crates/strata-verify/src/bin/strata_verify.rs` prints the migration report and then always runs `verify_store`, which reads `kernel.log` / `gate.log`. Return when the migration report is ok. Teach verify to read a `StrataStore` directory (`log/*.seg`, `store.meta`) and do not call `StrataLog::open` on a directory that has no segments (that call creates `strata.key` and a segment).

2. **A SIGKILL during migrate-to-strata must be rerunnable to one complete receipt.** `crates/strata-migrate/src/lib.rs` treats any non-empty destination without a matching `MIGRATION_RECEIPT` as fatal (`a killed run must be removed by hand`). A rerun has to finish, with no duplicate legacy ids, and a receipt whose per-table counts match the log.

3. **Refusing a non-empty `--to` must not write.** The same function calls `StrataLog::open` before returning `DestinationNotEmpty`, and open creates `strata.key` plus `00000000-*.seg`. Detect content without opening the log writer.

4. **Stop linking SQLite into the default-features `vestige` binary.** `vestige-mcp` default features include `legacy-sqlite` (rusqlite bundled), and `strata-migrate` depends on `rusqlite` with `bundled` unconditionally, so the shipped binary contains `libsqlite3`. The creation guard is on; the link is still there.

5. **Make `cargo fmt --all -- --check` pass.** 96 files differ at e5d04a3 (list in `verdict/logs/08-fmt-summary.txt`).

6. **Make excluded `strata-migrate` pass `cargo clippy -- -D warnings`.** `crates/strata-migrate/src/records.rs:65` empty line after a doc comment; `crates/strata-migrate/src/lib.rs:836` `row.get(&column)` needless borrow. `cargo clippy --workspace` does not fail on this because the crate is in the workspace `exclude` list.

Lane E, separate from #297:

7. **Refuse a damaged segment when `head.state` is missing.** `StrataLog::open` treats a missing watermark as 0 and truncates the active segment (observed 468→58 bytes, 0 frames kept).

8. **Do not regenerate `strata.key` for an existing unsealed log.** `load_or_create_key` writes a new 32-byte key when the file is absent, and open then succeeds.

9. **Error when `store.meta` is missing after a checkpoint.** `StrataStore::verify_checkpoint_chain` treats `(checkpoints, no meta)` as no anchor and opens successfully.

## Runtime

- Default-features `vestige` / `vestige-mcp` build: 4m 15s, finished 2026-09-29T13:03Z.
- Live script: 2026-09-29T13:04:46Z–13:04:48Z (2s; processes exited immediately).
- `cargo test --workspace`: ended 2026-09-29T13:11:28Z, exit 0.
- Strata crate tests: ended 2026-09-29T13:11:19Z, all six exit 0.
- `cargo clippy --workspace`: 29s, exit 0. Strata-crate clippy ended 2026-09-29T13:13:24Z (`strata-migrate` exit 101).
- Gate time from the start of the default build through clippy: about 15 minutes.
