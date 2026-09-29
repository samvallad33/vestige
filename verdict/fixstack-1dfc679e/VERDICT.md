# Verdict: fix stack at 1dfc679e

Date: 2026-09-29. Rust 1.98.1. Release builds used `CARGO_PROFILE_RELEASE_LTO=false`, `CODEGEN_UNITS=8`, `OPT_LEVEL=2`, `STRIP=none` (panic=abort kept). These are not the Cargo.toml LTO / opt-level=z binaries. Feature set is what was measured.

#297 is mergeable only if original checks 1–8 all pass. They do not. STACK-A passes 1–6 and fails 7 and 8. STACK-B passes 1–7 and fails 8.

## SHAs

| Piece | Commit | Tree |
|---|---|---|
| Rebuilt prefix 0a+#308+#309+#311 | `1567046c4cac6020f7dd9144e7c50ec562f10dfa` | `af94967b4ecf63200e9adebde4eb552a4eefddd6` |
| Claimed prefix `b304bfd1` | not an object in this clone | |
| STACK-A (#313 head) | `1dfc679e944dfb89d58ae21de5fa47e649a08dd5` | `1c44b19e78ce3f9256631a3291105d308adf932d` |
| STACK-B (A + #305 `00c982b1`) | `f0c01028e81355dc27c77e03d86b4e5bab2d3d66` | `f4eaf968624ff199e2e72dbb99e5139fe0436b54` |

`git merge-tree --write-tree 1567046c 1dfc679e` exited 0 and printed `1c44b19e`, which is `1dfc679e^{tree}`. The merge is clean. STACK-A content is exactly `1dfc679e`. The prefix commit SHA is not `b304bfd1`; merge commits are not reproducible across committer timestamps. `git merge-tree --write-tree 1dfc679e 00c982b1` exited 0 and printed `f4eaf968`, matching the STACK-B commit. No conflict was resolved by hand. Log: `logs/shas.txt`.

## Checks

| Check | STACK-A | STACK-B |
|---|---|---|
| 1. Empty dir, twice, CLI and MCP | PASS. Both starts exit 0. CLI files: `log/*.seg`, `log/strata.key`. MCP files: `actor.key`, `log/*.seg`, `log/strata.key`. No `.sqlite`/`.db`. | PASS. Same exits and file allowlist. |
| 2. v3 store refused, bytes unchanged | PASS. CLI and `vestige-mcp` exit 1. Message below. Combined sha unchanged `81c261d5…`. | PASS. Same exit, same message, same sha. |
| 3. Import all 10 stores | PASS. 10/10 exit 0, compare `problems: []`, source sha unchanged. Non-vocab edges are `derived_from` + `legacy_inferred`. | PASS. Same 10/10. |
| 4. strata-verify on import and live store | PASS. Standalone and feature `vestige strata-verify` exit 0, `ok: true`, files unchanged. Live store: 65 frames, 2 segments, exit 0. | PASS. Same. |
| 5. Byte flip and forged key | PASS. Flip: payload blake3 mismatch segment 0 offset 10750, verify exit 1. Derived key and random key: trailer signature invalid offset 23112, verify exit 1. | PASS. Same offsets and exits. |
| 6. SIGKILL mid-import; refuse non-empty `--to` | PASS. SIGKILL in staging, code -9, size 4491, dest absent. Rerun exit 0, 10 unique nodes, 1 receipt. `note.txt` dest exit 1, bytes unchanged, no `strata.key`. Foreign store exit 1. Same source idempotent exit 0. | PASS. Same. |
| 7. Lane E open guards | FAIL. Missing key regenerated. Missing `store.meta` opened (8 nodes). Unsealed+deleted head truncated. Nometa truncated 12788→6257 and opened 9 of 20 nodes. Sealed+deleted head refused and was not truncated. | PASS. Missing key refused, key stayed absent. Missing `store.meta` refused. Unsealed and sealed damage refused, not truncated. Nometa stayed 12788 bytes and open failed. |
| 8. test, clippy -D warnings, fmt --check | FAIL. `cargo test --workspace` exit 101, 0 tests: `install_open_storage_hook` configured out (`E0425`) once `tests/e2e`'s default `legacy-sqlite` is unified. Clippy exit 101: `redundant_closure` in `vestige-core` `sqlite/actors.rs:592`. `cargo fmt --all -- --check` exit 1, 72 files. Excluded crates' own tests passed (strata 10, gate 14, kernel 19+22, migrate 11, store 11, verify 7; 1 doc ignored). Their clippy passed. | FAIL. Same workspace compile error, same clippy error, same 72 fmt files. Excluded tests passed (strata 12, store 12; the rest match A). |
| 9. No walk follows `legacy_inferred` | INCONCLUSIVE. The importer writes inferred v3 links as `derived_from`, the same kind as a recorded edge, so walk and purge results cannot decide this check. Counts for the six stores are below. | INCONCLUSIVE. Same edge kinds as STACK-A. |

### Check 2 message (both stacks, CLI stderr and MCP stderr)

```
Error: v3 SQLite store at <dir>/vestige.db cannot be opened by 4.0. run: vestige migrate-to-strata --from <path>  (the original is never modified)
```

MCP also logs that sentence at ERROR and prints it again. The hint contains the literal token `<path>`. Schema version is not in the text. Exit code 1. Source bytes unchanged.

### Check 9 detail

INCONCLUSIVE on both stacks. The importer writes inferred v3 links as `derived_from`. That is also a recorded edge kind, so an imported inferred link is not distinguishable from a recorded edge by kind. Walk and purge output does not change the score.

`memory_connections` has no inferred column. An inferred source link is a row whose `link_type` is outside the eight-type vocabulary (`touched`, `anchored_to`, `derived_from`, `supersedes`, `corrects`, `closed_by`, `projected_to`, `evidence_of`). Source counts were read from a copy of `vestige.db*` in a temp directory. The original store directories were not modified. Post-import kinds are from the check 3 dumps. STACK-A and STACK-B match. Full rows: `logs/check9-edge-counts.json`.

| Store | Source inferred links | Source `link_type` | Edges after import, by kind |
|---|---|---|---|
| backfill-v31 | 1 | `backfill_candidate` 1 | `derived_from` 1 |
| demo-v38 | 0 | none | none |
| fresh-v38-ckpt | 1 | `narrative` 1 | `derived_from` 1 |
| fresh-v38-wal | 1 | `narrative` 1 | `derived_from` 1 |
| probe-v38-ckpt | 0 | none | none |
| vnc-v36 | 2 | `semantic` 2 | `derived_from` 2 |

The decoded records also carry `legacy_inferred: true` and `legacy_link_type` equal to the source type. The written kind is still `derived_from`. Check 9 will be rerun on these same six stores once the importer fix that writes `legacy_inferred` lands.

The tools did run. On every imported log, `causal_walk` and `backfill` return `pending_strata: … entity overlap is not a recorded edge`. `memory` purge with `confirm: true` returns `pending_strata: purge_node is not admitted on the Strata log yet`. `maintain gc` (dry run and real) reports `deleted: 0`. `graph associations` is empty. `graph chain` between the two ends of a narrative or semantic edge returns `No chain found` and `steps: []`. `StrataStore` replay ignores migration `NODE`/`EDGE` frames, so the runtime graph does not contain those links. That is recorded evidence. It is not a pass.

## SQLite facts

Default shipped binaries, both stacks (`nm -a` / `strings`):

| Binary | `sqlite3_*` nm | `sqlite3_*` strings | `rusqlite` nm | `ldd` |
|---|---|---|---|---|
| `vestige` | 0 | 0 | 0 | libc, libm, libgcc_s only |
| `vestige-mcp` | 0 | 0 | 0 | same |

Residual case-insensitive `sqlite` hits are the v3 header string `SQLite format 3`, the refusal text, `LegacySqliteDisabled`, and dashboard HTML. They are not engine symbols.

`cargo tree -p vestige-mcp -i rusqlite -e features` on the default build exits 101: `rusqlite` did not match any packages. With `--features migrate-to-strata`, `rusqlite v0.40.2` feature `bundled` is reached only through `strata-migrate` → `strata-verify` / `vestige-mcp` feature `migrate-to-strata`.

Feature-build `vestige`: `nm -a` `sqlite3_*` = 302, and `sqlite3_open`/`prepare`/`exec`/`step` = 13. Feature-build `vestige-mcp`: `sqlite3_*` = 0. Standalone `strata-verify`: `sqlite3_open`/`prepare`/`step` = 0 (three `libsqlite3_sys` CGU abs names, no engine entry points). `cargo tree -p strata-verify -i rusqlite` does reach `rusqlite` via `strata-migrate`; those symbols are not in the verify binary.

The default storage path does not call SQLite. The v3 guard is a header compare. SQLite is linked only into the `vestige` CLI when built with `migrate-to-strata`, and that is the importer. It is not on the MCP storage path.

Default `vestige strata-verify <dir>` does not work. Exit 1, stderr:

```
Error: strata-verify is not linked into this binary; rebuild with --features migrate-to-strata
```

The same command on the feature CLI, and the standalone `strata-verify` binary, exit 0 on an untouched imported log and on a live store.

## Release-e2e matrix

Commit `e47796755648dfc0fd13592dc5019968ad97828e`, tests copied into the STACK-A tree so the `strata-driver` path deps are `1dfc679e`. Product bins: STACK-A default `vestige` and `vestige-mcp`, plus the migrate-feature `strata-verify`. Linux. 83 tests, 513s.

| Bucket | Count |
|---|---|
| Passed | 36 |
| Failed only because the tool answered `pending_strata` | 10 |
| Upgrade rows failed, expected until #310 | 13 |
| Real failures | 24 |

Upgrade rows that passed anyway: `upgrade_failed_import_leaves_v3_and_names_v311`, `upgrade_fresh_install_started_twice_creates_no_sqlite_and_stays_up`.

`pending_strata`-only: `cli_backfill`, `cli_forgotten_lesson`, `cli_project`, `cli_selftest`, `mcp_causal_walk`, `mcp_forgotten_lesson`, `mcp_project`, `mcp_purge`, `mcp_selftest`, `mcp_suppress`.

The 24 real failures include three lane-E rows that match check 7 on STACK-A (`tamper_missing_head_state_and_damaged_segment_refuses_without_truncate`, `tamper_missing_key_on_unsealed_log_is_not_regenerated`, `tamper_missing_store_meta_refuses`). The matrix was not run on STACK-B. The other real rows are missing CLI flags (`recall --handle`, `migrate-to-strata --rollback`), unimplemented strata backends (`upsert_by_source`, `sync_portable_archive_file`, `set_protected`, `get_intentions_by_status`, `get_connector_cursor`), and MCP results that are not `handle_required` or a gate receipt. Names and one-line reasons: `logs/matrix-class.txt`.

`cli_dashboard` and `cli_serve` stayed up. The harness blocked reading stderr of a live process. Those two processes were killed from outside; both tests then recorded ok, with no sqlite files.

`engine::shipped_binary_links_no_sqlite` passed against the default STACK-A binaries.
