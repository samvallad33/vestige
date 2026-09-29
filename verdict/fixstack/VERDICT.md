# Verdict: fix stack on top of #297 (e5d04a3)

Date: 2026-09-29T15:26Z. Toolchain not used: no stack was built.

Base: `origin/strata/fix-00a-v3-guard` `e5d04a332aee768d5b122b313a28e0f97fd04a96` (PR #297 head, unchanged).

| PR | Branch | Head | Result |
|---|---|---|---|
| #308 | `cursor/strata-migrate-clippy-fmt-ada1` | `707e63aa31c300fad7e617fb010f766df763a0c0` | merged (`d7c6e424`) |
| #309 | `cursor/strata-verify-readonly-7201` | `3f52f9c5db669b46dd8f1188d62b2fb74313828c` | merged (`19c5caee`) |
| #311 | `cursor/strata-migrate-staging-7201` | `87cca7c4fd8a2384b3220438b8fcc41b33d18446` | merged (`4a564c43`) |
| #313 | `cursor/wire-strata-stdio-5bf1` | `dbb6daff47f6ab17d703764ba23abf7f294da649` | CONFLICT |
| #305 | `cursor/strata-log-open-guards-bf5f` | `00c982b19199566fb2bd0251efb939aac5d634cc` | not attempted |

STACK-A stopped at #313. One conflicting file: `crates/vestige-mcp/src/main.rs` (one region, the v3-store refusal at startup). Ours (after #308) inlines `detect_v3` and exits 1. Theirs (#313) calls `exit_if_v3(db_path)`. Merge aborted. Not resolved by hand. Log: `logs/313-conflict.txt`.

STACK-B is STACK-A + #305. STACK-A has no commit, so #305 was not merged.

Last clean prefix (0a+#308+#309+#311 only, not a stack head): `4a564c433310fdb4127c656af40bc1917c81c00d`.

No release binary was built. The 10 real stores were not imported. The release-e2e matrix was not run.

| Check | STACK-A | STACK-B |
|---|---|---|
| 1. Fresh boot, twice, no sqlite, MCP initialize on stdio | BLOCKED. No STACK-A commit. #313 conflicts in `crates/vestige-mcp/src/main.rs`. | BLOCKED. STACK-A did not complete, so #305 was not applied. |
| 2. Existing v3 store, sha256 unchanged | BLOCKED. Same conflict. No binary. | BLOCKED. Same. |
| 3. Import all 10 real stores | BLOCKED. Same conflict. Stores not opened. | BLOCKED. Same. |
| 4. strata-verify on import and live store | BLOCKED. Same conflict. | BLOCKED. Same. |
| 5. Byte-flip and forged key | BLOCKED. Same conflict. | BLOCKED. Same. |
| 6. SIGKILL mid-import; refuse non-empty `--to` | BLOCKED. Same conflict. | BLOCKED. Same. |
| 7. Lane E strata-store hardening | BLOCKED. Same conflict. #305 not merged. | BLOCKED. Same. |
| 8. `cargo test --workspace`, clippy `-D warnings`, fmt | BLOCKED. Same conflict. No stack tree to test. | BLOCKED. Same. |
| SQLite symbols and storage-path reachability | BLOCKED. No shipped binary. Symbol count, `cargo tree -i rusqlite`, and storage-path reachability were not measured. | BLOCKED. Same. |
| 9. No walk returns `legacy_inferred` | BLOCKED. No import, no tool calls. | BLOCKED. Same. |
| release-e2e matrix (`e4779675`, Linux, STACK-A binaries) | BLOCKED. No STACK-A binaries. `pending_strata` rows not listed because the matrix did not run. | n/a (matrix is STACK-A only). |

## SQLite facts

Not measured. There is no STACK-A or STACK-B binary.

## Check 9

Not run.

## Matrix

Not run. Totals: 0 passed, 0 failed, all rows blocked.

## Heads

- STACK-A: none
- STACK-B: none
