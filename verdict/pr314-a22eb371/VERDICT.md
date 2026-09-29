# PR #314 default `strata-verify` verdict

Product commit `a22eb37105c693820a323f4302d044933bb91c1c` (`fix(verify): check Strata logs without linking rusqlite`), parent `1dfc679e944dfb89d58ae21de5fa47e649a08dd5`. That parent is the STACK-A tree, so STACK-A + #314 is this commit. No merge, no conflict. Prior STACK-A / STACK-B logs under `verdict/fixstack-1dfc679e/` were not touched.

Default binary: `cargo build --locked --release -p vestige-mcp --bin vestige` at that commit, no `--features`, no profile overrides. Finished in 4m 41s, exit 0. Size 25865672. SHA256 `6ddbc21703a704b6ee97dcc343828ed8303716b82687f205bcd0c4247c9e4afa`.

Importer: same commit, `cargo build --locked --release -p vestige-mcp --bin vestige --features migrate-to-strata`. Finished in 1m 18s, exit 0. Size 27691576. SHA256 `a2eff407411a3fad5c07a12e0ce90f3bd10c63e6b3643d6d85eb90528b249a9d`. This is the binary that imported. The default binary cannot: `migrate-to-strata` exits 1 with `Error: migrate-to-strata is not linked into this binary; rebuild with --features migrate-to-strata` and writes no destination. #310 is not in this stack.

| Check | Result |
| --- | --- |
| 1. Fresh empty folder | PASS. `vestige strata-verify` exit 0. |
| 2. Migrated v3.1.1 log | PASS. Default binary exit 0. `checksum_ok`, `signature_ok`, `counts_match` all true. |
| 3. One-byte flip | PASS. Exit 1. Payload blake3 mismatch. |
| 4. No rusqlite on the default build | PASS. `cargo tree -i rusqlite` exit 101. `sqlite3_*` symbol count 0. |
| 5. Read-only | PASS. Store-dir content sha256 and mtimes identical before and after verify. |

## 1. Fresh empty folder

`vestige --data-dir <empty> stats` exit 0. Files created: `log/00000000-5a22b30730678cc3c61a8059664d9e4e.seg` (58 bytes), `log/strata.key` (32 bytes). No sqlite file.

`vestige strata-verify <empty>` exit 0. stderr empty. stdout:

```
{
  "ok": true,
  "frames_total": 0,
  "segments": 1,
  "failures": []
}
OK
```

## 2. Migrated v3.1.1 log

Source: `fresh-v38-ckpt` from the attached tarball. `schema_version` max 38, no WAL, 10 knowledge nodes. Importer command: the feature binary above, `migrate-to-strata --from <src> --to <dest>`. Import exit 0. Source file hashes unchanged. Importer stdout:

```
=== Vestige migrate-to-strata ===
Source (never modified): /tmp/verdict-fix/pr314-work/c2/src (keep this file; it is your pre-migration record)
Source BLAKE3: b2874018caca6771caf01a47e650b34d68ac9863ca48212f7bcd86f27db31cb7
Destination: /tmp/verdict-fix/pr314-work/c2/to
Migrated: 10 nodes, 1 edges, 0 fsrs events
Skipped tables (counted, not mapped): actor_endorsement_events, actor_policy_state, actor_role_weights, agent_runs, agent_traces, composition_events, composition_members, composition_outcomes, consolidation_history, embedding_profile_manifests, embedding_profile_state, embedding_profiles, erasure_generation_counter, fsrs_config, insights, intentions, memory_access_log, memory_prs, memory_receipts, merge_operations, merge_plans, retention_snapshots, retrieval_replay_capsules, retrieval_replay_items, schema_version, suppression_operations, synaptic_tags
MIGRATION_RECEIPT: f1b14fda472361fc7ad7262674db47c79397959668c56c650cd3726f01738612 (signature verified: true)
Replay verification: true
```

Default binary `vestige strata-verify <dest>` exit 0. stderr empty. stdout:

```
{
  "frames_total": 18,
  "checksum_ok": true,
  "signature_ok": true,
  "counts_match": true,
  "failures": [],
  "ok": true
}
OK
```

## 3. One-byte flip

Copy of the migrated log. One byte in `00000000-6a7c05599f83e6d19cabcc18ebd5e96a.seg` flipped at file offset 11608 (`0x64` -> `0x9b`). Default binary `vestige strata-verify <copy>` exit 1.

stderr:

```
  segment 0: damaged at offset 10750: payload blake3 mismatch
```

stdout:

```
{
  "failures": [
    "segment 0: damaged at offset 10750: payload blake3 mismatch"
  ],
  "ok": false
}
FAILED
```

## 4. No rusqlite

From the a22eb371 tree, default features, no `--features`:

```
cargo tree -p vestige-mcp -i rusqlite -e features
```

```
error: package ID specification `rusqlite` did not match any packages
```

Exit 101.

On the default binary (`nm -a`, `strings -a`):

| Probe | Count |
| --- | --- |
| `sqlite3_*` symbols | 0 |
| `sqlite3_open` / `prepare` / `exec` / `step` | 0 |
| `strings` `sqlite3_*` | 0 |
| `strings` `rusqlite` | 0 |
| `strata-verify is not linked` | 0 |

`ldd`: linux-vdso, libgcc_s, libm, libc, ld-linux. No libsqlite3.

## 5. Read-only

Fresh store, per-file sha256 before and after `strata-verify`:

```
82ef3bbc96efbfa10343e077318be263e2e145aba57ad9a68074112d4b4bb3ce  log/00000000-5a22b30730678cc3c61a8059664d9e4e.seg
6168c1288876d042332c246f32e8718358d71f1bfd15ed65d4a89fc1a28376a8  log/strata.key
```

Identical.

Migrated log directory content digest (sha256 of the sorted per-file sha256 listing) before and after verify: `deb8e550e3f36ede7f50a46c6e20ae0b3d6bf1018c9b91d9f27fdcb492b6c617`. Content listing identical. mtime listing identical.

Logs: `verdict/pr314-a22eb371/logs/`.
