# Security Policy

## Supported Versions

Security fixes ship in the latest 4.x release (4.2.0 at the time of writing). Older releases are not patched. Upgrade to the newest 4.x release before you report a problem that may already be fixed.

## Reporting a Vulnerability

If you discover a security vulnerability in Vestige, please report it responsibly:

1. **DO NOT** open a public GitHub issue
2. Email the maintainer directly (see GitHub profile)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

You can expect a response within 48 hours.

## Security Model

### Trust Boundaries

Vestige runs on your machine with your user permissions:

- **Trusted**: the MCP client or local agent that connects over stdio, and any other agent you run on the same machine. The first `vestige-mcp` to start owns the store. Every later one attaches to it through an owner-only socket (mode `0600`) that needs a random token from an owner-only endpoint file, so any process that can read your data directory can use the store.
- **Untrusted**: the content of tool arguments, which is validated before use, and every web page open in your browser.

HTTP and the dashboard are off unless you turn them on, and both answer only on `127.0.0.1`. The dashboard checks `Host` and `Origin` on every request. A request that changes anything must come from the dashboard's own page or carry `Authorization: Bearer <token>`. The HTTP MCP transport takes the same token. Details: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### What Vestige Does NOT Do

- Make network requests in a default build. There is no embedding model to download and no startup version check. `vestige update` contacts GitHub only when you run it. `source_sync` (`--features connectors`) and `vestige sync --cloud` (`--features cloud-sync`) are the only calls that use the network, and neither is in a 4.x release build.
- Require telemetry, a hosted service or a cloud account
- Send telemetry or analytics
- Phone home to any server
- Read or write outside its data directory unless you point it there: `project` writes only inside the `root` you give it, `codebase` reads the `repoPath` you pass, and `vestige backup` and `maintain` `export` write the folder you name

### Data Storage

Everything is stored locally in the data directory. Strata, an append-only, hash-chained log, is the store. It sits in `log/`. Sealed segments carry a signed trailer, and writes are checked by the gate before they are admitted.

| Item | Location |
|------|----------|
| Strata log | `<data-dir>/log/` |
| Signing and identity keys | `receipt-signing.key` and `actor.key` in `<data-dir>/`. `vestige backup` does not copy them |
| HTTP and dashboard token | `auth_token` in the default per-user data directory, or `VESTIGE_AUTH_TOKEN` |
| Backups and exports | `<data-dir>/backups/` and `<data-dir>/exports/` |

On macOS the default data directory is `~/Library/Application Support/com.vestige.core/`. Other platforms and overrides (`--data-dir`, `VESTIGE_DATA_DIR`) are in [docs/STORAGE.md](docs/STORAGE.md).

On Unix the data directory is `0700`, and the lock, socket and endpoint files are `0600`. Backups are owner-only (folders `0700`, files `0600`).

After an upgrade from v3 the old `vestige.db` and the owner-only backup the upgrade made stay in the data directory. They hold your pre-upgrade records. Delete them yourself when you no longer need a rollback.

### Encryption at Rest

The Strata log is not encrypted. Its hash chain and signed trailers prove integrity, not secrecy. Use OS-level encryption (FileVault, BitLocker, LUKS) and keep backups on encrypted storage. The SQLCipher option (`encryption` feature) belongs to the v3 SQLite engine and is not in any release build.

### Erasure

A Strata log cannot erase a record. `suppress` hides a record from every read and keeps its bytes. On Strata it cannot be undone. `purge`, `memory` actions `purge` and `delete`, and `delete_knowledge` return `unavailable_in_4_0`. If a secret reaches the log, rotate it. Real erasure is planned as crypto-erasure ([#402](https://github.com/samvallad33/vestige/issues/402)).

### Credential Gate

Every write is scanned before it is admitted, in every stored field (content, type, tags, source and provenance) and for intentions too. A match refuses the write and names the kind of credential, never the value. Refused kinds: Google API keys, GitHub tokens, AWS access key ids, Slack tokens, PEM private keys, Azure client secrets, Anthropic keys, OpenAI keys and Stripe live keys. A bare `password: ...` line is not blocked. `allowSecrets` lets one save store a credential in its own text; the scope is always checked, and intentions are always gated. `vestige scan-secrets` reports anything already in the log. It opens the log directly, so run it while no Vestige server holds the store, or on a copy.

### Input Validation

- Content is limited to 1,000,000 bytes per memory, and `items` to 20 per call
- Handles resolve exactly: an id, a unique id prefix, or an exact tag. There is no query language and no full-text index to inject into
- The dashboard, HTTP transport and attach socket refuse requests that fail their guards before any tool runs
- `project` refuses a target that escapes `root`, is a symlink, is not a regular file, or is larger than 2 MiB
- `codebase` `ingest_repo` reads git locally with `--no-show-signature` and `core.fsmonitor=false`, so a hostile checkout's config cannot make it run a program

### Dependencies

Dependencies are pinned in `Cargo.lock`. To see the current advisory status, run `cargo audit` against that lockfile. `strata-store`, `strata-kernel` and `strata-verify` set `#![forbid(unsafe_code)]`. Other crates use `unsafe` for a few operating-system calls: fsync on macOS, a disk-space check in the upgrader, and glibc shims in the server.

### Operator Lite

Operator Lite is a separate layer: a stdlib-only Python hook that applies pattern rules to commands before they run, and writes its own hash-chained receipts. It blocks what its closed rule set recognizes and that is routed through hooked tools. It cannot see around hooks. Receipts are hash-chained digests, not signatures. See [operator-lite/README.md](operator-lite/README.md).

## Security Checklist

- [x] No hardcoded secrets
- [x] The credential gate runs on every write
- [x] Exact handles only, with no query language to inject into
- [x] Local-only transports, with Host, Origin and token checks
- [x] Owner-only files and folders on Unix
