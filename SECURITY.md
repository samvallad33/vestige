# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.3.x   | :white_check_mark: |
| 2.2.x   | Critical fixes only |
| 2.1.x   | Critical fixes only |
| 2.0.x   | :x:                |
| 1.x     | :x:                |

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

Vestige is a **local MCP server** designed to run on your machine with your user permissions:

- **Trusted**: The MCP client or local agent that connects via stdio
- **Untrusted**: Content passed through MCP tool arguments (validated before use)

### What Vestige Does NOT Do

- ❌ Make network requests during normal memory use, except first-run model download from Hugging Face
- ❌ Require telemetry, hosted memory storage, or a cloud account
- ❌ Access files outside its data directory
- ❌ Send telemetry or analytics
- ❌ Phone home to any server

### Data Storage

All data is stored locally in SQLite:

| Platform | Location |
|----------|----------|
| macOS | `~/Library/Application Support/com.vestige.core/vestige.db` |
| Linux | `~/.local/share/vestige/core/vestige.db` |
| Windows | `%APPDATA%\vestige\core\vestige.db` |

**Default**: Data is stored in plaintext with owner-only file permissions (0600).

### Encryption at Rest

For database-level encryption, build with SQLCipher:
```bash
cargo build --no-default-features --features encryption,embeddings,vector-search
```
Set `VESTIGE_ENCRYPTION_KEY` environment variable. SQLCipher encrypts all database files including the WAL journal. Alternatively, use OS-level encryption (FileVault, BitLocker, LUKS).

### Input Validation

All MCP tool inputs are validated:

- Content size limit: 1MB max
- Query length limit: 1000 characters
- FTS5 queries are sanitized to prevent injection
- All SQL uses parameterized queries (`params![]` macro)

### Dependencies

We use well-maintained dependencies and run `cargo audit` as part of the release gate. Status as of the v2.3.0 audit (2026-07-25, 564 crate dependencies scanned):

- **Vulnerabilities**: 0
- **Warnings**: 9 (3 unmaintained, 5 unsound, 1 yanked), all transitive, none with a known exploit path in Vestige

That zero is not the default state. The 2.3.0 lockfile carried 5 vulnerabilities
until three transitive dependencies were updated. Resolved for this release:

| Advisory | Crate | Issue | Resolved by |
|---|---|---|---|
| RUSTSEC-2026-0185 (7.5 high) | `quinn-proto` | Remote memory exhaustion from unbounded out-of-order stream reassembly | 0.11.14 -> 0.11.16 |
| RUSTSEC-2026-0104 | `rustls-webpki` | Reachable panic parsing a certificate revocation list | 0.103.11 -> 0.103.13 |
| RUSTSEC-2026-0098 | `rustls-webpki` | Name constraints for URI names incorrectly accepted | 0.103.11 -> 0.103.13 |
| RUSTSEC-2026-0099 | `rustls-webpki` | Name constraints accepted for certificates asserting a wildcard name | 0.103.11 -> 0.103.13 |
| RUSTSEC-2026-0204 | `crossbeam-epoch` | Invalid pointer dereference in the `fmt::Pointer` impl | 0.9.18 -> 0.9.20 |

`rustls-webpki` is the certificate validation path that Vestige Pro cloud sync
depends on, so these versions are pinned forward in `Cargo.lock` and a downgrade
is a release blocker, not a lockfile detail.

The 9 remaining warnings are not vulnerabilities and have no available fix we can
apply, because each is pulled in transitively by `fastembed`, `candle`,
`usearch`, or `git2`:

- **Unmaintained**: `core2` (RUSTSEC-2026-0105, also yanked), `number_prefix`
  (RUSTSEC-2025-0119), `paste` (RUSTSEC-2024-0436)
- **Unsound**: `anyhow` (RUSTSEC-2026-0190), `cxx` (RUSTSEC-2026-0202), `git2`
  (RUSTSEC-2026-0183, RUSTSEC-2026-0184), `memmap2` (RUSTSEC-2026-0186)

We track these and will pull the fixes through as soon as our direct
dependencies publish releases that carry them. We would rather print the real
number here than a zero.

## Security Checklist

- [x] No hardcoded secrets
- [x] Parameterized SQL queries
- [x] Input validation on all tools
- [x] No command injection vectors
- [x] No unsafe Rust code
- [x] Dependencies audited
