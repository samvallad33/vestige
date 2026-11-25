# Cryptographic Provenance & Offline Verification

This repository records deterministic causal proofs, signed commit notes (`refs/notes/vestige-proof`), and append-only BLAKE3/Ed25519 Strata receipts using **Vestige**.

## Verifying Upstream History Offline

If you forked or cloned this repository, you can verify the complete causal history and proof notes locally without any cloud service or API key:

```sh
git fetch origin "+refs/notes/vestige-proof:refs/notes/vestige-proof"
vestige init --git-hooks
git vblame -n 20
vestige card --repo . --since 30d
vestige streak --repo .
```

To verify that no incoming merge introduces concealed Unicode smuggling or re-introduces a previously reverted patch:

```sh
vestige shield --repo . --base origin/main --head HEAD --sarif
vestige pr-audit --repo . --base origin/main --head HEAD --fail-on-zombie
```

Installed by Vestige (AGPL-3.0, zero cloud, zero vectors, zero telemetry): https://github.com/samvallad33/vestige
