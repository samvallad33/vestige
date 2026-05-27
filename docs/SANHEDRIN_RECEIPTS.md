# Sanhedrin Receipt Schema

Sanhedrin writes local, inspectable receipts so a Stop-hook veto is appealable
instead of opaque. The current schema is `vestige.sanhedrin.receipt.v1`.

## Locations

- Latest JSON: `~/.vestige/sanhedrin/latest.json`
- Latest HTML: `~/.vestige/sanhedrin/latest.html`
- Receipt archive: `~/.vestige/sanhedrin/receipts/<receipt-id>.json`
- Command receipt ledger: `~/.vestige/sanhedrin/command-receipts.jsonl`
- Appeals: `~/.vestige/sanhedrin/appeals.jsonl`
- Fail-open events: `~/.vestige/sanhedrin/fail-open.jsonl`

## v1 JSON Shape

```json
{
  "schema": "vestige.sanhedrin.receipt.v1",
  "id": "receipt_<stable hash>",
  "draftId": "draft_<stable hash>",
  "createdAt": "2026-05-25T18:00:00+00:00",
  "overall": "pass|pass_with_warnings|veto|appealed",
  "verdictBar": "PASS|NOTE|CAUTION|VETO|APPEALED",
  "summary": "Human-readable result",
  "draftPreview": "First 1000 chars of the assistant draft",
  "claims": [
    {
      "id": "c001",
      "text": "All tests passed.",
      "fingerprint": "16-char sha256 prefix",
      "class": "receipt_lock|TECHNICAL|ACHIEVEMENT|...",
      "subject": "Sam|draft|command receipt",
      "risk": "normal|hard",
      "evidence_state": "supported|missing_receipt|contradicted|appealed|...",
      "decision": "pass|pass_unverified|veto|appealed",
      "precedent": [
        {
          "type": "command|receipt_lock|vestige|appeal",
          "summary": "Why this claim passed or failed",
          "command": "cargo test --workspace",
          "exitCode": 0
        }
      ],
      "fix": "Suggested rewrite",
      "appeal": {
        "status": "open|appealed",
        "actions": ["stale", "wrong", "too_strict"]
      }
    }
  ],
  "receipts": [
    {
      "source": "transcript|codex-transcript",
      "command": "cargo test --workspace",
      "exitCode": 0,
      "success": true,
      "timestamp": "2026-05-25T18:00:00+00:00"
    }
  ],
  "source": {
    "stateDir": "~/.vestige/sanhedrin",
    "transcript": "/path/to/session.jsonl"
  }
}
```

## Compatibility Rules

- Readers should accept `vestige.sanhedrin.receipt.v1` without warning.
- Readers should keep rendering unknown schemas defensively, but surface a
  warning instead of silently treating them as v1.
- New schema versions must keep `id`, `createdAt`, `verdictBar`, `summary`, and
  `claims` stable or provide a dashboard migration.

## Staged Evidence Boundary

`VESTIGE_SANHEDRIN_STAGE_FILE` is a non-durable overlay for current-turn context.
It may help the executioner understand a draft, but code enforces that staged
evidence cannot satisfy durable evidence requirements for `SUPPORTED`,
`REFUTED`, or `REFUTED_BY_ABSENCE`. Durable support must come from Vestige memory
or command receipts.

## Receipt Lock Compatibility Flags

`VESTIGE_SANHEDRIN_ALLOW_COMMAND_LEDGER=1` lets Receipt Lock read
`command-receipts.jsonl` when no live transcript path is available.

`VESTIGE_SANHEDRIN_ALLOW_LOOSE_LEDGER=1` re-enables the legacy fallback that
regex-scans transcript JSON blobs for `command` or `cmd` fields. Keep this off
unless you are migrating old transcripts; structured tool-use receipts are safer
because loose scanning can mistake quoted text for a real command execution.

Hosted Sanhedrin backends should use `VESTIGE_SANHEDRIN_API_KEY` in
`~/.claude/hooks/vestige-sanhedrin.env`. The installer keeps that file at mode
`0600`; do not store shared or unrelated API keys there.
