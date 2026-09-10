# Code context evidence

`codebase.get_context` and `session_start` share current-memory selection and
source-anchor evaluation. Startup retains each code memory's ID, actionable
summary and evidence state, including code advice found through search or
predictions. Source-span equality does not prove a natural-language claim.

Pass an explicit checkout when reading context:

```json
{
  "queries": [],
  "scope": "user",
  "context": {
    "codebase": "example-project",
    "repoPath": "/path/to/checkout"
  },
  "include_predictions": false,
  "token_budget": 2000
}
```

Use the same `scope`, `codebase` and `repoPath` for `codebase` with
`action: "get_context"`. Codebase writes accept `scope` too; omitted scope
retains the `user` namespace. Project tags match exactly, so `app` does not
select `app-extra`. The dedicated code-context query excludes future, expired and
superseded memories before its per-type limit. Search and prediction candidates
are also checked for scope and current eligibility before delivery. Historical storage APIs remain
available; no memories are deleted by this selection.

## Reading evidence

Each code item carries `anchorStatus`, per-anchor details when checked, and
an `evidence` object:

| State | Meaning |
|---|---|
| `unchanged_evidence` | Every recorded anchor matches its observed source span |
| `needs_recheck` | At least one anchor has changed or is missing |
| `partial` | Some anchors match but others cannot be checked |
| `unavailable` | Evidence cannot be evaluated, or verification was disabled |
| `unanchored` | No anchor rows exist for this memory |

`checkedAnchors`, `matchingAnchors` and `totalAnchors` describe coverage.
`claimVerified` is always false: this check evaluates source identity, not
whether a remembered rule is semantically correct. Per-anchor `checkedAt`
records observation time, not when the source changed. The verification
summary identifies the canonical checkout used; all item checks are live and
no persistent latest-verdict cache is written by these context reads.

A missing or omitted `repoPath` on context reads produces unavailable evidence.
This deliberately replaces the implicit working-directory fallback for
`get_context`. Remember and verify actions retain their existing fallback.
Existing callers should pass the checkout they intend to inspect.
`verify: false` returns explicit unavailable evidence rather than silently
omitting evidence fields. Anchor-storage errors fail the tool call instead of
masquerading as successful empty results.

Relative anchors are evaluated against the selected checkout, so two worktrees
can yield different evidence for the same memory. Absolute anchors retain their
explicit absolute target. These checks are not an atomic filesystem snapshot:
files can change between observations. Symbol selection remains heuristic;
there is no whole-program semantic equivalence claim.

## Anchor format and compatibility

New content hashes have a `v2:` prefix and preserve indentation, blank lines
and whitespace within strings. Line endings and a final newline are normalized
by the line-based representation. Formatting-only changes may consequently
request rechecking. Unchanged spans can still be found after line relocation.

Old unprefixed hashes remain stored but return unverifiable with a legacy-format
explanation. They cannot be upgraded safely without reviewing the source: the
old format discarded information. No automatic re-anchoring or database
migration occurs. After reviewing the advice against the source, explicitly
replace its anchors without duplicating or promoting the memory:

```json
{
  "action": "reanchor",
  "memoryId": "existing-memory-id",
  "scope": "user",
  "repoPath": "/path/to/checkout",
  "files": ["src/example.py#example"]
}
```

Every supplied anchor must be captured successfully. Replacement is atomic;
wrong-scope and incomplete requests preserve the old anchors. The memory's
content, ID and strength do not change. The action replaces source evidence;
it does not verify the memory's claim.

A rollback to an older binary must not read newly captured v2 anchors as if they
were its own format; an old verifier can falsely label them drifted. Preserve a
pre-upgrade database backup before deploying and restore that backup with the
old binary if rollback is required. This development patch does not modify the
installed server or the live store.

## Startup budgets and partial output

`token_budget` bounds the serialized tool result using an explicitly labeled
estimate: UTF-8 bytes divided by four, rounded up. `tokensUsed` includes the
whole result envelope and is not a model-tokenizer measurement. It does not
include outer MCP transport framing or a client's text rendering.

Code summaries and their evidence are kept or omitted together. Other sections
are removed first under pressure. `expandable` supplies omitted IDs when they
fit; `omitted` counts items/sections left out, so it is not a total memory count.
At very small budgets the optional `codeContext` and automation fields may be
omitted. Increase the budget to expand context. Verification summary counts
cover evaluated candidates, including candidates subsequently omitted from the
packet. A due reminder is evaluated even with no context object, but a budget
can still omit the entire intention section.

Requests with more than sixteen search queries are rejected explicitly.
Retrieval telemetry records only IDs in the final rendered context and does not promote their strength.

## Verification

Run the standard library-only MCP fixture with a freshly built server:

```sh
cargo build -p vestige-mcp --no-default-features --features connectors,cloud-sync --bin vestige-mcp
python3 scripts/test-context-evidence.py --binary target/debug/vestige-mcp
```

The fixture creates disposable stores and checkout directories. It covers drift,
unrelated edits, partial coverage, namespace and project selection, checkout
switches, legacy anchors, output accounting, repeated reads, due intentions and
an injected anchor-storage failure. It never imports a live tracker or opens an
existing memory database. Core anchor unit tests additionally cover significant
whitespace in string content. The no-embeddings CI job runs the MCP fixture.

Source-sync coverage, dependency-triggered invalidation, delta handles and
conditional failed-approach memory are subsequent extensions. This patch
establishes consistent observed evidence on the two context tools.
