# Code context evidence

> Written for Vestige 4.1.1.

`codebase.get_context` and `session_start` share current-record selection and
source-anchor evaluation. Startup retains each code record's ID, actionable
summary and evidence state. A record is found by its tags (`codebase:<name>`),
never by wording. Source-span equality does not prove a natural-language claim.

Pass an explicit checkout when reading context:

```json
{
  "scope": "user",
  "context": {
    "codebase": "example-project",
    "repoPath": "/path/to/checkout"
  },
  "token_budget": 2000
}
```

On a Strata log `session_start` ignores `queries` and `changed_files`, and says so in `notices`. Predictions are not available.

Use the same `scope`, `codebase` and `repoPath` for `codebase` with
`action: "get_context"`. Codebase writes accept `scope` too; omitted scope
retains the `user` namespace. Project tags match exactly, so `app` does not
select `app-extra`. The dedicated code-context query excludes future, expired and
retired records before its per-type limit. Nothing is deleted by this selection.
If the requested scope holds no code records for the codebase, the response names
the scopes that do (`codeContext.elsewhere` and `notices`). `codebase`
`get_context` lists every scope that holds them in `scopes`, and `allScopes: true`
reads them all.

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
`get_context`. The verify action likewise requires an explicit `repoPath` and
refuses without one, so no verdict is persisted against the server's working
directory. Remember actions retain their existing fallback.
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

Old unprefixed hashes, carried over from a v3 store, remain stored but return
unverifiable with a legacy-format explanation. They cannot be upgraded safely
without reviewing the source: the old format discarded information. No automatic
re-anchoring occurs. After reviewing the advice against the source, explicitly
replace its anchors without duplicating or promoting the record:

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

Anchors you write after the upgrade live in the Strata log. v3.1.1 reads only the
old `vestige.db`, so a rollback to it does not see them, and an old verifier could
label v2 anchors drifted. Keep the backup the upgrade made, and take your own
with `vestige backup <new-dir>` before a rollback.

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

A request with more than sixteen `queries` is rejected, even though a Strata log
ignores them. Reading context does not change a record's strength.

`codebase` `verify` with a `codebase` also checks that codebase's change records
(the `event` records `ingest_repo` writes from git commits). `limit` applies to each
type. The response carries `totalByType`, `uncheckedByType` and `truncated`, so a
sweep that stopped short does not read as the whole scope.

## Verification

Run the standard library-only MCP fixture with a freshly built server:

```sh
cargo build -p vestige-mcp --bin vestige-mcp
python3 scripts/test-context-evidence.py --binary target/debug/vestige-mcp
```

The fixture creates a disposable data directory. It checks the argument gate (more
than sixteen `queries`, a blank scope), an exact get of a force-created record, an
empty code context, and that the data directory never grows a SQLite file. It does
not check drift or anchor coverage. The anchor unit tests in
`crates/vestige-core/src/codebase/anchor.rs` cover moved code, a rewritten body and
significant whitespace, and the `codebase` tests in
`crates/vestige-mcp/src/tools/codebase_unified.rs` cover remember, verify, drift and
reanchor on a Strata log. The fixture never opens an existing data directory. CI
runs it in its no-embeddings job.

Dependency-triggered invalidation and delta handles are not implemented.
`codebase` `ingest_repo` turns git history into anchored change records that
`verify` re-checks. This page covers observed evidence on the two context tools.
