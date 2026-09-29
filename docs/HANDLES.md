# Handles — the 4.0 lookup contract

Vestige 4.0 answers lookups by **exact handle**. Free text is never searched:
there is no keyword path, no embedding path, no fuzzy fallback, and no
`mode` dispatch on `recall`. An input that is not a byte-exact handle — or a
handle that resolves to zero nodes — fails closed with the `handle_required`
payload.

Source of truth: `crates/vestige-mcp/src/tools/recall.rs` (`ACCEPTED_PREFIXES`,
`handle_scheme`, `handle_required_payload`).

## The grammar

Fifteen schemes are accepted, in the order `handle_required` reports them:

```
mem:  sha:  line:  sym:  path:  test:  run:  call:  issue:  pr:  purl:
node:  receipt:  tag:  session:
```

`repo := <forge>/<numeric id>` wherever a repo appears. A forge is one or more
of `a-z`, `0-9`, `-`, `.`, `_`; the id is all digits (e.g. `github/1234`).

Matching is **byte-exact**: no trimming, no case folding, no globbing, no
prefix match, no short SHAs. `SHA:github/1@…`, `sha:github/abc@…` (non-numeric
id), and a bare or abbreviated SHA are all rejected.

| Handle | Shape | Notes |
|---|---|---|
| `mem:` | `mem:<uuid-v4 lowercase>` or `mem:mem-<digits>` | The bare alias `mem-<seq>` (no colon) is also accepted and reported as scheme `mem:`. Ingest mints ids of the form `mem-<16 hex>` from the log head. |
| `sha:` | `sha:<repo>@<40\|64 lowercase hex>` | A commit. |
| `path:` | `path:<repo>@<sha>:<path>` | Path may not contain whitespace or `#`. |
| `line:` | `line:<repo>@<sha>:<path>#L<n>[-L<m>]` | Optional `-L<m>` range; both numbers are digits. |
| `sym:` | `sym:<repo>@<sha>::<symbol path>` | Parser symbol path, e.g. `vestige::core::ingest`. |
| `test:` | `test:<framework>:<id>` | Framework has no whitespace or `#`; the id may contain `#` (e.g. `test:junit:com.x.YTest#run`). |
| `run:` | `run:<forge>:<run>[/<job>]` | CI run, optionally a job within it (e.g. `run:github:998877/1234`). |
| `call:` | `call:<session>/<call>` | One tool call inside a session. |
| `issue:` | `issue:<repo>#<n>` | |
| `pr:` | `pr:<repo>!<n>` | |
| `purl:` | `purl:<package-url>` | No whitespace. |
| `node:` | `node:<40\|64 lowercase hex>` | A store node id. |
| `receipt:` | `receipt:<40\|64 lowercase hex>` | A receipt id. |
| `tag:` | `tag:<name>` | **Filter only.** `tag:` narrows a lookup; it is not a text search over tag names. |
| `session:` | `session:<id>` | |

### Resolution semantics

- `mem:` (and `mem-<seq>`) resolve by exact id against the store.
- Every other scheme goes through the byte-exact resolver and must return
  **exactly one** exact hit. A prefix match, an ambiguous (>1) match, and a
  miss all fail closed to `handle_required`.
- One call accepts at most **16 handles** (`handle` is a string or an array of
  at most 16 strings). More than 16 fails closed.
- Optional controls: `as_of` (caller-supplied RFC3339 decision point),
  `k` (walk breadth, `1..=3`, default 1), and `edge_types` (a subset of the 8
  typed edges: `touched`, `anchored_to`, `derived_from`, `supersedes`,
  `corrects`, `closed_by`, `projected_to`, `evidence_of`).

### Lineage targets are handles too

`supersedes`/`corrects` lineage on ingest (admission) accepts only exact
`mem:` handles and bare `mem-<seq>` ids as targets; anything else is not an
exact handle and the proposal is refused with a receipt
(`crates/strata-store/src/admission.rs`).

## The `handle_required` payload

Every non-handle input — free text, an empty string, a malformed handle, or a
grammar-valid handle that resolves to nothing — returns the same fail-closed
shape:

```json
{
  "error": "handle_required",
  "got": "<your input, echoed, truncated to 200 characters>",
  "accepted": ["mem:", "sha:", "line:", "sym:", "path:", "test:", "run:",
               "call:", "issue:", "pr:", "purl:", "node:", "receipt:",
               "tag:", "session:"],
  "session_handles": [],
  "hint": "pass an exact handle; free text is never searched"
}
```

Notes:

- There is **no `candidates` field**. Token mining is gone: a full sentence
  that happens to contain a real memory id still returns `handle_required`
  with nothing resolved from the prose.
- The legacy `mode` and `query` arguments are not part of `recall` anymore;
  passing them returns `handle_required` (they are never interpreted as a
  search).
- `recall` never touches an FTS table; it works on stores where the FTS
  machinery does not exist at all.

## Why exact handles

Hard rule H5: retrieval identity comes from the caller, not from a ranker.
Handles make every lookup replayable — the same handle at the same `as_of`
resolves the same way — and they keep resemblance out of the decision path
entirely (H1/H2). If you have prose and need a handle, list session handles or
read receipts; do not paraphrase into a query.

## Where handles come from

- Node ids minted by ingest: `mem-<16 hex>` → use `mem:<id>` or the bare id.
- Store nodes and receipts: `node:` / `receipt:` with their hex ids.
- Code, CI, and issue artifacts: anchor them yourself with `sha:` / `path:` /
  `line:` / `sym:` / `test:` / `run:` / `call:` / `issue:` / `pr:` / `purl:`
  when you record facts about them, so the facts stay addressable later.
