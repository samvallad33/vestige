# INGEST V5 SPEC — the proof-carrying write path (feat/ingest-proof)

Three build lanes with DISJOINT file ownership. Read this whole file before editing. Repo: /Users/entity002/vestige, branch feat/ingest-proof (off release-412 @ d20413ba). **Do NOT `git commit`, `git push`, or touch files outside your lane — the coordinator integrates and commits once.** `cargo check`/`cargo test` may wait on a target-dir lock; that is expected (siblings are building too). Run `cargo fmt` on files you touch.

## Laws (violations = rework)

1. Append-only forever. No content mutation, no merges, no in-place updates. A duplicate NEVER touches the original.
2. Determinism: same store + same call = same response bytes, apart from honestly-labeled time fields. No clock reads for decisions, no randomness, no HashMap-iteration-order-dependent output (sort everything with a total order: bytes, then ids).
3. No LLM, no embeddings, no network in the write path.
4. No new `StoreOp` kinds (log format is frozen). Use existing ops: UpsertNode, SaveEdge.
5. Every new response field must be derivable from log facts or from the submitted bytes alone.

## Lane A — store layer (owns: crates/strata-store/** incl. Cargo.toml + tests, crates/vestige-mcp/src/strata_memory.rs, crates/vestige-core/src/storage/memory_store.rs trait defaults)

### A1. canonical.rs (new, in strata-store, pub)
```rust
pub const CANONICAL_PIPELINE_VERSION: &str = "nfc-lower-zwstrip-wscollapse-v1";
pub fn canonicalize(content: &str) -> String;   // Unicode NFC -> to_lowercase -> strip U+200B..U+200D,U+FEFF -> collapse any unicode-whitespace run to one ' ' -> trim
pub fn canonical_hash(content: &str) -> [u8; 32];      // blake3 over canonicalize(content) as bytes (strata-store already deps blake3)
pub fn canonical_hash_hex(content: &str) -> String;    // lowercase hex
```
(unicode-normalization is already a dep of vestige-mcp; ADD it to strata-store's Cargo.toml.)
Unit tests: NFC (e+combining vs precomposed equal), case, zero-width, whitespace collapse, multibyte safety, hex stability golden vector (print-once, pin exact hex in test).

### A2. Canonical index + duplicate lookup (StrataStore)
Follow the existing index pattern used by `effect_index` in crates/strata-store/src/store.rs (same storage substrate, created best-effort at open with IF NOT EXISTS so existing stores upgrade in place):
- Record `(scope, canonical_hash) -> node_id` for every NEW regular node created via `ingest_in_scope_with_receipt`. Do NOT index nodes whose source field == "duplicate" (echo nodes never become dedup targets).
- `pub fn find_node_by_canonical_hash(&self, scope: &str, hash: &[u8; 32]) -> Result<Option<String>>`
- Backfill is out of scope v1 (index starts at now; document in code comment).

### A3. Intent index (idempotent replay)
- `(scope, intent_id) -> (node_id, effect_seq, response_digest)` recorded at create time when the caller supplied intent_id. Table `intent_index`, PRIMARY KEY (scope, intent_id), INSERT OR REPLACE never (first write wins; replay returns the first).
- `pub fn find_intent(&self, scope: &str, intent_id: &str) -> Result<Option<(String, u64, String)>>`
- `pub fn record_intent(&self, scope: &str, intent_id: &str, node_id: &str, effect_seq: u64, response_digest: &str) -> Result<()>`

### A4. Trait surface (vestige-core/src/storage/memory_store.rs)
Add to the `MemoryStoreSend` trait with DEFAULT impls (so SQLite legacy still compiles unchanged):
```rust
fn find_duplicate_by_canonical_hash(&self, scope: &str, content: &str) -> Result<Option<String>> { Ok(None) }
fn find_intent_record(&self, scope: &str, intent_id: &str) -> Result<Option<(String, u64, String)>> { Ok(None) }
fn record_intent_entry(&self, scope: &str, intent_id: &str, node_id: &str, effect_seq: u64, response_digest: &str) -> Result<()> { Ok(()) }
fn latest_receipt_id_for_node(&self, node_id: &str) -> Option<String> { None }
```
Implement all four on `StrataMemory` (strata_memory.rs). `latest_receipt_id_for_node` = wrap the existing resolve_proof/latest_effect path, return `eff-...` string.
Response_digest for v1 = hex of blake3 over the JSON `{"content":canonical_hash_hex,"source":...,"tags":sorted tags}` — defined here so the tool layer can recompute it identically; export a helper `pub fn intent_digest(canonical_hash_hex: &str, source: &str, tags: &[String]) -> String` from strata-store canonical.rs (tags sorted+deduped first).

### A5. supersedes declarable (strata_memory.rs)
Extend `DeclaredLink` (+ NAMES, parse, edge()) with `Supersedes`: caller target = the OLD memory; record edge `new_node -[supersedes]-> old_node`. Keep 16-link cap and all existing checks (same scope, live target, no dupes) unchanged. Update the doc comment that called it review-gated (now caller-declarable on Strata; note in comment: for corrections use corrects via review, supersedes is for full replacement).

### A6. Tests (strata-store + strata_memory-level)
- canonical golden vectors; duplicate found on second identical ingest (via find_node_by_canonical_hash after ingest); intent record/find roundtrip; first-write-wins on double record_intent; supersedes edge persisted with correct direction; latest_receipt_id_for_node returns eff- id.
Runner: `cargo test -p strata-store` and `cargo test -p vestige-mcp --lib` (or wherever strata_memory unit tests live; if the mcp crate has no lib target for these, put StrataMemory-level tests in crates/vestige-mcp/tests/ as a plain integration test opening a StrataStore tempdir — follow existing test file patterns).

## Lane B — tool layer (owns: crates/vestige-mcp/src/tools/smart_ingest.rs, plus NEW file crates/vestige-mcp/tests/smart_ingest_proof_stdio.rs)

### B1. Schema additions (schema() fn)
- single + batch item: optional `intent_id` (string, max 128 chars, pattern `[A-Za-z0-9._:-]+`).
- links description: add `supersedes` to the declarable kinds list.
- NO removals; bump nothing; add only.

### B2. Single-write flow (execute_verbose, default-build branch only — do not touch cfg'd legacy branches except where the shared helpers force it)
Order of operations:
1. Parse optional intent_id (strip from args like links are stripped; error on invalid pattern).
2. If intent_id present: `storage.find_intent_record(scope, intent_id)` → if Some((node_id, effect_seq, digest)): return EARLY with decision `"replay"`, fields `{replayed: true, replayOf: node_id, effectSeq, intentDigest: digest, intentId}` and NO write of any kind. This must short-circuit BEFORE canonical check and before any storage write.
3. Canonical gate: `chex = strata_store::canonical::canonical_hash_hex(content)`; `storage.find_duplicate_by_canonical_hash(scope, content)` → if Some(orig):
   - Create the echo node via the EXISTING ingest call: content = `"duplicate of {orig}\ncanonical_hash: {chex}\nsubmitted_sha256_line_count: {n_lines}"`, source = `"duplicate"`, tags = `["duplicate"]` (plus nothing else), scope same.
   - Save edge echo -[evidence_of]-> orig via the existing links machinery (single link; it produces its own eff- receipt).
   - Response: decision `"reinforce"`, `{duplicateOf: orig, echoNodeId: <new id>, canonicalHash: chex, rawBytes: content.len(), pipeline: strata_store::canonical::CANONICAL_PIPELINE_VERSION, receiptId: <echo node receipt>}`. NEVER merge, never include full original content.
   - If intent_id present, record_intent_entry for the echo node too.
4. Else normal create (existing path unchanged) then enrich response with: `receiptId` (from latest_receipt_id_for_node), `canonicalHash` (chex), `entities` (from lane C module: `crate::intake::entities::extract_typed_spans(content)` serialized as `{kind, surface, byteStart, byteEnd}`, max 32 spans), `importance` (from lane C: `crate::intake::importance::compute_and_format(content, &spans, tags)`).
   - If intent_id present: record_intent_entry(scope, intent_id, node_id, effect_seq, digest) where digest from `strata_store::canonical::intent_digest(...)` and effect_seq from the receipt resolution (parse from eff- id or expose via A4 helper returning seq — coordinate: A4's find returns seq; record needs it at write time; simplest: A4 adds `fn node_effect_seq(&self, node_id) -> Option<u64>` default None. SPEC LOCK: use this.)
5. forceCreate:true skips step 3 only (duplicates still possible via intent replay? No — forceCreate skips canonical gate; intent replay still applies since it's idempotency, not dedup).

### B3. Batch flow (execute_batch)
1. Precompute per item: canonical_hash_hex (items with invalid/missing fields keep current behavior).
2. Processing order: sort item indices by `(canonical_hash_hex asc, original_index asc)`. Process in that order. Each result keeps its ORIGINAL `"index"` field; the results array must be re-sorted to original caller order before returning. Deterministic regardless of arrival order.
3. Per item: same intent-replay → canonical-gate → create flow as B2 (batch items CAN dedup against earlier items of the same batch — that is correct and desired; the canonical index naturally handles it because earlier-sorted identical items are indexed first).
4. Batch summary gains counters: `replayed`, `reinforced` (in addition to created/updated/skipped/errors).

### B4. lean_response compatibility
The lean path strips fields; ensure new fields survive lean_response for create/reinforce/replay OR are intentionally stripped — check how lean_response treats existing optional fields and match that behavior for: receiptId, canonicalHash (keep), entities (keep top 8), importance (keep score+weightsVersion only). Document choice in a comment.

### B5. New stdio test file crates/vestige-mcp/tests/smart_ingest_proof_stdio.rs
Follow the harness in crates/vestige-mcp/tests/common/mod.rs (spawns binary; VESTIGE_DATA_DIR tempdir). Cover, in one sequential story against ONE store:
1. forceCreate ingest A ("alpha content with src/store.py and commit a1b2c3d") → assert decision create, receiptId starts "eff-", canonicalHash 64-hex, entities include FilePath and CommitSha spans with correct surfaces, importance.score in (0,1] and importance.weightsVersion present.
2. Same content again (no forceCreate) → decision "reinforce", duplicateOf == A node, echo node exists, and a second call returns duplicateOf == A STILL (echo never becomes the dedup target).
3. content with NFC-different but canonically-identical bytes (e+combining accent vs precomposed, different case, extra whitespace) → still reinforce to A.
4. intent_id="run-42" with NEW content → create; replay same intent_id same content → decision "replay", replayOf set, and store node count UNCHANGED (prove no write: stats before/after or list length).
5. intent_id replay with DIFFERENT content → still replay (idempotency wins; include response field `requestCanonicalHash` of the NEW submission + `intentDigest` of the original so the divergence is visible, not hidden).
6. supersedes link: create B with links [{kind:"supersedes", to: A-node}] → assert edge persisted (via a recall/get or the response's links receipt), direction B -supersedes-> A.
7. batch of 4 (two canonically identical items, one new, one duplicate of A) → summary counters correct; results array in caller order with correct original index fields; the two identical items → one create + one reinforce (order-independence: run the same batch twice with intent_ids on the creates → replays).
Runner: `cargo test -p vestige-mcp --test smart_ingest_proof_stdio` (harness may need `--features` matching sibling stdio tests — copy their attributes exactly).

## Lane C — pure intake modules (owns: NEW dir crates/vestige-mcp/src/intake/ (mod.rs, entities.rs, importance.rs), plus ONE line `pub mod intake;` added to the crate root file where `mod auto_connect;` is declared)

ZERO new dependencies (use std + unicode-normalization if already a dep of vestige-mcp — it is).

### C1. entities.rs
```rust
pub const EXTRACTOR_VERSION: &str = "hand-scanners-v1";
#[derive(Clone, Debug, PartialEq, Eq)] pub enum EntityKind { CommitSha, Url, FilePath, IssueRef, Email, Version }
impl EntityKind { pub fn as_str(&self) -> &'static str { ... } }
#[derive(Clone, Debug, PartialEq, Eq)] pub struct EntitySpan { pub kind: EntityKind, pub surface: String, pub byte_start: usize, pub byte_end: usize }
pub fn extract_typed_spans(content: &str) -> Vec<EntitySpan>; // sorted by byte_start; non-overlapping (longest-leftmost wins); cap 64; deterministic
```
Hand-rolled scanners over char boundaries (byte offsets from char_indices; NEVER split multibyte). Rules:
- Url: `https?://` then run of non-whitespace, trailing punctuation `.,);]}'">` trimmed. Kind wins over FilePath inside it (skip scanning inside Url spans).
- Email: local@domain per RFC-lite (alnum + `._%+-` @ alnum + `.-` + `.` tld 2+).
- CommitSha: hex run of 7..=40, boundary-delimited (prev/next char not hex), must contain at least one a-f letter (rejects pure numbers), not preceded by `#`.
- IssueRef: `[A-Z][A-Z0-9]{1,}-\d+` OR `#\d+` (not preceded by another # or alnum) OR `[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+`.
- FilePath: token with len>=3, no whitespace, contains '/' AND '.' in final segment, first char alnum or `.~/`; reject if it looks like a Url or Version (checker order: Url > Email > CommitSha > IssueRef > Version > FilePath).
- Version: `v?\d+\.\d+(\.\d+)*` with word boundaries; also accept leading `@`? No — bare only.
Table-driven tests: every kind, overlap precedence (a URL containing a path and a sha-looking hex), multibyte content ("重要 fix in 🚒 src/store.py" — spans still byte-correct), determinism (same input twice → identical vec), cap behavior, empty input.
Golden test: pin EXTRACTOR_VERSION output for one mixed paragraph.

### C2. importance.rs
```rust
pub const WEIGHTS_VERSION: &str = "linear-v1";
#[derive(Clone, Copy, Debug, PartialEq)] pub struct ImportanceFactors { pub length_bytes: usize, pub entity_count: usize, pub unique_entity_count: usize, pub tag_count: usize, pub entropy_bits_per_byte: f64, pub type_token_ratio: f64 }
pub fn compute_factors(content: &str, spans: &[EntitySpan], tags: &[String]) -> ImportanceFactors;
pub fn score(f: &ImportanceFactors) -> f64;                 // clamped [0,1]
pub fn recompute_from_factors(f: &ImportanceFactors) -> f64; // MUST be byte-identical logic to score — implement score AS recompute_from_factors (single fn body, two names)
pub fn compute_and_format(content: &str, spans: &[EntitySpan], tags: &[String]) -> serde_json::Value; // {score, formula:"linear-v1", weightsVersion, factors:{...all six, 3-decimal f64s}}
```
Published formula (document it in a doc comment, every weight named):
`score = clamp01( 0.25*entity_density + 0.20*entropy_norm + 0.15*length_band + 0.15*type_token_norm + 0.15*tag_signal + 0.10*unique_entity_ratio )` where:
- entity_density = min(1, entity_count / 16)
- entropy_norm = clamp01((entropy_bits_per_byte - 3.0) / 5.0)  (byte-histogram Shannon entropy; ASCII text lands ~3-6)
- length_band = 1 - min(1, |ln(max(len,1)/512)| / ln(16))  (sweet spot 512 bytes, ±4x falloff)
- type_token_norm = clamp01(type_token_ratio) where ratio = unique words / total words (whitespace-split, lowercase)
- tag_signal = min(1, tag_count / 8)
- unique_entity_ratio = if entity_count==0 {0} else {unique_surfaces / entity_count}
unique = distinct (kind, surface) pairs.
Tests: score==recompute trivially but assert it; empty content; f64 output rounded to 3 decimals in JSON via format; determinism golden vector; factors all finite.

### C3. mod.rs
`pub mod entities; pub mod importance;` plus a crate-level doc comment stating the law: pure functions of bytes, no clocks, no RNG, no network; version-pinned.

## Coordination notes
- Lane B compiles against lane C's exact paths (`crate::intake::entities::extract_typed_spans`, `crate::intake::importance::compute_and_format`) and lane A's (`strata_store::canonical::canonical_hash_hex`, `CANONICAL_PIPELINE_VERSION`, `intent_digest`, trait methods `find_duplicate_by_canonical_hash`, `find_intent_record`, `record_intent_entry`, `latest_receipt_id_for_node`, `node_effect_seq`). If a signature must change, change it in YOUR lane only and note the delta in your final report; do not edit another lane's files.
- If your lane fails to compile because a sibling lane's code isn't there yet, gate the call sites with a minimal local `#[allow(dead_code)]` stub is FORBIDDEN — instead write the code to the spec signatures and let integration (coordinator) resolve; run only the tests that don't need siblings.
- cargo check -p vestige-mcp is the mandated pre-commit check; you run it for your lane's crate only if it compiles standalone, else run your crate's unit tests.
