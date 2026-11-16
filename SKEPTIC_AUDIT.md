# Skeptic audit: hand-typed word lists in engine code

Scope: everything under `crates/` (engine and MCP server). Excluded by the
owner for this pass: `benchmarks/real_engine/`, `crates/vestige-core/src/rungs.rs`,
and `rsb_seat_behind` (not touched). Enforcement: `scripts/hardcode_scan.sh`
(exit 0 on this commit; exit 1 with 96 findings on the previous `main`).

Rule applied: a hand-typed list of open-class words (topics, cues, sentiment,
domains, gazetteers) is removed. Where a list fed a signal, the signal is
either replaced by a structural one (an explicit recorded tag, document
frequency, timestamps, recorded links, grammar shape) or dropped. Closed-class
grammar and wire/schema protocol stay, each marked in code with
`// hardcode-scan: closed-class (...)` or `// hardcode-scan: protocol (...)`.

## Removed (file:line on the previous main)

| Where | What | Replacement |
|---|---|---|
| vestige-core/src/retrieval.rs:296 | `PREF_MARKERS` (preference verbs/phrases) | dropped; preference intent is not inferred from phrases |
| retrieval.rs:2189 | `PREF_AXIS` preference feature + `user_preference_speech_act`, `hedged_like_is_preference`, `commitment_manner`, `habitual_perfect`, `clause_has_manner` | dropped |
| retrieval.rs:457 | `HYPERNYM_LATTICE` (40 hand-built domain clusters) + per-session domain machinery | Princeton WordNet 3.1, gated query expansion (below) |
| zero_overlap.rs:140 | `CONCEPT_TABLE` gazetteer (countries, cities, landmarks, foods, sports, pets), Wilkins/Zatocode/ConceptNet/SME bitmasks | dropped; module now holds only the list-free zero-overlap gate |
| retrieval.rs:319 | `HARD_CONSTRAINT_MARKERS` diet/allergy words, `constraint_object_tokens` | dropped |
| retrieval.rs:516 | `SPATIAL_WORDS` place nouns | dropped |
| retrieval.rs:3155 | `BLIP_STOPLIST` caption stoplist | dropped; caption tokens use the normal content filter |
| advanced/retroactive_backfill.rs:56, strata-store/src/types.rs:325 | `FAILURE_MARKERS` (crash, error, outage, ...) | `looks_like_failure` = explicit `failure` / `failure:<detail>` tag |
| retrieval.rs:424 | `TASK_UNDER_RULES_CUES`, `RECALL_NOT_TASK_CUES` | dropped (`task_under_rules` is false) |
| retrieval.rs:388 | `MULTI_HOP_CUES` | structural: >=3 content groups and >=2 digit groups or >=2 wh-slots |
| advanced/contradiction.rs:52 | `ANTONYM_PAIRS`, `NEGATION_PAIRS` | token/value divergence on an established subject |
| retrieval.rs:286 | `QUERY_FRAMING_WORDS` | corpus-derived: a token is "framing" when its corpus df is high (`corpus_frame`) |
| retrieval.rs | "chronological" / "order in which" phrase checks, `TIMELINE_CUES` | `is_event_sequence` = query names >=2 years or >=2 dates |
| retrieval.rs | `PREF_QUERY_CUES`, `CURRENT_STATE_CUES`, `ASSISTANT_ATTRIBUTION_CUES`, `USER_ATTRIBUTION_CUES`, `COLLECTIVE_CHAT_CUES`, `USER_INITIATIVE_CUES`, `SUMMARY_REQUEST_CUES`, `INFERENCE_CUES`, `WHEN_INTERROGATIVE_CUES`, `TEMPORAL_REST_CUES`, `CONTINUATIONS`, `CAUSAL_WORDS`, `Q_LEAD_WORDS`, `Q_REACTION_*`, `BACKCHANNEL_LEADS`, `DIRECTIVE_VERBS`, `PSEUDO_VERBS`, `CONDITIONAL_CLAUSES`, `DONT_PSEUDO_NEXT` | dropped; standing-policy detection is now closed-class grammar only (`utterance_is_standing_policy`: please? + always/never/don't/do not + base verb + noun phrase) |
| retrieval.rs | `expand_abbrev` (edu/fam/pic/uni...), `strip_tech_major` (html5/css3), `fold_two_factor` | dropped |
| retrieval.rs:877 | `irregular_lemma` (28 hand-picked verbs plus a `career` special case) | dropped: it was a curated subset with a non-verb hack, not a complete closed-class table. Follow-up: a complete irregular-verb table could be restored as morphology |
| neuroscience/importance_signals.rs:641,911,951 | arousal intensity keywords, positive/negative sentiment lexicons | empty by default; arousal = punctuation + capitalization shape; callers may pass their own keywords (`with_keywords`). Negation words kept (closed class) |
| advanced/intent.rs | keyword cues in intent patterns (error/bug/fix, refactor/rename, how to/tutorial, npm/upgrade, config file names) and the fixed per-intent tag vocabulary | intents scored from action types only; tags are what the actions named |
| trace/review.rs:179 | `SENSITIVE_TOPICS` topic gazetteer (salary, payment, auth, ...) | review gate holds a write on a sensitive node type, a tag naming a sensitive node type, or a credential-shaped value (format-based secret scan). **Behavior change:** prose that only mentions such topics no longer goes to review |
| vestige-mcp/src/tools/forgotten_lesson.rs:40,44 | `LESSON_TAGS` synonyms, `FIX_MARKERS` | one explicit `lesson` / `lesson:<detail>` tag |
| vestige-mcp/src/dashboard/handlers.rs | `infer_category` keyword rules; generic source-label list | category only from a tag equal to the category's canonical name (`PatternCategory::from_tag`); source labels rejected by shape only |
| vestige-mcp/src/tools/ghostlink.rs | per-law keyword "signals" on the six embedded laws | removed (they were display-only and never matched) |
| vestige-core/src/lib.rs, test fixtures | benchmark names in comments; `symbol:lme:*` fixture tags | rephrased / renamed `symbol:demo:*` |

## WordNet (replaces HYPERNYM_LATTICE only)

- Data: Princeton WordNet 3.1 `dict/data.noun`, from
  https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz, bundled verbatim at
  `crates/vestige-core/data/wordnet-3.1/data.noun`.
  sha256 `2cad22fe43461ee7ae61a564ae6a518c57445c8597e53542caddb5c26a6a5d94`
  (archive sha256 `3f7d8be8ef6ecc7167d39b10d66954ec734280b5bdcd57f7d9eafe429d11c22a`),
  checked by a unit test. Full file loaded, unfiltered, not pruned.
- License: WordNet 3.1 license, Princeton University ("WordNet 3.1 Copyright
  2011 by Princeton University. All rights reserved."), reproduced in
  `crates/vestige-core/data/wordnet-3.1/LICENSE` and in the file header.
- Use: query expansion with synonyms, direct hypernyms and direct hyponyms of
  every noun sense (`wordnet::expansion_lemmas`). Morphy's noun detachment
  suffix rules are used for plurals. Nothing is added to documents.
- Three controls, from published IR work:
  1. **Gate.** Expansion runs only when the query's own terms have zero
     lexical overlap with the indexed corpus
     (`zero_overlap::ounis_zero_overlap_gate`, after Ounis' pre-retrieval
     query-performance-prediction gating). Any overlap means no expansion; a
     test pins this. Unconditional WordNet expansion hurts effective queries
     (Voorhees, SIGIR 1994, "Query expansion using lexical-semantic relations").
  2. **Weight.** An expansion term weighs `hypernym_weight = 0.5` of an
     original term (1.0), deterministic (Fang, ACL 2008, P08-1017, "A
     Re-examination of Query Expansion Using Lexical Resources").
  3. **Corpus filter.** Only expansion lemmas with corpus df > 0 are kept;
     the rest are dropped (Gong et al. 2006, WordNet-based query expansion
     filtered by term co-occurrence in the collection).
- Not used for: ANTONYM_PAIRS, CONCEPT_TABLE, or any other list.

## Kept (closed-class grammar and protocol), each marked in code

| Where (this commit) | What | Why kept |
|---|---|---|
| retrieval.rs:262 `STOPWORDS`, :279, :415, :2054 | stopwords, pronouns/anaphors, question words | closed-class function words |
| retrieval.rs:1123, :1903 | temporal prepositions, determiners | closed class |
| retrieval.rs:647, :710, :748 (+ `ise` exceptions) | UK/US spelling normalization | owner-approved orthography normalization |
| retrieval.rs:891, algos_c2.rs inflection suffixes, wordnet.rs `NOUN_DETACH` | inflectional suffix rules | morphology, not vocabulary |
| retrieval.rs months/weekdays/ordinals/number words/relative-time words, rungs.rs, memory_packet.rs, prospective_memory.rs, intention_unified.rs | dates and relative time | closed class |
| importance_signals.rs negation words | negation | closed class |
| intake/entities.rs:65 | unit symbols (ns, ms, sec, ...) | closed class |
| codebase/anchor.rs:95, :396; advanced/git_records.rs:870, :920 | programming-language keywords/literals | language syntax, not topics |
| source_sync.rs:257 | GitHub's documented issue-closing keywords | GitHub protocol |
| server.rs:392, compact.rs:83, :445, ghostlink.rs:47, attach.rs:1353, session_unified.rs:97, curate_unified.rs:39, intention_graph.rs:34 | MCP/JSON-RPC/JSON-schema keys and tool names | wire protocol |
| security.rs secret prefixes (`xoxb-`, `ghp_`, ...), precedent.rs secret key names | credential formats | security redaction, not retrieval |
| reflex/pairing.rs rejection strings, repo_ingest.rs git error strings | literal tool output being parsed | protocol |
| causal_git_proof.rs, causal_walk.rs `/home/alex/.cargo/registry/...` | verbatim stack frames quoted from a public quinn/tokio bug report | replay data, never opened as a path |

## Known limits

- The scan is a heuristic: it fails on retired list names, word-literal arrays
  (>= 8 lowercase literals) without a closed-class/protocol mark, benchmark
  names in code, hardcoded user paths, and the retired phrase checks.
- Comments in `algos_c2.rs` still mention benchmark history; no condition in
  code tests a benchmark, question id, category, or speaker.
- `rungs.rs` and `benchmarks/real_engine/` were out of scope by instruction.
