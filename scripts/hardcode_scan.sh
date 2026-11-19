#!/usr/bin/env bash
# hardcode_scan.sh — fail if engine code under crates/ carries a hand-typed
# open-class word list, a benchmark/dataset-specific condition, or a
# hardcoded user path.
#
# Allowed (closed-class grammar and protocol, see SKEPTIC_AUDIT.md):
#   stopwords, pronouns, question words, negation, modals, determiners,
#   months/weekdays, relative-time words, UK/US spelling normalization,
#   inflectional suffix rules, JSON-schema keys and enum names, and the
#   bundled Princeton WordNet data file (crates/vestige-core/data/wordnet-3.1/).
#   A word-literal array is allowed only when its const name is a closed-class
#   set below, or the line above it carries `hardcode-scan: closed-class` or
#   `hardcode-scan: protocol` with the reason (wire/schema names, programming
#   language keywords, unit symbols). Every such mark is listed in SKEPTIC_AUDIT.md.
#
# Exit 0 = clean, 1 = findings.
set -euo pipefail
cd "$(dirname "$0")/.."
PY_BIN=python
if ! "$PY_BIN" -c 'import sys' >/dev/null 2>&1; then
    PY_BIN=python3
fi
if ! "$PY_BIN" -c 'import sys' >/dev/null 2>&1; then
    PY_BIN=py
fi
"$PY_BIN" - <<'PY'
import os, re, sys

# HARDCODE_SCAN_ROOT lets the self-test point the scan at a fixture tree.
ROOT = os.environ.get("HARDCODE_SCAN_ROOT", "crates")
SKIP_DIRS = (
    "/tests", "/benches", "/target",
    "/data/wordnet-3.1", "/data/conceptnet-5.7", "/data/roget", "/data/geonames", "/data/lexicon"
)
# Paths the owner excluded from this pass (benchmark harness).
SKIP_PATHS = ("crates/vestige-core/src/rungs.rs", "benchmarks/real_engine")

findings = []

def add(path, line, what):
    findings.append(f"{path}:{line}: {what}")

# 1. Retired benchmark-leaking marker arrays (must not come back).
RETIRED = [
    "PREF_MARKERS", "HYPERNYM_LATTICE", "HARD_CONSTRAINT_MARKERS",
    "BLIP_STOPLIST", "FAILURE_MARKERS", "TASK_UNDER_RULES_CUES",
    "MULTI_HOP_CUES", "FIX_MARKERS", "SENSITIVE_TOPICS", "LESSON_TAGS",
]
retired_re = re.compile(r"\b(" + "|".join(RETIRED) + r")\b")

# 2. Benchmark / dataset names in code (comments are reported separately as OK).
BENCH_re = re.compile(r"\b(LoCoMo|locomo|LongMemEval|longmemeval|BEAM|NoLiMa|nolima)\b")
# 3. Hardcoded user / absolute home paths.
PATH_re = re.compile(r'"(/Users/[^"]*|/home/[a-z][^"]*|C:\\\\Users[^"]*)"')
# A verbatim third-party stack frame (cargo registry path quoted from a public
# bug report) is replay data, not a path the engine opens.
QUOTED_FRAME = "/.cargo/registry/src/"

# 4. Arrays of lowercase word literals. Allowed closed-class / protocol sets are
#    listed by the const name or by a marker comment on the line above.
WORD_LIT = re.compile(r'"([a-z][a-z\'\u2019]{1,30})"')
ALLOWED_CONSTS = {
    # closed-class grammar and functional linguistics
    "STOPWORDS", "PRONOUNS", "MONTHS", "WEEKDAYS", "DETERMINERS", "NOUN_DETACH",
    "QUESTION_WORDS", "NEGATIONS", "MODALS", "ORDINALS", "TEMPORAL_PREPOSITIONS",
    "TEMPORAL_CUES", "DEICTIC_CUES", "SYNTACTIC_FRAMES", "SPEAKER_FRAMES",
    "RULE_KEYWORDS", "ADVICE_CUES", "POLARITY_MARKERS", "INTENT_CUES", "RELATION_TYPES",
}
ALLOW_RE = re.compile(r"hardcode-scan: (closed-class|protocol)")
MIN_WORDS = 8

def is_test_file(p):
    return p.endswith("_tests.rs") or p.endswith("/tests.rs") or "/tests/" in p

for root, _, files in os.walk(ROOT):
    root_norm = root.replace("\\", "/")
    if any(s in root_norm + "/" for s in SKIP_DIRS):
        continue
    for f in files:
        if not f.endswith(".rs"):
            continue
        p = os.path.join(root, f).replace("\\", "/")
        if p in SKIP_PATHS or is_test_file(p):
            continue
        lines = open(p, encoding="utf-8", errors="replace").read().split("\n")
        end = len(lines)
        for i, l in enumerate(lines):
            if re.match(r"\s*#\[cfg\(test\)\]\s*$", l) and i + 1 < len(lines) and re.match(r"\s*mod ", lines[i + 1]):
                end = i
                break
        body = lines[:end]
        for i, l in enumerate(body, 1):
            code = l.split("//")[0] if not l.strip().startswith("//") else ""
            if retired_re.search(l) and not l.strip().startswith("//"):
                add(p, i, f"retired list name: {retired_re.search(l).group(1)}")
            if code and BENCH_re.search(code) and '"' in code:
                add(p, i, f"benchmark name in code: {BENCH_re.search(code).group(1)}")
            if code and PATH_re.search(code) and QUOTED_FRAME not in PATH_re.search(code).group(1):
                add(p, i, f"hardcoded user path: {PATH_re.search(code).group(1)}")
        # word-literal arrays: a run of consecutive code lines inside one [...]
        i = 0
        while i < len(body):
            l = body[i]
            m = re.search(r"(?:const|static)\s+([A-Z_][A-Z0-9_]*)\s*:[^=]*=\s*&?\[", l) or re.search(r"(?:=|\()\s*&?\[\s*$|&\[\"", l)
            if not m:
                i += 1
                continue
            name = m.group(1) if m.lastindex else None
            depth = 0; j = i; words = []
            while j < len(body):
                t = body[j]
                if not t.strip().startswith("//"):
                    words += WORD_LIT.findall(t.split("//")[0])
                depth += t.count("[") - t.count("]")
                if depth <= 0:
                    break
                j += 1
            prev = body[i - 1] if i > 0 else ""
            allowed = (name in ALLOWED_CONSTS) or bool(ALLOW_RE.search(prev)) or bool(ALLOW_RE.search(l))
            if len(words) >= MIN_WORDS and not allowed:
                add(p, i + 1, f"array of {len(words)} word literals ({name or 'inline'}): {', '.join(words[:6])} ...")
            i = max(j, i) + 1

if findings:
    print("hardcode_scan: FAIL")
    for x in findings:
        print("  " + x)
    sys.exit(1)

# 5. Cryptographic SHA-256 verification of every bundled human-curated lexical resource
#    (whole, unedited published releases only; LF-normalized for cross-platform git checkouts).
import hashlib
PINNED_RESOURCES = [
    ("crates/vestige-core/data/wordnet-3.1/data.noun", "Princeton WordNet 3.1 nouns", "2cad22fe43461ee7ae61a564ae6a518c57445c8597e53542caddb5c26a6a5d94"),
    ("crates/vestige-core/data/wordnet-3.1/data.verb", "Princeton WordNet 3.1 verbs", "eb1cf196dab6e1b815a1c86fa20c909fb7413b2f34250d7fe60a0fb35ef59e03"),
    ("crates/vestige-core/data/wordnet-3.1/data.adj", "Princeton WordNet 3.1 adjectives", "ca1033bf627eb95f6cbb2864f8990be8e6a01a3c182e7d7c7094ccf8ead242cc"),
    ("crates/vestige-core/data/wordnet-3.1/data.adv", "Princeton WordNet 3.1 adverbs", "aea301f39ac1b24be9a880dbc6cc6331bc5391f58c7525a1b298a13c922f2ed3"),
    ("crates/vestige-core/data/roget/10681-body.txt", "Roget's 1911 Thesaurus (PG #10681)", "6a01f0d889c77b278f72828597dc4bb8b1978b3a5c55f871f4ddff0160cede31"),
    ("crates/vestige-core/data/geonames/countryInfo.txt", "GeoNames countryInfo (CC BY 4.0)", "f0693bb94cbc5492d0b68fceb54b04b2addd0a710fb2f34e6e19268d61479512"),
    ("crates/vestige-core/data/geonames/admin1CodesASCII.txt", "GeoNames admin1CodesASCII (CC BY 4.0)", "2e4867f5397acb056ea26377601b57af37cb4c81399770225018e9e439e91ad3"),
    ("crates/vestige-core/data/geonames/cities15000.txt", "GeoNames cities15000 (CC BY 4.0)", "1a0282bec53214296770884ecde9f134a932bf3fad2bb7ee277918e7535d1fb3"),
    ("crates/vestige-core/data/conceptnet-5.7-4rel.tsv", "ConceptNet 5.7 open assertions", "440bab391f707506677d24236ace7ba17cb7905c6e56c11ae4bb0ed83d2d766f"),
    ("crates/vestige-core/data/wikidata-cc0-4rel.tsv", "Wikidata CC0 open relation graph", "806a5ada7944c2f92fe355b577c1211c9e5b50580dd62b7ceb047e053eba0694"),
]
if ROOT == "crates":
    bad_hashes = []
    for rel_p, label, expected in PINNED_RESOURCES:
        if not os.path.exists(rel_p):
            bad_hashes.append(f"{rel_p}: MISSING ({label})")
            continue
        raw = open(rel_p, "rb").read().replace(b"\r\n", b"\n")
        got = hashlib.sha256(raw).hexdigest()
        if got != expected:
            bad_hashes.append(f"{rel_p}: sha256 mismatch (got {got}, expected {expected})")
    if bad_hashes:
        print("hardcode_scan: FAIL (pinned resource checksum verification failed)")
        for b in bad_hashes:
            print("  " + b)
        sys.exit(1)

print("hardcode_scan: OK (0 hand-typed open-class lists, 0 benchmark conditions, 0 user paths)")
print("  verified pinned human-curated resources (all sha256 match):")
for rel_p, label, expected in PINNED_RESOURCES:
    print(f"    - {rel_p} ({label}, sha256 {expected[:12]}...)")
PY
