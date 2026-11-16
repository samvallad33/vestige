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
python3 - <<'PY'
import os, re, sys

# HARDCODE_SCAN_ROOT lets the self-test point the scan at a fixture tree.
ROOT = os.environ.get("HARDCODE_SCAN_ROOT", "crates")
SKIP_DIRS = ("/tests", "/benches", "/target", "/data/wordnet-3.1")
# Paths the owner excluded from this pass (benchmark harness).
SKIP_PATHS = ("crates/vestige-core/src/rungs.rs", "benchmarks/real_engine")

findings = []

def add(path, line, what):
    findings.append(f"{path}:{line}: {what}")

# 1. Retired list names must not come back (as a definition or a use).
RETIRED = [
    "PREF_MARKERS", "HYPERNYM_LATTICE", "CONCEPT_TABLE", "HARD_CONSTRAINT_MARKERS",
    "SPATIAL_WORDS", "BLIP_STOPLIST", "FAILURE_MARKERS", "TASK_UNDER_RULES_CUES",
    "MULTI_HOP_CUES", "ANTONYM_PAIRS", "QUERY_FRAMING_WORDS", "PREF_AXIS",
    "FIX_MARKERS", "SENSITIVE_TOPICS", "LESSON_TAGS",
]
retired_re = re.compile(r"\b(" + "|".join(RETIRED) + r")\b")

# 2. Benchmark / dataset names in code (comments are reported separately as OK).
BENCH_re = re.compile(r"\b(LoCoMo|locomo|LongMemEval|longmemeval|BEAM|NoLiMa|nolima)\b")
# 3. Hardcoded user / absolute home paths.
PATH_re = re.compile(r'"(/Users/[^"]*|/home/[a-z][^"]*|C:\\\\Users[^"]*)"')
# A verbatim third-party stack frame (cargo registry path quoted from a public
# bug report) is replay data, not a path the engine opens.
QUOTED_FRAME = "/.cargo/registry/src/"
# 4. Natural-language phrase checks that were retired.
PHRASE_re = re.compile(r'contains\("(chronological|order in which)')

# 5. Arrays of lowercase word literals. Allowed closed-class / protocol sets are
#    listed by the const name or by a marker comment on the line above.
WORD_LIT = re.compile(r'"([a-z][a-z\'\u2019]{1,30})"')
ALLOWED_CONSTS = {
    # closed-class grammar
    "STOPWORDS", "PRONOUNS", "MONTHS", "WEEKDAYS", "DETERMINERS", "NOUN_DETACH",
    "QUESTION_WORDS", "NEGATIONS", "MODALS",
}
ALLOW_RE = re.compile(r"hardcode-scan: (closed-class|protocol)")
MIN_WORDS = 8

def is_test_file(p):
    return p.endswith("_tests.rs") or p.endswith("/tests.rs") or "/tests/" in p

for root, _, files in os.walk(ROOT):
    if any(s in root + "/" for s in SKIP_DIRS):
        continue
    for f in files:
        if not f.endswith(".rs"):
            continue
        p = os.path.join(root, f)
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
            if code and PHRASE_re.search(code):
                add(p, i, "retired phrase check")
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
print("hardcode_scan: OK (0 hand-typed open-class lists, 0 benchmark conditions, 0 user paths)")
print("  allowed data file: crates/vestige-core/data/wordnet-3.1/data.noun (Princeton WordNet 3.1, sha256 2cad22fe43461ee7ae61a564ae6a518c57445c8597e53542caddb5c26a6a5d94)")
PY
