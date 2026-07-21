"""Lexical (BM25) and dense-cosine baseline over the Silent Rotation corpus.

WHY THIS EXISTS
---------------
The obvious objection to this benchmark is that the `rag` arm is dense-cosine
only, while the answer being hunted is a rare literal identifier (`k_nashira`),
which is supposed to be exact-match's home ground. If BM25 trivially finds the
causal memory, then the benchmark is measuring a baseline gap rather than
anything about memory architecture.

So we ran BM25. It does not find it either. On the symptom-shaped queries every
agent opens with, BM25 and dense cosine both bury the causal memory at rank 7 of
8, and on the reformulated queries that actually worked, BM25 is one to four
ranks WORSE than dense.

The mechanism: the planted decoys share the target's exact vocabulary ("signing
key", "kid", "keyring", "rotation", "active"), so lexical scoring rewards them at
least as much as the truth. What separates the right key from the wrong one is
which key is CURRENT, and recency carries no lexical signal.

There is also a practical ceiling on exact match here. The live key is randomized
per trial from a 50-key keyring and appears in no file the agent can read, so you
can only type it into a query after you already know the answer. The exact-key
probe below is included to show that even then it only reaches rank 3.

USAGE
-----
    python3 bm25_baseline.py <corpus.json> [--no-dense]

<corpus.json> is a `vestige export --format json` dump of the seeded trial store,
i.e. the exact same source the competitor arms read (see
`harness/agent/runner.py::_rag_load_facts`). The retrievable corpus excludes the
PRODUCTION OUTAGE memory, matching the arms' behaviour: a retriever indexes prior
history, not the live error it is being asked about.

Dense scoring needs ollama on localhost:11434 with nomic-embed-text, the same
embedder the `rag` arm uses. Pass --no-dense to run BM25 only.
"""

import json
import math
import re
import sys
import urllib.request
from collections import Counter

EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA = "http://localhost:11434/api/embeddings"

# The queries below are VERBATIM from the trial-1 agent transcripts in
# results/trial-1/. They are not authored for this script. Each is labelled with
# the agent that issued it so it can be traced back.
QUERIES = [
    ("rag a0, first query", "end-to-end charge test failing identity gateway ledger session token verify"),
    ("zep a0, first query", "end-to-end charge test failing session token identity gateway ledger"),
    ("zep a0, third query", "default key kid keyring shipped platform active key"),
    ("mem0 a0, second query", "which key id active signing key k_antares externalize secrets refactor blank config"),
    ("supermemory a0, first", "no active signing key selected signing key rotation identity-service keyring"),
]


def load_corpus(path):
    """Mirror runner.py::_rag_load_facts. content/body only, outage row dropped."""
    raw = json.load(open(path))
    rows = raw if isinstance(raw, list) else raw.get("memories", raw.get("nodes", []))
    docs = []
    for m in rows:
        body = m.get("content") or m.get("body") or ""
        if not body or "PRODUCTION OUTAGE" in body:
            continue
        docs.append({"body": body, "is_cause": "keyring_rotation" in (m.get("tags") or [])})
    return rows, docs


def tokenize(s):
    return re.findall(r"[a-z0-9_]+", s.lower())


class BM25:
    """Okapi BM25, k1=1.5, b=0.75. Standard parameters, no tuning."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus = [tokenize(d["body"]) for d in docs]
        self.n = len(self.corpus)
        self.avgdl = sum(len(c) for c in self.corpus) / self.n
        self.df = Counter()
        for c in self.corpus:
            for t in set(c):
                self.df[t] += 1

    def idf(self, t):
        return math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))

    def score(self, query, i):
        c = self.corpus[i]
        freq = Counter(c)
        dl = len(c)
        total = 0.0
        for t in tokenize(query):
            if t not in self.df:
                continue
            total += self.idf(t) * (freq[t] * (self.k1 + 1)) / (
                freq[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return total


def embed(text):
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps({"model": EMBED_MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-12)


def rank_of_cause(order, docs):
    return [k for k, i in enumerate(order) if docs[i]["is_cause"]][0] + 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    path = sys.argv[1]
    use_dense = "--no-dense" not in sys.argv

    rows, docs = load_corpus(path)
    n = len(docs)
    if n == 0:
        print(f"  ERROR: no retrievable memories found in {path}")
        print()
        print("  This is almost certainly the wrong input file. You want the")
        print("  memory-layer export, named corpus-export.json:")
        print()
        print("      python3 tests/bm25_baseline.py results/runA-trial-1/corpus-export.json --no-dense")
        print()
        print("  prod-corpus.json, which sits in every results/ directory, is a")
        print("  DIFFERENT artifact: it is the production replay oracle, not a corpus.")
        sys.exit(2)
    causes = sum(d["is_cause"] for d in docs)
    keys = sorted({w.strip(".,;:\"'()") for d in docs for w in d["body"].split() if w.startswith("k_")})

    print(f"  seeded memories in the store : {len(rows)}")
    print(f"  retrievable competitor corpus: {n}   (outage row excluded, as runner.py does)")
    print(f"  causal memories present      : {causes}")
    print(f"  key identifiers in corpus    : {keys}")
    if causes != 1:
        print("  WARNING: expected exactly one causal memory; the ranks below are not meaningful.")

    bm = BM25(docs)
    dense_vecs = [embed(d["body"]) for d in docs] if use_dense else None

    print()
    print(f"  rank of the causal memory out of {n}, lower is better")
    print(f"  {'query (verbatim from a transcript)':<34}{'dense':<9}{'bm25'}")
    print("  " + "-" * 52)
    for label, q in QUERIES:
        bm_rank = rank_of_cause(sorted(range(n), key=lambda i: -bm.score(q, i)), docs)
        if use_dense:
            qv = embed(q)
            d_rank = rank_of_cause(sorted(range(n), key=lambda i: -cosine(qv, dense_vecs[i])), docs)
            print(f"  {label:<34}#{d_rank:<8}#{bm_rank}")
        else:
            print(f"  {label:<34}{'-':<9}#{bm_rank}")

    # The exact-key probe. Only possible once you already know the answer, which
    # is why it is reported separately and not as evidence that lexical works.
    correct = [k for k in keys if k not in ("k_antares", "k_vela", "k_sirius", "k_spica")]
    if correct:
        probe = f"{correct[0]} active signing key"
        r = rank_of_cause(sorted(range(n), key=lambda i: -bm.score(probe, i)), docs)
        print()
        print(f"  exact-key probe {probe!r}: BM25 rank #{r} of {n}")
        print("  (impossible in practice: the live key is randomized per trial and appears")
        print("   in no file the agent can read, so this query requires knowing the answer)")


if __name__ == "__main__":
    main()
