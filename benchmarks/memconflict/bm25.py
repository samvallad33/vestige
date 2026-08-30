"""Okapi BM25 control retriever. Pure stdlib, no dependencies.

This is a MANDATORY control, not an optional extra. MemDelta (arXiv:2606.29914)
found that agent memory systems routinely fail to beat controlled baselines --
agent self-memory scored 42% against 47% for basic retrieval on LongMemEval-S.
A memory system that does not beat BM25 on the same corpus, with the same
reader and the same judge, has not earned its complexity.

Standard BM25 with k1=1.5, b=0.75. Tokenisation is intentionally simple and
shared with nothing else in the harness, so the control stays a control.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

K1 = 1.5
B = 0.75
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.documents = list(documents)
        self.doc_tokens = [tokenize(d) for d in self.documents]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.n = len(self.documents)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

        self.tf: List[Counter] = [Counter(t) for t in self.doc_tokens]
        df: Counter = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] += 1
        self.idf: Dict[str, float] = {}
        for term, freq in df.items():
            # Robertson/Sparck-Jones idf with +1 smoothing (always positive).
            self.idf[term] = math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))

    def score(self, query: str, index: int) -> float:
        if not self.n:
            return 0.0
        q_terms = tokenize(query)
        tf = self.tf[index]
        dl = self.doc_len[index] or 1
        total = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = self.idf.get(term, 0.0)
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
            total += idf * (f * (self.k1 + 1.0)) / denom
        return total

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(self.n)]
        scored = [(i, s) for i, s in scored if s > 0.0]
        # Deterministic: sort by score desc, then by document index asc.
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
