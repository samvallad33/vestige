"""Real BM25 and pinned local BGE baselines, with no fallback substitutes."""
from __future__ import annotations

from functools import lru_cache
import threading

# Official BAAI/bge-small-en-v1.5 revision. Constructor requires the pinned sha,
# rather than allowing a mutable branch or another embedding model.
BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
_MODEL_LOCK = threading.Lock()


def tokens(text: str) -> list[str]:
    # Lexical baseline tokenization only; no extraction, answer synthesis or
    # inference. Preserve whitespace-separated tokens after lowercasing.
    return text.lower().split()


class BM25Memory:
    def __init__(self, history: list[str]):
        from rank_bm25 import BM25Okapi
        self.history = list(history)
        self.index = BM25Okapi([tokens(x) for x in history]) if history else None

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.index:
            return []
        scores = self.index.get_scores(tokens(query))
        ids = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
        return [{"id": f"session-{i}", "content": self.history[i], "score": float(scores[i])} for i in ids]


@lru_cache(maxsize=1)
def _model(revision: str):
    if revision != BGE_REVISION:
        raise ValueError("BGE revision must match campaign pin")
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1)
    return SentenceTransformer(BGE_MODEL, revision=revision, device="cpu", trust_remote_code=False)


class BGEMemory:
    def __init__(self, history: list[str], revision: str = BGE_REVISION):
        self.history = list(history)
        self.revision = revision
        # One immutable CPU model shared across workers; corpus vectors remain
        # private to this memory instance. There is no BM25 or random fallback.
        with _MODEL_LOCK:
            self.model = _model(revision)
            # Split by the pinned tokenizer's capacity, with original character
            # offsets. Every source byte remains retrievable; search returns the
            # complete original record whose best chunk matched. No semantic
            # rules, answer metadata, or task-dependent boundaries are used.
            chunks, self.chunk_sources = [], []
            capacity = int(self.model.max_seq_length) - 32
            tokenizer = self.model.tokenizer
            for i, text in enumerate(history):
                offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                                    truncation=False)["offset_mapping"]
                if len(offsets) <= capacity:
                    chunks.append(text)
                    self.chunk_sources.append(i)
                    continue
                for start in range(0,len(offsets),capacity):
                    end=min(start+capacity,len(offsets))
                    left=0 if start==0 else offsets[start][0]
                    right=len(text) if end==len(offsets) else offsets[end][0]
                    chunk=text[left:right]
                    if len(tokenizer.encode(chunk,add_special_tokens=True))>self.model.max_seq_length:
                        raise RuntimeError("BGE chunk exceeds pinned tokenizer capacity")
                    chunks.append(chunk)
                    self.chunk_sources.append(i)
            self.chunk_count=len(chunks)
            self.vectors = self.model.encode(chunks, normalize_embeddings=True, show_progress_bar=False) if chunks else None

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.vectors is None:
            return []
        with _MODEL_LOCK:
            vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        chunk_scores = self.vectors @ vector
        scores = [float('-inf')]*len(self.history)
        for source,score in zip(self.chunk_sources,chunk_scores):
            scores[source]=max(scores[source],float(score))
        ids = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]
        return [{"id": f"session-{i}", "content": self.history[i], "score": float(scores[i])} for i in ids]
