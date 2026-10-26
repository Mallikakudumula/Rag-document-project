"""Step 3: find the chunks most relevant to a question.
- BM25: classic keyword ranking (fast, offline, great for exact terms like 'MFA').
- Embeddings (optional): semantic search ('refund window' ~ 'return period').
- Hybrid: merge both rankings with Reciprocal Rank Fusion (RRF)."""
import math
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk
from .textutil import tokenize


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.n, 1)
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for i, doc in enumerate(self.docs):
            s, dl = 0.0, len(doc)
            for t in set(query):
                f = self.tf[i].get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float          # BM25 score (used for the 'not found' guardrail)
    rank: int
    coverage: float       # share of question terms present in this chunk


class SentenceTransformerEmbedder:
    """Optional semantic search. pip install sentence-transformers numpy"""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True)


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across rankings."""
    fused: Counter = Counter()
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + pos + 1)
    return [i for i, _ in fused.most_common()]


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder=None, embeddings=None):
        self.chunks = chunks
        self.tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25(self.tokens)
        self.embedder, self.embeddings = embedder, embeddings

    def search(self, question: str, k: int = 4) -> list[Hit]:
        q = tokenize(question)
        if not q or not self.chunks:
            return []
        bm = self.bm25.scores(q)
        order = sorted(range(len(bm)), key=lambda i: bm[i], reverse=True)
        if self.embedder is not None and self.embeddings is not None:
            sims = (self.embeddings @ self.embedder.embed([question])[0]).tolist()
            sem = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
            order = rrf([order, sem])
        qset = set(q)
        hits = []
        for rank, i in enumerate(order[:k], 1):
            cov = len(qset & set(self.tokens[i])) / len(qset)
            hits.append(Hit(self.chunks[i], bm[i], rank, cov))
        return hits
