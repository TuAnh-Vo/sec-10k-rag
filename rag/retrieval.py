"""Hybrid retrieval: dense (FAISS) + BM25, fused with Reciprocal Rank Fusion.

The selected configuration is row 2 of the results table: dense + BM25 + RRF.
MMR and the cross-encoder are kept behind flags so the ablation table can be
reproduced through ONE code path (scripts/evaluate.py).

Dense search detail: the notebook used LangChain's FAISS wrapper, which builds an
IndexFlatL2. Here we use IndexFlatIP (inner product) directly. Because every
vector has length 1:

    ||a - b||^2 = 2 - 2 * cos(a, b)

so "smallest L2 distance" and "largest inner product" pick the SAME chunks in the
SAME order. Both are exact brute-force search over all 5,089 chunks.
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

from .config import (BGE_QUERY_PREFIX, CANDIDATE_K, EMBEDDING_MODEL_NAME,
                     MMR_LAMBDA, RERANKER_NAME, RRF_K)


# --------------------------------------------------------------------------- #
# BM25                                                                         #
# --------------------------------------------------------------------------- #
class BM25:
    """Sparse lexical scorer.
      1. rare words matter more        -> idf
      2. repetition saturates          -> k1 caps term-frequency gain
      3. long documents get discounted -> b normalises by length
    """

    def __init__(self, ids, texts, k1=1.5, b=0.75):
        self.k1, self.b, self.ids = k1, b, list(ids)
        self.docs = [self._tok(t) for t in texts]
        self.lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.lens) / len(self.lens)

        self.index = defaultdict(list)          # term -> [(doc_index, tf)]
        df = Counter()
        for i, doc in enumerate(self.docs):
            for term, tf in Counter(doc).items():
                self.index[term].append((i, tf))
                df[term] += 1

        n = len(self.docs)
        # floor stops very common words ("company", "risk") scoring negative
        self.idf = {t: max(math.log((n - m + 0.5) / (m + 0.5) + 1), 1e-6)
                    for t, m in df.items()}

    @staticmethod
    def _tok(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query, k=50):
        scores = defaultdict(float)
        for term in self._tok(query):
            if term not in self.index:
                continue
            idf = self.idf[term]
            for i, tf in self.index[term]:
                dl = 1 - self.b + self.b * self.lens[i] / self.avgdl
                scores[i] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * dl)
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [self.ids[i] for i, _ in top]


# --------------------------------------------------------------------------- #
# Fusion and diversity                                                         #
# --------------------------------------------------------------------------- #
def rrf_scored(rank_lists, k=RRF_K):
    """Reciprocal Rank Fusion: combine lists by POSITION, not by raw score."""
    scores = defaultdict(float)
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return scores


def rrf(rank_lists, k=RRF_K):
    s = rrf_scored(rank_lists, k=k)
    return sorted(s, key=lambda d: -s[d])


def mmr_hybrid(rel_scores, cand_ids, chunk_vec, lam=MMR_LAMBDA, top_n=10):
    """lam * relevance(fused RRF, min-max scaled) - (1-lam) * redundancy(cosine)."""
    cand_ids = [c for c in cand_ids if c in chunk_vec]
    if not cand_ids:
        return []
    m = np.stack([chunk_vec[c] for c in cand_ids])
    rel = np.array([rel_scores[c] for c in cand_ids], dtype="float32")
    span = rel.max() - rel.min()
    rel = (rel - rel.min()) / span if span > 1e-9 else np.ones_like(rel)

    selected, remaining = [], list(range(len(cand_ids)))
    while remaining and len(selected) < top_n:
        if not selected:
            best = max(remaining, key=lambda i: rel[i])
        else:
            red = (m[remaining] @ m[selected].T).max(axis=1)
            best = remaining[int(np.argmax(lam * rel[remaining] - (1 - lam) * red))]
        selected.append(best)
        remaining.remove(best)
    return [cand_ids[i] for i in selected]


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
def load_embedder(device="cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)


def embed_passages(embedder, texts, batch_size=32):
    """Passages go in WITHOUT the query prefix (bge-v1.5 is asymmetric)."""
    return embedder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True).astype("float32")


# --------------------------------------------------------------------------- #
# The retriever                                                                #
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self, chunks, vectors, embedder, reranker=None):
        import faiss

        self.ids = [c["chunk_id"] for c in chunks]
        self.text = {c["chunk_id"]: c["text"] for c in chunks}
        self.meta = {c["chunk_id"]: c["meta"] for c in chunks}
        self.chunk_vec = dict(zip(self.ids, vectors))
        self.embedder = embedder
        self.reranker = reranker

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.bm25 = BM25(self.ids, [c["text"] for c in chunks])

        # Same guard as the notebook: all stores describe the same corpus.
        assert self.index.ntotal == len(self.ids) == len(self.text) == len(self.bm25.ids)

    def dense(self, query, k=CANDIDATE_K):
        q = self.embedder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")
        _, idx = self.index.search(q, k)
        return [self.ids[i] for i in idx[0] if i >= 0]

    def rerank(self, query, cand_ids, top_n=10):
        if not cand_ids:
            return []
        scores = self.reranker.predict([(query, self.text[c]) for c in cand_ids],
                                       batch_size=32, show_progress_bar=False)
        return [cand_ids[i] for i in np.argsort(-scores)[:top_n]]

    def search(self, query, *, use_bm25=True, use_mmr=False, use_rerank=False,
               candidate_k=CANDIDATE_K, rerank_n=30, lam=MMR_LAMBDA, top_n=10):
        """ONE code path for every ablation row (same as the notebook's search())."""
        lists = [self.dense(query, k=candidate_k)]
        if use_bm25:
            lists.append(self.bm25.search(query, k=candidate_k))

        scores = rrf_scored(lists)
        cands = sorted(scores, key=lambda d: -scores[d])[:candidate_k]

        n_after = rerank_n if use_rerank else top_n
        cands = (mmr_hybrid(scores, cands, self.chunk_vec, lam=lam, top_n=n_after)
                 if use_mmr else cands[:n_after])
        if use_rerank:
            if self.reranker is None:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(RERANKER_NAME, max_length=512,
                                             device=str(self.embedder.device))
            cands = self.rerank(query, cands, top_n=top_n)
        return cands[:top_n]
