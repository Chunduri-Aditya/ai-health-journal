#!/usr/bin/env python3
"""Retrieval strategies for the ablation harness: dense, BM25, and hybrid (RRF).

Why BM25 is implemented here rather than imported:

1. Zero new dependencies. `rank_bm25` and `sentence-transformers` are both
   absent from this environment, and the whole point of the local-first design
   is that a fresh clone runs the eval without a heavyweight install.
2. Backend independence. Chroma ships a sparse embedding function, but binding
   the ablation to it would make these numbers unreproducible on the Pinecone
   path. A pure-Python ranker scores any list of documents from any backend.
3. It is ~40 lines of a well-specified formula. Inspectable beats opaque when
   the output is a benchmark someone is meant to trust.

Fusion is Reciprocal Rank Fusion (Cormack et al.), which combines ranked lists
without needing the two scorers to share a scale. That matters here: Chroma's
`1/(1+distance)` and a BM25 score are not comparable magnitudes, so any
weighted-sum fusion would be silently dominated by whichever scorer happens to
have the larger range. RRF only reads ranks, so it sidesteps the problem.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

# Standard BM25 parameters. k1 controls term-frequency saturation, b controls
# length normalisation. These are the conventional defaults and are NOT tuned
# on the eval set -- tuning them here would make the ablation self-congratulatory.
BM25_K1 = 1.5
BM25_B = 0.75

# RRF's rank-smoothing constant. 60 is the value from the original paper.
RRF_K = 60

# A deliberately small stoplist. Journal entries are first-person narrative, so
# "i", "my", and "me" appear in essentially every document and carry no
# discriminative signal; leaving them in lets document length dominate BM25.
_STOPWORDS = frozenset("""
a an and are as at be been but by for from had has have i if in into is it its
me my of on or our so than that the their them they this to was we were what
when which who will with you your
""".split())

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords."""
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Doc:
    id: str
    text: str


class BM25:
    """Okapi BM25 over a fixed document set.

    Built once per corpus; `rank` may be called for many queries.
    """

    def __init__(self, docs: Sequence[Doc]) -> None:
        self._ids = [d.id for d in docs]
        self._tokens = [tokenize(d.text) for d in docs]
        self._lengths = [len(t) for t in self._tokens]
        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._tf: List[Counter] = [Counter(t) for t in self._tokens]

        # Document frequency per term.
        df: Counter = Counter()
        for toks in self._tokens:
            for term in set(toks):
                df[term] += 1
        self._df = df
        self._n = len(docs)

    def _idf(self, term: str) -> float:
        # Robertson/Spärck Jones IDF with the +1 smoothing that keeps the value
        # non-negative for terms appearing in more than half the corpus.
        df = self._df.get(term, 0)
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def rank(self, query: str) -> List[Tuple[str, float]]:
        """Return (doc_id, score) for every document, sorted best first."""
        q_terms = tokenize(query)
        scored: List[Tuple[str, float]] = []
        for idx, doc_id in enumerate(self._ids):
            tf = self._tf[idx]
            dl = self._lengths[idx]
            norm = BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / self._avgdl if self._avgdl else 0.0))
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                score += self._idf(term) * (f * (BM25_K1 + 1.0)) / (f + norm)
            scored.append((doc_id, score))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """Fuse several ranked id lists into one.

    score(d) = sum over rankers of 1 / (k + rank(d)), rank being 1-based.
    A document missing from a ranker simply contributes nothing from it.
    """
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    out = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return out


# ── Strategies ────────────────────────────────────────────────────────────────
# Each strategy is (query, top_k) -> list of doc ids, best first.
Strategy = Callable[[str, int], List[str]]


def dense_strategy(store, namespace: str) -> Strategy:
    """Current production behaviour: single dense vector search."""

    def run(query: str, top_k: int) -> List[str]:
        return [h.id for h in store.query(query, top_k=top_k, namespace=namespace)]

    return run


def bm25_strategy(bm25: BM25) -> Strategy:
    """Lexical only. Included as a control, not as a proposal."""

    def run(query: str, top_k: int) -> List[str]:
        return [doc_id for doc_id, _ in bm25.rank(query)[:top_k]]

    return run


# ── Alternative embedders ─────────────────────────────────────────────────────
# Chroma's default is ONNX all-MiniLM-L6-v2 (384-dim). The ablation showed a
# target document sitting outside the top 20 for a semantically clear query,
# which is a retrieval-stage failure no reranker can repair, so the embedder
# itself becomes the variable to test.
#
# nomic-embed-text is used rather than a sentence-transformers model because it
# runs on the Ollama daemon this project already requires, keeping the eval
# local and installable-free. It is also trained for ASYMMETRIC retrieval: the
# model card specifies "search_document: " and "search_query: " prefixes, so
# prefixing is measured as its own variable rather than assumed to help.

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_EMBED_LEGACY_URL = "http://localhost:11434/api/embeddings"


def ollama_embed(texts: Sequence[str], model: str, *, timeout: int = 120) -> List[List[float]]:
    """Embed a batch of texts via the local Ollama daemon.

    Tries the batch /api/embed endpoint first and falls back to the older
    per-item /api/embeddings, so this works across Ollama versions.
    """
    import requests

    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": model, "input": list(texts)},
            timeout=timeout,
        )
        if response.status_code == 200:
            payload = response.json()
            if payload.get("embeddings"):
                return payload["embeddings"]
    except requests.exceptions.RequestException:
        pass

    out: List[List[float]] = []
    for text in texts:
        response = requests.post(
            OLLAMA_EMBED_LEGACY_URL,
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        response.raise_for_status()
        out.append(response.json()["embedding"])
    return out


def _cosine_ranker(doc_ids: Sequence[str], doc_vecs: Sequence[Sequence[float]]):
    """Return a function mapping a query vector to a ranked list of doc ids."""
    import numpy as np

    matrix = np.asarray(doc_vecs, dtype="float64")
    # Normalise once so scoring is a single matrix-vector product.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    def rank(query_vec: Sequence[float]) -> List[str]:
        q = np.asarray(query_vec, dtype="float64")
        q_norm = np.linalg.norm(q) or 1.0
        sims = matrix @ (q / q_norm)
        order = np.argsort(-sims)
        return [doc_ids[i] for i in order]

    return rank


def ollama_embedder_strategy(
    documents: Sequence[Doc],
    *,
    model: str = "nomic-embed-text",
    doc_prefix: str = "",
    query_prefix: str = "",
) -> Strategy:
    """Dense retrieval using an Ollama-hosted embedder, ranked by cosine.

    Deliberately bypasses Chroma. The corpus is small enough that an exact
    cosine scan is instant, and it removes the vector store as a confound so the
    only thing changing between this and the dense baseline is the embedder.
    """
    doc_ids = [d.id for d in documents]
    doc_vecs = ollama_embed([doc_prefix + d.text for d in documents], model)
    rank = _cosine_ranker(doc_ids, doc_vecs)

    def run(query: str, top_k: int) -> List[str]:
        query_vec = ollama_embed([query_prefix + query], model)[0]
        return rank(query_vec)[:top_k]

    return run


def with_valence_rerank(
    base: Strategy,
    documents: Sequence[Doc],
    *,
    depth: int = 10,
) -> Strategy:
    """Wrap any strategy with the demotion-only valence reorder.

    Generic over the base ranker so the valence question can be asked
    independently of which embedder produced the candidates. Composition matters
    here: a reranker that helps a weak retriever may be redundant or harmful on
    a strong one, and that is only visible by testing both.
    """
    import valence as _valence

    text_by_id = {d.id: d.text for d in documents}

    def run(query: str, top_k: int) -> List[str]:
        candidates = base(query, depth)
        order = _valence.partition_by_agreement(
            query, [text_by_id.get(i, "") for i in candidates]
        )
        return [candidates[i] for i in order][:top_k]

    return run


def valence_aware_strategy(store, namespace: str, *, depth: int = 10) -> Strategy:
    """Dense retrieval, then reorder so valence-matching entries rank first.

    Over-fetch to `depth`, partition the candidates by whether their emotional
    valence agrees with the query's, and keep the dense ordering inside each
    partition. Rank-only and parameter free, so there is no weight that could be
    quietly fitted to the eval corpus.

    Degrades to plain dense in the two cases where valence carries no signal:
    a neutral query, and a candidate set with no valence match. It can therefore
    reorder results but never remove them, which matters because retrieving
    nothing is worse than retrieving something imperfect.
    """
    import valence as _valence

    def run(query: str, top_k: int) -> List[str]:
        hits = store.query(query, top_k=depth, namespace=namespace)
        order = _valence.partition_by_agreement(query, [h.text for h in hits])
        return [hits[i].id for i in order][:top_k]

    return run


def hybrid_strategy(store, namespace: str, bm25: BM25, *, depth: int = 10) -> Strategy:
    """Dense + BM25 fused with RRF.

    `depth` is how far down each list is considered before fusing. It must
    exceed top_k or fusion has nothing to reorder.
    """

    def run(query: str, top_k: int) -> List[str]:
        dense_ids = [h.id for h in store.query(query, top_k=depth, namespace=namespace)]
        lexical_ids = [doc_id for doc_id, _ in bm25.rank(query)[:depth]]
        fused = reciprocal_rank_fusion([dense_ids, lexical_ids])
        return [doc_id for doc_id, _ in fused[:top_k]]

    return run
