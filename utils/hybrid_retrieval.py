"""
utils/hybrid_retrieval.py
Hybrid retrieval primitives used by the RAG engine:

    BM25 (sparse) ─┐
                   ├── RRF fusion ── optional cross-encoder rerank ── top-k
    Dense    ──────┘

Each component degrades gracefully:
  - if ``rank_bm25`` is missing → BM25 layer is skipped, only dense is used
  - if ``sentence_transformers`` cross-encoder fails to load → no rerank,
    fused list is returned as-is

The module exposes small, testable building blocks (`reciprocal_rank_fusion`,
`BM25Index`, `CrossEncoderReranker`) plus a high-level orchestrator
(`hybrid_search`) that the RAG engine can call without knowing the details.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Optional dependencies — never required for import-time correctness.
try:  # pragma: no cover — exercised via tests indirectly
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore[assignment]

try:  # pragma: no cover
    from sentence_transformers import CrossEncoder  # type: ignore
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokenizer. Stable across platforms; no NLTK needed."""
    return _TOKEN_RE.findall((text or "").lower())


# ---------------------------------------------------------------------------
# RRF — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    Each input ranking is a sequence of document IDs ordered best-first.
    Returns a list of ``(doc_id, fused_score)`` sorted by descending score.
    The ``k`` smoothing constant defaults to 60, the value used in the
    original paper.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# BM25 sparse index
# ---------------------------------------------------------------------------

@dataclass
class BM25Index:
    """Thin wrapper over rank_bm25.BM25Okapi keyed by stable doc IDs.

    The index is intentionally rebuildable from scratch — we don't try to do
    incremental updates, since rank_bm25's IDF is corpus-wide. ``add()``
    therefore appends to internal lists; call :meth:`build` once before
    querying.
    """

    ids: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    _tokens: list[list[str]] = field(default_factory=list)
    _bm25: Any = None

    def add(self, doc_id: str, text: str) -> None:
        """Queue a document for indexing. Call :meth:`build` afterwards."""
        self.ids.append(doc_id)
        self.docs.append(text)
        self._tokens.append(tokenize(text))
        self._bm25 = None  # Invalidate

    def build(self) -> None:
        """Materialise the BM25 index. No-op when rank_bm25 is unavailable."""
        if BM25Okapi is None:
            logger.info("rank_bm25 not installed — BM25 layer disabled.")
            self._bm25 = None
            return
        if not self._tokens:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._tokens)

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and bool(self.ids)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return up to *top_k* ``(doc_id, bm25_score)`` pairs, best first."""
        if not self.is_ready:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(
            zip(self.ids, scores), key=lambda x: x[1], reverse=True,
        )
        # Strip zero-score tail to keep the ranking honest.
        return [(d, s) for d, s in ranked[:top_k] if s > 0]

    def __len__(self) -> int:
        return len(self.ids)


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Lazy wrapper around a sentence-transformers CrossEncoder.

    Loads the model on first use and degrades to a no-op when the library or
    the network is unavailable.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Any = None
        self._failed = False

    def _ensure_loaded(self) -> bool:
        if self._model is not None or self._failed:
            return self._model is not None
        if CrossEncoder is None:
            self._failed = True
            return False
        try:
            self._model = CrossEncoder(self.model_name)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder load failed: %s — rerank disabled.", exc)
            self._failed = True
            return False

    @property
    def available(self) -> bool:
        if self._failed:
            return False
        return CrossEncoder is not None

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Score and reorder ``(doc_id, text)`` candidates.

        Returns ``[(doc_id, ce_score), ...]`` sorted best-first. If the
        cross-encoder cannot be loaded, returns the input order with
        placeholder scores.
        """
        if not candidates:
            return []
        if not self._ensure_loaded():
            # Stable identity rerank — preserves input ordering.
            return [(doc_id, 1.0 / (i + 1)) for i, (doc_id, _) in enumerate(candidates)][:top_k]

        try:
            pairs = [[query, text] for _, text in candidates]
            scores = self._model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder predict failed: %s — falling back.", exc)
            return [(doc_id, 1.0 / (i + 1)) for i, (doc_id, _) in enumerate(candidates)][:top_k]

        scored = [
            (candidates[i][0], float(scores[i])) for i in range(len(candidates))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    dense_results: list[tuple[str, str]],
    bm25: BM25Index | None,
    *,
    top_k: int = 5,
    fusion_candidates: int = 50,
    reranker: CrossEncoderReranker | None = None,
) -> list[tuple[str, float, str]]:
    """Run BM25 + dense fusion (+ optional rerank) over a candidate pool.

    Parameters
    ----------
    query
        The user query string.
    dense_results
        Top-N dense-retrieval candidates as ``(doc_id, text)`` pairs.
        The order encodes the dense ranking.
    bm25
        Optional populated :class:`BM25Index`. ``None`` or empty index
        ⇒ dense-only mode.
    top_k
        Final number of results to return.
    fusion_candidates
        How many candidates to take from each layer before fusion.
    reranker
        Optional :class:`CrossEncoderReranker`. ``None`` ⇒ no rerank.

    Returns
    -------
    List of ``(doc_id, score, text)`` ordered best-first.
    """
    # Build the dense ranking (just IDs for fusion)
    dense_ranking: list[str] = [doc_id for doc_id, _ in dense_results[:fusion_candidates]]

    # Build the sparse ranking, if available
    sparse_ranking: list[str] = []
    if bm25 is not None and bm25.is_ready:
        sparse_ranking = [doc_id for doc_id, _ in bm25.search(query, top_k=fusion_candidates)]

    # Fuse
    if sparse_ranking:
        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
    else:
        fused = [(d, 1.0 / (i + 1)) for i, d in enumerate(dense_ranking)]

    # Map IDs back to text. Prefer dense's text since it's freshest from
    # the embedding store; fall back to BM25's docs when missing.
    text_by_id: dict[str, str] = {doc_id: text for doc_id, text in dense_results}
    if bm25 is not None:
        for did, txt in zip(bm25.ids, bm25.docs):
            text_by_id.setdefault(did, txt)

    # Slice down to the rerank window
    rerank_window = fused[:max(top_k * 4, top_k)]

    # Optional rerank
    if reranker is not None and reranker.available:
        candidates = [(doc_id, text_by_id.get(doc_id, "")) for doc_id, _ in rerank_window]
        reranked = reranker.rerank(query, candidates, top_k=top_k)
        return [(doc_id, score, text_by_id.get(doc_id, "")) for doc_id, score in reranked]

    # No rerank: return fused top-k directly
    return [
        (doc_id, score, text_by_id.get(doc_id, ""))
        for doc_id, score in rerank_window[:top_k]
    ]


__all__ = [
    "BM25Index",
    "CrossEncoderReranker",
    "hybrid_search",
    "reciprocal_rank_fusion",
    "tokenize",
]
