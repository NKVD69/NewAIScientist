"""
agents/proximity.py — ProximityAgent for semantic similarity analysis.

Responsible for:
- Computing hypothesis similarity using vector embeddings (primary)
- Jaccard-based fallback for text similarity
- Building proximity graphs
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from models.hypothesis import Hypothesis
from utils.llm import ensure_str

logger = logging.getLogger(__name__)


class ProximityAgent:
    """Builds proximity graph using semantic vector similarity (cosine) with Jaccard fallback."""

    def __init__(self):
        self.name = "Proximity"
        self.proximity_graph = defaultdict(lambda: defaultdict(float))
        self._embedding_model = None

        # Try to load SentenceTransformer (already used by RAG system)
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("ProximityAgent: SentenceTransformer loaded for vector similarity.")
        except Exception as e:
            logger.warning("ProximityAgent: SentenceTransformer unavailable, using Jaccard fallback. (%s)", e)

    async def compute_proximity(self, hypotheses: List[Hypothesis]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute similarity between hypotheses.
        Returns dict mapping hypothesis IDs to list of (similar_id, similarity_score)
        """
        proximity_map = {}
        
        for i, hyp_a in enumerate(hypotheses):
            similarities = []
            
            for j, hyp_b in enumerate(hypotheses):
                if i != j:
                    similarity = await self._compute_similarity(hyp_a, hyp_b)
                    similarities.append((hyp_b.id, similarity))
                    self.proximity_graph[hyp_a.id][hyp_b.id] = similarity
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            proximity_map[hyp_a.id] = similarities
        
        return proximity_map
    
    async def _compute_similarity(self, hyp_a: Hypothesis, hyp_b: Hypothesis) -> float:
        """
        Compute semantic similarity score between two hypotheses (0–1).
        Primary: cosine similarity of SentenceTransformer embeddings.
        Fallback: weighted Jaccard over mechanism words + prediction overlap + citation overlap.
        """
        # --- Primary: vector similarity ---
        if self._embedding_model is not None:
            try:
                import numpy as np
                text_a = f"{hyp_a.title}. {ensure_str(hyp_a.mechanism)} {ensure_str(hyp_a.description)}"
                text_b = f"{hyp_b.title}. {ensure_str(hyp_b.mechanism)} {ensure_str(hyp_b.description)}"

                def _encode(texts):
                    return self._embedding_model.encode(texts, convert_to_tensor=False)

                embeddings = await asyncio.to_thread(_encode, [text_a, text_b])
                emb_a, emb_b = embeddings[0], embeddings[1]

                norm_a = np.linalg.norm(emb_a)
                norm_b = np.linalg.norm(emb_b)
                if norm_a > 0 and norm_b > 0:
                    cos_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                    return (cos_sim + 1.0) / 2.0
            except Exception as e:
                logger.warning("Vector similarity failed, using Jaccard fallback: %s", e)

        # --- Fallback: Jaccard-based heuristic ---
        similarity_score = 0.0
        total_weight = 0.0

        mech_a = set(ensure_str(hyp_a.mechanism).lower().split())
        mech_b = set(ensure_str(hyp_b.mechanism).lower().split())
        if mech_a or mech_b:
            shared = len(mech_a & mech_b)
            union = len(mech_a | mech_b)
            similarity_score += (shared / union if union > 0 else 0.0) * 0.5
            total_weight += 0.5

        pred_a = set(hyp_a.testable_predictions)
        pred_b = set(hyp_b.testable_predictions)
        if pred_a or pred_b:
            max_p = max(len(pred_a), len(pred_b))
            similarity_score += (len(pred_a & pred_b) / max_p if max_p > 0 else 0.0) * 0.3
            total_weight += 0.3

        cite_a = set(hyp_a.cited_papers)
        cite_b = set(hyp_b.cited_papers)
        if cite_a or cite_b:
            max_c = max(len(cite_a), len(cite_b))
            similarity_score += (len(cite_a & cite_b) / max_c if max_c > 0 else 0.0) * 0.2
            total_weight += 0.2

        if total_weight > 0:
            return min(1.0, max(0.0, similarity_score / total_weight))
        return 0.0


__all__ = ["ProximityAgent"]
