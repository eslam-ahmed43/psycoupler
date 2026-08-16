"""
Multidimensional embedding backend for PsyCoupler.

Replaces the scalar keyword extractor with sentence-transformers
embeddings, enabling multidimensional psychological state representation
as described in Rocca et al. (2026).

Usage:
    from psycoupler.embeddings import EmbeddingExtractor
    extractor = EmbeddingExtractor()
    sentiment_fn = extractor.as_sentiment_fn()
    result = analyze_conversation(turns, sentiment_fn=sentiment_fn)
"""
from __future__ import annotations

import numpy as np
from typing import Callable


class EmbeddingExtractor:
    """
    Sentence-transformer based psychological state extractor.

    Projects text onto a valence axis defined by two anchor sentences,
    producing a scalar sentiment score in [-1, 1] that captures
    semantic meaning rather than keyword matching.

    Parameters
    ----------
    model_name:
        Any sentence-transformers model. Default is all-MiniLM-L6-v2
        (fast, 384-dim, good zero-shot performance).
    positive_anchor:
        Reference sentence for the positive pole.
    negative_anchor:
        Reference sentence for the negative pole.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        positive_anchor: str = "I feel happy, hopeful, understood, and at peace.",
        negative_anchor: str = "I feel terrible, hopeless, alone, and completely broken.",
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for EmbeddingExtractor. "
                "Install it with: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(model_name)
        self._positive = self._model.encode(positive_anchor, normalize_embeddings=True)
        self._negative = self._model.encode(negative_anchor, normalize_embeddings=True)

    def score(self, text: str) -> float:
        """
        Score a text on the negative-positive psychological valence axis.

        Returns
        -------
        float
            Score in [-1, 1]. Positive = closer to positive anchor.
        """
        vec = self._model.encode(text, normalize_embeddings=True)
        pos_sim = float(np.dot(vec, self._positive))
        neg_sim = float(np.dot(vec, self._negative))
        return float(np.clip(pos_sim - neg_sim, -1.0, 1.0))

    def as_sentiment_fn(self) -> Callable[[str], float]:
        """
        Return a callable compatible with analyze_conversation's sentiment_fn.

        Example
        -------
        >>> extractor = EmbeddingExtractor()
        >>> result = analyze_conversation(turns, sentiment_fn=extractor.as_sentiment_fn())
        """
        return self.score

    def embed(self, text: str) -> np.ndarray:
        """
        Return the raw embedding vector for a text.
        Useful for custom distance metrics or visualization.
        """
        return self._model.encode(text, normalize_embeddings=True)


__all__ = ["EmbeddingExtractor"]