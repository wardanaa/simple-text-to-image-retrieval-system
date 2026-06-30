"""Cosine similarity ranking for normalized embeddings."""

from __future__ import annotations

import numpy as np

from packages.retrieval.exceptions import SimilarityError


def rank_by_similarity(
    text_embedding: np.ndarray,
    image_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank image embeddings by cosine similarity against one text embedding."""

    text = np.asarray(text_embedding, dtype=np.float32)
    images = np.asarray(image_embeddings, dtype=np.float32)
    if text.ndim == 2 and text.shape[0] == 1:
        text = text[0]
    if text.ndim != 1:
        raise SimilarityError("Text embedding must be one vector.")
    if images.ndim != 2:
        raise SimilarityError("Image embeddings must be a 2D matrix.")
    if images.shape[0] == 0:
        raise SimilarityError("At least one image embedding is required.")
    if text.shape[0] != images.shape[1]:
        raise SimilarityError(
            f"Dimension mismatch: text has {text.shape[0]}, images have {images.shape[1]}."
        )
    if top_k < 1:
        raise SimilarityError("top_k must be at least 1.")
    limit = min(top_k, images.shape[0])
    scores = images @ text
    order = np.argsort(scores)[::-1][:limit]
    return order.astype(np.int64), scores[order].astype(np.float32)
