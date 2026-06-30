"""Tests for similarity ranking."""

from __future__ import annotations

import numpy as np
import pytest

from packages.retrieval.exceptions import SimilarityError
from packages.retrieval.similarity import rank_by_similarity


def test_rank_by_similarity_orders_descending_scores():
    text = np.array([1.0, 0.0], dtype=np.float32)
    images = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.3]], dtype=np.float32)
    indices, scores = rank_by_similarity(text, images, top_k=2)
    assert indices.tolist() == [1, 2]
    assert scores.tolist() == [1.0, pytest.approx(0.7)]


def test_rank_by_similarity_caps_top_k_to_available_images():
    text = np.array([1.0, 0.0], dtype=np.float32)
    images = np.array([[1.0, 0.0]], dtype=np.float32)
    indices, scores = rank_by_similarity(text, images, top_k=10)
    assert indices.tolist() == [0]
    assert len(scores) == 1


def test_rank_by_similarity_rejects_invalid_top_k():
    with pytest.raises(SimilarityError):
        rank_by_similarity(np.array([1.0]), np.array([[1.0]]), top_k=0)


def test_rank_by_similarity_rejects_dimension_mismatch():
    with pytest.raises(SimilarityError):
        rank_by_similarity(np.array([1.0, 0.0]), np.array([[1.0]]), top_k=1)
