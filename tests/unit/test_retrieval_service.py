"""Tests for the high-level retrieval service."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.retrieval.config import AppConfig, EmbeddingConfig, RetrievalConfig
from packages.retrieval.embedding_store import EmbeddingStore
from packages.retrieval.exceptions import QueryValidationError, SimilarityError
from packages.retrieval.service import RetrievalService


class FakeTextEncoder:
    """Deterministic query encoder."""

    def encode_query(self, query: str) -> np.ndarray:
        return np.array([[1.0, 0.0]], dtype=np.float32)


def _config(tmp_path) -> AppConfig:
    return AppConfig(
        embedding=EmbeddingConfig(output_directory=tmp_path),
        retrieval=RetrievalConfig(default_top_k=2, maximum_top_k=5),
    )


def _save_index(tmp_path) -> EmbeddingStore:
    store = EmbeddingStore(tmp_path)
    store.save_index(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        pd.DataFrame(
            [
                {
                    "image_id": "blue",
                    "product_id": "1",
                    "product_name": "Blue Product",
                    "category": "shoes",
                    "image_path": "/tmp/blue.jpg",
                    "source": "test",
                },
                {
                    "image_id": "black",
                    "product_id": "2",
                    "product_name": "Black Product",
                    "category": "shoes",
                    "image_path": "/tmp/black.jpg",
                    "source": "test",
                },
            ]
        ),
        "test-model",
    )
    return store


def test_retrieval_service_returns_ranked_results(tmp_path):
    service = RetrievalService(
        _config(tmp_path), text_encoder=FakeTextEncoder(), store=_save_index(tmp_path)
    )
    results = service.search("black shoes", top_k=2)
    assert [result["image_id"] for result in results] == ["black", "blue"]
    assert results[0]["rank"] == 1
    assert results[0]["image_url"] == "/media/black"


def test_retrieval_service_rejects_empty_query(tmp_path):
    service = RetrievalService(
        _config(tmp_path), text_encoder=FakeTextEncoder(), store=_save_index(tmp_path)
    )
    with pytest.raises(QueryValidationError):
        service.search(" ")


def test_retrieval_service_rejects_top_k_above_configured_maximum(tmp_path):
    service = RetrievalService(
        _config(tmp_path), text_encoder=FakeTextEncoder(), store=_save_index(tmp_path)
    )
    with pytest.raises(SimilarityError):
        service.search("black shoes", top_k=6)
