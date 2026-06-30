"""Tests for embedding index persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.retrieval.embedding_store import EmbeddingStore
from packages.retrieval.exceptions import EmbeddingStoreError, IndexNotFoundError


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "img-1",
                "product_id": "prod-1",
                "product_name": "Black Shoe",
                "category": "shoes",
                "image_path": "data/raw/images/img-1.jpg",
                "source": "test",
            }
        ]
    )


def test_embedding_store_save_and_load_round_trip(tmp_path):
    store = EmbeddingStore(tmp_path)
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    store.save_index(embeddings, _metadata(), "test-model")
    index = store.load_index()
    assert index.embeddings.shape == (1, 2)
    assert index.manifest.model_identifier == "test-model"
    assert index.metadata.iloc[0]["image_id"] == "img-1"


def test_embedding_store_rejects_row_count_mismatch(tmp_path):
    store = EmbeddingStore(tmp_path)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(EmbeddingStoreError):
        store.save_index(embeddings, _metadata(), "test-model")


def test_embedding_store_reports_missing_index(tmp_path):
    with pytest.raises(IndexNotFoundError):
        EmbeddingStore(tmp_path).load_index()
