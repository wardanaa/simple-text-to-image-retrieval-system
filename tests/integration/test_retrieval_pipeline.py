"""Integration test for a synthetic retrieval pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.retrieval.config import AppConfig, EmbeddingConfig
from packages.retrieval.embedding_store import EmbeddingStore
from packages.retrieval.service import RetrievalService


class RedQueryEncoder:
    """Return a deterministic text embedding."""

    def encode_query(self, query: str) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)


def test_retrieval_pipeline_with_synthetic_index(tmp_path):
    store = EmbeddingStore(tmp_path)
    store.save_index(
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        pd.DataFrame(
            [
                {
                    "image_id": "green",
                    "product_id": "p-green",
                    "product_name": "Green Bag",
                    "category": "bags",
                    "image_path": "/tmp/green.jpg",
                    "source": "synthetic",
                },
                {
                    "image_id": "red",
                    "product_id": "p-red",
                    "product_name": "Red Bag",
                    "category": "bags",
                    "image_path": "/tmp/red.jpg",
                    "source": "synthetic",
                },
            ]
        ),
        "synthetic-model",
    )
    service = RetrievalService(
        AppConfig(embedding=EmbeddingConfig(output_directory=tmp_path)),
        text_encoder=RedQueryEncoder(),
        store=store,
    )
    results = service.search("red bag", top_k=1)
    assert results[0]["image_id"] == "red"
    assert results[0]["score"] == 1.0
