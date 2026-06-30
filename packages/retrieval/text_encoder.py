"""CLIP text query validation and embedding helpers."""

from __future__ import annotations

import re

import numpy as np

from packages.retrieval.exceptions import QueryValidationError
from packages.retrieval.model_loader import ClipResources
from packages.retrieval.preprocessing import normalize_embeddings


def validate_query(query: str, max_length: int = 200) -> str:
    """Validate and normalize a user text query."""

    normalized = re.sub(r"\s+", " ", query or "").strip()
    if not normalized:
        raise QueryValidationError("Search query cannot be empty.")
    if len(normalized) > max_length:
        raise QueryValidationError(f"Search query must be {max_length} characters or fewer.")
    return normalized


class TextEncoder:
    """Encode text queries with a loaded CLIP model."""

    def __init__(self, resources: ClipResources) -> None:
        self.resources = resources

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one text query and return a `(1, dimension)` float32 embedding."""

        return self.encode_queries([validate_query(query)])

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode one or more text queries and return normalized float32 embeddings."""

        import torch

        normalized_queries = [validate_query(query) for query in queries]
        inputs = self.resources.processor(
            text=normalized_queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.resources.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            features = self.resources.model.get_text_features(**inputs)
        return normalize_embeddings(features.detach().cpu().numpy())
