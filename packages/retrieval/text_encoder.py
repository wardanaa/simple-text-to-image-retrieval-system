"""CLIP text query validation and embedding helpers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

from packages.retrieval.exceptions import QueryValidationError
from packages.retrieval.model_loader import ClipResources
from packages.retrieval.preprocessing import normalize_embeddings

if TYPE_CHECKING:
    import torch


def validate_query(query: str, max_length: int = 200) -> str:
    """Validate and normalize a user text query."""

    normalized = re.sub(r"\s+", " ", query or "").strip()
    if not normalized:
        raise QueryValidationError("Search query cannot be empty.")
    if len(normalized) > max_length:
        raise QueryValidationError(f"Search query must be {max_length} characters or fewer.")
    return normalized


def _extract_text_features(output: Any) -> "torch.Tensor":
    """Extract projected CLIP text features across Transformers versions."""

    import torch

    # Older Transformers versions return the projected features directly.
    if isinstance(output, torch.Tensor):
        return output

    # Projection-model outputs may expose the embedding explicitly.
    text_embeds = getattr(output, "text_embeds", None)
    if isinstance(text_embeds, torch.Tensor):
        return text_embeds

    # Newer CLIPModel.get_text_features() versions wrap projected features
    # in BaseModelOutputWithPooling.pooler_output.
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    raise TypeError(
        "Unsupported CLIP text feature output: "
        f"{type(output).__module__}.{type(output).__name__}"
    )


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

        if not queries:
            raise ValueError("queries must not be empty.")

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
            output = self.resources.model.get_text_features(**inputs)
            features = _extract_text_features(output)

        embeddings = normalize_embeddings(features.detach().cpu().numpy())
        return embeddings.astype(np.float32, copy=False)
