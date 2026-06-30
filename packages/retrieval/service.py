"""High-level retrieval service used by all applications."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from packages.retrieval.config import AppConfig
from packages.retrieval.embedding_store import EmbeddingIndex, EmbeddingStore
from packages.retrieval.exceptions import SimilarityError
from packages.retrieval.model_loader import load_clip_resources
from packages.retrieval.schemas import SearchResult
from packages.retrieval.similarity import rank_by_similarity
from packages.retrieval.text_encoder import TextEncoder, validate_query


class QueryEncoder(Protocol):
    """Minimal protocol for query encoders."""

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query as a normalized embedding."""


class RetrievalService:
    """Search product image embeddings using text queries."""

    def __init__(
        self,
        config: AppConfig,
        text_encoder: QueryEncoder | None = None,
        store: EmbeddingStore | None = None,
    ) -> None:
        self.config = config
        self._text_encoder = text_encoder
        self.store = store or EmbeddingStore(config.embedding.output_directory)
        self._index: EmbeddingIndex | None = None

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, object]]:
        """Return ordered retrieval results for a query."""

        normalized_query = validate_query(query)
        requested_top_k = top_k if top_k is not None else self.config.retrieval.default_top_k
        self._validate_top_k(requested_top_k)
        index = self._load_index()
        text_embedding = self._get_text_encoder().encode_query(normalized_query)
        indices, scores = rank_by_similarity(text_embedding, index.embeddings, requested_top_k)
        results: list[dict[str, object]] = []
        for rank, (row_index, score) in enumerate(zip(indices, scores, strict=True), start=1):
            row = index.metadata.iloc[int(row_index)]
            image_id = str(row.get("image_id", row_index))
            image_path = str(row.get("image_path", ""))
            result = SearchResult(
                rank=rank,
                image_id=image_id,
                image_path=image_path,
                image_url=f"/media/{image_id}",
                product_name=str(row.get("product_name", "")),
                category=str(row.get("category", "")),
                score=float(score),
            )
            results.append(result.to_dict())
        return results

    def resolve_image_path(self, image_id: str) -> Path | None:
        """Resolve a media URL image identifier to a known indexed image path."""

        index = self._load_index()
        matches = index.metadata[index.metadata["image_id"].astype(str) == str(image_id)]
        if matches.empty:
            return None
        path = Path(str(matches.iloc[0]["image_path"])).resolve()
        if not path.exists() or not path.is_file():
            return None
        return path

    def _load_index(self) -> EmbeddingIndex:
        if self._index is None:
            self._index = self.store.load_index()
        return self._index

    def _get_text_encoder(self) -> QueryEncoder:
        if self._text_encoder is None:
            resources = load_clip_resources(self.config.model)
            self._text_encoder = TextEncoder(resources)
        return self._text_encoder

    def _validate_top_k(self, top_k: int) -> None:
        if top_k < 1:
            raise SimilarityError("top_k must be at least 1.")
        if top_k > self.config.retrieval.maximum_top_k:
            raise SimilarityError(f"top_k must be at most {self.config.retrieval.maximum_top_k}.")
