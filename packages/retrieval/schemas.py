"""Shared data schemas for retrieval components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class IndexManifest:
    """Metadata describing a persisted embedding index."""

    model_identifier: str
    embedding_dimension: int
    row_count: int
    created_at: str

    @classmethod
    def create(
        cls, model_identifier: str, embedding_dimension: int, row_count: int
    ) -> IndexManifest:
        """Create a manifest for a newly built index."""

        return cls(
            model_identifier=model_identifier,
            embedding_dimension=embedding_dimension,
            row_count=row_count,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    """One ranked retrieval result."""

    rank: int
    image_id: str
    image_path: str
    product_name: str
    category: str
    score: float
    image_url: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)
