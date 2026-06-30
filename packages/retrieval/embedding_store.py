"""Persistence for image embeddings and metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from packages.retrieval.exceptions import EmbeddingStoreError, IndexNotFoundError
from packages.retrieval.schemas import IndexManifest


@dataclass(frozen=True)
class EmbeddingIndex:
    """Loaded embedding index and metadata."""

    embeddings: np.ndarray
    metadata: pd.DataFrame
    manifest: IndexManifest


class EmbeddingStore:
    """Save and load a flat image embedding index."""

    embeddings_filename = "image_embeddings.npy"
    metadata_filename = "metadata.csv"
    manifest_filename = "index_manifest.json"

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)

    @property
    def embeddings_path(self) -> Path:
        """Path to the NumPy embedding matrix."""

        return self.output_directory / self.embeddings_filename

    @property
    def metadata_path(self) -> Path:
        """Path to the CSV metadata file."""

        return self.output_directory / self.metadata_filename

    @property
    def manifest_path(self) -> Path:
        """Path to the JSON manifest file."""

        return self.output_directory / self.manifest_filename

    def exists(self) -> bool:
        """Return whether all required index files exist."""

        return (
            self.embeddings_path.exists()
            and self.metadata_path.exists()
            and self.manifest_path.exists()
        )

    def save_index(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        model_identifier: str,
        overwrite: bool = False,
    ) -> IndexManifest:
        """Persist embeddings, metadata, and manifest."""

        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2:
            raise EmbeddingStoreError("Embeddings must be a 2D array.")
        if len(metadata) != array.shape[0]:
            raise EmbeddingStoreError("Metadata row count must match embedding row count.")
        if self.exists() and not overwrite:
            raise EmbeddingStoreError(f"Embedding index already exists in {self.output_directory}.")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        manifest = IndexManifest.create(
            model_identifier=model_identifier,
            embedding_dimension=int(array.shape[1]),
            row_count=int(array.shape[0]),
        )
        np.save(self.embeddings_path, array)
        metadata.to_csv(self.metadata_path, index=False)
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2)
        return manifest

    def load_index(self) -> EmbeddingIndex:
        """Load and validate a persisted embedding index."""

        missing = [
            path
            for path in [self.embeddings_path, self.metadata_path, self.manifest_path]
            if not path.exists()
        ]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise IndexNotFoundError(f"Embedding index has not been built. Missing: {joined}")
        try:
            embeddings = np.load(self.embeddings_path).astype(np.float32)
            metadata = pd.read_csv(self.metadata_path)
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                raw_manifest: dict[str, Any] = json.load(handle)
            manifest = IndexManifest(
                model_identifier=str(raw_manifest["model_identifier"]),
                embedding_dimension=int(raw_manifest["embedding_dimension"]),
                row_count=int(raw_manifest["row_count"]),
                created_at=str(raw_manifest["created_at"]),
            )
        except Exception as exc:
            raise EmbeddingStoreError(f"Could not load embedding index: {exc}") from exc
        self._validate_loaded(embeddings, metadata, manifest)
        return EmbeddingIndex(embeddings=embeddings, metadata=metadata, manifest=manifest)

    def _validate_loaded(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        manifest: IndexManifest,
    ) -> None:
        if embeddings.ndim != 2:
            raise EmbeddingStoreError("Loaded embeddings must be a 2D array.")
        if len(metadata) != embeddings.shape[0]:
            raise EmbeddingStoreError("Metadata row count does not match embedding row count.")
        if manifest.row_count != embeddings.shape[0]:
            raise EmbeddingStoreError("Manifest row count does not match embedding row count.")
        if manifest.embedding_dimension != embeddings.shape[1]:
            raise EmbeddingStoreError("Manifest embedding dimension does not match embeddings.")
