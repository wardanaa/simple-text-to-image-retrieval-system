"""Inspect a persisted embedding index."""

from __future__ import annotations

import argparse
from pathlib import Path

from packages.retrieval.config import load_config
from packages.retrieval.embedding_store import EmbeddingStore


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-directory", type=Path, default=config.embedding.output_directory)
    return parser.parse_args()


def main() -> int:
    """Print index manifest and sample metadata."""

    args = parse_args()
    index = EmbeddingStore(args.index_directory).load_index()
    print(f"Model: {index.manifest.model_identifier}")
    print(f"Rows: {index.manifest.row_count}")
    print(f"Embedding dimension: {index.manifest.embedding_dimension}")
    print(f"Created at: {index.manifest.created_at}")
    print(index.metadata.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
