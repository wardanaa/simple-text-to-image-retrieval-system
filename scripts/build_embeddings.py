"""Build and persist CLIP image embeddings from processed metadata."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from packages.retrieval.config import load_config
from packages.retrieval.embedding_store import EmbeddingStore
from packages.retrieval.image_encoder import ImageEncoder
from packages.retrieval.model_loader import load_clip_resources

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=config.dataset.processed_metadata_path)
    parser.add_argument("--output-directory", type=Path, default=config.embedding.output_directory)
    parser.add_argument("--model-identifier", default=config.model.identifier)
    parser.add_argument("--device", default=config.model.device)
    parser.add_argument("--batch-size", type=int, default=config.embedding.batch_size)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _batches(rows: list[dict[str, object]], batch_size: int) -> list[list[dict[str, object]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def main() -> int:
    """Build the embedding index."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not args.metadata.exists():
        raise FileNotFoundError(f"Processed metadata not found: {args.metadata}")
    metadata = pd.read_csv(args.metadata)
    if "image_path" not in metadata.columns:
        raise ValueError("Processed metadata must include image_path.")
    config = load_config(
        overrides={
            "model": {"identifier": args.model_identifier, "device": args.device},
            "embedding": {
                "output_directory": str(args.output_directory),
                "batch_size": args.batch_size,
            },
        }
    )
    resources = load_clip_resources(config.model)
    encoder = ImageEncoder(resources)
    encoded_batches: list[np.ndarray] = []
    successful_rows: list[dict[str, object]] = []
    failed = 0
    for batch in _batches(metadata.to_dict(orient="records"), args.batch_size):
        paths = [row["image_path"] for row in batch]
        try:
            embeddings = encoder.encode_batch([Path(str(path)) for path in paths])
        except Exception as exc:
            LOGGER.warning("Batch failed; retrying images individually: %s", exc)
            for row in batch:
                try:
                    embedding = encoder.encode_image(Path(str(row["image_path"])))
                except Exception as image_exc:
                    failed += 1
                    LOGGER.warning("Skipping %s: %s", row["image_path"], image_exc)
                    continue
                encoded_batches.append(embedding)
                successful_rows.append(row)
            continue
        encoded_batches.append(embeddings)
        successful_rows.extend(batch)
        LOGGER.info("Encoded %s/%s images", len(successful_rows), len(metadata))
    if not encoded_batches:
        raise RuntimeError("No images were successfully embedded.")
    all_embeddings = np.vstack(encoded_batches).astype(np.float32)
    successful_metadata = pd.DataFrame(successful_rows)
    manifest = EmbeddingStore(args.output_directory).save_index(
        embeddings=all_embeddings,
        metadata=successful_metadata,
        model_identifier=args.model_identifier,
        overwrite=args.overwrite,
    )
    LOGGER.info("Saved %s embeddings to %s", manifest.row_count, args.output_directory)
    LOGGER.info("Skipped %s failed images", failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
