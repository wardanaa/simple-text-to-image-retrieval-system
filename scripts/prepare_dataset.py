"""Prepare a deterministic local subset of product image metadata."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from packages.retrieval.config import load_config
from packages.retrieval.preprocessing import validate_image_path

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=config.dataset.raw_directory)
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument("--category", default=config.dataset.category)
    parser.add_argument("--max-images", type=int, default=config.dataset.max_images)
    parser.add_argument("--random-seed", type=int, default=config.dataset.random_seed)
    parser.add_argument("--output", type=Path, default=config.dataset.processed_metadata_path)
    parser.add_argument(
        "--skipped-output", type=Path, default=Path("data/interim/skipped_images.csv")
    )
    return parser.parse_args()


def _find_metadata_file(raw_directory: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit
    for name in ["metadata.csv", "metadata.jsonl", "metadata.json"]:
        candidate = raw_directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No metadata file found in {raw_directory}. Expected metadata.csv, metadata.jsonl, or metadata.json."
    )


def _load_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
        return pd.DataFrame(records)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported metadata format: {path.suffix}")


def prepare_dataset(
    raw_directory: Path,
    metadata_file: Path,
    category: str,
    max_images: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare valid metadata rows and skipped-row diagnostics."""

    metadata = _load_metadata(metadata_file)
    if "image_path" not in metadata.columns:
        raise ValueError("Metadata must include an image_path column.")
    if "category" in metadata.columns:
        metadata = metadata[metadata["category"].astype(str).str.lower() == category.lower()]
    if max_images > 0 and len(metadata) > max_images:
        metadata = metadata.sample(n=max_images, random_state=random_seed)
    valid_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for index, row in metadata.reset_index(drop=True).iterrows():
        raw_image_path = Path(str(row["image_path"]))
        image_path = (
            raw_image_path if raw_image_path.is_absolute() else raw_directory / raw_image_path
        )
        try:
            validate_image_path(image_path)
        except Exception as exc:
            skipped_rows.append(
                {"source_row": index, "image_path": str(image_path), "reason": str(exc)}
            )
            continue
        valid_rows.append(
            {
                "image_id": str(row.get("image_id", image_path.stem)),
                "product_id": str(row.get("product_id", row.get("image_id", image_path.stem))),
                "product_name": str(row.get("product_name", row.get("title", image_path.stem))),
                "category": str(row.get("category", category)),
                "image_path": str(image_path),
                "source": str(metadata_file),
            }
        )
    return pd.DataFrame(valid_rows), pd.DataFrame(skipped_rows)


def main() -> int:
    """Run dataset preparation."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    metadata_file = _find_metadata_file(args.raw_directory, args.metadata_file)
    prepared, skipped = prepare_dataset(
        raw_directory=args.raw_directory,
        metadata_file=metadata_file,
        category=args.category,
        max_images=args.max_images,
        random_seed=args.random_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.skipped_output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(args.output, index=False)
    skipped.to_csv(args.skipped_output, index=False)
    LOGGER.info("Wrote %s valid rows to %s", len(prepared), args.output)
    LOGGER.info("Wrote %s skipped rows to %s", len(skipped), args.skipped_output)
    if prepared.empty:
        LOGGER.warning("No valid rows were prepared. Check category filters and image paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
