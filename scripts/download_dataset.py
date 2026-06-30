"""Explain the supported Amazon Berkeley Objects dataset acquisition workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

INSTRUCTIONS = """Amazon Berkeley Objects acquisition

This repository does not redistribute the Amazon Berkeley Objects dataset and does not
silently download the full dataset. Download the dataset from the official Amazon
Berkeley Objects source, review its license and attribution requirements, then place
metadata and product images under the target directory.

Expected local layout:

  data/raw/
    metadata.csv
    images/
      example.jpg

The metadata file should include image_path and should preferably include image_id,
product_id, product_name, and category.
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--category", default="shoes")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-instructions", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Print dataset acquisition instructions and fail clearly by default."""

    args = parse_args()
    args.target_directory.mkdir(parents=True, exist_ok=True)
    print(INSTRUCTIONS)
    print(f"Target directory: {args.target_directory}")
    print(f"Requested category: {args.category}")
    print(f"Requested maximum images: {args.max_images}")
    if args.print_instructions:
        return 0
    print(
        "Automated download is not enabled because the dataset source and license must be reviewed."
    )
    print("After placing local files in the target directory, run scripts/prepare_dataset.py.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
