# Data Pipeline

1. Acquire Amazon Berkeley Objects data manually after reviewing license terms.
2. Place metadata and image files under `data/raw/`.
3. Run `python3 scripts/prepare_dataset.py`.
4. The script filters by category, samples up to the configured maximum, validates image paths, and
   records skipped rows in `data/interim/skipped_images.csv`.
5. Run `python3 scripts/build_embeddings.py --overwrite`.
6. The embedding index is written to `artifacts/embeddings/`.

The index can be rebuilt by passing `--overwrite`. Individual corrupt images are skipped during
embedding generation and are not included in saved metadata.
