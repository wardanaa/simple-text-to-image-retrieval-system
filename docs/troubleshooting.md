# Troubleshooting

- CLIP model download fails: check network access and Hugging Face cache permissions.
- Missing dataset files: confirm `data/raw/metadata.csv` and image paths exist.
- Corrupt images: inspect `data/interim/skipped_images.csv`.
- Missing embedding index: run `python3 scripts/build_embeddings.py`.
- Mismatched embedding dimensions: rebuild the index after changing the CLIP model.
- Out-of-memory errors: lower `embedding.batch_size` or use CPU.
- CUDA unavailable: use `TIR_DEVICE=cpu` or install a CUDA-compatible PyTorch build.
- Slow CPU inference: reduce subset size or use CUDA when available.
- Flask import errors: install dependencies and run from the repository root.
- Images not loading: ensure indexed `image_path` values still exist locally.
- Empty retrieval results: check processed metadata and index row count.
- Dependency conflicts: recreate the virtual environment and reinstall requirements.
