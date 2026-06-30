# Simple Text-to-Image Retrieval System

Educational monorepo for retrieving e-commerce product images from natural-language queries.
The system encodes product images and text queries with CLIP, compares normalized embeddings
with cosine similarity, and returns ranked product images.

## Objective

This is a retrieval system, not an image classifier. It demonstrates:

1. Offline image indexing with a pretrained CLIP image encoder.
2. Online text-query encoding with the CLIP text encoder.
3. Brute-force similarity ranking for a small product subset.
4. Flask as the primary web interface and Gradio as an optional demo.

## Architecture

```text
Product images -> CLIP image encoder -> image_embeddings.npy
User query     -> CLIP text encoder  -> cosine similarity -> top-k results
```

The shared implementation lives in `packages/retrieval/`. Flask and Gradio must call that
package instead of duplicating model loading, encoding, storage, or ranking logic.

## Stack

- Python 3.10+
- Flask
- PyTorch
- Hugging Face Transformers
- `openai/clip-vit-base-patch32`
- NumPy and Pandas
- Gradio, optional
- pytest and Ruff for development

## Repository Structure

```text
apps/                 Flask and Gradio applications
packages/retrieval/   Shared retrieval logic
scripts/              Dataset, embedding, evaluation, and validation commands
config/               YAML configuration
data/                 Local dataset files, ignored except placeholders
artifacts/            Generated embedding index, ignored except placeholders
docs/                 Architecture and operating documentation
tests/                Deterministic tests with synthetic data
changes/              Individual change notes
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
cp .env.example .env
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
copy .env.example .env
```

## Dataset Preparation

This repository does not include Amazon Berkeley Objects data. Download it from the official
source after reviewing license and attribution terms, then place metadata and images under
`data/raw/`.

Expected metadata fields:

```text
image_id, product_id, product_name, category, image_path
```

Prepare a small deterministic subset:

```bash
python3 scripts/download_dataset.py --print-instructions
python3 scripts/prepare_dataset.py --category shoes --max-images 500
```

## Build Embeddings

```bash
python3 scripts/build_embeddings.py --metadata data/processed/metadata.csv --overwrite
python3 scripts/inspect_index.py
```

The index is written to `artifacts/embeddings/`:

- `image_embeddings.npy`
- `metadata.csv`
- `index_manifest.json`

## Run Applications

Flask is the primary application:

```bash
flask --app apps.flask_app.app run
```

Optional Gradio demo:

```bash
python3 -m apps.gradio_app.app
```

Example queries:

```text
black running shoes
white leather sneakers
red shoulder bag
```

## Evaluation

Evaluation requires relevance labels. A tiny synthetic input example lives at
`tests/fixtures/evaluation_queries.csv`.

```bash
python3 scripts/evaluate_retrieval.py --queries tests/fixtures/evaluation_queries.csv --k 5
```

Metrics are only meaningful when the query file contains real relevance judgments.

## Configuration

Default configuration is in `config/default.yaml`. Precedence is:

1. Command-line arguments
2. Environment variables
3. YAML configuration
4. Application defaults

CPU is the default device. Set `TIR_DEVICE=auto` or pass `--device auto` to use CUDA when
available.

## Development

```bash
python3 -m compileall apps packages scripts
pytest
python3 scripts/validate_repository.py
ruff check .
ruff format --check .
```

Equivalent Make targets:

```bash
make install-dev
make test
make validate
make flask
make gradio
```

`make` may not be available on all Windows installations; use the direct commands above.

## Change Notes

Every logical repository change requires one Markdown file under `changes/` and a matching
entry in `CHANGELOG.md`. See `changes/README.md`.

## Limitations

- Small single-category subset by default.
- Brute-force search is acceptable for 500 to 1,000 images, not for large catalogs.
- Similarity scores are ranking signals, not calibrated probabilities.
- CLIP may reflect dataset bias and may not perfectly match product-domain language.
- The repository does not redistribute Amazon Berkeley Objects data.

## Documentation Index

- `docs/architecture.md`
- `docs/dataset.md`
- `docs/data-pipeline.md`
- `docs/retrieval-pipeline.md`
- `docs/development.md`
- `docs/evaluation.md`
- `docs/deployment.md`
- `docs/troubleshooting.md`
- `docs/limitations-and-future-work.md`
