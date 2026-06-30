# Agent Instructions

## Purpose

This repository is an educational Flask-first text-to-image retrieval system for e-commerce
product images. It uses CLIP embeddings to rank indexed product images against text queries.

## Boundaries

- Flask and Gradio must reuse `packages/retrieval/`.
- Do not duplicate model loading, encoding, similarity, or ranking logic in app routes.
- Do not commit datasets, generated embeddings, model caches, result artifacts, logs, or secrets.
- Use `pathlib.Path`, type hints for public functions, docstrings for public modules/classes/functions,
  and clear custom exceptions.
- Keep application startup free from hidden dataset downloads or mandatory CLIP downloads.
- Treat similarity scores as ranking values, not probabilities.

## Important Directories

- `apps/flask_app/`: primary web application.
- `apps/gradio_app/`: optional demo.
- `packages/retrieval/`: shared retrieval code.
- `scripts/`: dataset, indexing, evaluation, and repository validation commands.
- `data/`, `artifacts/`, `results/`: generated/local content, ignored except `.gitkeep`.
- `docs/`: current behavior and limitations.
- `changes/`: individual change notes.

## Configuration

Default values live in `config/default.yaml`. Configuration precedence is command-line arguments,
environment variables, YAML, then application defaults. Do not commit machine-specific absolute paths.

## Testing and Completion

Before completing a task, run or report blockers for:

```bash
python3 -m compileall apps packages scripts
pytest
python3 scripts/validate_repository.py
ruff check .
ruff format --check .
```

Completion checklist:

- Tests pass.
- Imports work.
- Configuration is valid.
- Documentation reflects behavior.
- No generated data is committed.
- A new individual change note has been added.
- `CHANGELOG.md` has been updated.
