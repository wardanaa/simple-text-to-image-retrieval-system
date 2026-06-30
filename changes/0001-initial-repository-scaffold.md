# Change 0001: Initial Repository Scaffold

- Date: 2026-06-30
- Type: added
- Status: completed
- Scope: repository

## Summary

Added a complete Python monorepo scaffold for a simple CLIP-based text-to-image retrieval system.

## Motivation

The project needs a maintainable educational baseline that separates applications, shared retrieval
logic, scripts, configuration, generated artifacts, tests, and documentation.

## Files Changed

- `apps/`
- `packages/retrieval/`
- `scripts/`
- `config/`
- `docs/`
- `tests/`
- Root project metadata and policy files

## Implementation Details

Flask is the primary app. Gradio is optional and reuses the shared retrieval service. Embeddings are
stored as NumPy arrays plus CSV metadata and a JSON manifest. Dataset and artifact directories are
ignored except for placeholders.

## Validation

Planned validation commands:

```bash
python3 -m compileall apps packages scripts
pytest
python3 scripts/validate_repository.py
ruff check .
ruff format --check .
```

## Risks and Limitations

The repository does not include Amazon Berkeley Objects data, generated embeddings, or model weights.
The first implementation uses brute-force search for a small local subset.

## Follow-up Work

Add real dataset metadata locally, build embeddings, and evaluate retrieval quality with relevance labels.
