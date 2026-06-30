# Development

Set up the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
```

Run Flask:

```bash
flask --app apps.flask_app.app run
```

Run Gradio:

```bash
python3 -m apps.gradio_app.app
```

Run validation:

```bash
python3 -m compileall apps packages scripts
pytest
python3 scripts/validate_repository.py
ruff check .
ruff format --check .
```

When modifying shared retrieval code, update tests and documentation together. Every logical change
requires a new file in `changes/` and an entry in `CHANGELOG.md`.
