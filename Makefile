PYTHON ?= python3

.PHONY: install install-dev prepare-data embeddings flask gradio test lint format validate

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

prepare-data:
	$(PYTHON) scripts/prepare_dataset.py

embeddings:
	$(PYTHON) scripts/build_embeddings.py

flask:
	flask --app apps.flask_app.app run

gradio:
	$(PYTHON) -m apps.gradio_app.app

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

validate:
	$(PYTHON) -m compileall apps packages scripts
	pytest
	$(PYTHON) scripts/validate_repository.py
	ruff check .
	ruff format --check .
