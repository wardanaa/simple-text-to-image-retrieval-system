"""Validate repository structure and lightweight project invariants."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REQUIRED_FILES = [
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "CHANGELOG.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "config/default.yaml",
    "packages/retrieval/service.py",
    "apps/flask_app/app.py",
    "apps/gradio_app/app.py",
    "changes/0001-initial-repository-scaffold.md",
]
REQUIRED_DIRECTORIES = [
    "apps/flask_app",
    "apps/gradio_app",
    "packages/retrieval",
    "scripts",
    "data/raw",
    "data/interim",
    "data/processed",
    "artifacts/embeddings",
    "results",
    "tests/unit",
    "tests/integration",
    "docs",
    "changes",
]


def _error(message: str, errors: list[str]) -> None:
    errors.append(message)


def check_required_paths(errors: list[str]) -> None:
    """Check required files and directories."""

    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            _error(f"Missing required file: {relative}", errors)
    for relative in REQUIRED_DIRECTORIES:
        if not (ROOT / relative).is_dir():
            _error(f"Missing required directory: {relative}", errors)


def check_config(errors: list[str]) -> None:
    """Check that YAML configuration can be parsed."""

    try:
        with (ROOT / "config/default.yaml").open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle)
        from packages.retrieval.config import load_config

        load_config(ROOT / "config/default.yaml")
    except Exception as exc:
        _error(f"Configuration validation failed: {exc}", errors)


def check_imports(errors: list[str]) -> None:
    """Check that project modules import."""

    try:
        import apps.flask_app.app  # noqa: F401
        import packages.retrieval.service  # noqa: F401
    except Exception as exc:
        _error(f"Import validation failed: {exc}", errors)


def check_manifest(errors: list[str]) -> None:
    """Validate manifest shape when an embedding index exists."""

    manifest_path = ROOT / "artifacts/embeddings/index_manifest.json"
    if not manifest_path.exists():
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {"model_identifier", "embedding_dimension", "row_count", "created_at"}
        missing = required - set(data)
        if missing:
            _error(f"Embedding manifest is missing keys: {sorted(missing)}", errors)
    except Exception as exc:
        _error(f"Embedding manifest validation failed: {exc}", errors)


def check_change_notes(errors: list[str]) -> None:
    """Check change-note naming and changelog references."""

    change_files = sorted((ROOT / "changes").glob("*.md"))
    pattern = re.compile(r"^\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
    for path in change_files:
        if path.name == "README.md":
            continue
        if not pattern.match(path.name):
            _error(f"Invalid change-note filename: {path.name}", errors)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for path in change_files:
        if path.name == "README.md":
            continue
        if f"changes/{path.name}" not in changelog:
            _error(f"CHANGELOG.md does not reference {path.name}", errors)


def check_generated_data_tracking(errors: list[str]) -> None:
    """Check that generated data is not tracked when the directory is a Git repo."""

    if not (ROOT / ".git").exists():
        return
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "data/raw",
            "data/interim",
            "data/processed",
            "artifacts/embeddings",
            "results",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    tracked = [
        line
        for line in completed.stdout.splitlines()
        if not line.endswith(".gitkeep") and line.strip()
    ]
    if tracked:
        _error(f"Generated data appears tracked: {tracked}", errors)


def main() -> int:
    """Run repository validation checks."""

    errors: list[str] = []
    check_required_paths(errors)
    check_config(errors)
    check_imports(errors)
    check_manifest(errors)
    check_change_notes(errors)
    check_generated_data_tracking(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Repository validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
