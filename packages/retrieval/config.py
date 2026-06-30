"""Configuration loading for the retrieval system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Project-level metadata."""

    name: str = "text-to-image-retrieval"


@dataclass(frozen=True)
class ModelConfig:
    """CLIP model configuration."""

    identifier: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset preparation configuration."""

    category: str = "shoes"
    max_images: int = 500
    random_seed: int = 42
    raw_directory: Path = Path("data/raw")
    processed_metadata_path: Path = Path("data/processed/metadata.csv")


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding index configuration."""

    batch_size: int = 16
    output_directory: Path = Path("artifacts/embeddings")


@dataclass(frozen=True)
class RetrievalConfig:
    """Online retrieval configuration."""

    default_top_k: int = 5
    maximum_top_k: int = 20


@dataclass(frozen=True)
class FlaskServerConfig:
    """Flask server configuration."""

    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    flask: FlaskServerConfig = field(default_factory=FlaskServerConfig)


DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge where `override` wins over `base`."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return data


def _env_overrides() -> dict[str, Any]:
    mapping: dict[str, tuple[str, str, Any]] = {
        "TIR_MODEL_IDENTIFIER": ("model", "identifier", str),
        "TIR_DEVICE": ("model", "device", str),
        "TIR_DATASET_CATEGORY": ("dataset", "category", str),
        "TIR_DATASET_MAX_IMAGES": ("dataset", "max_images", int),
        "TIR_DATASET_RANDOM_SEED": ("dataset", "random_seed", int),
        "TIR_RAW_DIRECTORY": ("dataset", "raw_directory", str),
        "TIR_PROCESSED_METADATA_PATH": ("dataset", "processed_metadata_path", str),
        "TIR_EMBEDDING_BATCH_SIZE": ("embedding", "batch_size", int),
        "TIR_EMBEDDING_OUTPUT_DIRECTORY": ("embedding", "output_directory", str),
        "TIR_RETRIEVAL_DEFAULT_TOP_K": ("retrieval", "default_top_k", int),
        "TIR_RETRIEVAL_MAXIMUM_TOP_K": ("retrieval", "maximum_top_k", int),
        "FLASK_HOST": ("flask", "host", str),
        "FLASK_PORT": ("flask", "port", int),
        "FLASK_DEBUG": ("flask", "debug", _parse_bool),
    }
    result: dict[str, Any] = {}
    for env_name, (section, key, parser) in mapping.items():
        if env_name not in os.environ:
            continue
        result.setdefault(section, {})[key] = parser(os.environ[env_name])
    return result


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _defaults_dict() -> dict[str, Any]:
    defaults = AppConfig()
    return {
        "project": {"name": defaults.project.name},
        "model": {"identifier": defaults.model.identifier, "device": defaults.model.device},
        "dataset": {
            "category": defaults.dataset.category,
            "max_images": defaults.dataset.max_images,
            "random_seed": defaults.dataset.random_seed,
            "raw_directory": str(defaults.dataset.raw_directory),
            "processed_metadata_path": str(defaults.dataset.processed_metadata_path),
        },
        "embedding": {
            "batch_size": defaults.embedding.batch_size,
            "output_directory": str(defaults.embedding.output_directory),
        },
        "retrieval": {
            "default_top_k": defaults.retrieval.default_top_k,
            "maximum_top_k": defaults.retrieval.maximum_top_k,
        },
        "flask": {
            "host": defaults.flask.host,
            "port": defaults.flask.port,
            "debug": defaults.flask.debug,
        },
    }


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load configuration using defaults, YAML, environment variables, and overrides."""

    path = Path(config_path or os.getenv("TIR_CONFIG_PATH", DEFAULT_CONFIG_PATH))
    raw = deep_merge(_defaults_dict(), _read_yaml(path))
    raw = deep_merge(raw, _env_overrides())
    if overrides:
        raw = deep_merge(raw, overrides)
    return AppConfig(
        project=ProjectConfig(name=str(raw["project"]["name"])),
        model=ModelConfig(
            identifier=str(raw["model"]["identifier"]),
            device=str(raw["model"]["device"]),
        ),
        dataset=DatasetConfig(
            category=str(raw["dataset"]["category"]),
            max_images=int(raw["dataset"]["max_images"]),
            random_seed=int(raw["dataset"]["random_seed"]),
            raw_directory=Path(raw["dataset"]["raw_directory"]),
            processed_metadata_path=Path(raw["dataset"]["processed_metadata_path"]),
        ),
        embedding=EmbeddingConfig(
            batch_size=int(raw["embedding"]["batch_size"]),
            output_directory=Path(raw["embedding"]["output_directory"]),
        ),
        retrieval=RetrievalConfig(
            default_top_k=int(raw["retrieval"]["default_top_k"]),
            maximum_top_k=int(raw["retrieval"]["maximum_top_k"]),
        ),
        flask=FlaskServerConfig(
            host=str(raw["flask"]["host"]),
            port=int(raw["flask"]["port"]),
            debug=bool(raw["flask"]["debug"]),
        ),
    )
