"""Flask-specific configuration helpers."""

from packages.retrieval.config import AppConfig, load_config


def load_app_config() -> AppConfig:
    """Load the shared application configuration for Flask."""

    return load_config()
