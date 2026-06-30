"""Flask application factory."""

from __future__ import annotations

from flask import Flask

from apps.flask_app.config import load_app_config
from apps.flask_app.routes import bp
from packages.retrieval.service import RetrievalService


def create_app() -> Flask:
    """Create and configure the Flask application."""

    app_config = load_app_config()
    app = Flask(__name__)
    app.config["APP_CONFIG"] = app_config
    app.config["RETRIEVAL_SERVICE"] = RetrievalService(app_config)
    app.register_blueprint(bp)
    return app


app = create_app()
