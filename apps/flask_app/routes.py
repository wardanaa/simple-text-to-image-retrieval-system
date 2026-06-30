"""HTTP routes for the Flask retrieval application."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, render_template, request, send_file

from packages.retrieval.exceptions import IndexNotFoundError, QueryValidationError, RetrievalError
from packages.retrieval.service import RetrievalService

bp = Blueprint("retrieval", __name__)


def _service() -> RetrievalService:
    return current_app.config["RETRIEVAL_SERVICE"]


def _parse_top_k(value: str | None) -> int:
    if value is None or value == "":
        return int(current_app.config["APP_CONFIG"].retrieval.default_top_k)
    try:
        return int(value)
    except ValueError as exc:
        raise QueryValidationError("Top-k must be an integer.") from exc


@bp.get("/")
def index() -> str:
    """Render the search form."""

    app_config = current_app.config["APP_CONFIG"]
    return render_template(
        "index.html",
        default_top_k=app_config.retrieval.default_top_k,
        maximum_top_k=app_config.retrieval.maximum_top_k,
    )


@bp.post("/search")
def search() -> str | tuple[str, int]:
    """Handle server-rendered search results."""

    query = request.form.get("query", "")
    try:
        top_k = _parse_top_k(request.form.get("top_k"))
        results = _service().search(query=query, top_k=top_k)
        return render_template("results.html", query=query.strip(), top_k=top_k, results=results)
    except (QueryValidationError, IndexNotFoundError, RetrievalError) as exc:
        app_config = current_app.config["APP_CONFIG"]
        return (
            render_template(
                "index.html",
                error=str(exc),
                query=query,
                default_top_k=request.form.get("top_k", app_config.retrieval.default_top_k),
                maximum_top_k=app_config.retrieval.maximum_top_k,
            ),
            HTTPStatus.BAD_REQUEST,
        )


@bp.post("/api/search")
def api_search() -> tuple[Response, int] | Response:
    """Return JSON search results."""

    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", ""))
    try:
        top_k = _parse_top_k(str(payload["top_k"]) if "top_k" in payload else None)
        results = _service().search(query=query, top_k=top_k)
        return jsonify(
            {"query": query.strip(), "top_k": top_k, "count": len(results), "results": results}
        )
    except QueryValidationError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST
    except IndexNotFoundError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.SERVICE_UNAVAILABLE
    except RetrievalError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST


@bp.get("/health")
def health() -> Response:
    """Return basic application health."""

    return jsonify({"status": "ok", "service": current_app.config["APP_CONFIG"].project.name})


@bp.get("/media/<image_id>")
def media(image_id: str) -> Response | tuple[str, int]:
    """Serve only images referenced by the loaded embedding index."""

    path = _service().resolve_image_path(image_id)
    if path is None:
        return "Image not found", HTTPStatus.NOT_FOUND
    return send_file(Path(path))
