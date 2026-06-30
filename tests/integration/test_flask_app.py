"""Integration tests for Flask routes."""

from __future__ import annotations

from http import HTTPStatus

from apps.flask_app.app import create_app


class FakeService:
    """Fake retrieval service for successful route tests."""

    def search(self, query: str, top_k: int):
        return [
            {
                "rank": 1,
                "image_id": "img-1",
                "image_path": "/tmp/img-1.jpg",
                "image_url": "/media/img-1",
                "product_name": "Black Shoe",
                "category": "shoes",
                "score": 0.91,
            }
        ]

    def resolve_image_path(self, image_id: str):
        return None


def test_flask_home_page_loads():
    app = create_app()
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert b"Search product images" in response.data


def test_flask_health_endpoint_loads():
    app = create_app()
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == HTTPStatus.OK
    assert response.json["status"] == "ok"


def test_flask_search_validation_rejects_empty_query():
    app = create_app()
    client = app.test_client()
    response = client.post("/search", data={"query": "", "top_k": "5"})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert b"Search query cannot be empty" in response.data


def test_flask_api_reports_missing_index_without_stack_trace():
    app = create_app()
    client = app.test_client()
    response = client.post("/api/search", json={"query": "black shoes", "top_k": 5})
    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert "error" in response.json


def test_flask_api_success_with_fake_service():
    app = create_app()
    app.config["RETRIEVAL_SERVICE"] = FakeService()
    client = app.test_client()
    response = client.post("/api/search", json={"query": "black shoes", "top_k": 1})
    assert response.status_code == HTTPStatus.OK
    assert response.json["count"] == 1
    assert response.json["results"][0]["image_id"] == "img-1"
