"""Tests for preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from packages.retrieval.exceptions import PreprocessingError, QueryValidationError
from packages.retrieval.preprocessing import (
    load_rgb_image,
    normalize_embeddings,
    validate_image_path,
)
from packages.retrieval.text_encoder import validate_query


def test_validate_image_path_accepts_readable_image(tiny_image):
    assert validate_image_path(tiny_image) == tiny_image
    assert load_rgb_image(tiny_image).mode == "RGB"


def test_validate_image_path_rejects_missing_file(tmp_path):
    with pytest.raises(PreprocessingError):
        validate_image_path(tmp_path / "missing.jpg")


def test_validate_query_strips_whitespace():
    assert validate_query("  black   shoes ") == "black shoes"


def test_validate_query_rejects_empty_text():
    with pytest.raises(QueryValidationError):
        validate_query("   ")


def test_normalize_embeddings_returns_unit_vectors():
    normalized = normalize_embeddings(np.array([[3.0, 4.0]], dtype=np.float32))
    assert normalized.dtype == np.float32
    assert np.allclose(np.linalg.norm(normalized, axis=1), [1.0])


def test_normalize_embeddings_rejects_zero_vector():
    with pytest.raises(PreprocessingError):
        normalize_embeddings(np.array([[0.0, 0.0]], dtype=np.float32))
