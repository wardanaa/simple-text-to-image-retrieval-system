"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def tiny_image(tmp_path: Path) -> Path:
    """Create a tiny valid image fixture."""

    path = tmp_path / "product.jpg"
    Image.new("RGB", (4, 4), color=(12, 34, 56)).save(path)
    return path
