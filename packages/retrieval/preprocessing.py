"""Image validation and preprocessing helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from packages.retrieval.exceptions import PreprocessingError

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def validate_image_path(path: str | Path) -> Path:
    """Validate that a path points to a readable supported image."""

    image_path = Path(path)
    if not image_path.exists():
        raise PreprocessingError(f"Image file does not exist: {image_path}")
    if not image_path.is_file():
        raise PreprocessingError(f"Image path is not a file: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise PreprocessingError(f"Unsupported image format: {image_path.suffix}")
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise PreprocessingError(f"Image is missing or corrupt: {image_path}") from exc
    return image_path


def load_rgb_image(path: str | Path) -> Image.Image:
    """Open an image and return an RGB copy suitable for CLIP processing."""

    image_path = validate_image_path(path)
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise PreprocessingError(f"Could not load image as RGB: {image_path}") from exc


def prepare_image_inputs(
    image_paths: Iterable[str | Path],
    processor: Any,
    device: str,
) -> dict[str, Any]:
    """Prepare a batch of images with the CLIP processor."""

    images = [load_rgb_image(path) for path in image_paths]
    if not images:
        raise PreprocessingError("At least one image is required for encoding.")
    inputs = processor(images=images, return_tensors="pt")
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings along the last dimension."""

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise PreprocessingError("Cannot normalize zero-vector embeddings.")
    return (array / norms).astype(np.float32)
