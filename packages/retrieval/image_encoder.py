"""CLIP image embedding helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from packages.retrieval.model_loader import ClipResources
from packages.retrieval.preprocessing import normalize_embeddings, prepare_image_inputs


class ImageEncoder:
    """Encode product images with a loaded CLIP model."""

    def __init__(self, resources: ClipResources) -> None:
        self.resources = resources

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        """Encode one image and return a `(1, dimension)` float32 embedding."""

        return self.encode_batch([image_path])

    def encode_batch(self, image_paths: list[str | Path]) -> np.ndarray:
        """Encode a batch of images and return normalized float32 embeddings."""

        import torch

        inputs = prepare_image_inputs(image_paths, self.resources.processor, self.resources.device)
        with torch.inference_mode():
            features = self.resources.model.get_image_features(**inputs)
        return normalize_embeddings(features.detach().cpu().numpy())
