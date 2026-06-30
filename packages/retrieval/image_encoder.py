"""CLIP image embedding helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from packages.retrieval.model_loader import ClipResources
from packages.retrieval.preprocessing import normalize_embeddings, prepare_image_inputs

if TYPE_CHECKING:
    import torch


def _extract_image_features(output: Any) -> torch.Tensor:
    """Extract projected image features across Transformers API versions."""

    import torch

    # Older Transformers versions return the projected features directly.
    if isinstance(output, torch.Tensor):
        return output

    # Projection-specific model outputs expose the tensor as image_embeds.
    image_embeds = getattr(output, "image_embeds", None)
    if isinstance(image_embeds, torch.Tensor):
        return image_embeds

    # Newer CLIPModel.get_image_features() returns BaseModelOutputWithPooling
    # and stores the projected image features in pooler_output.
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    raise TypeError(
        "Unsupported CLIP image feature output: "
        f"{type(output).__module__}.{type(output).__name__}. "
        "Expected a torch.Tensor or an output containing image_embeds "
        "or pooler_output."
    )


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

        if not image_paths:
            raise ValueError("image_paths must not be empty.")

        inputs = prepare_image_inputs(
            image_paths,
            self.resources.processor,
            self.resources.device,
        )
        with torch.inference_mode():
            output = self.resources.model.get_image_features(**inputs)
            features = _extract_image_features(output)

        embeddings = features.detach().cpu().numpy().astype(np.float32, copy=False)
        return normalize_embeddings(embeddings).astype(np.float32, copy=False)
