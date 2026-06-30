"""CLIP model loading and device selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from packages.retrieval.config import ModelConfig
from packages.retrieval.exceptions import ModelLoadError


@dataclass(frozen=True)
class ClipResources:
    """Loaded CLIP model resources."""

    model: Any
    processor: Any
    device: str
    model_identifier: str


def resolve_device(device: str) -> str:
    """Resolve a configured device string into `cpu` or `cuda`."""

    normalized = device.lower()
    if normalized == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise ModelLoadError("PyTorch is required to resolve CUDA availability.") from exc
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError("CUDA was requested but is not available.")
        return "cuda"
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ModelLoadError(f"Unsupported device setting: {device}")


def load_clip_resources(config: ModelConfig) -> ClipResources:
    """Load and cache the configured CLIP model and processor."""

    return _load_clip_resources(config.identifier, resolve_device(config.device))


@lru_cache(maxsize=4)
def _load_clip_resources(model_identifier: str, device: str) -> ClipResources:
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(model_identifier)
        model = CLIPModel.from_pretrained(model_identifier)
        model.to(device)
        model.eval()
        torch.set_grad_enabled(False)
        return ClipResources(
            model=model,
            processor=processor,
            device=device,
            model_identifier=model_identifier,
        )
    except Exception as exc:
        raise ModelLoadError(f"Failed to load CLIP model '{model_identifier}': {exc}") from exc
