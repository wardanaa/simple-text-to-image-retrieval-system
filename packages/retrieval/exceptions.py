"""Custom exceptions for retrieval workflows."""


class RetrievalError(Exception):
    """Base exception for retrieval system failures."""


class ConfigurationError(RetrievalError):
    """Raised when configuration is invalid."""


class ModelLoadError(RetrievalError):
    """Raised when the CLIP model or processor cannot be loaded."""


class PreprocessingError(RetrievalError):
    """Raised when input images or text cannot be preprocessed."""


class IndexNotFoundError(RetrievalError):
    """Raised when an embedding index is unavailable."""


class EmbeddingStoreError(RetrievalError):
    """Raised when embedding persistence or validation fails."""


class QueryValidationError(RetrievalError):
    """Raised when a text query is invalid."""


class SimilarityError(RetrievalError):
    """Raised when similarity ranking inputs are invalid."""
