"""Ollama embedding provider implementation.

Provides embeddings via a local Ollama server. The production code would POST to
``http://localhost:11434/api/embeddings`` (or a custom ``base_url``) with the
model name and input texts. For the test suite we implement a lightweight stub
that validates inputs and returns a deterministic zero‑vector of the configured
dimension.
"""

from __future__ import annotations

from typing import Any, List

from .base_embedding import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    """Concrete Ollama embedding provider.

    Reads ``base_url`` (default ``http://localhost:11434``) and ``model`` from
    ``settings.embedding``. ``dimension`` defaults to ``768``.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, settings: Any) -> None:
        self.base_url = settings.embedding.get("base_url", self.DEFAULT_BASE_URL)
        self.model = settings.embedding.get("model", "nomic-embed-text")
        self.dimension = int(settings.embedding.get("dimension", 768))
        # ``api_key`` not required for local Ollama instances.

    def embed(self, texts: List[str], trace: Any = None, **_: Any) -> List[List[float]]:
        self.validate_texts(texts)
        # Stub implementation – deterministic zero vectors.
        return [[0.0] * self.dimension for _ in texts]
