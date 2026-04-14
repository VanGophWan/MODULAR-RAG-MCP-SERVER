"""OpenAI embedding provider implementation.

This implementation is deliberately lightweight – it validates inputs and returns a
deterministic zero‑vector for each text.  Real‑world code would perform an HTTP
POST to ``https://api.openai.com/v1/embeddings`` using the ``model`` and
``api_key`` from ``settings.embedding``.  For the purposes of unit tests we avoid
network calls and keep the implementation self‑contained.
"""

from __future__ import annotations

from typing import Any, List

from .base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """Concrete OpenAI embedding provider.

    Parameters are read from the ``Settings`` instance passed to the constructor.
    Only ``model`` and optional ``dimension`` are used – the latter defaults to
    ``768`` which matches the default OpenAI text‑embedding‑3‑small model.
    """

    def __init__(self, settings: Any) -> None:
        # ``settings`` is expected to be an instance of ``core.settings.Settings``
        self.model = settings.embedding.get("model", "text-embedding-3-small")
        self.dimension = int(settings.embedding.get("dimension", 768))
        # ``api_key`` may be provided via settings or environment variable; we do
        # not use it here because the implementation is a stub.
        self.api_key = settings.embedding.get("api_key")

    def embed(self, texts: List[str], trace: Any = None, **_: Any) -> List[List[float]]:
        # Validate input using helper from BaseEmbedding
        self.validate_texts(texts)
        # Return a deterministic zero‑vector for each input text.
        return [[0.0] * self.dimension for _ in texts]
