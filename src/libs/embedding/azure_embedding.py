"""Azure OpenAI embedding provider implementation.

Mirrors :class:`OpenAIEmbedding` but uses Azure‑specific configuration keys.
The real implementation would POST to the Azure endpoint constructed from
``settings.embedding.azure_endpoint`` and ``deployment_name`` (the model name).
For unit testing we provide a stub that returns a deterministic zero‑vector.
"""

from __future__ import annotations

from typing import Any, List

from .base_embedding import BaseEmbedding


class AzureEmbedding(BaseEmbedding):
    """Concrete Azure OpenAI embedding provider.

    Reads ``azure_endpoint`` and ``model`` (deployment name) from ``settings``.
    ``dimension`` defaults to ``768`` if not provided.
    """

    def __init__(self, settings: Any) -> None:
        self.endpoint = settings.embedding.get("azure_endpoint", "")
        self.model = settings.embedding.get("model", "text-embedding-3-small")
        self.dimension = int(settings.embedding.get("dimension", 768))
        self.api_key = settings.embedding.get("api_key")

    def embed(self, texts: List[str], trace: Any = None, **_: Any) -> List[List[float]]:
        self.validate_texts(texts)
        # Stub implementation – returns zero vectors of the configured dimension.
        return [[0.0] * self.dimension for _ in texts]
