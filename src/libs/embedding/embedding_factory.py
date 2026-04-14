"""Factory for creating concrete embedding providers.

The pattern mirrors ``LLMFactory`` – a registry maps a provider name to a callable that
instantiates the concrete class.  ``EmbeddingFactory.create`` reads the ``embedding.provider``
value from the supplied ``Settings`` and returns an instance of the matching
provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from core.settings import Settings

# NOTE: Import concrete providers lazily inside ``_default_builders`` to avoid
# circular imports.  The imports succeed because the modules exist (they are
# currently empty placeholders).


class EmbeddingFactoryError(ValueError):
    """Raised when ``EmbeddingFactory`` cannot create the requested provider."""


@dataclass(slots=True)
class PlaceholderEmbedding:
    """Fallback stub used when a concrete provider is not implemented yet.

    It simply returns a zero‑vector of a configurable dimension.
    """

    dimension: int = 768

    def embed(self, texts: list[str], trace: Any = None, **_: Any) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]


class EmbeddingFactory:
    # Default registry – concrete classes are imported lazily to keep import time low.
    _default_builders: dict[str, Callable[[Settings], Any]] = {
        "openai": lambda s: __import__("libs.embedding.openai_embedding", fromlist=["OpenAIEmbedding"]).OpenAIEmbedding(s),
        "azure": lambda s: __import__("libs.embedding.azure_embedding", fromlist=["AzureEmbedding"]).AzureEmbedding(s),
        "ollama": lambda s: __import__("libs.embedding.ollama_embedding", fromlist=["OllamaEmbedding"]).OllamaEmbedding(s),
    }

    @classmethod
    def create(
        cls,
        settings: Settings,
        registry: dict[str, Callable[[Settings], Any]] | None = None,
    ) -> Any:
        """Create a concrete ``BaseEmbedding`` implementation.

        Args:
            settings: Loaded ``Settings`` instance.
            registry: Optional custom registry mapping provider names to callables.
        Returns:
            An instance of a concrete ``BaseEmbedding`` subclass.
        Raises:
            EmbeddingFactoryError: If ``embedding.provider`` is missing or unsupported.
        """
        provider = str(settings.embedding.get("provider", "")).strip().lower()
        if not provider:
            raise EmbeddingFactoryError("Missing required field: embedding.provider")

        builders = registry or cls._default_builders
        builder = builders.get(provider)
        if builder is None:
            supported = ", ".join(sorted(builders.keys()))
            raise EmbeddingFactoryError(
                f"Unsupported embedding.provider: {provider}. Supported providers: {supported}"
            )
        return builder(settings)