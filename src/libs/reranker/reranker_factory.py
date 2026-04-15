"""Factory for creating concrete reranker backends.

The pattern mirrors the other factories in the project. It reads
``settings.rerank.backend`` and returns an instance of a class that implements
:class:`BaseReranker`. Two concrete backends are provided in this repository:

- ``llm`` – an LLM‑based reranker (stub implementation).
- ``cross_encoder`` – a cross‑encoder‑based reranker (stub implementation).

If the configured backend is missing or unsupported, a clear ``RerankerFactoryError``
is raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from core.settings import Settings

from .base_reranker import BaseReranker


class RerankerFactoryError(ValueError):
    """Raised when the factory cannot create a requested reranker backend."""


@dataclass(slots=True)
class PlaceholderReranker(BaseReranker):
    """Fallback implementation that returns candidates unchanged.

    Used when no concrete backend is configured. It satisfies the contract of
    :class:`BaseReranker` without performing any computation.
    """

    def rerank(self, query: Any, candidates: list[Any], trace: Any = None, **_: Any) -> list[Any]:
        return candidates


class RerankerFactory:
    """Factory for reranker backends.

    The default registry lazily imports concrete implementations to avoid circular
    imports. Users may supply a custom ``registry`` mapping provider names to callables.
    """

    _default_builders: dict[str, Callable[[Settings], BaseReranker]] = {
        "none": lambda s: PlaceholderReranker(),
        "llm": lambda s: __import__("libs.reranker.llm_reranker", fromlist=["LLMReranker"]).LLMReranker(s),
        "cross_encoder": lambda s: __import__("libs.reranker.cross_encoder_reranker", fromlist=["CrossEncoderReranker"]).CrossEncoderReranker(s),
    }

    @classmethod
    def create(
        cls,
        settings: Settings,
        registry: dict[str, Callable[[Settings], BaseReranker]] | None = None,
    ) -> BaseReranker:
        """Instantiate the configured reranker backend.

        Args:
            settings: The global ``Settings`` instance containing ``rerank.backend``.
            registry: Optional custom registry mapping backend names to callables.
        Returns:
            An instance of a concrete ``BaseReranker`` subclass.
        Raises:
            RerankerFactoryError: If the ``backend`` is missing or unsupported.
        """
        provider = str(settings.rerank.get("backend", "")).strip().lower()
        if not provider:
            raise RerankerFactoryError("Missing required field: rerank.backend")
        builders = registry or cls._default_builders
        builder = builders.get(provider)
        if builder is None:
            supported = ", ".join(sorted(builders.keys()))
            raise RerankerFactoryError(
                f"Unsupported rerank.backend: {provider}. Supported: {supported}"
            )
        return builder(settings)
