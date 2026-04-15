"""Factory for creating concrete splitter implementations.

The design mirrors other factories in the project (LLMFactory, EmbeddingFactory).
It reads ``settings.splitter.provider`` and returns an instance of a class that
inherits from :class:`BaseSplitter`.  A ``PlaceholderSplitter`` is provided as a
fallback implementation that simply returns the original text as a single
chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from core.settings import Settings

# Import the placeholder lazily to avoid circular imports.


class SplitterFactoryError(ValueError):
    """Raised when ``SplitterFactory`` cannot create the requested provider."""


@dataclass(slots=True)
class PlaceholderSplitter:
    """A trivial splitter that returns the input text unchanged.

    This implementation satisfies the contract for testing and for early phases
    of development before a real splitter (e.g., RecursiveSplitter) is added.
    """

    def split_text(self, text: str, trace: Any = None, **_: Any) -> list[str]:  # type: ignore[override]
        return [text]


class SplitterFactory:
    # Default registry – concrete implementations are imported lazily.
    _default_builders: dict[str, Callable[[Settings], Any]] = {
        "placeholder": lambda s: PlaceholderSplitter(),
        "recursive": lambda s: __import__("libs.splitter.recursive_splitter", fromlist=["RecursiveSplitter"]).RecursiveSplitter()
        # Real implementations (recursive, fixed_length, etc.) will be added
        # in later phases (B7).
    }

    @classmethod
    def create(
        cls,
        settings: Settings,
        registry: dict[str, Callable[[Settings], Any]] | None = None,
    ) -> Any:
        """Create a concrete :class:`BaseSplitter` implementation.

        Args:
            settings: The loaded ``Settings`` object.
            registry: Optional custom registry mapping provider names to callables.
        Returns:
            An instance of a concrete splitter.
        Raises:
            SplitterFactoryError: If the ``splitter.provider`` setting is missing
                or does not map to a registered implementation.
        """
        provider = str(settings.splitter.get("provider", "")).strip().lower()
        if not provider:
            raise SplitterFactoryError("Missing required field: splitter.provider")

        builders = registry or cls._default_builders
        builder = builders.get(provider)
        if builder is None:
            supported = ", ".join(sorted(builders.keys()))
            raise SplitterFactoryError(
                f"Unsupported splitter.provider: {provider}. Supported: {supported}"
            )
        return builder(settings)
