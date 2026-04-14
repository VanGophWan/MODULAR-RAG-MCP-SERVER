"""Factory for creating concrete vector store implementations.

The design follows the same pattern as the other factories in the project.
A ``PlaceholderVectorStore`` provides an in‑memory store for early testing.
Later phases (B7) will register real backends such as Chroma.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, List, Dict

from core.settings import Settings


class VectorStoreFactoryError(ValueError):
    """Raised when ``VectorStoreFactory`` cannot create the requested backend."""


@dataclass(slots=True)
class PlaceholderVectorStore:
    """In‑memory placeholder vector store.

    Records are stored in a simple list. ``query`` returns the first ``top_k``
    records (deterministic for tests). This satisfies the contract without any
    external dependencies.
    """

    _records: List[Dict[str, Any]] = field(default_factory=list)

    def upsert(self, records: List[Dict[str, Any]], trace: Any = None, **_: Any) -> None:
        self._records.extend(records)

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Dict[str, Any] | None = None,
        trace: Any = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        # Simple deterministic slice; ignore vector/filters for placeholder.
        return self._records[:top_k]


class VectorStoreFactory:
    _default_builders: dict[str, Callable[[Settings], Any]] = {
        "none": lambda s: PlaceholderVectorStore(),
        # Real backends (e.g., "chroma") will be added in B7.
    }

    @classmethod
    def create(
        cls,
        settings: Settings,
        registry: dict[str, Callable[[Settings], Any]] | None = None,
    ) -> Any:
        """Create a concrete ``BaseVectorStore`` implementation.

        Args:
            settings: Settings instance containing ``vector_store.backend``.
            registry: Optional custom registry.
        Returns:
            An instance of a concrete vector store.
        Raises:
            VectorStoreFactoryError: If the backend is missing or unsupported.
        """
        provider = str(settings.vector_store.get("backend", "")).strip().lower()
        if not provider:
            raise VectorStoreFactoryError("Missing required field: vector_store.backend")
        builders = registry or cls._default_builders
        builder = builders.get(provider)
        if builder is None:
            supported = ", ".join(sorted(builders.keys()))
            raise VectorStoreFactoryError(
                f"Unsupported vector_store.backend: {provider}. Supported: {supported}"
            )
        return builder(settings)
