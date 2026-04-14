"""Embedding abstract base class and utilities.

All concrete embedding providers must inherit from :class:`BaseEmbedding` and implement
the :meth:`embed` method, which receives a list of texts and returns a list of
vector embeddings (list of float lists).  The method optionally accepts a
``TraceContext`` for observability – the concrete implementations may ignore it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Any


class BaseEmbedding(ABC):
    """Abstract interface for vector embedding providers.

    Implementations must provide a deterministic ``embed`` method that returns a
    list of ``float`` vectors, one per input text.
    """

    @abstractmethod
    def embed(
        self, texts: List[str], trace: Optional[Any] = None, **kwargs: Any
    ) -> List[List[float]]:
        """Return embeddings for *texts*.

        Args:
            texts: List of non‑empty strings to embed.
            trace: Optional ``TraceContext`` for tracing; may be ignored.
            **kwargs: Provider‑specific options.
        """
        raise NotImplementedError

    def validate_texts(self, texts: List[str]) -> None:
        """Validate that *texts* is a non‑empty list of non‑empty strings.
        Raises:
            ValueError: If validation fails.
        """
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non‑empty list of strings")
        for t in texts:
            if not isinstance(t, str) or not t:
                raise ValueError("each text must be a non‑empty string")
