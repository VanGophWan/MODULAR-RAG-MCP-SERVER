"""Abstract base class for text splitters.

All concrete splitters must inherit from :class:`BaseSplitter` and implement the
``split_text`` method, which receives a raw string and returns a list of string
chunks.  The method optionally accepts a ``TraceContext`` for observability –
concrete implementations may ignore it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Any


class BaseSplitter(ABC):
    """Abstract interface for splitting a piece of text into chunks.

    Implementations should ensure that the returned list covers the entire input
    and that each chunk is a non‑empty string.
    """

    @abstractmethod
    def split_text(self, text: str, trace: Any = None, **kwargs: Any) -> List[str]:
        """Split *text* into a list of chunks.

        Args:
            text: The raw text to split.
            trace: Optional ``TraceContext`` for observability; may be ignored.
            **kwargs: Provider‑specific options.
        Returns:
            List of string chunks.
        """
        raise NotImplementedError

    def validate_text(self, text: str) -> None:
        """Validate that *text* is a non‑empty string.

        Raises:
            ValueError: If *text* is empty or not a string.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non‑empty string")
