"""Abstract base class for reranking candidates.

Implementations must provide a ``rerank`` method that takes a query and a list
of candidate items and returns them ordered by relevance. The placeholder
implementation ``NoneReranker`` (in the factory module) simply returns the
candidates unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class BaseReranker(ABC):
    """Contract for a reranker.

    ``rerank`` should be pure‑functional – given the same inputs it must always
    return the same ordering.
    """

    @abstractmethod
    def rerank(
        self, query: Any, candidates: List[Any], trace: Any = None, **kwargs: Any
    ) -> List[Any]:
        """Return ``candidates`` ordered by relevance to ``query``.

        Args:
            query: The original query string or embedding.
            candidates: List of items to rank.
            trace: Optional ``TraceContext``.
        Returns:
            Re‑ordered list of candidates.
        """
        raise NotImplementedError
