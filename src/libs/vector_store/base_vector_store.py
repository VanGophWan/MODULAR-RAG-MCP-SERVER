"""Abstract base class for vector store backends.

Implementations must provide ``upsert`` (store records) and ``query`` (retrieve
nearest vectors).  The contract is deliberately minimal – enough for downstream
code to rely on the interface while we build concrete backends later (e.g.,
ChromaStore in B7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Dict


class BaseVectorStore(ABC):
    """Contract for a vector store.

    * ``upsert`` stores a list of records (each record is a dict containing at
      least a ``vector`` field and optional metadata).
    * ``query`` returns the top‑k records closest to a query vector.  The exact
      similarity metric is implementation‑specific.
    """

    @abstractmethod
    def upsert(self, records: List[Dict[str, Any]], trace: Any = None, **kwargs: Any) -> None:
        """Insert or replace ``records`` in the store.

        Args:
            records: List of dicts, each containing a ``vector`` key.
            trace: Optional ``TraceContext`` for observability.
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Dict[str, Any] | None = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Return the ``top_k`` nearest records to ``vector``.

        Args:
            vector: Query embedding.
            top_k: Number of results to return.
            filters: Optional metadata filters.
            trace: Optional ``TraceContext``.
        Returns:
            List of record dicts (same shape as stored).
        """
        raise NotImplementedError
