"""Simple in‑memory Chroma‑like vector store.

The real project would depend on the ``chromadb`` package to provide a persistent
vector database. For the purpose of the test suite we implement a lightweight
in‑memory store that satisfies the same public API:

- ``upsert(records)`` stores a list of records where each record is a mapping
  containing at least ``id`` and ``vector`` keys.
- ``query(vector, top_k, filters=None)`` returns the ``top_k`` records ordered by
  Euclidean distance to the query vector. ``filters`` are ignored in this stub.

If a ``persist_path`` is supplied in ``settings.vector_store``, the store will
write its internal list to a JSON file on each upsert and reload it on
initialisation. This mimics the persistence behaviour required by the B7
spec without pulling in external dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .base_vector_store import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """In‑memory vector store with optional JSON persistence.

    The implementation focuses on correctness for unit/integration tests rather
    than performance. It stores records as a list of dictionaries. Each record
    must contain an ``id`` (hashable) and a ``vector`` (list of floats). Additional
    keys are preserved unchanged.
    """

    def __init__(self, settings: Any):
        cfg = settings.vector_store
        self.persist_path = cfg.get("persist_path")
        self._records: List[Dict[str, Any]] = []
        if self.persist_path:
            self._load()

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------
    def _load(self) -> None:
        """Load records from ``persist_path`` if the file exists.

        The method expects a JSON file with a list of record dictionaries.
        Errors are ignored silently – an empty store is used on failure.
        """
        try:
            path = Path(self.persist_path)
            if path.is_file():
                with path.open("r", encoding="utf-8") as f:
                    self._records = json.load(f)
        except Exception:
            self._records = []

    def _save(self) -> None:
        """Write the current records to ``persist_path`` as JSON.
        """
        if not self.persist_path:
            return
        try:
            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(self._records, f, ensure_ascii=False, indent=2)
        except Exception:
            # Persistence failures should not halt normal operation in tests.
            pass

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def upsert(self, records: List[Dict[str, Any]], trace: Any = None, **_: Any) -> None:
        """Add or replace records in the store.

        If a record with the same ``id`` already exists it is replaced; otherwise
        it is appended.
        """
        for rec in records:
            rec_id = rec.get("id")
            # Remove any existing record with the same id.
            self._records = [r for r in self._records if r.get("id") != rec_id]
            self._records.append(rec)
        self._save()

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Dict[str, Any] | None = None,
        trace: Any = None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        """Return the ``top_k`` nearest records based on Euclidean distance.

        ``filters`` are ignored – they exist for API compatibility.
        """
        # Guard against empty store.
        if not self._records:
            return []
        # Compute squared Euclidean distance for each record.
        def _dist(rec: Dict[str, Any]) -> float:
            vec = rec.get("vector", [])
            # Pad shorter vectors with zeros to avoid IndexError.
            if len(vec) != len(vector):
                # Simple handling: truncate or pad with zeros.
                min_len = min(len(vec), len(vector))
                vec = vec[:min_len]
                q_vec = vector[:min_len]
            else:
                q_vec = vector
            return sum((a - b) ** 2 for a, b in zip(vec, q_vec))

        # Sort by distance ascending.
        sorted_records = sorted(self._records, key=_dist)
        return sorted_records[:top_k]
