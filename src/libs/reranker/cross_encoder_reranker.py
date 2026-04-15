"""Cross‑Encoder reranker placeholder.

In a production setting this class would load a cross‑encoder model (e.g., a
Sentence‑Transformers model) to score a set of candidate passages. For the unit
tests we provide a deterministic stub that returns the candidates as‑is.
"""

from __future__ import annotations

from typing import Any, List

from .base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Reranker that would use a cross‑encoder model – stub for testing.

    The constructor accepts ``settings`` for future extensibility but does not
    load any model.
    """

    def __init__(self, settings: Any, **_: Any) -> None:
        self.settings = settings
        self.model_name = settings.rerank.get("model", "cross_encoder_stub")

    def rerank(self, query: Any, candidates: List[Any], trace: Any = None, **_: Any) -> List[Any]:
        """Return ``candidates`` unchanged.

        A real implementation would score each candidate with the cross‑encoder and
        sort the list. The stub provides deterministic behaviour for tests.
        """
        return candidates
