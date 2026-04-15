"""LLM‑based reranker placeholder.

The real implementation would invoke an LLM with a prompt (e.g., from
``config/prompts/rerank.txt``) to obtain relevance scores for a list of candidate
IDs. For the purpose of the current test suite we provide a deterministic stub
that simply returns the candidates unchanged, which satisfies the contract of
:class:`BaseReranker`.
"""

from __future__ import annotations

from typing import Any, List

from .base_reranker import BaseReranker


class LLMReranker(BaseReranker):
    """Reranker that would call an LLM – stub implementation for testing.

    Parameters
    ----------
    settings: Settings
        The global settings object; currently unused but kept for API
        compatibility.
    """

    def __init__(self, settings: Any, **_: Any) -> None:
        # In a full implementation we would load a prompt template here.
        self.prompt_template = """Rank the following candidates for the query:\n{query}\nCandidates: {candidates}\nReturn a JSON list of IDs in order of relevance."""
        self.settings = settings

    def rerank(self, query: Any, candidates: List[Any], trace: Any = None, **_: Any) -> List[Any]:
        """Return ``candidates`` unchanged.

        A real implementation would score each candidate using an LLM and sort
        accordingly. For now we keep the behaviour deterministic and side‑effect
        free to simplify testing.
        """
        return candidates
