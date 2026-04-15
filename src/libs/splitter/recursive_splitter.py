"""Recursive text splitter implementation.

A lightweight implementation that recursively splits a document into smaller chunks
based on a maximum chunk size. The algorithm tries to split on paragraph
boundaries (double new‑lines) first, then on single new‑lines, and finally on
spaces if the resulting pieces are still too large.

This implementation satisfies the :class:`BaseSplitter` contract without pulling
in external dependencies such as LangChain, keeping the repository self‑contained
and fast for unit tests.
"""

from __future__ import annotations

from typing import List, Any

from .base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """Split text recursively into chunks of at most ``max_length`` characters.

    Parameters
    ----------
    max_length: int, optional
        Desired maximum length of each chunk. Defaults to 500 characters, which
        works well for typical LLM context windows while keeping chunks
        manageable for downstream processing.
    """

    def __init__(self, max_length: int = 500):
        self.max_length = max_length

    def split_text(self, text: str, trace: Any = None, **_: Any) -> List[str]:
        self.validate_text(text)
        # Initial split on double‑newlines (paragraphs).
        chunks = self._split_on_delimiter(text, delimiter="\n\n")
        # Further split any chunk larger than max_length.
        final_chunks: List[str] = []
        for chunk in chunks:
            if len(chunk) <= self.max_length:
                final_chunks.append(chunk.strip())
            else:
                # Try splitting on single newline (lines).
                sub_chunks = self._split_on_delimiter(chunk, delimiter="\n")
                for sub in sub_chunks:
                    if len(sub) <= self.max_length:
                        final_chunks.append(sub.strip())
                    else:
                        # Fallback split on spaces.
                        final_chunks.extend(self._split_on_spaces(sub))
        # Remove empty strings.
        return [c for c in final_chunks if c]

    def _split_on_delimiter(self, text: str, delimiter: str) -> List[str]:
        """Split *text* using *delimiter*; preserve the delimiter at the end of each chunk.

        The delimiter itself is not kept – only the resulting pieces are returned.
        """
        return [part for part in text.split(delimiter) if part]

    def _split_on_spaces(self, text: str) -> List[str]:
        """Split *text* into roughly ``max_length`` sized pieces on word boundaries.
        """
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for w in words:
            # +1 for space when joining later (except first word).
            projected = current_len + len(w) + (1 if current else 0)
            if projected > self.max_length and current:
                chunks.append(" ".join(current))
                current = [w]
                current_len = len(w)
            else:
                current.append(w)
                current_len = projected
        if current:
            chunks.append(" ".join(current))
        return chunks
