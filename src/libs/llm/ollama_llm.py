"""Ollama LLM stub implementation.

Provides a deterministic ``chat`` method suitable for unit tests. The implementation
does not perform real HTTP requests; it simply echoes the supplied messages in a
predictable format.
"""

from __future__ import annotations

from typing import Any, List

from ..base_llm import BaseLLM, ChatMessage


class OllamaLLM(BaseLLM):
    """Ollama LLM provider stub.

    Reads ``base_url`` (default ``http://localhost:11434``) and ``model`` from the
    ``settings.llm`` mapping. No network traffic is performed.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, settings: Any, base_url: str | None = None, **_: Any) -> None:
        cfg = settings.llm
        self.api_key = cfg.get("api_key", "")  # Ollama typically does not need an API key.
        self.model = cfg.get("model", "")
        self.base_url = base_url or cfg.get("base_url", self.DEFAULT_BASE_URL)

    def chat(self, messages: List[ChatMessage | dict[str, Any]]) -> str:
        """Return a deterministic placeholder response.

        The format mirrors other providers: ``[ollama:<model>] placeholder response``.
        """
        normalized = self._normalize_messages(messages)
        return f"[ollama:{self.model}] placeholder response (messages={len(normalized)})"

    @staticmethod
    def _normalize_messages(messages: List[ChatMessage | dict[str, Any]]) -> List[dict[str, str]]:
        """Normalise messages to a ``{'role': ..., 'content': ...}`` dict.

        This helper is identical to the one used in ``OpenAILLM`` for consistency.
        """
        normalized = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                normalized.append({"role": msg.role, "content": msg.content})
            else:
                raise ValueError(f"Invalid message format: {msg}")
        return normalized
