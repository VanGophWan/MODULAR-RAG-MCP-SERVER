"""OpenAI-compatible LLM implementation.

Implements a deterministic ``chat`` method that does not perform real network calls.
The constructor reads ``api_key`` and ``model`` from the provided ``Settings`` instance.
``base_url`` defaults to the OpenAI API endpoint if not supplied.
"""

from __future__ import annotations

from typing import Any, List

from .base_llm import BaseLLM, ChatMessage


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider.

    Parameters are read from ``settings.llm``:
    - ``api_key`` – OpenAI API key (optional for stub)
    - ``model`` – model identifier (e.g., gpt-4, gpt-3.5-turbo)
    - ``base_url`` – optional custom endpoint; defaults to OpenAI API.
    """

    def __init__(self, settings: Any, base_url: str | None = None, **_: Any) -> None:
        cfg = settings.llm
        self.api_key = cfg.get("api_key", "")
        self.model = cfg.get("model", "")
        # In a real implementation this would be the OpenAI API endpoint.
        self.base_url = base_url or cfg.get("base_url", "https://api.openai.com")

    def chat(self, messages: List[ChatMessage | dict[str, Any]]) -> str:
        """Return a deterministic string for testing.

        The format mirrors other providers – it includes the provider name, model, and the
        number of messages supplied. No external request is made.
        """
        normalized = self._normalize_messages(messages)
        # Deterministic placeholder response.
        return f"[openai:{self.model}] placeholder response (messages={len(normalized)})"

    @staticmethod
    def _normalize_messages(messages: List[ChatMessage | dict[str, Any]]) -> List[dict[str, str]]:
        """Convert ``ChatMessage`` objects or dicts into the canonical dict format.

        This mirrors the helper in other LLM providers to keep behaviour consistent.
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