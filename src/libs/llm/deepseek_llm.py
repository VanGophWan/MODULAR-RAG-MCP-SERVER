"""DeepSeek LLM stub implementation.

Provides a deterministic ``chat`` method for unit testing without external calls.
The class reads ``api_key`` and ``model`` from the supplied ``Settings`` instance.
"""

from __future__ import annotations

from typing import Any, List

from ..base_llm import BaseLLM, ChatMessage


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM provider stub.

    The real DeepSeek API is not invoked – this class only stores configuration and
    returns a predictable response useful for test suites.
    """

    def __init__(self, settings: Any, base_url: str | None = None, **_: Any) -> None:
        cfg = settings.llm
        self.api_key = cfg.get("api_key", "")
        self.model = cfg.get("model", "")
        # ``base_url`` is not used for the stub but accepted for API compatibility.
        self.base_url = base_url or cfg.get("base_url", "https://api.deepseek.com")

    def chat(self, messages: List[ChatMessage | dict[str, Any]]) -> str:
        """Return a deterministic placeholder response.

        The output contains the provider name, model and the number of messages.
        """
        normalized = self._normalize_messages(messages)
        return f"[deepseek:{self.model}] placeholder response (messages={len(normalized)})"

    @staticmethod
    def _normalize_messages(messages: List[ChatMessage | dict[str, Any]]) -> List[dict[str, str]]:
        """Same normalisation logic as other LLM providers."""
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
