"""Azure-compatible LLM implementation.

Implements a deterministic ``chat`` method that does not perform real network calls.
The constructor reads ``api_key`` and ``model`` (or deployment name) from the
provided ``Settings`` instance. ``base_url`` defaults to the Azure OpenAI endpoint
if not supplied.
"""

from __future__ import annotations

from typing import Any, List

from ..base_llm import BaseLLM, ChatMessage


class AzureLLM(BaseLLM):
    """Azure OpenAI LLM provider.

    Parameters are read from ``settings.llm``:
    - ``api_key`` – Azure OpenAI API key (optional for stub)
    - ``model`` – deployment name used as the model identifier
    - ``base_url`` – optional custom endpoint; defaults to a placeholder URL.
    """

    def __init__(self, settings: Any, base_url: str | None = None, **_: Any) -> None:
        cfg = settings.llm
        self.api_key = cfg.get("api_key", "")
        self.model = cfg.get("model", "")
        # In a real implementation this would be the Azure resource endpoint.
        self.base_url = base_url or cfg.get("azure_endpoint", "https://example.azure.openai.com")

    def chat(self, messages: List[ChatMessage | dict[str, Any]]) -> str:
        """Return a deterministic string for testing.

        The format mirrors ``OpenAILLM`` – it includes the provider name, model, and the
        number of messages supplied. No external request is made.
        """
        normalized = self._normalize_messages(messages)
        # Deterministic placeholder response.
        return f"[azure:{self.model}] placeholder response (messages={len(normalized)})"

    @staticmethod
    def _normalize_messages(messages: List[ChatMessage | dict[str, Any]]) -> List[dict[str, str]]:
        """Convert ``ChatMessage`` objects or dicts into the canonical dict format.

        This mirrors the helper in ``OpenAILLM`` to keep behaviour consistent across
        providers.
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
