from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.settings import Settings

from .base_llm import BaseLLM, ChatMessage


class LLMFactoryError(ValueError):
    """Raised when LLMFactory cannot create a requested provider."""


@dataclass(slots=True)
class PlaceholderLLM(BaseLLM):
    provider: str
    model: str

    def chat(self, messages: list[ChatMessage | dict]) -> str:
        # Placeholder implementation for B1: concrete providers land in B7.
        return f"[{self.provider}:{self.model}] placeholder response"


class LLMFactory:
    _default_builders: dict[str, Callable[[Settings], BaseLLM]] = {
        "azure": lambda s: PlaceholderLLM("azure", s.llm.get("model", "unknown")),
        "openai": lambda s: PlaceholderLLM("openai", s.llm.get("model", "unknown")),
        "ollama": lambda s: PlaceholderLLM("ollama", s.llm.get("model", "unknown")),
        "deepseek": lambda s: PlaceholderLLM("deepseek", s.llm.get("model", "unknown")),
    }

    @classmethod
    def create(
        cls,
        settings: Settings,
        registry: dict[str, Callable[[Settings], BaseLLM]] | None = None,
    ) -> BaseLLM:
        provider = str(settings.llm.get("provider", "")).strip().lower()
        if not provider:
            raise LLMFactoryError("Missing required field: llm.provider")

        builders = registry or cls._default_builders
        builder = builders.get(provider)
        if builder is None:
            supported = ", ".join(sorted(builders.keys()))
            raise LLMFactoryError(
                f"Unsupported llm.provider: {provider}. Supported providers: {supported}"
            )

        return builder(settings)
