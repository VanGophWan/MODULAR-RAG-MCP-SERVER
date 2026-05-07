"""Factory for creating Vision LLM provider instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.settings import Settings

from .base_vision_llm import BaseVisionLLM


class VisionLLMFactoryError(ValueError):
    """Raised when VisionLLMFactory cannot create a requested provider."""


@dataclass(slots=True)
class PlaceholderVisionLLM(BaseVisionLLM):
    """Placeholder Vision LLM used when a concrete provider is not yet implemented."""

    provider: str
    model: str

    def caption(self, images: list, text: str | None = None) -> str:
        return f"[{self.provider}:{self.model}] placeholder caption"


class VisionLLMFactory:
    _default_builders: dict[str, Callable[[Settings], BaseVisionLLM]] = {
        "azure": lambda s: __import__(
            "libs.llm.azure_vision_llm", fromlist=["AzureVisionLLM"]
        ).AzureVisionLLM(s),
        "openai": lambda s: PlaceholderVisionLLM("openai", s.vision_llm.get("model", "unknown")),
        "ollama": lambda s: PlaceholderVisionLLM("ollama", s.vision_llm.get("model", "unknown")),
        "deepseek": lambda s: PlaceholderVisionLLM("deepseek", s.vision_llm.get("model", "unknown")),
    }

    @classmethod
    def create(
        cls,
        settings: Settings,
        registry: dict[str, Callable[[Settings], BaseVisionLLM]] | None = None,
    ) -> BaseVisionLLM:
        provider = str(settings.vision_llm.get("provider", "")).strip().lower()
        if not provider:
            raise VisionLLMFactoryError("Missing required field: vision_llm.provider")

        builders = registry or cls._default_builders
        builder = builders.get(provider)
        if builder is None:
            supported = ", ".join(sorted(builders.keys()))
            raise VisionLLMFactoryError(
                f"Unsupported vision_llm.provider: {provider}. Supported: {supported}"
            )

        return builder(settings)