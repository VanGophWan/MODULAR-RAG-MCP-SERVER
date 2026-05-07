"""Azure Vision LLM implementation using GPT-4o.

Implements a deterministic ``caption`` method that does not perform real network calls.
The constructor reads ``api_key`` and ``model`` from the provided ``Settings`` instance.
"""

from __future__ import annotations

from typing import Any

from .base_vision_llm import BaseVisionLLM, ImageContent


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision LLM provider.

    Used for multimodal tasks such as image captioning in the Ingestion pipeline.

    Parameters are read from ``settings.vision_llm``:
    - ``api_key`` – Azure OpenAI API key (optional for stub)
    - ``model`` – deployment name used as the model identifier (e.g., gpt-4o)
    - ``azure_endpoint`` – optional custom endpoint; defaults to Azure OpenAI endpoint.
    """

    def __init__(self, settings: Any, **_: Any) -> None:
        cfg = settings.vision_llm
        self.api_key = cfg.get("api_key", "")
        self.model = cfg.get("model", "gpt-4o")
        self.azure_endpoint = cfg.get(
            "azure_endpoint", "https://example.azure.openai.com"
        )

    def caption(self, images: list[ImageContent], text: str | None = None) -> str:
        """Return a deterministic caption for testing.

        The format includes the provider name, model, image count, and optional text.
        No external request is made.
        """
        image_count = len(images)
        prompt_hint = f", prompt: {text}" if text else ""
        return (
            f"[azure_vision:{self.model}] placeholder caption "
            f"for {image_count} image(s){prompt_hint}"
        )