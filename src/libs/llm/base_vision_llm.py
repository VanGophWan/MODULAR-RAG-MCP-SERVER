"""Abstract base class for Vision LLM providers.

Vision LLMs support multimodal input (text + images) and are used for tasks
like image captioning in the Ingestion pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ImageContent:
    """Represents an image input for Vision LLM."""

    image_path: str | None = None
    image_bytes: bytes | None = None
    image_url: str | None = None


class BaseVisionLLM(ABC):
    """Abstract interface for Vision LLM providers.

    Subclasses must implement ``caption`` for image-to-text generation.
    """

    @abstractmethod
    def caption(self, images: list[ImageContent], text: str | None = None) -> str:
        """Generate a caption/description for the given images.

        Args:
            images: List of ImageContent objects containing image data.
            text: Optional supplemental text prompt.

        Returns:
            Generated caption/description as a string.
        """
        raise NotImplementedError