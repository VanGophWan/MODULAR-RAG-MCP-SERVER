from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: list[ChatMessage | dict[str, Any]]) -> str:
        """Generate a text response from input messages."""
