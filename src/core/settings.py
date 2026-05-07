from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class SettingsError(ValueError):
    """Raised when settings loading or validation fails."""


@dataclass(slots=True)
class Settings:
    llm: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)
    vector_store: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    rerank: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)
    splitter: dict[str, Any] = field(default_factory=dict)
    vision_llm: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def _get_nested(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise SettingsError(f"Missing required field: {path}")
        current = current[key]
    return current


def validate_settings(settings: Settings) -> None:
    data = {
        "llm": settings.llm,
        "embedding": settings.embedding,
        "vector_store": settings.vector_store,
        "retrieval": settings.retrieval,
        "rerank": settings.rerank,
        "evaluation": settings.evaluation,
        "observability": settings.observability,
        "vision_llm": settings.vision_llm,
    }

    required_fields = [
        "llm.provider",
        "llm.model",
        "embedding.provider",
        "embedding.model",
        "vector_store.backend",
        "retrieval.sparse_backend",
        "retrieval.fusion_algorithm",
        "rerank.backend",
        "evaluation.backends",
        "observability.enabled",
        "vision_llm.provider",
        "vision_llm.model",
    ]

    for field_path in required_fields:
        _get_nested(data, field_path)


def load_settings(path: str) -> Settings:
    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    try:
        raw = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SettingsError(f"Invalid YAML in settings file: {exc}") from exc

    if not isinstance(raw, dict):
        raise SettingsError("Settings root must be a mapping object")

    settings = Settings(
        llm=raw.get("llm") or {},
        embedding=raw.get("embedding") or {},
        vector_store=raw.get("vector_store") or {},
        retrieval=raw.get("retrieval") or {},
        rerank=raw.get("rerank") or {},
        evaluation=raw.get("evaluation") or {},
        observability=raw.get("observability") or {},
        vision_llm=raw.get("vision_llm") or {},
        extras={k: v for k, v in raw.items() if k not in {
            "llm",
            "embedding",
            "vector_store",
            "retrieval",
            "rerank",
            "evaluation",
            "observability",
            "vision_llm",
        }},
    )
    validate_settings(settings)
    return settings
