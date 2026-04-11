from pathlib import Path
import sys

import pytest


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_add_src_to_path()

from core.settings import SettingsError, load_settings  # noqa: E402


def test_load_settings_success() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings_path = project_root / "config" / "settings.yaml"
    settings = load_settings(str(settings_path))

    assert settings.llm["provider"] == "azure"
    assert settings.embedding["provider"] == "openai"


def test_load_settings_missing_required_field_raises() -> None:
    import tempfile
    import yaml

    data = {
        "llm": {"provider": "azure", "model": "gpt-4o"},
        "embedding": {"model": "text-embedding-3-small"},
        "vector_store": {"backend": "chroma"},
        "retrieval": {
            "sparse_backend": "bm25",
            "fusion_algorithm": "rrf",
        },
        "rerank": {"backend": "cross_encoder"},
        "evaluation": {"backends": ["ragas"]},
        "observability": {"enabled": True},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml.safe_dump(data))
        temp_path = f.name

    try:
        with pytest.raises(SettingsError, match=r"embedding\.provider"):
            load_settings(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)
