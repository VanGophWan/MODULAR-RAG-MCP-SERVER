"""Tests for the ``SplitterFactory`` routing logic.

The tests mimic the pattern used for ``LLMFactory`` and ``EmbeddingFactory``.
They verify that the factory correctly selects a provider from a custom registry,
and that missing/unknown providers raise the appropriate error.
"""

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

from core.settings import Settings  # noqa: E402
from libs.splitter.base_splitter import BaseSplitter  # noqa: E402
from libs.splitter.splitter_factory import (
    SplitterFactory,
    SplitterFactoryError,
)  # noqa: E402


class FakeSplitter(BaseSplitter):
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def split_text(self, text: str, trace=None, **_: any):  # type: ignore[override]
        # Simple deterministic split: return a list with the tag repeated.
        return [self.tag for _ in range(len(text))]


def _settings(provider: str = "placeholder") -> Settings:
    return Settings(
        llm={},
        embedding={},
        vector_store={},
        retrieval={},
        rerank={},
        evaluation={},
        observability={},
        extras={},
        # Include splitter section for this test
        splitter={"provider": provider},
    )


def test_splitter_factory_routes_via_registry() -> None:
    registry = {
        "fake_a": lambda _: FakeSplitter("A"),
        "fake_b": lambda _: FakeSplitter("B"),
    }
    settings = _settings("fake_b")
    splitter = SplitterFactory.create(settings, registry=registry)
    assert isinstance(splitter, FakeSplitter)
    # Verify deterministic split length matches input length
    result = splitter.split_text("hello")
    assert len(result) == 5
    assert all(chunk == "B" for chunk in result)


def test_splitter_factory_missing_provider_raises() -> None:
    settings = _settings("")
    with pytest.raises(SplitterFactoryError, match=r"splitter\.provider"):
        SplitterFactory.create(settings)


def test_splitter_factory_unknown_provider_raises() -> None:
    settings = _settings("unknown")
    with pytest.raises(SplitterFactoryError, match="Unsupported splitter.provider"):
        SplitterFactory.create(settings)
