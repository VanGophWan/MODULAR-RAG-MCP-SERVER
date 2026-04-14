"""Tests for the ``EmbeddingFactory`` routing logic.

The pattern mirrors ``tests/unit/test_llm_factory.py`` – we provide a minimal
``Settings`` instance, override the factory's ``_default_builders`` with simple
fakes, and verify that the correct class is instantiated.  Additionally we
exercise error handling for missing or unknown providers.
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
from libs.embedding.base_embedding import BaseEmbedding  # noqa: E402
from libs.embedding.embedding_factory import (
    EmbeddingFactory,
    EmbeddingFactoryError,
)  # noqa: E402


class FakeEmbedding(BaseEmbedding):
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def embed(self, texts, trace=None, **_: any):  # type: ignore[override]
        # Return a deterministic vector based on the tag length, just for testing.
        dim = len(self.tag) + 1
        return [[float(i) for i in range(dim)] for _ in texts]


def _settings(provider: str = "openai") -> Settings:
    return Settings(
        llm={},
        embedding={"provider": provider, "model": "test-model"},
        vector_store={},
        retrieval={},
        rerank={},
        evaluation={},
        observability={},
        extras={},
    )


def test_embedding_factory_routes_provider_via_registry() -> None:
    registry = {
        "fake_a": lambda _: FakeEmbedding("a"),
        "fake_b": lambda _: FakeEmbedding("bb"),
    }
    settings = _settings("fake_b")
    embedder = EmbeddingFactory.create(settings, registry=registry)
    assert isinstance(embedder, FakeEmbedding)
    # Verify embed returns vectors of length len(tag)+1 == 3
    vectors = embedder.embed(["one", "two"])
    assert all(len(v) == 3 for v in vectors)


def test_embedding_factory_missing_provider_raises() -> None:
    settings = _settings("")
    with pytest.raises(EmbeddingFactoryError, match=r"embedding\.provider"):
        EmbeddingFactory.create(settings)


def test_embedding_factory_unknown_provider_raises() -> None:
    settings = _settings("not_supported")
    with pytest.raises(EmbeddingFactoryError, match="Unsupported embedding.provider"):
        EmbeddingFactory.create(settings)
