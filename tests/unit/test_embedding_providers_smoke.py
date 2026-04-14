"""Smoke tests for concrete embedding providers.

These tests replace the real provider classes in ``EmbeddingFactory._default_builders``
with lightweight fakes that return deterministic vectors.  The goal is to verify
that the factory can instantiate a provider, that ``embed`` validates input, and
that the returned vectors have the correct dimension.
"""

from pathlib import Path
import sys
import types

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
from libs.embedding.embedding_factory import EmbeddingFactory, EmbeddingFactoryError  # noqa: E402


class FakeProvider(BaseEmbedding):
    """Simple fake that returns a vector of a fixed size filled with the provider tag.
    """

    def __init__(self, tag: str, dim: int = 4):
        self.tag = tag
        self.dim = dim

    def embed(self, texts, trace=None, **_: any):  # type: ignore[override]
        self.validate_texts(texts)
        # Each vector = [len(tag)] * dim for determinism
        value = float(len(self.tag))
        return [[value for _ in range(self.dim)] for _ in texts]


def _settings(provider: str) -> Settings:
    return Settings(
        llm={},
        embedding={"provider": provider, "model": "test-model", "dimension": 5},
        vector_store={},
        retrieval={},
        rerank={},
        evaluation={},
        observability={},
        extras={},
    )


def test_openai_embedding_provider_returns_vectors_of_configured_dim():
    # Monkey‑patch the factory registry for this test only.
    original = EmbeddingFactory._default_builders.copy()
    try:
        EmbeddingFactory._default_builders["openai"] = lambda _: FakeProvider("openai", dim=5)
        embedder = EmbeddingFactory.create(_settings("openai"))
        vectors = embedder.embed(["hello", "world"])
        assert len(vectors) == 2
        assert all(len(v) == 5 for v in vectors)
        # All values should be the float representation of the tag length (6.0)
        assert all(v[0] == 6.0 for v in vectors)
    finally:
        EmbeddingFactory._default_builders = original


def test_azure_embedding_provider_returns_vectors_of_configured_dim():
    original = EmbeddingFactory._default_builders.copy()
    try:
        EmbeddingFactory._default_builders["azure"] = lambda _: FakeProvider("azure", dim=5)
        embedder = EmbeddingFactory.create(_settings("azure"))
        vectors = embedder.embed(["test"])
        assert len(vectors) == 1
        assert len(vectors[0]) == 5
        # Tag "azure" length is 5 → vector values should be 5.0
        assert vectors[0][0] == 5.0
    finally:
        EmbeddingFactory._default_builders = original


def test_ollama_embedding_provider_returns_vectors_of_configured_dim():
    original = EmbeddingFactory._default_builders.copy()
    try:
        EmbeddingFactory._default_builders["ollama"] = lambda _: FakeProvider("ollama", dim=5)
        embedder = EmbeddingFactory.create(_settings("ollama"))
        vectors = embedder.embed(["sample", "data"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 5
        # Tag "ollama" length is 6 → vector values should be 6.0
        assert vectors[0][0] == 6.0
    finally:
        EmbeddingFactory._default_builders = original


def test_embedding_factory_invalid_texts_raise_error():
    # Use a fake provider that would otherwise work.
    original = EmbeddingFactory._default_builders.copy()
    try:
        EmbeddingFactory._default_builders["openai"] = lambda _: FakeProvider("openai", dim=5)
        embedder = EmbeddingFactory.create(_settings("openai"))
        with pytest.raises(ValueError, match="texts must be a non‑empty list"):
            embedder.embed([])
        with pytest.raises(ValueError, match="each text must be a non‑empty string"):
            embedder.embed(["valid", ""])
    finally:
        EmbeddingFactory._default_builders = original
