"""Tests for Vision LLM factory and providers."""

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

from core.settings import Settings
from libs.llm.base_vision_llm import BaseVisionLLM, ImageContent
from libs.llm.vision_llm_factory import VisionLLMFactory, VisionLLMFactoryError
from libs.llm.azure_vision_llm import AzureVisionLLM


class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def caption(self, images: list, text: str | None = None) -> str:
        return f"fake:{self.tag}:{len(images)}"


def _settings(provider: str = "azure") -> Settings:
    return Settings(
        llm={"provider": "openai", "model": "gpt-4o"},
        embedding={"provider": "openai", "model": "text-embedding-3-small"},
        vector_store={"backend": "chroma"},
        retrieval={"sparse_backend": "bm25", "fusion_algorithm": "rrf"},
        rerank={"backend": "cross_encoder"},
        evaluation={"backends": ["ragas"]},
        observability={"enabled": True},
        vision_llm={"provider": provider, "model": "gpt-4o"},
    )


def test_vision_llm_factory_routes_provider_via_registry() -> None:
    registry = {
        "fake_a": lambda _: FakeVisionLLM("a"),
        "fake_b": lambda _: FakeVisionLLM("b"),
    }
    settings = _settings("fake_b")
    vision_llm = VisionLLMFactory.create(settings, registry=registry)
    assert isinstance(vision_llm, FakeVisionLLM)
    images = [ImageContent(image_path="/test/image.png")]
    result = vision_llm.caption(images, text="describe")
    assert result == "fake:b:1"


def test_vision_llm_factory_missing_provider_raises() -> None:
    settings = _settings("")
    with pytest.raises(VisionLLMFactoryError, match=r"vision_llm\.provider"):
        VisionLLMFactory.create(settings)


def test_vision_llm_factory_unknown_provider_raises() -> None:
    settings = _settings("not_supported")
    with pytest.raises(VisionLLMFactoryError, match="Unsupported vision_llm.provider"):
        VisionLLMFactory.create(settings)


def test_azure_vision_llm_caption() -> None:
    settings = _settings("azure")
    vision_llm = AzureVisionLLM(settings)
    images = [
        ImageContent(image_path="/test/image1.png"),
        ImageContent(image_path="/test/image2.png"),
    ]
    result = vision_llm.caption(images, text="What is shown?")
    assert "[azure_vision:gpt-4o]" in result
    assert "2 image(s)" in result
    assert "What is shown?" in result


def test_azure_vision_llm_caption_no_text() -> None:
    settings = _settings("azure")
    vision_llm = AzureVisionLLM(settings)
    images = [ImageContent(image_path="/test/image.png")]
    result = vision_llm.caption(images)
    assert "[azure_vision:gpt-4o]" in result
    assert "1 image(s)" in result


def test_base_vision_llm_is_abstract() -> None:
    """Verify BaseVisionLLM cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseVisionLLM()


def test_vision_llm_exposes_correct_interface() -> None:
    """Verify AzureVisionLLM implements the expected interface."""
    settings = _settings()
    vision_llm = VisionLLMFactory.create(settings)
    assert hasattr(vision_llm, "caption")
    images = [ImageContent(image_path="/test.png")]
    result = vision_llm.caption(images)
    assert isinstance(result, str)
    assert len(result) > 0