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
from libs.llm.base_llm import BaseLLM  # noqa: E402
from libs.llm.llm_factory import LLMFactory, LLMFactoryError  # noqa: E402


class FakeLLM(BaseLLM):
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def chat(self, messages: list[dict]) -> str:
        return f"fake:{self.tag}:{len(messages)}"


def _settings(provider: str = "openai") -> Settings:
    return Settings(
        llm={"provider": provider, "model": "gpt-test"},
        embedding={},
        vector_store={},
        retrieval={},
        rerank={},
        evaluation={},
        observability={},
        extras={},
    )


def test_llm_factory_routes_provider_via_registry() -> None:
    registry = {
        "fake_a": lambda _: FakeLLM("a"),
        "fake_b": lambda _: FakeLLM("b"),
    }
    settings = _settings("fake_b")
    llm = LLMFactory.create(settings, registry=registry)
    assert isinstance(llm, FakeLLM)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "fake:b:1"


def test_llm_factory_missing_provider_raises() -> None:
    settings = _settings("")
    with pytest.raises(LLMFactoryError, match=r"llm\.provider"):
        LLMFactory.create(settings)


def test_llm_factory_unknown_provider_raises() -> None:
    settings = _settings("not_supported")
    with pytest.raises(LLMFactoryError, match="Unsupported llm.provider"):
        LLMFactory.create(settings)
