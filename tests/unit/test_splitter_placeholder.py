"""Smoke test for the ``PlaceholderSplitter`` implementation.

The placeholder splitter should return the input text unchanged (wrapped in a list).
This test ensures the deterministic behaviour expected by downstream code
before the real splitter implementations are added in later phases.
"""

from pathlib import Path
import sys


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_add_src_to_path()

from libs.splitter import PlaceholderSplitter  # noqa: E402


def test_placeholder_splitter_returns_single_chunk() -> None:
    splitter = PlaceholderSplitter()
    text = "This is a test string."
    chunks = splitter.split_text(text)
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0] == text
