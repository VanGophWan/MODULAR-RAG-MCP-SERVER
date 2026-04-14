"""Splitter package public API.

Exports the abstract base class and the factory for external use.
"""

from .base_splitter import BaseSplitter  # noqa: F401
from .splitter_factory import SplitterFactory, SplitterFactoryError, PlaceholderSplitter  # noqa: F401
