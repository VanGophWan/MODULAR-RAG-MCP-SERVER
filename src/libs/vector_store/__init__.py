"""Vector store package public API.

Exports the abstract base class and the factory for external use.
"""

from .base_vector_store import BaseVectorStore  # noqa: F401
from .vector_store_factory import VectorStoreFactory, VectorStoreFactoryError, PlaceholderVectorStore  # noqa: F401
