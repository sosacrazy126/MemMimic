"""
Providers module for CXD Classifier.

Contains implementations for data providers, vector stores, and embedding models
that support the CXD classification system with various backends and configurations.
"""

from .embedding_models import (
    SENTENCE_TRANSFORMERS_AVAILABLE,
    TORCH_AVAILABLE,
    CachedEmbeddingModel,
    MockEmbeddingModel,
    SentenceTransformerModel,
    create_cached_model,
    create_embedding_model,
)

# Import providers
from .examples import (
    CompositeExampleProvider,
    InMemoryExampleProvider,
    JsonExampleProvider,
    YamlExampleProvider,
    create_default_provider,
    create_json_provider,
    create_yaml_provider,
)
from .vector_store import (
    FAISS_AVAILABLE,
    FAISSVectorStore,
    NumpyVectorStore,
    create_faiss_store,
    create_numpy_store,
    create_vector_store,
)

__all__ = [
    # Example providers
    "YamlExampleProvider",
    "JsonExampleProvider",
    "InMemoryExampleProvider",
    "CompositeExampleProvider",
    "create_yaml_provider",
    "create_json_provider",
    "create_default_provider",
    # Vector stores
    "FAISSVectorStore",
    "NumpyVectorStore",
    "create_vector_store",
    "create_faiss_store",
    "create_numpy_store",
    "FAISS_AVAILABLE",
    # Embedding models
    "SentenceTransformerModel",
    "MockEmbeddingModel",
    "CachedEmbeddingModel",
    "create_embedding_model",
    "create_cached_model",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "TORCH_AVAILABLE",
]

