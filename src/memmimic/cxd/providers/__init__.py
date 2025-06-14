"""
Providers module for CXD Classifier.

Contains implementations for data providers, vector stores, and embedding models
that support the CXD classification system with various backends and configurations.
"""

# Import providers
from .examples import (
    YamlExampleProvider,
    JsonExampleProvider,
    InMemoryExampleProvider,
    CompositeExampleProvider,
    create_yaml_provider,
    create_json_provider,
    create_default_provider,
)

from .vector_store import (
    FAISSVectorStore,
    NumpyVectorStore,
    create_vector_store,
    create_faiss_store,
    create_numpy_store,
    FAISS_AVAILABLE,
)

from .embedding_models import (
    SentenceTransformerModel,
    MockEmbeddingModel,
    CachedEmbeddingModel,
    create_embedding_model,
    create_cached_model,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    TORCH_AVAILABLE,
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
