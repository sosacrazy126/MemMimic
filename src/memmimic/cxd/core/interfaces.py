"""
Abstract interfaces for CXD Classifier.

This module defines the core abstract base classes and interfaces that
establish the contract for all CXD classifier components. These interfaces
enable plugin architecture and ensure consistent behavior across different
implementations.

Key interfaces:
- EmbeddingModel: Text-to-vector embedding generation
- CXDClassifier: Text-to-CXD classification
- VectorStore: Vector storage and similarity search
- CanonicalExampleProvider: Canonical examples management
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np

from .types import CXDSequence, CXDFunction, MetaClassificationResult


# =============================================================================
# EMBEDDING MODEL INTERFACE
# =============================================================================

class EmbeddingModel(ABC):
    """
    Abstract interface for text embedding models.
    
    This interface defines the contract for converting text into numerical
    vector representations. Implementations can use various backends like
    SentenceTransformers, OpenAI embeddings, or custom models.
    """
    
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            np.ndarray: Embedding vector (normalized, float32)
        """
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embedding vectors for multiple texts efficiently.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            np.ndarray: Matrix of embedding vectors (n_texts x embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension."""
        pass
    
    @property
    def model_name(self) -> str:
        """Get the model name/identifier."""
        return self.__class__.__name__
    
    @property
    def max_sequence_length(self) -> Optional[int]:
        """Get maximum sequence length supported by model."""
        return None
    
    def encode_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Generate embedding with additional metadata.
        
        Args:
            text: Input text to embed
            
        Returns:
            Dict: Containing 'embedding' and metadata
        """
        embedding = self.encode(text)
        return {
            "embedding": embedding,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "text_length": len(text),
            "normalized": True  # Assume embeddings are normalized
        }
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Cosine similarity (-1.0 to 1.0)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# =============================================================================
# CXD CLASSIFIER INTERFACE
# =============================================================================

class CXDClassifier(ABC):
    """
    Abstract interface for CXD text classifiers.
    
    This interface defines the contract for classifying text into CXD
    (Control/Context/Data) sequences. Implementations can use lexical
    patterns, semantic similarity, or hybrid approaches.
    """
    
    @abstractmethod
    def classify(self, text: str) -> CXDSequence:
        """
        Classify text into CXD sequence.
        
        Args:
            text: Input text to classify
            
        Returns:
            CXDSequence: Classified CXD sequence
        """
        pass
    
    def classify_batch(self, texts: List[str]) -> List[CXDSequence]:
        """
        Classify multiple texts efficiently.
        
        Default implementation uses single classification.
        Subclasses should override for batch optimization.
        
        Args:
            texts: List of input texts
            
        Returns:
            List[CXDSequence]: List of classified sequences
        """
        return [self.classify(text) for text in texts]
    
    def classify_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Classify text with additional metadata.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dict: Classification result with metadata
        """
        import time
        start_time = time.time()
        
        sequence = self.classify(text)
        processing_time = (time.time() - start_time) * 1000  # ms
        
        return {
            "sequence": sequence,
            "text": text,
            "processing_time_ms": processing_time,
            "classifier_type": self.__class__.__name__,
            "pattern": sequence.pattern,
            "confidence": sequence.average_confidence
        }
    
    @abstractmethod
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics and metrics.
        
        Returns:
            Dict: Performance statistics
        """
        pass
    
    @property
    def classifier_name(self) -> str:
        """Get classifier name/identifier."""
        return self.__class__.__name__
    
    @property
    def supports_batch(self) -> bool:
        """Check if classifier supports optimized batch processing."""
        # Check if classify_batch is overridden
        return (self.__class__.classify_batch != 
                CXDClassifier.classify_batch)


# =============================================================================
# VECTOR STORE INTERFACE
# =============================================================================

class VectorStore(ABC):
    """
    Abstract interface for vector storage and similarity search.
    
    This interface defines the contract for storing and querying vector
    embeddings. Implementations can use FAISS, numpy, or other backends.
    """
    
    @abstractmethod
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add vectors with associated metadata to the store.
        
        Args:
            vectors: Matrix of vectors to add (n_vectors x dimension)
            metadata: List of metadata dicts for each vector
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k most similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            Tuple[similarities, indices]: Similarity scores and vector indices
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> bool:
        """
        Save vector store to disk.
        
        Args:
            path: File path to save to
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> bool:
        """
        Load vector store from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            bool: True if successful
        """
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """Get number of vectors in store."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get vector dimension."""
        pass
    
    def search_with_metadata(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search and return results with metadata.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List[Dict]: Results with similarity scores and metadata
        """
        similarities, indices = self.search(query_vector, k)
        
        results = []
        for sim, idx in zip(similarities, indices):
            result = {
                "similarity": float(sim),
                "index": int(idx),
                "metadata": self.get_metadata(int(idx))
            }
            results.append(result)
        
        return results
    
    @abstractmethod
    def get_metadata(self, index: int) -> Dict[str, Any]:
        """
        Get metadata for vector at index.
        
        Args:
            index: Vector index
            
        Returns:
            Dict: Metadata for vector
        """
        pass
    
    def clear(self) -> None:
        """Clear all vectors from store."""
        pass
    
    @property
    def is_empty(self) -> bool:
        """Check if store is empty."""
        return self.size == 0


# =============================================================================
# CANONICAL EXAMPLE PROVIDER INTERFACE
# =============================================================================

class CanonicalExampleProvider(ABC):
    """
    Abstract interface for managing canonical examples.
    
    This interface defines the contract for loading, managing, and accessing
    canonical examples used for CXD classification. Implementations can use
    YAML files, JSON files, databases, or other storage backends.
    """
    
    @abstractmethod
    def load_examples(self) -> Dict[CXDFunction, List['CanonicalExample']]:
        """
        Load canonical examples organized by CXD function.
        
        Returns:
            Dict: Mapping from CXDFunction to list of examples
        """
        pass
    
    @abstractmethod
    def get_checksum(self) -> str:
        """
        Get checksum/hash of current examples for cache invalidation.
        
        Returns:
            str: Checksum string
        """
        pass
    
    def get_all_examples(self) -> List['CanonicalExample']:
        """
        Get all examples as a flat list.
        
        Returns:
            List[CanonicalExample]: All examples
        """
        examples_by_function = self.load_examples()
        all_examples = []
        for examples in examples_by_function.values():
            all_examples.extend(examples)
        return all_examples
    
    def get_examples_for_function(self, function: CXDFunction) -> List['CanonicalExample']:
        """
        Get examples for a specific CXD function.
        
        Args:
            function: CXD function to get examples for
            
        Returns:
            List[CanonicalExample]: Examples for function
        """
        examples_by_function = self.load_examples()
        return examples_by_function.get(function, [])
    
    def get_example_texts(self) -> List[str]:
        """
        Get all example texts as strings.
        
        Returns:
            List[str]: Example text strings
        """
        examples = self.get_all_examples()
        return [example.text for example in examples]
    
    def get_examples_by_category(self, category: str) -> List['CanonicalExample']:
        """
        Get examples by category (e.g., 'search', 'filter').
        
        Args:
            category: Category name
            
        Returns:
            List[CanonicalExample]: Examples in category
        """
        examples = self.get_all_examples()
        return [ex for ex in examples if ex.category == category]
    
    def get_high_quality_examples(self, min_quality: float = 0.8) -> List['CanonicalExample']:
        """
        Get examples above quality threshold.
        
        Args:
            min_quality: Minimum quality score
            
        Returns:
            List[CanonicalExample]: High quality examples
        """
        examples = self.get_all_examples()
        return [ex for ex in examples if ex.quality_score >= min_quality]
    
    @property
    def example_count(self) -> int:
        """Get total number of examples."""
        return len(self.get_all_examples())
    
    @property
    def function_counts(self) -> Dict[CXDFunction, int]:
        """Get count of examples per function."""
        examples_by_function = self.load_examples()
        return {func: len(examples) for func, examples in examples_by_function.items()}


# =============================================================================
# CONFIGURATION PROVIDER INTERFACE
# =============================================================================

class ConfigProvider(ABC):
    """
    Abstract interface for configuration management.
    
    This interface defines the contract for loading and managing
    configuration settings from various sources (files, environment, etc.).
    """
    
    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from source.
        
        Returns:
            Dict: Configuration dictionary
        """
        pass
    
    @abstractmethod
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration to source.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if successful
        """
        pass
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get specific configuration value.
        
        Args:
            key: Configuration key (can be nested with dots)
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        config = self.load_config()
        
        # Handle nested keys like "models.embedding.name"
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


# =============================================================================
# METRICS COLLECTOR INTERFACE
# =============================================================================

class MetricsCollector(ABC):
    """
    Abstract interface for collecting and reporting metrics.
    
    This interface defines the contract for gathering performance metrics,
    classification statistics, and other operational data.
    """
    
    @abstractmethod
    def record_classification(self, 
                            text: str, 
                            result: CXDSequence, 
                            processing_time: float) -> None:
        """
        Record a classification event.
        
        Args:
            text: Input text
            result: Classification result
            processing_time: Processing time in milliseconds
        """
        pass
    
    @abstractmethod
    def record_error(self, error_type: str, error_message: str, context: Dict[str, Any]) -> None:
        """
        Record an error event.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context information
        """
        pass
    
    @abstractmethod
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected metrics.
        
        Returns:
            Dict: Metrics summary
        """
        pass
    
    def record_performance_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
        """
        pass


# =============================================================================
# CACHE INTERFACE
# =============================================================================

class CacheProvider(ABC):
    """
    Abstract interface for caching operations.
    
    This interface defines the contract for caching embeddings, classifications,
    and other expensive computations.
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = no expiration)
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if key existed and was deleted
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            bool: True if successful
        """
        pass
    
    def get_or_set(self, key: str, factory_func, ttl: Optional[int] = None) -> Any:
        """
        Get value from cache or compute and cache it.
        
        Args:
            key: Cache key
            factory_func: Function to compute value if not in cache
            ttl: Time to live in seconds
            
        Returns:
            Any: Cached or computed value
        """
        value = self.get(key)
        if value is None:
            value = factory_func()
            self.set(key, value, ttl)
        return value
    
    @property
    def size(self) -> int:
        """Get number of entries in cache."""
        return 0  # Default implementation


# =============================================================================
# LOGGER INTERFACE
# =============================================================================

class StructuredLogger(ABC):
    """
    Abstract interface for structured logging.
    
    This interface defines the contract for logging classification events,
    performance metrics, and debug information in a structured format.
    """
    
    @abstractmethod
    def log_classification(self, 
                          text: str, 
                          result: Union[CXDSequence, MetaClassificationResult],
                          processing_time: float,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a classification event.
        
        Args:
            text: Input text
            result: Classification result
            processing_time: Processing time in milliseconds
            metadata: Additional metadata
        """
        pass
    
    @abstractmethod
    def log_performance(self, 
                       metric_name: str, 
                       value: float, 
                       context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            context: Additional context
        """
        pass
    
    @abstractmethod
    def log_error(self, 
                  error: Exception, 
                  context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error with context.
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        pass
    
    def log_debug(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log debug information.
        
        Args:
            message: Debug message
            data: Additional debug data
        """
        pass


# Export all interfaces
__all__ = [
    # Core interfaces
    "EmbeddingModel",
    "CXDClassifier", 
    "VectorStore",
    "CanonicalExampleProvider",
    
    # Support interfaces
    "ConfigProvider",
    "MetricsCollector",
    "CacheProvider",
    "StructuredLogger",
]
