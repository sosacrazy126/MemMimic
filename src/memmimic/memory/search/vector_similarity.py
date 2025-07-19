"""
High-performance vector similarity calculation module for MemMimic Memory Search System.

This module provides optimized implementations for various vector similarity metrics
with caching, batch processing, and comprehensive error handling for high-throughput
memory search operations.
"""

import hashlib
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .interfaces import SimilarityCalculator, SimilarityMetric, SimilarityCalculationError
from .search_config import SearchConfig


class OptimizedVectorSimilarity(SimilarityCalculator):
    """
    High-performance vector similarity calculator with multiple metrics and caching.
    
    Supports cosine similarity, euclidean distance, and dot product calculations
    with optimized batch processing using numpy vectorization and LRU caching
    for frequently computed similarities.
    """
    
    def __init__(self, config: SearchConfig, cache_size: int = 1024):
        """
        Initialize the similarity calculator with configuration and cache settings.
        
        Args:
            config: Search configuration containing similarity metric preferences
            cache_size: Maximum number of similarity calculations to cache
        """
        self.config = config
        self.cache_size = cache_size
        self.similarity_metric = config.get_similarity_metric()
        
        # Performance metrics tracking
        self._metrics = {
            'total_calculations': 0,
            'batch_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_calculation_time': 0.0,
            'avg_calculation_time': 0.0,
        }
        
        # Initialize LRU cache for similarity calculations
        from collections import OrderedDict
        self._cache = OrderedDict()
        
        # Precompiled similarity functions for performance
        self._similarity_functions = {
            SimilarityMetric.COSINE: self._cosine_similarity,
            SimilarityMetric.EUCLIDEAN: self._euclidean_distance,
            SimilarityMetric.DOT_PRODUCT: self._dot_product_similarity,
        }
        
        # Batch similarity functions
        self._batch_similarity_functions = {
            SimilarityMetric.COSINE: self._batch_cosine_similarity,
            SimilarityMetric.EUCLIDEAN: self._batch_euclidean_distance,
            SimilarityMetric.DOT_PRODUCT: self._batch_dot_product_similarity,
        }
    
    def calculate_similarity(self, query_embedding: List[float], 
                           memory_embedding: List[float]) -> float:
        """
        Calculate similarity score between two embeddings using configured metric.
        
        Args:
            query_embedding: Query vector as list of floats
            memory_embedding: Memory vector as list of floats
            
        Returns:
            Similarity score between 0.0 and 1.0 (higher = more similar)
            
        Raises:
            SimilarityCalculationError: If vectors are invalid or calculation fails
        """
        start_time = time.perf_counter()
        
        try:
            # Input validation
            self._validate_embeddings(query_embedding, memory_embedding)
            
            # Convert to numpy arrays for efficient computation
            query_vec = np.array(query_embedding, dtype=np.float32)
            memory_vec = np.array(memory_embedding, dtype=np.float32)
            
            # Check cache first
            cache_key = self._generate_cache_key(query_vec, memory_vec)
            cached_result = self._get_cached_similarity(cache_key)
            if cached_result is not None:
                self._metrics['cache_hits'] += 1
                return cached_result
            
            # Calculate similarity using configured metric
            similarity_func = self._similarity_functions[self.similarity_metric]
            similarity_score = similarity_func(query_vec, memory_vec)
            
            # Cache the result
            self._cache_similarity(cache_key, similarity_score)
            self._metrics['cache_misses'] += 1
            
            # Update metrics
            calculation_time = time.perf_counter() - start_time
            self._update_metrics(calculation_time)
            
            return float(similarity_score)
            
        except Exception as e:
            raise SimilarityCalculationError(
                f"Failed to calculate similarity: {str(e)}",
                error_code="SIMILARITY_CALCULATION_FAILED",
                context={
                    'query_dim': len(query_embedding),
                    'memory_dim': len(memory_embedding),
                    'similarity_metric': self.similarity_metric.value,
                }
            )
    
    def batch_calculate_similarity(self, query_embedding: List[float],
                                 memory_embeddings: List[List[float]]) -> List[float]:
        """
        Calculate similarity scores for multiple embeddings efficiently using vectorization.
        
        Args:
            query_embedding: Query vector as list of floats
            memory_embeddings: List of memory vectors as lists of floats
            
        Returns:
            List of similarity scores corresponding to each memory embedding
            
        Raises:
            SimilarityCalculationError: If vectors are invalid or calculation fails
        """
        if not memory_embeddings:
            return []
        
        start_time = time.perf_counter()
        
        try:
            # Input validation
            query_vec = np.array(query_embedding, dtype=np.float32)
            self._validate_single_embedding(query_vec)
            
            # Convert memory embeddings to numpy array for vectorized operations
            memory_matrix = np.array(memory_embeddings, dtype=np.float32)
            
            # Validate dimensions
            if memory_matrix.shape[1] != len(query_embedding):
                raise ValueError(
                    f"Dimension mismatch: query has {len(query_embedding)} dimensions, "
                    f"but memory embeddings have {memory_matrix.shape[1]} dimensions"
                )
            
            # Use vectorized batch calculation
            batch_func = self._batch_similarity_functions[self.similarity_metric]
            similarity_scores = batch_func(query_vec, memory_matrix)
            
            # Update metrics
            calculation_time = time.perf_counter() - start_time
            self._metrics['batch_calculations'] += 1
            self._metrics['total_calculations'] += len(memory_embeddings)
            self._metrics['total_calculation_time'] += calculation_time
            self._metrics['avg_calculation_time'] = (
                self._metrics['total_calculation_time'] / self._metrics['total_calculations']
            )
            
            return similarity_scores.tolist()
            
        except Exception as e:
            raise SimilarityCalculationError(
                f"Failed to calculate batch similarities: {str(e)}",
                error_code="BATCH_SIMILARITY_CALCULATION_FAILED",
                context={
                    'query_dim': len(query_embedding),
                    'batch_size': len(memory_embeddings),
                    'similarity_metric': self.similarity_metric.value,
                }
            )
    
    def _cosine_similarity(self, vec1: NDArray, vec2: NDArray) -> float:
        """Calculate cosine similarity between two vectors."""
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
    
    def _euclidean_distance(self, vec1: NDArray, vec2: NDArray) -> float:
        """Calculate normalized euclidean distance (converted to similarity)."""
        distance = np.linalg.norm(vec1 - vec2)
        max_possible_distance = np.sqrt(2 * len(vec1))  # Assuming normalized vectors
        
        # Convert distance to similarity (0 = far apart, 1 = identical)
        similarity = 1.0 - min(distance / max_possible_distance, 1.0)
        return similarity
    
    def _dot_product_similarity(self, vec1: NDArray, vec2: NDArray) -> float:
        """Calculate normalized dot product similarity."""
        dot_product = np.dot(vec1, vec2)
        
        # Normalize by vector magnitudes to get similarity in [0, 1]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        normalized_dot = dot_product / (norm1 * norm2)
        return max(0.0, normalized_dot)
    
    def _batch_cosine_similarity(self, query_vec: NDArray, memory_matrix: NDArray) -> NDArray:
        """Vectorized cosine similarity calculation."""
        # Compute norms
        query_norm = np.linalg.norm(query_vec)
        memory_norms = np.linalg.norm(memory_matrix, axis=1)
        
        # Handle zero vectors
        valid_mask = (query_norm > 0) & (memory_norms > 0)
        
        # Compute dot products
        dot_products = np.dot(memory_matrix, query_vec)
        
        # Calculate similarities
        similarities = np.zeros(len(memory_matrix))
        similarities[valid_mask] = dot_products[valid_mask] / (query_norm * memory_norms[valid_mask])
        
        # Normalize to [0, 1] range
        similarities = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)
        
        return similarities
    
    def _batch_euclidean_distance(self, query_vec: NDArray, memory_matrix: NDArray) -> NDArray:
        """Vectorized euclidean distance calculation."""
        # Calculate squared distances
        distances_squared = np.sum((memory_matrix - query_vec) ** 2, axis=1)
        distances = np.sqrt(distances_squared)
        
        # Normalize and convert to similarity
        max_possible_distance = np.sqrt(2 * len(query_vec))
        similarities = 1.0 - np.clip(distances / max_possible_distance, 0.0, 1.0)
        
        return similarities
    
    def _batch_dot_product_similarity(self, query_vec: NDArray, memory_matrix: NDArray) -> NDArray:
        """Vectorized dot product similarity calculation."""
        # Compute dot products
        dot_products = np.dot(memory_matrix, query_vec)
        
        # Normalize by vector magnitudes
        query_norm = np.linalg.norm(query_vec)
        memory_norms = np.linalg.norm(memory_matrix, axis=1)
        
        # Handle zero vectors
        valid_mask = (query_norm > 0) & (memory_norms > 0)
        similarities = np.zeros(len(memory_matrix))
        similarities[valid_mask] = dot_products[valid_mask] / (query_norm * memory_norms[valid_mask])
        
        return np.clip(similarities, 0.0, 1.0)
    
    def _validate_embeddings(self, embedding1: List[float], embedding2: List[float]) -> None:
        """Validate that embeddings are valid for similarity calculation."""
        if not embedding1 or not embedding2:
            raise ValueError("Embeddings cannot be empty")
        
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimensions must match: {len(embedding1)} != {len(embedding2)}"
            )
        
        # Check for invalid values
        for i, val in enumerate(embedding1):
            if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                raise ValueError(f"Invalid value in query embedding at index {i}: {val}")
        
        for i, val in enumerate(embedding2):
            if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                raise ValueError(f"Invalid value in memory embedding at index {i}: {val}")
    
    def _validate_single_embedding(self, embedding: NDArray) -> None:
        """Validate a single embedding vector."""
        if len(embedding) == 0:
            raise ValueError("Embedding cannot be empty")
        
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            raise ValueError("Embedding contains invalid values (NaN or Inf)")
    
    def _generate_cache_key(self, vec1: NDArray, vec2: NDArray) -> str:
        """Generate a hash-based cache key for two vectors."""
        # Create a deterministic hash from the vectors
        vec1_bytes = vec1.tobytes()
        vec2_bytes = vec2.tobytes()
        
        # Sort to ensure consistent key regardless of order
        if vec1_bytes < vec2_bytes:
            combined = vec1_bytes + vec2_bytes
        else:
            combined = vec2_bytes + vec1_bytes
        
        return hashlib.md5(combined).hexdigest()
    
    def _get_cached_similarity(self, cache_key: str) -> Optional[float]:
        """Retrieve cached similarity score."""
        return self._cache.get(cache_key)
    
    def _cache_similarity(self, cache_key: str, similarity: float) -> None:
        """Cache a similarity calculation result."""
        # Implement LRU eviction if cache is full
        if len(self._cache) >= self.cache_size:
            # Remove oldest item (first in OrderedDict)
            self._cache.popitem(last=False)
        
        self._cache[cache_key] = similarity
    
    def _update_metrics(self, calculation_time: float) -> None:
        """Update performance metrics."""
        self._metrics['total_calculations'] += 1
        self._metrics['total_calculation_time'] += calculation_time
        
        if self._metrics['total_calculations'] > 0:
            self._metrics['avg_calculation_time'] = (
                self._metrics['total_calculation_time'] / self._metrics['total_calculations']
            )
    
    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Get performance metrics for similarity calculations.
        
        Returns:
            Dictionary containing performance metrics including cache hit rates,
            calculation times, and operation counts
        """
        total_cache_operations = self._metrics['cache_hits'] + self._metrics['cache_misses']
        cache_hit_rate = (
            self._metrics['cache_hits'] / total_cache_operations
            if total_cache_operations > 0 else 0.0
        )
        
        return {
            **self._metrics,
            'cache_hit_rate': cache_hit_rate,
            'similarity_metric': self.similarity_metric.value,
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics to zero."""
        for key in self._metrics:
            self._metrics[key] = 0 if isinstance(self._metrics[key], int) else 0.0
    
    def change_similarity_metric(self, new_metric: SimilarityMetric) -> None:
        """
        Change the similarity metric used for calculations.
        
        Args:
            new_metric: New similarity metric to use
        """
        if new_metric not in self._similarity_functions:
            raise ValueError(f"Unsupported similarity metric: {new_metric}")
        
        self.similarity_metric = new_metric
        # Clear cache when metric changes as cached results are no longer valid
        if hasattr(self, '_similarity_cache'):
            self._similarity_cache.clear()


def create_similarity_calculator(config: SearchConfig, 
                               cache_size: int = 1024) -> OptimizedVectorSimilarity:
    """
    Factory function to create an optimized vector similarity calculator.
    
    Args:
        config: Search configuration containing similarity preferences
        cache_size: Maximum number of calculations to cache
        
    Returns:
        Configured OptimizedVectorSimilarity instance
    """
    return OptimizedVectorSimilarity(config=config, cache_size=cache_size)