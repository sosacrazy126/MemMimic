#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Search Processor Module

Handles vector similarity processing and semantic search operations.
Extracted from memmimic_recall_cxd.py for improved maintainability.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import hashlib

from ...utils.caching import cached_embedding_operation, lru_cached

logger = logging.getLogger(__name__)


class SemanticProcessor:
    """
    Semantic search processor using vector similarity and embedding techniques.
    
    Provides high-level semantic search capabilities with various similarity metrics
    and optimization strategies for memory retrieval.
    """
    
    def __init__(self, similarity_threshold: float = 0.1):
        """
        Initialize semantic processor.
        
        Args:
            similarity_threshold: Minimum similarity score for relevant results
        """
        self.similarity_threshold = similarity_threshold
        self.cache = {}  # Simple in-memory cache for embeddings
    
    def search(
        self,
        query: str,
        memory_store,
        limit: int = 10,
        similarity_metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Search query text
            memory_store: Memory storage instance
            limit: Maximum number of results
            similarity_metric: Similarity metric to use
            
        Returns:
            List of semantically similar memories with scores
        """
        try:
            # Get query embedding
            query_embedding = self._get_query_embedding(query, memory_store)
            if query_embedding is None:
                logger.warning("Failed to get query embedding, falling back to keyword search")
                return self._fallback_keyword_search(query, memory_store, limit)
            
            # Search for similar memories
            similar_memories = self._find_similar_memories(
                query_embedding=query_embedding,
                memory_store=memory_store,
                limit=limit * 2,  # Get more for better ranking
                similarity_metric=similarity_metric
            )
            
            # Format results
            formatted_results = []
            for memory, similarity_score in similar_memories[:limit]:
                if similarity_score >= self.similarity_threshold:
                    result = self._format_semantic_result(memory, similarity_score)
                    formatted_results.append(result)
            
            logger.debug(f"Semantic search found {len(formatted_results)} relevant results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self._fallback_keyword_search(query, memory_store, limit)
    
    def _get_query_embedding(self, query: str, memory_store) -> Optional[np.ndarray]:
        """Get embedding vector for query text."""
        try:
            # Check cache first
            query_hash = hashlib.md5(query.encode()).hexdigest()
            if query_hash in self.cache:
                return self.cache[query_hash]
            
            # Try to get embedding from memory store if it has embedding capabilities
            if hasattr(memory_store, 'get_embedding'):
                embedding = memory_store.get_embedding(query)
                if embedding is not None:
                    self.cache[query_hash] = embedding
                    return embedding
            
            # Fallback: create simple embedding (placeholder for real implementation)
            embedding = self._create_simple_embedding(query)
            self.cache[query_hash] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            return None
    
    @cached_embedding_operation(ttl=7200)  # Cache embeddings for 2 hours
    def _create_simple_embedding(self, text: str) -> np.ndarray:
        """
        Create a simple embedding as fallback.
        
        Note: This is a placeholder. In production, this should use
        a proper embedding model like sentence-transformers, OpenAI embeddings, etc.
        """
        # Simple character-based embedding (for demonstration)
        # Replace with actual embedding model
        text_lower = text.lower()
        
        # Create a fixed-size vector based on character frequencies
        embedding_size = 384  # Common embedding dimension
        embedding = np.zeros(embedding_size)
        
        for i, char in enumerate(text_lower[:embedding_size]):
            embedding[i] = ord(char) / 255.0  # Normalize to 0-1
        
        # Add some randomness based on text hash for better distribution
        text_hash = hash(text_lower)
        np.random.seed(abs(text_hash) % 2**32)
        noise = np.random.normal(0, 0.1, embedding_size)
        embedding = embedding + noise
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def _find_similar_memories(
        self,
        query_embedding: np.ndarray,
        memory_store,
        limit: int,
        similarity_metric: str = "cosine"
    ) -> List[Tuple[Any, float]]:
        """Find memories similar to the query embedding."""
        try:
            similar_memories = []
            
            # Get all memories from store
            all_memories = memory_store.get_all() if hasattr(memory_store, 'get_all') else []
            
            for memory in all_memories:
                try:
                    # Get memory embedding
                    memory_embedding = self._get_memory_embedding(memory, memory_store)
                    if memory_embedding is None:
                        continue
                    
                    # Calculate similarity
                    similarity = self._calculate_similarity(
                        query_embedding, memory_embedding, similarity_metric
                    )
                    
                    if similarity >= self.similarity_threshold:
                        similar_memories.append((memory, similarity))
                        
                except Exception as e:
                    logger.debug(f"Error processing memory: {e}")
                    continue
            
            # Sort by similarity (highest first)
            similar_memories.sort(key=lambda x: x[1], reverse=True)
            
            return similar_memories[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar memories: {e}")
            return []
    
    def _get_memory_embedding(self, memory, memory_store) -> Optional[np.ndarray]:
        """Get embedding for a memory object."""
        try:
            # Extract content from memory
            content = getattr(memory, 'content', str(memory))
            if not content:
                return None
            
            # Check if memory store can provide embeddings
            if hasattr(memory_store, 'get_memory_embedding'):
                embedding = memory_store.get_memory_embedding(memory)
                if embedding is not None:
                    return embedding
            
            # Fallback: create simple embedding
            return self._create_simple_embedding(content)
            
        except Exception as e:
            logger.debug(f"Failed to get memory embedding: {e}")
            return None
    
    @lru_cached(maxsize=256)  # Cache similarity calculations
    def _calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """Calculate similarity between two embeddings."""
        try:
            if metric == "cosine":
                # Cosine similarity
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = dot_product / (norm1 * norm2)
                return max(0.0, similarity)  # Clamp to non-negative
                
            elif metric == "euclidean":
                # Inverse Euclidean distance (converted to similarity)
                distance = np.linalg.norm(embedding1 - embedding2)
                similarity = 1.0 / (1.0 + distance)
                return similarity
                
            elif metric == "manhattan":
                # Inverse Manhattan distance
                distance = np.sum(np.abs(embedding1 - embedding2))
                similarity = 1.0 / (1.0 + distance)
                return similarity
                
            else:
                # Default to cosine
                return self._calculate_similarity(embedding1, embedding2, "cosine")
                
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _format_semantic_result(self, memory, similarity_score: float) -> Dict[str, Any]:
        """Format a semantic search result."""
        try:
            # Extract memory attributes
            content = getattr(memory, 'content', str(memory))
            memory_type = getattr(memory, 'memory_type', 'unknown')
            created_at = getattr(memory, 'created_at', None)
            
            # Create formatted result
            result = {
                "content": content,
                "semantic_score": float(similarity_score),
                "search_method": "semantic",
                "memory_type": memory_type,
                "created_at": str(created_at) if created_at else "",
            }
            
            # Add additional attributes if available
            if hasattr(memory, '__dict__'):
                for key, value in memory.__dict__.items():
                    if key not in result and not key.startswith('_'):
                        result[key] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to format semantic result: {e}")
            return {
                "content": str(memory),
                "semantic_score": float(similarity_score),
                "search_method": "semantic",
                "error": str(e)
            }
    
    def _fallback_keyword_search(self, query: str, memory_store, limit: int) -> List[Dict]:
        """Fallback to simple keyword search when semantic search fails."""
        try:
            logger.debug("Using fallback keyword search")
            
            # Simple keyword matching
            results = []
            if hasattr(memory_store, 'search_memories'):
                memories = memory_store.search_memories(query, limit=limit)
                
                for memory in memories:
                    result = {
                        "content": getattr(memory, 'content', str(memory)),
                        "semantic_score": 0.5,  # Default score for keyword matches
                        "search_method": "keyword_fallback",
                        "memory_type": getattr(memory, 'memory_type', 'unknown')
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback keyword search failed: {e}")
            return []
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        logger.debug("Semantic processor cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_memory_mb": sum(
                arr.nbytes for arr in self.cache.values() if isinstance(arr, np.ndarray)
            ) / (1024 * 1024)
        }