"""
Semantic CXD Classifier implementation.

This module implements the semantic CXD classifier that uses embedding similarity
and canonical examples to identify cognitive functions in text.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

from ..core.interfaces import CXDClassifier, EmbeddingModel, CanonicalExampleProvider, VectorStore
from ..core.types import CXDFunction, ExecutionState, CXDTag, CXDSequence
from ..core.config import CXDConfig
from ..core.canonical import CanonicalExample
from ..providers.embedding_models import create_embedding_model
from ..providers.examples import create_default_provider
from ..providers.vector_store import create_vector_store

logger = logging.getLogger(__name__)


class SemanticCXDClassifier(CXDClassifier):
    """
    Semantic classifier for CXD functions using embedding similarity.
    
    Uses canonical examples and vector similarity to classify text into
    CXD functions based on semantic meaning rather than lexical patterns.
    """
    
    def __init__(self,
                 embedding_model: Optional[EmbeddingModel] = None,
                 example_provider: Optional[CanonicalExampleProvider] = None,
                 vector_store: Optional[VectorStore] = None,
                 config: Optional[CXDConfig] = None):
        """
        Initialize semantic classifier.
        
        Args:
            embedding_model: Embedding model for text encoding
            example_provider: Provider for canonical examples
            vector_store: Vector store for similarity search
            config: Configuration object
        """
        self.config = config or CXDConfig()
        
        # Initialize components
        self.embedding_model = embedding_model or self._create_default_embedding_model()
        self.example_provider = example_provider or self._create_default_example_provider()
        self.vector_store = vector_store or self._create_default_vector_store()
        
        # Configuration parameters
        self.semantic_threshold = self.config.algorithms.thresholds.semantic
        self.search_k = self.config.algorithms.search.k
        self.min_confidence = self.config.algorithms.thresholds.confidence_min
        
        # Index state
        self._index_built = False
        self._examples_by_function: Optional[Dict[CXDFunction, List[CanonicalExample]]] = None
        self._example_metadata: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            "total_classifications": 0,
            "index_builds": 0,
            "processing_times": [],
            "search_times": [],
            "function_detections": {func: 0 for func in CXDFunction},
            "semantic_confidences": []
        }
        
        logger.info(f"Initialized SemanticCXDClassifier with {self.embedding_model.model_name}")
    
    def _create_default_embedding_model(self) -> EmbeddingModel:
        """Create default embedding model based on configuration."""
        try:
            # Try to create SentenceTransformer model with config settings
            return create_embedding_model(
                model_type="auto",
                model_name=self.config.models.embedding.name,
                device=self.config.get_effective_device()
            )
        except Exception as e:
            logger.warning(f"Failed to create configured embedding model: {e}")
            # Fallback to mock model
            return create_embedding_model(
                model_type="mock",
                dimension=self.config.models.mock.dimension,
                seed=self.config.models.mock.seed
            )
    
    def _create_default_example_provider(self) -> CanonicalExampleProvider:
        """Create default example provider."""
        try:
            return create_default_provider(self.config.paths.canonical_examples.parent)
        except Exception as e:
            logger.warning(f"Failed to create example provider: {e}")
            # Create in-memory provider with basic examples
            from ..providers.examples import InMemoryExampleProvider
            return InMemoryExampleProvider()
    
    def _create_default_vector_store(self) -> VectorStore:
        """Create default vector store."""
        return create_vector_store(
            dimension=self.embedding_model.dimension,
            metric=self.config.algorithms.search.metric.value,
            prefer_faiss=self.config.features.faiss_indexing
        )
    
    def _ensure_index_built(self) -> None:
        """Ensure that the vector index is built."""
        if not self._index_built:
            self._build_index()
    
    def _build_index(self) -> None:
        """Build vector index from canonical examples."""
        start_time = time.time()
        
        logger.info("Building semantic index from canonical examples...")
        
        try:
            # Load canonical examples
            self._examples_by_function = self.example_provider.load_examples()
            
            # Collect all example texts and metadata
            all_texts = []
            all_metadata = []
            
            for function, examples in self._examples_by_function.items():
                for example in examples:
                    all_texts.append(example.text)
                    metadata = {
                        "function": function,
                        "example_id": example.id,
                        "category": example.category,
                        "quality_score": example.quality_score,
                        "tags": example.tags,
                        "text": example.text
                    }
                    all_metadata.append(metadata)
            
            if not all_texts:
                logger.warning("No canonical examples found")
                self._index_built = True
                return
            
            # Generate embeddings for all examples
            logger.info(f"Generating embeddings for {len(all_texts)} examples...")
            embeddings = self.embedding_model.encode_batch(all_texts)
            
            # Add to vector store
            self.vector_store.add(embeddings, all_metadata)
            self._example_metadata = all_metadata
            
            # Update statistics
            build_time = time.time() - start_time
            self.stats["index_builds"] += 1
            
            self._index_built = True
            
            logger.info(f"Semantic index built in {build_time:.2f}s with {len(all_texts)} examples")
            
        except Exception as e:
            logger.error(f"Failed to build semantic index: {e}")
            # Set as built to avoid infinite retry
            self._index_built = True
            raise
    
    def classify(self, text: str) -> CXDSequence:
        """
        Classify text using semantic similarity to canonical examples.
        
        Args:
            text: Input text to classify
            
        Returns:
            CXDSequence: Classified CXD sequence
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return CXDSequence([])
        
        # Ensure index is built
        self._ensure_index_built()
        
        if self.vector_store.size == 0:
            logger.warning("Empty vector store, cannot classify")
            return CXDSequence([])
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(text)
        
        # Search for similar examples
        search_start = time.time()
        similarities, indices = self.vector_store.search(query_embedding, self.search_k)
        search_time = (time.time() - search_start) * 1000
        
        # Aggregate scores by function
        function_scores = self._aggregate_function_scores(similarities, indices)
        
        # Create CXD tags
        tags = self._create_cxd_tags(function_scores, text)
        
        # Update statistics
        processing_time = (time.time() - start_time) * 1000
        self._update_stats(function_scores, processing_time, search_time)
        
        return CXDSequence(tags)
    
    def _aggregate_function_scores(self, 
                                  similarities: np.ndarray, 
                                  indices: np.ndarray) -> Dict[CXDFunction, float]:
        """
        Aggregate similarity scores by CXD function.
        
        Args:
            similarities: Similarity scores from vector search
            indices: Indices of similar examples
            
        Returns:
            Dict: Aggregated scores by function
        """
        function_scores = {}
        function_similarities = {}
        
        # Group similarities by function
        for sim, idx in zip(similarities, indices):
            if 0 <= idx < len(self._example_metadata):
                metadata = self._example_metadata[idx]
                function = metadata["function"]
                quality_score = metadata.get("quality_score", 1.0)
                
                # Weight similarity by example quality
                weighted_sim = float(sim) * quality_score
                
                if function not in function_similarities:
                    function_similarities[function] = []
                function_similarities[function].append(weighted_sim)
        
        # Aggregate scores for each function
        for function, sims in function_similarities.items():
            if sims:
                # Use weighted average with higher weight for top similarities
                sims_array = np.array(sims)
                weights = np.exp(np.arange(len(sims_array), 0, -1))  # Exponential decay
                weighted_avg = np.average(sims_array, weights=weights[:len(sims_array)])
                
                # Apply threshold
                if weighted_avg >= self.semantic_threshold:
                    function_scores[function] = min(weighted_avg, 0.95)  # Cap at 95%
        
        return function_scores
    
    def _create_cxd_tags(self, 
                        function_scores: Dict[CXDFunction, float], 
                        original_text: str) -> List[CXDTag]:
        """
        Create CXD tags from function scores.
        
        Args:
            function_scores: Aggregated scores by function
            original_text: Original input text
            
        Returns:
            List[CXDTag]: Created tags with semantic confidence
        """
        tags = []
        
        for function, score in function_scores.items():
            # Determine execution state based on score
            if score >= 0.7:
                state = ExecutionState.SUCCESS
            elif score >= 0.4:
                state = ExecutionState.PARTIAL
            elif score >= 0.2:
                state = ExecutionState.UNCERTAIN
            else:
                state = ExecutionState.FAILURE
            
            # Create tag with semantic metadata
            tag = CXDTag(
                function=function,
                state=state,
                confidence=score,
                evidence=[f"Semantic similarity: {score:.3f}"],
                metadata={
                    "semantic_confidence": score,
                    "classification_method": "semantic_similarity",
                    "search_k": self.search_k,
                    "threshold": self.semantic_threshold
                }
            )
            
            # Store semantic confidence for backward compatibility
            tag.semantic_confidence = score
            
            tags.append(tag)
        
        # Sort by confidence (highest first)
        tags.sort(key=lambda t: t.confidence, reverse=True)
        
        return tags
    
    def _update_stats(self, 
                     function_scores: Dict[CXDFunction, float], 
                     processing_time: float,
                     search_time: float) -> None:
        """Update classification statistics."""
        self.stats["total_classifications"] += 1
        self.stats["processing_times"].append(processing_time)
        self.stats["search_times"].append(search_time)
        
        for function, score in function_scores.items():
            self.stats["function_detections"][function] += 1
            self.stats["semantic_confidences"].append(score)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the classifier.
        
        Returns:
            Dict: Performance statistics
        """
        stats = self.stats.copy()
        
        # Processing time statistics
        if self.stats["processing_times"]:
            times = self.stats["processing_times"]
            stats.update({
                "avg_processing_time_ms": sum(times) / len(times),
                "max_processing_time_ms": max(times),
                "min_processing_time_ms": min(times),
                "total_processing_time_ms": sum(times)
            })
        
        # Search time statistics
        if self.stats["search_times"]:
            search_times = self.stats["search_times"]
            stats.update({
                "avg_search_time_ms": sum(search_times) / len(search_times),
                "max_search_time_ms": max(search_times),
                "min_search_time_ms": min(search_times)
            })
        
        # Semantic confidence statistics
        if self.stats["semantic_confidences"]:
            confidences = self.stats["semantic_confidences"]
            stats.update({
                "avg_semantic_confidence": sum(confidences) / len(confidences),
                "max_semantic_confidence": max(confidences),
                "min_semantic_confidence": min(confidences)
            })
        
        # Component statistics
        stats.update({
            "index_built": self._index_built,
            "vector_store_size": self.vector_store.size,
            "embedding_dimension": self.embedding_model.dimension,
            "search_k": self.search_k,
            "semantic_threshold": self.semantic_threshold
        })
        
        # Function detection rates
        total_classifications = self.stats["total_classifications"]
        if total_classifications > 0:
            for function, count in self.stats["function_detections"].items():
                stats[f"{function.value}_detection_rate"] = count / total_classifications
        
        # Add component stats if available
        if hasattr(self.embedding_model, 'get_stats'):
            stats["embedding_model_stats"] = self.embedding_model.get_stats()
        
        if hasattr(self.vector_store, 'get_stats'):
            stats["vector_store_stats"] = self.vector_store.get_stats()
        
        return stats
    
    def get_similar_examples(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar canonical examples for a given text.
        
        Args:
            text: Input text
            k: Number of similar examples to return
            
        Returns:
            List[Dict]: Similar examples with similarity scores
        """
        self._ensure_index_built()
        
        if self.vector_store.size == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(text)
        
        # Search for similar examples
        similarities, indices = self.vector_store.search(query_embedding, k)
        
        # Format results
        results = []
        for sim, idx in zip(similarities, indices):
            if 0 <= idx < len(self._example_metadata):
                metadata = self._example_metadata[idx]
                result = {
                    "similarity": float(sim),
                    "function": metadata["function"].value,
                    "text": metadata["text"],
                    "category": metadata.get("category", ""),
                    "quality_score": metadata.get("quality_score", 1.0),
                    "example_id": metadata.get("example_id", ""),
                    "tags": metadata.get("tags", [])
                }
                results.append(result)
        
        return results
    
    def explain_classification(self, text: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of semantic classification.
        
        Args:
            text: Input text to explain
            
        Returns:
            Dict: Detailed explanation with similar examples
        """
        self._ensure_index_built()
        
        explanation = {
            "input_text": text,
            "embedding_model": self.embedding_model.model_name,
            "vector_store_size": self.vector_store.size,
            "search_k": self.search_k,
            "semantic_threshold": self.semantic_threshold,
            "similar_examples": self.get_similar_examples(text, self.search_k),
            "function_analysis": {}
        }
        
        # Get classification result
        result = self.classify(text)
        
        # Add function analysis
        for tag in result.tags:
            explanation["function_analysis"][tag.function.value] = {
                "confidence": tag.confidence,
                "semantic_confidence": getattr(tag, 'semantic_confidence', tag.confidence),
                "state": tag.state.value,
                "evidence": tag.evidence,
                "metadata": tag.metadata
            }
        
        return explanation
    
    def rebuild_index(self) -> None:
        """Rebuild the vector index from canonical examples."""
        logger.info("Rebuilding semantic index...")
        self._index_built = False
        self.vector_store.clear()
        self._example_metadata.clear()
        self._build_index()
    
    def update_examples(self, new_provider: CanonicalExampleProvider) -> None:
        """
        Update canonical examples and rebuild index.
        
        Args:
            new_provider: New example provider
        """
        self.example_provider = new_provider
        self.rebuild_index()


# Export semantic classifier
__all__ = [
    "SemanticCXDClassifier",
]
