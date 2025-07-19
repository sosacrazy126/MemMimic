"""
Result Processing Module for Memory Search System.

Handles ranking, filtering, and post-processing of search results with 
configurable algorithms and performance optimization.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass

from .interfaces import (
    ResultProcessor, SearchQuery, SearchResult, SearchConfig,
    SearchType, SearchError
)

logger = logging.getLogger(__name__)


class RankingAlgorithm(Enum):
    """Available ranking algorithms"""
    RELEVANCE_ONLY = "relevance_only"
    RELEVANCE_WITH_CXD = "relevance_with_cxd"
    TEMPORAL_BOOST = "temporal_boost"
    METADATA_WEIGHTED = "metadata_weighted"
    HYBRID_RANKING = "hybrid_ranking"


@dataclass
class RankingWeights:
    """Configurable weights for hybrid ranking algorithm"""
    relevance_weight: float = 0.6
    cxd_confidence_weight: float = 0.2
    recency_weight: float = 0.1
    metadata_quality_weight: float = 0.1


class AdvancedResultProcessor(ResultProcessor):
    """
    Advanced result processor with multiple ranking algorithms and filtering strategies.
    
    Features:
    - Multiple ranking algorithms (relevance, CXD-aware, temporal, hybrid)
    - Sophisticated filtering with metadata analysis
    - Performance optimization for large result sets
    - Configurable weights and thresholds
    - Result diversity promotion
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """
        Initialize result processor.
        
        Args:
            config: Search configuration for processor settings
        """
        self.config = config
        
        # Default ranking weights
        self.ranking_weights = RankingWeights()
        
        # Performance metrics
        self._metrics = {
            'total_processed': 0,
            'total_filtered': 0,
            'avg_processing_time_ms': 0.0,
            'ranking_algorithm_usage': {},
        }
        
        # Ranking algorithm implementations
        self._ranking_algorithms = {
            RankingAlgorithm.RELEVANCE_ONLY: self._rank_by_relevance,
            RankingAlgorithm.RELEVANCE_WITH_CXD: self._rank_by_relevance_and_cxd,
            RankingAlgorithm.TEMPORAL_BOOST: self._rank_with_temporal_boost,
            RankingAlgorithm.METADATA_WEIGHTED: self._rank_by_metadata_quality,
            RankingAlgorithm.HYBRID_RANKING: self._rank_hybrid,
        }
        
        logger.info("Advanced Result Processor initialized")
    
    def rank_results(self, query: SearchQuery, 
                    candidates: List[SearchResult]) -> List[SearchResult]:
        """
        Apply ranking algorithm to search result candidates.
        
        Args:
            query: Original search query
            candidates: List of search result candidates
            
        Returns:
            Ranked list of search results
        """
        if not candidates:
            return candidates
        
        try:
            start_time = time.time()
            
            # Determine ranking algorithm based on query and configuration
            algorithm = self._select_ranking_algorithm(query)
            
            # Apply ranking algorithm
            ranking_func = self._ranking_algorithms[algorithm]
            ranked_results = ranking_func(query, candidates)
            
            # Apply diversity promotion if enabled
            if self._should_promote_diversity(query):
                ranked_results = self._promote_result_diversity(ranked_results)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics('rank', processing_time, len(candidates), algorithm)
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Ranking failed, returning original order: {e}")
            return candidates
    
    def filter_results(self, query: SearchQuery,
                      candidates: List[SearchResult]) -> List[SearchResult]:
        """
        Apply filtering to search result candidates.
        
        Args:
            query: Original search query
            candidates: List of search result candidates
            
        Returns:
            Filtered list of search results
        """
        if not candidates:
            return candidates
        
        try:
            start_time = time.time()
            filtered_results = []
            
            for candidate in candidates:
                if self._should_include_result(query, candidate):
                    filtered_results.append(candidate)
            
            # Apply limit
            if query.limit > 0:
                filtered_results = filtered_results[:query.limit]
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics('filter', processing_time, len(candidates))
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Filtering failed, returning limited results: {e}")
            return candidates[:query.limit] if query.limit > 0 else candidates
    
    def _select_ranking_algorithm(self, query: SearchQuery) -> RankingAlgorithm:
        """Select appropriate ranking algorithm based on query characteristics."""
        
        # Use hybrid ranking for complex queries
        if (len(query.text.split()) > 3 or 
            query.filters or 
            query.search_type == SearchType.HYBRID):
            return RankingAlgorithm.HYBRID_RANKING
        
        # Use CXD-aware ranking for semantic searches
        if query.search_type == SearchType.SEMANTIC:
            return RankingAlgorithm.RELEVANCE_WITH_CXD
        
        # Use temporal boost for queries that might benefit from recency
        temporal_keywords = ['recent', 'new', 'latest', 'current', 'today']
        if any(keyword in query.text.lower() for keyword in temporal_keywords):
            return RankingAlgorithm.TEMPORAL_BOOST
        
        # Default to relevance-only for simple keyword searches
        return RankingAlgorithm.RELEVANCE_ONLY
    
    def _rank_by_relevance(self, query: SearchQuery, 
                          candidates: List[SearchResult]) -> List[SearchResult]:
        """Simple relevance-based ranking."""
        return sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
    
    def _rank_by_relevance_and_cxd(self, query: SearchQuery,
                                  candidates: List[SearchResult]) -> List[SearchResult]:
        """Ranking that considers both relevance and CXD classification confidence."""
        
        def combined_score(result: SearchResult) -> float:
            base_score = result.relevance_score
            
            # Add CXD confidence boost
            if result.cxd_classification:
                cxd_boost = result.cxd_classification.confidence * 0.1
                return min(base_score + cxd_boost, 1.0)
            
            return base_score
        
        return sorted(candidates, key=combined_score, reverse=True)
    
    def _rank_with_temporal_boost(self, query: SearchQuery,
                                 candidates: List[SearchResult]) -> List[SearchResult]:
        """Ranking with recency boost for recent memories."""
        
        def temporal_score(result: SearchResult) -> float:
            base_score = result.relevance_score
            
            # Try to extract timestamp from metadata
            metadata = result.metadata or {}
            created_at = metadata.get('created_at') or metadata.get('timestamp')
            
            if created_at:
                try:
                    from datetime import datetime, timedelta
                    if isinstance(created_at, str):
                        # Parse ISO format timestamp
                        timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        timestamp = created_at
                    
                    # Calculate age in days
                    age_days = (datetime.now() - timestamp.replace(tzinfo=None)).days
                    
                    # Apply temporal boost (stronger for newer content)
                    if age_days <= 1:
                        temporal_boost = 0.2  # Recent content gets significant boost
                    elif age_days <= 7:
                        temporal_boost = 0.1  # Week-old content gets moderate boost
                    elif age_days <= 30:
                        temporal_boost = 0.05  # Month-old content gets small boost
                    else:
                        temporal_boost = 0.0  # Older content gets no boost
                    
                    return min(base_score + temporal_boost, 1.0)
                    
                except Exception:
                    # If timestamp parsing fails, just use base score
                    pass
            
            return base_score
        
        return sorted(candidates, key=temporal_score, reverse=True)
    
    def _rank_by_metadata_quality(self, query: SearchQuery,
                                 candidates: List[SearchResult]) -> List[SearchResult]:
        """Ranking based on metadata quality and completeness."""
        
        def metadata_score(result: SearchResult) -> float:
            base_score = result.relevance_score
            metadata = result.metadata or {}
            
            # Quality indicators
            quality_indicators = [
                'confidence' in metadata,
                'type' in metadata,
                'created_at' in metadata or 'timestamp' in metadata,
                'source' in metadata,
                len(result.content) > 50,  # Substantial content
            ]
            
            # Calculate quality boost
            quality_ratio = sum(quality_indicators) / len(quality_indicators)
            quality_boost = quality_ratio * 0.15
            
            return min(base_score + quality_boost, 1.0)
        
        return sorted(candidates, key=metadata_score, reverse=True)
    
    def _rank_hybrid(self, query: SearchQuery,
                    candidates: List[SearchResult]) -> List[SearchResult]:
        """Advanced hybrid ranking considering multiple factors."""
        
        def hybrid_score(result: SearchResult) -> float:
            weights = self.ranking_weights
            
            # Base relevance score
            relevance_component = result.relevance_score * weights.relevance_weight
            
            # CXD confidence component
            cxd_component = 0.0
            if result.cxd_classification:
                cxd_component = result.cxd_classification.confidence * weights.cxd_confidence_weight
            
            # Recency component
            recency_component = self._calculate_recency_score(result) * weights.recency_weight
            
            # Metadata quality component
            metadata_component = self._calculate_metadata_quality(result) * weights.metadata_quality_weight
            
            # Combine all components
            total_score = (relevance_component + cxd_component + 
                          recency_component + metadata_component)
            
            return min(total_score, 1.0)
        
        return sorted(candidates, key=hybrid_score, reverse=True)
    
    def _calculate_recency_score(self, result: SearchResult) -> float:
        """Calculate recency score for a result."""
        try:
            metadata = result.metadata or {}
            created_at = metadata.get('created_at') or metadata.get('timestamp')
            
            if not created_at:
                return 0.5  # Neutral score for unknown age
            
            from datetime import datetime, timedelta
            if isinstance(created_at, str):
                timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                timestamp = created_at
            
            age_days = (datetime.now() - timestamp.replace(tzinfo=None)).days
            
            # Exponential decay: newer content scores higher
            if age_days <= 0:
                return 1.0
            elif age_days <= 7:
                return 0.8
            elif age_days <= 30:
                return 0.6
            elif age_days <= 90:
                return 0.4
            else:
                return 0.2
                
        except Exception:
            return 0.5  # Neutral score if calculation fails
    
    def _calculate_metadata_quality(self, result: SearchResult) -> float:
        """Calculate metadata quality score."""
        metadata = result.metadata or {}
        
        quality_factors = []
        
        # Check for confidence score
        if 'confidence' in metadata:
            confidence = metadata.get('confidence', 0)
            if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                quality_factors.append(confidence)
        
        # Check for content completeness
        content_length = len(result.content)
        if content_length > 100:
            quality_factors.append(1.0)
        elif content_length > 50:
            quality_factors.append(0.8)
        elif content_length > 20:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.3)
        
        # Check for metadata completeness
        required_fields = ['type', 'source', 'created_at']
        completeness_score = sum(1 for field in required_fields if field in metadata) / len(required_fields)
        quality_factors.append(completeness_score)
        
        # Return average quality
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
    
    def _should_include_result(self, query: SearchQuery, result: SearchResult) -> bool:
        """Determine if a result should be included based on filtering criteria."""
        
        # Apply confidence threshold
        if result.relevance_score < query.min_confidence:
            return False
        
        # Apply query filters
        if query.filters:
            result_metadata = result.metadata or {}
            
            for filter_key, filter_value in query.filters.items():
                if filter_key not in result_metadata:
                    return False
                
                result_value = result_metadata[filter_key]
                
                # Handle different filter types
                if isinstance(filter_value, list):
                    # Include if result value is in the filter list
                    if result_value not in filter_value:
                        return False
                elif isinstance(filter_value, dict):
                    # Handle range filters
                    if 'min' in filter_value and result_value < filter_value['min']:
                        return False
                    if 'max' in filter_value and result_value > filter_value['max']:
                        return False
                else:
                    # Exact match
                    if result_value != filter_value:
                        return False
        
        return True
    
    def _should_promote_diversity(self, query: SearchQuery) -> bool:
        """Determine if result diversity should be promoted."""
        # Promote diversity for complex queries or when explicitly requested
        return (len(query.text.split()) > 2 or 
                'diverse' in query.text.lower() or
                'different' in query.text.lower())
    
    def _promote_result_diversity(self, results: List[SearchResult]) -> List[SearchResult]:
        """Promote diversity in results to avoid too much similarity."""
        if len(results) <= 3:
            return results
        
        diverse_results = []
        used_content_hashes = set()
        
        for result in results:
            # Create content hash for similarity detection
            content_hash = hash(result.content[:200].lower())
            
            # Check if this content is too similar to already selected results
            if content_hash not in used_content_hashes:
                diverse_results.append(result)
                used_content_hashes.add(content_hash)
            elif len(diverse_results) < len(results) // 2:
                # Still include some similar results if we don't have enough diversity
                diverse_results.append(result)
        
        # Fill remaining slots with original order if needed
        remaining_slots = len(results) - len(diverse_results)
        if remaining_slots > 0:
            for result in results:
                if result not in diverse_results and remaining_slots > 0:
                    diverse_results.append(result)
                    remaining_slots -= 1
        
        return diverse_results
    
    def _update_metrics(self, operation: str, processing_time_ms: float, 
                       result_count: int, algorithm: RankingAlgorithm = None):
        """Update processing metrics."""
        self._metrics['total_processed'] += result_count
        
        # Update average processing time
        current_avg = self._metrics['avg_processing_time_ms']
        total_ops = self._metrics.get('total_operations', 0) + 1
        self._metrics['avg_processing_time_ms'] = (
            (current_avg * (total_ops - 1) + processing_time_ms) / total_ops
        )
        self._metrics['total_operations'] = total_ops
        
        # Track algorithm usage
        if algorithm:
            algo_key = algorithm.value
            self._metrics['ranking_algorithm_usage'][algo_key] = (
                self._metrics['ranking_algorithm_usage'].get(algo_key, 0) + 1
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get result processor performance metrics."""
        return {
            'total_processed': self._metrics['total_processed'],
            'total_filtered': self._metrics['total_filtered'],
            'avg_processing_time_ms': self._metrics['avg_processing_time_ms'],
            'ranking_algorithm_usage': self._metrics['ranking_algorithm_usage'],
            'total_operations': self._metrics.get('total_operations', 0),
        }
    
    def configure_ranking_weights(self, weights: RankingWeights):
        """Configure ranking weights for hybrid algorithm."""
        self.ranking_weights = weights
        logger.info(f"Ranking weights updated: {weights}")


def create_result_processor(config: Optional[SearchConfig] = None) -> ResultProcessor:
    """
    Factory function to create result processor.
    
    Args:
        config: Search configuration
        
    Returns:
        ResultProcessor instance
    """
    return AdvancedResultProcessor(config)