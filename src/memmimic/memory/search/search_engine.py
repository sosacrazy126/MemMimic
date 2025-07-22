"""
Core search engine implementation for the MemMimic Memory Search System.

Provides hybrid search combining semantic and keyword matching with CXD classification,
performance optimization, and comprehensive error handling.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import replace

from .interfaces import (
    SearchEngine, SearchQuery, SearchResults, SearchResult, SearchContext,
    SearchMetrics, SearchType, SimilarityMetric, SimilarityCalculator,
    CXDIntegrationBridge, PerformanceCache, ResultProcessor, SearchConfig,
    SearchEngineError, SimilarityCalculationError, CXDIntegrationError,
    CacheError
)
from .search_config import DefaultSearchConfig

logger = logging.getLogger(__name__)


class HybridMemorySearchEngine:
    """
    High-performance hybrid search engine with semantic and keyword matching.
    
    Combines vector similarity search with traditional keyword matching,
    enhanced with CXD classification and performance optimization.
    """
    
    def __init__(
        self,
        similarity_calculator: SimilarityCalculator,
        cxd_bridge: CXDIntegrationBridge,
        cache: PerformanceCache,
        result_processor: ResultProcessor,
        config: Optional[SearchConfig] = None
    ):
        """
        Initialize the hybrid search engine.
        
        Args:
            similarity_calculator: Vector similarity computation
            cxd_bridge: CXD classification integration
            cache: Result caching system
            result_processor: Result ranking and filtering
            config: Search configuration (uses defaults if None)
        """
        self.similarity_calculator = similarity_calculator
        self.cxd_bridge = cxd_bridge
        self.cache = cache
        self.result_processor = result_processor
        self.config = config or DefaultSearchConfig()
        
        # Performance metrics
        self._metrics = SearchMetrics()
        self._search_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0
        self._error_count = 0
        
        logger.info("HybridMemorySearchEngine initialized")
    
    def search(self, query: SearchQuery) -> SearchResults:
        """
        Execute hybrid search and return ranked results.
        
        Args:
            query: Search query with parameters
            
        Returns:
            SearchResults with ranked and filtered results
            
        Raises:
            SearchEngineError: On search execution failure
        """
        start_time = time.perf_counter()
        cache_used = False
        
        try:
            # Validate query
            self._validate_query(query)
            
            # Check cache first
            cache_key = self._generate_cache_key(query)
            cached_results = self._try_get_cached_results(cache_key)
            
            if cached_results:
                cache_used = True
                self._cache_hits += 1
                logger.debug(f"Cache hit for query: {query.text[:50]}...")
                return self._update_search_context(cached_results, cache_used)
            
            self._cache_misses += 1
            
            # Preprocess query
            processed_query = self._preprocess_query(query)
            
            # Retrieve candidates based on search type
            candidates = self._retrieve_candidates(processed_query)
            
            # Score and rank candidates
            scored_candidates = self._score_candidates(processed_query, candidates)
            
            # Apply CXD enhancement if enabled
            if self.config.is_cxd_enabled():
                scored_candidates = self._enhance_with_cxd(processed_query, scored_candidates)
            
            # Process results (filter and rank)
            final_results = self._process_results(processed_query, scored_candidates)
            
            # Create search context
            search_time = (time.perf_counter() - start_time) * 1000
            context = SearchContext(
                search_time_ms=search_time,
                total_candidates=len(candidates),
                cache_used=cache_used,
                similarity_metric=self.config.get_similarity_metric(),
                cxd_classification_used=self.config.is_cxd_enabled()
            )
            
            # Build final results
            results = SearchResults(
                results=final_results,
                total_found=len(final_results),
                query=processed_query,
                search_context=context
            )
            
            # Cache results
            self._cache_results(cache_key, results)
            
            # Update metrics
            self._update_metrics(search_time)
            
            logger.debug(f"Search completed: {len(final_results)} results in {search_time:.2f}ms")
            return results
            
        except Exception as e:
            self._error_count += 1
            search_time = (time.perf_counter() - start_time) * 1000
            self._update_metrics(search_time)
            
            if isinstance(e, (SearchEngineError, SimilarityCalculationError, CXDIntegrationError)):
                raise
            
            logger.error(f"Search failed for query '{query.text}': {e}")
            raise SearchEngineError(
                f"Search execution failed: {str(e)}",
                error_code="SEARCH_EXECUTION_ERROR",
                context={"query": query.text, "search_time_ms": search_time}
            )
    
    def warm_cache(self, queries: List[str]) -> None:
        """
        Preload cache with likely search terms.
        
        Args:
            queries: List of query strings to preload
        """
        if not self.config.enable_cache_warming:
            logger.debug("Cache warming disabled")
            return
        
        logger.info(f"Warming cache with {len(queries)} queries")
        
        for query_text in queries:
            try:
                query = SearchQuery(text=query_text, limit=5)
                self.search(query)
            except Exception as e:
                logger.warning(f"Failed to warm cache for query '{query_text}': {e}")
        
        logger.info("Cache warming completed")
    
    def get_metrics(self) -> SearchMetrics:
        """Get current performance metrics."""
        return replace(self._metrics)
    
    def health_check(self) -> bool:
        """
        Check if search engine is operational.
        
        Returns:
            True if all components are healthy
        """
        try:
            # Test basic search functionality
            test_query = SearchQuery(text="test", limit=1)
            self.search(test_query)
            
            # Check cache health
            cache_stats = self.cache.get_stats()
            if cache_stats.get('errors', 0) > 0:
                logger.warning("Cache reporting errors")
                return False
            
            # Check error rate
            if self._get_error_rate() > 0.1:  # 10% error threshold
                logger.warning(f"High error rate: {self._get_error_rate():.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _validate_query(self, query: SearchQuery) -> None:
        """Validate search query parameters."""
        if not query.text or not query.text.strip():
            raise SearchEngineError(
                "Query text cannot be empty",
                error_code="EMPTY_QUERY"
            )
        
        if query.limit <= 0 or query.limit > self.config.get_max_results():
            raise SearchEngineError(
                f"Limit must be between 1 and {self.config.get_max_results()}",
                error_code="INVALID_LIMIT"
            )
        
        if not (0.0 <= query.min_confidence <= 1.0):
            raise SearchEngineError(
                "min_confidence must be between 0.0 and 1.0",
                error_code="INVALID_CONFIDENCE"
            )
    
    def _preprocess_query(self, query: SearchQuery) -> SearchQuery:
        """Preprocess query for optimal search performance."""
        if not self.config.enable_query_preprocessing:
            return query
        
        # Normalize whitespace and trim
        processed_text = " ".join(query.text.strip().split())
        
        # Apply any additional preprocessing here
        # (e.g., stemming, stop word removal, etc.)
        
        return replace(query, text=processed_text)
    
    def _retrieve_candidates(self, query: SearchQuery) -> List[SearchResult]:
        """Retrieve initial candidate results based on search type."""
        # This is a stub - in a real implementation, this would:
        # 1. Query the vector database for semantic candidates
        # 2. Query the keyword index for keyword candidates
        # 3. Combine results based on search type
        
        logger.debug(f"Retrieving candidates for {query.search_type.value} search")
        
        # Mock implementation - replace with actual retrieval logic
        candidates = []
        
        # In production, this would interface with:
        # - Vector database (e.g., Pinecone, Weaviate, Chroma)
        # - Full-text search (e.g., Elasticsearch, Solr)
        # - Database queries
        
        return candidates
    
    def _score_candidates(self, query: SearchQuery, candidates: List[SearchResult]) -> List[SearchResult]:
        """Apply similarity scoring to candidates."""
        if not candidates:
            return candidates
        
        try:
            # For hybrid search, combine semantic and keyword scores
            if query.search_type == SearchType.HYBRID:
                semantic_weight, keyword_weight = self.config.get_weighted_scores()
                
                for candidate in candidates:
                    # Calculate semantic similarity (mock - replace with actual embedding comparison)
                    semantic_score = 0.5  # Mock score
                    
                    # Calculate keyword similarity (mock - replace with actual keyword matching)
                    keyword_score = 0.3  # Mock score
                    
                    # Combine scores
                    combined_score = (semantic_score * semantic_weight + 
                                    keyword_score * keyword_weight)
                    
                    candidate.relevance_score = combined_score
            
            return candidates
            
        except Exception as e:
            raise SimilarityCalculationError(
                f"Similarity calculation failed: {str(e)}",
                error_code="SIMILARITY_CALC_ERROR",
                context={"query": query.text, "candidate_count": len(candidates)}
            )
    
    def _enhance_with_cxd(self, query: SearchQuery, candidates: List[SearchResult]) -> List[SearchResult]:
        """Enhance results with CXD classification."""
        try:
            return self.cxd_bridge.enhance_results(query, candidates)
        except Exception as e:
            logger.warning(f"CXD enhancement failed: {e}")
            # Don't fail the search if CXD enhancement fails
            return candidates
    
    def _process_results(self, query: SearchQuery, candidates: List[SearchResult]) -> List[SearchResult]:
        """Apply final processing (filtering and ranking) to results."""
        try:
            # Apply filters
            filtered = self.result_processor.filter_results(query, candidates)
            
            # Apply ranking
            ranked = self.result_processor.rank_results(query, filtered)
            
            # Apply confidence threshold
            final_results = [
                result for result in ranked 
                if result.relevance_score >= query.min_confidence
            ]
            
            # Limit results
            return final_results[:query.limit]
            
        except Exception as e:
            logger.warning(f"Result processing failed: {e}")
            # Return unprocessed candidates as fallback
            return candidates[:query.limit]
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for query."""
        return f"search:{hash((query.text, query.limit, str(query.filters), query.search_type.value))}"
    
    def _try_get_cached_results(self, cache_key: str) -> Optional[SearchResults]:
        """Attempt to retrieve cached results."""
        try:
            return self.cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def _cache_results(self, cache_key: str, results: SearchResults) -> None:
        """Cache search results."""
        try:
            self.cache.set(cache_key, results, self.config.get_cache_ttl())
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _update_search_context(self, results: SearchResults, cache_used: bool) -> SearchResults:
        """Update search context for cached results."""
        context = replace(results.search_context, cache_used=cache_used)
        return replace(results, search_context=context)
    
    def _update_metrics(self, search_time_ms: float) -> None:
        """Update performance metrics."""
        self._search_times.append(search_time_ms)
        
        # Keep only recent search times (last 1000)
        if len(self._search_times) > 1000:
            self._search_times = self._search_times[-1000:]
        
        # Update metrics
        self._metrics.total_searches += 1
        self._metrics.avg_response_time_ms = sum(self._search_times) / len(self._search_times)
        
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            self._metrics.cache_hit_rate = self._cache_hits / total_requests
        
        self._metrics.error_rate = self._get_error_rate()
        self._metrics.last_updated = datetime.now()
    
    def _get_error_rate(self) -> float:
        """Calculate current error rate."""
        total = self._metrics.total_searches
        return self._error_count / total if total > 0 else 0.0