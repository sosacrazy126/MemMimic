"""
Integration tests for the Memory Search System.

Tests the complete search pipeline with all components working together.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock

# Set up path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from memmimic.memory.search.interfaces import (
    SearchQuery, SearchResult, SearchResults, SearchContext, CXDClassification,
    SearchType, SimilarityMetric, SearchEngine
)
from memmimic.memory.search.search_config import DefaultSearchConfig
from memmimic.memory.search.vector_similarity import OptimizedVectorSimilarity
from memmimic.memory.search.performance_cache import LRUMemoryCache, generate_search_cache_key
from memmimic.memory.search.search_engine import HybridMemorySearchEngine


class MockMemoryStore:
    """Mock memory store for testing"""
    
    def __init__(self):
        self.memories = [
            {
                'id': '1',
                'content': 'This is about machine learning and artificial intelligence',
                'embedding': [0.8, 0.6, 0.2, 0.1],
                'metadata': {'type': 'technical', 'confidence': 0.9}
            },
            {
                'id': '2', 
                'content': 'User asked about cooking recipes and food preparation',
                'embedding': [0.2, 0.8, 0.7, 0.3],
                'metadata': {'type': 'interaction', 'confidence': 0.8}
            },
            {
                'id': '3',
                'content': 'Discussion about project management and planning',
                'embedding': [0.6, 0.3, 0.8, 0.5],
                'metadata': {'type': 'planning', 'confidence': 0.7}
            },
            {
                'id': '4',
                'content': 'Technical documentation about API development',
                'embedding': [0.9, 0.1, 0.4, 0.8],
                'metadata': {'type': 'technical', 'confidence': 0.95}
            }
        ]
    
    def search_candidates(self, query_text: str, limit: int = 10):
        """Return all memories as candidates for simplicity"""
        return self.memories[:limit]
    
    def get_embedding(self, content: str):
        """Generate mock embedding for content"""
        # Simple hash-based embedding for testing
        hash_val = hash(content.lower()) % 1000
        return [
            (hash_val % 100) / 100.0,
            ((hash_val + 1) % 100) / 100.0,
            ((hash_val + 2) % 100) / 100.0,
            ((hash_val + 3) % 100) / 100.0,
        ]


class MockCXDClassifier:
    """Mock CXD classifier for testing"""
    
    def classify(self, content: str):
        """Return mock CXD classification"""
        if 'technical' in content.lower() or 'api' in content.lower():
            return CXDClassification('Data', 0.9, {'reasoning': 'technical content'})
        elif 'user' in content.lower() or 'asked' in content.lower():
            return CXDClassification('Context', 0.8, {'reasoning': 'user interaction'})
        else:
            return CXDClassification('Control', 0.7, {'reasoning': 'general content'})


class MockResultProcessor:
    """Mock result processor for testing"""
    
    def rank_results(self, query, candidates):
        """Sort by relevance score descending"""
        return sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
    
    def filter_results(self, query, candidates):
        """Apply basic filtering"""
        filtered = []
        for candidate in candidates:
            if candidate.relevance_score >= query.min_confidence:
                filtered.append(candidate)
        return filtered[:query.limit]


class TestMemorySearchIntegration:
    """Integration tests for complete memory search system"""
    
    @pytest.fixture
    def search_config(self):
        """Create test search configuration"""
        return DefaultSearchConfig(
            similarity_metric=SimilarityMetric.COSINE,
            max_results=10,
            cache_ttl=60,
            enable_cxd_classification=True
        )
    
    @pytest.fixture
    def memory_store(self):
        """Create mock memory store"""
        return MockMemoryStore()
    
    @pytest.fixture
    def similarity_calculator(self, search_config):
        """Create similarity calculator"""
        return OptimizedVectorSimilarity(search_config)
    
    @pytest.fixture
    def performance_cache(self):
        """Create performance cache"""
        return LRUMemoryCache(max_size=100, default_ttl=60)
    
    @pytest.fixture
    def cxd_classifier(self):
        """Create mock CXD classifier"""
        return MockCXDClassifier()
    
    @pytest.fixture
    def result_processor(self):
        """Create mock result processor"""
        return MockResultProcessor()
    
    @pytest.fixture
    def search_engine(self, search_config, memory_store, similarity_calculator, 
                     performance_cache, cxd_classifier, result_processor):
        """Create complete search engine with all dependencies"""
        
        # Mock the CXD integration bridge
        cxd_bridge = Mock()
        cxd_bridge.enhance_results.side_effect = self._mock_cxd_enhance
        cxd_bridge.classify_content = cxd_classifier.classify
        
        engine = HybridMemorySearchEngine(
            similarity_calculator=similarity_calculator,
            cxd_bridge=cxd_bridge,
            cache=performance_cache,
            result_processor=result_processor,
            config=search_config
        )
        
        # Inject memory store for testing
        engine.memory_store = memory_store
        
        return engine
    
    def _mock_cxd_enhance(self, query, candidates):
        """Mock CXD enhancement of search results"""
        enhanced = []
        classifier = MockCXDClassifier()
        
        for candidate in candidates:
            classification = classifier.classify(candidate.content)
            enhanced_candidate = SearchResult(
                memory_id=candidate.memory_id,
                content=candidate.content,
                relevance_score=candidate.relevance_score,
                cxd_classification=classification,
                metadata=candidate.metadata,
                search_context=candidate.search_context
            )
            enhanced.append(enhanced_candidate)
        
        return enhanced
    
    def test_basic_search_functionality(self, search_engine):
        """Test basic search functionality works end-to-end"""
        query = SearchQuery(
            text="machine learning artificial intelligence",
            limit=5,
            search_type=SearchType.HYBRID
        )
        
        results = search_engine.search(query)
        
        # Verify results structure
        assert isinstance(results, SearchResults)
        assert len(results.results) > 0
        assert results.query.text == query.text
        assert results.search_context is not None
        
        # Verify first result has expected fields
        first_result = results.results[0]
        assert first_result.memory_id is not None
        assert first_result.content is not None
        assert 0.0 <= first_result.relevance_score <= 1.0
        assert first_result.cxd_classification is not None
    
    def test_search_with_caching(self, search_engine):
        """Test that caching works correctly"""
        query = SearchQuery(text="test caching functionality")
        
        # First search - should miss cache
        start_time = time.time()
        results1 = search_engine.search(query)
        first_search_time = time.time() - start_time
        
        # Second search - should hit cache
        start_time = time.time()
        results2 = search_engine.search(query)
        second_search_time = time.time() - start_time
        
        # Verify results are identical
        assert len(results1.results) == len(results2.results)
        assert results1.results[0].memory_id == results2.results[0].memory_id
        
        # Verify second search was faster (cache hit)
        assert second_search_time < first_search_time
        
        # Verify cache metrics
        metrics = search_engine.get_metrics()
        assert metrics.total_searches >= 2
    
    def test_similarity_calculation_integration(self, search_engine):
        """Test similarity calculations work in search pipeline"""
        query = SearchQuery(
            text="artificial intelligence machine learning",
            limit=3,
            search_type=SearchType.SEMANTIC
        )
        
        results = search_engine.search(query)
        
        # Verify results are ranked by relevance
        relevance_scores = [r.relevance_score for r in results.results]
        assert relevance_scores == sorted(relevance_scores, reverse=True)
        
        # Verify all scores are in valid range
        for score in relevance_scores:
            assert 0.0 <= score <= 1.0
    
    def test_cxd_classification_integration(self, search_engine):
        """Test CXD classification is properly integrated"""
        query = SearchQuery(text="technical API documentation")
        
        results = search_engine.search(query)
        
        # Verify CXD classifications are present
        for result in results.results:
            assert result.cxd_classification is not None
            assert result.cxd_classification.function in ['Control', 'Context', 'Data']
            assert 0.0 <= result.cxd_classification.confidence <= 1.0
    
    def test_filtering_and_ranking(self, search_engine):
        """Test result filtering and ranking"""
        query = SearchQuery(
            text="test filtering",
            limit=2,
            min_confidence=0.3
        )
        
        results = search_engine.search(query)
        
        # Verify filtering by confidence
        for result in results.results:
            assert result.relevance_score >= query.min_confidence
        
        # Verify limit is respected
        assert len(results.results) <= query.limit
        
        # Verify ranking (highest scores first)
        scores = [r.relevance_score for r in results.results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_metrics_collection(self, search_engine):
        """Test that search metrics are properly collected"""
        # Perform multiple searches
        queries = [
            SearchQuery(text="query 1"),
            SearchQuery(text="query 2"),
            SearchQuery(text="query 3"),
        ]
        
        for query in queries:
            search_engine.search(query)
        
        # Check metrics
        metrics = search_engine.get_metrics()
        assert metrics.total_searches >= 3
        assert metrics.avg_response_time_ms > 0
        assert 0.0 <= metrics.cache_hit_rate <= 1.0
        assert metrics.error_rate >= 0.0
    
    def test_health_check(self, search_engine):
        """Test search engine health check"""
        health_status = search_engine.health_check()
        assert health_status is True
    
    def test_cache_warming(self, search_engine):
        """Test cache warming functionality"""
        queries = ["query 1", "query 2", "query 3"]
        
        search_engine.warm_cache(queries)
        
        # Perform searches that should hit warmed cache
        for query_text in queries:
            query = SearchQuery(text=query_text)
            results = search_engine.search(query)
            assert results is not None
        
        # Verify cache hit rate improved
        metrics = search_engine.get_metrics()
        assert metrics.cache_hit_rate > 0
    
    def test_error_handling(self, search_engine):
        """Test error handling in search pipeline"""
        # Test with empty query
        empty_query = SearchQuery(text="")
        results = search_engine.search(empty_query)
        assert isinstance(results, SearchResults)
        
        # Test with very long query
        long_query = SearchQuery(text="x" * 10000)
        results = search_engine.search(long_query)
        assert isinstance(results, SearchResults)
    
    def test_different_search_types(self, search_engine):
        """Test different search types work correctly"""
        base_query_text = "machine learning AI"
        
        # Test semantic search
        semantic_query = SearchQuery(text=base_query_text, search_type=SearchType.SEMANTIC)
        semantic_results = search_engine.search(semantic_query)
        
        # Test keyword search  
        keyword_query = SearchQuery(text=base_query_text, search_type=SearchType.KEYWORD)
        keyword_results = search_engine.search(keyword_query)
        
        # Test hybrid search
        hybrid_query = SearchQuery(text=base_query_text, search_type=SearchType.HYBRID)
        hybrid_results = search_engine.search(hybrid_query)
        
        # All should return results
        assert len(semantic_results.results) > 0
        assert len(keyword_results.results) > 0
        assert len(hybrid_results.results) > 0
        
        # Results may differ between search types
        # (exact comparison would depend on implementation details)
    
    def test_concurrent_searches(self, search_engine):
        """Test system handles concurrent searches correctly"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def perform_search(query_text):
            query = SearchQuery(text=f"concurrent search {query_text}")
            results = search_engine.search(query)
            results_queue.put(results)
        
        # Start multiple concurrent searches
        threads = []
        for i in range(5):
            thread = threading.Thread(target=perform_search, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Verify all searches completed successfully
        assert results_queue.qsize() == 5
        
        while not results_queue.empty():
            results = results_queue.get()
            assert isinstance(results, SearchResults)
            assert len(results.results) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])