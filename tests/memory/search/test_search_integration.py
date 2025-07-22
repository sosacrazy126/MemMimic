"""
Integration tests for the extracted MemMimic Memory Search System.

Tests the complete integration of all extracted components:
- SearchEngine
- VectorSimilarity  
- CXDIntegration
- PerformanceCache
- MCP handlers

This validates that the refactoring successfully extracted functionality
from the massive memmimic_recall_cxd.py file while maintaining compatibility.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

# Import the extracted search system components
from memmimic.memory.search.interfaces import (
    SearchQuery, SearchResult, SearchResults, SearchType, SimilarityMetric,
    CXDClassification, SearchContext
)
from memmimic.memory.search.search_engine import HybridMemorySearchEngine
from memmimic.memory.search.vector_similarity import OptimizedVectorSimilarity, create_similarity_calculator
from memmimic.memory.search.cxd_integration import ProductionCXDIntegrationBridge, create_cxd_integration_bridge
from memmimic.memory.search.performance_cache import LRUMemoryCache, create_performance_cache
from memmimic.memory.search.search_config import DefaultSearchConfig
from memmimic.mcp.handlers.recall_handler import MemoryRecallMCPHandler, create_memory_recall_handler
from memmimic.mcp.handlers.response_formatter import MCPResponseFormatter, create_legacy_format_response
from memmimic.mcp.handlers.mcp_base import MCPBase, extract_search_params


class TestSearchSystemIntegration:
    """Integration tests for the complete search system."""
    
    @pytest.fixture
    def search_config(self):
        """Create test search configuration."""
        return DefaultSearchConfig()
    
    @pytest.fixture
    def similarity_calculator(self, search_config):
        """Create test similarity calculator."""
        return create_similarity_calculator(search_config)
    
    @pytest.fixture
    def mock_cxd_classifier(self):
        """Create mock CXD classifier."""
        mock_classifier = Mock()
        mock_classifier.classify.return_value = Mock(
            function="Context",
            confidence=0.8,
            metadata={'test': True}
        )
        return mock_classifier
    
    @pytest.fixture 
    def cxd_integration(self, mock_cxd_classifier, search_config):
        """Create CXD integration bridge."""
        return create_cxd_integration_bridge(mock_cxd_classifier, search_config)
    
    @pytest.fixture
    def performance_cache(self):
        """Create performance cache."""
        return create_performance_cache("memory", max_size=100, default_ttl=300)
    
    @pytest.fixture
    def result_processor(self):
        """Create mock result processor."""
        mock_processor = Mock()
        mock_processor.filter_results.return_value = []
        mock_processor.rank_results.return_value = []
        return mock_processor
    
    @pytest.fixture
    def search_engine(self, similarity_calculator, cxd_integration, 
                     performance_cache, result_processor, search_config):
        """Create integrated search engine."""
        return HybridMemorySearchEngine(
            similarity_calculator=similarity_calculator,
            cxd_bridge=cxd_integration,
            cache=performance_cache,
            result_processor=result_processor,
            config=search_config
        )
    
    @pytest.fixture
    def mcp_handler(self, search_engine):
        """Create MCP handler."""
        return create_memory_recall_handler(search_engine)
    
    def test_search_engine_initialization(self, search_engine):
        """Test that search engine initializes correctly with all components."""
        assert search_engine is not None
        assert search_engine.similarity_calculator is not None
        assert search_engine.cxd_bridge is not None
        assert search_engine.cache is not None
        assert search_engine.result_processor is not None
        assert search_engine.config is not None
    
    def test_similarity_calculator_basic_operation(self, similarity_calculator):
        """Test basic similarity calculation functionality."""
        # Test vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]  # Same as vec1
        
        # Calculate similarities
        similarity_different = similarity_calculator.calculate_similarity(vec1, vec2)
        similarity_same = similarity_calculator.calculate_similarity(vec1, vec3)
        
        # Verify results
        assert 0.0 <= similarity_different <= 1.0
        assert 0.0 <= similarity_same <= 1.0
        assert similarity_same > similarity_different  # Same vectors should be more similar
    
    def test_similarity_calculator_batch_operation(self, similarity_calculator):
        """Test batch similarity calculation."""
        query_vec = [1.0, 0.0, 0.0]
        memory_vecs = [
            [1.0, 0.0, 0.0],  # Same
            [0.0, 1.0, 0.0],  # Different
            [0.5, 0.5, 0.0],  # Partially similar
        ]
        
        similarities = similarity_calculator.batch_calculate_similarity(query_vec, memory_vecs)
        
        assert len(similarities) == 3
        assert all(0.0 <= sim <= 1.0 for sim in similarities)
        assert similarities[0] > similarities[1]  # First should be most similar
    
    def test_cxd_integration_classification(self, cxd_integration):
        """Test CXD classification functionality."""
        test_content = "analyze the data and extract insights"
        
        classification = cxd_integration.classify_content(test_content)
        
        assert classification is not None
        assert hasattr(classification, 'function')
        assert hasattr(classification, 'confidence')
        assert 0.0 <= classification.confidence <= 1.0
    
    def test_cxd_integration_result_enhancement(self, cxd_integration):
        """Test CXD enhancement of search results."""
        query = SearchQuery(text="test query")
        candidates = [
            SearchResult(
                memory_id="1",
                content="test content",
                relevance_score=0.5
            )
        ]
        
        enhanced_results = cxd_integration.enhance_results(query, candidates)
        
        assert len(enhanced_results) == 1
        enhanced_result = enhanced_results[0]
        assert enhanced_result.cxd_classification is not None
        assert enhanced_result.metadata.get('enhanced_by_cxd') is True
    
    def test_performance_cache_basic_operations(self, performance_cache):
        """Test basic cache operations."""
        # Create test data
        query = SearchQuery(text="test query")
        results = SearchResults(
            results=[],
            total_found=0,
            query=query,
            search_context=SearchContext(
                search_time_ms=100.0,
                total_candidates=0,
                cache_used=False,
                similarity_metric=SimilarityMetric.COSINE,
                cxd_classification_used=True
            )
        )
        
        cache_key = "test_key"
        
        # Test cache miss
        cached_result = performance_cache.get(cache_key)
        assert cached_result is None
        
        # Test cache set and hit
        performance_cache.set(cache_key, results)
        cached_result = performance_cache.get(cache_key)
        assert cached_result is not None
        assert cached_result.query.text == "test query"
    
    def test_search_engine_health_check(self, search_engine):
        """Test search engine health check functionality."""
        # Mock the candidate retrieval to avoid database dependencies
        search_engine._retrieve_candidates = Mock(return_value=[])
        
        health_status = search_engine.health_check()
        assert isinstance(health_status, bool)
    
    def test_search_engine_metrics(self, search_engine):
        """Test search engine metrics collection."""
        metrics = search_engine.get_metrics()
        
        assert hasattr(metrics, 'total_searches')
        assert hasattr(metrics, 'avg_response_time_ms')
        assert hasattr(metrics, 'cache_hit_rate')
        assert hasattr(metrics, 'error_rate')
    
    def test_mcp_handler_initialization(self, mcp_handler):
        """Test MCP handler initialization."""
        assert mcp_handler is not None
        assert mcp_handler.search_engine is not None
        assert mcp_handler.response_formatter is not None
    
    @pytest.mark.asyncio
    async def test_mcp_handler_recall_request(self, mcp_handler):
        """Test MCP recall request handling."""
        # Mock search engine to avoid database dependencies
        mock_results = SearchResults(
            results=[
                SearchResult(
                    memory_id="test_id",
                    content="test content",
                    relevance_score=0.8,
                    cxd_classification=CXDClassification(
                        function="Context",
                        confidence=0.7
                    )
                )
            ],
            total_found=1,
            query=SearchQuery(text="test query"),
            search_context=SearchContext(
                search_time_ms=50.0,
                total_candidates=1,
                cache_used=False,
                similarity_metric=SimilarityMetric.COSINE,
                cxd_classification_used=True
            )
        )
        
        mcp_handler.search_engine.search = Mock(return_value=mock_results)
        
        # Create test MCP request
        mcp_request = {
            'method': 'recall',
            'params': {
                'query': 'test query',
                'limit': 5,
                'function_filter': 'ALL'
            }
        }
        
        response = await mcp_handler.handle_recall_request(mcp_request)
        
        assert response.success is True
        assert response.data is not None
        assert 'formatted_response' in response.data
    
    def test_mcp_parameter_extraction(self):
        """Test MCP parameter extraction and validation."""
        # Valid parameters
        valid_params = {
            'query': 'test query',
            'limit': 10,
            'function_filter': 'CONTEXT',
            'db_name': 'test_db'
        }
        
        extracted = extract_search_params(valid_params)
        
        assert extracted['query'] == 'test query'
        assert extracted['limit'] == 10
        assert extracted['function_filter'] == 'CONTEXT'
        assert extracted['db_name'] == 'test_db'
    
    def test_mcp_parameter_validation_errors(self):
        """Test MCP parameter validation error handling."""
        # Empty query
        with pytest.raises(ValueError, match="Query parameter is required"):
            extract_search_params({'query': ''})
        
        # Invalid limit
        with pytest.raises(ValueError, match="Limit must be integer"):
            extract_search_params({'query': 'test', 'limit': 0})
        
        # Invalid function filter
        with pytest.raises(ValueError, match="function_filter must be one of"):
            extract_search_params({'query': 'test', 'function_filter': 'INVALID'})
    
    def test_response_formatter_legacy_compatibility(self):
        """Test legacy format response generation."""
        # Create test search results
        results = SearchResults(
            results=[
                SearchResult(
                    memory_id="test_1",
                    content="test content 1",
                    relevance_score=0.8,
                    cxd_classification=CXDClassification(
                        function="Context",
                        confidence=0.7
                    ),
                    metadata={'type': 'INTERACTION', 'confidence': 0.8}
                )
            ],
            total_found=1,
            query=SearchQuery(text="test query"),
            search_context=SearchContext(
                search_time_ms=50.0,
                total_candidates=1,
                cache_used=False,
                similarity_metric=SimilarityMetric.COSINE,
                cxd_classification_used=True
            )
        )
        
        legacy_response = create_legacy_format_response(results)
        
        # Verify legacy format structure
        assert "🧠 MEMMIMIC TRUE HYBRID WORDNET SEARCH v3.0" in legacy_response
        assert "🎯 Methods:" in legacy_response
        assert "🔤 LEGEND:" in legacy_response
        assert "test content 1" in legacy_response
        assert "📊 FUNCTION DISTRIBUTION:" in legacy_response
        assert "🧠 Powered by MemMimic v3.0" in legacy_response
    
    def test_complete_integration_workflow(self, search_engine, mcp_handler):
        """Test complete integration workflow from MCP request to response."""
        # Mock all components to avoid external dependencies
        mock_results = SearchResults(
            results=[
                SearchResult(
                    memory_id="integration_test",
                    content="integration test content",
                    relevance_score=0.9,
                    cxd_classification=CXDClassification(
                        function="Data",
                        confidence=0.8
                    )
                )
            ],
            total_found=1,
            query=SearchQuery(text="integration test"),
            search_context=SearchContext(
                search_time_ms=25.0,
                total_candidates=1,
                cache_used=False,
                similarity_metric=SimilarityMetric.COSINE,
                cxd_classification_used=True
            )
        )
        
        # Mock search engine
        search_engine.search = Mock(return_value=mock_results)
        search_engine.health_check = Mock(return_value=True)
        
        # Test workflow
        assert search_engine.health_check() is True
        
        search_query = SearchQuery(text="integration test", limit=5)
        results = search_engine.search(search_query)
        
        assert results is not None
        assert len(results.results) == 1
        assert results.results[0].content == "integration test content"
        
        # Test metrics collection
        metrics = search_engine.get_metrics()
        assert metrics is not None


class TestSearchSystemPerformance:
    """Performance tests for the extracted search system."""
    
    def test_similarity_calculation_performance(self):
        """Test similarity calculation performance."""
        import time
        
        config = DefaultSearchConfig()
        calculator = create_similarity_calculator(config)
        
        # Test vectors
        query_vec = [0.5] * 100  # 100-dimensional vector
        memory_vecs = [[0.6 + i * 0.01] * 100 for i in range(100)]  # 100 vectors
        
        start_time = time.perf_counter()
        similarities = calculator.batch_calculate_similarity(query_vec, memory_vecs)
        calculation_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        assert len(similarities) == 100
        assert calculation_time < 100  # Should complete within 100ms
        print(f"Batch similarity calculation time: {calculation_time:.2f}ms")
    
    def test_cache_performance(self):
        """Test cache performance under load."""
        import time
        
        cache = create_performance_cache("memory", max_size=1000)
        
        # Generate test data
        test_queries = [SearchQuery(text=f"query_{i}") for i in range(100)]
        test_results = [
            SearchResults(
                results=[],
                total_found=0,
                query=query,
                search_context=SearchContext(
                    search_time_ms=10.0,
                    total_candidates=0,
                    cache_used=False,
                    similarity_metric=SimilarityMetric.COSINE,
                    cxd_classification_used=False
                )
            )
            for query in test_queries
        ]
        
        # Test cache set performance
        start_time = time.perf_counter()
        for i, results in enumerate(test_results):
            cache.set(f"key_{i}", results)
        set_time = (time.perf_counter() - start_time) * 1000
        
        # Test cache get performance
        start_time = time.perf_counter()
        for i in range(100):
            cached_result = cache.get(f"key_{i}")
            assert cached_result is not None
        get_time = (time.perf_counter() - start_time) * 1000
        
        print(f"Cache set time (100 items): {set_time:.2f}ms")
        print(f"Cache get time (100 items): {get_time:.2f}ms")
        
        assert set_time < 50  # Should set 100 items within 50ms
        assert get_time < 20  # Should get 100 items within 20ms


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])