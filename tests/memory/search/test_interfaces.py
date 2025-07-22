"""
Test suite for memory search interfaces and data models.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from memmimic.memory.search.interfaces import (
    SearchQuery,
    SearchResult,
    SearchResults,
    SearchContext,
    CXDClassification,
    SearchMetrics,
    SearchType,
    SimilarityMetric,
    SearchError,
    SearchEngineError,
    ConfigurationError,
)


class TestSearchQuery:
    """Test SearchQuery data model"""
    
    def test_default_values(self):
        """Test SearchQuery with default values"""
        query = SearchQuery(text="test query")
        
        assert query.text == "test query"
        assert query.limit == 10
        assert query.filters == {}
        assert query.include_metadata is True
        assert query.min_confidence == 0.0
        assert query.search_type == SearchType.HYBRID
        assert query.timeout_ms == 5000
    
    def test_custom_values(self):
        """Test SearchQuery with custom values"""
        query = SearchQuery(
            text="custom query",
            limit=20,
            filters={"type": "interaction"},
            include_metadata=False,
            min_confidence=0.5,
            search_type=SearchType.SEMANTIC,
            timeout_ms=1000
        )
        
        assert query.text == "custom query"
        assert query.limit == 20
        assert query.filters == {"type": "interaction"}
        assert query.include_metadata is False
        assert query.min_confidence == 0.5
        assert query.search_type == SearchType.SEMANTIC
        assert query.timeout_ms == 1000


class TestSearchResult:
    """Test SearchResult data model"""
    
    def test_basic_result(self):
        """Test basic SearchResult creation"""
        result = SearchResult(
            memory_id="123",
            content="test content",
            relevance_score=0.85
        )
        
        assert result.memory_id == "123"
        assert result.content == "test content"
        assert result.relevance_score == 0.85
        assert result.cxd_classification is None
        assert result.metadata == {}
        assert result.search_context is None
    
    def test_result_with_cxd(self):
        """Test SearchResult with CXD classification"""
        cxd = CXDClassification(
            function="Context",
            confidence=0.9,
            metadata={"reasoning": "contextual information"}
        )
        
        result = SearchResult(
            memory_id="456",
            content="contextual content",
            relevance_score=0.75,
            cxd_classification=cxd
        )
        
        assert result.cxd_classification.function == "Context"
        assert result.cxd_classification.confidence == 0.9


class TestSearchResults:
    """Test SearchResults container"""
    
    def test_search_results_creation(self):
        """Test SearchResults creation"""
        query = SearchQuery(text="test")
        context = SearchContext(
            search_time_ms=50.0,
            total_candidates=100,
            cache_used=True,
            similarity_metric=SimilarityMetric.COSINE,
            cxd_classification_used=True
        )
        
        results = [
            SearchResult("1", "content1", 0.9),
            SearchResult("2", "content2", 0.8),
        ]
        
        search_results = SearchResults(
            results=results,
            total_found=2,
            query=query,
            search_context=context
        )
        
        assert len(search_results.results) == 2
        assert search_results.total_found == 2
        assert search_results.query.text == "test"
        assert search_results.search_context.search_time_ms == 50.0
        assert isinstance(search_results.timestamp, datetime)


class TestSearchMetrics:
    """Test SearchMetrics data model"""
    
    def test_default_metrics(self):
        """Test default SearchMetrics values"""
        metrics = SearchMetrics()
        
        assert metrics.total_searches == 0
        assert metrics.avg_response_time_ms == 0.0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.error_rate == 0.0
        assert isinstance(metrics.last_updated, datetime)
    
    def test_custom_metrics(self):
        """Test SearchMetrics with custom values"""
        metrics = SearchMetrics(
            total_searches=100,
            avg_response_time_ms=75.5,
            cache_hit_rate=0.85,
            error_rate=0.02
        )
        
        assert metrics.total_searches == 100
        assert metrics.avg_response_time_ms == 75.5
        assert metrics.cache_hit_rate == 0.85
        assert metrics.error_rate == 0.02


class TestCXDClassification:
    """Test CXDClassification data model"""
    
    def test_basic_classification(self):
        """Test basic CXD classification"""
        classification = CXDClassification(
            function="Control",
            confidence=0.95
        )
        
        assert classification.function == "Control"
        assert classification.confidence == 0.95
        assert classification.metadata == {}
    
    def test_classification_with_metadata(self):
        """Test CXD classification with metadata"""
        classification = CXDClassification(
            function="Data",
            confidence=0.8,
            metadata={
                "reasoning": "contains factual information",
                "keywords": ["fact", "data", "information"]
            }
        )
        
        assert classification.function == "Data"
        assert classification.confidence == 0.8
        assert "reasoning" in classification.metadata
        assert len(classification.metadata["keywords"]) == 3


class TestSearchExceptions:
    """Test search exception hierarchy"""
    
    def test_search_error_basic(self):
        """Test basic SearchError"""
        error = SearchError("Test error")
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code == "SearchError"
        assert error.context == {}
        assert isinstance(error.timestamp, datetime)
    
    def test_search_error_with_context(self):
        """Test SearchError with context"""
        context = {"query": "test", "timeout": 5000}
        error = SearchError(
            "Search timeout",
            error_code="SEARCH_TIMEOUT",
            context=context
        )
        
        assert error.message == "Search timeout"
        assert error.error_code == "SEARCH_TIMEOUT"
        assert error.context == context
    
    def test_search_engine_error(self):
        """Test SearchEngineError inheritance"""
        error = SearchEngineError("Engine failure")
        
        assert isinstance(error, SearchError)
        assert error.error_code == "SearchEngineError"
    
    def test_configuration_error(self):
        """Test ConfigurationError inheritance"""
        error = ConfigurationError("Invalid config")
        
        assert isinstance(error, SearchError)
        assert error.error_code == "ConfigurationError"


class TestEnums:
    """Test enumeration types"""
    
    def test_search_type_enum(self):
        """Test SearchType enum values"""
        assert SearchType.SEMANTIC.value == "semantic"
        assert SearchType.KEYWORD.value == "keyword"
        assert SearchType.HYBRID.value == "hybrid"
    
    def test_similarity_metric_enum(self):
        """Test SimilarityMetric enum values"""
        assert SimilarityMetric.COSINE.value == "cosine"
        assert SimilarityMetric.EUCLIDEAN.value == "euclidean"
        assert SimilarityMetric.DOT_PRODUCT.value == "dot_product"


if __name__ == "__main__":
    pytest.main([__file__])