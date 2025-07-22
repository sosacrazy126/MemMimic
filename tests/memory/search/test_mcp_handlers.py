"""
Unit tests for MCP handlers in the Memory Search System.

Tests the MCP protocol handling functionality including request validation,
response formatting, and error handling.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

# Set up path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from memmimic.memory.search.interfaces import (
    SearchQuery, SearchResult, SearchResults, SearchContext, CXDClassification,
    SearchType, SimilarityMetric, SearchEngineError
)
from memmimic.memory.search.mcp_handlers import (
    MemoryRecallMCPHandler, MemoryRecallCXDHandler, create_memory_recall_handler
)
from memmimic.memory.search.mcp_base import (
    MCPRequest, MCPResponse, MCPError, MCPBaseHandler, MCPHandlerRegistry,
    create_mcp_request, create_success_response, create_error_response
)
from memmimic.memory.search.response_formatter import MCPResponseFormatter


class MockSearchEngine:
    """Mock search engine for testing MCP handlers"""
    
    def __init__(self, return_results=True):
        self.return_results = return_results
        self.search_calls = []
        
    def search(self, query):
        self.search_calls.append(query)
        
        if not self.return_results:
            raise SearchEngineError("Mock search failure", error_code="MOCK_ERROR")
        
        # Create mock results
        mock_result = SearchResult(
            memory_id="test_id_1",
            content="This is a test memory about machine learning",
            relevance_score=0.85,
            cxd_classification=CXDClassification(
                function="Data",
                confidence=0.9,
                metadata={"test": True}
            ),
            metadata={"type": "technical", "confidence": 0.9},
            search_context={"source": "test"}
        )
        
        context = SearchContext(
            search_time_ms=50.0,
            total_candidates=5,
            cache_used=False,
            similarity_metric=SimilarityMetric.COSINE,
            cxd_classification_used=True
        )
        
        return SearchResults(
            results=[mock_result],
            total_found=1,
            query=query,
            search_context=context
        )
    
    def health_check(self):
        return True


class TestMCPBaseClasses:
    """Test base MCP classes and utilities"""
    
    def test_mcp_request_creation(self):
        """Test MCPRequest creation and parameter access"""
        parameters = {"query": "test query", "limit": 5}
        request = create_mcp_request("req_123", "memory_recall", parameters)
        
        assert request.request_id == "req_123"
        assert request.method == "memory_recall"
        assert request.parameters == parameters
        assert isinstance(request.timestamp, datetime)
        
        # Test parameter access
        assert request.get_parameter("query") == "test query"
        assert request.get_parameter("limit") == 5
        assert request.get_parameter("missing", "default") == "default"
        assert request.has_parameter("query") is True
        assert request.has_parameter("missing") is False
    
    def test_mcp_response_creation(self):
        """Test MCPResponse creation and serialization"""
        data = {"results": [], "total": 0}
        response = create_success_response("req_123", data, 100.5)
        
        assert response.request_id == "req_123"
        assert response.success is True
        assert response.data == data
        assert response.error is None
        assert response.processing_time_ms == 100.5
        
        # Test serialization
        response_dict = response.to_dict()
        assert response_dict["success"] is True
        assert response_dict["data"] == data
        assert "timestamp" in response_dict
    
    def test_mcp_error_creation(self):
        """Test MCPError creation and formatting"""
        error = MCPError(
            "Test error message",
            error_code="TEST_ERROR",
            request_id="req_123",
            context={"test": True}
        )
        
        assert str(error) == "Test error message"
        assert error.error_code == "TEST_ERROR"
        assert error.request_id == "req_123"
        assert error.context == {"test": True}
        
        # Test error dict conversion
        error_dict = error.to_dict()
        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error message"
        assert error_dict["context"] == {"test": True}


class TestMemoryRecallMCPHandler:
    """Test MemoryRecallMCPHandler functionality"""
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine"""
        return MockSearchEngine()
    
    @pytest.fixture
    def handler(self, mock_search_engine):
        """Create MemoryRecallMCPHandler instance"""
        return MemoryRecallMCPHandler(mock_search_engine)
    
    def test_handler_initialization(self, handler, mock_search_engine):
        """Test handler initialization"""
        assert handler.handler_type == "memory_recall"
        assert handler.search_engine == mock_search_engine
        assert handler.formatter is not None
        assert isinstance(handler._handler_metrics, dict)
    
    def test_successful_request_handling(self, handler):
        """Test successful memory recall request"""
        request = create_mcp_request(
            "req_123",
            "memory_recall",
            {
                "query": "machine learning",
                "limit": 5,
                "min_confidence": 0.3
            }
        )
        
        response = handler.handle_request(request)
        
        assert isinstance(response, MCPResponse)
        assert response.success is True
        assert response.data is not None
        assert "results" in response.data
        assert "metadata" in response.data
        
        # Verify search was called with correct parameters
        search_calls = handler.search_engine.search_calls
        assert len(search_calls) == 1
        
        search_query = search_calls[0]
        assert search_query.text == "machine learning"
        assert search_query.limit == 5
        assert search_query.min_confidence == 0.3
    
    def test_request_validation(self, handler):
        """Test request parameter validation"""
        # Test missing query parameter
        request = create_mcp_request("req_123", "memory_recall", {})
        response = handler.handle_request(request)
        
        assert response.success is False
        assert response.error["error_code"] == "MISSING_PARAMETER"
        
        # Test empty query
        request = create_mcp_request("req_123", "memory_recall", {"query": ""})
        response = handler.handle_request(request)
        
        assert response.success is False
        assert response.error["error_code"] == "EMPTY_QUERY"
        
        # Test invalid limit
        request = create_mcp_request("req_123", "memory_recall", {
            "query": "test",
            "limit": 200  # Too high
        })
        response = handler.handle_request(request)
        
        assert response.success is False
        assert response.error["error_code"] == "INVALID_LIMIT"
        
        # Test invalid confidence
        request = create_mcp_request("req_123", "memory_recall", {
            "query": "test",
            "min_confidence": 1.5  # Out of range
        })
        response = handler.handle_request(request)
        
        assert response.success is False
        assert response.error["error_code"] == "INVALID_CONFIDENCE"
    
    def test_search_engine_error_handling(self, mock_search_engine):
        """Test handling of search engine errors"""
        # Configure mock to raise error
        failing_engine = MockSearchEngine(return_results=False)
        handler = MemoryRecallMCPHandler(failing_engine)
        
        request = create_mcp_request("req_123", "memory_recall", {"query": "test"})
        response = handler.handle_request(request)
        
        assert response.success is False
        assert response.error is not None
        assert "Mock search failure" in str(response.error)
    
    def test_search_type_parameter_handling(self, handler):
        """Test search type parameter parsing"""
        request = create_mcp_request("req_123", "memory_recall", {
            "query": "test",
            "search_type": "semantic"
        })
        
        response = handler.handle_request(request)
        search_query = handler.search_engine.search_calls[0]
        
        assert search_query.search_type == SearchType.SEMANTIC
        
        # Test invalid search type (should default to hybrid)
        request = create_mcp_request("req_123", "memory_recall", {
            "query": "test",
            "search_type": "invalid_type"
        })
        
        response = handler.handle_request(request)
        search_query = handler.search_engine.search_calls[1]
        
        assert search_query.search_type == SearchType.HYBRID
    
    def test_metrics_collection(self, handler):
        """Test handler metrics collection"""
        # Make successful request
        request = create_mcp_request("req_123", "memory_recall", {"query": "test"})
        handler.handle_request(request)
        
        # Make failed request
        request = create_mcp_request("req_124", "memory_recall", {})  # Missing query
        handler.handle_request(request)
        
        metrics = handler.get_handler_metrics()
        
        assert metrics["total_requests"] >= 2
        assert metrics["successful_requests"] >= 1
        assert metrics["failed_requests"] >= 1
        assert "success_rate" in metrics
        assert "avg_processing_time_ms" in metrics
        assert "search_type_usage" in metrics
        assert "error_types" in metrics
    
    def test_health_check(self, handler):
        """Test handler health check"""
        health = handler.health_check()
        
        assert isinstance(health, dict)
        assert "healthy" in health
        assert "search_engine_healthy" in health
        assert "error_rate" in health
        assert "status" in health


class TestMemoryRecallCXDHandler:
    """Test MemoryRecallCXDHandler functionality"""
    
    @pytest.fixture
    def mock_search_engine(self):
        """Create mock search engine"""
        return MockSearchEngine()
    
    @pytest.fixture
    def cxd_handler(self, mock_search_engine):
        """Create MemoryRecallCXDHandler instance"""
        return MemoryRecallCXDHandler(mock_search_engine)
    
    def test_cxd_handler_initialization(self, cxd_handler, mock_search_engine):
        """Test CXD handler initialization"""
        assert cxd_handler.handler_type == "memory_recall_cxd"
        assert cxd_handler.search_engine == mock_search_engine
        assert isinstance(cxd_handler._cxd_metrics, dict)
    
    def test_cxd_function_filtering(self, cxd_handler):
        """Test CXD function filtering"""
        request = create_mcp_request("req_123", "memory_recall_cxd", {
            "query": "test query",
            "function_filter": "Data"
        })
        
        response = cxd_handler.handle_request(request)
        
        assert response.success is True
        assert response.data is not None
        
        # Check that CXD metrics were updated
        cxd_metrics = cxd_handler.get_cxd_metrics()
        assert cxd_metrics["function_filter_usage"]["Data"] == 1
    
    def test_cxd_function_filter_validation(self, cxd_handler):
        """Test CXD function filter validation"""
        request = create_mcp_request("req_123", "memory_recall_cxd", {
            "query": "test query",
            "function_filter": "InvalidFunction"
        })
        
        response = cxd_handler.handle_request(request)
        
        assert response.success is False
        assert response.error["error_code"] == "INVALID_FUNCTION_FILTER"
    
    def test_cxd_enhanced_response_format(self, cxd_handler):
        """Test CXD-enhanced response formatting"""
        request = create_mcp_request("req_123", "memory_recall_cxd", {
            "query": "test query",
            "include_cxd_analysis": True
        })
        
        response = cxd_handler.handle_request(request)
        
        assert response.success is True
        assert "cxd_analysis" in response.data
        
        # Check that results include CXD classification
        results = response.data["results"]
        assert len(results) > 0
        assert "cxd_classification" in results[0]


class TestMCPHandlerRegistry:
    """Test MCP handler registry functionality"""
    
    def test_handler_registration_and_retrieval(self):
        """Test handler registration and retrieval"""
        registry = MCPHandlerRegistry()
        mock_engine = MockSearchEngine()
        handler = MemoryRecallMCPHandler(mock_engine)
        
        # Register handler
        registry.register_handler("memory_recall", handler)
        
        # Retrieve handler
        retrieved_handler = registry.get_handler("memory_recall")
        assert retrieved_handler == handler
        
        # Test unknown method
        unknown_handler = registry.get_handler("unknown_method")
        assert unknown_handler is None
    
    def test_request_routing(self):
        """Test request routing through registry"""
        registry = MCPHandlerRegistry()
        mock_engine = MockSearchEngine()
        handler = MemoryRecallMCPHandler(mock_engine)
        
        registry.register_handler("memory_recall", handler)
        
        request = create_mcp_request("req_123", "memory_recall", {"query": "test"})
        response = registry.route_request(request)
        
        assert response.success is True
        
        # Test unknown method routing
        unknown_request = create_mcp_request("req_124", "unknown_method", {})
        unknown_response = registry.route_request(unknown_request)
        
        assert unknown_response.success is False
        assert unknown_response.error["error_code"] == "ROUTING_ERROR"
    
    def test_registry_status(self):
        """Test registry status reporting"""
        registry = MCPHandlerRegistry()
        mock_engine = MockSearchEngine()
        handler = MemoryRecallMCPHandler(mock_engine)
        
        registry.register_handler("memory_recall", handler)
        
        status = registry.get_registry_status()
        
        assert status["total_handlers"] == 1
        assert "memory_recall" in status["registered_methods"]
        assert "handlers" in status
        assert "registry_uptime" in status


class TestMCPResponseFormatter:
    """Test MCP response formatter"""
    
    @pytest.fixture
    def formatter(self):
        """Create MCPResponseFormatter instance"""
        return MCPResponseFormatter(include_debug_info=True)
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results for testing"""
        result = SearchResult(
            memory_id="test_id",
            content="Test content",
            relevance_score=0.8,
            cxd_classification=CXDClassification("Data", 0.9, {}),
            metadata={"test": True}
        )
        
        context = SearchContext(
            search_time_ms=100.0,
            total_candidates=5,
            cache_used=False,
            similarity_metric=SimilarityMetric.COSINE,
            cxd_classification_used=True
        )
        
        query = SearchQuery(text="test query")
        
        return SearchResults(
            results=[result],
            total_found=1,
            query=query,
            search_context=context
        )
    
    def test_detailed_format(self, formatter, sample_search_results):
        """Test detailed result formatting"""
        response = formatter.format_search_results(sample_search_results, "detailed")
        
        assert response.success is True
        assert "results" in response.data
        assert "metadata" in response.data
        assert len(response.data["results"]) == 1
        
        result = response.data["results"][0]
        assert "memory_id" in result
        assert "content" in result
        assert "relevance_score" in result
        assert "cxd_classification" in result
    
    def test_compact_format(self, formatter, sample_search_results):
        """Test compact result formatting"""
        response = formatter.format_search_results(sample_search_results, "compact")
        
        assert response.success is True
        assert "results" in response.data
        assert "metadata" in response.data
        
        # Compact format should have fewer fields
        result = response.data["results"][0]
        assert "memory_id" in result
        assert "content" in result
        assert "relevance_score" in result
        assert "cxd_function" in result
    
    def test_summary_format(self, formatter, sample_search_results):
        """Test summary result formatting"""
        response = formatter.format_search_results(sample_search_results, "summary")
        
        assert response.success is True
        assert "summary" in response.data
        assert "query" in response.data
        assert "top_result" in response.data
        
        summary = response.data["summary"]
        assert "total_results" in summary
        assert "avg_relevance_score" in summary
        assert "cxd_function_distribution" in summary
    
    def test_cxd_analysis_generation(self, formatter, sample_search_results):
        """Test CXD analysis generation"""
        response = formatter.format_cxd_search_results(sample_search_results)
        
        assert response.success is True
        assert "cxd_analysis" in response.data
        
        analysis = response.data["cxd_analysis"]
        assert "total_classified" in analysis
        assert "classification_coverage" in analysis
        assert "function_distribution" in analysis
        assert "insights" in analysis
    
    def test_error_formatting(self, formatter):
        """Test error response formatting"""
        test_error = MCPError("Test error", error_code="TEST_ERROR")
        response = formatter.format_error_response(test_error, "req_123")
        
        assert response.success is False
        assert response.error["error_code"] == "TEST_ERROR"
        assert response.error["message"] == "Test error"


class TestHandlerFactory:
    """Test handler factory functions"""
    
    def test_create_memory_recall_handler(self):
        """Test memory recall handler factory"""
        mock_engine = MockSearchEngine()
        
        # Test standard handler
        standard_handler = create_memory_recall_handler(mock_engine, "standard")
        assert isinstance(standard_handler, MemoryRecallMCPHandler)
        assert standard_handler.handler_type == "memory_recall"
        
        # Test CXD handler
        cxd_handler = create_memory_recall_handler(mock_engine, "cxd")
        assert isinstance(cxd_handler, MemoryRecallCXDHandler)
        assert cxd_handler.handler_type == "memory_recall_cxd"
        
        # Test invalid handler type
        with pytest.raises(ValueError):
            create_memory_recall_handler(mock_engine, "invalid_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])