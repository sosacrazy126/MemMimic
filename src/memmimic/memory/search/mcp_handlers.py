"""
MCP (Model Context Protocol) handlers for Memory Search System.

Provides clean protocol handlers separated from business logic for handling
memory recall requests through the MCP interface.
"""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime

from .interfaces import (
    SearchQuery, SearchResults, SearchEngine, SearchType, SimilarityMetric,
    SearchError, SearchEngineError
)
from .response_formatter import MCPResponseFormatter
from .mcp_base import MCPBaseHandler, MCPRequest, MCPResponse, MCPError

logger = logging.getLogger(__name__)


class MemoryRecallMCPHandler(MCPBaseHandler):
    """
    MCP handler for memory recall operations.
    
    Handles memory search requests through the MCP protocol with proper
    error handling, validation, and response formatting.
    """
    
    def __init__(self, search_engine: SearchEngine, formatter: Optional[MCPResponseFormatter] = None):
        """
        Initialize memory recall MCP handler.
        
        Args:
            search_engine: Search engine instance for processing queries
            formatter: Response formatter (creates default if None)
        """
        super().__init__(handler_type="memory_recall")
        self.search_engine = search_engine
        self.formatter = formatter or MCPResponseFormatter()
        
        # Handler-specific metrics
        self._handler_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time_ms': 0.0,
            'search_type_usage': {},
            'error_types': {},
        }
        
        logger.info("Memory Recall MCP Handler initialized")
    
    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle MCP memory recall request.
        
        Args:
            request: MCP request object
            
        Returns:
            MCP response with search results or error information
        """
        start_time = time.time()
        self._handler_metrics['total_requests'] += 1
        
        try:
            # Validate request
            self._validate_recall_request(request)
            
            # Extract search parameters from request
            search_query = self._extract_search_query(request)
            
            # Execute search
            search_results = self.search_engine.search(search_query)
            
            # Format response
            response = self.formatter.format_search_results(
                search_results, 
                request.get_parameter('format', 'detailed')
            )
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_success_metrics(processing_time, search_query.search_type)
            
            logger.debug(f"Memory recall completed: {len(search_results.results)} results in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_type = type(e).__name__
            self._update_error_metrics(processing_time, error_type)
            
            logger.error(f"Memory recall failed: {e}")
            
            # Format error response
            return self.formatter.format_error_response(
                error=e,
                request_id=request.request_id,
                context={'processing_time_ms': processing_time}
            )
    
    def _validate_recall_request(self, request: MCPRequest) -> None:
        """Validate memory recall request parameters."""
        required_params = ['query']
        
        for param in required_params:
            if not request.has_parameter(param):
                raise MCPError(
                    f"Missing required parameter: {param}",
                    error_code="MISSING_PARAMETER",
                    request_id=request.request_id
                )
        
        # Validate query parameter
        query_text = request.get_parameter('query')
        if not query_text or not query_text.strip():
            raise MCPError(
                "Query parameter cannot be empty",
                error_code="EMPTY_QUERY",
                request_id=request.request_id
            )
        
        # Validate optional parameters
        limit = request.get_parameter('limit', 10)
        if not isinstance(limit, int) or limit <= 0 or limit > 100:
            raise MCPError(
                "Limit must be a positive integer between 1 and 100",
                error_code="INVALID_LIMIT",
                request_id=request.request_id
            )
        
        min_confidence = request.get_parameter('min_confidence', 0.0)
        if not isinstance(min_confidence, (int, float)) or not (0.0 <= min_confidence <= 1.0):
            raise MCPError(
                "min_confidence must be a number between 0.0 and 1.0",
                error_code="INVALID_CONFIDENCE",
                request_id=request.request_id
            )
    
    def _extract_search_query(self, request: MCPRequest) -> SearchQuery:
        """Extract SearchQuery from MCP request."""
        # Required parameters
        query_text = request.get_parameter('query')
        
        # Optional parameters with defaults
        limit = request.get_parameter('limit', 10)
        min_confidence = request.get_parameter('min_confidence', 0.0)
        include_metadata = request.get_parameter('include_metadata', True)
        
        # Parse search type
        search_type_str = request.get_parameter('search_type', 'hybrid')
        try:
            search_type = SearchType(search_type_str.lower())
        except ValueError:
            search_type = SearchType.HYBRID
            logger.warning(f"Invalid search_type '{search_type_str}', using hybrid")
        
        # Parse filters
        filters = request.get_parameter('filters', {})
        if not isinstance(filters, dict):
            filters = {}
        
        # Create SearchQuery
        return SearchQuery(
            text=query_text.strip(),
            limit=limit,
            min_confidence=min_confidence,
            include_metadata=include_metadata,
            search_type=search_type,
            filters=filters
        )
    
    def _update_success_metrics(self, processing_time_ms: float, search_type: SearchType):
        """Update metrics for successful request."""
        self._handler_metrics['successful_requests'] += 1
        
        # Update average processing time
        total_requests = self._handler_metrics['total_requests']
        current_avg = self._handler_metrics['avg_processing_time_ms']
        self._handler_metrics['avg_processing_time_ms'] = (
            (current_avg * (total_requests - 1) + processing_time_ms) / total_requests
        )
        
        # Track search type usage
        search_type_key = search_type.value
        self._handler_metrics['search_type_usage'][search_type_key] = (
            self._handler_metrics['search_type_usage'].get(search_type_key, 0) + 1
        )
    
    def _update_error_metrics(self, processing_time_ms: float, error_type: str):
        """Update metrics for failed request."""
        self._handler_metrics['failed_requests'] += 1
        
        # Track error types
        self._handler_metrics['error_types'][error_type] = (
            self._handler_metrics['error_types'].get(error_type, 0) + 1
        )
    
    def get_handler_metrics(self) -> Dict[str, Any]:
        """Get handler-specific metrics."""
        total_requests = self._handler_metrics['total_requests']
        success_rate = (
            self._handler_metrics['successful_requests'] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        return {
            'handler_type': self.handler_type,
            'total_requests': total_requests,
            'successful_requests': self._handler_metrics['successful_requests'],
            'failed_requests': self._handler_metrics['failed_requests'],
            'success_rate': success_rate,
            'avg_processing_time_ms': self._handler_metrics['avg_processing_time_ms'],
            'search_type_usage': self._handler_metrics['search_type_usage'],
            'error_types': self._handler_metrics['error_types'],
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check handler health status."""
        try:
            # Test search engine health
            engine_healthy = self.search_engine.health_check()
            
            # Check error rate
            total_requests = self._handler_metrics['total_requests']
            error_rate = (
                self._handler_metrics['failed_requests'] / total_requests 
                if total_requests > 0 else 0.0
            )
            
            # Determine overall health
            healthy = engine_healthy and error_rate < 0.1  # Less than 10% error rate
            
            return {
                'healthy': healthy,
                'search_engine_healthy': engine_healthy,
                'error_rate': error_rate,
                'total_requests_processed': total_requests,
                'last_check': datetime.now().isoformat(),
                'status': 'healthy' if healthy else 'degraded'
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'last_check': datetime.now().isoformat(),
                'status': 'unhealthy'
            }


class MemoryRecallCXDHandler(MCPBaseHandler):
    """
    MCP handler for CXD-enhanced memory recall operations.
    
    Specialized handler that provides CXD function filtering and enhanced
    search capabilities for Control/Context/Data classification.
    """
    
    def __init__(self, search_engine: SearchEngine, formatter: Optional[MCPResponseFormatter] = None):
        """
        Initialize CXD memory recall MCP handler.
        
        Args:
            search_engine: Search engine instance with CXD capabilities
            formatter: Response formatter (creates default if None)
        """
        super().__init__(handler_type="memory_recall_cxd")
        self.search_engine = search_engine
        self.formatter = formatter or MCPResponseFormatter()
        
        # CXD-specific metrics
        self._cxd_metrics = {
            'function_filter_usage': {},
            'cxd_enhanced_requests': 0,
            'fallback_to_regular_search': 0,
        }
        
        logger.info("Memory Recall CXD MCP Handler initialized")
    
    def handle_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle CXD-enhanced memory recall request.
        
        Args:
            request: MCP request with CXD parameters
            
        Returns:
            MCP response with CXD-enhanced search results
        """
        start_time = time.time()
        
        try:
            # Validate CXD-specific request
            self._validate_cxd_request(request)
            
            # Extract search query with CXD parameters
            search_query = self._extract_cxd_search_query(request)
            
            # Execute CXD-enhanced search
            search_results = self.search_engine.search(search_query)
            
            # Apply CXD function filtering if requested
            function_filter = request.get_parameter('function_filter')
            if function_filter:
                search_results = self._filter_by_cxd_function(search_results, function_filter)
                self._cxd_metrics['function_filter_usage'][function_filter] = (
                    self._cxd_metrics['function_filter_usage'].get(function_filter, 0) + 1
                )
            
            # Format CXD-enhanced response
            response = self.formatter.format_cxd_search_results(
                search_results,
                include_cxd_analysis=request.get_parameter('include_cxd_analysis', True)
            )
            
            self._cxd_metrics['cxd_enhanced_requests'] += 1
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"CXD memory recall completed in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"CXD memory recall failed: {e}")
            
            return self.formatter.format_error_response(
                error=e,
                request_id=request.request_id,
                context={'processing_time_ms': processing_time}
            )
    
    def _validate_cxd_request(self, request: MCPRequest) -> None:
        """Validate CXD-specific request parameters."""
        # Validate function filter if provided
        function_filter = request.get_parameter('function_filter')
        if function_filter:
            valid_functions = ['Control', 'Context', 'Data', 'ALL']
            if function_filter not in valid_functions:
                raise MCPError(
                    f"Invalid function_filter. Must be one of: {valid_functions}",
                    error_code="INVALID_FUNCTION_FILTER",
                    request_id=request.request_id
                )
    
    def _extract_cxd_search_query(self, request: MCPRequest) -> SearchQuery:
        """Extract SearchQuery with CXD-specific parameters."""
        # Start with basic search query
        base_handler = MemoryRecallMCPHandler(self.search_engine, self.formatter)
        search_query = base_handler._extract_search_query(request)
        
        # Add CXD-specific filters
        cxd_filters = {}
        function_filter = request.get_parameter('function_filter')
        if function_filter and function_filter != 'ALL':
            cxd_filters['cxd_function'] = function_filter
        
        # Merge with existing filters
        if cxd_filters:
            merged_filters = {**search_query.filters, **cxd_filters}
            return SearchQuery(
                text=search_query.text,
                limit=search_query.limit,
                min_confidence=search_query.min_confidence,
                include_metadata=search_query.include_metadata,
                search_type=search_query.search_type,
                filters=merged_filters
            )
        
        return search_query
    
    def _filter_by_cxd_function(self, results: SearchResults, function_filter: str) -> SearchResults:
        """Filter search results by CXD function."""
        if function_filter == 'ALL':
            return results
        
        filtered_results = []
        for result in results.results:
            if (result.cxd_classification and 
                result.cxd_classification.function == function_filter):
                filtered_results.append(result)
        
        # Create new SearchResults with filtered results
        return SearchResults(
            results=filtered_results,
            total_found=len(filtered_results),
            query=results.query,
            search_context=results.search_context
        )
    
    def get_cxd_metrics(self) -> Dict[str, Any]:
        """Get CXD-specific metrics."""
        return {
            'cxd_enhanced_requests': self._cxd_metrics['cxd_enhanced_requests'],
            'function_filter_usage': self._cxd_metrics['function_filter_usage'],
            'fallback_to_regular_search': self._cxd_metrics['fallback_to_regular_search'],
        }


def create_memory_recall_handler(search_engine: SearchEngine, 
                               handler_type: str = "standard") -> MCPBaseHandler:
    """
    Factory function to create memory recall MCP handlers.
    
    Args:
        search_engine: Search engine instance
        handler_type: Type of handler ("standard" or "cxd")
        
    Returns:
        MCPBaseHandler instance
    """
    if handler_type == "standard":
        return MemoryRecallMCPHandler(search_engine)
    elif handler_type == "cxd":
        return MemoryRecallCXDHandler(search_engine)
    else:
        raise ValueError(f"Unknown handler type: {handler_type}")