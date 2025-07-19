"""
Response formatting utilities for MCP (Model Context Protocol) handlers.

Provides standardized formatting of search results and error responses
for consistent MCP protocol communication.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
from datetime import datetime

from .interfaces import (
    SearchResults, SearchResult, SearchQuery, SearchContext, CXDClassification
)
from .mcp_base import MCPResponse, MCPError, create_success_response, create_error_response

logger = logging.getLogger(__name__)


class MCPResponseFormatter:
    """
    Formatter for MCP protocol responses.
    
    Provides standardized formatting of search results, errors, and metadata
    for consistent communication through the MCP protocol.
    """
    
    def __init__(self, include_debug_info: bool = False):
        """
        Initialize response formatter.
        
        Args:
            include_debug_info: Whether to include debug information in responses
        """
        self.include_debug_info = include_debug_info
        
        logger.info("MCP Response Formatter initialized")
    
    def format_search_results(self, search_results: SearchResults, 
                            format_type: str = "detailed") -> MCPResponse:
        """
        Format search results for MCP response.
        
        Args:
            search_results: SearchResults to format
            format_type: Format type ("detailed", "compact", "summary")
            
        Returns:
            MCPResponse with formatted search results
        """
        try:
            if format_type == "detailed":
                data = self._format_detailed_results(search_results)
            elif format_type == "compact":
                data = self._format_compact_results(search_results)
            elif format_type == "summary":
                data = self._format_summary_results(search_results)
            else:
                raise ValueError(f"Unknown format type: {format_type}")
            
            return create_success_response(
                request_id="",  # Will be set by handler
                data=data
            )
            
        except Exception as e:
            logger.error(f"Failed to format search results: {e}")
            raise MCPError(
                f"Response formatting failed: {str(e)}",
                error_code="RESPONSE_FORMAT_ERROR"
            )
    
    def format_cxd_search_results(self, search_results: SearchResults,
                                include_cxd_analysis: bool = True) -> MCPResponse:
        """
        Format CXD-enhanced search results for MCP response.
        
        Args:
            search_results: SearchResults with CXD classifications
            include_cxd_analysis: Whether to include CXD analysis
            
        Returns:
            MCPResponse with CXD-enhanced formatting
        """
        try:
            # Start with detailed format
            data = self._format_detailed_results(search_results)
            
            # Add CXD-specific analysis
            if include_cxd_analysis:
                data['cxd_analysis'] = self._generate_cxd_analysis(search_results)
            
            # Add CXD metadata to results
            for i, result in enumerate(search_results.results):
                if result.cxd_classification:
                    data['results'][i]['cxd_classification'] = self._format_cxd_classification(
                        result.cxd_classification
                    )
            
            return create_success_response(
                request_id="",  # Will be set by handler
                data=data
            )
            
        except Exception as e:
            logger.error(f"Failed to format CXD search results: {e}")
            raise MCPError(
                f"CXD response formatting failed: {str(e)}",
                error_code="CXD_FORMAT_ERROR"
            )
    
    def format_error_response(self, error: Union[Exception, Dict[str, Any]], 
                            request_id: str = "",
                            context: Optional[Dict[str, Any]] = None) -> MCPResponse:
        """
        Format error response for MCP protocol.
        
        Args:
            error: Error to format (exception or dict)
            request_id: Request identifier
            context: Additional error context
            
        Returns:
            MCPResponse with formatted error
        """
        try:
            # Prepare error context
            error_context = context or {}
            
            # Add debug information if enabled
            if self.include_debug_info:
                error_context.update({
                    'debug_timestamp': datetime.now().isoformat(),
                    'formatter_version': '1.0.0'
                })
            
            # Format error based on type
            if isinstance(error, MCPError):
                error_dict = error.to_dict()
                error_dict['context'].update(error_context)
            elif isinstance(error, Exception):
                error_dict = {
                    'error_code': 'INTERNAL_ERROR',
                    'message': str(error),
                    'error_type': type(error).__name__,
                    'timestamp': datetime.now().isoformat(),
                    'context': error_context
                }
            else:
                error_dict = {**error, 'context': {**error.get('context', {}), **error_context}}
            
            return create_error_response(
                request_id=request_id,
                error=error_dict
            )
            
        except Exception as e:
            logger.error(f"Failed to format error response: {e}")
            # Fallback error response
            return MCPResponse(
                request_id=request_id,
                success=False,
                error={
                    'error_code': 'FORMATTER_ERROR',
                    'message': f"Error formatting failed: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                }
            )
    
    def _format_detailed_results(self, search_results: SearchResults) -> Dict[str, Any]:
        """Format detailed search results."""
        return {
            'results': [self._format_search_result(result) for result in search_results.results],
            'metadata': {
                'total_found': search_results.total_found,
                'query': self._format_search_query(search_results.query),
                'search_context': self._format_search_context(search_results.search_context),
                'result_count': len(search_results.results),
                'has_more_results': search_results.total_found > len(search_results.results)
            }
        }
    
    def _format_compact_results(self, search_results: SearchResults) -> Dict[str, Any]:
        """Format compact search results with essential information only."""
        return {
            'results': [
                {
                    'memory_id': result.memory_id,
                    'content': result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    'relevance_score': round(result.relevance_score, 3),
                    'cxd_function': (
                        result.cxd_classification.function 
                        if result.cxd_classification else None
                    )
                }
                for result in search_results.results
            ],
            'metadata': {
                'total_found': search_results.total_found,
                'query_text': search_results.query.text,
                'search_time_ms': search_results.search_context.search_time_ms
            }
        }
    
    def _format_summary_results(self, search_results: SearchResults) -> Dict[str, Any]:
        """Format summary of search results."""
        # Calculate summary statistics
        avg_relevance = (
            sum(r.relevance_score for r in search_results.results) / len(search_results.results)
            if search_results.results else 0.0
        )
        
        # CXD function distribution
        cxd_distribution = {}
        for result in search_results.results:
            if result.cxd_classification:
                function = result.cxd_classification.function
                cxd_distribution[function] = cxd_distribution.get(function, 0) + 1
        
        return {
            'summary': {
                'total_results': len(search_results.results),
                'total_found': search_results.total_found,
                'avg_relevance_score': round(avg_relevance, 3),
                'search_time_ms': search_results.search_context.search_time_ms,
                'cxd_function_distribution': cxd_distribution
            },
            'query': search_results.query.text,
            'top_result': (
                self._format_search_result(search_results.results[0])
                if search_results.results else None
            )
        }
    
    def _format_search_result(self, result: SearchResult) -> Dict[str, Any]:
        """Format individual search result."""
        formatted_result = {
            'memory_id': result.memory_id,
            'content': result.content,
            'relevance_score': round(result.relevance_score, 4),
            'metadata': result.metadata or {}
        }
        
        # Add CXD classification if present
        if result.cxd_classification:
            formatted_result['cxd_classification'] = self._format_cxd_classification(
                result.cxd_classification
            )
        
        # Add search context if present
        if result.search_context:
            formatted_result['search_context'] = result.search_context
        
        return formatted_result
    
    def _format_search_query(self, query: SearchQuery) -> Dict[str, Any]:
        """Format search query information."""
        return {
            'text': query.text,
            'limit': query.limit,
            'min_confidence': query.min_confidence,
            'search_type': query.search_type.value,
            'include_metadata': query.include_metadata,
            'filters': query.filters or {}
        }
    
    def _format_search_context(self, context: SearchContext) -> Dict[str, Any]:
        """Format search context information."""
        return {
            'search_time_ms': round(context.search_time_ms, 2),
            'total_candidates': context.total_candidates,
            'cache_used': context.cache_used,
            'similarity_metric': context.similarity_metric.value,
            'cxd_classification_used': context.cxd_classification_used
        }
    
    def _format_cxd_classification(self, classification: CXDClassification) -> Dict[str, Any]:
        """Format CXD classification information."""
        return {
            'function': classification.function,
            'confidence': round(classification.confidence, 4),
            'metadata': classification.metadata or {}
        }
    
    def _generate_cxd_analysis(self, search_results: SearchResults) -> Dict[str, Any]:
        """Generate CXD analysis of search results."""
        try:
            total_results = len(search_results.results)
            if total_results == 0:
                return {
                    'total_classified': 0,
                    'classification_coverage': 0.0,
                    'function_distribution': {},
                    'avg_confidence_by_function': {},
                    'insights': []
                }
            
            # Count classifications by function
            function_counts = {}
            function_confidences = {}
            classified_count = 0
            
            for result in search_results.results:
                if result.cxd_classification:
                    classified_count += 1
                    function = result.cxd_classification.function
                    confidence = result.cxd_classification.confidence
                    
                    function_counts[function] = function_counts.get(function, 0) + 1
                    if function not in function_confidences:
                        function_confidences[function] = []
                    function_confidences[function].append(confidence)
            
            # Calculate average confidences
            avg_confidence_by_function = {}
            for function, confidences in function_confidences.items():
                avg_confidence_by_function[function] = sum(confidences) / len(confidences)
            
            # Generate insights
            insights = []
            
            # Dominant function insight
            if function_counts:
                dominant_function = max(function_counts, key=function_counts.get)
                dominant_percentage = (function_counts[dominant_function] / total_results) * 100
                insights.append(
                    f"{dominant_function} is the dominant function ({dominant_percentage:.1f}% of results)"
                )
            
            # Confidence insight
            overall_confidences = [c for conf_list in function_confidences.values() for c in conf_list]
            if overall_confidences:
                avg_confidence = sum(overall_confidences) / len(overall_confidences)
                if avg_confidence > 0.8:
                    insights.append("High confidence in CXD classifications")
                elif avg_confidence < 0.5:
                    insights.append("Low confidence in CXD classifications - review may be needed")
            
            # Coverage insight
            coverage = classified_count / total_results
            if coverage < 0.8:
                insights.append(f"Only {coverage*100:.1f}% of results have CXD classification")
            
            return {
                'total_classified': classified_count,
                'classification_coverage': round(coverage, 3),
                'function_distribution': function_counts,
                'avg_confidence_by_function': {
                    func: round(conf, 3) 
                    for func, conf in avg_confidence_by_function.items()
                },
                'insights': insights
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate CXD analysis: {e}")
            return {
                'error': f"Analysis generation failed: {str(e)}",
                'total_classified': 0,
                'classification_coverage': 0.0
            }


def create_mcp_response_formatter(debug_mode: bool = False) -> MCPResponseFormatter:
    """
    Factory function to create MCP response formatter.
    
    Args:
        debug_mode: Whether to include debug information in responses
        
    Returns:
        MCPResponseFormatter instance
    """
    return MCPResponseFormatter(include_debug_info=debug_mode)