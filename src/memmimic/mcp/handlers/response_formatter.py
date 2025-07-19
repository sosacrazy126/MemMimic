"""
MCP response formatting utilities for memory search results.

Provides consistent formatting for search results, error responses,
and metadata in MCP protocol responses.
"""

import logging
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..memory.search.interfaces import SearchResults, SearchResult, CXDClassification
from .mcp_base import MCPResponse

logger = logging.getLogger(__name__)


class MCPResponseFormatter:
    """
    Formats search results and responses for MCP protocol transmission.
    
    Handles the conversion between internal search result objects and
    MCP-compatible response formats with proper encoding and metadata.
    """
    
    def __init__(self, include_debug_info: bool = False):
        """
        Initialize response formatter.
        
        Args:
            include_debug_info: Whether to include debug information in responses
        """
        self.include_debug_info = include_debug_info
        
        # Response format configuration
        self.config = {
            'max_content_length': 2000,  # Truncate long content
            'include_search_context': True,
            'include_cxd_metadata': True,
            'unicode_safe': True,  # Ensure safe Unicode handling
        }
        
        logger.debug("MCPResponseFormatter initialized")
    
    def format_search_results(self, search_results: SearchResults) -> MCPResponse:
        """
        Format SearchResults for MCP response.
        
        Args:
            search_results: SearchResults object to format
            
        Returns:
            MCPResponse with formatted search results
        """
        try:
            # Format individual results
            formatted_results = []
            for result in search_results.results:
                formatted_result = self._format_single_result(result)
                formatted_results.append(formatted_result)
            
            # Create response data
            response_data = {
                'results': formatted_results,
                'total_found': search_results.total_found,
                'query': {
                    'text': search_results.query.text,
                    'limit': search_results.query.limit,
                    'filters': search_results.query.filters,
                    'search_type': search_results.query.search_type.value,
                },
                'search_metadata': self._format_search_context(search_results.search_context),
                'timestamp': search_results.timestamp.isoformat(),
            }
            
            # Add debug information if enabled
            if self.include_debug_info:
                response_data['debug'] = {
                    'formatter_version': '1.0',
                    'response_size_bytes': len(str(response_data)),
                    'result_count': len(formatted_results),
                }
            
            return MCPResponse(
                success=True,
                data=response_data,
                metadata={
                    'response_type': 'search_results',
                    'result_count': len(formatted_results),
                    'search_time_ms': search_results.search_context.search_time_ms,
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to format search results: {e}")
            return MCPResponse(
                success=False,
                error=f"Failed to format search results: {str(e)}",
                error_code="RESPONSE_FORMAT_ERROR"
            )
    
    def format_error_response(self, error_code: str, error_message: str,
                             context: Optional[Dict[str, Any]] = None) -> MCPResponse:
        """
        Format error response for MCP transmission.
        
        Args:
            error_code: Machine-readable error code
            error_message: Human-readable error message
            context: Optional error context information
            
        Returns:
            MCPResponse with error information
        """
        return MCPResponse(
            success=False,
            error=error_message,
            error_code=error_code,
            metadata={
                'response_type': 'error',
                'error_context': context or {},
                'timestamp': datetime.now().isoformat(),
            }
        )
    
    def format_status_response(self, status_data: Dict[str, Any]) -> MCPResponse:
        """
        Format status/health check response.
        
        Args:
            status_data: Status information to format
            
        Returns:
            MCPResponse with status information
        """
        return MCPResponse(
            success=True,
            data=status_data,
            metadata={
                'response_type': 'status',
                'timestamp': datetime.now().isoformat(),
            }
        )
    
    def _format_single_result(self, result: SearchResult) -> Dict[str, Any]:
        """Format a single search result for MCP response."""
        # Safe content truncation
        content = self._safe_decode_text(result.content)
        if len(content) > self.config['max_content_length']:
            content = content[:self.config['max_content_length']] + "..."
        
        formatted_result = {
            'memory_id': str(result.memory_id),
            'content': content,
            'relevance_score': round(result.relevance_score, 3),
            'metadata': result.metadata or {},
        }
        
        # Add CXD classification if available
        if result.cxd_classification and self.config['include_cxd_metadata']:
            formatted_result['cxd_classification'] = self._format_cxd_classification(
                result.cxd_classification
            )
        
        # Add search context if available
        if result.search_context and self.config['include_search_context']:
            formatted_result['search_context'] = self._format_search_context(
                result.search_context
            )
        
        return formatted_result
    
    def _format_cxd_classification(self, classification: CXDClassification) -> Dict[str, Any]:
        """Format CXD classification for response."""
        return {
            'function': classification.function,
            'confidence': round(classification.confidence, 3),
            'metadata': classification.metadata or {},
        }
    
    def _format_search_context(self, context) -> Dict[str, Any]:
        """Format search context for response."""
        if not context:
            return {}
        
        return {
            'search_time_ms': round(context.search_time_ms, 2),
            'total_candidates': context.total_candidates,
            'cache_used': context.cache_used,
            'similarity_metric': context.similarity_metric.value if hasattr(context.similarity_metric, 'value') else str(context.similarity_metric),
            'cxd_classification_used': context.cxd_classification_used,
        }
    
    def _safe_decode_text(self, text: Any) -> str:
        """
        Safely decode text content with proper Unicode handling.
        
        Extracted from the original massive file's safe_decode_text function.
        """
        if text is None:
            return ""
        
        if isinstance(text, str):
            return text
        
        if isinstance(text, bytes):
            # Try UTF-8 first, fallback to other encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            for encoding in encodings:
                try:
                    decoded = text.decode(encoding)
                    # Ensure compatibility with terminal output
                    if self.config['unicode_safe'] and sys.platform.startswith("win"):
                        # On Windows, replace problematic Unicode characters
                        decoded = decoded.encode('ascii', errors='replace').decode('ascii')
                    return decoded
                except (UnicodeDecodeError, UnicodeEncodeError):
                    continue
            
            # Last resort: decode with replacement characters
            return text.decode('utf-8', errors='replace')
        
        # Convert other types to string
        try:
            return str(text)
        except Exception:
            return "<unprintable content>"


def create_legacy_format_response(search_results: SearchResults) -> str:
    """
    Create response in the legacy format for backward compatibility.
    
    This maintains compatibility with the original massive file's output format
    while using the new modular architecture.
    
    Args:
        search_results: SearchResults to format
        
    Returns:
        Formatted string in legacy format
    """
    try:
        formatter = MCPResponseFormatter()
        response_parts = []
        
        # Header
        result_count = len(search_results.results)
        response_parts.append(f"🧠 MEMMIMIC TRUE HYBRID WORDNET SEARCH v3.0 ({result_count} results)")
        
        # Search method summary
        semantic_count = sum(1 for r in search_results.results if r.relevance_score > 0)
        wordnet_count = 0  # Would need to track this in search context
        response_parts.append(f"🎯 Methods: {semantic_count} semantic + {wordnet_count} WordNet")
        response_parts.append("")
        
        # Legend
        response_parts.append("🔤 LEGEND: 🎛️CONTROL=Search/filter 🔗CONTEXT=Relations 📊DATA=Processing")
        response_parts.append("🔍 METHODS: 🧠=Semantic 🔍=WordNet 🎯🧠=Hybrid-Convergence 🎯⭐=Convergence+Original ⭐=Original-Terms")
        response_parts.append("")
        
        # Results
        for i, result in enumerate(search_results.results, 1):
            content = formatter._safe_decode_text(result.content)
            
            # Memory confidence and scores
            memory_confidence = result.metadata.get('confidence', 0.8)
            relevance_score = result.relevance_score
            
            # Format confidence bars
            memory_bar = "█" * int(memory_confidence * 10) + "░" * (10 - int(memory_confidence * 10))
            combined_bar = "█" * int(relevance_score * 10) + "░" * (10 - int(relevance_score * 10))
            
            # CXD classification
            cxd_symbol = "❓"
            cxd_confidence = 0.5
            if result.cxd_classification:
                cxd_map = {"Control": "🎛️", "Context": "🔗", "Data": "📊"}
                cxd_symbol = cxd_map.get(result.cxd_classification.function, "❓")
                cxd_confidence = result.cxd_classification.confidence
            
            # Memory type
            memory_type = result.metadata.get('type', 'UNKNOWN').upper()
            
            response_parts.append(f"{i}. 🧠 {cxd_symbol}[?] [{memory_type}] {content}")
            response_parts.append(f"   Memory: [{memory_bar}] {memory_confidence:.2f} | Combined: [{combined_bar}] {relevance_score:.3f}")
            response_parts.append(f"   Semantic: [{combined_bar}] {relevance_score:.3f} | WordNet: [░░░░░░░░░░] 0.000")
            response_parts.append(f"   Method: semantic_dominant")
            response_parts.append(f"   CXD: [{'█' * int(cxd_confidence * 10)}{'░' * (10 - int(cxd_confidence * 10))}] {cxd_confidence:.3f}")
            
            if 'created_at' in result.metadata:
                response_parts.append(f"   Created: {result.metadata['created_at']}")
            
            response_parts.append("")
        
        # Function distribution (simplified)
        response_parts.append("📊 FUNCTION DISTRIBUTION:")
        cxd_counts = {}
        for result in search_results.results:
            if result.cxd_classification:
                func = result.cxd_classification.function
                short_func = func[0] if func else "?"
                cxd_counts[short_func] = cxd_counts.get(short_func, 0) + 1
        
        for func, count in cxd_counts.items():
            response_parts.append(f"   {func}: {count}")
        
        response_parts.append("")
        response_parts.append("🧠 Powered by MemMimic v3.0 | True Hybrid Search = Semantic + Lexical + NLTK WordNet")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Failed to create legacy format response: {e}")
        return f"Error formatting response: {str(e)}"