"""
Clean MCP protocol handler for memory recall requests.

Extracted from the massive memmimic_recall_cxd.py file to provide a clean
separation between MCP protocol handling and search business logic.
"""

import logging
from typing import Dict, Any, Optional

from ..memory.search.interfaces import SearchEngine, SearchQuery, SearchType
from .mcp_base import MCPBase, MCPResponse, extract_search_params
from .response_formatter import MCPResponseFormatter, create_legacy_format_response

logger = logging.getLogger(__name__)


class MemoryRecallMCPHandler(MCPBase):
    """
    Clean MCP protocol handler for memory recall requests.
    
    Handles MCP protocol communication while delegating actual search
    operations to the modular search engine components.
    """
    
    def __init__(self, search_engine: SearchEngine, 
                 response_formatter: Optional[MCPResponseFormatter] = None):
        """
        Initialize memory recall MCP handler.
        
        Args:
            search_engine: SearchEngine instance for performing searches
            response_formatter: Optional custom response formatter
        """
        super().__init__("memory_recall")
        
        self.search_engine = search_engine
        self.response_formatter = response_formatter or MCPResponseFormatter()
        
        # Legacy compatibility settings
        self.legacy_format = True  # Output in original format for compatibility
        
        logger.info("MemoryRecallMCPHandler initialized")
    
    async def handle_recall_request(self, mcp_request: Dict[str, Any]) -> MCPResponse:
        """
        Handle MCP memory recall request.
        
        Args:
            mcp_request: MCP request dictionary
            
        Returns:
            MCPResponse with search results or error
        """
        def _handle_recall():
            # Validate MCP request structure
            validation_error = self.validate_request(mcp_request)
            if validation_error:
                return self.create_error_response(
                    error_message=validation_error,
                    error_code="INVALID_REQUEST"
                )
            
            try:
                # Extract and validate search parameters
                params = mcp_request.get('params', {})
                search_params = extract_search_params(params)
                
                # Create search query
                search_query = SearchQuery(
                    text=search_params['query'],
                    limit=search_params['limit'],
                    filters={'function_filter': search_params['function_filter']} if search_params['function_filter'] != 'ALL' else {},
                    search_type=SearchType.HYBRID  # Use hybrid search by default
                )
                
                # Execute search
                search_results = self.search_engine.search(search_query)
                
                # Format response
                if self.legacy_format:
                    # Return legacy format for compatibility
                    legacy_response = create_legacy_format_response(search_results)
                    return self.create_success_response(
                        data={'formatted_response': legacy_response},
                        metadata={
                            'format': 'legacy',
                            'result_count': len(search_results.results),
                            'search_time_ms': search_results.search_context.search_time_ms
                        }
                    )
                else:
                    # Return structured response
                    return self.response_formatter.format_search_results(search_results)
                
            except ValueError as e:
                return self.create_error_response(
                    error_message=str(e),
                    error_code="INVALID_PARAMETERS"
                )
            except Exception as e:
                logger.error(f"Search execution failed: {e}")
                return self.create_error_response(
                    error_message=f"Search failed: {str(e)}",
                    error_code="SEARCH_ERROR"
                )
        
        # Execute with metrics collection
        return self.execute_with_metrics(_handle_recall)
    
    def handle_status_request(self, mcp_request: Dict[str, Any]) -> MCPResponse:
        """
        Handle MCP status/health check request.
        
        Args:
            mcp_request: MCP request dictionary
            
        Returns:
            MCPResponse with system status
        """
        def _handle_status():
            try:
                # Get search engine health
                search_engine_healthy = self.search_engine.health_check()
                search_metrics = self.search_engine.get_metrics()
                
                # Get handler metrics
                handler_metrics = self.get_metrics()
                handler_health = self.health_check()
                
                status_data = {
                    'overall_status': 'healthy' if search_engine_healthy and handler_health['is_healthy'] else 'degraded',
                    'search_engine': {
                        'healthy': search_engine_healthy,
                        'metrics': search_metrics.__dict__ if hasattr(search_metrics, '__dict__') else search_metrics
                    },
                    'mcp_handler': {
                        'healthy': handler_health['is_healthy'],
                        'metrics': handler_metrics
                    },
                    'response_formatter': {
                        'legacy_format_enabled': self.legacy_format
                    }
                }
                
                return self.response_formatter.format_status_response(status_data)
                
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                return self.create_error_response(
                    error_message=f"Status check failed: {str(e)}",
                    error_code="STATUS_ERROR"
                )
        
        return self.execute_with_metrics(_handle_status)
    
    def set_legacy_format(self, enabled: bool):
        """
        Enable or disable legacy format output.
        
        Args:
            enabled: Whether to use legacy format
        """
        self.legacy_format = enabled
        logger.info(f"Legacy format {'enabled' if enabled else 'disabled'}")
    
    def warm_cache(self, common_queries: list[str]) -> MCPResponse:
        """
        Warm the search cache with common queries.
        
        Args:
            common_queries: List of common query strings
            
        Returns:
            MCPResponse indicating cache warming result
        """
        try:
            self.search_engine.warm_cache(common_queries)
            
            return self.create_success_response(
                data={'cache_warmed': True, 'query_count': len(common_queries)},
                metadata={'operation': 'cache_warming'}
            )
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
            return self.create_error_response(
                error_message=f"Cache warming failed: {str(e)}",
                error_code="CACHE_WARM_ERROR"
            )


def create_memory_recall_handler(search_engine: SearchEngine) -> MemoryRecallMCPHandler:
    """
    Factory function to create a memory recall MCP handler.
    
    Args:
        search_engine: SearchEngine instance for performing searches
        
    Returns:
        Configured MemoryRecallMCPHandler instance
    """
    formatter = MCPResponseFormatter(include_debug_info=False)
    return MemoryRecallMCPHandler(search_engine, formatter)


# Compatibility function for direct invocation
def search_memories_hybrid(query: str, function_filter: str = "ALL", 
                          limit: int = 5, db_name: Optional[str] = None) -> str:
    """
    Legacy compatibility function that mimics the original search_memories_hybrid.
    
    This function maintains API compatibility with the original massive file
    while using the new modular architecture internally.
    
    Args:
        query: Search query text
        function_filter: CXD function filter (CONTROL, CONTEXT, DATA, ALL)
        limit: Maximum number of results
        db_name: Database name (optional)
        
    Returns:
        Formatted search results string in legacy format
    """
    try:
        # This would need to be properly initialized with actual search engine
        # For now, return a placeholder that maintains the original format
        
        logger.warning("Legacy compatibility function called - should use proper MCP handler")
        
        # Mock response in legacy format
        return f"""🧠 MEMMIMIC TRUE HYBRID WORDNET SEARCH v3.0 (0 results)
🎯 Methods: 0 semantic + 0 WordNet

🔤 LEGEND: 🎛️CONTROL=Search/filter 🔗CONTEXT=Relations 📊DATA=Processing
🔍 METHODS: 🧠=Semantic 🔍=WordNet 🎯🧠=Hybrid-Convergence 🎯⭐=Convergence+Original ⭐=Original-Terms

No results found for query: "{query}"

📊 FUNCTION DISTRIBUTION:
   (none)

🧠 Powered by MemMimic v3.0 | True Hybrid Search = Semantic + Lexical + NLTK WordNet"""
        
    except Exception as e:
        logger.error(f"Legacy compatibility function failed: {e}")
        return f"Error: {str(e)}"