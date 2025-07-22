"""
Complete integration tests for the Memory Search System.

Tests the complete system with factory functions and full component integration.
"""

import pytest
import sys
import os

# Set up path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from memmimic.memory.search import (
    create_search_system,
    create_search_engine,
    create_complete_mcp_system,
    SearchQuery,
    SearchType,
    DefaultSearchConfig,
    create_mcp_request,
)


class TestCompleteSystemIntegration:
    """Test complete system integration with factory functions"""
    
    def test_create_search_system_basic(self):
        """Test basic search system creation"""
        system = create_search_system()
        
        assert 'search_engine' in system
        assert 'config' in system
        assert 'components' in system
        
        # Verify components are properly connected
        search_engine = system['search_engine']
        assert search_engine.similarity_calculator is not None
        assert search_engine.cxd_bridge is not None
        assert search_engine.cache is not None
        assert search_engine.result_processor is not None
        assert search_engine.config is not None
        
        # Test health check
        assert search_engine.health_check() is True
    
    def test_create_search_system_with_mcp(self):
        """Test search system creation with MCP handlers"""
        system = create_search_system(enable_mcp=True)
        
        # Verify MCP components are created
        assert 'mcp_registry' in system
        assert 'mcp_handlers' in system
        
        mcp_registry = system['mcp_registry']
        mcp_handlers = system['mcp_handlers']
        
        # Verify handlers are registered
        assert 'standard' in mcp_handlers
        assert 'cxd' in mcp_handlers
        
        # Test handler retrieval
        standard_handler = mcp_registry.get_handler("memory_recall")
        cxd_handler = mcp_registry.get_handler("memory_recall_cxd")
        
        assert standard_handler is not None
        assert cxd_handler is not None
        assert standard_handler.handler_type == "memory_recall"
        assert cxd_handler.handler_type == "memory_recall_cxd"
    
    def test_create_search_system_with_custom_config(self):
        """Test search system creation with custom configuration"""
        config = DefaultSearchConfig(
            max_results=25,
            cache_ttl=1800,
            enable_cxd_classification=True
        )
        
        system = create_search_system(config=config)
        
        # Verify config is used
        assert system['config'] == config
        assert system['search_engine'].config == config
    
    def test_create_search_engine_factory(self):
        """Test search engine factory function"""
        engine = create_search_engine()
        
        # Verify all components are created
        assert engine.similarity_calculator is not None
        assert engine.cxd_bridge is not None
        assert engine.cache is not None
        assert engine.result_processor is not None
        assert engine.config is not None
        
        # Test basic functionality
        assert engine.health_check() is True
    
    def test_create_complete_mcp_system_factory(self):
        """Test complete MCP system factory function"""
        registry = create_complete_mcp_system()
        
        # Verify handlers are registered
        standard_handler = registry.get_handler("memory_recall")
        cxd_handler = registry.get_handler("memory_recall_cxd")
        
        assert standard_handler is not None
        assert cxd_handler is not None
        
        # Test registry status
        status = registry.get_registry_status()
        assert status['total_handlers'] == 2
        assert "memory_recall" in status['registered_methods']
        assert "memory_recall_cxd" in status['registered_methods']
    
    def test_mcp_request_routing_integration(self):
        """Test MCP request routing through complete system"""
        # Create system with MCP enabled
        system = create_search_system(enable_mcp=True)
        mcp_registry = system['mcp_registry']
        
        # Test standard memory recall request
        request = create_mcp_request("req_1", "memory_recall", {
            "query": "test query",
            "limit": 5
        })
        
        response = mcp_registry.route_request(request)
        
        # Verify response structure (even with empty results due to stub implementation)
        assert response.success is True
        assert response.data is not None
        assert "results" in response.data
        assert "metadata" in response.data
        
        # Test CXD-enhanced request
        cxd_request = create_mcp_request("req_2", "memory_recall_cxd", {
            "query": "test query",
            "function_filter": "Data",
            "include_cxd_analysis": True
        })
        
        cxd_response = mcp_registry.route_request(cxd_request)
        
        # Verify CXD response structure
        assert cxd_response.success is True
        assert cxd_response.data is not None
        assert "cxd_analysis" in cxd_response.data
    
    def test_component_health_checks(self):
        """Test health checks across all components"""
        system = create_search_system(enable_mcp=True)
        
        # Test search engine health
        search_engine = system['search_engine']
        assert search_engine.health_check() is True
        
        # Test component metrics
        metrics = search_engine.get_metrics()
        assert metrics.total_searches >= 0
        assert metrics.avg_response_time_ms >= 0
        assert 0.0 <= metrics.cache_hit_rate <= 1.0
        assert metrics.error_rate >= 0.0
        
        # Test cache stats
        cache = system['components']['cache']
        cache_stats = cache.get_stats()
        assert 'cache_size' in cache_stats
        assert 'hit_rate' in cache_stats
        assert 'total_requests' in cache_stats
        
        # Test MCP handler health
        mcp_handlers = system['mcp_handlers']
        for handler_name, handler in mcp_handlers.items():
            health = handler.health_check()
            assert isinstance(health, dict)
            assert 'healthy' in health
            assert 'status' in health
    
    def test_backwards_compatibility_aliases(self):
        """Test backwards compatibility aliases"""
        from memmimic.memory.search import (
            MemorySearchEngine,
            VectorSimilarity,
            MemoryCache,
            CXDIntegration,
            ResultProcessor,
            HybridMemorySearchEngine,
            OptimizedVectorSimilarity,
            LRUMemoryCache,
            ProductionCXDIntegrationBridge,
            AdvancedResultProcessor,
        )
        
        # Verify aliases point to correct classes
        assert MemorySearchEngine == HybridMemorySearchEngine
        assert VectorSimilarity == OptimizedVectorSimilarity
        assert MemoryCache == LRUMemoryCache
        assert CXDIntegration == ProductionCXDIntegrationBridge
        assert ResultProcessor == AdvancedResultProcessor
    
    def test_factory_function_error_handling(self):
        """Test factory function error handling"""
        # Test invalid configurations gracefully handled
        try:
            # This should work even with minimal config
            system = create_search_system(config=DefaultSearchConfig())
            assert system is not None
        except Exception as e:
            pytest.fail(f"Factory function should handle basic config: {e}")
    
    def test_module_imports(self):
        """Test that all module imports work correctly"""
        try:
            from memmimic.memory.search import (
                # Core classes
                SearchQuery, SearchResult, SearchResults,
                SearchType, SimilarityMetric,
                HybridMemorySearchEngine,
                
                # Factory functions
                create_search_system,
                create_search_engine,
                create_complete_mcp_system,
                
                # MCP support
                MCPRequest, MCPResponse, MCPError,
                create_mcp_request,
                
                # Configuration
                DefaultSearchConfig,
            )
            
            # Test basic object creation
            query = SearchQuery("test")
            assert query.text == "test"
            
            config = DefaultSearchConfig()
            assert config.max_results > 0
            
            request = create_mcp_request("test", "memory_recall", {"query": "test"})
            assert request.request_id == "test"
            
        except ImportError as e:
            pytest.fail(f"Module import failed: {e}")
    
    def test_version_info(self):
        """Test version information is available"""
        import memmimic.memory.search as search_module
        
        assert hasattr(search_module, '__version__')
        assert hasattr(search_module, '__author__')
        assert search_module.__version__ == "2.0.0"
        assert "MemMimic" in search_module.__author__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])