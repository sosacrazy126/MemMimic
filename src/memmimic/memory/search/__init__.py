"""
Memory Search System - Modern modular search engine for MemMimic.

A high-performance, modular memory search system that combines semantic and keyword 
matching with CXD classification support, advanced caching, and comprehensive 
performance monitoring.

Main Components:
- HybridMemorySearchEngine: Core search orchestration
- OptimizedVectorSimilarity: High-performance similarity calculations  
- LRUMemoryCache: Advanced caching with TTL support
- ProductionCXDIntegrationBridge: CXD classification integration
- AdvancedResultProcessor: Result ranking and filtering
- MCP handlers: Protocol handlers for memory recall operations

Usage Examples:
    Basic search setup:
    >>> from memmimic.memory.search import create_search_system
    >>> search_system = create_search_system()
    >>> results = search_system.search(SearchQuery("machine learning"))
    
    Advanced configuration:
    >>> from memmimic.memory.search import (
    ...     create_search_engine, DefaultSearchConfig, SimilarityMetric
    ... )
    >>> config = DefaultSearchConfig(
    ...     similarity_metric=SimilarityMetric.COSINE,
    ...     max_results=50,
    ...     enable_cxd_classification=True
    ... )
    >>> engine = create_search_engine(config=config)
    
    MCP handler setup:
    >>> from memmimic.memory.search.mcp_handlers import create_memory_recall_handler
    >>> handler = create_memory_recall_handler(search_engine, "standard")
"""

# Core interfaces and data models
from .interfaces import (
    # Search interfaces
    SearchEngine,
    SimilarityCalculator, 
    CXDIntegrationBridge,
    PerformanceCache,
    ResultProcessor,
    
    # Data models
    SearchQuery,
    SearchResult,
    SearchResults,
    SearchContext,
    SearchMetrics,
    CXDClassification,
    
    # Enums
    SearchType,
    SimilarityMetric,
    
    # Configuration
    SearchConfig,
    
    # Exceptions
    SearchError,
    SearchEngineError,
    SimilarityCalculationError,
    CXDIntegrationError,
    CacheError,
)

# Configuration
from .search_config import (
    DefaultSearchConfig,
    create_search_config,
)

# Core implementations
from .search_engine import HybridMemorySearchEngine
from .vector_similarity import (
    OptimizedVectorSimilarity,
    create_similarity_calculator,
)
from .performance_cache import (
    LRUMemoryCache,
    create_performance_cache,
    generate_search_cache_key,
)
from .cxd_integration import (
    ProductionCXDIntegrationBridge,
    create_cxd_integration_bridge,
)
from .result_processor import (
    AdvancedResultProcessor,
    RankingAlgorithm,
    RankingWeights,
    create_result_processor,
)

# MCP protocol support
from .mcp_base import (
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPBaseHandler,
    MCPHandlerRegistry,
    create_mcp_request,
    create_success_response,
    create_error_response,
)
from .mcp_handlers import (
    MemoryRecallMCPHandler,
    MemoryRecallCXDHandler,
    create_memory_recall_handler,
)
from .response_formatter import (
    MCPResponseFormatter,
    create_mcp_response_formatter,
)

# Version info
__version__ = "2.0.0"
__author__ = "MemMimic Memory Search Team"

# Export main factory functions
__all__ = [
    # Factory functions
    "create_search_system",
    "create_search_engine", 
    "create_complete_mcp_system",
    
    # Core classes
    "HybridMemorySearchEngine",
    "OptimizedVectorSimilarity",
    "LRUMemoryCache",
    "ProductionCXDIntegrationBridge", 
    "AdvancedResultProcessor",
    
    # Data models
    "SearchQuery",
    "SearchResult", 
    "SearchResults",
    "SearchContext",
    "SearchMetrics",
    "CXDClassification",
    
    # Enums
    "SearchType",
    "SimilarityMetric",
    
    # Configuration
    "SearchConfig",
    "DefaultSearchConfig",
    
    # MCP Support
    "MCPRequest",
    "MCPResponse", 
    "MCPError",
    "MCPBaseHandler",
    "MCPHandlerRegistry",
    "MemoryRecallMCPHandler",
    "MemoryRecallCXDHandler",
    "MCPResponseFormatter",
    
    # Exceptions
    "SearchError",
    "SearchEngineError",
    "SimilarityCalculationError",
    "CXDIntegrationError",
    "CacheError",
]


def create_search_system(config: SearchConfig = None, 
                        cxd_classifier=None,
                        enable_mcp: bool = False) -> dict:
    """
    Create complete memory search system with all components.
    
    This is the main factory function for setting up a complete memory search 
    system with all necessary components properly configured and integrated.
    
    Args:
        config: Search configuration (uses defaults if None)
        cxd_classifier: CXD classifier instance (optional)
        enable_mcp: Whether to create MCP handlers
        
    Returns:
        Dictionary containing:
        - 'search_engine': Configured HybridMemorySearchEngine
        - 'mcp_registry': MCPHandlerRegistry (if enable_mcp=True)
        - 'config': Final configuration used
        - 'components': Individual component instances
        
    Example:
        >>> system = create_search_system(enable_mcp=True)
        >>> search_engine = system['search_engine']
        >>> mcp_registry = system['mcp_registry']
        >>> 
        >>> # Use search engine directly
        >>> results = search_engine.search(SearchQuery("test query"))
        >>> 
        >>> # Use MCP protocol
        >>> request = create_mcp_request("req_1", "memory_recall", {"query": "test"})
        >>> response = mcp_registry.route_request(request)
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Use default config if none provided
    if config is None:
        config = DefaultSearchConfig()
    
    # Create core components
    logger.info("Creating memory search system components...")
    
    # Similarity calculator
    similarity_calculator = create_similarity_calculator(config=config)
    
    # Performance cache
    cache = create_performance_cache(
        cache_type="memory",
        max_size=getattr(config, 'cache_max_size', 1000),
        default_ttl=getattr(config, 'cache_ttl', 3600)
    )
    
    # CXD integration bridge
    cxd_bridge = create_cxd_integration_bridge(
        cxd_classifier=cxd_classifier,
        config=config
    )
    
    # Result processor
    result_processor = create_result_processor(config=config)
    
    # Main search engine
    search_engine = HybridMemorySearchEngine(
        similarity_calculator=similarity_calculator,
        cxd_bridge=cxd_bridge,
        cache=cache,
        result_processor=result_processor,
        config=config
    )
    
    components = {
        'similarity_calculator': similarity_calculator,
        'cache': cache,
        'cxd_bridge': cxd_bridge,
        'result_processor': result_processor,
    }
    
    system = {
        'search_engine': search_engine,
        'config': config,
        'components': components,
    }
    
    # Create MCP handlers if requested
    if enable_mcp:
        logger.info("Creating MCP handler system...")
        
        mcp_registry = MCPHandlerRegistry()
        
        # Create standard memory recall handler
        standard_handler = create_memory_recall_handler(search_engine, "standard")
        mcp_registry.register_handler("memory_recall", standard_handler)
        
        # Create CXD-enhanced handler
        cxd_handler = create_memory_recall_handler(search_engine, "cxd")
        mcp_registry.register_handler("memory_recall_cxd", cxd_handler)
        
        system['mcp_registry'] = mcp_registry
        system['mcp_handlers'] = {
            'standard': standard_handler,
            'cxd': cxd_handler,
        }
    
    logger.info(f"Memory search system created successfully (MCP: {enable_mcp})")
    return system


def create_search_engine(similarity_calculator: SimilarityCalculator = None,
                        cxd_bridge: CXDIntegrationBridge = None,
                        cache: PerformanceCache = None,
                        result_processor: ResultProcessor = None,
                        config: SearchConfig = None) -> HybridMemorySearchEngine:
    """
    Create search engine with specified or default components.
    
    Args:
        similarity_calculator: Vector similarity calculator (creates default if None)
        cxd_bridge: CXD integration bridge (creates default if None)
        cache: Performance cache (creates default if None)
        result_processor: Result processor (creates default if None)
        config: Search configuration (creates default if None)
        
    Returns:
        Configured HybridMemorySearchEngine instance
        
    Example:
        >>> # Use all defaults
        >>> engine = create_search_engine()
        >>> 
        >>> # Custom configuration
        >>> config = DefaultSearchConfig(max_results=50)
        >>> engine = create_search_engine(config=config)
    """
    if config is None:
        config = DefaultSearchConfig()
    
    if similarity_calculator is None:
        similarity_calculator = create_similarity_calculator(config=config)
    
    if cache is None:
        cache = create_performance_cache("memory")
    
    if cxd_bridge is None:
        cxd_bridge = create_cxd_integration_bridge(config=config)
    
    if result_processor is None:
        result_processor = create_result_processor(config=config)
    
    return HybridMemorySearchEngine(
        similarity_calculator=similarity_calculator,
        cxd_bridge=cxd_bridge,
        cache=cache,
        result_processor=result_processor,
        config=config
    )


def create_complete_mcp_system(search_engine: SearchEngine = None,
                             config: SearchConfig = None) -> MCPHandlerRegistry:
    """
    Create complete MCP handler system for memory recall operations.
    
    Args:
        search_engine: Search engine instance (creates default if None)
        config: Search configuration (creates default if None)
        
    Returns:
        MCPHandlerRegistry with all handlers registered
        
    Example:
        >>> registry = create_complete_mcp_system()
        >>> 
        >>> # Handle memory recall request
        >>> request = create_mcp_request("req_1", "memory_recall", {"query": "test"})
        >>> response = registry.route_request(request)
        >>> 
        >>> # Handle CXD-enhanced request  
        >>> cxd_request = create_mcp_request("req_2", "memory_recall_cxd", {
        ...     "query": "test",
        ...     "function_filter": "Data"
        ... })
        >>> cxd_response = registry.route_request(cxd_request)
    """
    if search_engine is None:
        search_engine = create_search_engine(config=config)
    
    registry = MCPHandlerRegistry()
    
    # Register standard handler
    standard_handler = create_memory_recall_handler(search_engine, "standard")
    registry.register_handler("memory_recall", standard_handler)
    
    # Register CXD-enhanced handler
    cxd_handler = create_memory_recall_handler(search_engine, "cxd")
    registry.register_handler("memory_recall_cxd", cxd_handler)
    
    return registry


# Backwards compatibility aliases
MemorySearchEngine = HybridMemorySearchEngine
VectorSimilarity = OptimizedVectorSimilarity
MemoryCache = LRUMemoryCache
CXDIntegration = ProductionCXDIntegrationBridge
ResultProcessor = AdvancedResultProcessor