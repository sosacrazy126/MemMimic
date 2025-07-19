"""
Core interfaces and protocols for the MemMimic Memory Search System.

This module defines the abstract contracts that all search components must implement,
ensuring clean separation of concerns and testability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class SearchType(Enum):
    """Types of search operations supported"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SimilarityMetric(Enum):
    """Vector similarity calculation methods"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


@dataclass
class SearchQuery:
    """Represents a memory search query with all parameters"""
    text: str
    limit: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    include_metadata: bool = True
    min_confidence: float = 0.0
    search_type: SearchType = SearchType.HYBRID
    timeout_ms: int = 5000


@dataclass
class SearchContext:
    """Context information about how a search was performed"""
    search_time_ms: float
    total_candidates: int
    cache_used: bool
    similarity_metric: SimilarityMetric
    cxd_classification_used: bool


@dataclass
class CXDClassification:
    """CXD classification result"""
    function: str  # Control, Context, or Data
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Individual search result with relevance and metadata"""
    memory_id: str
    content: str
    relevance_score: float
    cxd_classification: Optional[CXDClassification] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    search_context: Optional[SearchContext] = None


@dataclass
class SearchResults:
    """Complete search results with metadata"""
    results: List[SearchResult]
    total_found: int
    query: SearchQuery
    search_context: SearchContext
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SearchMetrics:
    """Performance and operational metrics for search operations"""
    total_searches: int = 0
    avg_response_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class SearchEngine(Protocol):
    """Core interface for memory search engines"""
    
    @abstractmethod
    def search(self, query: SearchQuery) -> SearchResults:
        """Execute search and return ranked results"""
        pass
    
    @abstractmethod
    def warm_cache(self, queries: List[str]) -> None:
        """Preload cache with likely search terms"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> SearchMetrics:
        """Get current performance metrics"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if search engine is operational"""
        pass


class SimilarityCalculator(Protocol):
    """Interface for vector similarity calculations"""
    
    @abstractmethod
    def calculate_similarity(self, query_embedding: List[float], 
                           memory_embedding: List[float]) -> float:
        """Calculate similarity score between two embeddings"""
        pass
    
    @abstractmethod
    def batch_calculate_similarity(self, query_embedding: List[float],
                                 memory_embeddings: List[List[float]]) -> List[float]:
        """Calculate similarity scores for multiple embeddings efficiently"""
        pass


class CXDIntegrationBridge(Protocol):
    """Interface for CXD classification integration"""
    
    @abstractmethod
    def enhance_results(self, query: SearchQuery,
                       candidates: List[SearchResult]) -> List[SearchResult]:
        """Add CXD classification information to search results"""
        pass
    
    @abstractmethod
    def classify_content(self, content: str) -> CXDClassification:
        """Classify content using CXD framework"""
        pass


class PerformanceCache(Protocol):
    """Interface for search result caching"""
    
    @abstractmethod
    def get(self, cache_key: str) -> Optional[SearchResults]:
        """Retrieve cached search results"""
        pass
    
    @abstractmethod
    def set(self, cache_key: str, results: SearchResults, ttl: int = 3600) -> None:
        """Store search results in cache"""
        pass
    
    @abstractmethod
    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        pass


class ResultProcessor(Protocol):
    """Interface for result ranking and filtering"""
    
    @abstractmethod
    def rank_results(self, query: SearchQuery, 
                    candidates: List[SearchResult]) -> List[SearchResult]:
        """Apply ranking algorithm to candidates"""
        pass
    
    @abstractmethod
    def filter_results(self, query: SearchQuery,
                      candidates: List[SearchResult]) -> List[SearchResult]:
        """Apply filters to candidate results"""
        pass


class SearchConfig(ABC):
    """Abstract base for search configuration"""
    
    @abstractmethod
    def get_similarity_metric(self) -> SimilarityMetric:
        """Get configured similarity metric"""
        pass
    
    @abstractmethod
    def get_cache_ttl(self) -> int:
        """Get cache time-to-live in seconds"""
        pass
    
    @abstractmethod
    def get_max_results(self) -> int:
        """Get maximum results to return"""
        pass


# Exception hierarchy for search operations
class SearchError(Exception):
    """Base exception for search operations"""
    
    def __init__(self, message: str, error_code: str = None, 
                 context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.timestamp = datetime.now()


class SearchEngineError(SearchError):
    """Errors in core search engine operations"""
    pass


class SimilarityCalculationError(SearchError):
    """Errors in similarity calculations"""
    pass


class CXDIntegrationError(SearchError):
    """Errors in CXD classification integration"""
    pass


class CacheError(SearchError):
    """Errors in caching operations"""
    pass


class ConfigurationError(SearchError):
    """Configuration validation errors"""
    pass