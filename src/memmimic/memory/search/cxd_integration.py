"""
CXD Classification Integration Bridge for Memory Search System.

Provides seamless integration between the memory search system and MemMimic's
CXD (Control/Context/Data) classification framework with caching and fallback.
"""

import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .interfaces import (
    CXDIntegrationBridge, SearchQuery, SearchResult, CXDClassification,
    CXDIntegrationError, SearchConfig
)

logger = logging.getLogger(__name__)


class ProductionCXDIntegrationBridge(CXDIntegrationBridge):
    """
    Production-ready CXD classification integration with caching and fallback.
    
    Features:
    - Seamless integration with existing CXD classification system
    - Classification result caching with TTL
    - Graceful fallback when CXD system unavailable
    - Performance monitoring and metrics
    - Batch classification optimization
    """
    
    def __init__(self, cxd_classifier=None, config: Optional[SearchConfig] = None):
        """
        Initialize CXD integration bridge.
        
        Args:
            cxd_classifier: MemMimic CXD classifier instance (optional for fallback)
            config: Search configuration for CXD settings
        """
        self.cxd_classifier = cxd_classifier
        self.config = config
        
        # Classification cache with TTL
        self._classification_cache = OrderedDict()
        self._cache_timestamps = OrderedDict()
        self._max_cache_size = 1000
        self._cache_ttl = 7200 if not config else config.cxd_cache_ttl  # 2 hours default
        
        # Performance metrics
        self._metrics = {
            'total_classifications': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'classification_errors': 0,
            'batch_classifications': 0,
            'avg_classification_time_ms': 0.0,
            'total_classification_time': 0.0,
            'fallback_activations': 0,
        }
        
        # Classification confidence boosters based on CXD function
        self._cxd_relevance_boosters = {
            'Control': {
                'keywords': ['manage', 'control', 'execute', 'run', 'command', 'action'],
                'boost_factor': 0.1
            },
            'Context': {
                'keywords': ['context', 'relate', 'reference', 'previous', 'similar', 'background'],
                'boost_factor': 0.15
            },
            'Data': {
                'keywords': ['data', 'information', 'fact', 'analysis', 'technical', 'specific'],
                'boost_factor': 0.12
            }
        }
        
        logger.info("CXD Integration Bridge initialized")
    
    def enhance_results(self, query: SearchQuery, 
                       candidates: List[SearchResult]) -> List[SearchResult]:
        """
        Add CXD classification information to search results.
        
        Args:
            query: Original search query
            candidates: List of search result candidates
            
        Returns:
            Enhanced search results with CXD classifications
        """
        if not candidates:
            return candidates
        
        try:
            start_time = time.time()
            enhanced_results = []
            
            # Batch classify for efficiency if multiple candidates
            if len(candidates) > 1:
                enhanced_results = self._batch_enhance_results(query, candidates)
                self._metrics['batch_classifications'] += 1
            else:
                enhanced_results = self._single_enhance_results(query, candidates)
            
            # Update performance metrics
            classification_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_classification_metrics(classification_time, len(candidates))
            
            return enhanced_results
            
        except Exception as e:
            self._metrics['classification_errors'] += 1
            logger.warning(f"CXD enhancement failed, returning original results: {e}")
            
            # Return original results with basic CXD classification fallback
            return self._apply_fallback_classification(candidates)
    
    def classify_content(self, content: str) -> CXDClassification:
        """
        Classify content using CXD framework with caching.
        
        Args:
            content: Text content to classify
            
        Returns:
            CXD classification result
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(content)
            cached_classification = self._get_cached_classification(cache_key)
            
            if cached_classification:
                self._metrics['cache_hits'] += 1
                return cached_classification
            
            # Perform classification
            classification = self._perform_classification(content)
            
            # Cache the result
            self._cache_classification(cache_key, classification)
            self._metrics['cache_misses'] += 1
            self._metrics['total_classifications'] += 1
            
            return classification
            
        except Exception as e:
            self._metrics['classification_errors'] += 1
            logger.warning(f"Content classification failed: {e}")
            
            # Return fallback classification
            return self._create_fallback_classification(content)
    
    def _batch_enhance_results(self, query: SearchQuery, 
                              candidates: List[SearchResult]) -> List[SearchResult]:
        """Efficiently enhance multiple results using batch classification."""
        enhanced_results = []
        
        # Prepare content for batch classification
        contents = [candidate.content for candidate in candidates]
        
        # Batch classify (if CXD classifier supports it, otherwise fallback to individual)
        try:
            if hasattr(self.cxd_classifier, 'batch_classify'):
                classifications = self.cxd_classifier.batch_classify(contents)
            else:
                # Fallback to individual classifications
                classifications = [self.classify_content(content) for content in contents]
            
            # Enhance each result
            for candidate, classification in zip(candidates, classifications):
                enhanced_candidate = self._enhance_single_result(
                    candidate, classification, query
                )
                enhanced_results.append(enhanced_candidate)
                
        except Exception as e:
            logger.warning(f"Batch classification failed, using fallback: {e}")
            enhanced_results = self._apply_fallback_classification(candidates)
        
        return enhanced_results
    
    def _single_enhance_results(self, query: SearchQuery,
                               candidates: List[SearchResult]) -> List[SearchResult]:
        """Enhance results one by one."""
        enhanced_results = []
        
        for candidate in candidates:
            try:
                classification = self.classify_content(candidate.content)
                enhanced_candidate = self._enhance_single_result(
                    candidate, classification, query
                )
                enhanced_results.append(enhanced_candidate)
                
            except Exception as e:
                logger.warning(f"Single classification failed for {candidate.memory_id}: {e}")
                # Add original candidate with fallback classification
                fallback_classification = self._create_fallback_classification(candidate.content)
                enhanced_candidate = self._enhance_single_result(
                    candidate, fallback_classification, query
                )
                enhanced_results.append(enhanced_candidate)
        
        return enhanced_results
    
    def _enhance_single_result(self, candidate: SearchResult, 
                              classification: CXDClassification,
                              query: SearchQuery) -> SearchResult:
        """Enhance a single search result with CXD classification."""
        
        # Calculate CXD-based relevance boost
        cxd_boost = self._calculate_cxd_relevance_boost(query, classification)
        
        # Apply boost to relevance score (capped at 1.0)
        boosted_score = min(candidate.relevance_score + cxd_boost, 1.0)
        
        # Create enhanced result
        enhanced_result = SearchResult(
            memory_id=candidate.memory_id,
            content=candidate.content,
            relevance_score=boosted_score,
            cxd_classification=classification,
            metadata={
                **candidate.metadata,
                'cxd_boost_applied': cxd_boost,
                'original_relevance': candidate.relevance_score,
                'enhanced_by_cxd': True
            },
            search_context=candidate.search_context
        )
        
        return enhanced_result
    
    def _perform_classification(self, content: str) -> CXDClassification:
        """Perform actual CXD classification."""
        if self.cxd_classifier is None:
            return self._create_fallback_classification(content)
        
        try:
            # Use existing MemMimic CXD classifier
            classification_result = self.cxd_classifier.classify(content)
            
            # Convert to our CXDClassification format
            if hasattr(classification_result, 'function'):
                function = classification_result.function
                confidence = getattr(classification_result, 'confidence', 0.5)
                metadata = getattr(classification_result, 'metadata', {})
            else:
                # Handle string result or other formats
                function = str(classification_result) if classification_result else 'Data'
                confidence = 0.5
                metadata = {'source': 'cxd_classifier'}
            
            return CXDClassification(
                function=function,
                confidence=confidence,
                metadata={
                    **metadata,
                    'classification_method': 'cxd_classifier',
                    'classified_at': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.warning(f"CXD classifier failed: {e}")
            return self._create_fallback_classification(content)
    
    def _create_fallback_classification(self, content: str) -> CXDClassification:
        """Create fallback classification using heuristics."""
        self._metrics['fallback_activations'] += 1
        
        content_lower = content.lower()
        
        # Simple heuristic-based classification
        if any(keyword in content_lower for keyword in ['command', 'execute', 'run', 'manage', 'control']):
            function = 'Control'
            confidence = 0.6
        elif any(keyword in content_lower for keyword in ['context', 'reference', 'previous', 'related']):
            function = 'Context'
            confidence = 0.6
        else:
            function = 'Data'
            confidence = 0.5
        
        return CXDClassification(
            function=function,
            confidence=confidence,
            metadata={
                'classification_method': 'fallback_heuristic',
                'classified_at': datetime.now().isoformat(),
                'fallback_reason': 'cxd_classifier_unavailable'
            }
        )
    
    def _calculate_cxd_relevance_boost(self, query: SearchQuery, 
                                     classification: CXDClassification) -> float:
        """Calculate relevance boost based on CXD classification and query."""
        if not classification or classification.confidence < 0.3:
            return 0.0
        
        function = classification.function
        query_text = query.text.lower()
        
        # Get boost configuration for this CXD function
        boost_config = self._cxd_relevance_boosters.get(function, {})
        if not boost_config:
            return 0.0
        
        # Check if query contains relevant keywords for this function
        keywords = boost_config.get('keywords', [])
        keyword_matches = sum(1 for keyword in keywords if keyword in query_text)
        
        if keyword_matches == 0:
            return 0.0
        
        # Calculate boost based on classification confidence and keyword matches
        base_boost = boost_config.get('boost_factor', 0.1)
        confidence_multiplier = classification.confidence
        keyword_multiplier = min(keyword_matches / len(keywords), 1.0)
        
        boost = base_boost * confidence_multiplier * keyword_multiplier
        
        # Cap boost at reasonable level
        return min(boost, 0.2)
    
    def _apply_fallback_classification(self, candidates: List[SearchResult]) -> List[SearchResult]:
        """Apply fallback classification to all candidates."""
        enhanced_results = []
        
        for candidate in candidates:
            fallback_classification = self._create_fallback_classification(candidate.content)
            enhanced_candidate = SearchResult(
                memory_id=candidate.memory_id,
                content=candidate.content,
                relevance_score=candidate.relevance_score,
                cxd_classification=fallback_classification,
                metadata={
                    **candidate.metadata,
                    'enhanced_by_cxd': True,
                    'fallback_classification': True
                },
                search_context=candidate.search_context
            )
            enhanced_results.append(enhanced_candidate)
        
        return enhanced_results
    
    def _generate_cache_key(self, content: str) -> str:
        """Generate cache key for content."""
        import hashlib
        # Use first 500 chars to avoid huge cache keys
        content_sample = content[:500]
        return hashlib.md5(content_sample.encode('utf-8')).hexdigest()
    
    def _get_cached_classification(self, cache_key: str) -> Optional[CXDClassification]:
        """Retrieve cached classification if not expired."""
        if cache_key not in self._classification_cache:
            return None
        
        # Check if expired
        timestamp = self._cache_timestamps.get(cache_key)
        if timestamp and time.time() - timestamp > self._cache_ttl:
            # Remove expired entry
            self._classification_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
            return None
        
        # Move to end (LRU)
        classification = self._classification_cache.pop(cache_key)
        self._classification_cache[cache_key] = classification
        
        return classification
    
    def _cache_classification(self, cache_key: str, classification: CXDClassification):
        """Cache classification result with TTL."""
        # Evict oldest if cache full
        if len(self._classification_cache) >= self._max_cache_size:
            oldest_key = next(iter(self._classification_cache))
            self._classification_cache.pop(oldest_key)
            self._cache_timestamps.pop(oldest_key, None)
        
        # Add new entry
        self._classification_cache[cache_key] = classification
        self._cache_timestamps[cache_key] = time.time()
    
    def _update_classification_metrics(self, classification_time_ms: float, result_count: int):
        """Update performance metrics."""
        self._metrics['total_classification_time'] += classification_time_ms
        
        total_classifications = self._metrics['total_classifications']
        if total_classifications > 0:
            self._metrics['avg_classification_time_ms'] = (
                self._metrics['total_classification_time'] / total_classifications
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get CXD integration performance metrics."""
        total_requests = self._metrics['cache_hits'] + self._metrics['cache_misses']
        cache_hit_rate = (
            self._metrics['cache_hits'] / total_requests if total_requests > 0 else 0.0
        )
        
        return {
            'total_classifications': self._metrics['total_classifications'],
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self._metrics['cache_hits'],
            'cache_misses': self._metrics['cache_misses'],
            'classification_errors': self._metrics['classification_errors'],
            'batch_classifications': self._metrics['batch_classifications'],
            'avg_classification_time_ms': self._metrics['avg_classification_time_ms'],
            'fallback_activations': self._metrics['fallback_activations'],
            'cache_size': len(self._classification_cache),
            'cache_max_size': self._max_cache_size,
            'cache_ttl_seconds': self._cache_ttl,
        }
    
    def clear_cache(self):
        """Clear classification cache."""
        self._classification_cache.clear()
        self._cache_timestamps.clear()
        logger.info("CXD classification cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of CXD integration."""
        health_status = {
            'cxd_classifier_available': self.cxd_classifier is not None,
            'cache_operational': True,
            'fallback_ready': True,
            'metrics_tracking': True,
        }
        
        # Test classification if classifier available
        if self.cxd_classifier:
            try:
                test_classification = self.classify_content("test content")
                health_status['test_classification_successful'] = True
                health_status['last_test_confidence'] = test_classification.confidence
            except Exception as e:
                health_status['test_classification_successful'] = False
                health_status['test_error'] = str(e)
        
        return health_status


def create_cxd_integration_bridge(cxd_classifier=None, config=None) -> CXDIntegrationBridge:
    """
    Factory function to create CXD integration bridge.
    
    Args:
        cxd_classifier: MemMimic CXD classifier instance
        config: Search configuration
        
    Returns:
        CXDIntegrationBridge instance
    """
    return ProductionCXDIntegrationBridge(cxd_classifier, config)