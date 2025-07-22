"""
MemMimic-Specific Performance Tracking

Specialized performance tracking for MemMimic operations including:
- CXD classification performance
- Cache hit/miss rates and efficiency
- Search response times and accuracy
- Memory consolidation metrics
- Quality gate performance
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics

from .metrics_collector import get_metrics_collector, metrics_timer, metrics_counter

logger = logging.getLogger(__name__)


@dataclass
class CXDPerformanceMetrics:
    """CXD classification performance metrics"""
    total_classifications: int = 0
    successful_classifications: int = 0
    failed_classifications: int = 0
    classification_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    classification_accuracy: float = 0.0
    pattern_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def get_success_rate(self) -> float:
        if self.total_classifications == 0:
            return 0.0
        return self.successful_classifications / self.total_classifications
    
    def get_average_time(self) -> float:
        if not self.classification_times:
            return 0.0
        return statistics.mean(self.classification_times)
    
    def get_p95_time(self) -> float:
        if len(self.classification_times) < 20:
            return 0.0
        sorted_times = sorted(self.classification_times)
        return statistics.quantiles(sorted_times, n=20)[18]


@dataclass
class CachePerformanceMetrics:
    """Cache performance metrics"""
    total_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    miss_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    cache_size_bytes: int = 0
    cache_entries: int = 0
    evictions: int = 0
    
    def get_hit_rate(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return self.cache_hits / self.total_operations
    
    def get_average_hit_time(self) -> float:
        if not self.hit_times:
            return 0.0
        return statistics.mean(self.hit_times)
    
    def get_average_miss_time(self) -> float:
        if not self.miss_times:
            return 0.0
        return statistics.mean(self.miss_times)
    
    def get_efficiency_score(self) -> float:
        """Calculate cache efficiency (0-1 score)"""
        hit_rate = self.get_hit_rate()
        avg_hit_time = self.get_average_hit_time()
        avg_miss_time = self.get_average_miss_time()
        
        # Efficiency based on hit rate and time savings
        if avg_miss_time == 0:
            return hit_rate
        
        time_efficiency = min(1.0, avg_hit_time / avg_miss_time) if avg_miss_time > 0 else 1.0
        return (hit_rate * 0.7) + (time_efficiency * 0.3)


@dataclass
class SearchPerformanceMetrics:
    """Search operation performance metrics"""
    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    search_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    relevance_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    results_returned: deque = field(default_factory=lambda: deque(maxlen=1000))
    search_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def get_success_rate(self) -> float:
        if self.total_searches == 0:
            return 0.0
        return self.successful_searches / self.total_searches
    
    def get_average_time(self) -> float:
        if not self.search_times:
            return 0.0
        return statistics.mean(self.search_times)
    
    def get_average_relevance(self) -> float:
        if not self.relevance_scores:
            return 0.0
        return statistics.mean(self.relevance_scores)
    
    def get_average_results(self) -> float:
        if not self.results_returned:
            return 0.0
        return statistics.mean(self.results_returned)


@dataclass
class MemoryConsolidationMetrics:
    """Memory consolidation performance metrics"""
    total_consolidations: int = 0
    successful_consolidations: int = 0
    failed_consolidations: int = 0
    consolidation_times: deque = field(default_factory=lambda: deque(maxlen=100))
    memories_processed: deque = field(default_factory=lambda: deque(maxlen=100))
    space_saved_bytes: int = 0
    duplicate_memories_found: int = 0
    
    def get_success_rate(self) -> float:
        if self.total_consolidations == 0:
            return 0.0
        return self.successful_consolidations / self.total_consolidations
    
    def get_average_time(self) -> float:
        if not self.consolidation_times:
            return 0.0
        return statistics.mean(self.consolidation_times)
    
    def get_efficiency_score(self) -> float:
        """Calculate consolidation efficiency"""
        if self.total_consolidations == 0:
            return 0.0
        
        success_rate = self.get_success_rate()
        space_efficiency = min(1.0, self.space_saved_bytes / max(1024*1024, self.space_saved_bytes))  # MB threshold
        
        return (success_rate * 0.6) + (space_efficiency * 0.4)


@dataclass
class QualityGateMetrics:
    """Quality gate performance metrics"""
    total_evaluations: int = 0
    passed_evaluations: int = 0
    failed_evaluations: int = 0
    evaluation_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    quality_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    rejection_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def get_pass_rate(self) -> float:
        if self.total_evaluations == 0:
            return 0.0
        return self.passed_evaluations / self.total_evaluations
    
    def get_average_time(self) -> float:
        if not self.evaluation_times:
            return 0.0
        return statistics.mean(self.evaluation_times)
    
    def get_average_quality_score(self) -> float:
        if not self.quality_scores:
            return 0.0
        return statistics.mean(self.quality_scores)


class MemMimicPerformanceTracker:
    """
    Specialized performance tracker for MemMimic operations.
    
    Tracks detailed performance metrics for all major MemMimic subsystems
    and provides analytics and insights for optimization.
    """
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        
        # Component-specific metrics
        self.cxd_metrics = CXDPerformanceMetrics()
        self.cache_metrics = CachePerformanceMetrics()
        self.search_metrics = SearchPerformanceMetrics()
        self.consolidation_metrics = MemoryConsolidationMetrics()
        self.quality_gate_metrics = QualityGateMetrics()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance baselines (learned over time)
        self.baselines = {
            'cxd_classification_time': 0.1,  # seconds
            'cache_hit_time': 0.01,  # seconds
            'search_time': 0.5,  # seconds
            'consolidation_time': 2.0,  # seconds
            'quality_evaluation_time': 0.05  # seconds
        }
        
        logger.info("MemMimic Performance Tracker initialized")
    
    @metrics_timer("memmimic_cxd_classification_duration_seconds")
    def track_cxd_classification(self, content: str, start_time: Optional[float] = None) -> str:
        """Context manager for tracking CXD classification performance"""
        if start_time is None:
            start_time = time.time()
        
        operation_id = f"cxd_{int(time.time() * 1000000)}"
        return CXDClassificationTracker(self, operation_id, content, start_time)
    
    @metrics_timer("memmimic_cache_operation_duration_seconds")
    def track_cache_operation(self, operation_type: str, key: str) -> str:
        """Context manager for tracking cache operations"""
        start_time = time.time()
        operation_id = f"cache_{operation_type}_{int(time.time() * 1000000)}"
        return CacheOperationTracker(self, operation_id, operation_type, key, start_time)
    
    @metrics_timer("memmimic_search_duration_seconds")
    def track_search_operation(self, query: str, search_type: str = "hybrid") -> str:
        """Context manager for tracking search operations"""
        start_time = time.time()
        operation_id = f"search_{int(time.time() * 1000000)}"
        return SearchOperationTracker(self, operation_id, query, search_type, start_time)
    
    @metrics_timer("memmimic_consolidation_duration_seconds")
    def track_consolidation_operation(self) -> str:
        """Context manager for tracking memory consolidation"""
        start_time = time.time()
        operation_id = f"consolidation_{int(time.time() * 1000000)}"
        return ConsolidationTracker(self, operation_id, start_time)
    
    @metrics_timer("memmimic_quality_gate_duration_seconds")
    def track_quality_evaluation(self, content: str) -> str:
        """Context manager for tracking quality gate evaluations"""
        start_time = time.time()
        operation_id = f"quality_{int(time.time() * 1000000)}"
        return QualityGateTracker(self, operation_id, content, start_time)
    
    def record_cxd_classification(self, duration_seconds: float, success: bool, 
                                pattern: Optional[str] = None, accuracy: Optional[float] = None):
        """Record CXD classification metrics"""
        with self._lock:
            self.cxd_metrics.total_classifications += 1
            self.cxd_metrics.classification_times.append(duration_seconds)
            
            if success:
                self.cxd_metrics.successful_classifications += 1
                if pattern:
                    self.cxd_metrics.pattern_distribution[pattern] += 1
            else:
                self.cxd_metrics.failed_classifications += 1
            
            if accuracy is not None:
                self.cxd_metrics.classification_accuracy = (
                    (self.cxd_metrics.classification_accuracy * (self.cxd_metrics.total_classifications - 1) + accuracy)
                    / self.cxd_metrics.total_classifications
                )
            
            # Update metrics
            self.metrics_collector.observe_histogram("memmimic_cxd_classification_duration_seconds", duration_seconds)
            self.metrics_collector.increment_counter("memmimic_cxd_classifications_total")
            if success:
                self.metrics_collector.increment_counter("memmimic_cxd_classifications_successful_total")
            
            # Performance analysis
            if duration_seconds > self.baselines['cxd_classification_time'] * 3:
                logger.warning(f"Slow CXD classification detected: {duration_seconds:.3f}s")
    
    def record_cache_operation(self, operation_type: str, duration_seconds: float, 
                             hit: bool, cache_size_bytes: Optional[int] = None,
                             cache_entries: Optional[int] = None):
        """Record cache operation metrics"""
        with self._lock:
            self.cache_metrics.total_operations += 1
            
            if hit:
                self.cache_metrics.cache_hits += 1
                self.cache_metrics.hit_times.append(duration_seconds)
                self.metrics_collector.increment_counter("memmimic_cache_hits_total")
            else:
                self.cache_metrics.cache_misses += 1
                self.cache_metrics.miss_times.append(duration_seconds)
                self.metrics_collector.increment_counter("memmimic_cache_misses_total")
            
            if cache_size_bytes is not None:
                self.cache_metrics.cache_size_bytes = cache_size_bytes
                self.metrics_collector.set_gauge("memmimic_cache_size_bytes", cache_size_bytes)
            
            if cache_entries is not None:
                self.cache_metrics.cache_entries = cache_entries
                self.metrics_collector.set_gauge("memmimic_cache_entries_count", cache_entries)
            
            # Update hit rate metric
            hit_rate = self.cache_metrics.get_hit_rate()
            self.metrics_collector.set_gauge("memmimic_cache_hit_rate", hit_rate)
            
            # Performance analysis
            expected_time = self.baselines['cache_hit_time'] if hit else self.baselines['cache_hit_time'] * 10
            if duration_seconds > expected_time * 3:
                logger.warning(f"Slow cache {operation_type} detected: {duration_seconds:.3f}s (hit: {hit})")
    
    def record_search_operation(self, search_type: str, duration_seconds: float, 
                              success: bool, results_count: int = 0, 
                              relevance_score: Optional[float] = None):
        """Record search operation metrics"""
        with self._lock:
            self.search_metrics.total_searches += 1
            self.search_metrics.search_times.append(duration_seconds)
            self.search_metrics.search_types[search_type] += 1
            
            if success:
                self.search_metrics.successful_searches += 1
                self.search_metrics.results_returned.append(results_count)
                
                if relevance_score is not None:
                    self.search_metrics.relevance_scores.append(relevance_score)
                    
                self.metrics_collector.increment_counter("memmimic_searches_successful_total")
            else:
                self.search_metrics.failed_searches += 1
            
            self.metrics_collector.observe_histogram("memmimic_search_duration_seconds", duration_seconds)
            self.metrics_collector.increment_counter("memmimic_searches_total")
            
            # Performance analysis
            if duration_seconds > self.baselines['search_time'] * 3:
                logger.warning(f"Slow search detected: {duration_seconds:.3f}s (type: {search_type})")
    
    def record_consolidation_operation(self, duration_seconds: float, success: bool,
                                     memories_processed: int = 0, space_saved_bytes: int = 0,
                                     duplicates_found: int = 0):
        """Record memory consolidation metrics"""
        with self._lock:
            self.consolidation_metrics.total_consolidations += 1
            self.consolidation_metrics.consolidation_times.append(duration_seconds)
            
            if success:
                self.consolidation_metrics.successful_consolidations += 1
                self.consolidation_metrics.memories_processed.append(memories_processed)
                self.consolidation_metrics.space_saved_bytes += space_saved_bytes
                self.consolidation_metrics.duplicate_memories_found += duplicates_found
                
                self.metrics_collector.increment_counter("memmimic_consolidations_successful_total")
            
            self.metrics_collector.observe_histogram("memmimic_consolidation_duration_seconds", duration_seconds)
            self.metrics_collector.increment_counter("memmimic_consolidations_total")
            
            # Performance analysis
            if duration_seconds > self.baselines['consolidation_time'] * 3:
                logger.warning(f"Slow consolidation detected: {duration_seconds:.3f}s")
    
    def record_quality_evaluation(self, duration_seconds: float, passed: bool,
                                quality_score: Optional[float] = None,
                                rejection_reason: Optional[str] = None):
        """Record quality gate evaluation metrics"""
        with self._lock:
            self.quality_gate_metrics.total_evaluations += 1
            self.quality_gate_metrics.evaluation_times.append(duration_seconds)
            
            if passed:
                self.quality_gate_metrics.passed_evaluations += 1
                self.metrics_collector.increment_counter("memmimic_quality_gate_pass_total")
            else:
                self.quality_gate_metrics.failed_evaluations += 1
                self.metrics_collector.increment_counter("memmimic_quality_gate_fail_total")
                
                if rejection_reason:
                    self.quality_gate_metrics.rejection_reasons[rejection_reason] += 1
            
            if quality_score is not None:
                self.quality_gate_metrics.quality_scores.append(quality_score)
                self.metrics_collector.observe_histogram("memmimic_quality_score", quality_score)
            
            # Performance analysis
            if duration_seconds > self.baselines['quality_evaluation_time'] * 5:
                logger.warning(f"Slow quality evaluation detected: {duration_seconds:.3f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'cxd_classification': {
                    'total_operations': self.cxd_metrics.total_classifications,
                    'success_rate': self.cxd_metrics.get_success_rate(),
                    'average_time_seconds': self.cxd_metrics.get_average_time(),
                    'p95_time_seconds': self.cxd_metrics.get_p95_time(),
                    'pattern_distribution': dict(self.cxd_metrics.pattern_distribution),
                    'accuracy': self.cxd_metrics.classification_accuracy
                },
                'cache_performance': {
                    'total_operations': self.cache_metrics.total_operations,
                    'hit_rate': self.cache_metrics.get_hit_rate(),
                    'average_hit_time_seconds': self.cache_metrics.get_average_hit_time(),
                    'average_miss_time_seconds': self.cache_metrics.get_average_miss_time(),
                    'efficiency_score': self.cache_metrics.get_efficiency_score(),
                    'cache_size_mb': self.cache_metrics.cache_size_bytes / (1024*1024),
                    'cache_entries': self.cache_metrics.cache_entries,
                    'evictions': self.cache_metrics.evictions
                },
                'search_performance': {
                    'total_searches': self.search_metrics.total_searches,
                    'success_rate': self.search_metrics.get_success_rate(),
                    'average_time_seconds': self.search_metrics.get_average_time(),
                    'average_relevance_score': self.search_metrics.get_average_relevance(),
                    'average_results_returned': self.search_metrics.get_average_results(),
                    'search_types': dict(self.search_metrics.search_types)
                },
                'memory_consolidation': {
                    'total_consolidations': self.consolidation_metrics.total_consolidations,
                    'success_rate': self.consolidation_metrics.get_success_rate(),
                    'average_time_seconds': self.consolidation_metrics.get_average_time(),
                    'efficiency_score': self.consolidation_metrics.get_efficiency_score(),
                    'space_saved_mb': self.consolidation_metrics.space_saved_bytes / (1024*1024),
                    'duplicates_found': self.consolidation_metrics.duplicate_memories_found
                },
                'quality_gate': {
                    'total_evaluations': self.quality_gate_metrics.total_evaluations,
                    'pass_rate': self.quality_gate_metrics.get_pass_rate(),
                    'average_time_seconds': self.quality_gate_metrics.get_average_time(),
                    'average_quality_score': self.quality_gate_metrics.get_average_quality_score(),
                    'rejection_reasons': dict(self.quality_gate_metrics.rejection_reasons)
                },
                'performance_baselines': self.baselines.copy()
            }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        with self._lock:
            # CXD Classification recommendations
            if (self.cxd_metrics.total_classifications > 100 and
                self.cxd_metrics.get_average_time() > self.baselines['cxd_classification_time'] * 2):
                recommendations.append({
                    'component': 'CXD Classification',
                    'priority': 'high',
                    'issue': 'Slow classification performance',
                    'recommendation': 'Consider optimizing classifier model or implementing classification caching',
                    'metrics': {
                        'current_avg_time': self.cxd_metrics.get_average_time(),
                        'baseline_time': self.baselines['cxd_classification_time']
                    }
                })
            
            # Cache recommendations
            if (self.cache_metrics.total_operations > 100 and
                self.cache_metrics.get_hit_rate() < 0.6):
                recommendations.append({
                    'component': 'Cache System',
                    'priority': 'medium',
                    'issue': 'Low cache hit rate',
                    'recommendation': 'Review caching strategy, increase cache size, or improve cache key design',
                    'metrics': {
                        'current_hit_rate': self.cache_metrics.get_hit_rate(),
                        'target_hit_rate': 0.8
                    }
                })
            
            # Search recommendations
            if (self.search_metrics.total_searches > 50 and
                self.search_metrics.get_average_time() > self.baselines['search_time'] * 2):
                recommendations.append({
                    'component': 'Search System', 
                    'priority': 'high',
                    'issue': 'Slow search performance',
                    'recommendation': 'Optimize search indexes, implement search result caching, or tune search algorithm',
                    'metrics': {
                        'current_avg_time': self.search_metrics.get_average_time(),
                        'baseline_time': self.baselines['search_time']
                    }
                })
            
            # Quality gate recommendations
            if (self.quality_gate_metrics.total_evaluations > 100 and
                self.quality_gate_metrics.get_pass_rate() < 0.7):
                recommendations.append({
                    'component': 'Quality Gate',
                    'priority': 'medium', 
                    'issue': 'High rejection rate',
                    'recommendation': 'Review quality thresholds or improve input data quality',
                    'metrics': {
                        'current_pass_rate': self.quality_gate_metrics.get_pass_rate(),
                        'target_pass_rate': 0.8
                    }
                })
        
        return recommendations
    
    def update_baselines(self):
        """Update performance baselines based on recent performance"""
        with self._lock:
            # Update baselines to 80th percentile of recent performance
            if len(self.cxd_metrics.classification_times) >= 50:
                times = sorted(list(self.cxd_metrics.classification_times))
                self.baselines['cxd_classification_time'] = times[int(len(times) * 0.8)]
            
            if len(self.cache_metrics.hit_times) >= 50:
                times = sorted(list(self.cache_metrics.hit_times))
                self.baselines['cache_hit_time'] = times[int(len(times) * 0.8)]
            
            if len(self.search_metrics.search_times) >= 50:
                times = sorted(list(self.search_metrics.search_times))
                self.baselines['search_time'] = times[int(len(times) * 0.8)]
            
            logger.info(f"Performance baselines updated: {self.baselines}")


# Context manager classes for operation tracking

class CXDClassificationTracker:
    def __init__(self, tracker: MemMimicPerformanceTracker, operation_id: str, content: str, start_time: float):
        self.tracker = tracker
        self.operation_id = operation_id
        self.content = content
        self.start_time = start_time
        self.pattern = None
        self.accuracy = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        self.tracker.record_cxd_classification(duration, success, self.pattern, self.accuracy)
    
    def set_result(self, pattern: Optional[str], accuracy: Optional[float] = None):
        self.pattern = pattern
        self.accuracy = accuracy


class CacheOperationTracker:
    def __init__(self, tracker: MemMimicPerformanceTracker, operation_id: str, operation_type: str, key: str, start_time: float):
        self.tracker = tracker
        self.operation_id = operation_id
        self.operation_type = operation_type
        self.key = key
        self.start_time = start_time
        self.hit = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.tracker.record_cache_operation(self.operation_type, duration, self.hit)
    
    def set_hit(self, hit: bool):
        self.hit = hit


class SearchOperationTracker:
    def __init__(self, tracker: MemMimicPerformanceTracker, operation_id: str, query: str, search_type: str, start_time: float):
        self.tracker = tracker
        self.operation_id = operation_id
        self.query = query
        self.search_type = search_type
        self.start_time = start_time
        self.results_count = 0
        self.relevance_score = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        self.tracker.record_search_operation(self.search_type, duration, success, self.results_count, self.relevance_score)
    
    def set_results(self, results_count: int, relevance_score: Optional[float] = None):
        self.results_count = results_count
        self.relevance_score = relevance_score


class ConsolidationTracker:
    def __init__(self, tracker: MemMimicPerformanceTracker, operation_id: str, start_time: float):
        self.tracker = tracker
        self.operation_id = operation_id
        self.start_time = start_time
        self.memories_processed = 0
        self.space_saved = 0
        self.duplicates_found = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        self.tracker.record_consolidation_operation(
            duration, success, self.memories_processed, self.space_saved, self.duplicates_found
        )
    
    def set_results(self, memories_processed: int, space_saved: int, duplicates_found: int):
        self.memories_processed = memories_processed
        self.space_saved = space_saved
        self.duplicates_found = duplicates_found


class QualityGateTracker:
    def __init__(self, tracker: MemMimicPerformanceTracker, operation_id: str, content: str, start_time: float):
        self.tracker = tracker
        self.operation_id = operation_id
        self.content = content
        self.start_time = start_time
        self.passed = False
        self.quality_score = None
        self.rejection_reason = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        # Consider it passed if no exception and explicitly set to passed
        self.passed = exc_type is None and self.passed
        self.tracker.record_quality_evaluation(duration, self.passed, self.quality_score, self.rejection_reason)
    
    def set_result(self, passed: bool, quality_score: Optional[float] = None, rejection_reason: Optional[str] = None):
        self.passed = passed
        self.quality_score = quality_score
        self.rejection_reason = rejection_reason


# Global performance tracker instance
_performance_tracker_instance = None
_tracker_lock = threading.Lock()


def get_performance_tracker() -> MemMimicPerformanceTracker:
    """Get global performance tracker instance (thread-safe singleton)"""
    global _performance_tracker_instance
    
    if _performance_tracker_instance is None:
        with _tracker_lock:
            if _performance_tracker_instance is None:
                _performance_tracker_instance = MemMimicPerformanceTracker()
    
    return _performance_tracker_instance