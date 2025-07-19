"""
Automatic optimization engine for active memory management system.

Provides intelligent optimization decisions, lifecycle coordination,
and automatic performance tuning based on usage patterns and metrics.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import statistics

from .interfaces import (
    LifecycleCoordinator, MemoryIndexingEngine, CacheManager, 
    DatabasePool, PerformanceMonitor, PerformanceSnapshot,
    MemoryQuery, MemoryStatus, ActiveMemoryError
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization engine"""
    # Lifecycle management
    importance_decay_rate: float = 0.1  # Daily decay rate
    archival_importance_threshold: float = 0.3
    deletion_importance_threshold: float = 0.1
    min_access_age_days: int = 7  # Minimum age before archival consideration
    
    # Performance optimization
    target_cache_hit_rate: float = 0.85
    target_query_time_ms: float = 50.0
    target_memory_utilization: float = 0.75
    
    # Optimization intervals
    lifecycle_optimization_hours: int = 6
    performance_optimization_hours: int = 1
    emergency_optimization_threshold: float = 0.95  # Memory utilization
    
    # Predictive settings
    prediction_window_hours: int = 24
    trend_analysis_points: int = 48  # Data points for trend analysis


class AutomaticOptimizationEngine(LifecycleCoordinator):
    """
    Intelligent optimization engine for automatic memory management.
    
    Features:
    - Automatic lifecycle management based on usage patterns
    - Performance optimization recommendations
    - Predictive capacity planning
    - Emergency optimization triggers
    - Learning from historical patterns
    """
    
    def __init__(self,
                 indexing_engine: MemoryIndexingEngine,
                 cache_manager: CacheManager,
                 database_pool: DatabasePool,
                 performance_monitor: PerformanceMonitor,
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize optimization engine.
        
        Args:
            indexing_engine: Memory indexing engine
            cache_manager: Cache management system
            database_pool: Database connection pool
            performance_monitor: Performance monitoring system
            config: Optimization configuration
        """
        self.indexing_engine = indexing_engine
        self.cache_manager = cache_manager
        self.database_pool = database_pool
        self.performance_monitor = performance_monitor
        self.config = config or OptimizationConfig()
        
        # Optimization state
        self._last_lifecycle_optimization = datetime.now()
        self._last_performance_optimization = datetime.now()
        self._optimization_history: List[Dict[str, Any]] = []
        
        # Memory access patterns
        self._access_patterns: Dict[str, List[datetime]] = {}
        self._importance_scores: Dict[str, float] = {}
        self._memory_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Learning data
        self._successful_optimizations: List[Dict[str, Any]] = []
        self._optimization_effectiveness: Dict[str, float] = {}
        
        # Background optimization
        self._optimization_thread = None
        self._stop_optimization = threading.Event()
        self._lock = threading.RLock()
        
        # Start background optimization
        self._start_background_optimization()
        
        logger.info("AutomaticOptimizationEngine initialized")
    
    def evaluate_memory_importance(self, memory_id: str) -> float:
        """
        Evaluate current importance score for memory.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Current importance score (0.0-1.0)
        """
        try:
            with self._lock:
                # Get base importance score
                base_importance = self._importance_scores.get(memory_id, 0.5)
                
                # Get access patterns
                access_times = self._access_patterns.get(memory_id, [])
                if not access_times:
                    return base_importance * 0.5  # Reduce if never accessed
                
                # Calculate recency factor
                last_access = max(access_times)
                days_since_access = (datetime.now() - last_access).days
                recency_factor = max(0.1, 1.0 - (days_since_access * self.config.importance_decay_rate))
                
                # Calculate frequency factor
                access_count = len(access_times)
                frequency_factor = min(1.0, access_count / 10.0)  # Normalize to 10 accesses
                
                # Calculate temporal distribution factor
                if len(access_times) > 1:
                    recent_accesses = [
                        t for t in access_times 
                        if (datetime.now() - t).days <= 7
                    ]
                    temporal_factor = len(recent_accesses) / len(access_times)
                else:
                    temporal_factor = 0.5
                
                # Get memory metadata factors
                metadata = self._memory_metadata.get(memory_id, {})
                type_factor = self._get_type_importance_factor(metadata.get('type', 'unknown'))
                confidence_factor = metadata.get('confidence', 0.5)
                
                # Combine factors
                importance = (
                    base_importance * 0.3 +
                    recency_factor * 0.25 +
                    frequency_factor * 0.2 +
                    temporal_factor * 0.15 +
                    type_factor * 0.07 +
                    confidence_factor * 0.03
                )
                
                return min(1.0, max(0.0, importance))
                
        except Exception as e:
            logger.error(f"Failed to evaluate importance for {memory_id}: {e}")
            return 0.5  # Default importance
    
    def suggest_archival_candidates(self, max_candidates: int = 100) -> List[str]:
        """
        Suggest memories for archival based on usage patterns.
        
        Args:
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of memory IDs recommended for archival
        """
        try:
            candidates = []
            
            with self._lock:
                # Evaluate all memories
                for memory_id in self._importance_scores.keys():
                    importance = self.evaluate_memory_importance(memory_id)
                    
                    # Check archival criteria
                    if importance < self.config.archival_importance_threshold:
                        # Check minimum age
                        metadata = self._memory_metadata.get(memory_id, {})
                        created_at = metadata.get('created_at')
                        
                        if created_at:
                            age_days = (datetime.now() - created_at).days
                            if age_days >= self.config.min_access_age_days:
                                candidates.append((memory_id, importance))
                
                # Sort by importance (lowest first)
                candidates.sort(key=lambda x: x[1])
                
                # Return memory IDs only
                return [memory_id for memory_id, _ in candidates[:max_candidates]]
                
        except Exception as e:
            logger.error(f"Failed to suggest archival candidates: {e}")
            return []
    
    def suggest_deletion_candidates(self, max_candidates: int = 50) -> List[str]:
        """
        Suggest memories for deletion based on age and importance.
        
        Args:
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of memory IDs recommended for deletion
        """
        try:
            candidates = []
            
            with self._lock:
                # Find very low importance, old memories
                for memory_id in self._importance_scores.keys():
                    importance = self.evaluate_memory_importance(memory_id)
                    
                    # Check deletion criteria
                    if importance < self.config.deletion_importance_threshold:
                        metadata = self._memory_metadata.get(memory_id, {})
                        created_at = metadata.get('created_at')
                        
                        if created_at:
                            age_days = (datetime.now() - created_at).days
                            # Require longer age for deletion
                            if age_days >= self.config.min_access_age_days * 4:
                                # Additional safety check - no recent access
                                access_times = self._access_patterns.get(memory_id, [])
                                if not access_times or (datetime.now() - max(access_times)).days > 30:
                                    candidates.append((memory_id, importance, age_days))
                
                # Sort by importance and age
                candidates.sort(key=lambda x: (x[1], -x[2]))  # Low importance, high age first
                
                return [memory_id for memory_id, _, _ in candidates[:max_candidates]]
                
        except Exception as e:
            logger.error(f"Failed to suggest deletion candidates: {e}")
            return []
    
    def optimize_memory_distribution(self) -> Dict[str, Any]:
        """
        Optimize memory distribution across tiers.
        
        Returns:
            Dictionary with optimization results
        """
        try:
            optimization_start = time.time()
            results = {
                'archival_suggestions': 0,
                'deletion_suggestions': 0,
                'cache_optimizations': 0,
                'index_optimizations': 0,
                'performance_improvements': [],
            }
            
            # Get current performance snapshot
            current_performance = self.performance_monitor.get_current_snapshot()
            
            # Suggest archival candidates
            archival_candidates = self.suggest_archival_candidates()
            results['archival_suggestions'] = len(archival_candidates)
            
            # Suggest deletion candidates if needed
            if current_performance.total_memory_mb > self.config.target_memory_utilization * 1024:
                deletion_candidates = self.suggest_deletion_candidates()
                results['deletion_suggestions'] = len(deletion_candidates)
            
            # Optimize cache if hit rate is low
            if current_performance.cache_hit_rate < self.config.target_cache_hit_rate:
                cache_optimization = self._optimize_cache()
                results['cache_optimizations'] = cache_optimization.get('optimizations_applied', 0)
                if cache_optimization.get('improvements'):
                    results['performance_improvements'].extend(cache_optimization['improvements'])
            
            # Optimize indexes if query time is high
            if current_performance.avg_query_time_ms > self.config.target_query_time_ms:
                index_optimization = self._optimize_indexes()
                results['index_optimizations'] = index_optimization.get('optimizations_applied', 0)
                if index_optimization.get('improvements'):
                    results['performance_improvements'].extend(index_optimization['improvements'])
            
            # Record optimization
            optimization_time = time.time() - optimization_start
            optimization_record = {
                'timestamp': datetime.now(),
                'type': 'memory_distribution',
                'duration_seconds': optimization_time,
                'results': results,
                'performance_before': current_performance.__dict__,
            }
            
            self._optimization_history.append(optimization_record)
            
            # Evaluate effectiveness after a delay (simplified)
            self._evaluate_optimization_effectiveness(optimization_record)
            
            logger.info(f"Memory distribution optimization completed in {optimization_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Memory distribution optimization failed: {e}")
            raise ActiveMemoryError(f"Optimization failed: {e}")
    
    def predict_memory_usage(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """
        Predict future memory usage patterns.
        
        Args:
            hours_ahead: Hours to predict ahead
            
        Returns:
            Dictionary with usage predictions
        """
        try:
            # Get historical data
            historical_data = self.performance_monitor.get_historical_data(
                hours=self.config.prediction_window_hours
            )
            
            if len(historical_data) < 5:
                return {'error': 'Insufficient historical data for prediction'}
            
            # Extract time series
            timestamps = [snapshot.timestamp for snapshot in historical_data]
            memory_usage = [snapshot.total_memory_mb for snapshot in historical_data]
            query_rates = [snapshot.queries_per_second for snapshot in historical_data]
            cache_hit_rates = [snapshot.cache_hit_rate for snapshot in historical_data]
            
            # Simple linear trend prediction
            def predict_trend(values: List[float], hours_ahead: int) -> float:
                if len(values) < 2:
                    return values[-1] if values else 0.0
                
                # Calculate hourly trend
                time_span_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
                value_change = values[-1] - values[0]
                hourly_trend = value_change / time_span_hours if time_span_hours > 0 else 0.0
                
                # Project forward
                predicted_value = values[-1] + (hourly_trend * hours_ahead)
                return max(0.0, predicted_value)
            
            # Make predictions
            predicted_memory = predict_trend(memory_usage, hours_ahead)
            predicted_query_rate = predict_trend(query_rates, hours_ahead)
            predicted_cache_hit_rate = max(0.0, min(1.0, predict_trend(cache_hit_rates, hours_ahead)))
            
            # Capacity analysis
            max_memory_capacity = 2048  # Assume 2GB max
            utilization_prediction = predicted_memory / max_memory_capacity
            
            # Risk assessment
            risks = []
            if utilization_prediction > 0.9:
                risks.append('High memory utilization predicted')
            if predicted_query_rate > 100:
                risks.append('High query load predicted')
            if predicted_cache_hit_rate < 0.7:
                risks.append('Poor cache performance predicted')
            
            # Recommendations
            recommendations = []
            if utilization_prediction > 0.8:
                recommendations.append('Consider memory cleanup or capacity increase')
            if predicted_cache_hit_rate < self.config.target_cache_hit_rate:
                recommendations.append('Optimize cache configuration')
            if predicted_query_rate > current_capacity_estimate():
                recommendations.append('Consider query optimization or load balancing')
            
            return {
                'prediction_horizon_hours': hours_ahead,
                'current_memory_mb': memory_usage[-1] if memory_usage else 0,
                'predicted_memory_mb': predicted_memory,
                'memory_utilization_prediction': utilization_prediction,
                'predicted_query_rate': predicted_query_rate,
                'predicted_cache_hit_rate': predicted_cache_hit_rate,
                'risks': risks,
                'recommendations': recommendations,
                'confidence': self._calculate_prediction_confidence(historical_data),
                'data_points_used': len(historical_data),
            }
            
        except Exception as e:
            logger.error(f"Memory usage prediction failed: {e}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def record_memory_access(self, memory_id: str, metadata: Dict[str, Any] = None) -> None:
        """
        Record memory access for pattern learning.
        
        Args:
            memory_id: Memory identifier
            metadata: Optional memory metadata
        """
        try:
            with self._lock:
                # Record access time
                if memory_id not in self._access_patterns:
                    self._access_patterns[memory_id] = []
                
                self._access_patterns[memory_id].append(datetime.now())
                
                # Keep only recent accesses (last 30 days)
                cutoff_time = datetime.now() - timedelta(days=30)
                self._access_patterns[memory_id] = [
                    access_time for access_time in self._access_patterns[memory_id]
                    if access_time >= cutoff_time
                ]
                
                # Update metadata
                if metadata:
                    self._memory_metadata[memory_id] = metadata
                    if 'importance_score' in metadata:
                        self._importance_scores[memory_id] = metadata['importance_score']
                
        except Exception as e:
            logger.error(f"Failed to record memory access for {memory_id}: {e}")
    
    def _get_type_importance_factor(self, memory_type: str) -> float:
        """Get importance factor based on memory type"""
        type_weights = {
            'synthetic_wisdom': 1.0,
            'milestone': 0.9,
            'consciousness_evolution': 0.95,
            'reflection': 0.7,
            'interaction': 0.5,
            'project_info': 0.6,
            'system': 0.8,
            'configuration': 0.3,
            'temporary': 0.1,
        }
        return type_weights.get(memory_type.lower(), 0.5)
    
    def _optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache configuration and performance"""
        try:
            optimization_results = {
                'optimizations_applied': 0,
                'improvements': [],
            }
            
            # Get cache statistics
            cache_stats = self.cache_manager.get_stats()
            
            # Force eviction if memory pressure is high
            if cache_stats.get('memory_utilization', 0) > 0.9:
                evicted = self.cache_manager.force_eviction(0.7)
                if evicted > 0:
                    optimization_results['optimizations_applied'] += 1
                    optimization_results['improvements'].append(
                        f'Evicted {evicted} entries to reduce memory pressure'
                    )
            
            # Clean expired entries
            expired = self.cache_manager.evict_expired()
            if expired > 0:
                optimization_results['optimizations_applied'] += 1
                optimization_results['improvements'].append(
                    f'Removed {expired} expired cache entries'
                )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return {'optimizations_applied': 0, 'improvements': [], 'error': str(e)}
    
    def _optimize_indexes(self) -> Dict[str, Any]:
        """Optimize indexing engine performance"""
        try:
            optimization_results = {
                'optimizations_applied': 0,
                'improvements': [],
            }
            
            # Run index optimization
            index_optimization = self.indexing_engine.optimize_indexes()
            
            if 'optimization_time_ms' in index_optimization:
                optimization_results['optimizations_applied'] += 1
                optimization_results['improvements'].append(
                    f'Optimized indexes in {index_optimization["optimization_time_ms"]:.2f}ms'
                )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            return {'optimizations_applied': 0, 'improvements': [], 'error': str(e)}
    
    def _calculate_prediction_confidence(self, historical_data: List[PerformanceSnapshot]) -> float:
        """Calculate confidence level for predictions based on data quality"""
        if len(historical_data) < 5:
            return 0.3  # Low confidence
        
        # Check data consistency (simplified)
        memory_values = [snapshot.total_memory_mb for snapshot in historical_data]
        if len(set(memory_values)) == 1:
            return 0.5  # Medium confidence - static data
        
        # Check for sufficient data points
        if len(historical_data) >= 24:  # At least 24 hours of data
            return 0.8  # High confidence
        elif len(historical_data) >= 12:
            return 0.7  # Good confidence
        else:
            return 0.5  # Medium confidence
    
    def _evaluate_optimization_effectiveness(self, optimization_record: Dict[str, Any]) -> None:
        """Evaluate how effective an optimization was (simplified)"""
        # This would compare performance before/after optimization
        # For now, just record that it was performed
        self._successful_optimizations.append(optimization_record)
        
        # Keep only recent optimizations
        if len(self._successful_optimizations) > 100:
            self._successful_optimizations = self._successful_optimizations[-100:]
    
    def _start_background_optimization(self) -> None:
        """Start background optimization thread"""
        def optimization_worker():
            while not self._stop_optimization.wait(3600):  # Check every hour
                try:
                    current_time = datetime.now()
                    
                    # Check if lifecycle optimization is due
                    lifecycle_hours = (current_time - self._last_lifecycle_optimization).total_seconds() / 3600
                    if lifecycle_hours >= self.config.lifecycle_optimization_hours:
                        self.optimize_memory_distribution()
                        self._last_lifecycle_optimization = current_time
                    
                    # Check for emergency optimization
                    current_performance = self.performance_monitor.get_current_snapshot()
                    memory_utilization = current_performance.total_memory_mb / 2048  # Assume 2GB limit
                    
                    if memory_utilization >= self.config.emergency_optimization_threshold:
                        logger.warning("Emergency optimization triggered due to high memory usage")
                        self.optimize_memory_distribution()
                    
                except Exception as e:
                    logger.error(f"Background optimization failed: {e}")
        
        self._optimization_thread = threading.Thread(target=optimization_worker, daemon=True)
        self._optimization_thread.start()
        logger.debug("Background optimization thread started")
    
    def shutdown(self) -> None:
        """Shutdown optimization engine"""
        try:
            self._stop_optimization.set()
            if self._optimization_thread and self._optimization_thread.is_alive():
                self._optimization_thread.join(timeout=5.0)
            
            logger.info("Optimization engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Optimization engine shutdown failed: {e}")


def current_capacity_estimate() -> float:
    """Estimate current system capacity for queries per second"""
    # Simplified capacity estimate
    return 50.0  # queries per second


def create_optimization_engine(
    indexing_engine: MemoryIndexingEngine,
    cache_manager: CacheManager,
    database_pool: DatabasePool,
    performance_monitor: PerformanceMonitor,
    config: Optional[OptimizationConfig] = None
) -> LifecycleCoordinator:
    """
    Factory function to create optimization engine.
    
    Args:
        indexing_engine: Memory indexing engine
        cache_manager: Cache management system  
        database_pool: Database connection pool
        performance_monitor: Performance monitoring system
        config: Optimization configuration
        
    Returns:
        LifecycleCoordinator instance
    """
    return AutomaticOptimizationEngine(
        indexing_engine=indexing_engine,
        cache_manager=cache_manager,
        database_pool=database_pool,
        performance_monitor=performance_monitor,
        config=config
    )