"""
ML-Driven Predictive Cache Warming System for MemMimic.

Implements intelligent cache warming based on usage pattern analysis, temporal predictions,
and user behavior learning to optimize cache hit rates and reduce response times.
"""

import asyncio
import logging
import pickle
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from .cache_manager import CacheManager, LRUMemoryCache
from .interfaces import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class CacheAccessPattern:
    """Represents a cache access pattern for ML analysis."""
    key: str
    access_times: List[datetime] = field(default_factory=list)
    access_count: int = 0
    last_access: Optional[datetime] = None
    data_size: int = 0
    access_frequency: float = 0.0  # accesses per hour
    temporal_pattern: str = "unknown"  # hourly, daily, weekly, irregular
    content_hash: str = ""
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictiveCacheConfig:
    """Configuration for predictive cache warming system."""
    # Learning parameters
    learning_window_hours: int = 168  # 1 week
    min_accesses_for_prediction: int = 3
    pattern_analysis_interval_minutes: int = 60
    
    # Warming parameters
    warm_ahead_hours: int = 2
    max_warm_items: int = 100
    warm_confidence_threshold: float = 0.7
    
    # Model parameters
    temporal_clusters: int = 5
    prediction_horizon_hours: int = 24
    model_retrain_hours: int = 24
    
    # Cache optimization
    enable_preemptive_eviction: bool = True
    enable_content_based_warming: bool = True
    enable_user_behavior_learning: bool = True
    
    # Performance tuning
    max_pattern_history: int = 10000
    pattern_cleanup_hours: int = 72
    background_processing_interval_seconds: int = 30


@dataclass
class WarmingPrediction:
    """Prediction for cache warming."""
    key: str
    predicted_access_time: datetime
    confidence: float
    reason: str
    priority: float
    data_loader: Optional[callable] = None


class TemporalPatternAnalyzer:
    """Analyzes temporal patterns in cache access for predictive warming."""
    
    def __init__(self, config: PredictiveCacheConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=config.temporal_clusters, random_state=42)
        self.pattern_labels = {}
        self.is_fitted = False
    
    def analyze_pattern(self, access_times: List[datetime]) -> Tuple[str, float]:
        """
        Analyze temporal pattern of access times.
        
        Args:
            access_times: List of access timestamps
            
        Returns:
            Tuple of (pattern_type, regularity_score)
        """
        if len(access_times) < 2:
            return "insufficient_data", 0.0
        
        # Convert to hour-of-day and day-of-week features
        hours = [t.hour for t in access_times]
        days = [t.weekday() for t in access_times]
        
        # Analyze hour-of-day pattern
        hour_distribution = np.bincount(hours, minlength=24)
        hour_entropy = self._calculate_entropy(hour_distribution)
        
        # Analyze day-of-week pattern
        day_distribution = np.bincount(days, minlength=7)
        day_entropy = self._calculate_entropy(day_distribution)
        
        # Analyze intervals between accesses
        intervals = []
        for i in range(1, len(access_times)):
            interval = (access_times[i] - access_times[i-1]).total_seconds() / 3600  # hours
            intervals.append(interval)
        
        if not intervals:
            return "single_access", 0.0
        
        interval_std = np.std(intervals)
        interval_mean = np.mean(intervals)
        
        # Classify pattern based on entropy and interval regularity
        regularity_score = 1.0 / (1.0 + interval_std / max(interval_mean, 1.0))
        
        if hour_entropy < 2.0 and regularity_score > 0.7:
            if interval_mean <= 24:
                return "daily", regularity_score
            elif interval_mean <= 168:
                return "weekly", regularity_score
            else:
                return "monthly", regularity_score
        elif hour_entropy < 3.0:
            return "time_based", regularity_score * 0.8
        else:
            return "irregular", regularity_score * 0.5
    
    def _calculate_entropy(self, distribution: np.ndarray) -> float:
        """Calculate Shannon entropy of a distribution."""
        # Normalize and avoid log(0)
        prob = distribution / max(distribution.sum(), 1)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))
    
    def predict_next_access(self, pattern: CacheAccessPattern, 
                           current_time: datetime) -> Optional[Tuple[datetime, float]]:
        """
        Predict next access time based on pattern analysis.
        
        Args:
            pattern: Cache access pattern
            current_time: Current timestamp
            
        Returns:
            Tuple of (predicted_time, confidence) or None if unpredictable
        """
        if len(pattern.access_times) < self.config.min_accesses_for_prediction:
            return None
        
        pattern_type, regularity = self.analyze_pattern(pattern.access_times)
        
        if regularity < 0.3:  # Too irregular to predict
            return None
        
        # Get recent access times for prediction
        recent_accesses = [
            t for t in pattern.access_times 
            if (current_time - t).total_seconds() / 3600 <= self.config.learning_window_hours
        ]
        
        if len(recent_accesses) < 2:
            return None
        
        if pattern_type == "daily":
            return self._predict_daily_pattern(recent_accesses, current_time, regularity)
        elif pattern_type == "weekly":
            return self._predict_weekly_pattern(recent_accesses, current_time, regularity)
        elif pattern_type == "time_based":
            return self._predict_time_based_pattern(recent_accesses, current_time, regularity)
        elif pattern_type == "irregular":
            return self._predict_statistical_pattern(recent_accesses, current_time, regularity)
        else:
            return None
    
    def _predict_daily_pattern(self, accesses: List[datetime], 
                              current_time: datetime, regularity: float) -> Tuple[datetime, float]:
        """Predict based on daily access pattern."""
        # Find most common hour
        hours = [t.hour for t in accesses]
        most_common_hour = max(set(hours), key=hours.count)
        
        # Predict next occurrence
        next_day = current_time.replace(hour=most_common_hour, minute=0, second=0, microsecond=0)
        if next_day <= current_time:
            next_day += timedelta(days=1)
        
        confidence = regularity * 0.9
        return next_day, confidence
    
    def _predict_weekly_pattern(self, accesses: List[datetime], 
                               current_time: datetime, regularity: float) -> Tuple[datetime, float]:
        """Predict based on weekly access pattern."""
        # Find most common day of week and hour
        day_hours = [(t.weekday(), t.hour) for t in accesses]
        most_common_day_hour = max(set(day_hours), key=day_hours.count)
        target_weekday, target_hour = most_common_day_hour
        
        # Find next occurrence
        days_until_target = (target_weekday - current_time.weekday()) % 7
        if days_until_target == 0 and current_time.hour >= target_hour:
            days_until_target = 7
        
        next_access = current_time.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        next_access += timedelta(days=days_until_target)
        
        confidence = regularity * 0.8
        return next_access, confidence
    
    def _predict_time_based_pattern(self, accesses: List[datetime], 
                                   current_time: datetime, regularity: float) -> Tuple[datetime, float]:
        """Predict based on general time-based pattern."""
        # Calculate average interval
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds() / 3600
            intervals.append(interval)
        
        avg_interval = np.mean(intervals)
        last_access = max(accesses)
        
        predicted_time = last_access + timedelta(hours=avg_interval)
        confidence = regularity * 0.6
        
        return predicted_time, confidence
    
    def _predict_statistical_pattern(self, accesses: List[datetime], 
                                    current_time: datetime, regularity: float) -> Tuple[datetime, float]:
        """Predict using statistical methods for irregular patterns."""
        if len(accesses) < 3:
            return None
        
        # Use linear regression on time series
        timestamps = [(t - accesses[0]).total_seconds() for t in accesses]
        X = np.array(range(len(timestamps))).reshape(-1, 1)
        y = np.array(timestamps)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next timestamp
        next_index = len(accesses)
        predicted_offset = model.predict([[next_index]])[0]
        predicted_time = accesses[0] + timedelta(seconds=predicted_offset)
        
        # Confidence based on model accuracy and regularity
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        confidence = regularity * 0.4 * max(0, 1 - mae / np.mean(y))
        
        return predicted_time, max(0.1, confidence)


class ContentSimilarityAnalyzer:
    """Analyzes content similarity for cache warming recommendations."""
    
    def __init__(self, config: PredictiveCacheConfig):
        self.config = config
        self.content_clusters = {}
        self.similarity_threshold = 0.7
    
    def analyze_content_similarity(self, patterns: List[CacheAccessPattern]) -> Dict[str, List[str]]:
        """
        Analyze content similarity between cache items for warming recommendations.
        
        Args:
            patterns: List of cache access patterns
            
        Returns:
            Dictionary mapping cache keys to similar keys
        """
        similarity_groups = defaultdict(list)
        
        # Group by content characteristics
        content_features = {}
        for pattern in patterns:
            if pattern.content_hash:
                # Simple content-based grouping
                # In a real implementation, this would use semantic embeddings
                content_type = self._infer_content_type(pattern)
                size_bucket = self._get_size_bucket(pattern.data_size)
                feature_key = f"{content_type}_{size_bucket}"
                
                if feature_key not in content_features:
                    content_features[feature_key] = []
                content_features[feature_key].append(pattern.key)
        
        # Create similarity groups
        for feature_key, keys in content_features.items():
            if len(keys) > 1:
                for key in keys:
                    similarity_groups[key].extend([k for k in keys if k != key])
        
        return dict(similarity_groups)
    
    def _infer_content_type(self, pattern: CacheAccessPattern) -> str:
        """Infer content type from pattern metadata."""
        key_lower = pattern.key.lower()
        
        if "embedding" in key_lower or "vector" in key_lower:
            return "embedding"
        elif "search" in key_lower or "query" in key_lower:
            return "search_result"
        elif "classification" in key_lower or "cxd" in key_lower:
            return "classification"
        elif "memory" in key_lower:
            return "memory_data"
        else:
            return "unknown"
    
    def _get_size_bucket(self, size: int) -> str:
        """Get size bucket for content grouping."""
        if size < 1024:
            return "small"
        elif size < 10240:
            return "medium"
        elif size < 102400:
            return "large"
        else:
            return "xlarge"


class PredictiveCacheWarmer:
    """
    ML-driven predictive cache warming system.
    
    Features:
    - Temporal pattern analysis and prediction
    - Content-based similarity warming
    - User behavior learning and adaptation
    - Performance-optimized warming strategies
    """
    
    def __init__(self, 
                 cache_manager: CacheManager,
                 performance_monitor: PerformanceMonitor,
                 config: Optional[PredictiveCacheConfig] = None):
        """
        Initialize predictive cache warmer.
        
        Args:
            cache_manager: Cache management system
            performance_monitor: Performance monitoring system
            config: Predictive cache configuration
        """
        self.cache_manager = cache_manager
        self.performance_monitor = performance_monitor
        self.config = config or PredictiveCacheConfig()
        
        # Analysis components
        self.temporal_analyzer = TemporalPatternAnalyzer(self.config)
        self.content_analyzer = ContentSimilarityAnalyzer(self.config)
        
        # Pattern tracking
        self.access_patterns: Dict[str, CacheAccessPattern] = {}
        self.warming_history: Dict[str, List[datetime]] = defaultdict(list)
        self.pattern_lock = threading.RLock()
        
        # ML models and data
        self.usage_predictor = None
        self.content_similarity_map = {}
        self.user_behavior_models = {}
        
        # Performance tracking
        self.warming_stats = {
            'predictions_made': 0,
            'successful_predictions': 0,
            'cache_hits_from_warming': 0,
            'warming_overhead_ms': 0.0,
            'patterns_learned': 0,
        }
        
        # Background processing
        self._background_thread = None
        self._shutdown_event = threading.Event()
        self._start_background_processing()
        
        logger.info("PredictiveCacheWarmer initialized")
    
    def record_access(self, cache_key: str, 
                     data_size: int = 0,
                     user_context: Optional[Dict[str, Any]] = None):
        """
        Record a cache access for pattern learning.
        
        Args:
            cache_key: Cache key that was accessed
            data_size: Size of the cached data in bytes
            user_context: Optional user context information
        """
        current_time = datetime.now()
        
        with self.pattern_lock:
            if cache_key not in self.access_patterns:
                self.access_patterns[cache_key] = CacheAccessPattern(
                    key=cache_key,
                    data_size=data_size,
                    user_context=user_context or {}
                )
            
            pattern = self.access_patterns[cache_key]
            pattern.access_times.append(current_time)
            pattern.access_count += 1
            pattern.last_access = current_time
            
            # Update frequency calculation
            if len(pattern.access_times) > 1:
                time_span = (pattern.access_times[-1] - pattern.access_times[0]).total_seconds() / 3600
                pattern.access_frequency = len(pattern.access_times) / max(time_span, 0.1)
            
            # Limit access time history
            max_history = min(1000, self.config.max_pattern_history // len(self.access_patterns))
            if len(pattern.access_times) > max_history:
                pattern.access_times = pattern.access_times[-max_history:]
        
        # Update warming stats
        self.warming_stats['patterns_learned'] += 1
    
    def predict_cache_needs(self, 
                          hours_ahead: int = None,
                          max_predictions: int = None) -> List[WarmingPrediction]:
        """
        Predict future cache needs based on learned patterns.
        
        Args:
            hours_ahead: Hours to predict ahead (uses config default if None)
            max_predictions: Maximum predictions to return
            
        Returns:
            List of warming predictions sorted by priority
        """
        hours_ahead = hours_ahead or self.config.warm_ahead_hours
        max_predictions = max_predictions or self.config.max_warm_items
        current_time = datetime.now()
        predictions = []
        
        with self.pattern_lock:
            for key, pattern in self.access_patterns.items():
                # Skip patterns with insufficient data
                if len(pattern.access_times) < self.config.min_accesses_for_prediction:
                    continue
                
                # Get temporal prediction
                temporal_pred = self.temporal_analyzer.predict_next_access(pattern, current_time)
                if temporal_pred:
                    predicted_time, confidence = temporal_pred
                    
                    # Check if prediction is within our warming window
                    time_until_access = (predicted_time - current_time).total_seconds() / 3600
                    if 0 < time_until_access <= hours_ahead and confidence >= self.config.warm_confidence_threshold:
                        
                        # Calculate priority based on multiple factors
                        priority = self._calculate_warming_priority(
                            pattern, confidence, time_until_access
                        )
                        
                        prediction = WarmingPrediction(
                            key=key,
                            predicted_access_time=predicted_time,
                            confidence=confidence,
                            reason=f"temporal_pattern_{pattern.temporal_pattern}",
                            priority=priority
                        )
                        predictions.append(prediction)
                
                # Add content-based predictions
                content_predictions = self._get_content_based_predictions(
                    pattern, current_time, hours_ahead
                )
                predictions.extend(content_predictions)
        
        # Sort by priority and limit results
        predictions.sort(key=lambda p: p.priority, reverse=True)
        self.warming_stats['predictions_made'] += len(predictions[:max_predictions])
        
        return predictions[:max_predictions]
    
    def warm_cache_proactively(self, 
                              predictions: Optional[List[WarmingPrediction]] = None) -> Dict[str, Any]:
        """
        Proactively warm cache based on predictions.
        
        Args:
            predictions: Optional list of predictions (will generate if None)
            
        Returns:
            Dictionary with warming results
        """
        if predictions is None:
            predictions = self.predict_cache_needs()
        
        warming_start = time.perf_counter()
        warming_results = {
            'items_warmed': 0,
            'warming_time_ms': 0.0,
            'failed_warmings': 0,
            'cache_space_used_mb': 0.0,
            'predictions_processed': len(predictions),
        }
        
        for prediction in predictions:
            try:
                # Check if item is already in cache
                if self._is_in_cache(prediction.key):
                    continue
                
                # Check cache capacity before warming
                cache_stats = self.cache_manager.get_stats()
                if cache_stats.get('memory_utilization', 0) > 0.85:
                    # Cache is nearly full, skip warming
                    break
                
                # Attempt to warm cache
                if self._warm_cache_item(prediction):
                    warming_results['items_warmed'] += 1
                    
                    # Estimate cache space used
                    pattern = self.access_patterns.get(prediction.key)
                    if pattern:
                        warming_results['cache_space_used_mb'] += pattern.data_size / (1024 * 1024)
                else:
                    warming_results['failed_warmings'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to warm cache for key {prediction.key}: {e}")
                warming_results['failed_warmings'] += 1
        
        warming_time = (time.perf_counter() - warming_start) * 1000
        warming_results['warming_time_ms'] = warming_time
        self.warming_stats['warming_overhead_ms'] += warming_time
        
        logger.debug(f"Cache warming completed: {warming_results}")
        return warming_results
    
    def _calculate_warming_priority(self, 
                                   pattern: CacheAccessPattern,
                                   confidence: float,
                                   time_until_access: float) -> float:
        """Calculate priority score for cache warming."""
        # Base priority from confidence
        priority = confidence
        
        # Boost priority for frequently accessed items
        frequency_boost = min(1.0, pattern.access_frequency / 10.0)  # Normalize to 10 accesses/hour
        priority += frequency_boost * 0.3
        
        # Boost priority for items accessed soon
        urgency_boost = 1.0 / max(time_until_access, 0.1)  # Inverse of time until access
        priority += min(urgency_boost, 1.0) * 0.2
        
        # Boost priority for large items (expensive to compute)
        size_boost = min(1.0, pattern.data_size / (1024 * 1024))  # Normalize to 1MB
        priority += size_boost * 0.1
        
        return min(2.0, priority)  # Cap at 2.0
    
    def _get_content_based_predictions(self, 
                                     pattern: CacheAccessPattern,
                                     current_time: datetime,
                                     hours_ahead: int) -> List[WarmingPrediction]:
        """Get predictions based on content similarity."""
        predictions = []
        
        if not self.config.enable_content_based_warming:
            return predictions
        
        # Get similar content keys
        similar_keys = self.content_similarity_map.get(pattern.key, [])
        
        for similar_key in similar_keys:
            if similar_key in self.access_patterns:
                similar_pattern = self.access_patterns[similar_key]
                
                # If original pattern is accessed frequently, similar content might be too
                if (pattern.access_frequency > 1.0 and  # At least once per hour
                    similar_pattern.access_frequency > 0.5):
                    
                    # Predict similar content will be accessed soon
                    predicted_time = current_time + timedelta(hours=hours_ahead * 0.5)
                    confidence = 0.6 * (pattern.access_frequency / 10.0)
                    
                    prediction = WarmingPrediction(
                        key=similar_key,
                        predicted_access_time=predicted_time,
                        confidence=confidence,
                        reason="content_similarity",
                        priority=confidence * 0.8  # Slightly lower priority than temporal
                    )
                    predictions.append(prediction)
        
        return predictions
    
    def _is_in_cache(self, cache_key: str) -> bool:
        """Check if item is already in cache."""
        try:
            result = self.cache_manager.get(cache_key)
            return result is not None
        except:
            return False
    
    def _warm_cache_item(self, prediction: WarmingPrediction) -> bool:
        """
        Warm a specific cache item.
        
        Args:
            prediction: Warming prediction
            
        Returns:
            True if warming succeeded, False otherwise
        """
        try:
            # This is a placeholder - in reality, you would need to implement
            # the data loading logic specific to your cache content types
            
            # For now, we'll just mark that we would warm this item
            # In a real implementation, this would:
            # 1. Identify the data source for the cache key
            # 2. Load the data using the appropriate loader
            # 3. Store it in the cache with appropriate TTL
            
            if prediction.data_loader:
                data = prediction.data_loader()
                self.cache_manager.put(prediction.key, data)
                return True
            else:
                # Can't warm without a data loader
                return False
                
        except Exception as e:
            logger.error(f"Cache warming failed for {prediction.key}: {e}")
            return False
    
    def _start_background_processing(self):
        """Start background thread for pattern analysis and cache warming."""
        def background_worker():
            while not self._shutdown_event.wait(self.config.background_processing_interval_seconds):
                try:
                    # Periodic pattern analysis
                    if datetime.now().minute % self.config.pattern_analysis_interval_minutes == 0:
                        self._analyze_patterns()
                        self._update_content_similarity()
                    
                    # Proactive cache warming
                    predictions = self.predict_cache_needs()
                    if predictions:
                        self.warm_cache_proactively(predictions)
                    
                    # Cleanup old patterns
                    self._cleanup_old_patterns()
                    
                except Exception as e:
                    logger.error(f"Background processing error: {e}")
        
        self._background_thread = threading.Thread(target=background_worker, daemon=True)
        self._background_thread.start()
        logger.debug("Predictive cache background processing started")
    
    def _analyze_patterns(self):
        """Analyze access patterns to update temporal classifications."""
        with self.pattern_lock:
            for pattern in self.access_patterns.values():
                if len(pattern.access_times) >= 3:
                    pattern_type, regularity = self.temporal_analyzer.analyze_pattern(
                        pattern.access_times
                    )
                    pattern.temporal_pattern = pattern_type
    
    def _update_content_similarity(self):
        """Update content similarity mappings."""
        with self.pattern_lock:
            patterns = list(self.access_patterns.values())
        
        self.content_similarity_map = self.content_analyzer.analyze_content_similarity(patterns)
    
    def _cleanup_old_patterns(self):
        """Clean up old access patterns to prevent memory bloat."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.pattern_cleanup_hours)
        
        with self.pattern_lock:
            keys_to_remove = []
            for key, pattern in self.access_patterns.items():
                if pattern.last_access and pattern.last_access < cutoff_time:
                    if pattern.access_count < 5:  # Only remove infrequently accessed patterns
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.access_patterns[key]
                if key in self.content_similarity_map:
                    del self.content_similarity_map[key]
        
        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} old access patterns")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about predictive cache warming."""
        with self.pattern_lock:
            total_patterns = len(self.access_patterns)
            active_patterns = sum(
                1 for p in self.access_patterns.values() 
                if p.last_access and (datetime.now() - p.last_access).hours < 24
            )
            
            pattern_types = defaultdict(int)
            for pattern in self.access_patterns.values():
                pattern_types[pattern.temporal_pattern] += 1
        
        cache_stats = self.cache_manager.get_stats()
        
        # Calculate prediction accuracy
        successful_predictions = self.warming_stats['successful_predictions']
        total_predictions = self.warming_stats['predictions_made']
        prediction_accuracy = (
            successful_predictions / total_predictions 
            if total_predictions > 0 else 0.0
        )
        
        return {
            'pattern_analysis': {
                'total_patterns': total_patterns,
                'active_patterns': active_patterns,
                'pattern_types': dict(pattern_types),
                'avg_accesses_per_pattern': np.mean([
                    p.access_count for p in self.access_patterns.values()
                ]) if self.access_patterns else 0,
            },
            'warming_performance': {
                'prediction_accuracy': prediction_accuracy,
                'cache_hits_from_warming': self.warming_stats['cache_hits_from_warming'],
                'warming_overhead_ms': self.warming_stats['warming_overhead_ms'],
                'patterns_learned': self.warming_stats['patterns_learned'],
            },
            'cache_integration': {
                'cache_hit_rate': cache_stats.get('hit_rate', 0.0),
                'cache_utilization': cache_stats.get('memory_utilization', 0.0),
                'current_cache_items': cache_stats.get('current_items', 0),
            },
            'configuration': {
                'learning_window_hours': self.config.learning_window_hours,
                'warm_ahead_hours': self.config.warm_ahead_hours,
                'max_warm_items': self.config.max_warm_items,
                'warm_confidence_threshold': self.config.warm_confidence_threshold,
            }
        }
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """Optimize configuration based on learned patterns and performance."""
        optimization_results = {
            'optimizations_applied': 0,
            'improvements': [],
        }
        
        try:
            cache_stats = self.cache_manager.get_stats()
            hit_rate = cache_stats.get('hit_rate', 0.0)
            
            # Optimize warming threshold based on cache hit rate
            if hit_rate < 0.7:  # Low hit rate
                # Lower confidence threshold to warm more aggressively
                old_threshold = self.config.warm_confidence_threshold
                self.config.warm_confidence_threshold = max(0.5, old_threshold - 0.1)
                
                if self.config.warm_confidence_threshold != old_threshold:
                    optimization_results['optimizations_applied'] += 1
                    optimization_results['improvements'].append(
                        f'Lowered warming threshold from {old_threshold} to {self.config.warm_confidence_threshold}'
                    )
            
            elif hit_rate > 0.9:  # Very high hit rate
                # Raise confidence threshold to be more selective
                old_threshold = self.config.warm_confidence_threshold
                self.config.warm_confidence_threshold = min(0.9, old_threshold + 0.05)
                
                if self.config.warm_confidence_threshold != old_threshold:
                    optimization_results['optimizations_applied'] += 1
                    optimization_results['improvements'].append(
                        f'Raised warming threshold from {old_threshold} to {self.config.warm_confidence_threshold}'
                    )
            
            # Optimize warming window based on pattern regularity
            with self.pattern_lock:
                regular_patterns = sum(
                    1 for p in self.access_patterns.values()
                    if p.temporal_pattern in ['daily', 'weekly', 'time_based']
                )
                total_patterns = len(self.access_patterns)
            
            if total_patterns > 0:
                regularity_ratio = regular_patterns / total_patterns
                
                if regularity_ratio > 0.7:  # Many regular patterns
                    # Can predict further ahead
                    old_hours = self.config.warm_ahead_hours
                    self.config.warm_ahead_hours = min(6, old_hours + 1)
                    
                    if self.config.warm_ahead_hours != old_hours:
                        optimization_results['optimizations_applied'] += 1
                        optimization_results['improvements'].append(
                            f'Increased warming window from {old_hours} to {self.config.warm_ahead_hours} hours'
                        )
                
                elif regularity_ratio < 0.3:  # Many irregular patterns
                    # Reduce warming window to be more conservative
                    old_hours = self.config.warm_ahead_hours
                    self.config.warm_ahead_hours = max(1, old_hours - 1)
                    
                    if self.config.warm_ahead_hours != old_hours:
                        optimization_results['optimizations_applied'] += 1
                        optimization_results['improvements'].append(
                            f'Reduced warming window from {old_hours} to {self.config.warm_ahead_hours} hours'
                        )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Configuration optimization failed: {e}")
            optimization_results['error'] = str(e)
            return optimization_results
    
    def shutdown(self):
        """Shutdown predictive cache warmer."""
        logger.info("Shutting down predictive cache warmer...")
        
        self._shutdown_event.set()
        
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5.0)
        
        logger.info("Predictive cache warmer shutdown complete")


def create_predictive_cache_warmer(
    cache_manager: CacheManager,
    performance_monitor: PerformanceMonitor,
    config: Optional[PredictiveCacheConfig] = None
) -> PredictiveCacheWarmer:
    """
    Factory function to create a predictive cache warmer.
    
    Args:
        cache_manager: Cache management system
        performance_monitor: Performance monitoring system
        config: Optional predictive cache configuration
        
    Returns:
        PredictiveCacheWarmer instance
    """
    return PredictiveCacheWarmer(
        cache_manager=cache_manager,
        performance_monitor=performance_monitor,
        config=config
    )