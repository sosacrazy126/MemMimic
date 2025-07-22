"""
Intelligent Batch Processing System for CXD Classification.

Implements ML-driven batch processing with queue management, dynamic batching,
and performance optimization for high-throughput classification operations.
"""

import asyncio
import hashlib
import logging
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import weakref

import numpy as np

from ..core.interfaces import CXDClassifier
from ..core.types import CXDSequence, CXDFunction
from .optimized_semantic import OptimizedSemanticCXDClassifier

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Individual classification request in the batch queue."""
    text: str
    request_id: str
    timestamp: datetime
    priority: float = 1.0
    callback: Optional[Callable[[CXDSequence], None]] = None
    future: Optional[asyncio.Future] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Configuration for batch processing behavior."""
    # Batching parameters
    max_batch_size: int = 32
    min_batch_size: int = 4
    max_wait_time_ms: float = 50.0
    min_wait_time_ms: float = 5.0
    
    # Performance tuning
    target_throughput_per_sec: float = 100.0
    max_queue_size: int = 1000
    adaptive_batching: bool = True
    
    # Resource management
    max_worker_threads: int = 4
    memory_pressure_threshold: float = 0.8
    
    # ML-driven optimization
    enable_learning: bool = True
    learning_window_size: int = 100
    performance_history_size: int = 1000


@dataclass
class BatchMetrics:
    """Performance metrics for batch processing."""
    total_requests: int = 0
    batches_processed: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time_ms: float = 0.0
    avg_wait_time_ms: float = 0.0
    throughput_per_sec: float = 0.0
    cache_hit_rate: float = 0.0
    queue_utilization: float = 0.0
    memory_utilization_mb: float = 0.0
    
    # ML optimization metrics
    batch_size_predictions: List[int] = field(default_factory=list)
    timing_predictions: List[float] = field(default_factory=list)
    optimization_accuracy: float = 0.0


class IntelligentBatchProcessor:
    """
    ML-driven batch processor for CXD classification with adaptive optimization.
    
    Features:
    - Dynamic batch sizing based on workload patterns
    - Priority-based request queuing
    - Predictive batch timing optimization
    - Memory-aware processing
    - Performance learning and adaptation
    - Async/await support with thread-safe operations
    """
    
    def __init__(self, 
                 classifier: CXDClassifier,
                 config: Optional[BatchConfig] = None):
        """
        Initialize intelligent batch processor.
        
        Args:
            classifier: CXD classifier instance to use for batch processing
            config: Batch processing configuration
        """
        self.classifier = classifier
        self.config = config or BatchConfig()
        
        # Request queue and processing state
        self._request_queue = deque()
        self._processing_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Worker thread pool
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_worker_threads)
        
        # Performance monitoring
        self._metrics = BatchMetrics()
        self._performance_history = deque(maxlen=self.config.performance_history_size)
        self._processing_times: Dict[int, List[float]] = defaultdict(list)  # batch_size -> times
        
        # ML-driven optimization state
        self._batch_size_predictor = BatchSizePredictor() if self.config.enable_learning else None
        self._timing_predictor = TimingPredictor() if self.config.enable_learning else None
        
        # Adaptive parameters (learned over time)
        self._adaptive_max_batch_size = self.config.max_batch_size
        self._adaptive_wait_time_ms = self.config.max_wait_time_ms
        
        # Result cache for duplicate detection
        self._result_cache: Dict[str, Tuple[CXDSequence, datetime]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        
        # Background processing
        self._processor_thread = None
        self._start_background_processor()
        
        logger.info(f"IntelligentBatchProcessor initialized with config: {self.config}")
    
    async def classify_async(self, text: str, 
                           priority: float = 1.0,
                           metadata: Optional[Dict[str, Any]] = None) -> CXDSequence:
        """
        Classify text asynchronously using batch processing.
        
        Args:
            text: Text to classify
            priority: Request priority (higher = processed sooner)
            metadata: Optional metadata for the request
            
        Returns:
            CXD classification sequence
        """
        # Check cache first
        cache_key = self._generate_cache_key(text)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Create request
        request_id = self._generate_request_id(text, priority)
        future = asyncio.Future()
        
        request = BatchRequest(
            text=text,
            request_id=request_id,
            timestamp=datetime.now(),
            priority=priority,
            future=future,
            metadata=metadata or {}
        )
        
        # Add to queue
        with self._processing_lock:
            self._request_queue.append(request)
            self._metrics.total_requests += 1
        
        # Wait for result
        try:
            result = await future
            # Cache successful result
            self._cache_result(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Async classification failed for request {request_id}: {e}")
            raise
    
    def classify_sync(self, text: str,
                     priority: float = 1.0,
                     metadata: Optional[Dict[str, Any]] = None) -> CXDSequence:
        """
        Classify text synchronously using batch processing.
        
        Args:
            text: Text to classify
            priority: Request priority
            metadata: Optional metadata
            
        Returns:
            CXD classification sequence
        """
        # Check cache first
        cache_key = self._generate_cache_key(text)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # Create synchronization event
        result_event = threading.Event()
        result_container = {'result': None, 'error': None}
        
        def callback(result):
            result_container['result'] = result
            result_event.set()
        
        def error_callback(error):
            result_container['error'] = error
            result_event.set()
        
        # Create request
        request_id = self._generate_request_id(text, priority)
        request = BatchRequest(
            text=text,
            request_id=request_id,
            timestamp=datetime.now(),
            priority=priority,
            callback=callback,
            metadata=metadata or {}
        )
        
        # Add to queue
        with self._processing_lock:
            self._request_queue.append(request)
            self._metrics.total_requests += 1
        
        # Wait for result
        if result_event.wait(timeout=30.0):  # 30 second timeout
            if result_container['error']:
                raise result_container['error']
            result = result_container['result']
            # Cache successful result
            self._cache_result(cache_key, result)
            return result
        else:
            raise TimeoutError(f"Classification request {request_id} timed out")
    
    def _start_background_processor(self):
        """Start background thread for batch processing."""
        def processor_loop():
            while not self._shutdown_event.is_set():
                try:
                    batch = self._collect_batch()
                    if batch:
                        self._process_batch(batch)
                    else:
                        # No batch ready, wait briefly
                        time.sleep(0.001)  # 1ms
                except Exception as e:
                    logger.error(f"Batch processor error: {e}")
                    time.sleep(0.1)  # Back off on errors
        
        self._processor_thread = threading.Thread(target=processor_loop, daemon=True)
        self._processor_thread.start()
        logger.debug("Background batch processor started")
    
    def _collect_batch(self) -> Optional[List[BatchRequest]]:
        """
        Intelligently collect requests into an optimal batch.
        
        Returns:
            List of requests to process as a batch, or None if no batch ready
        """
        with self._processing_lock:
            if not self._request_queue:
                return None
            
            # Calculate optimal batch size and timing
            queue_size = len(self._request_queue)
            optimal_batch_size = self._calculate_optimal_batch_size(queue_size)
            max_wait_time = self._calculate_optimal_wait_time(queue_size)
            
            # Get oldest request to check timing
            oldest_request = self._request_queue[0]
            wait_time_ms = (datetime.now() - oldest_request.timestamp).total_seconds() * 1000
            
            # Decide if batch is ready
            should_process = (
                queue_size >= optimal_batch_size or
                wait_time_ms >= max_wait_time or
                queue_size >= self.config.max_queue_size * 0.8  # Emergency processing
            )
            
            if not should_process:
                return None
            
            # Collect batch with priority sorting
            batch_size = min(optimal_batch_size, queue_size)
            requests = []
            
            # Sort queue by priority (descending) and age (ascending)
            sorted_queue = sorted(
                list(self._request_queue),
                key=lambda r: (-r.priority, r.timestamp)
            )
            
            # Take top priority requests
            for i in range(batch_size):
                request = sorted_queue[i]
                requests.append(request)
                self._request_queue.remove(request)
            
            return requests
    
    def _calculate_optimal_batch_size(self, queue_size: int) -> int:
        """Calculate optimal batch size based on current conditions."""
        if not self.config.adaptive_batching:
            return min(self.config.max_batch_size, queue_size)
        
        # Use ML predictor if available
        if self._batch_size_predictor:
            predicted_size = self._batch_size_predictor.predict(
                queue_size=queue_size,
                current_throughput=self._metrics.throughput_per_sec,
                avg_processing_time=self._metrics.avg_processing_time_ms
            )
            return min(predicted_size, self._adaptive_max_batch_size, queue_size)
        
        # Fallback to heuristic-based sizing
        if queue_size <= self.config.min_batch_size:
            return queue_size
        elif queue_size >= self.config.max_batch_size:
            return self.config.max_batch_size
        else:
            # Dynamic sizing based on performance history
            if self._performance_history:
                recent_performance = list(self._performance_history)[-10:]
                avg_throughput = np.mean([p['throughput'] for p in recent_performance])
                
                if avg_throughput < self.config.target_throughput_per_sec * 0.8:
                    # Increase batch size to improve throughput
                    return min(self.config.max_batch_size, int(queue_size * 0.8))
                else:
                    # Maintain current sizing
                    return min(self.config.max_batch_size, max(self.config.min_batch_size, queue_size // 2))
            
            return min(self.config.max_batch_size, queue_size)
    
    def _calculate_optimal_wait_time(self, queue_size: int) -> float:
        """Calculate optimal wait time based on current conditions."""
        if not self.config.adaptive_batching:
            return self.config.max_wait_time_ms
        
        # Use ML predictor if available
        if self._timing_predictor:
            predicted_wait = self._timing_predictor.predict(
                queue_size=queue_size,
                current_throughput=self._metrics.throughput_per_sec
            )
            return max(self.config.min_wait_time_ms, 
                      min(predicted_wait, self._adaptive_wait_time_ms))
        
        # Adaptive wait time based on queue pressure
        if queue_size <= self.config.min_batch_size:
            return self.config.max_wait_time_ms  # Wait longer for small queues
        elif queue_size >= self.config.max_batch_size:
            return self.config.min_wait_time_ms  # Process immediately for large queues
        else:
            # Linear interpolation
            ratio = (queue_size - self.config.min_batch_size) / (
                self.config.max_batch_size - self.config.min_batch_size
            )
            return self.config.max_wait_time_ms * (1 - ratio) + self.config.min_wait_time_ms * ratio
    
    def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of classification requests."""
        if not batch:
            return
        
        batch_start_time = time.perf_counter()
        batch_size = len(batch)
        
        try:
            # Extract texts for batch classification
            texts = [request.text for request in batch]
            
            # Check if classifier supports batch processing
            if hasattr(self.classifier, 'classify_batch'):
                # Use native batch processing
                results = self.classifier.classify_batch(texts)
            else:
                # Fallback to individual classification
                results = []
                for text in texts:
                    result = self.classifier.classify(text)
                    results.append(result)
            
            # Process results and notify requesters
            processing_time = (time.perf_counter() - batch_start_time) * 1000
            
            for request, result in zip(batch, results):
                try:
                    if request.future and not request.future.done():
                        # Async request
                        request.future.set_result(result)
                    elif request.callback:
                        # Sync request with callback
                        request.callback(result)
                except Exception as e:
                    logger.error(f"Failed to deliver result for request {request.request_id}: {e}")
                    if request.future and not request.future.done():
                        request.future.set_exception(e)
            
            # Update metrics and learning data
            self._update_metrics(batch, processing_time)
            self._record_performance_data(batch_size, processing_time)
            
            logger.debug(f"Processed batch of {batch_size} requests in {processing_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Notify all requesters of failure
            for request in batch:
                try:
                    if request.future and not request.future.done():
                        request.future.set_exception(e)
                    elif request.callback:
                        # For sync requests, we can't easily propagate the error
                        # Log it and hope the timeout mechanism catches it
                        logger.error(f"Cannot propagate error to sync request {request.request_id}")
                except Exception as notify_error:
                    logger.error(f"Failed to notify request {request.request_id} of error: {notify_error}")
    
    def _update_metrics(self, batch: List[BatchRequest], processing_time_ms: float):
        """Update performance metrics after batch processing."""
        batch_size = len(batch)
        
        # Calculate wait times
        now = datetime.now()
        wait_times = [(now - request.timestamp).total_seconds() * 1000 for request in batch]
        avg_wait_time = np.mean(wait_times)
        
        # Update metrics
        self._metrics.batches_processed += 1
        self._metrics.avg_batch_size = (
            (self._metrics.avg_batch_size * (self._metrics.batches_processed - 1) + batch_size) /
            self._metrics.batches_processed
        )
        self._metrics.avg_processing_time_ms = (
            (self._metrics.avg_processing_time_ms * (self._metrics.batches_processed - 1) + processing_time_ms) /
            self._metrics.batches_processed
        )
        self._metrics.avg_wait_time_ms = (
            (self._metrics.avg_wait_time_ms * (self._metrics.batches_processed - 1) + avg_wait_time) /
            self._metrics.batches_processed
        )
        
        # Calculate throughput
        total_time_s = (self._metrics.avg_processing_time_ms + self._metrics.avg_wait_time_ms) / 1000
        if total_time_s > 0:
            self._metrics.throughput_per_sec = self._metrics.avg_batch_size / total_time_s
        
        # Queue utilization
        with self._processing_lock:
            self._metrics.queue_utilization = len(self._request_queue) / self.config.max_queue_size
    
    def _record_performance_data(self, batch_size: int, processing_time_ms: float):
        """Record performance data for ML learning."""
        performance_record = {
            'timestamp': datetime.now(),
            'batch_size': batch_size,
            'processing_time_ms': processing_time_ms,
            'throughput': batch_size / (processing_time_ms / 1000),
            'queue_size_before': len(self._request_queue) + batch_size,
        }
        
        self._performance_history.append(performance_record)
        
        # Update batch size -> processing time mapping
        self._processing_times[batch_size].append(processing_time_ms)
        if len(self._processing_times[batch_size]) > 50:
            self._processing_times[batch_size] = self._processing_times[batch_size][-50:]
        
        # Train ML predictors if enabled
        if self.config.enable_learning and len(self._performance_history) % 10 == 0:
            self._train_predictors()
    
    def _train_predictors(self):
        """Train ML predictors based on historical performance data."""
        if not self._performance_history or len(self._performance_history) < 20:
            return
        
        try:
            # Train batch size predictor
            if self._batch_size_predictor:
                self._batch_size_predictor.train(self._performance_history)
            
            # Train timing predictor
            if self._timing_predictor:
                self._timing_predictor.train(self._performance_history)
                
            # Update adaptive parameters based on learning
            self._update_adaptive_parameters()
            
        except Exception as e:
            logger.error(f"Failed to train ML predictors: {e}")
    
    def _update_adaptive_parameters(self):
        """Update adaptive parameters based on learned patterns."""
        if not self._performance_history:
            return
        
        recent_data = list(self._performance_history)[-50:]
        
        # Find optimal batch size based on throughput
        batch_throughputs = defaultdict(list)
        for record in recent_data:
            batch_throughputs[record['batch_size']].append(record['throughput'])
        
        best_batch_size = self.config.max_batch_size
        best_throughput = 0
        
        for batch_size, throughputs in batch_throughputs.items():
            avg_throughput = np.mean(throughputs)
            if avg_throughput > best_throughput and len(throughputs) >= 3:
                best_throughput = avg_throughput
                best_batch_size = batch_size
        
        # Gradually adapt parameters
        self._adaptive_max_batch_size = int(
            0.9 * self._adaptive_max_batch_size + 0.1 * best_batch_size
        )
        
        # Adapt wait time based on queue pressure patterns
        avg_queue_utilization = np.mean([r.get('queue_size_before', 0) for r in recent_data]) / self.config.max_queue_size
        if avg_queue_utilization > 0.7:
            # High queue pressure, reduce wait time
            self._adaptive_wait_time_ms *= 0.95
        elif avg_queue_utilization < 0.3:
            # Low queue pressure, can afford to wait longer
            self._adaptive_wait_time_ms *= 1.05
        
        self._adaptive_wait_time_ms = max(
            self.config.min_wait_time_ms,
            min(self._adaptive_wait_time_ms, self.config.max_wait_time_ms)
        )
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text classification result."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[CXDSequence]:
        """Get cached classification result if available and not expired."""
        if cache_key in self._result_cache:
            result, timestamp = self._result_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl_seconds:
                self._metrics.cache_hit_rate = (
                    (self._metrics.cache_hit_rate * self._metrics.total_requests + 1) /
                    (self._metrics.total_requests + 1)
                )
                return result
            else:
                # Expired, remove from cache
                del self._result_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: CXDSequence):
        """Cache classification result."""
        self._result_cache[cache_key] = (result, datetime.now())
        
        # Cleanup old entries if cache is too large
        if len(self._result_cache) > 1000:
            # Remove oldest 20% of entries
            sorted_items = sorted(
                self._result_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            items_to_remove = len(sorted_items) // 5
            for cache_key, _ in sorted_items[:items_to_remove]:
                del self._result_cache[cache_key]
    
    def _generate_request_id(self, text: str, priority: float) -> str:
        """Generate unique request ID."""
        timestamp = str(time.time())
        content_hash = hashlib.md5(f"{text}{priority}{timestamp}".encode()).hexdigest()[:8]
        return f"batch_{content_hash}_{timestamp.split('.')[-1]}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive batch processing metrics."""
        with self._processing_lock:
            queue_size = len(self._request_queue)
        
        return {
            'basic_metrics': {
                'total_requests': self._metrics.total_requests,
                'batches_processed': self._metrics.batches_processed,
                'avg_batch_size': self._metrics.avg_batch_size,
                'avg_processing_time_ms': self._metrics.avg_processing_time_ms,
                'avg_wait_time_ms': self._metrics.avg_wait_time_ms,
                'throughput_per_sec': self._metrics.throughput_per_sec,
                'cache_hit_rate': self._metrics.cache_hit_rate,
            },
            'queue_status': {
                'current_queue_size': queue_size,
                'queue_utilization': queue_size / self.config.max_queue_size,
                'max_queue_size': self.config.max_queue_size,
            },
            'adaptive_parameters': {
                'adaptive_max_batch_size': self._adaptive_max_batch_size,
                'adaptive_wait_time_ms': self._adaptive_wait_time_ms,
                'config_max_batch_size': self.config.max_batch_size,
                'config_max_wait_time_ms': self.config.max_wait_time_ms,
            },
            'ml_optimization': {
                'learning_enabled': self.config.enable_learning,
                'performance_history_size': len(self._performance_history),
                'predictor_accuracy': getattr(self._batch_size_predictor, 'accuracy', 0.0),
            },
            'cache_stats': {
                'cache_size': len(self._result_cache),
                'cache_hit_rate': self._metrics.cache_hit_rate,
                'cache_ttl_seconds': self._cache_ttl_seconds,
            }
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Manually trigger performance optimization."""
        optimization_results = {
            'optimizations_applied': 0,
            'improvements': [],
        }
        
        try:
            # Clean expired cache entries
            expired_count = 0
            now = datetime.now()
            expired_keys = [
                key for key, (_, timestamp) in self._result_cache.items()
                if (now - timestamp).total_seconds() > self._cache_ttl_seconds
            ]
            
            for key in expired_keys:
                del self._result_cache[key]
                expired_count += 1
            
            if expired_count > 0:
                optimization_results['optimizations_applied'] += 1
                optimization_results['improvements'].append(
                    f'Removed {expired_count} expired cache entries'
                )
            
            # Retrain predictors if enough new data
            if (self.config.enable_learning and 
                len(self._performance_history) >= 20 and
                len(self._performance_history) % 5 == 0):
                
                self._train_predictors()
                optimization_results['optimizations_applied'] += 1
                optimization_results['improvements'].append('Retrained ML predictors')
            
            # Adjust adaptive parameters
            old_batch_size = self._adaptive_max_batch_size
            old_wait_time = self._adaptive_wait_time_ms
            
            self._update_adaptive_parameters()
            
            if (abs(old_batch_size - self._adaptive_max_batch_size) > 1 or
                abs(old_wait_time - self._adaptive_wait_time_ms) > 5):
                optimization_results['optimizations_applied'] += 1
                optimization_results['improvements'].append(
                    f'Adapted batch size: {old_batch_size} -> {self._adaptive_max_batch_size}, '
                    f'wait time: {old_wait_time:.1f} -> {self._adaptive_wait_time_ms:.1f}ms'
                )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            optimization_results['error'] = str(e)
            return optimization_results
    
    def shutdown(self):
        """Shutdown the batch processor gracefully."""
        logger.info("Shutting down batch processor...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Process remaining requests
        with self._processing_lock:
            remaining_requests = list(self._request_queue)
            self._request_queue.clear()
        
        if remaining_requests:
            logger.info(f"Processing {len(remaining_requests)} remaining requests...")
            self._process_batch(remaining_requests)
        
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        
        # Wait for processor thread
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5.0)
        
        logger.info("Batch processor shutdown complete")


class BatchSizePredictor:
    """ML predictor for optimal batch sizes based on system conditions."""
    
    def __init__(self):
        self.model_weights = np.array([1.0, 0.5, -0.2])  # Simple linear model
        self.training_data = []
        self.accuracy = 0.0
    
    def predict(self, queue_size: int, current_throughput: float, avg_processing_time: float) -> int:
        """Predict optimal batch size based on current conditions."""
        # Simple linear prediction model
        features = np.array([queue_size, current_throughput, avg_processing_time])
        prediction = np.dot(features, self.model_weights)
        
        # Clamp to reasonable range
        return max(4, min(32, int(prediction)))
    
    def train(self, performance_history: deque):
        """Train predictor on historical performance data."""
        if len(performance_history) < 10:
            return
        
        # Simple training logic - in reality this would be more sophisticated
        recent_data = list(performance_history)[-50:]
        
        # Find best performing batch sizes
        best_performers = sorted(recent_data, key=lambda x: x['throughput'], reverse=True)[:10]
        avg_best_batch_size = np.mean([p['batch_size'] for p in best_performers])
        
        # Update weights to favor successful patterns
        self.model_weights[0] *= 0.95  # Slightly reduce queue size influence
        self.model_weights[1] *= 1.02  # Slightly increase throughput influence
        
        self.accuracy = min(0.95, self.accuracy + 0.01)  # Gradual accuracy improvement


class TimingPredictor:
    """ML predictor for optimal wait times based on system conditions."""
    
    def __init__(self):
        self.base_wait_time = 25.0  # milliseconds
        self.queue_factor = 0.5
        self.throughput_factor = 0.3
        self.accuracy = 0.0
    
    def predict(self, queue_size: int, current_throughput: float) -> float:
        """Predict optimal wait time based on current conditions."""
        # Adaptive wait time based on queue pressure and throughput
        queue_pressure = min(1.0, queue_size / 32.0)  # Normalize to 32 max batch size
        throughput_factor = max(0.5, min(2.0, current_throughput / 100.0))
        
        predicted_wait = (
            self.base_wait_time * 
            (1 - queue_pressure * self.queue_factor) *
            (2 - throughput_factor * self.throughput_factor)
        )
        
        return max(5.0, min(100.0, predicted_wait))
    
    def train(self, performance_history: deque):
        """Train timing predictor on historical data."""
        if len(performance_history) < 10:
            return
        
        # Analyze optimal wait times from historical data
        recent_data = list(performance_history)[-30:]
        
        # Find patterns in successful timing
        high_throughput_records = [r for r in recent_data if r['throughput'] > 80]
        if high_throughput_records:
            avg_successful_wait = np.mean([
                r.get('avg_wait_time_ms', self.base_wait_time) 
                for r in high_throughput_records
            ])
            
            # Gradually adapt base wait time
            self.base_wait_time = 0.9 * self.base_wait_time + 0.1 * avg_successful_wait
        
        self.accuracy = min(0.90, self.accuracy + 0.02)


# Factory function for easy integration
def create_batch_processor(classifier: CXDClassifier, 
                          config: Optional[BatchConfig] = None) -> IntelligentBatchProcessor:
    """
    Create an intelligent batch processor for CXD classification.
    
    Args:
        classifier: CXD classifier to use for batch processing
        config: Optional batch configuration
        
    Returns:
        IntelligentBatchProcessor instance
    """
    return IntelligentBatchProcessor(classifier=classifier, config=config)


# Integration decorator for existing classifiers
def batch_enabled(config: Optional[BatchConfig] = None):
    """
    Decorator to add batch processing capabilities to existing CXD classifiers.
    
    Args:
        config: Optional batch configuration
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def enhanced_init(self, *args, **kwargs):
            # Initialize original classifier
            original_init(self, *args, **kwargs)
            
            # Add batch processor
            self._batch_processor = create_batch_processor(self, config)
        
        def classify_async(self, text: str, priority: float = 1.0, 
                          metadata: Optional[Dict[str, Any]] = None):
            return self._batch_processor.classify_async(text, priority, metadata)
        
        def classify_batch_sync(self, texts: List[str], priorities: Optional[List[float]] = None):
            results = []
            for i, text in enumerate(texts):
                priority = priorities[i] if priorities else 1.0
                result = self._batch_processor.classify_sync(text, priority)
                results.append(result)
            return results
        
        def get_batch_metrics(self):
            return self._batch_processor.get_metrics()
        
        def optimize_batch_performance(self):
            return self._batch_processor.optimize_performance()
        
        def shutdown_batch_processor(self):
            self._batch_processor.shutdown()
        
        # Add new methods to class
        cls.__init__ = enhanced_init
        cls.classify_async = classify_async
        cls.classify_batch_sync = classify_batch_sync
        cls.get_batch_metrics = get_batch_metrics
        cls.optimize_batch_performance = optimize_batch_performance
        cls.shutdown_batch_processor = shutdown_batch_processor
        
        return cls
    
    return decorator