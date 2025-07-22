"""
Dynamic Resource Allocation System with Workload Prediction and Auto-Scaling.

Implements intelligent resource management that predicts workload patterns,
dynamically allocates system resources, and auto-scales components based on
demand to optimize performance and efficiency.
"""

import asyncio
import logging
import numpy as np
import pickle
import psutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Current system resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_recv_mb: float
    network_io_sent_mb: float
    active_threads: int
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class WorkloadPrediction:
    """Workload prediction for resource planning."""
    prediction_time: datetime
    predicted_cpu: float
    predicted_memory_mb: float
    predicted_requests_per_second: float
    predicted_queue_sizes: Dict[str, int]
    confidence: float
    prediction_horizon_minutes: int
    model_used: str
    features_used: List[str] = field(default_factory=list)


@dataclass
class ResourceAllocation:
    """Current resource allocation configuration."""
    component_name: str
    allocated_cpu_cores: float
    allocated_memory_mb: int
    thread_pool_size: int
    process_pool_size: int
    cache_size_mb: int
    batch_size_limits: Dict[str, int] = field(default_factory=dict)
    priority: float = 1.0
    scaling_enabled: bool = True
    min_resources: Dict[str, Any] = field(default_factory=dict)
    max_resources: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationConfig:
    """Configuration for dynamic resource allocation."""
    # Prediction parameters
    prediction_window_minutes: int = 60
    prediction_horizon_minutes: int = 30
    model_retrain_hours: int = 6
    min_history_points: int = 20
    
    # Scaling parameters
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.4
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.75
    min_scale_interval_minutes: int = 5
    
    # Resource limits
    max_cpu_cores: int = os.cpu_count() or 4
    max_memory_mb: int = int(psutil.virtual_memory().total / (1024**2))
    max_thread_pool_size: int = 50
    max_process_pool_size: int = 8
    
    # Component priorities
    component_priorities: Dict[str, float] = field(default_factory=lambda: {
        'search_engine': 1.0,
        'cxd_classifier': 0.9,
        'cache_manager': 0.8,
        'consolidator': 0.6,
        'background_tasks': 0.4
    })
    
    # System health thresholds
    critical_cpu_threshold: float = 0.9
    critical_memory_threshold: float = 0.95
    emergency_scale_factor: float = 2.0
    
    # Monitoring parameters
    metrics_collection_interval_seconds: int = 30
    allocation_update_interval_minutes: int = 2
    performance_evaluation_interval_minutes: int = 15


class WorkloadPredictor:
    """Machine learning based workload predictor."""
    
    def __init__(self, config: AllocationConfig):
        self.config = config
        
        # ML models for different metrics
        self.cpu_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.memory_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.requests_model = LinearRegression()
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Training data
        self.training_data = deque(maxlen=10000)
        self.feature_names = [
            'hour_of_day', 'day_of_week', 'cpu_trend', 'memory_trend',
            'requests_trend', 'queue_pressure', 'recent_avg_cpu', 
            'recent_avg_memory', 'recent_avg_requests'
        ]
        
        # Model state
        self.is_trained = False
        self.last_training = None
        self.prediction_accuracy = {'cpu': 0.0, 'memory': 0.0, 'requests': 0.0}
    
    def add_training_data(self, metrics: ResourceMetrics, 
                         requests_per_second: float = 0.0):
        """Add new training data point."""
        try:
            # Extract features
            features = self._extract_features(metrics, requests_per_second)
            
            # Add to training data
            training_point = {
                'timestamp': metrics.timestamp,
                'features': features,
                'targets': {
                    'cpu': metrics.cpu_percent,
                    'memory': metrics.memory_percent,
                    'requests': requests_per_second
                }
            }
            
            self.training_data.append(training_point)
            
            # Retrain if enough data and time has passed
            if (len(self.training_data) >= self.config.min_history_points and
                (not self.last_training or 
                 (datetime.now() - self.last_training).total_seconds() > 
                 self.config.model_retrain_hours * 3600)):
                self._train_models()
            
        except Exception as e:
            logger.error(f"Failed to add training data: {e}")
    
    def _extract_features(self, metrics: ResourceMetrics, 
                         requests_per_second: float) -> np.ndarray:
        """Extract features from metrics for ML models."""
        timestamp = metrics.timestamp
        
        features = [
            timestamp.hour / 23.0,  # hour_of_day
            timestamp.weekday() / 6.0,  # day_of_week
            0.0,  # cpu_trend (will be calculated)
            0.0,  # memory_trend (will be calculated)
            0.0,  # requests_trend (will be calculated)
            sum(metrics.queue_sizes.values()) / 1000.0,  # queue_pressure
            0.0,  # recent_avg_cpu
            0.0,  # recent_avg_memory  
            0.0,  # recent_avg_requests
        ]
        
        # Calculate trends and averages from recent data
        if len(self.training_data) >= 5:
            recent_points = list(self.training_data)[-5:]
            
            # CPU trend
            cpu_values = [p['targets']['cpu'] for p in recent_points]
            if len(cpu_values) > 1:
                features[2] = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
            
            # Memory trend
            memory_values = [p['targets']['memory'] for p in recent_points]
            if len(memory_values) > 1:
                features[3] = (memory_values[-1] - memory_values[0]) / len(memory_values)
            
            # Requests trend
            request_values = [p['targets']['requests'] for p in recent_points]
            if len(request_values) > 1:
                features[4] = (request_values[-1] - request_values[0]) / len(request_values)
            
            # Recent averages
            features[6] = np.mean(cpu_values)
            features[7] = np.mean(memory_values)
            features[8] = np.mean(request_values)
        
        return np.array(features)
    
    def _train_models(self):
        """Train ML models on accumulated data."""
        try:
            if len(self.training_data) < self.config.min_history_points:
                return
            
            # Prepare training data
            X = np.array([point['features'] for point in self.training_data])
            y_cpu = np.array([point['targets']['cpu'] for point in self.training_data])
            y_memory = np.array([point['targets']['memory'] for point in self.training_data])
            y_requests = np.array([point['targets']['requests'] for point in self.training_data])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.cpu_model.fit(X_scaled, y_cpu)
            self.memory_model.fit(X_scaled, y_memory)
            self.requests_model.fit(X_scaled, y_requests)
            
            # Evaluate accuracy
            self._evaluate_model_accuracy(X_scaled, y_cpu, y_memory, y_requests)
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            logger.info(f"Trained workload prediction models with {len(self.training_data)} data points")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def _evaluate_model_accuracy(self, X, y_cpu, y_memory, y_requests):
        """Evaluate model prediction accuracy."""
        try:
            # CPU model accuracy
            cpu_pred = self.cpu_model.predict(X)
            self.prediction_accuracy['cpu'] = 1.0 - mean_absolute_error(y_cpu, cpu_pred) / np.mean(y_cpu)
            
            # Memory model accuracy
            memory_pred = self.memory_model.predict(X)
            self.prediction_accuracy['memory'] = 1.0 - mean_absolute_error(y_memory, memory_pred) / np.mean(y_memory)
            
            # Requests model accuracy
            requests_pred = self.requests_model.predict(X)
            if np.mean(y_requests) > 0:
                self.prediction_accuracy['requests'] = 1.0 - mean_absolute_error(y_requests, requests_pred) / np.mean(y_requests)
            else:
                self.prediction_accuracy['requests'] = 0.0
            
        except Exception as e:
            logger.warning(f"Failed to evaluate model accuracy: {e}")
    
    def predict_workload(self, 
                        current_metrics: ResourceMetrics,
                        current_requests_per_second: float = 0.0,
                        horizon_minutes: int = None) -> WorkloadPrediction:
        """Predict future workload based on current state."""
        horizon_minutes = horizon_minutes or self.config.prediction_horizon_minutes
        
        if not self.is_trained:
            # Fallback prediction based on current state
            return WorkloadPrediction(
                prediction_time=datetime.now() + timedelta(minutes=horizon_minutes),
                predicted_cpu=current_metrics.cpu_percent * 1.1,  # Slight increase
                predicted_memory_mb=current_metrics.memory_available_mb * 0.9,  # Slight decrease
                predicted_requests_per_second=current_requests_per_second * 1.05,
                predicted_queue_sizes={k: int(v * 1.1) for k, v in current_metrics.queue_sizes.items()},
                confidence=0.3,  # Low confidence
                prediction_horizon_minutes=horizon_minutes,
                model_used='fallback'
            )
        
        try:
            # Extract features for prediction
            features = self._extract_features(current_metrics, current_requests_per_second)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make predictions
            predicted_cpu = float(self.cpu_model.predict(features_scaled)[0])
            predicted_memory_percent = float(self.memory_model.predict(features_scaled)[0])
            predicted_requests = float(self.requests_model.predict(features_scaled)[0])
            
            # Convert memory percentage to available MB
            total_memory_mb = psutil.virtual_memory().total / (1024**2)
            predicted_memory_mb = total_memory_mb * (1 - predicted_memory_percent / 100)
            
            # Predict queue sizes (simple heuristic)
            predicted_queues = {}
            for queue_name, current_size in current_metrics.queue_sizes.items():
                # Queue size correlates with request rate
                growth_factor = max(0.5, predicted_requests / max(current_requests_per_second, 1))
                predicted_queues[queue_name] = int(current_size * growth_factor)
            
            # Calculate prediction confidence
            confidence = np.mean(list(self.prediction_accuracy.values()))
            
            return WorkloadPrediction(
                prediction_time=datetime.now() + timedelta(minutes=horizon_minutes),
                predicted_cpu=max(0, min(100, predicted_cpu)),
                predicted_memory_mb=max(0, predicted_memory_mb),
                predicted_requests_per_second=max(0, predicted_requests),
                predicted_queue_sizes=predicted_queues,
                confidence=confidence,
                prediction_horizon_minutes=horizon_minutes,
                model_used='ml_ensemble',
                features_used=self.feature_names
            )
            
        except Exception as e:
            logger.error(f"Workload prediction failed: {e}")
            # Return fallback prediction
            return self.predict_workload(current_metrics, current_requests_per_second, horizon_minutes)


class ResourceAllocator:
    """Manages dynamic allocation of system resources."""
    
    def __init__(self, config: AllocationConfig):
        self.config = config
        
        # Current allocations
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_history = deque(maxlen=1000)
        
        # Resource pools
        self.thread_pools: Dict[str, ThreadPoolExecutor] = {}
        self.process_pools: Dict[str, ProcessPoolExecutor] = {}
        
        # Scaling state
        self.last_scaling_action = {}
        self.scaling_lock = threading.RLock()
        
        # Initialize default allocations
        self._initialize_default_allocations()
    
    def _initialize_default_allocations(self):
        """Initialize default resource allocations for components."""
        total_cpu = self.config.max_cpu_cores
        total_memory = self.config.max_memory_mb
        
        # Default allocations based on priorities
        priorities = self.config.component_priorities
        total_priority = sum(priorities.values())
        
        for component, priority in priorities.items():
            cpu_allocation = (total_cpu * priority / total_priority) * 0.7  # Reserve 30%
            memory_allocation = int((total_memory * priority / total_priority) * 0.8)  # Reserve 20%
            
            allocation = ResourceAllocation(
                component_name=component,
                allocated_cpu_cores=cpu_allocation,
                allocated_memory_mb=memory_allocation,
                thread_pool_size=min(20, max(2, int(cpu_allocation * 4))),
                process_pool_size=min(4, max(1, int(cpu_allocation))),
                cache_size_mb=min(memory_allocation // 4, 128),
                priority=priority,
                min_resources={
                    'cpu_cores': cpu_allocation * 0.3,
                    'memory_mb': memory_allocation // 4,
                    'threads': 2,
                    'processes': 1
                },
                max_resources={
                    'cpu_cores': cpu_allocation * 2.0,
                    'memory_mb': memory_allocation * 3,
                    'threads': 50,
                    'processes': 8
                }
            )
            
            self.allocations[component] = allocation
            
            # Initialize thread pools
            self.thread_pools[component] = ThreadPoolExecutor(
                max_workers=allocation.thread_pool_size,
                thread_name_prefix=f"{component}_pool"
            )
        
        logger.info(f"Initialized default resource allocations for {len(self.allocations)} components")
    
    def update_allocations(self, 
                          prediction: WorkloadPrediction,
                          current_metrics: ResourceMetrics) -> Dict[str, Any]:
        """Update resource allocations based on workload prediction."""
        update_results = {
            'allocations_updated': 0,
            'scaling_actions': [],
            'resource_adjustments': {}
        }
        
        try:
            with self.scaling_lock:
                # Check if emergency scaling is needed
                if self._is_emergency_state(current_metrics):
                    emergency_results = self._perform_emergency_scaling(current_metrics)
                    update_results['emergency_scaling'] = emergency_results
                
                # Calculate resource demand
                demand_analysis = self._analyze_resource_demand(prediction, current_metrics)
                
                # Update allocations for each component
                for component_name, allocation in self.allocations.items():
                    if not allocation.scaling_enabled:
                        continue
                    
                    component_demand = demand_analysis.get(component_name, {})
                    adjustment = self._calculate_allocation_adjustment(
                        allocation, component_demand, prediction
                    )
                    
                    if self._should_apply_adjustment(component_name, adjustment):
                        old_allocation = self._copy_allocation(allocation)
                        self._apply_allocation_adjustment(allocation, adjustment)
                        
                        update_results['allocations_updated'] += 1
                        update_results['resource_adjustments'][component_name] = {
                            'old': old_allocation.__dict__,
                            'new': allocation.__dict__,
                            'adjustment': adjustment
                        }
                        
                        # Update thread pools if needed
                        if adjustment.get('thread_pool_size'):
                            self._update_thread_pool(component_name, allocation.thread_pool_size)
                            update_results['scaling_actions'].append(
                                f"Scaled {component_name} thread pool to {allocation.thread_pool_size}"
                            )
                
                # Record allocation history
                self.allocation_history.append({
                    'timestamp': datetime.now(),
                    'prediction': prediction.__dict__,
                    'current_metrics': current_metrics.__dict__,
                    'allocations': {k: v.__dict__ for k, v in self.allocations.items()},
                    'update_results': update_results
                })
                
                return update_results
                
        except Exception as e:
            logger.error(f"Failed to update allocations: {e}")
            update_results['error'] = str(e)
            return update_results
    
    def _is_emergency_state(self, metrics: ResourceMetrics) -> bool:
        """Check if system is in emergency state requiring immediate action."""
        return (
            metrics.cpu_percent > self.config.critical_cpu_threshold * 100 or
            metrics.memory_percent > self.config.critical_memory_threshold * 100
        )
    
    def _perform_emergency_scaling(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Perform emergency scaling to handle critical resource usage."""
        emergency_actions = []
        
        try:
            # Scale up highest priority components
            high_priority_components = sorted(
                self.allocations.items(),
                key=lambda x: x[1].priority,
                reverse=True
            )
            
            for component_name, allocation in high_priority_components[:2]:  # Top 2 components
                if allocation.scaling_enabled:
                    # Emergency scale up
                    old_threads = allocation.thread_pool_size
                    allocation.thread_pool_size = min(
                        allocation.max_resources.get('threads', 50),
                        int(allocation.thread_pool_size * self.config.emergency_scale_factor)
                    )
                    
                    if allocation.thread_pool_size != old_threads:
                        self._update_thread_pool(component_name, allocation.thread_pool_size)
                        emergency_actions.append(
                            f"Emergency scaled {component_name} threads: {old_threads} -> {allocation.thread_pool_size}"
                        )
                        
                        # Record scaling action
                        self.last_scaling_action[component_name] = datetime.now()
            
            return {'actions': emergency_actions, 'triggered_at': datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Emergency scaling failed: {e}")
            return {'error': str(e)}
    
    def _analyze_resource_demand(self, 
                                prediction: WorkloadPrediction,
                                current_metrics: ResourceMetrics) -> Dict[str, Dict]:
        """Analyze resource demand for each component."""
        demand_analysis = {}
        
        # CPU demand increase
        cpu_demand_ratio = prediction.predicted_cpu / max(current_metrics.cpu_percent, 1)
        
        # Memory demand increase
        memory_demand_ratio = (
            (self.config.max_memory_mb - prediction.predicted_memory_mb) /
            max(current_metrics.memory_available_mb, 1)
        )
        
        # Request rate increase
        current_requests = current_metrics.custom_metrics.get('requests_per_second', 1)
        request_demand_ratio = prediction.predicted_requests_per_second / max(current_requests, 1)
        
        for component_name, allocation in self.allocations.items():
            # Component-specific demand based on type
            if 'search' in component_name.lower():
                # Search components are CPU and request-rate sensitive
                demand_factor = (cpu_demand_ratio * 0.5 + request_demand_ratio * 0.5)
            elif 'cache' in component_name.lower():
                # Cache components are memory sensitive
                demand_factor = memory_demand_ratio
            elif 'classifier' in component_name.lower():
                # Classifiers are CPU intensive
                demand_factor = cpu_demand_ratio
            else:
                # Default: balanced demand
                demand_factor = (cpu_demand_ratio + memory_demand_ratio + request_demand_ratio) / 3
            
            demand_analysis[component_name] = {
                'demand_factor': demand_factor,
                'cpu_pressure': cpu_demand_ratio,
                'memory_pressure': memory_demand_ratio,
                'request_pressure': request_demand_ratio,
                'predicted_queue_size': prediction.predicted_queue_sizes.get(component_name, 0)
            }
        
        return demand_analysis
    
    def _calculate_allocation_adjustment(self, 
                                       allocation: ResourceAllocation,
                                       demand: Dict,
                                       prediction: WorkloadPrediction) -> Dict[str, Any]:
        """Calculate required adjustment for resource allocation."""
        adjustment = {}
        
        demand_factor = demand.get('demand_factor', 1.0)
        
        # Determine scaling direction
        if demand_factor > self.config.scale_up_threshold:
            # Scale up
            scale_factor = min(self.config.scale_up_factor, demand_factor * 0.8)
            
            # Thread pool adjustment
            new_thread_pool_size = int(allocation.thread_pool_size * scale_factor)
            max_threads = allocation.max_resources.get('threads', self.config.max_thread_pool_size)
            adjustment['thread_pool_size'] = min(new_thread_pool_size, max_threads)
            
            # Memory adjustment  
            new_memory = int(allocation.allocated_memory_mb * scale_factor)
            max_memory = allocation.max_resources.get('memory_mb', allocation.allocated_memory_mb * 2)
            adjustment['allocated_memory_mb'] = min(new_memory, max_memory)
            
        elif demand_factor < self.config.scale_down_threshold:
            # Scale down
            scale_factor = max(self.config.scale_down_factor, demand_factor * 1.2)
            
            # Thread pool adjustment
            new_thread_pool_size = int(allocation.thread_pool_size * scale_factor)
            min_threads = allocation.min_resources.get('threads', 2)
            adjustment['thread_pool_size'] = max(new_thread_pool_size, min_threads)
            
            # Memory adjustment
            new_memory = int(allocation.allocated_memory_mb * scale_factor)
            min_memory = allocation.min_resources.get('memory_mb', allocation.allocated_memory_mb // 4)
            adjustment['allocated_memory_mb'] = max(new_memory, min_memory)
        
        # Cache size adjustment based on memory prediction
        if prediction.predicted_memory_mb < allocation.cache_size_mb * 2:
            # Increase cache if memory is available
            new_cache_size = min(
                int(allocation.cache_size_mb * 1.2),
                allocation.allocated_memory_mb // 4
            )
            adjustment['cache_size_mb'] = new_cache_size
        
        return adjustment
    
    def _should_apply_adjustment(self, component_name: str, adjustment: Dict) -> bool:
        """Check if adjustment should be applied based on scaling constraints."""
        if not adjustment:
            return False
        
        # Check minimum time between scaling actions
        last_scaling = self.last_scaling_action.get(component_name)
        if last_scaling:
            time_since_last = datetime.now() - last_scaling
            if time_since_last.total_seconds() < self.config.min_scale_interval_minutes * 60:
                return False
        
        # Check if adjustment is significant enough
        current_allocation = self.allocations[component_name]
        
        thread_change = abs(
            adjustment.get('thread_pool_size', current_allocation.thread_pool_size) -
            current_allocation.thread_pool_size
        )
        
        memory_change = abs(
            adjustment.get('allocated_memory_mb', current_allocation.allocated_memory_mb) -
            current_allocation.allocated_memory_mb
        )
        
        # Apply if change is at least 10% or 2 threads
        significant_thread_change = thread_change >= 2 or thread_change / current_allocation.thread_pool_size >= 0.1
        significant_memory_change = memory_change / current_allocation.allocated_memory_mb >= 0.1
        
        return significant_thread_change or significant_memory_change
    
    def _apply_allocation_adjustment(self, 
                                   allocation: ResourceAllocation,
                                   adjustment: Dict):
        """Apply adjustment to resource allocation."""
        for key, value in adjustment.items():
            if hasattr(allocation, key):
                setattr(allocation, key, value)
        
        # Record scaling action
        self.last_scaling_action[allocation.component_name] = datetime.now()
    
    def _copy_allocation(self, allocation: ResourceAllocation) -> ResourceAllocation:
        """Create a copy of allocation for comparison."""
        return ResourceAllocation(
            component_name=allocation.component_name,
            allocated_cpu_cores=allocation.allocated_cpu_cores,
            allocated_memory_mb=allocation.allocated_memory_mb,
            thread_pool_size=allocation.thread_pool_size,
            process_pool_size=allocation.process_pool_size,
            cache_size_mb=allocation.cache_size_mb,
            batch_size_limits=allocation.batch_size_limits.copy(),
            priority=allocation.priority,
            scaling_enabled=allocation.scaling_enabled
        )
    
    def _update_thread_pool(self, component_name: str, new_size: int):
        """Update thread pool size for a component."""
        try:
            if component_name in self.thread_pools:
                # Shutdown old pool
                old_pool = self.thread_pools[component_name]
                old_pool.shutdown(wait=False)
                
                # Create new pool with updated size
                self.thread_pools[component_name] = ThreadPoolExecutor(
                    max_workers=new_size,
                    thread_name_prefix=f"{component_name}_pool"
                )
                
                logger.debug(f"Updated {component_name} thread pool size to {new_size}")
                
        except Exception as e:
            logger.error(f"Failed to update thread pool for {component_name}: {e}")
    
    def get_current_allocations(self) -> Dict[str, Dict]:
        """Get current resource allocations."""
        return {
            component_name: allocation.__dict__
            for component_name, allocation in self.allocations.items()
        }
    
    def get_allocation_statistics(self) -> Dict[str, Any]:
        """Get allocation statistics and history."""
        total_allocated_cpu = sum(a.allocated_cpu_cores for a in self.allocations.values())
        total_allocated_memory = sum(a.allocated_memory_mb for a in self.allocations.values())
        total_threads = sum(a.thread_pool_size for a in self.allocations.values())
        
        return {
            'current_allocations': self.get_current_allocations(),
            'resource_utilization': {
                'cpu_cores_allocated': total_allocated_cpu,
                'cpu_utilization_percent': (total_allocated_cpu / self.config.max_cpu_cores) * 100,
                'memory_mb_allocated': total_allocated_memory,
                'memory_utilization_percent': (total_allocated_memory / self.config.max_memory_mb) * 100,
                'total_threads': total_threads
            },
            'scaling_statistics': {
                'components_with_scaling': sum(1 for a in self.allocations.values() if a.scaling_enabled),
                'recent_scaling_actions': len([
                    timestamp for timestamp in self.last_scaling_action.values()
                    if (datetime.now() - timestamp).total_seconds() < 3600  # Last hour
                ]),
                'allocation_history_size': len(self.allocation_history)
            }
        }


class SystemMetricsCollector:
    """Collects comprehensive system metrics."""
    
    def __init__(self, config: AllocationConfig):
        self.config = config
        self.last_network_stats = None
        self.last_disk_stats = None
        
    def collect_metrics(self, 
                       custom_metrics: Optional[Dict[str, float]] = None,
                       queue_sizes: Optional[Dict[str, int]] = None) -> ResourceMetrics:
        """Collect current system resource metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_write_mb = 0.0
            if disk_io and self.last_disk_stats:
                disk_read_mb = (disk_io.read_bytes - self.last_disk_stats.read_bytes) / (1024**2)
                disk_write_mb = (disk_io.write_bytes - self.last_disk_stats.write_bytes) / (1024**2)
            if disk_io:
                self.last_disk_stats = disk_io
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_recv_mb = network_sent_mb = 0.0
            if network_io and self.last_network_stats:
                network_recv_mb = (network_io.bytes_recv - self.last_network_stats.bytes_recv) / (1024**2)
                network_sent_mb = (network_io.bytes_sent - self.last_network_stats.bytes_sent) / (1024**2)
            if network_io:
                self.last_network_stats = network_io
            
            # Thread count
            active_threads = threading.active_count()
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024**2),
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_io_recv_mb=network_recv_mb,
                network_io_sent_mb=network_sent_mb,
                active_threads=active_threads,
                queue_sizes=queue_sizes or {},
                custom_metrics=custom_metrics or {}
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return basic fallback metrics
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=1024.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_io_recv_mb=0.0,
                network_io_sent_mb=0.0,
                active_threads=1,
                queue_sizes={},
                custom_metrics={}
            )


class DynamicResourceManager:
    """
    Main dynamic resource allocation manager with workload prediction and auto-scaling.
    
    Features:
    - ML-based workload prediction
    - Dynamic resource allocation
    - Auto-scaling of thread pools and memory
    - Performance-based optimization
    - Emergency scaling capabilities
    """
    
    def __init__(self, 
                 config: Optional[AllocationConfig] = None,
                 component_registry: Optional[Dict[str, Any]] = None):
        """
        Initialize dynamic resource manager.
        
        Args:
            config: Resource allocation configuration
            component_registry: Registry of system components
        """
        self.config = config or AllocationConfig()
        self.component_registry = component_registry or {}
        
        # Core components
        self.predictor = WorkloadPredictor(self.config)
        self.allocator = ResourceAllocator(self.config)
        self.metrics_collector = SystemMetricsCollector(self.config)
        
        # State tracking
        self.metrics_history = deque(maxlen=10000)
        self.prediction_history = deque(maxlen=1000)
        self.performance_stats = {
            'total_allocations': 0,
            'successful_predictions': 0,
            'scaling_actions': 0,
            'emergency_interventions': 0,
            'average_cpu_utilization': 0.0,
            'average_memory_utilization': 0.0
        }
        
        # Background processing
        self._running = True
        self._background_thread = None
        self._management_lock = threading.RLock()
        
        # Component integration
        self._integrated_components = {}
        
        # Start background management
        self._start_background_management()
        
        logger.info("DynamicResourceManager initialized")
    
    def register_component(self, 
                          name: str,
                          component: Any,
                          resource_callbacks: Optional[Dict[str, Callable]] = None):
        """
        Register a component for resource management.
        
        Args:
            name: Component name
            component: Component instance
            resource_callbacks: Optional callbacks for resource updates
        """
        self._integrated_components[name] = {
            'component': component,
            'callbacks': resource_callbacks or {},
            'registration_time': datetime.now()
        }
        
        # Initialize allocation if not exists
        if name not in self.allocator.allocations:
            default_allocation = ResourceAllocation(
                component_name=name,
                allocated_cpu_cores=1.0,
                allocated_memory_mb=256,
                thread_pool_size=4,
                process_pool_size=1,
                cache_size_mb=64,
                priority=0.5
            )
            self.allocator.allocations[name] = default_allocation
        
        logger.info(f"Registered component: {name}")
    
    def update_component_metrics(self, 
                               component_name: str,
                               metrics: Dict[str, float]):
        """Update custom metrics for a component."""
        # Store component-specific metrics
        if not hasattr(self, '_component_metrics'):
            self._component_metrics = defaultdict(dict)
        
        self._component_metrics[component_name].update(metrics)
    
    def get_resource_allocation(self, component_name: str) -> Optional[ResourceAllocation]:
        """Get current resource allocation for a component."""
        return self.allocator.allocations.get(component_name)
    
    def force_reallocation(self, 
                          component_specific_demands: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """Force immediate resource reallocation."""
        try:
            with self._management_lock:
                # Collect current metrics
                current_metrics = self._collect_comprehensive_metrics()
                
                # Make prediction
                prediction = self.predictor.predict_workload(
                    current_metrics,
                    current_metrics.custom_metrics.get('requests_per_second', 0)
                )
                
                # Apply any specific demands
                if component_specific_demands:
                    for component_name, demands in component_specific_demands.items():
                        if component_name in self.allocator.allocations:
                            allocation = self.allocator.allocations[component_name]
                            for key, value in demands.items():
                                if hasattr(allocation, key):
                                    setattr(allocation, key, value)
                
                # Update allocations
                results = self.allocator.update_allocations(prediction, current_metrics)
                
                # Apply allocations to integrated components
                self._apply_allocations_to_components()
                
                self.performance_stats['total_allocations'] += 1
                
                return results
                
        except Exception as e:
            logger.error(f"Forced reallocation failed: {e}")
            return {'error': str(e)}
    
    def _start_background_management(self):
        """Start background resource management thread."""
        def management_loop():
            while self._running:
                try:
                    # Collect metrics
                    current_metrics = self._collect_comprehensive_metrics()
                    self.metrics_history.append(current_metrics)
                    
                    # Add to predictor training data
                    requests_per_second = current_metrics.custom_metrics.get('requests_per_second', 0)
                    self.predictor.add_training_data(current_metrics, requests_per_second)
                    
                    # Check if reallocation is needed
                    if self._should_update_allocations(current_metrics):
                        # Make prediction
                        prediction = self.predictor.predict_workload(current_metrics, requests_per_second)
                        self.prediction_history.append(prediction)
                        
                        # Update allocations
                        results = self.allocator.update_allocations(prediction, current_metrics)
                        
                        if results['allocations_updated'] > 0:
                            self.performance_stats['scaling_actions'] += results['allocations_updated']
                            self._apply_allocations_to_components()
                            
                            logger.info(f"Updated {results['allocations_updated']} component allocations")
                        
                        if results.get('emergency_scaling'):
                            self.performance_stats['emergency_interventions'] += 1
                    
                    # Update performance stats
                    self._update_performance_stats(current_metrics)
                    
                    # Sleep until next update
                    time.sleep(self.config.metrics_collection_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Background management error: {e}")
                    time.sleep(60)  # Back off on errors
        
        self._background_thread = threading.Thread(target=management_loop, daemon=True)
        self._background_thread.start()
        logger.debug("Background resource management started")
    
    def _collect_comprehensive_metrics(self) -> ResourceMetrics:
        """Collect comprehensive system and component metrics."""
        # Collect queue sizes from integrated components
        queue_sizes = {}
        custom_metrics = {}
        
        for component_name, info in self._integrated_components.items():
            component = info['component']
            
            # Try to get queue size
            if hasattr(component, 'get_queue_size'):
                try:
                    queue_sizes[component_name] = component.get_queue_size()
                except:
                    pass
            
            # Try to get custom metrics
            if hasattr(component, 'get_metrics'):
                try:
                    comp_metrics = component.get_metrics()
                    if isinstance(comp_metrics, dict):
                        for key, value in comp_metrics.items():
                            if isinstance(value, (int, float)):
                                custom_metrics[f"{component_name}_{key}"] = float(value)
                except:
                    pass
        
        # Add component-specific metrics if available
        if hasattr(self, '_component_metrics'):
            for component_name, metrics in self._component_metrics.items():
                for key, value in metrics.items():
                    custom_metrics[f"{component_name}_{key}"] = value
        
        return self.metrics_collector.collect_metrics(custom_metrics, queue_sizes)
    
    def _should_update_allocations(self, metrics: ResourceMetrics) -> bool:
        """Check if allocations should be updated."""
        # Always update if in emergency state
        if self.allocator._is_emergency_state(metrics):
            return True
        
        # Update based on time interval
        if not hasattr(self, '_last_allocation_update'):
            self._last_allocation_update = datetime.now()
            return True
        
        time_since_last = datetime.now() - self._last_allocation_update
        if time_since_last.total_seconds() >= self.config.allocation_update_interval_minutes * 60:
            self._last_allocation_update = datetime.now()
            return True
        
        # Update if significant resource pressure
        high_cpu = metrics.cpu_percent > 70
        high_memory = metrics.memory_percent > 80
        high_queues = any(size > 100 for size in metrics.queue_sizes.values())
        
        return high_cpu or high_memory or high_queues
    
    def _apply_allocations_to_components(self):
        """Apply resource allocations to integrated components."""
        for component_name, info in self._integrated_components.items():
            try:
                allocation = self.allocator.allocations.get(component_name)
                if not allocation:
                    continue
                
                callbacks = info.get('callbacks', {})
                
                # Apply thread pool size
                if 'update_thread_pool' in callbacks:
                    callbacks['update_thread_pool'](allocation.thread_pool_size)
                
                # Apply cache size
                if 'update_cache_size' in callbacks:
                    callbacks['update_cache_size'](allocation.cache_size_mb)
                
                # Apply batch size limits
                if 'update_batch_limits' in callbacks and allocation.batch_size_limits:
                    callbacks['update_batch_limits'](allocation.batch_size_limits)
                
                # Apply memory allocation
                if 'update_memory_limit' in callbacks:
                    callbacks['update_memory_limit'](allocation.allocated_memory_mb)
                
            except Exception as e:
                logger.error(f"Failed to apply allocation to {component_name}: {e}")
    
    def _update_performance_stats(self, metrics: ResourceMetrics):
        """Update performance statistics."""
        self.performance_stats['average_cpu_utilization'] = (
            self.performance_stats['average_cpu_utilization'] * 0.95 + 
            metrics.cpu_percent * 0.05
        )
        
        self.performance_stats['average_memory_utilization'] = (
            self.performance_stats['average_memory_utilization'] * 0.95 + 
            metrics.memory_percent * 0.05
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_metrics = self._collect_comprehensive_metrics()
        
        return {
            'current_metrics': current_metrics.__dict__,
            'resource_allocations': self.allocator.get_allocation_statistics(),
            'prediction_performance': {
                'model_trained': self.predictor.is_trained,
                'prediction_accuracy': self.predictor.prediction_accuracy,
                'training_data_size': len(self.predictor.training_data)
            },
            'performance_stats': self.performance_stats,
            'integrated_components': list(self._integrated_components.keys()),
            'system_health': {
                'cpu_healthy': current_metrics.cpu_percent < 80,
                'memory_healthy': current_metrics.memory_percent < 85,
                'emergency_state': self.allocator._is_emergency_state(current_metrics)
            }
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance based on historical data."""
        optimization_results = {
            'optimizations_applied': 0,
            'improvements': [],
        }
        
        try:
            # Analyze historical performance
            if len(self.metrics_history) >= 50:
                recent_metrics = list(self.metrics_history)[-50:]
                
                # Calculate average resource utilization
                avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
                avg_memory = np.mean([m.memory_percent for m in recent_metrics])
                
                # Optimize based on patterns
                if avg_cpu < 30 and avg_memory < 40:
                    # System is underutilized, can scale down some components
                    for component_name, allocation in self.allocator.allocations.items():
                        if allocation.thread_pool_size > allocation.min_resources.get('threads', 2):
                            old_size = allocation.thread_pool_size
                            allocation.thread_pool_size = max(
                                allocation.min_resources.get('threads', 2),
                                int(allocation.thread_pool_size * 0.8)
                            )
                            
                            if allocation.thread_pool_size != old_size:
                                self.allocator._update_thread_pool(component_name, allocation.thread_pool_size)
                                optimization_results['optimizations_applied'] += 1
                                optimization_results['improvements'].append(
                                    f'Scaled down {component_name} threads: {old_size} -> {allocation.thread_pool_size}'
                                )
                
                elif avg_cpu > 75 or avg_memory > 80:
                    # System is overutilized, prioritize high-priority components
                    high_priority_components = sorted(
                        self.allocator.allocations.items(),
                        key=lambda x: x[1].priority,
                        reverse=True
                    )
                    
                    for component_name, allocation in high_priority_components[:2]:
                        if allocation.thread_pool_size < allocation.max_resources.get('threads', 50):
                            old_size = allocation.thread_pool_size
                            allocation.thread_pool_size = min(
                                allocation.max_resources.get('threads', 50),
                                int(allocation.thread_pool_size * 1.2)
                            )
                            
                            if allocation.thread_pool_size != old_size:
                                self.allocator._update_thread_pool(component_name, allocation.thread_pool_size)
                                optimization_results['optimizations_applied'] += 1
                                optimization_results['improvements'].append(
                                    f'Scaled up {component_name} threads: {old_size} -> {allocation.thread_pool_size}'
                                )
            
            # Apply optimizations to components
            if optimization_results['optimizations_applied'] > 0:
                self._apply_allocations_to_components()
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
            optimization_results['error'] = str(e)
            return optimization_results
    
    def shutdown(self):
        """Shutdown dynamic resource manager."""
        logger.info("Shutting down dynamic resource manager...")
        
        self._running = False
        
        # Wait for background thread
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5.0)
        
        # Shutdown thread pools
        for pool in self.allocator.thread_pools.values():
            pool.shutdown(wait=True)
        
        for pool in self.allocator.process_pools.values():
            pool.shutdown(wait=True)
        
        logger.info("Dynamic resource manager shutdown complete")


def create_resource_manager(
    config: Optional[AllocationConfig] = None,
    component_registry: Optional[Dict[str, Any]] = None
) -> DynamicResourceManager:
    """
    Factory function to create dynamic resource manager.
    
    Args:
        config: Optional resource allocation configuration
        component_registry: Optional component registry
        
    Returns:
        DynamicResourceManager instance
    """
    return DynamicResourceManager(config=config, component_registry=component_registry)