"""
ML Performance Monitoring Dashboard with Anomaly Detection.

Implements comprehensive performance monitoring with machine learning-based anomaly detection,
real-time dashboards, and intelligent alerting for MemMimic system optimization.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric with metadata."""
    name: str
    value: float
    timestamp: datetime
    category: str  # system, memory, search, cxd, cache, etc.
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    source_component: str = ""
    importance: float = 1.0


@dataclass
class Anomaly:
    """Detected anomaly with context and severity."""
    id: str
    metric_name: str
    value: float
    expected_value: float
    deviation_score: float
    severity: str  # low, medium, high, critical
    category: str
    timestamp: datetime
    description: str
    detection_method: str
    confidence: float
    suggested_actions: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)


@dataclass
class DashboardConfig:
    """Configuration for ML performance dashboard."""
    # Data collection
    metrics_buffer_size: int = 10000
    anomaly_buffer_size: int = 1000
    collection_interval_seconds: int = 30
    
    # Anomaly detection
    anomaly_detection_window: int = 100  # Number of recent points to analyze
    isolation_forest_contamination: float = 0.05
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    zscore_threshold: float = 3.0
    
    # Dashboard settings
    dashboard_update_interval_seconds: int = 10
    max_chart_points: int = 500
    retention_hours: int = 168  # 1 week
    
    # Alerting
    enable_alerting: bool = True
    alert_cooldown_minutes: int = 15
    critical_alert_threshold: float = 0.9
    
    # ML models
    retrain_interval_hours: int = 12
    min_training_points: int = 100
    model_save_path: str = "models/performance_monitoring"
    
    # Performance tracking
    track_system_metrics: bool = True
    track_component_metrics: bool = True
    track_custom_metrics: bool = True


class AnomalyDetector:
    """Machine learning based anomaly detector for performance metrics."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        
        # ML models for different types of anomaly detection
        self.isolation_forest = IsolationForest(
            contamination=config.isolation_forest_contamination,
            random_state=42
        )
        self.dbscan = DBSCAN(
            eps=config.dbscan_eps,
            min_samples=config.dbscan_min_samples
        )
        
        # Feature scaling
        self.scaler = RobustScaler()
        
        # Model state
        self.is_trained = False
        self.last_training = None
        self.training_data = deque(maxlen=config.metrics_buffer_size)
        
        # Anomaly tracking
        self.detected_anomalies = deque(maxlen=config.anomaly_buffer_size)
        self.anomaly_history = defaultdict(list)
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'detection_accuracy': 0.0
        }
    
    def add_metrics_batch(self, metrics: List[PerformanceMetric]):
        """Add a batch of metrics for analysis."""
        try:
            # Convert metrics to feature vectors
            for metric in metrics:
                feature_vector = self._metric_to_features(metric, metrics)
                
                training_point = {
                    'timestamp': metric.timestamp,
                    'metric_name': metric.name,
                    'features': feature_vector,
                    'value': metric.value,
                    'category': metric.category
                }
                
                self.training_data.append(training_point)
            
            # Retrain if needed
            if self._should_retrain():
                self._train_models()
            
        except Exception as e:
            logger.error(f"Failed to add metrics batch: {e}")
    
    def detect_anomalies(self, 
                        current_metrics: List[PerformanceMetric]) -> List[Anomaly]:
        """Detect anomalies in current metrics using multiple methods."""
        anomalies = []
        
        try:
            if not current_metrics:
                return anomalies
            
            # Statistical anomaly detection
            statistical_anomalies = self._detect_statistical_anomalies(current_metrics)
            anomalies.extend(statistical_anomalies)
            
            # ML-based anomaly detection (if trained)
            if self.is_trained and len(current_metrics) > 1:
                ml_anomalies = self._detect_ml_anomalies(current_metrics)
                anomalies.extend(ml_anomalies)
            
            # Pattern-based anomaly detection
            pattern_anomalies = self._detect_pattern_anomalies(current_metrics)
            anomalies.extend(pattern_anomalies)
            
            # Update anomaly tracking
            for anomaly in anomalies:
                self.detected_anomalies.append(anomaly)
                self.anomaly_history[anomaly.metric_name].append(anomaly)
                self.detection_stats['total_detections'] += 1
            
            # Remove duplicates and rank by severity
            unique_anomalies = self._deduplicate_anomalies(anomalies)
            ranked_anomalies = sorted(
                unique_anomalies, 
                key=lambda a: (a.severity == 'critical', a.confidence),
                reverse=True
            )
            
            return ranked_anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def _metric_to_features(self, 
                           metric: PerformanceMetric,
                           context_metrics: List[PerformanceMetric]) -> np.ndarray:
        """Convert metric to feature vector for ML analysis."""
        features = [
            metric.value,
            metric.timestamp.hour / 23.0,
            metric.timestamp.weekday() / 6.0,
            metric.importance,
            len(metric.tags),
        ]
        
        # Add context features from related metrics
        same_category_metrics = [
            m for m in context_metrics 
            if m.category == metric.category and m.name != metric.name
        ]
        
        if same_category_metrics:
            features.extend([
                np.mean([m.value for m in same_category_metrics]),
                np.std([m.value for m in same_category_metrics]),
                len(same_category_metrics)
            ])
        else:
            features.extend([0.0, 0.0, 0])
        
        return np.array(features)
    
    def _should_retrain(self) -> bool:
        """Check if models should be retrained."""
        if not self.is_trained and len(self.training_data) >= self.config.min_training_points:
            return True
        
        if (self.last_training and 
            (datetime.now() - self.last_training).total_seconds() > 
            self.config.retrain_interval_hours * 3600):
            return True
        
        return False
    
    def _train_models(self):
        """Train anomaly detection models."""
        try:
            if len(self.training_data) < self.config.min_training_points:
                return
            
            # Prepare training data
            features_list = []
            for point in list(self.training_data)[-self.config.min_training_points*2:]:
                features_list.append(point['features'])
            
            if not features_list:
                return
            
            X = np.array(features_list)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Isolation Forest
            self.isolation_forest.fit(X_scaled)
            
            # Train DBSCAN (unsupervised, no explicit training needed)
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            logger.info(f"Trained anomaly detection models with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def _detect_statistical_anomalies(self, 
                                    metrics: List[PerformanceMetric]) -> List[Anomaly]:
        """Detect anomalies using statistical methods."""
        anomalies = []
        
        try:
            # Group metrics by name for time series analysis
            metric_groups = defaultdict(list)
            for metric in metrics:
                metric_groups[metric.name].append(metric)
            
            # Analyze each metric group
            for metric_name, metric_list in metric_groups.items():
                if len(metric_list) < 3:  # Need at least 3 points
                    continue
                
                values = [m.value for m in metric_list]
                latest_metric = metric_list[-1]  # Most recent
                
                # Z-score based detection
                if len(values) > 1:
                    z_scores = np.abs(zscore(values))
                    if z_scores[-1] > self.config.zscore_threshold:
                        anomaly = Anomaly(
                            id=f"zscore_{metric_name}_{int(time.time())}",
                            metric_name=metric_name,
                            value=latest_metric.value,
                            expected_value=np.mean(values[:-1]),
                            deviation_score=z_scores[-1],
                            severity=self._calculate_severity(z_scores[-1] / self.config.zscore_threshold),
                            category=latest_metric.category,
                            timestamp=latest_metric.timestamp,
                            description=f"Statistical anomaly: {metric_name} value {latest_metric.value:.2f} deviates {z_scores[-1]:.2f} standard deviations from normal",
                            detection_method="z_score",
                            confidence=min(0.95, z_scores[-1] / self.config.zscore_threshold)
                        )
                        anomalies.append(anomaly)
                
                # Interquartile range (IQR) based detection
                if len(values) >= 5:
                    Q1, Q3 = np.percentile(values, [25, 75])
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    if latest_metric.value < lower_bound or latest_metric.value > upper_bound:
                        deviation = max(
                            abs(latest_metric.value - lower_bound),
                            abs(latest_metric.value - upper_bound)
                        )
                        
                        anomaly = Anomaly(
                            id=f"iqr_{metric_name}_{int(time.time())}",
                            metric_name=metric_name,
                            value=latest_metric.value,
                            expected_value=(Q1 + Q3) / 2,
                            deviation_score=deviation / IQR if IQR > 0 else 0,
                            severity=self._calculate_severity(deviation / max(IQR, 1)),
                            category=latest_metric.category,
                            timestamp=latest_metric.timestamp,
                            description=f"IQR anomaly: {metric_name} value {latest_metric.value:.2f} outside expected range [{lower_bound:.2f}, {upper_bound:.2f}]",
                            detection_method="iqr",
                            confidence=0.8
                        )
                        anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
        
        return anomalies
    
    def _detect_ml_anomalies(self, metrics: List[PerformanceMetric]) -> List[Anomaly]:
        """Detect anomalies using trained ML models."""
        anomalies = []
        
        try:
            # Prepare features for current metrics
            features_list = []
            for metric in metrics:
                features = self._metric_to_features(metric, metrics)
                features_list.append(features)
            
            if not features_list:
                return anomalies
            
            X = np.array(features_list)
            X_scaled = self.scaler.transform(X)
            
            # Isolation Forest predictions
            if_predictions = self.isolation_forest.predict(X_scaled)
            if_scores = self.isolation_forest.decision_function(X_scaled)
            
            for i, (metric, prediction, score) in enumerate(zip(metrics, if_predictions, if_scores)):
                if prediction == -1:  # Anomaly detected
                    anomaly = Anomaly(
                        id=f"isolation_{metric.name}_{int(time.time())}",
                        metric_name=metric.name,
                        value=metric.value,
                        expected_value=0.0,  # Not available from Isolation Forest
                        deviation_score=abs(score),
                        severity=self._calculate_severity(abs(score)),
                        category=metric.category,
                        timestamp=metric.timestamp,
                        description=f"ML anomaly: {metric.name} flagged by Isolation Forest (score: {score:.3f})",
                        detection_method="isolation_forest",
                        confidence=min(0.9, abs(score) * 2)
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")
        
        return anomalies
    
    def _detect_pattern_anomalies(self, metrics: List[PerformanceMetric]) -> List[Anomaly]:
        """Detect anomalies based on patterns and relationships."""
        anomalies = []
        
        try:
            # Check for sudden spikes or drops
            metric_groups = defaultdict(list)
            for metric in metrics:
                if len(self.anomaly_history[metric.name]) > 0:
                    # Compare with recent history
                    recent_values = [
                        a.value for a in self.anomaly_history[metric.name][-10:]
                        if (datetime.now() - a.timestamp).total_seconds() < 3600  # Last hour
                    ]
                    
                    if recent_values:
                        avg_recent = np.mean(recent_values)
                        if abs(metric.value - avg_recent) / max(avg_recent, 1) > 2.0:  # 200% change
                            anomaly = Anomaly(
                                id=f"pattern_{metric.name}_{int(time.time())}",
                                metric_name=metric.name,
                                value=metric.value,
                                expected_value=avg_recent,
                                deviation_score=abs(metric.value - avg_recent) / max(avg_recent, 1),
                                severity=self._calculate_severity(2.0),
                                category=metric.category,
                                timestamp=metric.timestamp,
                                description=f"Pattern anomaly: {metric.name} shows {abs(metric.value - avg_recent) / max(avg_recent, 1) * 100:.1f}% change from recent average",
                                detection_method="pattern_analysis",
                                confidence=0.7
                            )
                            anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
        
        return anomalies
    
    def _calculate_severity(self, deviation_score: float) -> str:
        """Calculate severity level based on deviation score."""
        if deviation_score >= 3.0:
            return "critical"
        elif deviation_score >= 2.0:
            return "high"
        elif deviation_score >= 1.5:
            return "medium"
        else:
            return "low"
    
    def _deduplicate_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Remove duplicate anomalies and keep the most severe."""
        unique_anomalies = {}
        
        for anomaly in anomalies:
            key = f"{anomaly.metric_name}_{anomaly.detection_method}"
            
            if key not in unique_anomalies:
                unique_anomalies[key] = anomaly
            else:
                # Keep the one with higher confidence
                if anomaly.confidence > unique_anomalies[key].confidence:
                    unique_anomalies[key] = anomaly
        
        return list(unique_anomalies.values())


class MetricsCollector:
    """Collects performance metrics from various MemMimic components."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.registered_sources = {}
        self.custom_collectors = []
        
    def register_source(self, 
                       name: str, 
                       component: Any, 
                       metric_extractor: Optional[Callable] = None):
        """Register a component as a metrics source."""
        self.registered_sources[name] = {
            'component': component,
            'extractor': metric_extractor or self._default_extractor,
            'last_collection': None
        }
    
    def add_custom_collector(self, collector_func: Callable[[], List[PerformanceMetric]]):
        """Add a custom metrics collector function."""
        self.custom_collectors.append(collector_func)
    
    def collect_all_metrics(self) -> List[PerformanceMetric]:
        """Collect metrics from all registered sources."""
        all_metrics = []
        
        # System metrics
        if self.config.track_system_metrics:
            system_metrics = self._collect_system_metrics()
            all_metrics.extend(system_metrics)
        
        # Component metrics
        if self.config.track_component_metrics:
            for source_name, source_info in self.registered_sources.items():
                try:
                    component_metrics = source_info['extractor'](
                        source_name, source_info['component']
                    )
                    all_metrics.extend(component_metrics)
                    source_info['last_collection'] = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Failed to collect metrics from {source_name}: {e}")
        
        # Custom metrics
        if self.config.track_custom_metrics:
            for collector in self.custom_collectors:
                try:
                    custom_metrics = collector()
                    all_metrics.extend(custom_metrics)
                except Exception as e:
                    logger.error(f"Custom metrics collection failed: {e}")
        
        return all_metrics
    
    def _collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect system-level performance metrics."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            import psutil
            
            # CPU metrics
            metrics.append(PerformanceMetric(
                name="system_cpu_percent",
                value=psutil.cpu_percent(interval=0.1),
                timestamp=timestamp,
                category="system",
                unit="%",
                source_component="system"
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.extend([
                PerformanceMetric(
                    name="system_memory_percent",
                    value=memory.percent,
                    timestamp=timestamp,
                    category="system",
                    unit="%",
                    source_component="system"
                ),
                PerformanceMetric(
                    name="system_memory_available_mb",
                    value=memory.available / (1024**2),
                    timestamp=timestamp,
                    category="system", 
                    unit="MB",
                    source_component="system"
                )
            ])
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.append(PerformanceMetric(
                name="system_disk_usage_percent",
                value=(disk.used / disk.total) * 100,
                timestamp=timestamp,
                category="system",
                unit="%",
                source_component="system"
            ))
            
            # Process metrics
            metrics.extend([
                PerformanceMetric(
                    name="system_active_threads",
                    value=threading.active_count(),
                    timestamp=timestamp,
                    category="system",
                    unit="count",
                    source_component="system"
                ),
                PerformanceMetric(
                    name="system_process_count",
                    value=len(psutil.pids()),
                    timestamp=timestamp,
                    category="system",
                    unit="count", 
                    source_component="system"
                )
            ])
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
        
        return metrics
    
    def _default_extractor(self, 
                          source_name: str, 
                          component: Any) -> List[PerformanceMetric]:
        """Default metrics extractor for components."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Try common metric methods
            if hasattr(component, 'get_metrics'):
                component_metrics = component.get_metrics()
                if isinstance(component_metrics, dict):
                    for key, value in component_metrics.items():
                        if isinstance(value, (int, float)):
                            metrics.append(PerformanceMetric(
                                name=f"{source_name}_{key}",
                                value=float(value),
                                timestamp=timestamp,
                                category=source_name.split('_')[0],  # Use prefix as category
                                source_component=source_name
                            ))
            
            # Try performance stats methods
            if hasattr(component, 'get_performance_stats'):
                perf_stats = component.get_performance_stats()
                if isinstance(perf_stats, dict):
                    for key, value in perf_stats.items():
                        if isinstance(value, (int, float)):
                            metrics.append(PerformanceMetric(
                                name=f"{source_name}_perf_{key}",
                                value=float(value),
                                timestamp=timestamp,
                                category="performance",
                                source_component=source_name
                            ))
            
            # Try stats methods
            if hasattr(component, 'get_stats'):
                stats = component.get_stats()
                if isinstance(stats, dict):
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            metrics.append(PerformanceMetric(
                                name=f"{source_name}_stat_{key}",
                                value=float(value),
                                timestamp=timestamp,
                                category="statistics",
                                source_component=source_name
                            ))
        
        except Exception as e:
            logger.error(f"Default extraction failed for {source_name}: {e}")
        
        return metrics


class DashboardGenerator:
    """Generates visual dashboards and reports."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.metrics_history = deque(maxlen=config.metrics_buffer_size)
        self.anomalies_history = deque(maxlen=config.anomaly_buffer_size)
        
    def update_data(self, 
                   metrics: List[PerformanceMetric],
                   anomalies: List[Anomaly]):
        """Update dashboard data with new metrics and anomalies."""
        # Store metrics with timestamp grouping
        timestamp_group = datetime.now().replace(second=0, microsecond=0)  # Group by minute
        
        metrics_by_timestamp = {
            'timestamp': timestamp_group,
            'metrics': {metric.name: metric for metric in metrics}
        }
        self.metrics_history.append(metrics_by_timestamp)
        
        # Store anomalies
        for anomaly in anomalies:
            self.anomalies_history.append(anomaly)
        
        # Clean old data
        self._clean_old_data()
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate dashboard data for visualization."""
        try:
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'summary': self._generate_summary(),
                'charts': self._generate_chart_data(),
                'anomalies': self._generate_anomaly_data(),
                'alerts': self._generate_alerts(),
                'system_health': self._generate_health_overview()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.metrics_history:
            return {'error': 'No metrics data available'}
        
        # Get latest metrics
        latest_metrics_group = list(self.metrics_history)[-1]
        latest_metrics = latest_metrics_group['metrics']
        
        # Calculate summary stats
        total_metrics = len(latest_metrics)
        categories = set(m.category for m in latest_metrics.values())
        
        # Anomaly summary
        recent_anomalies = [
            a for a in self.anomalies_history
            if (datetime.now() - a.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        anomaly_by_severity = defaultdict(int)
        for anomaly in recent_anomalies:
            anomaly_by_severity[anomaly.severity] += 1
        
        return {
            'total_metrics': total_metrics,
            'categories': list(categories),
            'recent_anomalies': len(recent_anomalies),
            'anomalies_by_severity': dict(anomaly_by_severity),
            'data_points': len(self.metrics_history),
            'time_range_hours': self._get_time_range_hours()
        }
    
    def _generate_chart_data(self) -> Dict[str, Any]:
        """Generate data for various charts."""
        charts = {}
        
        try:
            # Time series data for key metrics
            charts['time_series'] = self._generate_time_series_data()
            
            # Category breakdown
            charts['category_breakdown'] = self._generate_category_breakdown()
            
            # Anomaly trends
            charts['anomaly_trends'] = self._generate_anomaly_trends()
            
            # Performance heatmap
            charts['performance_heatmap'] = self._generate_performance_heatmap()
            
        except Exception as e:
            logger.error(f"Chart data generation failed: {e}")
            charts['error'] = str(e)
        
        return charts
    
    def _generate_time_series_data(self) -> Dict[str, Any]:
        """Generate time series data for key metrics."""
        time_series = defaultdict(lambda: {'timestamps': [], 'values': []})
        
        # Get recent data points
        recent_points = list(self.metrics_history)[-self.config.max_chart_points:]
        
        # Priority metrics to always include
        priority_metrics = {
            'system_cpu_percent', 'system_memory_percent', 
            'search_engine_response_time', 'cache_hit_rate'
        }
        
        for point in recent_points:
            timestamp = point['timestamp'].isoformat()
            
            for metric_name, metric in point['metrics'].items():
                # Include priority metrics or top metrics by category
                if (metric_name in priority_metrics or 
                    metric.category in ['system', 'performance', 'search']):
                    
                    time_series[metric_name]['timestamps'].append(timestamp)
                    time_series[metric_name]['values'].append(metric.value)
                    time_series[metric_name]['category'] = metric.category
                    time_series[metric_name]['unit'] = metric.unit
        
        return dict(time_series)
    
    def _generate_category_breakdown(self) -> Dict[str, Any]:
        """Generate breakdown by metric category."""
        if not self.metrics_history:
            return {}
        
        latest_metrics = list(self.metrics_history)[-1]['metrics']
        
        category_breakdown = defaultdict(lambda: {'count': 0, 'avg_value': 0.0, 'metrics': []})
        
        for metric in latest_metrics.values():
            category_breakdown[metric.category]['count'] += 1
            category_breakdown[metric.category]['metrics'].append({
                'name': metric.name,
                'value': metric.value,
                'unit': metric.unit
            })
        
        # Calculate averages
        for category, data in category_breakdown.items():
            if data['metrics']:
                data['avg_value'] = np.mean([m['value'] for m in data['metrics']])
        
        return dict(category_breakdown)
    
    def _generate_anomaly_trends(self) -> Dict[str, Any]:
        """Generate anomaly trend data."""
        # Group anomalies by hour
        anomaly_trends = defaultdict(lambda: defaultdict(int))
        
        for anomaly in self.anomalies_history:
            hour_key = anomaly.timestamp.strftime('%Y-%m-%d %H:00')
            anomaly_trends[hour_key][anomaly.severity] += 1
            anomaly_trends[hour_key]['total'] += 1
        
        # Convert to chart format
        trend_data = {
            'timestamps': sorted(anomaly_trends.keys()),
            'series': {
                'critical': [],
                'high': [],
                'medium': [], 
                'low': [],
                'total': []
            }
        }
        
        for timestamp in trend_data['timestamps']:
            data = anomaly_trends[timestamp]
            trend_data['series']['critical'].append(data.get('critical', 0))
            trend_data['series']['high'].append(data.get('high', 0))
            trend_data['series']['medium'].append(data.get('medium', 0))
            trend_data['series']['low'].append(data.get('low', 0))
            trend_data['series']['total'].append(data.get('total', 0))
        
        return trend_data
    
    def _generate_performance_heatmap(self) -> Dict[str, Any]:
        """Generate performance heatmap data."""
        if len(self.metrics_history) < 2:
            return {}
        
        # Get metrics for heatmap (last 24 hours, hourly buckets)
        now = datetime.now()
        heatmap_data = defaultdict(lambda: defaultdict(list))
        
        for point in self.metrics_history:
            hour_key = point['timestamp'].strftime('%H:00')
            
            for metric_name, metric in point['metrics'].items():
                if metric.category in ['system', 'performance']:
                    heatmap_data[metric_name][hour_key].append(metric.value)
        
        # Calculate averages for each hour
        heatmap_matrix = {}
        for metric_name, hours_data in heatmap_data.items():
            heatmap_matrix[metric_name] = {}
            for hour, values in hours_data.items():
                heatmap_matrix[metric_name][hour] = np.mean(values) if values else 0
        
        return {
            'matrix': heatmap_matrix,
            'hours': [f"{h:02d}:00" for h in range(24)],
            'metrics': list(heatmap_matrix.keys())
        }
    
    def _generate_anomaly_data(self) -> List[Dict]:
        """Generate recent anomaly data for display."""
        recent_anomalies = [
            a for a in self.anomalies_history
            if (datetime.now() - a.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        return [asdict(anomaly) for anomaly in recent_anomalies[-50:]]  # Last 50
    
    def _generate_alerts(self) -> List[Dict]:
        """Generate current alerts."""
        alerts = []
        
        # Critical anomalies become alerts
        critical_anomalies = [
            a for a in self.anomalies_history
            if (a.severity == 'critical' and 
                (datetime.now() - a.timestamp).total_seconds() < 1800)  # Last 30 min
        ]
        
        for anomaly in critical_anomalies:
            alerts.append({
                'id': anomaly.id,
                'title': f"Critical Anomaly: {anomaly.metric_name}",
                'message': anomaly.description,
                'timestamp': anomaly.timestamp.isoformat(),
                'severity': anomaly.severity,
                'suggested_actions': anomaly.suggested_actions
            })
        
        return alerts
    
    def _generate_health_overview(self) -> Dict[str, Any]:
        """Generate system health overview."""
        if not self.metrics_history:
            return {'status': 'unknown'}
        
        latest_metrics = list(self.metrics_history)[-1]['metrics']
        
        # Health indicators
        health_scores = []
        
        # CPU health
        cpu_metrics = [m for m in latest_metrics.values() if 'cpu' in m.name.lower()]
        if cpu_metrics:
            avg_cpu = np.mean([m.value for m in cpu_metrics])
            cpu_health = 1.0 - min(avg_cpu / 100.0, 1.0)
            health_scores.append(cpu_health)
        
        # Memory health
        memory_metrics = [m for m in latest_metrics.values() if 'memory' in m.name.lower()]
        if memory_metrics:
            # Assume lower memory usage is better for available memory metrics
            memory_values = [m.value for m in memory_metrics if 'available' not in m.name.lower()]
            if memory_values:
                avg_memory = np.mean(memory_values)
                memory_health = 1.0 - min(avg_memory / 100.0, 1.0)
                health_scores.append(memory_health)
        
        # Anomaly impact on health
        recent_anomalies = [
            a for a in self.anomalies_history
            if (datetime.now() - a.timestamp).total_seconds() < 1800  # Last 30 min
        ]
        
        anomaly_impact = len(recent_anomalies) * 0.1  # Each anomaly reduces health by 10%
        anomaly_health = max(0.0, 1.0 - anomaly_impact)
        health_scores.append(anomaly_health)
        
        # Overall health score
        overall_health = np.mean(health_scores) if health_scores else 0.5
        
        # Determine status
        if overall_health >= 0.8:
            status = 'healthy'
        elif overall_health >= 0.6:
            status = 'warning'
        elif overall_health >= 0.4:
            status = 'degraded'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'overall_score': overall_health,
            'component_health': {
                'cpu': health_scores[0] if len(health_scores) > 0 else 0.5,
                'memory': health_scores[1] if len(health_scores) > 1 else 0.5,
                'anomalies': anomaly_health
            },
            'recent_anomaly_count': len(recent_anomalies),
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_time_range_hours(self) -> float:
        """Get the time range of stored data in hours."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        oldest = self.metrics_history[0]['timestamp']
        newest = self.metrics_history[-1]['timestamp']
        return (newest - oldest).total_seconds() / 3600.0
    
    def _clean_old_data(self):
        """Clean old data beyond retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.retention_hours)
        
        # Clean metrics history
        while (self.metrics_history and 
               self.metrics_history[0]['timestamp'] < cutoff_time):
            self.metrics_history.popleft()
        
        # Clean anomalies history
        while (self.anomalies_history and 
               self.anomalies_history[0].timestamp < cutoff_time):
            self.anomalies_history.popleft()


class MLPerformanceDashboard:
    """
    Main ML Performance Monitoring Dashboard with comprehensive anomaly detection.
    
    Features:
    - Real-time metrics collection from multiple sources
    - ML-based anomaly detection with multiple algorithms
    - Interactive dashboard with charts and visualizations
    - Intelligent alerting and suggested actions
    - Performance trend analysis and prediction
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize ML performance dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        
        # Core components
        self.metrics_collector = MetricsCollector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.dashboard_generator = DashboardGenerator(self.config)
        
        # State management
        self._running = False
        self._collection_thread = None
        self._dashboard_thread = None
        self._dashboard_lock = threading.RLock()
        
        # Alert management
        self._alert_history = deque(maxlen=1000)
        self._alert_cooldowns = defaultdict(datetime)
        
        # Performance tracking
        self.performance_stats = {
            'metrics_collected': 0,
            'anomalies_detected': 0,
            'dashboards_generated': 0,
            'alerts_sent': 0,
            'uptime_start': datetime.now()
        }
        
        # Dashboard data cache
        self._latest_dashboard_data = {}
        
        logger.info("MLPerformanceDashboard initialized")
    
    def start_monitoring(self):
        """Start performance monitoring and dashboard generation."""
        if self._running:
            logger.warning("Dashboard already running")
            return
        
        self._running = True
        
        # Start collection thread
        self._collection_thread = threading.Thread(
            target=self._metrics_collection_loop,
            daemon=True
        )
        self._collection_thread.start()
        
        # Start dashboard generation thread
        self._dashboard_thread = threading.Thread(
            target=self._dashboard_generation_loop,
            daemon=True
        )
        self._dashboard_thread.start()
        
        logger.info("ML Performance Dashboard monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._running = False
        
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        
        if self._dashboard_thread:
            self._dashboard_thread.join(timeout=5.0)
        
        logger.info("ML Performance Dashboard monitoring stopped")
    
    def register_component(self, 
                          name: str,
                          component: Any,
                          metric_extractor: Optional[Callable] = None):
        """Register a component for metrics collection."""
        self.metrics_collector.register_source(name, component, metric_extractor)
        logger.info(f"Registered component for monitoring: {name}")
    
    def add_custom_collector(self, collector_func: Callable[[], List[PerformanceMetric]]):
        """Add custom metrics collector."""
        self.metrics_collector.add_custom_collector(collector_func)
        logger.info("Added custom metrics collector")
    
    def get_current_dashboard(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        with self._dashboard_lock:
            return self._latest_dashboard_data.copy()
    
    def get_recent_anomalies(self, hours: int = 24) -> List[Anomaly]:
        """Get recent anomalies within specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            anomaly for anomaly in self.anomaly_detector.detected_anomalies
            if anomaly.timestamp >= cutoff_time
        ]
    
    def force_metrics_collection(self) -> Dict[str, Any]:
        """Force immediate metrics collection and analysis."""
        try:
            # Collect metrics
            metrics = self.metrics_collector.collect_all_metrics()
            self.performance_stats['metrics_collected'] += len(metrics)
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(metrics)
            self.performance_stats['anomalies_detected'] += len(anomalies)
            
            # Update dashboard
            self.dashboard_generator.update_data(metrics, anomalies)
            
            # Generate alerts if needed
            alerts = self._process_alerts(anomalies)
            
            return {
                'metrics_collected': len(metrics),
                'anomalies_detected': len(anomalies),
                'alerts_generated': len(alerts),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Forced collection failed: {e}")
            return {'error': str(e)}
    
    def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while self._running:
            try:
                start_time = time.time()
                
                # Collect metrics
                metrics = self.metrics_collector.collect_all_metrics()
                self.performance_stats['metrics_collected'] += len(metrics)
                
                # Add to anomaly detector training data
                self.anomaly_detector.add_metrics_batch(metrics)
                
                # Detect anomalies
                anomalies = self.anomaly_detector.detect_anomalies(metrics)
                self.performance_stats['anomalies_detected'] += len(anomalies)
                
                # Update dashboard data
                self.dashboard_generator.update_data(metrics, anomalies)
                
                # Process alerts
                alerts = self._process_alerts(anomalies)
                self.performance_stats['alerts_sent'] += len(alerts)
                
                # Log collection stats
                collection_time = time.time() - start_time
                if collection_time > self.config.collection_interval_seconds * 0.5:
                    logger.warning(f"Slow metrics collection: {collection_time:.2f}s")
                
                # Sleep until next collection
                time.sleep(max(0, self.config.collection_interval_seconds - collection_time))
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(60)  # Back off on errors
    
    def _dashboard_generation_loop(self):
        """Background dashboard generation loop."""
        while self._running:
            try:
                start_time = time.time()
                
                # Generate dashboard data
                dashboard_data = self.dashboard_generator.generate_dashboard_data()
                
                with self._dashboard_lock:
                    self._latest_dashboard_data = dashboard_data
                
                self.performance_stats['dashboards_generated'] += 1
                
                # Log generation time
                generation_time = time.time() - start_time
                if generation_time > 5.0:  # Warn if takes more than 5 seconds
                    logger.warning(f"Slow dashboard generation: {generation_time:.2f}s")
                
                # Sleep until next update
                time.sleep(max(0, self.config.dashboard_update_interval_seconds - generation_time))
                
            except Exception as e:
                logger.error(f"Dashboard generation error: {e}")
                time.sleep(30)  # Back off on errors
    
    def _process_alerts(self, anomalies: List[Anomaly]) -> List[Dict]:
        """Process anomalies and generate alerts."""
        alerts = []
        
        if not self.config.enable_alerting:
            return alerts
        
        for anomaly in anomalies:
            # Check cooldown
            cooldown_key = f"{anomaly.metric_name}_{anomaly.severity}"
            last_alert = self._alert_cooldowns.get(cooldown_key)
            
            if (last_alert and 
                (datetime.now() - last_alert).total_seconds() < 
                self.config.alert_cooldown_minutes * 60):
                continue
            
            # Check if alert should be sent based on severity
            should_alert = (
                anomaly.severity == 'critical' or
                (anomaly.severity == 'high' and anomaly.confidence > 0.7) or
                (anomaly.severity == 'medium' and anomaly.confidence > self.config.critical_alert_threshold)
            )
            
            if should_alert:
                alert = {
                    'id': f"alert_{anomaly.id}",
                    'anomaly_id': anomaly.id,
                    'title': f"{anomaly.severity.upper()}: {anomaly.metric_name}",
                    'message': anomaly.description,
                    'severity': anomaly.severity,
                    'timestamp': anomaly.timestamp.isoformat(),
                    'confidence': anomaly.confidence,
                    'suggested_actions': anomaly.suggested_actions,
                    'metric_value': anomaly.value,
                    'expected_value': anomaly.expected_value
                }
                
                alerts.append(alert)
                self._alert_history.append(alert)
                self._alert_cooldowns[cooldown_key] = datetime.now()
                
                logger.warning(f"ALERT: {alert['title']} - {alert['message']}")
        
        return alerts
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        uptime = datetime.now() - self.performance_stats['uptime_start']
        
        return {
            'performance_stats': self.performance_stats,
            'uptime_hours': uptime.total_seconds() / 3600,
            'anomaly_detector_stats': {
                'is_trained': self.anomaly_detector.is_trained,
                'training_data_points': len(self.anomaly_detector.training_data),
                'detection_accuracy': self.anomaly_detector.detection_stats['detection_accuracy'],
                'total_detections': self.anomaly_detector.detection_stats['total_detections']
            },
            'dashboard_stats': {
                'metrics_history_size': len(self.dashboard_generator.metrics_history),
                'anomalies_history_size': len(self.dashboard_generator.anomalies_history),
                'time_range_hours': self.dashboard_generator._get_time_range_hours()
            },
            'alert_stats': {
                'total_alerts': len(self._alert_history),
                'active_cooldowns': len(self._alert_cooldowns),
                'recent_alerts': len([
                    a for a in self._alert_history
                    if (datetime.now() - datetime.fromisoformat(a['timestamp'])).total_seconds() < 3600
                ])
            }
        }
    
    def export_dashboard_data(self, filepath: Path) -> bool:
        """Export dashboard data to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self._latest_dashboard_data, f, indent=2, default=str)
            
            logger.info(f"Dashboard data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export dashboard data: {e}")
            return False
    
    def clear_alert_history(self):
        """Clear alert history and cooldowns."""
        self._alert_history.clear()
        self._alert_cooldowns.clear()
        logger.info("Cleared alert history and cooldowns")


def create_ml_dashboard(config: Optional[DashboardConfig] = None) -> MLPerformanceDashboard:
    """
    Factory function to create ML performance dashboard.
    
    Args:
        config: Optional dashboard configuration
        
    Returns:
        MLPerformanceDashboard instance
    """
    return MLPerformanceDashboard(config=config)