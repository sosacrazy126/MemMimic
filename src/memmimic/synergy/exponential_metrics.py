"""
Exponential Metrics - Performance tracking for cognitive synergy

Tracks and optimizes exponential collaboration performance metrics including
understanding alignment, velocity multipliers, quality scores, and pattern reusability.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque

from ..errors import get_error_logger


@dataclass
class CollaborationMetrics:
    """Real-time collaboration performance metrics"""
    understanding: float = 0.8      # How well human and AI are aligned (0-1)
    velocity: float = 1.0           # Output multiplier vs solo work (1-20x)
    quality: float = 0.8            # Solution elegance score (0-1)
    reusability: float = 0.5        # Pattern library growth rate (0-1)
    
    # Tracking data
    timestamp: float = field(default_factory=time.time)
    interaction_count: int = 0
    pattern_matches: int = 0
    successful_outcomes: int = 0
    
    def get_composite_score(self) -> float:
        """Get weighted composite performance score"""
        return (self.understanding * 0.3 + 
                min(self.velocity / 10.0, 1.0) * 0.4 + 
                self.quality * 0.2 + 
                self.reusability * 0.1)


class ExponentialMetrics:
    """
    Tracks and optimizes exponential collaboration performance
    
    Monitors real-time metrics and triggers optimization when performance
    drops below thresholds, ensuring sustained exponential value creation.
    """
    
    def __init__(self, history_size: int = 1000):
        self.logger = get_error_logger("exponential_metrics")
        
        # Current metrics
        self.current_metrics = CollaborationMetrics()
        
        # Performance history
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        
        # Optimization thresholds
        self.thresholds = {
            'understanding_min': 0.8,
            'velocity_min': 5.0,
            'quality_min': 0.7,
            'reusability_min': 0.5
        }
        
        # Performance tracking
        self.total_interactions = 0
        self.exponential_activations = 0
        self.optimization_triggers = 0
        self.pattern_evolutions = 0
        
        # Trend analysis
        self.trend_window = 10
        self.performance_trends = {
            'understanding': [],
            'velocity': [],
            'quality': [],
            'reusability': []
        }
        
        self.logger.info("Exponential Metrics tracking initialized")
    
    def update_metrics(self, understanding: float = None, velocity: float = None,
                      quality: float = None, reusability: float = None,
                      pattern_matches: int = 0, successful_outcome: bool = True) -> CollaborationMetrics:
        """Update collaboration metrics with new performance data"""
        
        # Update current metrics using exponential moving average
        alpha = 0.1  # Learning rate
        
        if understanding is not None:
            self.current_metrics.understanding = (
                (1 - alpha) * self.current_metrics.understanding + alpha * understanding
            )
        
        if velocity is not None:
            self.current_metrics.velocity = (
                (1 - alpha) * self.current_metrics.velocity + alpha * velocity
            )
        
        if quality is not None:
            self.current_metrics.quality = (
                (1 - alpha) * self.current_metrics.quality + alpha * quality
            )
        
        if reusability is not None:
            self.current_metrics.reusability = (
                (1 - alpha) * self.current_metrics.reusability + alpha * reusability
            )
        
        # Update interaction tracking
        self.current_metrics.interaction_count += 1
        self.current_metrics.pattern_matches += pattern_matches
        if successful_outcome:
            self.current_metrics.successful_outcomes += 1
        self.current_metrics.timestamp = time.time()
        
        # Add to history
        metrics_snapshot = CollaborationMetrics(
            understanding=self.current_metrics.understanding,
            velocity=self.current_metrics.velocity,
            quality=self.current_metrics.quality,
            reusability=self.current_metrics.reusability,
            timestamp=self.current_metrics.timestamp,
            interaction_count=self.current_metrics.interaction_count,
            pattern_matches=pattern_matches,
            successful_outcomes=self.current_metrics.successful_outcomes
        )
        self.metrics_history.append(metrics_snapshot)
        
        # Update trend analysis
        self._update_trends()
        
        # Check for optimization triggers
        optimization_needed = self._check_optimization_triggers()
        if optimization_needed:
            self.optimization_triggers += 1
            self.logger.info(f"Optimization trigger activated: {optimization_needed}")
        
        self.total_interactions += 1
        
        self.logger.debug(
            f"Metrics updated: U={self.current_metrics.understanding:.2f}, "
            f"V={self.current_metrics.velocity:.1f}x, Q={self.current_metrics.quality:.2f}, "
            f"R={self.current_metrics.reusability:.2f}",
            extra={
                "understanding": self.current_metrics.understanding,
                "velocity": self.current_metrics.velocity,
                "quality": self.current_metrics.quality,
                "reusability": self.current_metrics.reusability
            }
        )
        
        return self.current_metrics
    
    def _update_trends(self):
        """Update performance trend analysis"""
        if len(self.metrics_history) < 2:
            return
        
        # Calculate trends over recent window
        recent_metrics = list(self.metrics_history)[-self.trend_window:]
        
        for metric_name in self.performance_trends.keys():
            values = [getattr(m, metric_name) for m in recent_metrics]
            if len(values) >= 2:
                # Simple trend: positive if improving, negative if declining
                trend = (values[-1] - values[0]) / max(abs(values[0]), 0.001)
                self.performance_trends[metric_name] = trend
    
    def _check_optimization_triggers(self) -> Optional[str]:
        """Check if performance optimization is needed"""
        metrics = self.current_metrics
        
        if metrics.understanding < self.thresholds['understanding_min']:
            return f"Understanding below threshold: {metrics.understanding:.2f} < {self.thresholds['understanding_min']}"
        
        if metrics.velocity < self.thresholds['velocity_min']:
            return f"Velocity below threshold: {metrics.velocity:.1f}x < {self.thresholds['velocity_min']}x"
        
        if metrics.quality < self.thresholds['quality_min']:
            return f"Quality below threshold: {metrics.quality:.2f} < {self.thresholds['quality_min']}"
        
        if metrics.reusability < self.thresholds['reusability_min']:
            return f"Reusability below threshold: {metrics.reusability:.2f} < {self.thresholds['reusability_min']}"
        
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics_history:
            return {"status": "no_data", "interactions": 0}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 interactions
        avg_understanding = sum(m.understanding for m in recent_metrics) / len(recent_metrics)
        avg_velocity = sum(m.velocity for m in recent_metrics) / len(recent_metrics)
        avg_quality = sum(m.quality for m in recent_metrics) / len(recent_metrics)
        avg_reusability = sum(m.reusability for m in recent_metrics) / len(recent_metrics)
        
        composite_score = (avg_understanding * 0.3 + 
                         min(avg_velocity / 10.0, 1.0) * 0.4 + 
                         avg_quality * 0.2 + 
                         avg_reusability * 0.1)
        
        return {
            "current_metrics": {
                "understanding": self.current_metrics.understanding,
                "velocity": self.current_metrics.velocity,
                "quality": self.current_metrics.quality,
                "reusability": self.current_metrics.reusability,
                "composite_score": composite_score
            },
            "averages_recent_10": {
                "understanding": avg_understanding,
                "velocity": avg_velocity,
                "quality": avg_quality,
                "reusability": avg_reusability
            },
            "performance_trends": self.performance_trends,
            "tracking_stats": {
                "total_interactions": self.total_interactions,
                "exponential_activations": self.exponential_activations,
                "optimization_triggers": self.optimization_triggers,
                "pattern_evolutions": self.pattern_evolutions,
                "success_rate": (self.current_metrics.successful_outcomes / 
                               max(self.current_metrics.interaction_count, 1))
            },
            "threshold_status": {
                "understanding_ok": self.current_metrics.understanding >= self.thresholds['understanding_min'],
                "velocity_ok": self.current_metrics.velocity >= self.thresholds['velocity_min'],
                "quality_ok": self.current_metrics.quality >= self.thresholds['quality_min'],
                "reusability_ok": self.current_metrics.reusability >= self.thresholds['reusability_min']
            }
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Get specific optimization recommendations based on current performance"""
        recommendations = []
        metrics = self.current_metrics
        
        if metrics.understanding < self.thresholds['understanding_min']:
            recommendations.append({
                "area": "understanding",
                "issue": f"Alignment score {metrics.understanding:.2f} below target {self.thresholds['understanding_min']}",
                "recommendation": "Clarify core objectives and ensure shared context"
            })
        
        if metrics.velocity < self.thresholds['velocity_min']:
            recommendations.append({
                "area": "velocity",
                "issue": f"Velocity {metrics.velocity:.1f}x below target {self.thresholds['velocity_min']}x",
                "recommendation": "Try different collaboration approach or break down complexity"
            })
        
        if metrics.quality < self.thresholds['quality_min']:
            recommendations.append({
                "area": "quality",
                "issue": f"Quality score {metrics.quality:.2f} below target {self.thresholds['quality_min']}",
                "recommendation": "Focus on solution elegance and thorough validation"
            })
        
        if metrics.reusability < self.thresholds['reusability_min']:
            recommendations.append({
                "area": "reusability",
                "issue": f"Pattern reuse {metrics.reusability:.2f} below target {self.thresholds['reusability_min']}",
                "recommendation": "Extract more patterns and increase pattern library utilization"
            })
        
        return recommendations
    
    def record_exponential_activation(self):
        """Record an exponential collaboration mode activation"""
        self.exponential_activations += 1
        self.logger.info(f"Exponential activation recorded (total: {self.exponential_activations})")
    
    def record_pattern_evolution(self, evolved_patterns: int = 1):
        """Record pattern evolution events"""
        self.pattern_evolutions += evolved_patterns
        self.logger.debug(f"Pattern evolution recorded: +{evolved_patterns} (total: {self.pattern_evolutions})")
    
    def set_thresholds(self, **thresholds):
        """Update performance thresholds"""
        for key, value in thresholds.items():
            if key in self.thresholds:
                old_value = self.thresholds[key]
                self.thresholds[key] = value
                self.logger.info(f"Threshold updated: {key} {old_value} → {value}")
    
    def reset_metrics(self):
        """Reset all metrics and history"""
        self.current_metrics = CollaborationMetrics()
        self.metrics_history.clear()
        self.total_interactions = 0
        self.exponential_activations = 0
        self.optimization_triggers = 0
        self.pattern_evolutions = 0
        for trend_list in self.performance_trends.values():
            trend_list.clear()
        
        self.logger.info("All metrics reset to initial state")