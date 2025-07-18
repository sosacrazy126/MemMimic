#!/usr/bin/env python3
"""
Advanced Analytics Dashboard - Phase 3 Advanced Features
Real-time visualization and insights for memory system analytics
"""

import sys
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import statistics

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .pattern_analyzer import create_pattern_analyzer, AnalyticsMetrics
from .predictive_manager import create_predictive_manager, PredictiveMetrics

@dataclass
class DashboardMetrics:
    """Comprehensive dashboard metrics"""
    timestamp: datetime
    system_health: str
    total_memories: int
    active_memories: int
    
    # Analytics metrics
    patterns_detected: int
    consciousness_patterns: int
    trend_analysis: Dict[str, Any]
    
    # Predictive metrics
    predictions_generated: int
    recommendations_active: int
    high_priority_actions: int
    
    # Performance metrics
    response_time_ms: float
    memory_coherence: float
    prediction_accuracy: float
    
    # Insights
    key_insights: List[str]
    recommendations: List[str]
    alerts: List[str]

class AdvancedAnalyticsDashboard:
    """
    Advanced analytics dashboard for comprehensive memory system insights
    
    Provides real-time visualization and analysis of memory patterns,
    predictions, system health, and consciousness evolution tracking.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Cache directory for dashboard data
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent / "memmimic_cache" / "dashboard"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analytics components
        self.pattern_analyzer = create_pattern_analyzer(str(self.cache_dir))
        self.predictive_manager = create_predictive_manager(str(self.cache_dir))
        
        # Dashboard configuration
        self.config = {
            'refresh_interval_seconds': 30,
            'alert_thresholds': {
                'low_system_health': 0.5,
                'high_error_rate': 0.05,
                'low_prediction_accuracy': 0.6,
                'high_memory_usage': 0.8
            },
            'insight_generators': [
                'pattern_insights',
                'consciousness_insights',
                'predictive_insights',
                'health_insights'
            ]
        }
        
        # Historical data storage
        self.metrics_history: List[DashboardMetrics] = []
        
        # Load historical data
        self._load_dashboard_data()
        
        self.logger.info("Advanced Analytics Dashboard initialized")
    
    def generate_dashboard_report(self, memory_store) -> DashboardMetrics:
        """
        Generate comprehensive dashboard report
        
        Args:
            memory_store: MemMimic memory store instance
            
        Returns:
            DashboardMetrics with comprehensive analytics
        """
        try:
            start_time = time.time()
            
            # Get analytics metrics
            analytics_metrics = self.pattern_analyzer.analyze_memory_patterns(memory_store)
            
            # Get predictive metrics
            predictive_metrics = self.predictive_manager.generate_predictions(memory_store, self.pattern_analyzer)
            
            # Generate insights and recommendations
            insights = self._generate_insights(analytics_metrics, predictive_metrics)
            recommendations = self._generate_recommendations(analytics_metrics, predictive_metrics)
            alerts = self._generate_alerts(analytics_metrics, predictive_metrics)
            
            # Calculate performance metrics
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Create dashboard metrics
            dashboard_metrics = DashboardMetrics(
                timestamp=datetime.now(),
                system_health=self._calculate_system_health(analytics_metrics, predictive_metrics),
                total_memories=analytics_metrics.total_memories,
                active_memories=analytics_metrics.active_memories,
                
                patterns_detected=analytics_metrics.patterns_detected,
                consciousness_patterns=len([p for p in self.pattern_analyzer.detected_patterns.values() 
                                          if p.pattern_type == 'consciousness']),
                trend_analysis=self._analyze_trends(analytics_metrics),
                
                predictions_generated=predictive_metrics.total_predictions,
                recommendations_active=predictive_metrics.active_recommendations,
                high_priority_actions=len(self.predictive_manager.get_high_priority_recommendations()),
                
                response_time_ms=response_time,
                memory_coherence=analytics_metrics.memory_coherence_score,
                prediction_accuracy=predictive_metrics.system_accuracy,
                
                key_insights=insights,
                recommendations=recommendations,
                alerts=alerts
            )
            
            # Store in history
            self.metrics_history.append(dashboard_metrics)
            
            # Keep only last 100 entries
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            # Save dashboard data
            self._save_dashboard_data()
            
            self.logger.info(f"Dashboard report generated in {response_time:.1f}ms")
            return dashboard_metrics
            
        except Exception as e:
            self.logger.error(f"Dashboard report generation failed: {e}")
            return self._empty_dashboard_metrics()
    
    def _calculate_system_health(self, analytics: AnalyticsMetrics, predictive: PredictiveMetrics) -> str:
        """Calculate overall system health status"""
        try:
            health_score = (
                analytics.system_health_score * 0.4 +
                predictive.system_accuracy * 0.3 +
                analytics.memory_coherence_score * 0.3
            )
            
            if health_score >= 0.8:
                return "EXCELLENT"
            elif health_score >= 0.6:
                return "GOOD"
            elif health_score >= 0.4:
                return "FAIR"
            else:
                return "NEEDS_ATTENTION"
                
        except Exception as e:
            self.logger.debug(f"System health calculation failed: {e}")
            return "UNKNOWN"
    
    def _analyze_trends(self, analytics: AnalyticsMetrics) -> Dict[str, Any]:
        """Analyze trends in memory usage and patterns"""
        try:
            trends = {
                'memory_growth': 'stable',
                'pattern_evolution': 'stable',
                'consciousness_development': 'stable',
                'system_performance': 'stable'
            }
            
            # Analyze historical data if available
            if len(self.metrics_history) >= 3:
                recent_metrics = self.metrics_history[-3:]
                
                # Memory growth trend
                memory_counts = [m.total_memories for m in recent_metrics]
                if len(set(memory_counts)) > 1:
                    if memory_counts[-1] > memory_counts[0]:
                        trends['memory_growth'] = 'increasing'
                    elif memory_counts[-1] < memory_counts[0]:
                        trends['memory_growth'] = 'decreasing'
                
                # Pattern evolution trend
                pattern_counts = [m.patterns_detected for m in recent_metrics]
                if len(set(pattern_counts)) > 1:
                    if pattern_counts[-1] > pattern_counts[0]:
                        trends['pattern_evolution'] = 'increasing'
                    elif pattern_counts[-1] < pattern_counts[0]:
                        trends['pattern_evolution'] = 'decreasing'
                
                # Consciousness development trend
                consciousness_counts = [m.consciousness_patterns for m in recent_metrics]
                if len(set(consciousness_counts)) > 1:
                    if consciousness_counts[-1] > consciousness_counts[0]:
                        trends['consciousness_development'] = 'accelerating'
                    elif consciousness_counts[-1] < consciousness_counts[0]:
                        trends['consciousness_development'] = 'stabilizing'
                
                # System performance trend
                response_times = [m.response_time_ms for m in recent_metrics]
                if len(response_times) > 1:
                    avg_response = statistics.mean(response_times)
                    if avg_response > 1000:  # > 1 second
                        trends['system_performance'] = 'degrading'
                    elif avg_response < 200:  # < 200ms
                        trends['system_performance'] = 'excellent'
            
            return trends
            
        except Exception as e:
            self.logger.debug(f"Trend analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_insights(self, analytics: AnalyticsMetrics, predictive: PredictiveMetrics) -> List[str]:
        """Generate key insights from analytics and predictive data"""
        insights = []
        
        try:
            # Pattern insights
            if analytics.patterns_detected > 0:
                insights.append(f"Detected {analytics.patterns_detected} memory usage patterns")
                
                if analytics.high_confidence_patterns > 0:
                    insights.append(f"{analytics.high_confidence_patterns} high-confidence patterns identified")
            
            # Consciousness insights
            consciousness_strength = analytics.consciousness_pattern_strength
            if consciousness_strength > 0.3:
                insights.append(f"Strong consciousness evolution patterns detected (strength: {consciousness_strength:.3f})")
            elif consciousness_strength > 0.1:
                insights.append(f"Emerging consciousness patterns observed (strength: {consciousness_strength:.3f})")
            
            # Predictive insights
            if predictive.total_predictions > 0:
                insights.append(f"Generated {predictive.total_predictions} predictions for memory lifecycle")
                
                if predictive.high_confidence_predictions > 0:
                    insights.append(f"{predictive.high_confidence_predictions} high-confidence predictions ready for action")
            
            # System health insights
            if analytics.system_health_score > 0.8:
                insights.append("System operating at optimal health levels")
            elif analytics.system_health_score < 0.5:
                insights.append("System health requires attention - check alerts")
            
            # Memory coherence insights
            if analytics.memory_coherence_score > 0.7:
                insights.append("High memory coherence indicates well-organized knowledge")
            elif analytics.memory_coherence_score < 0.3:
                insights.append("Low memory coherence suggests need for consolidation")
            
            # Trend insights
            if analytics.rising_importance_memories > 0:
                insights.append(f"{analytics.rising_importance_memories} memories showing increasing importance")
            
            if analytics.falling_importance_memories > 0:
                insights.append(f"{analytics.falling_importance_memories} memories may need archival consideration")
            
        except Exception as e:
            self.logger.debug(f"Insight generation failed: {e}")
            insights.append(f"Insight generation error: {str(e)}")
        
        return insights[:10]  # Limit to top 10 insights
    
    def _generate_recommendations(self, analytics: AnalyticsMetrics, predictive: PredictiveMetrics) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # System optimization recommendations
            if analytics.system_health_score < 0.6:
                recommendations.append("Consider system optimization - health score below threshold")
            
            # Memory management recommendations
            if predictive.active_recommendations > 0:
                recommendations.append(f"Review {predictive.active_recommendations} active lifecycle recommendations")
            
            # Pattern-based recommendations
            if analytics.patterns_detected > 0:
                recommendations.append("Leverage detected patterns for memory optimization")
            
            # Consciousness evolution recommendations
            if predictive.consciousness_predictions > 0:
                recommendations.append("Monitor consciousness evolution patterns for insights")
            
            # Performance recommendations
            if analytics.memory_coherence_score < 0.5:
                recommendations.append("Consider memory consolidation to improve coherence")
            
            # Predictive recommendations
            if predictive.prediction_coverage < 0.7:
                recommendations.append("Increase prediction coverage for better lifecycle management")
            
        except Exception as e:
            self.logger.debug(f"Recommendation generation failed: {e}")
            recommendations.append(f"Recommendation generation error: {str(e)}")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _generate_alerts(self, analytics: AnalyticsMetrics, predictive: PredictiveMetrics) -> List[str]:
        """Generate system alerts based on thresholds"""
        alerts = []
        
        try:
            # System health alerts
            if analytics.system_health_score < self.config['alert_thresholds']['low_system_health']:
                alerts.append(f"LOW SYSTEM HEALTH: {analytics.system_health_score:.3f}")
            
            # Prediction accuracy alerts
            if predictive.system_accuracy < self.config['alert_thresholds']['low_prediction_accuracy']:
                alerts.append(f"LOW PREDICTION ACCURACY: {predictive.system_accuracy:.3f}")
            
            # Memory coherence alerts
            if analytics.memory_coherence_score < 0.3:
                alerts.append(f"LOW MEMORY COHERENCE: {analytics.memory_coherence_score:.3f}")
            
            # Pattern detection alerts
            if analytics.patterns_detected == 0 and analytics.total_memories > 10:
                alerts.append("NO PATTERNS DETECTED - System may need recalibration")
            
            # Consciousness evolution alerts
            if analytics.consciousness_pattern_strength > 0.8:
                alerts.append("HIGH CONSCIOUSNESS ACTIVITY - Monitor for significant developments")
            
        except Exception as e:
            self.logger.debug(f"Alert generation failed: {e}")
            alerts.append(f"Alert generation error: {str(e)}")
        
        return alerts
    
    def _empty_dashboard_metrics(self) -> DashboardMetrics:
        """Return empty dashboard metrics for error cases"""
        return DashboardMetrics(
            timestamp=datetime.now(),
            system_health="UNKNOWN",
            total_memories=0,
            active_memories=0,
            patterns_detected=0,
            consciousness_patterns=0,
            trend_analysis={},
            predictions_generated=0,
            recommendations_active=0,
            high_priority_actions=0,
            response_time_ms=0.0,
            memory_coherence=0.0,
            prediction_accuracy=0.0,
            key_insights=[],
            recommendations=[],
            alerts=["Dashboard initialization error"]
        )
    
    def _save_dashboard_data(self):
        """Save dashboard data to cache"""
        try:
            # Save only recent metrics history
            recent_metrics = self.metrics_history[-20:] if len(self.metrics_history) > 20 else self.metrics_history
            
            data = {
                'metrics_history': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'system_health': m.system_health,
                        'total_memories': m.total_memories,
                        'active_memories': m.active_memories,
                        'patterns_detected': m.patterns_detected,
                        'consciousness_patterns': m.consciousness_patterns,
                        'trend_analysis': m.trend_analysis,
                        'predictions_generated': m.predictions_generated,
                        'recommendations_active': m.recommendations_active,
                        'high_priority_actions': m.high_priority_actions,
                        'response_time_ms': m.response_time_ms,
                        'memory_coherence': m.memory_coherence,
                        'prediction_accuracy': m.prediction_accuracy,
                        'key_insights': m.key_insights,
                        'recommendations': m.recommendations,
                        'alerts': m.alerts
                    }
                    for m in recent_metrics
                ],
                'last_updated': datetime.now().isoformat()
            }
            
            cache_file = self.cache_dir / "dashboard_data.json"
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Dashboard data saved to {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save dashboard data: {e}")
    
    def _load_dashboard_data(self):
        """Load dashboard data from cache"""
        try:
            cache_file = self.cache_dir / "dashboard_data.json"
            if not cache_file.exists():
                return
            
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Load metrics history
            for metric_data in data.get('metrics_history', []):
                metric = DashboardMetrics(
                    timestamp=datetime.fromisoformat(metric_data['timestamp']),
                    system_health=metric_data['system_health'],
                    total_memories=metric_data['total_memories'],
                    active_memories=metric_data['active_memories'],
                    patterns_detected=metric_data['patterns_detected'],
                    consciousness_patterns=metric_data['consciousness_patterns'],
                    trend_analysis=metric_data['trend_analysis'],
                    predictions_generated=metric_data['predictions_generated'],
                    recommendations_active=metric_data['recommendations_active'],
                    high_priority_actions=metric_data['high_priority_actions'],
                    response_time_ms=metric_data['response_time_ms'],
                    memory_coherence=metric_data['memory_coherence'],
                    prediction_accuracy=metric_data['prediction_accuracy'],
                    key_insights=metric_data['key_insights'],
                    recommendations=metric_data['recommendations'],
                    alerts=metric_data['alerts']
                )
                self.metrics_history.append(metric)
            
            self.logger.info(f"Loaded {len(self.metrics_history)} dashboard metrics from cache")
            
        except Exception as e:
            self.logger.warning(f"Failed to load dashboard data: {e}")
    
    def format_dashboard_report(self, metrics: DashboardMetrics) -> str:
        """Format dashboard metrics into a comprehensive report"""
        try:
            report_lines = []
            
            # Header
            report_lines.append("🚀 MEMMIMIC ADVANCED ANALYTICS DASHBOARD")
            report_lines.append("=" * 60)
            report_lines.append(f"📅 Generated: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"🎯 System Health: {metrics.system_health}")
            report_lines.append("")
            
            # Core Metrics
            report_lines.append("📊 CORE METRICS:")
            report_lines.append(f"  📚 Total Memories: {metrics.total_memories}")
            report_lines.append(f"  ⚡ Active Memories: {metrics.active_memories}")
            report_lines.append(f"  🔍 Patterns Detected: {metrics.patterns_detected}")
            report_lines.append(f"  🧠 Consciousness Patterns: {metrics.consciousness_patterns}")
            report_lines.append("")
            
            # Predictive Metrics
            report_lines.append("🔮 PREDICTIVE ANALYTICS:")
            report_lines.append(f"  📈 Predictions Generated: {metrics.predictions_generated}")
            report_lines.append(f"  💡 Active Recommendations: {metrics.recommendations_active}")
            report_lines.append(f"  🚨 High Priority Actions: {metrics.high_priority_actions}")
            report_lines.append(f"  🎯 Prediction Accuracy: {metrics.prediction_accuracy:.3f}")
            report_lines.append("")
            
            # Performance Metrics
            report_lines.append("⚡ PERFORMANCE METRICS:")
            report_lines.append(f"  ⏱️ Response Time: {metrics.response_time_ms:.1f}ms")
            report_lines.append(f"  🧩 Memory Coherence: {metrics.memory_coherence:.3f}")
            report_lines.append("")
            
            # Trend Analysis
            if metrics.trend_analysis:
                report_lines.append("📈 TREND ANALYSIS:")
                for trend_type, trend_value in metrics.trend_analysis.items():
                    if trend_type != 'error':
                        report_lines.append(f"  📊 {trend_type.replace('_', ' ').title()}: {trend_value}")
                report_lines.append("")
            
            # Key Insights
            if metrics.key_insights:
                report_lines.append("💡 KEY INSIGHTS:")
                for insight in metrics.key_insights:
                    report_lines.append(f"  • {insight}")
                report_lines.append("")
            
            # Recommendations
            if metrics.recommendations:
                report_lines.append("🎯 RECOMMENDATIONS:")
                for rec in metrics.recommendations:
                    report_lines.append(f"  ✅ {rec}")
                report_lines.append("")
            
            # Alerts
            if metrics.alerts:
                report_lines.append("🚨 ALERTS:")
                for alert in metrics.alerts:
                    report_lines.append(f"  ⚠️ {alert}")
                report_lines.append("")
            
            # Footer
            report_lines.append("🏆 Advanced Analytics Dashboard v1.0")
            report_lines.append("🔗 Powered by MemMimic AMMS with Phase 3 Analytics")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.logger.error(f"Dashboard report formatting failed: {e}")
            return f"Dashboard report formatting error: {str(e)}"
    
    def get_historical_trends(self, days: int = 7) -> Dict[str, List[Any]]:
        """Get historical trend data for visualization"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_date]
            
            if not recent_metrics:
                return {}
            
            trends = {
                'timestamps': [m.timestamp.isoformat() for m in recent_metrics],
                'total_memories': [m.total_memories for m in recent_metrics],
                'patterns_detected': [m.patterns_detected for m in recent_metrics],
                'consciousness_patterns': [m.consciousness_patterns for m in recent_metrics],
                'system_health_scores': [
                    {'EXCELLENT': 1.0, 'GOOD': 0.8, 'FAIR': 0.6, 'NEEDS_ATTENTION': 0.4, 'UNKNOWN': 0.0}
                    .get(m.system_health, 0.0) for m in recent_metrics
                ],
                'memory_coherence': [m.memory_coherence for m in recent_metrics],
                'prediction_accuracy': [m.prediction_accuracy for m in recent_metrics],
                'response_times': [m.response_time_ms for m in recent_metrics]
            }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Historical trends calculation failed: {e}")
            return {}

def create_analytics_dashboard(cache_dir: Optional[str] = None) -> AdvancedAnalyticsDashboard:
    """Create advanced analytics dashboard instance"""
    return AdvancedAnalyticsDashboard(cache_dir)

if __name__ == "__main__":
    # Test the analytics dashboard
    dashboard = create_analytics_dashboard()
    
    # Test with mock data
    from unittest.mock import Mock
    
    # Create mock memories
    mock_memories = []
    for i in range(15):
        mock_memory = Mock()
        mock_memory.id = i
        mock_memory.content = f"Test memory {i} with consciousness evolution and recursive unity patterns"
        mock_memory.importance_score = 0.2 + (i * 0.05)
        mock_memory.access_count = i % 5
        mock_memory.created_at = (datetime.now() - timedelta(days=i*2)).isoformat()
        mock_memories.append(mock_memory)
    
    # Create mock memory store
    mock_store = Mock()
    mock_store.get_all.return_value = mock_memories
    
    # Generate dashboard report
    metrics = dashboard.generate_dashboard_report(mock_store)
    
    # Format and display report
    report = dashboard.format_dashboard_report(metrics)
    print(report)
    
    # Test historical trends
    trends = dashboard.get_historical_trends(7)
    print(f"\nHistorical Trends Available: {list(trends.keys())}")