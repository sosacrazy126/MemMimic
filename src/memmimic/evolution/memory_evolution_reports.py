"""
Memory Evolution Reporting and Visualization System

Comprehensive reporting and visualization capabilities for memory evolution analysis.
Generates detailed reports, dashboards, and insights for memory system optimization.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
from pathlib import Path

from .memory_evolution_tracker import MemoryEvolutionTracker
from .memory_lifecycle_manager import MemoryLifecycleManager, LifecycleStage
from .memory_usage_analytics import MemoryUsageAnalytics, UsagePatternType
from .memory_evolution_metrics import MemoryEvolutionMetrics, MetricCategory
from ..errors.exceptions import MemMimicError


class ReportType(Enum):
    """Types of evolution reports"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    PERFORMANCE_DASHBOARD = "performance_dashboard"
    HEALTH_ASSESSMENT = "health_assessment"
    OPTIMIZATION_GUIDE = "optimization_guide"
    TREND_ANALYSIS = "trend_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"


@dataclass
class ReportConfiguration:
    """Configuration for report generation"""
    report_type: ReportType
    format: ReportFormat = ReportFormat.JSON
    include_visualizations: bool = True
    include_recommendations: bool = True
    include_raw_data: bool = False
    memory_filter: Optional[List[str]] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    detail_level: str = "standard"  # "minimal", "standard", "detailed"


@dataclass
class VisualizationData:
    """Data structure for visualization elements"""
    chart_type: str  # "line", "bar", "pie", "scatter", "heatmap"
    title: str
    data: Dict[str, Any]
    labels: List[str]
    colors: Optional[List[str]] = None
    description: str = ""


class MemoryEvolutionReporter:
    """
    Comprehensive reporting system for memory evolution analysis and insights.
    
    Core capabilities:
    - Executive dashboards: High-level system health and performance
    - Detailed analysis: Deep-dive into memory patterns and trends
    - Health assessments: Comprehensive health scoring and diagnostics
    - Optimization guides: Actionable recommendations for improvement
    - Trend analysis: Temporal patterns and predictive insights
    - Comparative analysis: Memory-to-memory and period-to-period comparisons
    - Visualization generation: Charts, graphs, and data visualizations
    """
    
    def __init__(self,
                 evolution_tracker: MemoryEvolutionTracker,
                 lifecycle_manager: MemoryLifecycleManager,
                 usage_analytics: MemoryUsageAnalytics,
                 evolution_metrics: MemoryEvolutionMetrics,
                 output_directory: str = "reports"):
        self.evolution_tracker = evolution_tracker
        self.lifecycle_manager = lifecycle_manager
        self.usage_analytics = usage_analytics
        self.evolution_metrics = evolution_metrics
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        self._report_cache: Dict[str, Any] = {}
    
    async def generate_executive_summary(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate executive summary report"""
        system_metrics = await self.evolution_metrics.calculate_system_metrics()
        
        # Key performance indicators
        kpis = {
            "total_memories": system_metrics.total_memories,
            "avg_evolution_score": round(system_metrics.avg_evolution_score, 3),
            "healthy_memories_percentage": round(
                (system_metrics.healthy_memories / system_metrics.total_memories * 100) 
                if system_metrics.total_memories > 0 else 0, 1
            ),
            "system_evolution_velocity": round(system_metrics.evolution_velocity, 3),
            "pattern_diversity": system_metrics.pattern_diversity
        }
        
        # Health status
        health_status = "excellent"
        if kpis["avg_evolution_score"] < 0.5:
            health_status = "critical"
        elif kpis["avg_evolution_score"] < 0.7:
            health_status = "needs_attention"
        elif kpis["avg_evolution_score"] < 0.9:
            health_status = "good"
        
        # Top insights
        insights = []
        
        if kpis["healthy_memories_percentage"] < 60:
            insights.append({
                "type": "warning",
                "message": f"Only {kpis['healthy_memories_percentage']}% of memories are healthy",
                "impact": "high",
                "recommendation": "Focus on improving low-scoring memories"
            })
        
        if system_metrics.evolution_velocity < 0.1:
            insights.append({
                "type": "info",
                "message": "Low system evolution velocity detected",
                "impact": "medium",
                "recommendation": "Encourage more memory updates and refinements"
            })
        
        if system_metrics.pattern_diversity < 3:
            insights.append({
                "type": "info",
                "message": "Limited usage pattern diversity",
                "impact": "medium",
                "recommendation": "Diversify memory usage across different contexts"
            })
        
        # Visualizations
        visualizations = []
        
        if config.include_visualizations:
            # Score distribution pie chart
            visualizations.append(VisualizationData(
                chart_type="pie",
                title="Memory Score Distribution",
                data=system_metrics.score_distribution,
                labels=list(system_metrics.score_distribution.keys()),
                colors=["#28a745", "#17a2b8", "#ffc107", "#fd7e14", "#dc3545"],
                description="Distribution of memories by evolution score ranges"
            ))
            
            # Lifecycle distribution bar chart
            lifecycle_data = {stage.value: count for stage, count in system_metrics.lifecycle_distribution.items()}
            visualizations.append(VisualizationData(
                chart_type="bar",
                title="Memory Lifecycle Distribution",
                data=lifecycle_data,
                labels=list(lifecycle_data.keys()),
                description="Distribution of memories across lifecycle stages"
            ))
        
        return {
            "report_type": "executive_summary",
            "generated_at": datetime.now().isoformat(),
            "reporting_period": self._get_reporting_period(config),
            "kpis": kpis,
            "health_status": health_status,
            "insights": insights,
            "visualizations": [asdict(viz) for viz in visualizations] if config.include_visualizations else [],
            "optimization_opportunities": system_metrics.optimization_opportunities[:5],  # Top 5
            "next_actions": await self._generate_next_actions(system_metrics)
        }
    
    async def generate_detailed_analysis(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate detailed analysis report"""
        # Get memory list
        memory_ids = config.memory_filter or list(
            self.lifecycle_manager._memory_lifecycle_status.keys()
        )[:50]  # Limit to 50 for detailed analysis
        
        # Analyze each memory
        memory_analyses = []
        for memory_id in memory_ids:
            try:
                analysis = await self._analyze_individual_memory(memory_id)
                memory_analyses.append(analysis)
            except Exception as e:
                memory_analyses.append({
                    "memory_id": memory_id,
                    "error": str(e),
                    "analysis_failed": True
                })
        
        # System-wide patterns
        detected_patterns = await self.usage_analytics.analyze_usage_patterns()
        pattern_summary = {}
        for pattern in detected_patterns:
            pattern_type = pattern.pattern_type.value
            if pattern_type not in pattern_summary:
                pattern_summary[pattern_type] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "memory_ids": []
                }
            pattern_summary[pattern_type]["count"] += 1
            pattern_summary[pattern_type]["avg_confidence"] += pattern.confidence
            pattern_summary[pattern_type]["memory_ids"].extend(pattern.memory_ids)
        
        # Calculate averages
        for pattern_type in pattern_summary:
            count = pattern_summary[pattern_type]["count"]
            if count > 0:
                pattern_summary[pattern_type]["avg_confidence"] /= count
                pattern_summary[pattern_type]["avg_confidence"] = round(
                    pattern_summary[pattern_type]["avg_confidence"], 3
                )
                # Remove duplicates from memory_ids
                pattern_summary[pattern_type]["memory_ids"] = list(
                    set(pattern_summary[pattern_type]["memory_ids"])
                )
        
        # Memory clusters
        clusters = await self.usage_analytics.cluster_memories_by_usage(memory_ids)
        cluster_summary = []
        for cluster in clusters:
            cluster_summary.append({
                "cluster_id": cluster.cluster_id,
                "size": len(cluster.memory_ids),
                "primary_pattern": cluster.primary_pattern.value,
                "cohesion_score": round(cluster.cohesion_score, 3),
                "characteristics": cluster.usage_characteristics
            })
        
        # Performance trends
        performance_trends = await self._analyze_performance_trends(memory_ids)
        
        return {
            "report_type": "detailed_analysis",
            "generated_at": datetime.now().isoformat(),
            "reporting_period": self._get_reporting_period(config),
            "summary": {
                "total_memories_analyzed": len(memory_ids),
                "successful_analyses": len([a for a in memory_analyses if not a.get("analysis_failed")]),
                "patterns_detected": len(detected_patterns),
                "clusters_identified": len(clusters)
            },
            "memory_analyses": memory_analyses,
            "pattern_analysis": pattern_summary,
            "cluster_analysis": cluster_summary,
            "performance_trends": performance_trends,
            "insights": await self._generate_detailed_insights(memory_analyses, pattern_summary, clusters)
        }
    
    async def generate_health_assessment(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate comprehensive health assessment report"""
        memory_ids = config.memory_filter or list(
            self.lifecycle_manager._memory_lifecycle_status.keys()
        )
        
        # Health metrics for each memory
        health_assessments = []
        critical_memories = []
        healthy_memories = []
        
        for memory_id in memory_ids:
            try:
                score = await self.evolution_metrics.calculate_evolution_score(memory_id)
                
                health_assessment = {
                    "memory_id": memory_id,
                    "overall_score": round(score.overall_score, 3),
                    "health_rating": self.evolution_metrics._get_score_rating(score.overall_score),
                    "category_scores": {
                        category.value: round(score_val, 3)
                        for category, score_val in score.category_scores.items()
                    },
                    "health_flags": score.health_flags,
                    "trend": score.score_trend,
                    "confidence": round(score.confidence, 3)
                }
                
                health_assessments.append(health_assessment)
                
                if score.overall_score < 0.5:
                    critical_memories.append(memory_id)
                elif score.overall_score >= 0.7:
                    healthy_memories.append(memory_id)
                    
            except Exception as e:
                health_assessments.append({
                    "memory_id": memory_id,
                    "error": str(e),
                    "assessment_failed": True
                })
        
        # System health metrics
        system_health = {
            "overall_health_score": statistics.mean([
                a["overall_score"] for a in health_assessments 
                if not a.get("assessment_failed")
            ]) if health_assessments else 0.0,
            "healthy_percentage": len(healthy_memories) / len(memory_ids) * 100 if memory_ids else 0,
            "critical_percentage": len(critical_memories) / len(memory_ids) * 100 if memory_ids else 0,
            "total_health_flags": sum(len(a.get("health_flags", [])) for a in health_assessments)
        }
        
        # Health trends
        health_trends = {
            "improving": len([a for a in health_assessments if a.get("trend") == "improving"]),
            "stable": len([a for a in health_assessments if a.get("trend") == "stable"]),
            "declining": len([a for a in health_assessments if a.get("trend") == "declining"])
        }
        
        # Category analysis
        category_health = {}
        for category in MetricCategory:
            category_scores = [
                a["category_scores"].get(category.value, 0)
                for a in health_assessments
                if not a.get("assessment_failed") and category.value in a.get("category_scores", {})
            ]
            if category_scores:
                category_health[category.value] = {
                    "avg_score": round(statistics.mean(category_scores), 3),
                    "min_score": round(min(category_scores), 3),
                    "max_score": round(max(category_scores), 3),
                    "std_dev": round(statistics.stdev(category_scores), 3) if len(category_scores) > 1 else 0.0
                }
        
        # Anomaly detection
        anomalies = []
        for memory_id in memory_ids[:20]:  # Check top 20 for anomalies
            try:
                memory_anomalies = await self.usage_analytics.detect_usage_anomalies(memory_id)
                if memory_anomalies:
                    anomalies.append({
                        "memory_id": memory_id,
                        "anomalies": memory_anomalies
                    })
            except Exception:
                continue
        
        return {
            "report_type": "health_assessment",
            "generated_at": datetime.now().isoformat(),
            "reporting_period": self._get_reporting_period(config),
            "system_health": system_health,
            "health_trends": health_trends,
            "category_health": category_health,
            "critical_memories": critical_memories[:10],  # Top 10 critical
            "healthy_memories": healthy_memories[:10],   # Top 10 healthy
            "memory_health_details": health_assessments,
            "anomalies_detected": anomalies,
            "health_recommendations": await self._generate_health_recommendations(
                system_health, critical_memories, category_health
            )
        }
    
    async def generate_optimization_guide(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate optimization guide with actionable recommendations"""
        system_metrics = await self.evolution_metrics.calculate_system_metrics()
        
        # Identify optimization opportunities
        optimization_opportunities = []
        
        # Performance optimizations
        if system_metrics.efficiency_metrics.get("avg_response_efficiency", 0) < 0.7:
            optimization_opportunities.append({
                "category": "performance",
                "priority": "high",
                "title": "Improve Response Efficiency",
                "description": "System response efficiency is below optimal levels",
                "impact": "Faster memory access and better user experience",
                "effort": "medium",
                "actions": [
                    "Implement better caching strategies",
                    "Optimize memory indexing",
                    "Review query optimization"
                ],
                "metrics_to_watch": ["response_efficiency", "cache_efficiency"]
            })
        
        # Usage pattern optimizations
        detected_patterns = await self.usage_analytics.analyze_usage_patterns()
        burst_patterns = [p for p in detected_patterns if p.pattern_type == UsagePatternType.BURST]
        
        if len(burst_patterns) > 3:
            optimization_opportunities.append({
                "category": "usage_patterns",
                "priority": "medium",
                "title": "Optimize Burst Usage Patterns",
                "description": f"Detected {len(burst_patterns)} burst usage patterns",
                "impact": "Better resource utilization and smoother performance",
                "effort": "low",
                "actions": [
                    "Implement burst detection and caching",
                    "Pre-load frequently accessed memories",
                    "Optimize for peak usage periods"
                ],
                "affected_memories": [m for pattern in burst_patterns for m in pattern.memory_ids]
            })
        
        # Lifecycle optimizations
        lifecycle_distribution = system_metrics.lifecycle_distribution
        dormant_count = lifecycle_distribution.get(LifecycleStage.DORMANT, 0)
        
        if dormant_count > system_metrics.total_memories * 0.2:  # More than 20% dormant
            optimization_opportunities.append({
                "category": "lifecycle",
                "priority": "medium",
                "title": "Address Dormant Memories",
                "description": f"{dormant_count} memories are dormant and may need attention",
                "impact": "Better resource utilization and system cleanliness",
                "effort": "low",
                "actions": [
                    "Review dormant memories for archival",
                    "Implement reactivation strategies",
                    "Optimize retention policies"
                ],
                "affected_count": dormant_count
            })
        
        # Quality optimizations
        memory_ids = list(self.lifecycle_manager._memory_lifecycle_status.keys())[:50]
        low_quality_memories = []
        
        for memory_id in memory_ids:
            try:
                score = await self.evolution_metrics.calculate_evolution_score(memory_id)
                if score.overall_score < 0.5:
                    low_quality_memories.append(memory_id)
            except Exception:
                continue
        
        if len(low_quality_memories) > len(memory_ids) * 0.15:  # More than 15% low quality
            optimization_opportunities.append({
                "category": "quality",
                "priority": "high",
                "title": "Improve Low-Quality Memories",
                "description": f"{len(low_quality_memories)} memories have quality scores below 0.5",
                "impact": "Better system reliability and user satisfaction",
                "effort": "high",
                "actions": [
                    "Review and improve memory content",
                    "Enhance memory tagging and metadata",
                    "Implement quality gates for new memories"
                ],
                "affected_memories": low_quality_memories[:10]  # Top 10
            })
        
        # Collaboration optimizations
        avg_collaboration = system_metrics.efficiency_metrics.get("avg_collaboration_score", 0)
        if avg_collaboration < 0.4:
            optimization_opportunities.append({
                "category": "collaboration",
                "priority": "low",
                "title": "Enhance Collaboration Features",
                "description": "Low cross-context usage detected across memories",
                "impact": "Better knowledge sharing and memory utilization",
                "effort": "medium",
                "actions": [
                    "Encourage cross-context memory usage",
                    "Implement memory sharing features",
                    "Create collaboration incentives"
                ],
                "current_score": round(avg_collaboration, 3)
            })
        
        # Priority ranking
        optimization_opportunities.sort(key=lambda x: {
            "high": 3, "medium": 2, "low": 1
        }.get(x["priority"], 0), reverse=True)
        
        # Implementation roadmap
        roadmap = {
            "immediate_actions": [
                op for op in optimization_opportunities 
                if op["priority"] == "high" and op["effort"] in ["low", "medium"]
            ][:3],
            "short_term_goals": [
                op for op in optimization_opportunities 
                if op["priority"] in ["high", "medium"]
            ][:5],
            "long_term_initiatives": optimization_opportunities
        }
        
        return {
            "report_type": "optimization_guide",
            "generated_at": datetime.now().isoformat(),
            "system_overview": {
                "total_memories": system_metrics.total_memories,
                "avg_score": round(system_metrics.avg_evolution_score, 3),
                "optimization_potential": len(optimization_opportunities)
            },
            "optimization_opportunities": optimization_opportunities,
            "implementation_roadmap": roadmap,
            "success_metrics": {
                "target_avg_score": 0.8,
                "target_healthy_percentage": 80,
                "target_response_efficiency": 0.9,
                "review_interval_days": 30
            },
            "recommendations": await self._generate_optimization_recommendations(optimization_opportunities)
        }
    
    async def generate_trend_analysis(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate trend analysis report"""
        days_back = 30 if not config.date_range else (
            datetime.now() - config.date_range[0]
        ).days
        
        # Get historical data
        system_report = await self.evolution_tracker.get_system_evolution_report(days_back)
        
        # Analyze trends in key metrics
        trends = {
            "activity_trends": {
                "total_events": system_report["total_events"],
                "daily_average": system_report["total_events"] / days_back,
                "active_memories": system_report["active_memories"]
            },
            "pattern_trends": system_report["detected_patterns"],
            "lifecycle_trends": await self._analyze_lifecycle_trends(days_back),
            "performance_trends": await self._analyze_performance_trends_over_time(days_back)
        }
        
        # Predictive insights
        predictions = {}
        memory_ids = list(self.lifecycle_manager._memory_lifecycle_status.keys())[:20]
        
        for memory_id in memory_ids:
            try:
                prediction = await self.evolution_metrics.predict_evolution_score(memory_id)
                predictions[memory_id] = {
                    "current_score": prediction["current_score"],
                    "predicted_score": prediction["predicted_score"],
                    "trend": prediction["trend"],
                    "confidence": prediction["confidence"]
                }
            except Exception:
                continue
        
        # Trend visualizations
        visualizations = []
        if config.include_visualizations:
            # Activity trend line chart
            activity_data = {
                "dates": [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back, 0, -1)],
                "events": [system_report["total_events"] // days_back] * days_back  # Simplified
            }
            visualizations.append(VisualizationData(
                chart_type="line",
                title="System Activity Trend",
                data=activity_data,
                labels=activity_data["dates"],
                description="Daily activity levels over time"
            ))
        
        return {
            "report_type": "trend_analysis",
            "generated_at": datetime.now().isoformat(),
            "analysis_period_days": days_back,
            "trends": trends,
            "predictions": predictions,
            "visualizations": [asdict(viz) for viz in visualizations] if config.include_visualizations else [],
            "key_insights": await self._generate_trend_insights(trends, predictions)
        }
    
    async def _analyze_individual_memory(self, memory_id: str) -> Dict[str, Any]:
        """Analyze an individual memory in detail"""
        evolution_score = await self.evolution_metrics.calculate_evolution_score(memory_id)
        usage_metrics = await self.usage_analytics.calculate_usage_metrics(memory_id)
        lifecycle_status = self.lifecycle_manager.get_memory_lifecycle_status(memory_id)
        evolution_summary = await self.evolution_tracker.get_memory_evolution_summary(memory_id)
        
        return {
            "memory_id": memory_id,
            "evolution_score": asdict(evolution_score),
            "usage_metrics": {
                "access_frequency": round(usage_metrics.access_frequency, 3),
                "usage_efficiency": round(usage_metrics.usage_efficiency, 3),
                "context_diversity": usage_metrics.context_diversity,
                "collaboration_score": round(usage_metrics.collaboration_score, 3),
                "evolution_velocity": round(usage_metrics.evolution_velocity, 3),
                "peak_usage_periods": len(usage_metrics.peak_usage_periods),
                "temporal_distribution": usage_metrics.temporal_distribution
            },
            "lifecycle_status": {
                "current_stage": lifecycle_status.current_stage.value if lifecycle_status else "unknown",
                "stage_duration_days": lifecycle_status.stage_duration.days if lifecycle_status else 0,
                "lifecycle_score": round(lifecycle_status.lifecycle_score, 3) if lifecycle_status else 0.0
            } if lifecycle_status else None,
            "activity_summary": {
                "total_events": evolution_summary.get("total_events", 0),
                "lifecycle_stage": evolution_summary.get("lifecycle_stage", "unknown"),
                "health_indicators": evolution_summary.get("health_indicators", {})
            }
        }
    
    async def _analyze_performance_trends(self, memory_ids: List[str]) -> Dict[str, Any]:
        """Analyze performance trends across memories"""
        performance_data = []
        
        for memory_id in memory_ids:
            try:
                score = await self.evolution_metrics.calculate_evolution_score(memory_id)
                performance_data.append({
                    "memory_id": memory_id,
                    "response_efficiency": score.component_scores.get("response_efficiency", 0),
                    "cache_efficiency": score.component_scores.get("cache_efficiency", 0),
                    "utilization_efficiency": score.component_scores.get("utilization_efficiency", 0)
                })
            except Exception:
                continue
        
        if not performance_data:
            return {"error": "No performance data available"}
        
        # Calculate averages and trends
        avg_response = statistics.mean(p["response_efficiency"] for p in performance_data)
        avg_cache = statistics.mean(p["cache_efficiency"] for p in performance_data)
        avg_utilization = statistics.mean(p["utilization_efficiency"] for p in performance_data)
        
        return {
            "avg_response_efficiency": round(avg_response, 3),
            "avg_cache_efficiency": round(avg_cache, 3),
            "avg_utilization_efficiency": round(avg_utilization, 3),
            "performance_distribution": {
                "high_performance": len([p for p in performance_data if p["response_efficiency"] > 0.8]),
                "medium_performance": len([p for p in performance_data if 0.5 <= p["response_efficiency"] <= 0.8]),
                "low_performance": len([p for p in performance_data if p["response_efficiency"] < 0.5])
            }
        }
    
    async def _analyze_lifecycle_trends(self, days_back: int) -> Dict[str, Any]:
        """Analyze lifecycle progression trends"""
        # Get lifecycle analytics
        lifecycle_analytics = await self.lifecycle_manager.get_lifecycle_analytics(days_back)
        
        return {
            "transition_patterns": lifecycle_analytics.get("transition_patterns", {}),
            "average_stage_durations": lifecycle_analytics.get("average_stage_durations", {}),
            "health_percentage": lifecycle_analytics.get("health_metrics", {}).get("health_percentage", 0)
        }
    
    async def _analyze_performance_trends_over_time(self, days_back: int) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        # This is a simplified implementation
        # In a real system, you'd track historical performance data
        
        return {
            "trend_direction": "stable",  # Would be calculated from historical data
            "performance_variance": 0.1,  # Would be calculated from historical data
            "improvement_rate": 0.02      # Would be calculated from historical data
        }
    
    async def _generate_next_actions(self, system_metrics: Any) -> List[str]:
        """Generate next action recommendations"""
        actions = []
        
        if system_metrics.avg_evolution_score < 0.6:
            actions.append("Focus on improving overall system health")
        
        if system_metrics.healthy_memories / system_metrics.total_memories < 0.7:
            actions.append("Address low-scoring memories")
        
        if system_metrics.evolution_velocity < 0.1:
            actions.append("Encourage more memory evolution activity")
        
        if len(system_metrics.optimization_opportunities) > 3:
            actions.append("Implement top optimization recommendations")
        
        return actions[:5]  # Top 5 actions
    
    async def _generate_detailed_insights(self, 
                                        memory_analyses: List[Dict],
                                        pattern_summary: Dict,
                                        clusters: List) -> List[str]:
        """Generate detailed insights from analysis"""
        insights = []
        
        # Memory analysis insights
        successful_analyses = [a for a in memory_analyses if not a.get("analysis_failed")]
        if len(successful_analyses) > 0:
            avg_score = statistics.mean(
                a["evolution_score"]["overall_score"] for a in successful_analyses
            )
            insights.append(f"Average evolution score across analyzed memories: {avg_score:.3f}")
        
        # Pattern insights
        if pattern_summary:
            dominant_pattern = max(pattern_summary.items(), key=lambda x: x[1]["count"])
            insights.append(
                f"Most common usage pattern: {dominant_pattern[0]} "
                f"({dominant_pattern[1]['count']} instances)"
            )
        
        # Cluster insights
        if clusters:
            largest_cluster = max(clusters, key=lambda c: c["size"])
            insights.append(
                f"Largest memory cluster has {largest_cluster['size']} memories "
                f"with {largest_cluster['primary_pattern']} pattern"
            )
        
        return insights
    
    async def _generate_health_recommendations(self, 
                                             system_health: Dict,
                                             critical_memories: List[str],
                                             category_health: Dict) -> List[str]:
        """Generate health-based recommendations"""
        recommendations = []
        
        if system_health["critical_percentage"] > 20:
            recommendations.append(
                f"Priority: Address {len(critical_memories)} critical memories immediately"
            )
        
        if system_health["overall_health_score"] < 0.6:
            recommendations.append("System health is below acceptable levels - comprehensive review needed")
        
        # Category-specific recommendations
        for category, health in category_health.items():
            if health["avg_score"] < 0.5:
                recommendations.append(f"Focus on improving {category.replace('_', ' ')} metrics")
        
        return recommendations
    
    async def _generate_optimization_recommendations(self, opportunities: List[Dict]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        high_priority = [op for op in opportunities if op["priority"] == "high"]
        if high_priority:
            recommendations.append(f"Address {len(high_priority)} high-priority optimization opportunities")
        
        performance_ops = [op for op in opportunities if op["category"] == "performance"]
        if performance_ops:
            recommendations.append("Focus on performance optimizations for immediate impact")
        
        return recommendations
    
    async def _generate_trend_insights(self, trends: Dict, predictions: Dict) -> List[str]:
        """Generate insights from trend analysis"""
        insights = []
        
        # Activity insights
        daily_avg = trends["activity_trends"]["daily_average"]
        if daily_avg > 50:
            insights.append("High system activity indicates healthy memory usage")
        elif daily_avg < 10:
            insights.append("Low system activity may indicate underutilization")
        
        # Prediction insights
        if predictions:
            improving_count = len([p for p in predictions.values() if p["trend"] == "improving"])
            declining_count = len([p for p in predictions.values() if p["trend"] == "declining"])
            
            if improving_count > declining_count:
                insights.append("Most memories show improving trends")
            elif declining_count > improving_count:
                insights.append("Several memories show declining trends - attention needed")
        
        return insights
    
    def _get_reporting_period(self, config: ReportConfiguration) -> str:
        """Get reporting period description"""
        if config.date_range:
            start, end = config.date_range
            return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        else:
            return f"Last 30 days (as of {datetime.now().strftime('%Y-%m-%d')})"
    
    async def export_report(self, report_data: Dict[str, Any], config: ReportConfiguration) -> str:
        """Export report to specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.report_type.value}_{timestamp}"
        
        if config.format == ReportFormat.JSON:
            filepath = self.output_directory / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        
        elif config.format == ReportFormat.MARKDOWN:
            filepath = self.output_directory / f"{filename}.md"
            markdown_content = self._convert_to_markdown(report_data)
            with open(filepath, 'w') as f:
                f.write(markdown_content)
        
        elif config.format == ReportFormat.HTML:
            filepath = self.output_directory / f"{filename}.html"
            html_content = self._convert_to_html(report_data)
            with open(filepath, 'w') as f:
                f.write(html_content)
        
        else:  # TEXT
            filepath = self.output_directory / f"{filename}.txt"
            text_content = self._convert_to_text(report_data)
            with open(filepath, 'w') as f:
                f.write(text_content)
        
        return str(filepath)
    
    def _convert_to_markdown(self, report_data: Dict[str, Any]) -> str:
        """Convert report data to Markdown format"""
        md_content = []
        
        # Title
        report_type = report_data.get("report_type", "Memory Evolution Report")
        md_content.append(f"# {report_type.replace('_', ' ').title()}")
        md_content.append(f"Generated: {report_data.get('generated_at', datetime.now().isoformat())}")
        md_content.append("")
        
        # Key sections
        if "kpis" in report_data:
            md_content.append("## Key Performance Indicators")
            for key, value in report_data["kpis"].items():
                md_content.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            md_content.append("")
        
        if "insights" in report_data:
            md_content.append("## Key Insights")
            for insight in report_data["insights"]:
                if isinstance(insight, dict):
                    md_content.append(f"- **{insight.get('type', 'info').upper()}**: {insight.get('message', '')}")
                else:
                    md_content.append(f"- {insight}")
            md_content.append("")
        
        if "optimization_opportunities" in report_data:
            md_content.append("## Optimization Opportunities")
            for i, opp in enumerate(report_data["optimization_opportunities"][:5], 1):
                if isinstance(opp, dict):
                    md_content.append(f"{i}. **{opp.get('title', '')}** ({opp.get('priority', 'medium')} priority)")
                    md_content.append(f"   - {opp.get('description', '')}")
                else:
                    md_content.append(f"{i}. {opp}")
            md_content.append("")
        
        return "\n".join(md_content)
    
    def _convert_to_html(self, report_data: Dict[str, Any]) -> str:
        """Convert report data to HTML format"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data.get('report_type', 'Memory Evolution Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .kpi {{ background: #f5f5f5; padding: 10px; margin: 5px 0; }}
                .insight {{ padding: 8px; margin: 5px 0; }}
                .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
                .info {{ background: #d1ecf1; border-left: 4px solid #17a2b8; }}
            </style>
        </head>
        <body>
            <h1>{report_data.get('report_type', 'Memory Evolution Report').replace('_', ' ').title()}</h1>
            <p>Generated: {report_data.get('generated_at', datetime.now().isoformat())}</p>
        """
        
        if "kpis" in report_data:
            html_content += "<h2>Key Performance Indicators</h2>"
            for key, value in report_data["kpis"].items():
                html_content += f'<div class="kpi"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        
        if "insights" in report_data:
            html_content += "<h2>Key Insights</h2>"
            for insight in report_data["insights"]:
                if isinstance(insight, dict):
                    css_class = insight.get('type', 'info')
                    html_content += f'<div class="insight {css_class}"><strong>{insight.get("type", "info").upper()}:</strong> {insight.get("message", "")}</div>'
                else:
                    html_content += f'<div class="insight info">{insight}</div>'
        
        html_content += "</body></html>"
        return html_content
    
    def _convert_to_text(self, report_data: Dict[str, Any]) -> str:
        """Convert report data to plain text format"""
        text_lines = []
        
        # Title
        title = report_data.get("report_type", "Memory Evolution Report").replace('_', ' ').title()
        text_lines.append(title)
        text_lines.append("=" * len(title))
        text_lines.append(f"Generated: {report_data.get('generated_at', datetime.now().isoformat())}")
        text_lines.append("")
        
        # KPIs
        if "kpis" in report_data:
            text_lines.append("KEY PERFORMANCE INDICATORS")
            text_lines.append("-" * 30)
            for key, value in report_data["kpis"].items():
                text_lines.append(f"{key.replace('_', ' ').title()}: {value}")
            text_lines.append("")
        
        # Insights
        if "insights" in report_data:
            text_lines.append("KEY INSIGHTS")
            text_lines.append("-" * 12)
            for insight in report_data["insights"]:
                if isinstance(insight, dict):
                    text_lines.append(f"[{insight.get('type', 'info').upper()}] {insight.get('message', '')}")
                else:
                    text_lines.append(f"• {insight}")
            text_lines.append("")
        
        return "\n".join(text_lines)
    
    async def generate_report(self, config: ReportConfiguration) -> Dict[str, Any]:
        """Generate report based on configuration"""
        if config.report_type == ReportType.EXECUTIVE_SUMMARY:
            return await self.generate_executive_summary(config)
        elif config.report_type == ReportType.DETAILED_ANALYSIS:
            return await self.generate_detailed_analysis(config)
        elif config.report_type == ReportType.HEALTH_ASSESSMENT:
            return await self.generate_health_assessment(config)
        elif config.report_type == ReportType.OPTIMIZATION_GUIDE:
            return await self.generate_optimization_guide(config)
        elif config.report_type == ReportType.TREND_ANALYSIS:
            return await self.generate_trend_analysis(config)
        else:
            raise MemMimicError(f"Unsupported report type: {config.report_type}")
    
    async def schedule_regular_reports(self, report_configs: List[ReportConfiguration]) -> Dict[str, Any]:
        """Schedule regular report generation"""
        # This would integrate with a job scheduler in a real system
        scheduled_reports = []
        
        for config in report_configs:
            report_data = await self.generate_report(config)
            filepath = await self.export_report(report_data, config)
            
            scheduled_reports.append({
                "report_type": config.report_type.value,
                "filepath": filepath,
                "generated_at": datetime.now().isoformat()
            })
        
        return {
            "scheduled_reports": scheduled_reports,
            "next_run": (datetime.now() + timedelta(days=1)).isoformat(),
            "status": "completed"
        }