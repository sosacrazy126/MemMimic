"""
Memory Evolution Metrics and Scoring System

Comprehensive metrics and scoring algorithms for evaluating memory evolution quality,
health, and optimization opportunities. Provides multi-dimensional scoring and analysis.
"""

import asyncio
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .memory_evolution_tracker import MemoryEvolutionTracker, MemoryEventType
from .memory_lifecycle_manager import MemoryLifecycleManager, LifecycleStage
from .memory_usage_analytics import MemoryUsageAnalytics, UsagePatternType
from ..errors.exceptions import MemMimicError


class MetricCategory(Enum):
    """Categories of evolution metrics"""
    USAGE_QUALITY = "usage_quality"
    EVOLUTION_HEALTH = "evolution_health"
    PERFORMANCE = "performance"
    LIFECYCLE_FITNESS = "lifecycle_fitness"
    PATTERN_STABILITY = "pattern_stability"
    COLLABORATION_VALUE = "collaboration_value"
    PREDICTIVE_ACCURACY = "predictive_accuracy"


@dataclass
class EvolutionScore:
    """Comprehensive evolution score for a memory"""
    memory_id: str
    overall_score: float  # 0.0 to 1.0
    category_scores: Dict[MetricCategory, float]
    component_scores: Dict[str, float]
    confidence: float
    calculated_at: datetime
    score_trend: Optional[str] = None  # 'improving', 'stable', 'declining'
    recommendations: List[str] = field(default_factory=list)
    health_flags: List[str] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """System-wide evolution metrics"""
    total_memories: int
    avg_evolution_score: float
    score_distribution: Dict[str, int]  # Score ranges
    healthy_memories: int
    evolution_velocity: float  # System-wide change rate
    pattern_diversity: int
    lifecycle_distribution: Dict[LifecycleStage, int]
    efficiency_metrics: Dict[str, float]
    optimization_opportunities: List[str]


@dataclass
class EvolutionBenchmark:
    """Benchmarks for evolution quality assessment"""
    excellent_threshold: float = 0.9
    good_threshold: float = 0.7
    fair_threshold: float = 0.5
    poor_threshold: float = 0.3
    
    usage_frequency_target: float = 0.5  # Accesses per day
    efficiency_target: float = 0.8
    lifecycle_progression_target: float = 0.7
    pattern_stability_target: float = 0.6


class MemoryEvolutionMetrics:
    """
    Advanced metrics and scoring system for memory evolution analysis.
    
    Core capabilities:
    - Multi-dimensional scoring: Evaluates memories across multiple quality dimensions
    - Health assessment: Provides comprehensive health indicators
    - Trend analysis: Tracks score evolution over time
    - Benchmarking: Compares against quality standards
    - Optimization identification: Identifies improvement opportunities
    - System analytics: Provides system-wide evolution insights
    - Predictive scoring: Forecasts future evolution quality
    """
    
    def __init__(self, 
                 evolution_tracker: MemoryEvolutionTracker,
                 lifecycle_manager: MemoryLifecycleManager,
                 usage_analytics: MemoryUsageAnalytics,
                 benchmarks: Optional[EvolutionBenchmark] = None):
        self.evolution_tracker = evolution_tracker
        self.lifecycle_manager = lifecycle_manager
        self.usage_analytics = usage_analytics
        self.benchmarks = benchmarks or EvolutionBenchmark()
        self._score_cache: Dict[str, EvolutionScore] = {}
        self._score_history: Dict[str, List[EvolutionScore]] = {}
    
    async def calculate_evolution_score(self, memory_id: str) -> EvolutionScore:
        """Calculate comprehensive evolution score for a memory"""
        # Get base data
        evolution_summary = await self.evolution_tracker.get_memory_evolution_summary(memory_id)
        usage_metrics = await self.usage_analytics.calculate_usage_metrics(memory_id)
        lifecycle_status = self.lifecycle_manager.get_memory_lifecycle_status(memory_id)
        
        # Calculate component scores
        component_scores = {}
        category_scores = {}
        
        # 1. Usage Quality Metrics
        usage_quality_components = await self._calculate_usage_quality_metrics(
            memory_id, evolution_summary, usage_metrics
        )
        component_scores.update(usage_quality_components)
        category_scores[MetricCategory.USAGE_QUALITY] = self._aggregate_component_scores(
            usage_quality_components
        )
        
        # 2. Evolution Health Metrics
        evolution_health_components = await self._calculate_evolution_health_metrics(
            memory_id, evolution_summary, usage_metrics
        )
        component_scores.update(evolution_health_components)
        category_scores[MetricCategory.EVOLUTION_HEALTH] = self._aggregate_component_scores(
            evolution_health_components
        )
        
        # 3. Performance Metrics
        performance_components = await self._calculate_performance_metrics(
            memory_id, evolution_summary, usage_metrics
        )
        component_scores.update(performance_components)
        category_scores[MetricCategory.PERFORMANCE] = self._aggregate_component_scores(
            performance_components
        )
        
        # 4. Lifecycle Fitness Metrics
        lifecycle_fitness_components = await self._calculate_lifecycle_fitness_metrics(
            memory_id, lifecycle_status, usage_metrics
        )
        component_scores.update(lifecycle_fitness_components)
        category_scores[MetricCategory.LIFECYCLE_FITNESS] = self._aggregate_component_scores(
            lifecycle_fitness_components
        )
        
        # 5. Pattern Stability Metrics
        pattern_stability_components = await self._calculate_pattern_stability_metrics(
            memory_id, usage_metrics
        )
        component_scores.update(pattern_stability_components)
        category_scores[MetricCategory.PATTERN_STABILITY] = self._aggregate_component_scores(
            pattern_stability_components
        )
        
        # 6. Collaboration Value Metrics
        collaboration_components = await self._calculate_collaboration_value_metrics(
            memory_id, usage_metrics
        )
        component_scores.update(collaboration_components)
        category_scores[MetricCategory.COLLABORATION_VALUE] = self._aggregate_component_scores(
            collaboration_components
        )
        
        # Calculate overall score (weighted average)
        weights = {
            MetricCategory.USAGE_QUALITY: 0.25,
            MetricCategory.EVOLUTION_HEALTH: 0.20,
            MetricCategory.PERFORMANCE: 0.15,
            MetricCategory.LIFECYCLE_FITNESS: 0.15,
            MetricCategory.PATTERN_STABILITY: 0.15,
            MetricCategory.COLLABORATION_VALUE: 0.10
        }
        
        overall_score = sum(
            category_scores.get(category, 0.0) * weight
            for category, weight in weights.items()
        )
        
        # Calculate confidence based on data availability
        confidence = self._calculate_score_confidence(evolution_summary, usage_metrics)
        
        # Generate recommendations and health flags
        recommendations = await self._generate_recommendations(
            memory_id, category_scores, component_scores
        )
        health_flags = self._identify_health_flags(category_scores, component_scores)
        
        # Determine score trend
        score_trend = self._calculate_score_trend(memory_id, overall_score)
        
        evolution_score = EvolutionScore(
            memory_id=memory_id,
            overall_score=overall_score,
            category_scores=category_scores,
            component_scores=component_scores,
            confidence=confidence,
            calculated_at=datetime.now(),
            score_trend=score_trend,
            recommendations=recommendations,
            health_flags=health_flags
        )
        
        # Cache and store history
        self._score_cache[memory_id] = evolution_score
        if memory_id not in self._score_history:
            self._score_history[memory_id] = []
        self._score_history[memory_id].append(evolution_score)
        
        # Keep only last 30 scores for history
        self._score_history[memory_id] = self._score_history[memory_id][-30:]
        
        return evolution_score
    
    async def _calculate_usage_quality_metrics(self, 
                                             memory_id: str,
                                             evolution_summary: Dict[str, Any],
                                             usage_metrics: Any) -> Dict[str, float]:
        """Calculate usage quality component scores"""
        scores = {}
        
        # Access frequency score
        target_frequency = self.benchmarks.usage_frequency_target
        actual_frequency = usage_metrics.access_frequency
        scores['access_frequency'] = min(1.0, actual_frequency / target_frequency)
        
        # Usage efficiency score
        target_efficiency = self.benchmarks.efficiency_target
        actual_efficiency = usage_metrics.usage_efficiency
        scores['usage_efficiency'] = actual_efficiency / target_efficiency if target_efficiency > 0 else actual_efficiency
        
        # Context diversity score (normalized)
        context_diversity = usage_metrics.context_diversity
        max_expected_contexts = 10  # Reasonable maximum
        scores['context_diversity'] = min(1.0, context_diversity / max_expected_contexts)
        
        # Recall success rate score
        recall_success_rate = evolution_summary.get('usage_stats', {}).get('recall_success_rate', 1.0)
        scores['recall_success'] = recall_success_rate
        
        # Consistency score (inverse of variance)
        if usage_metrics.access_variance > 0 and usage_metrics.access_frequency > 0:
            cv = math.sqrt(usage_metrics.access_variance) / usage_metrics.access_frequency
            scores['access_consistency'] = max(0.0, 1.0 - min(1.0, cv / 2.0))  # Normalize CV
        else:
            scores['access_consistency'] = 1.0 if usage_metrics.access_frequency > 0 else 0.5
        
        return scores
    
    async def _calculate_evolution_health_metrics(self, 
                                                memory_id: str,
                                                evolution_summary: Dict[str, Any],
                                                usage_metrics: Any) -> Dict[str, float]:
        """Calculate evolution health component scores"""
        scores = {}
        
        # Evolution activity score
        evolution_velocity = usage_metrics.evolution_velocity
        target_evolution = 0.1  # Changes per day
        scores['evolution_activity'] = min(1.0, evolution_velocity / target_evolution)
        
        # Health indicators score
        health_indicators = evolution_summary.get('health_indicators', {})
        usage_health = health_indicators.get('usage_health', 'good')
        evolution_health = health_indicators.get('evolution_health', 'good')
        
        health_score_map = {'good': 1.0, 'fair': 0.6, 'poor': 0.2}
        scores['usage_health'] = health_score_map.get(usage_health, 0.5)
        scores['evolution_health'] = health_score_map.get(evolution_health, 0.5)
        
        # Data completeness score
        total_events = evolution_summary.get('total_events', 0)
        if total_events >= 20:
            scores['data_completeness'] = 1.0
        elif total_events >= 10:
            scores['data_completeness'] = 0.8
        elif total_events >= 5:
            scores['data_completeness'] = 0.6
        else:
            scores['data_completeness'] = max(0.2, total_events / 5.0)
        
        # Age vs. activity balance
        evolution_age_days = (datetime.now() - datetime.fromisoformat(evolution_summary['events'][0]['timestamp'])).days if evolution_summary.get('events') else 0
        if evolution_age_days > 0:
            activity_per_day = total_events / evolution_age_days
            target_activity = 1.0  # 1 event per day
            scores['age_activity_balance'] = min(1.0, activity_per_day / target_activity)
        else:
            scores['age_activity_balance'] = 0.5
        
        return scores
    
    async def _calculate_performance_metrics(self, 
                                           memory_id: str,
                                           evolution_summary: Dict[str, Any],
                                           usage_metrics: Any) -> Dict[str, float]:
        """Calculate performance component scores"""
        scores = {}
        
        # Response efficiency (inverse of access time - simulated)
        # In a real system, this would measure actual response times
        total_accesses = evolution_summary.get('usage_stats', {}).get('total_accesses', 0)
        if total_accesses > 0:
            # Simulate response efficiency based on usage patterns
            efficiency_factor = min(1.0, usage_metrics.usage_efficiency + 0.2)
            scores['response_efficiency'] = efficiency_factor
        else:
            scores['response_efficiency'] = 0.5
        
        # Cache hit rate (simulated based on access patterns)
        if usage_metrics.access_frequency > 0.5:  # Frequently accessed
            scores['cache_efficiency'] = 0.9
        elif usage_metrics.access_frequency > 0.1:
            scores['cache_efficiency'] = 0.7
        else:
            scores['cache_efficiency'] = 0.4
        
        # Memory utilization efficiency
        peak_periods = len(usage_metrics.peak_usage_periods)
        total_periods = max(1, (datetime.now() - datetime.now().replace(day=1)).days // 7)  # Weeks this month
        utilization_ratio = min(1.0, peak_periods / total_periods)
        scores['utilization_efficiency'] = utilization_ratio
        
        # Evolution overhead (inverse of excessive changes)
        if usage_metrics.evolution_velocity > 0:
            if usage_metrics.evolution_velocity > 1.0:  # Too many changes
                scores['evolution_overhead'] = max(0.2, 1.0 - (usage_metrics.evolution_velocity - 1.0) / 2.0)
            else:
                scores['evolution_overhead'] = 1.0
        else:
            scores['evolution_overhead'] = 0.8  # Some evolution is expected
        
        return scores
    
    async def _calculate_lifecycle_fitness_metrics(self, 
                                                 memory_id: str,
                                                 lifecycle_status: Any,
                                                 usage_metrics: Any) -> Dict[str, float]:
        """Calculate lifecycle fitness component scores"""
        scores = {}
        
        if not lifecycle_status:
            # Default scores if no lifecycle data
            return {
                'stage_appropriateness': 0.5,
                'progression_health': 0.5,
                'retention_value': 0.5
            }
        
        # Stage appropriateness score
        current_stage = lifecycle_status.current_stage
        access_frequency = usage_metrics.access_frequency
        
        # Define ideal frequency ranges for each stage
        stage_frequency_ranges = {
            LifecycleStage.CREATED: (0.0, 0.1),
            LifecycleStage.WARMING: (0.1, 0.3),
            LifecycleStage.ACTIVE: (0.3, 1.0),
            LifecycleStage.PEAK: (1.0, float('inf')),
            LifecycleStage.MATURE: (0.2, 0.8),
            LifecycleStage.DECLINING: (0.01, 0.2),
            LifecycleStage.DORMANT: (0.0, 0.01),
            LifecycleStage.ARCHIVED: (0.0, 0.0)
        }
        
        min_freq, max_freq = stage_frequency_ranges.get(current_stage, (0.0, 1.0))
        if min_freq <= access_frequency <= max_freq:
            scores['stage_appropriateness'] = 1.0
        else:
            # Calculate distance from appropriate range
            if access_frequency < min_freq:
                distance = min_freq - access_frequency
            else:
                distance = access_frequency - max_freq
            scores['stage_appropriateness'] = max(0.0, 1.0 - distance / 2.0)
        
        # Progression health score
        lifecycle_score = lifecycle_status.lifecycle_score
        scores['progression_health'] = lifecycle_score
        
        # Retention value score (based on stage and activity)
        stage_value_map = {
            LifecycleStage.CREATED: 0.7,
            LifecycleStage.WARMING: 0.8,
            LifecycleStage.ACTIVE: 1.0,
            LifecycleStage.PEAK: 1.0,
            LifecycleStage.MATURE: 0.9,
            LifecycleStage.DECLINING: 0.4,
            LifecycleStage.DORMANT: 0.2,
            LifecycleStage.ARCHIVED: 0.1
        }
        
        base_value = stage_value_map.get(current_stage, 0.5)
        activity_bonus = min(0.3, usage_metrics.access_frequency / 2.0)
        scores['retention_value'] = min(1.0, base_value + activity_bonus)
        
        return scores
    
    async def _calculate_pattern_stability_metrics(self, 
                                                 memory_id: str,
                                                 usage_metrics: Any) -> Dict[str, float]:
        """Calculate pattern stability component scores"""
        scores = {}
        
        # Access pattern stability
        if usage_metrics.access_variance > 0 and usage_metrics.access_frequency > 0:
            cv = math.sqrt(usage_metrics.access_variance) / usage_metrics.access_frequency
            stability = max(0.0, 1.0 - min(1.0, cv / 1.5))  # Lower CV = higher stability
            scores['access_stability'] = stability
        else:
            scores['access_stability'] = 0.8 if usage_metrics.access_frequency > 0 else 0.3
        
        # Temporal pattern consistency
        temporal_dist = usage_metrics.temporal_distribution
        if temporal_dist:
            # Calculate entropy of temporal distribution
            total_events = sum(temporal_dist.values())
            if total_events > 0:
                probabilities = [count / total_events for count in temporal_dist.values()]
                entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
                max_entropy = math.log2(len(temporal_dist))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                # Lower entropy = more predictable/stable pattern
                scores['temporal_consistency'] = 1.0 - normalized_entropy
            else:
                scores['temporal_consistency'] = 0.5
        else:
            scores['temporal_consistency'] = 0.5
        
        # Evolution pattern stability
        if usage_metrics.evolution_velocity > 0:
            # Stable evolution should be moderate, not too high or too low
            optimal_velocity = 0.2  # Changes per day
            velocity_score = 1.0 - abs(usage_metrics.evolution_velocity - optimal_velocity) / 1.0
            scores['evolution_stability'] = max(0.0, velocity_score)
        else:
            scores['evolution_stability'] = 0.6  # Some evolution is expected
        
        return scores
    
    async def _calculate_collaboration_value_metrics(self, 
                                                   memory_id: str,
                                                   usage_metrics: Any) -> Dict[str, float]:
        """Calculate collaboration value component scores"""
        scores = {}
        
        # Collaboration frequency score
        collaboration_score = usage_metrics.collaboration_score
        scores['collaboration_frequency'] = collaboration_score
        
        # Context sharing score
        context_diversity = usage_metrics.context_diversity
        if context_diversity >= 5:
            scores['context_sharing'] = 1.0
        elif context_diversity >= 3:
            scores['context_sharing'] = 0.8
        elif context_diversity >= 2:
            scores['context_sharing'] = 0.6
        else:
            scores['context_sharing'] = 0.3
        
        # Knowledge transfer potential
        access_frequency = usage_metrics.access_frequency
        if collaboration_score > 0.5 and access_frequency > 0.2:
            scores['knowledge_transfer'] = min(1.0, collaboration_score + access_frequency / 2.0)
        else:
            scores['knowledge_transfer'] = collaboration_score * 0.5
        
        return scores
    
    def _aggregate_component_scores(self, component_scores: Dict[str, float]) -> float:
        """Aggregate component scores into a category score"""
        if not component_scores:
            return 0.5
        
        # Weighted average with slight penalty for missing components
        total_weight = len(component_scores)
        score_sum = sum(component_scores.values())
        
        return score_sum / total_weight if total_weight > 0 else 0.5
    
    def _calculate_score_confidence(self, 
                                   evolution_summary: Dict[str, Any],
                                   usage_metrics: Any) -> float:
        """Calculate confidence in the evolution score"""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Data availability
        total_events = evolution_summary.get('total_events', 0)
        if total_events >= 20:
            confidence += 0.3
        elif total_events >= 10:
            confidence += 0.2
        elif total_events >= 5:
            confidence += 0.1
        
        # Factor 2: Time span of data
        events = evolution_summary.get('events', [])
        if len(events) >= 2:
            first_event = datetime.fromisoformat(events[0]['timestamp'])
            last_event = datetime.fromisoformat(events[-1]['timestamp'])
            time_span_days = (last_event - first_event).days
            
            if time_span_days >= 30:
                confidence += 0.2
            elif time_span_days >= 7:
                confidence += 0.1
        
        # Factor 3: Metric completeness
        if usage_metrics.access_frequency > 0:
            confidence += 0.1
        if usage_metrics.context_diversity > 0:
            confidence += 0.1
        if usage_metrics.collaboration_score > 0:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def _generate_recommendations(self, 
                                      memory_id: str,
                                      category_scores: Dict[MetricCategory, float],
                                      component_scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on scores"""
        recommendations = []
        
        # Usage quality recommendations
        if category_scores.get(MetricCategory.USAGE_QUALITY, 0) < 0.6:
            if component_scores.get('access_frequency', 0) < 0.5:
                recommendations.append("Increase memory usage frequency through better discoverability")
            if component_scores.get('usage_efficiency', 0) < 0.7:
                recommendations.append("Improve memory content quality for better recall success")
            if component_scores.get('context_diversity', 0) < 0.4:
                recommendations.append("Encourage usage across more diverse contexts")
        
        # Evolution health recommendations
        if category_scores.get(MetricCategory.EVOLUTION_HEALTH, 0) < 0.6:
            if component_scores.get('evolution_activity', 0) < 0.3:
                recommendations.append("Memory may benefit from periodic updates and refinements")
            if component_scores.get('data_completeness', 0) < 0.6:
                recommendations.append("Gather more usage data for better analysis")
        
        # Performance recommendations
        if category_scores.get(MetricCategory.PERFORMANCE, 0) < 0.7:
            if component_scores.get('response_efficiency', 0) < 0.7:
                recommendations.append("Optimize memory structure for faster access")
            if component_scores.get('cache_efficiency', 0) < 0.6:
                recommendations.append("Implement better caching strategy for this memory")
        
        # Lifecycle recommendations
        if category_scores.get(MetricCategory.LIFECYCLE_FITNESS, 0) < 0.6:
            if component_scores.get('stage_appropriateness', 0) < 0.5:
                recommendations.append("Review lifecycle stage assignment")
            if component_scores.get('retention_value', 0) < 0.4:
                recommendations.append("Consider archiving or improving memory content")
        
        # Pattern stability recommendations
        if category_scores.get(MetricCategory.PATTERN_STABILITY, 0) < 0.5:
            recommendations.append("Work on establishing more consistent usage patterns")
        
        # Collaboration recommendations
        if category_scores.get(MetricCategory.COLLABORATION_VALUE, 0) < 0.4:
            recommendations.append("Increase cross-context usage for better collaboration value")
        
        return recommendations
    
    def _identify_health_flags(self, 
                              category_scores: Dict[MetricCategory, float],
                              component_scores: Dict[str, float]) -> List[str]:
        """Identify health flags based on scores"""
        flags = []
        
        # Critical flags (score < 0.3)
        if category_scores.get(MetricCategory.USAGE_QUALITY, 0) < 0.3:
            flags.append("CRITICAL: Very low usage quality")
        
        if component_scores.get('usage_efficiency', 0) < 0.3:
            flags.append("CRITICAL: Poor recall success rate")
        
        # Warning flags (score < 0.5)
        if category_scores.get(MetricCategory.EVOLUTION_HEALTH, 0) < 0.5:
            flags.append("WARNING: Evolution health concerns")
        
        if component_scores.get('access_frequency', 0) < 0.2:
            flags.append("WARNING: Very low usage frequency")
        
        if category_scores.get(MetricCategory.PERFORMANCE, 0) < 0.5:
            flags.append("WARNING: Performance issues detected")
        
        # Info flags
        if component_scores.get('collaboration_frequency', 0) < 0.3:
            flags.append("INFO: Limited collaboration value")
        
        if component_scores.get('evolution_activity', 0) < 0.2:
            flags.append("INFO: Memory not evolving")
        
        return flags
    
    def _calculate_score_trend(self, memory_id: str, current_score: float) -> Optional[str]:
        """Calculate score trend based on history"""
        if memory_id not in self._score_history or len(self._score_history[memory_id]) < 3:
            return None
        
        history = self._score_history[memory_id]
        recent_scores = [score.overall_score for score in history[-5:]]  # Last 5 scores
        
        if len(recent_scores) < 3:
            return None
        
        # Simple trend calculation
        first_half = statistics.mean(recent_scores[:len(recent_scores)//2])
        second_half = statistics.mean(recent_scores[len(recent_scores)//2:])
        
        difference = second_half - first_half
        
        if difference > 0.05:
            return "improving"
        elif difference < -0.05:
            return "declining"
        else:
            return "stable"
    
    async def calculate_system_metrics(self) -> SystemMetrics:
        """Calculate system-wide evolution metrics"""
        # Get all memories with lifecycle status
        all_memory_ids = list(self.lifecycle_manager._memory_lifecycle_status.keys())
        
        # Calculate scores for all memories
        evolution_scores = []
        for memory_id in all_memory_ids:
            try:
                score = await self.calculate_evolution_score(memory_id)
                evolution_scores.append(score)
            except Exception:
                continue  # Skip memories that can't be scored
        
        if not evolution_scores:
            return SystemMetrics(
                total_memories=0,
                avg_evolution_score=0.0,
                score_distribution={},
                healthy_memories=0,
                evolution_velocity=0.0,
                pattern_diversity=0,
                lifecycle_distribution={},
                efficiency_metrics={},
                optimization_opportunities=[]
            )
        
        # Calculate metrics
        total_memories = len(evolution_scores)
        avg_score = statistics.mean(score.overall_score for score in evolution_scores)
        
        # Score distribution
        score_ranges = {
            "excellent_0.9+": 0,
            "good_0.7-0.9": 0,
            "fair_0.5-0.7": 0,
            "poor_0.3-0.5": 0,
            "critical_<0.3": 0
        }
        
        for score in evolution_scores:
            if score.overall_score >= 0.9:
                score_ranges["excellent_0.9+"] += 1
            elif score.overall_score >= 0.7:
                score_ranges["good_0.7-0.9"] += 1
            elif score.overall_score >= 0.5:
                score_ranges["fair_0.5-0.7"] += 1
            elif score.overall_score >= 0.3:
                score_ranges["poor_0.3-0.5"] += 1
            else:
                score_ranges["critical_<0.3"] += 1
        
        # Healthy memories (score >= 0.7)
        healthy_memories = sum(1 for score in evolution_scores if score.overall_score >= 0.7)
        
        # System evolution velocity
        evolution_velocities = []
        for memory_id in all_memory_ids:
            try:
                usage_metrics = await self.usage_analytics.calculate_usage_metrics(memory_id)
                evolution_velocities.append(usage_metrics.evolution_velocity)
            except Exception:
                continue
        
        system_evolution_velocity = statistics.mean(evolution_velocities) if evolution_velocities else 0.0
        
        # Pattern diversity
        detected_patterns = await self.usage_analytics.analyze_usage_patterns()
        pattern_types = set(pattern.pattern_type for pattern in detected_patterns)
        pattern_diversity = len(pattern_types)
        
        # Lifecycle distribution
        lifecycle_distribution = {}
        for status in self.lifecycle_manager._memory_lifecycle_status.values():
            stage = status.current_stage
            lifecycle_distribution[stage] = lifecycle_distribution.get(stage, 0) + 1
        
        # Efficiency metrics
        efficiency_metrics = {
            "avg_usage_efficiency": statistics.mean(
                score.component_scores.get('usage_efficiency', 0)
                for score in evolution_scores
            ),
            "avg_response_efficiency": statistics.mean(
                score.component_scores.get('response_efficiency', 0)
                for score in evolution_scores
            ),
            "avg_collaboration_score": statistics.mean(
                score.component_scores.get('collaboration_frequency', 0)
                for score in evolution_scores
            )
        }
        
        # Optimization opportunities
        optimization_opportunities = []
        
        if score_ranges["poor_0.3-0.5"] + score_ranges["critical_<0.3"] > total_memories * 0.2:
            optimization_opportunities.append("High number of low-scoring memories need attention")
        
        if efficiency_metrics["avg_usage_efficiency"] < 0.7:
            optimization_opportunities.append("System-wide usage efficiency needs improvement")
        
        if healthy_memories / total_memories < 0.6:
            optimization_opportunities.append("Less than 60% of memories are healthy")
        
        if pattern_diversity < 3:
            optimization_opportunities.append("Limited usage pattern diversity detected")
        
        return SystemMetrics(
            total_memories=total_memories,
            avg_evolution_score=avg_score,
            score_distribution=score_ranges,
            healthy_memories=healthy_memories,
            evolution_velocity=system_evolution_velocity,
            pattern_diversity=pattern_diversity,
            lifecycle_distribution=lifecycle_distribution,
            efficiency_metrics=efficiency_metrics,
            optimization_opportunities=optimization_opportunities
        )
    
    async def predict_evolution_score(self, memory_id: str, days_ahead: int = 30) -> Dict[str, Any]:
        """Predict future evolution score based on current trends"""
        current_score = await self.calculate_evolution_score(memory_id)
        
        # Get historical trend
        trend = current_score.score_trend
        
        # Predict future score based on trend
        if trend == "improving":
            trend_factor = 1.02  # 2% improvement
        elif trend == "declining":
            trend_factor = 0.98  # 2% decline
        else:
            trend_factor = 1.0   # Stable
        
        # Predict usage trends
        usage_prediction = await self.usage_analytics.predict_usage_trends(memory_id, days_ahead)
        
        # Simulate score evolution
        predicted_score = current_score.overall_score
        confidence = current_score.confidence
        
        for day in range(1, days_ahead + 1):
            # Apply trend factor
            predicted_score *= trend_factor
            
            # Apply lifecycle influence
            lifecycle_status = self.lifecycle_manager.get_memory_lifecycle_status(memory_id)
            if lifecycle_status:
                stage = lifecycle_status.current_stage
                if stage in [LifecycleStage.DECLINING, LifecycleStage.DORMANT]:
                    predicted_score *= 0.999  # Gradual decline
                elif stage in [LifecycleStage.ACTIVE, LifecycleStage.PEAK]:
                    predicted_score *= 1.001  # Gradual improvement
            
            # Ensure bounds
            predicted_score = max(0.0, min(1.0, predicted_score))
            
            # Decrease confidence over time
            confidence *= 0.95
        
        return {
            'memory_id': memory_id,
            'current_score': current_score.overall_score,
            'predicted_score': predicted_score,
            'confidence': confidence,
            'trend': trend,
            'prediction_horizon_days': days_ahead,
            'usage_prediction': usage_prediction,
            'predicted_at': datetime.now().isoformat()
        }
    
    def get_cached_score(self, memory_id: str) -> Optional[EvolutionScore]:
        """Get cached evolution score for a memory"""
        return self._score_cache.get(memory_id)
    
    def get_score_history(self, memory_id: str) -> List[EvolutionScore]:
        """Get score history for a memory"""
        return self._score_history.get(memory_id, [])
    
    async def benchmark_against_standards(self, memory_id: str) -> Dict[str, Any]:
        """Benchmark a memory's evolution against quality standards"""
        score = await self.calculate_evolution_score(memory_id)
        
        benchmarks = {
            'overall_score': {
                'current': score.overall_score,
                'excellent_threshold': self.benchmarks.excellent_threshold,
                'good_threshold': self.benchmarks.good_threshold,
                'fair_threshold': self.benchmarks.fair_threshold,
                'poor_threshold': self.benchmarks.poor_threshold,
                'rating': self._get_score_rating(score.overall_score)
            }
        }
        
        # Category benchmarks
        for category, category_score in score.category_scores.items():
            benchmarks[category.value] = {
                'current': category_score,
                'target': 0.7,  # General target for categories
                'rating': self._get_score_rating(category_score)
            }
        
        return benchmarks
    
    def _get_score_rating(self, score: float) -> str:
        """Convert numeric score to rating"""
        if score >= self.benchmarks.excellent_threshold:
            return "excellent"
        elif score >= self.benchmarks.good_threshold:
            return "good"
        elif score >= self.benchmarks.fair_threshold:
            return "fair"
        elif score >= self.benchmarks.poor_threshold:
            return "poor"
        else:
            return "critical"