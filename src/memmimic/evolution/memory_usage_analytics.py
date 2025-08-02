"""
Memory Usage Analytics and Pattern Detection System

Advanced analytics for memory usage patterns, behavioral insights, and predictive modeling.
Provides deep insights into how memories are used and how usage patterns evolve over time.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
from collections import defaultdict, Counter
import statistics

from .memory_evolution_tracker import MemoryEvolutionTracker, MemoryEventType
from .memory_lifecycle_manager import MemoryLifecycleManager, LifecycleStage
from ..errors.exceptions import MemMimicError


class UsagePatternType(Enum):
    """Types of usage patterns"""
    BURST = "burst"                    # Intensive usage in short periods
    STEADY = "steady"                  # Consistent regular usage
    PERIODIC = "periodic"              # Regular intervals
    RANDOM = "random"                  # Unpredictable access
    DECAY = "decay"                    # Decreasing usage over time
    GROWTH = "growth"                  # Increasing usage over time
    SEASONAL = "seasonal"              # Time-based patterns
    CONTEXTUAL = "contextual"          # Context-dependent usage
    COLLABORATIVE = "collaborative"    # Multi-user patterns
    DORMANT = "dormant"               # Little to no usage


@dataclass
class UsagePattern:
    """Represents a detected usage pattern"""
    pattern_id: str
    pattern_type: UsagePatternType
    memory_ids: List[str]
    confidence: float  # 0.0 to 1.0
    description: str
    detected_at: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    frequency: Optional[float] = None  # Events per day
    duration: Optional[timedelta] = None  # Pattern duration
    strength: float = 0.5  # Pattern strength/significance


@dataclass
class MemoryCluster:
    """Represents a cluster of memories with similar usage patterns"""
    cluster_id: str
    memory_ids: List[str]
    primary_pattern: UsagePatternType
    cluster_center: Dict[str, float]  # Feature center
    cohesion_score: float  # How similar memories are
    usage_characteristics: Dict[str, Any]
    created_at: datetime


@dataclass
class UsageMetrics:
    """Comprehensive usage metrics for analysis"""
    memory_id: str
    total_events: int = 0
    access_frequency: float = 0.0
    access_variance: float = 0.0  # Consistency of access
    peak_usage_periods: List[Tuple[datetime, datetime]] = field(default_factory=list)
    usage_efficiency: float = 0.0  # Successful recalls / total accesses
    context_diversity: int = 0
    temporal_distribution: Dict[str, int] = field(default_factory=dict)  # Hour/day patterns
    collaboration_score: float = 0.0  # Multi-context usage
    evolution_velocity: float = 0.0  # Rate of change


class MemoryUsageAnalytics:
    """
    Advanced analytics engine for memory usage patterns and behavioral insights.
    
    Core capabilities:
    - Pattern detection: Identifies usage patterns across memories
    - Clustering analysis: Groups memories by similar usage characteristics
    - Predictive modeling: Forecasts future usage patterns
    - Anomaly detection: Identifies unusual usage behaviors
    - Performance analytics: Measures usage efficiency and effectiveness
    - Temporal analysis: Analyzes time-based usage patterns
    - Collaborative patterns: Detects multi-user/context usage patterns
    """
    
    def __init__(self, 
                 evolution_tracker: MemoryEvolutionTracker,
                 lifecycle_manager: MemoryLifecycleManager):
        self.evolution_tracker = evolution_tracker
        self.lifecycle_manager = lifecycle_manager
        self._detected_patterns: List[UsagePattern] = []
        self._memory_clusters: List[MemoryCluster] = []
        self._usage_metrics_cache: Dict[str, UsageMetrics] = {}
        self._last_analysis = datetime.now() - timedelta(days=1)
    
    async def analyze_usage_patterns(self, lookback_days: int = 30) -> List[UsagePattern]:
        """Comprehensive analysis of memory usage patterns"""
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        
        # Get all usage data
        usage_data = await self._collect_usage_data(cutoff_time)
        
        # Detect different types of patterns
        patterns = []
        
        # 1. Burst patterns
        burst_patterns = await self._detect_burst_patterns(usage_data)
        patterns.extend(burst_patterns)
        
        # 2. Periodic patterns
        periodic_patterns = await self._detect_periodic_patterns(usage_data)
        patterns.extend(periodic_patterns)
        
        # 3. Decay/Growth patterns
        trend_patterns = await self._detect_trend_patterns(usage_data)
        patterns.extend(trend_patterns)
        
        # 4. Contextual patterns
        contextual_patterns = await self._detect_contextual_patterns(usage_data)
        patterns.extend(contextual_patterns)
        
        # 5. Collaborative patterns
        collaborative_patterns = await self._detect_collaborative_patterns(usage_data)
        patterns.extend(collaborative_patterns)
        
        # 6. Temporal patterns
        temporal_patterns = await self._detect_temporal_patterns(usage_data)
        patterns.extend(temporal_patterns)
        
        self._detected_patterns = patterns
        return patterns
    
    async def _collect_usage_data(self, cutoff_time: datetime) -> Dict[str, List[Dict]]:
        """Collect and organize usage data for analysis"""
        try:
            conn = sqlite3.connect(self.evolution_tracker.db_path)
            cursor = conn.cursor()
            
            # Get all relevant events
            cursor.execute('''
                SELECT memory_id, event_type, timestamp, context, trigger
                FROM memory_events 
                WHERE timestamp > ? AND event_type IN ('accessed', 'recalled', 'modified')
                ORDER BY memory_id, timestamp ASC
            ''', (cutoff_time.isoformat(),))
            
            usage_data = defaultdict(list)
            for row in cursor.fetchall():
                memory_id = row[0]
                event_data = {
                    'event_type': row[1],
                    'timestamp': datetime.fromisoformat(row[2]),
                    'context': json.loads(row[3]) if row[3] else {},
                    'trigger': row[4]
                }
                usage_data[memory_id].append(event_data)
            
            conn.close()
            return dict(usage_data)
            
        except Exception as e:
            raise MemMimicError(f"Failed to collect usage data: {e}")
    
    async def _detect_burst_patterns(self, usage_data: Dict[str, List[Dict]]) -> List[UsagePattern]:
        """Detect burst usage patterns (intensive usage in short periods)"""
        burst_patterns = []
        
        for memory_id, events in usage_data.items():
            if len(events) < 5:  # Need sufficient data
                continue
            
            # Group events by time windows (1-hour windows)
            window_size = timedelta(hours=1)
            windows = defaultdict(list)
            
            for event in events:
                window_key = event['timestamp'].replace(minute=0, second=0, microsecond=0)
                windows[window_key].append(event)
            
            # Find burst windows (>= 5 events in 1 hour)
            burst_windows = []
            for window_time, window_events in windows.items():
                if len(window_events) >= 5:
                    burst_windows.append((window_time, len(window_events)))
            
            if len(burst_windows) >= 2:  # At least 2 burst periods
                avg_burst_intensity = sum(count for _, count in burst_windows) / len(burst_windows)
                
                pattern = UsagePattern(
                    pattern_id=f"burst_{memory_id}_{int(datetime.now().timestamp())}",
                    pattern_type=UsagePatternType.BURST,
                    memory_ids=[memory_id],
                    confidence=min(0.9, len(burst_windows) / 10.0 + 0.5),
                    description=f"Burst usage pattern with {len(burst_windows)} intense periods",
                    detected_at=datetime.now(),
                    parameters={
                        'burst_windows': len(burst_windows),
                        'avg_intensity': avg_burst_intensity,
                        'max_intensity': max(count for _, count in burst_windows)
                    },
                    frequency=len(burst_windows) / 30,  # Bursts per day
                    strength=min(1.0, avg_burst_intensity / 10.0)
                )
                burst_patterns.append(pattern)
        
        return burst_patterns
    
    async def _detect_periodic_patterns(self, usage_data: Dict[str, List[Dict]]) -> List[UsagePattern]:
        """Detect periodic usage patterns (regular intervals)"""
        periodic_patterns = []
        
        for memory_id, events in usage_data.items():
            if len(events) < 10:  # Need more data for periodicity
                continue
            
            # Calculate intervals between events
            intervals = []
            for i in range(1, len(events)):
                interval = (events[i]['timestamp'] - events[i-1]['timestamp']).total_seconds()
                intervals.append(interval)
            
            if len(intervals) < 5:
                continue
            
            # Check for periodicity using standard deviation
            mean_interval = statistics.mean(intervals)
            if mean_interval > 0:
                std_interval = statistics.stdev(intervals)
                coefficient_of_variation = std_interval / mean_interval
                
                # Low coefficient of variation indicates periodicity
                if coefficient_of_variation < 0.5:  # Less than 50% variation
                    pattern = UsagePattern(
                        pattern_id=f"periodic_{memory_id}_{int(datetime.now().timestamp())}",
                        pattern_type=UsagePatternType.PERIODIC,
                        memory_ids=[memory_id],
                        confidence=max(0.3, 1.0 - coefficient_of_variation),
                        description=f"Periodic usage with {mean_interval/3600:.1f} hour intervals",
                        detected_at=datetime.now(),
                        parameters={
                            'mean_interval_hours': mean_interval / 3600,
                            'coefficient_of_variation': coefficient_of_variation,
                            'regularity_score': 1.0 - coefficient_of_variation
                        },
                        frequency=86400 / mean_interval if mean_interval > 0 else 0,  # Events per day
                        strength=max(0.1, 1.0 - coefficient_of_variation)
                    )
                    periodic_patterns.append(pattern)
        
        return periodic_patterns
    
    async def _detect_trend_patterns(self, usage_data: Dict[str, List[Dict]]) -> List[UsagePattern]:
        """Detect decay/growth trends in usage patterns"""
        trend_patterns = []
        
        for memory_id, events in usage_data.items():
            if len(events) < 15:  # Need sufficient data for trend analysis
                continue
            
            # Group events by week to analyze trends
            weekly_counts = defaultdict(int)
            for event in events:
                week_key = event['timestamp'].isocalendar()[:2]  # (year, week)
                weekly_counts[week_key] += 1
            
            if len(weekly_counts) < 3:  # Need at least 3 weeks
                continue
            
            # Calculate trend using simple linear regression
            weeks = sorted(weekly_counts.keys())
            counts = [weekly_counts[week] for week in weeks]
            
            # Simple trend calculation
            x_values = list(range(len(counts)))
            if len(x_values) > 1:
                # Calculate slope
                x_mean = statistics.mean(x_values)
                y_mean = statistics.mean(counts)
                
                numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, counts))
                denominator = sum((x - x_mean) ** 2 for x in x_values)
                
                if denominator > 0:
                    slope = numerator / denominator
                    
                    # Determine trend type and significance
                    avg_count = statistics.mean(counts)
                    relative_slope = slope / (avg_count + 1)  # Normalize by average
                    
                    if abs(relative_slope) > 0.1:  # Significant trend
                        pattern_type = UsagePatternType.GROWTH if slope > 0 else UsagePatternType.DECAY
                        
                        pattern = UsagePattern(
                            pattern_id=f"trend_{memory_id}_{int(datetime.now().timestamp())}",
                            pattern_type=pattern_type,
                            memory_ids=[memory_id],
                            confidence=min(0.9, abs(relative_slope) * 2),
                            description=f"{'Growing' if slope > 0 else 'Decaying'} usage trend",
                            detected_at=datetime.now(),
                            parameters={
                                'slope': slope,
                                'relative_slope': relative_slope,
                                'weeks_analyzed': len(weeks),
                                'avg_weekly_count': avg_count
                            },
                            strength=min(1.0, abs(relative_slope) * 3)
                        )
                        trend_patterns.append(pattern)
        
        return trend_patterns
    
    async def _detect_contextual_patterns(self, usage_data: Dict[str, List[Dict]]) -> List[UsagePattern]:
        """Detect context-dependent usage patterns"""
        contextual_patterns = []
        
        # Group by context similarity
        context_groups = defaultdict(list)
        
        for memory_id, events in usage_data.items():
            if len(events) < 5:
                continue
            
            # Analyze context diversity
            contexts = [event.get('context', {}) for event in events]
            context_keys = set()
            for context in contexts:
                context_keys.update(context.keys())
            
            if len(context_keys) > 1:  # Has contextual information
                # Simple context pattern detection
                context_frequency = Counter()
                for context in contexts:
                    context_signature = tuple(sorted(context.items()))
                    context_frequency[context_signature] += 1
                
                # Check for dominant contexts
                total_events = len(events)
                dominant_contexts = [
                    (ctx, count) for ctx, count in context_frequency.items()
                    if count >= total_events * 0.3  # At least 30% of events
                ]
                
                if len(dominant_contexts) >= 2:  # Multiple significant contexts
                    pattern = UsagePattern(
                        pattern_id=f"contextual_{memory_id}_{int(datetime.now().timestamp())}",
                        pattern_type=UsagePatternType.CONTEXTUAL,
                        memory_ids=[memory_id],
                        confidence=min(0.9, len(dominant_contexts) / 5.0 + 0.4),
                        description=f"Context-dependent usage with {len(dominant_contexts)} primary contexts",
                        detected_at=datetime.now(),
                        parameters={
                            'dominant_contexts': len(dominant_contexts),
                            'context_diversity': len(context_keys),
                            'max_context_frequency': max(count for _, count in dominant_contexts) / total_events
                        },
                        strength=len(dominant_contexts) / 10.0
                    )
                    contextual_patterns.append(pattern)
        
        return contextual_patterns
    
    async def _detect_collaborative_patterns(self, usage_data: Dict[str, List[Dict]]) -> List[UsagePattern]:
        """Detect collaborative usage patterns (multi-user/session patterns)"""
        collaborative_patterns = []
        
        # Look for memories accessed across multiple triggers/contexts
        for memory_id, events in usage_data.items():
            if len(events) < 8:  # Need sufficient data
                continue
            
            # Analyze trigger diversity
            triggers = [event.get('trigger') for event in events if event.get('trigger')]
            unique_triggers = set(triggers)
            
            # Analyze context diversity for collaboration indicators
            contexts = [event.get('context', {}) for event in events]
            session_indicators = set()
            user_indicators = set()
            
            for context in contexts:
                if 'session_id' in context:
                    session_indicators.add(context['session_id'])
                if 'user_context' in context:
                    user_indicators.add(str(context['user_context']))
                if 'context_hash' in context:
                    session_indicators.add(context['context_hash'])
            
            # Determine if collaborative
            collaboration_score = 0.0
            
            if len(unique_triggers) >= 3:  # Multiple trigger types
                collaboration_score += 0.3
            
            if len(session_indicators) >= 3:  # Multiple sessions
                collaboration_score += 0.4
            
            if len(user_indicators) >= 2:  # Multiple users/contexts
                collaboration_score += 0.5
            
            if collaboration_score >= 0.5:  # Threshold for collaboration
                pattern = UsagePattern(
                    pattern_id=f"collaborative_{memory_id}_{int(datetime.now().timestamp())}",
                    pattern_type=UsagePatternType.COLLABORATIVE,
                    memory_ids=[memory_id],
                    confidence=min(0.9, collaboration_score),
                    description=f"Collaborative usage across {len(session_indicators)} sessions",
                    detected_at=datetime.now(),
                    parameters={
                        'unique_triggers': len(unique_triggers),
                        'session_diversity': len(session_indicators),
                        'user_diversity': len(user_indicators),
                        'collaboration_score': collaboration_score
                    },
                    strength=collaboration_score
                )
                collaborative_patterns.append(pattern)
        
        return collaborative_patterns
    
    async def _detect_temporal_patterns(self, usage_data: Dict[str, List[Dict]]) -> List[UsagePattern]:
        """Detect temporal usage patterns (time-of-day, day-of-week patterns)"""
        temporal_patterns = []
        
        for memory_id, events in usage_data.items():
            if len(events) < 20:  # Need more data for temporal analysis
                continue
            
            # Analyze hour-of-day patterns
            hour_distribution = defaultdict(int)
            day_distribution = defaultdict(int)
            
            for event in events:
                timestamp = event['timestamp']
                hour_distribution[timestamp.hour] += 1
                day_distribution[timestamp.weekday()] += 1
            
            # Check for temporal clustering
            total_events = len(events)
            
            # Hour pattern analysis
            max_hour_count = max(hour_distribution.values()) if hour_distribution else 0
            hour_concentration = max_hour_count / total_events if total_events > 0 else 0
            
            # Day pattern analysis
            max_day_count = max(day_distribution.values()) if day_distribution else 0
            day_concentration = max_day_count / total_events if total_events > 0 else 0
            
            # Detect strong temporal patterns
            if hour_concentration > 0.4 or day_concentration > 0.5:  # Concentrated usage
                pattern = UsagePattern(
                    pattern_id=f"temporal_{memory_id}_{int(datetime.now().timestamp())}",
                    pattern_type=UsagePatternType.SEASONAL,
                    memory_ids=[memory_id],
                    confidence=max(hour_concentration, day_concentration),
                    description=f"Temporal usage pattern concentrated in specific times",
                    detected_at=datetime.now(),
                    parameters={
                        'hour_concentration': hour_concentration,
                        'day_concentration': day_concentration,
                        'peak_hour': max(hour_distribution, key=hour_distribution.get) if hour_distribution else None,
                        'peak_day': max(day_distribution, key=day_distribution.get) if day_distribution else None,
                        'hour_distribution': dict(hour_distribution),
                        'day_distribution': dict(day_distribution)
                    },
                    strength=max(hour_concentration, day_concentration)
                )
                temporal_patterns.append(pattern)
        
        return temporal_patterns
    
    async def calculate_usage_metrics(self, memory_id: str) -> UsageMetrics:
        """Calculate comprehensive usage metrics for a memory"""
        # Get evolution summary
        evolution_summary = await self.evolution_tracker.get_memory_evolution_summary(memory_id)
        events = evolution_summary.get('events', [])
        
        if not events:
            return UsageMetrics(memory_id=memory_id)
        
        # Basic metrics
        total_events = len(events)
        access_events = [e for e in events if e['event_type'] in ['accessed', 'recalled']]
        
        # Calculate access frequency and variance
        if len(access_events) > 1:
            timestamps = [datetime.fromisoformat(e['timestamp']) for e in access_events]
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            
            if intervals:
                mean_interval = statistics.mean(intervals)
                access_frequency = 86400 / mean_interval if mean_interval > 0 else 0  # Events per day
                access_variance = statistics.variance(intervals) if len(intervals) > 1 else 0
            else:
                access_frequency = 0
                access_variance = 0
        else:
            access_frequency = 0
            access_variance = 0
        
        # Calculate usage efficiency (successful recalls)
        recall_events = [e for e in events if e['event_type'] == 'recalled']
        successful_recalls = sum(1 for e in recall_events if e.get('context', {}).get('recall_success', True))
        usage_efficiency = successful_recalls / len(recall_events) if recall_events else 1.0
        
        # Context diversity
        contexts = [e.get('context', {}) for e in events]
        unique_contexts = set()
        for context in contexts:
            context_signature = tuple(sorted(context.items()))
            unique_contexts.add(context_signature)
        context_diversity = len(unique_contexts)
        
        # Temporal distribution
        temporal_distribution = defaultdict(int)
        for event in events:
            timestamp = datetime.fromisoformat(event['timestamp'])
            hour_key = f"hour_{timestamp.hour}"
            day_key = f"day_{timestamp.weekday()}"
            temporal_distribution[hour_key] += 1
            temporal_distribution[day_key] += 1
        
        # Peak usage periods (periods with high activity)
        peak_periods = []
        if len(access_events) >= 5:
            # Group by day and find high-activity days
            daily_counts = defaultdict(int)
            for event in access_events:
                day_key = datetime.fromisoformat(event['timestamp']).date()
                daily_counts[day_key] += 1
            
            avg_daily = statistics.mean(daily_counts.values()) if daily_counts else 0
            threshold = avg_daily * 1.5  # 50% above average
            
            for day, count in daily_counts.items():
                if count >= threshold:
                    start_time = datetime.combine(day, datetime.min.time())
                    end_time = start_time + timedelta(days=1)
                    peak_periods.append((start_time, end_time))
        
        # Collaboration score (based on context diversity and trigger variety)
        triggers = [e.get('trigger') for e in events if e.get('trigger')]
        unique_triggers = len(set(triggers))
        collaboration_score = min(1.0, (context_diversity + unique_triggers) / 20.0)
        
        # Evolution velocity (rate of changes over time)
        change_events = [e for e in events if e['event_type'] in ['modified', 'importance_changed']]
        if events:
            first_event = datetime.fromisoformat(events[0]['timestamp'])
            last_event = datetime.fromisoformat(events[-1]['timestamp'])
            time_span = (last_event - first_event).total_seconds()
            evolution_velocity = len(change_events) / (time_span / 86400) if time_span > 0 else 0  # Changes per day
        else:
            evolution_velocity = 0
        
        metrics = UsageMetrics(
            memory_id=memory_id,
            total_events=total_events,
            access_frequency=access_frequency,
            access_variance=access_variance,
            peak_usage_periods=peak_periods,
            usage_efficiency=usage_efficiency,
            context_diversity=context_diversity,
            temporal_distribution=dict(temporal_distribution),
            collaboration_score=collaboration_score,
            evolution_velocity=evolution_velocity
        )
        
        # Cache metrics
        self._usage_metrics_cache[memory_id] = metrics
        
        return metrics
    
    async def cluster_memories_by_usage(self, memory_ids: List[str] = None) -> List[MemoryCluster]:
        """Cluster memories based on usage patterns"""
        if memory_ids is None:
            # Get all tracked memories
            memory_ids = list(self._usage_metrics_cache.keys())
            # Add any memories from lifecycle manager
            lifecycle_memories = [
                status.memory_id 
                for status in self.lifecycle_manager._memory_lifecycle_status.values()
            ]
            memory_ids.extend(lifecycle_memories)
            memory_ids = list(set(memory_ids))  # Remove duplicates
        
        # Calculate metrics for all memories
        metrics_data = []
        for memory_id in memory_ids:
            if memory_id not in self._usage_metrics_cache:
                await self.calculate_usage_metrics(memory_id)
            
            metrics = self._usage_metrics_cache.get(memory_id)
            if metrics:
                metrics_data.append(metrics)
        
        if len(metrics_data) < 2:
            return []
        
        # Simple clustering based on usage characteristics
        clusters = []
        
        # Feature extraction for clustering
        features = []
        for metrics in metrics_data:
            feature_vector = [
                metrics.access_frequency,
                metrics.usage_efficiency,
                metrics.context_diversity / 10.0,  # Normalize
                metrics.collaboration_score,
                metrics.evolution_velocity
            ]
            features.append((metrics.memory_id, feature_vector))
        
        # Simple k-means-like clustering (manual implementation)
        k = min(5, len(features) // 3 + 1)  # Adaptive number of clusters
        
        # Initialize cluster centers randomly
        import random
        random.seed(42)  # For reproducibility
        cluster_centers = []
        for i in range(k):
            center = [random.uniform(0, 1) for _ in range(5)]
            cluster_centers.append(center)
        
        # Assign memories to clusters
        cluster_assignments = {}
        for memory_id, feature_vector in features:
            best_cluster = 0
            best_distance = float('inf')
            
            for i, center in enumerate(cluster_centers):
                distance = sum((a - b) ** 2 for a, b in zip(feature_vector, center)) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_cluster = i
            
            cluster_assignments[memory_id] = best_cluster
        
        # Create cluster objects
        for cluster_id in range(k):
            cluster_memory_ids = [
                memory_id for memory_id, assigned_cluster in cluster_assignments.items()
                if assigned_cluster == cluster_id
            ]
            
            if not cluster_memory_ids:
                continue
            
            # Calculate cluster characteristics
            cluster_features = [
                feature_vector for memory_id, feature_vector in features
                if memory_id in cluster_memory_ids
            ]
            
            # Average features for cluster center
            center = [
                sum(feature[i] for feature in cluster_features) / len(cluster_features)
                for i in range(5)
            ]
            
            # Determine primary pattern based on feature dominance
            if center[0] > 0.5:  # High access frequency
                primary_pattern = UsagePatternType.BURST
            elif center[3] > 0.7:  # High collaboration
                primary_pattern = UsagePatternType.COLLABORATIVE
            elif center[4] > 0.3:  # High evolution
                primary_pattern = UsagePatternType.GROWTH
            elif center[1] > 0.8:  # High efficiency
                primary_pattern = UsagePatternType.STEADY
            else:
                primary_pattern = UsagePatternType.RANDOM
            
            # Calculate cohesion (average distance from center)
            distances = [
                sum((feature[i] - center[i]) ** 2 for i in range(5)) ** 0.5
                for feature in cluster_features
            ]
            cohesion_score = 1.0 - (statistics.mean(distances) if distances else 0)
            
            cluster = MemoryCluster(
                cluster_id=f"cluster_{cluster_id}",
                memory_ids=cluster_memory_ids,
                primary_pattern=primary_pattern,
                cluster_center={
                    'access_frequency': center[0],
                    'usage_efficiency': center[1],
                    'context_diversity': center[2],
                    'collaboration_score': center[3],
                    'evolution_velocity': center[4]
                },
                cohesion_score=max(0.0, cohesion_score),
                usage_characteristics={
                    'avg_access_frequency': center[0],
                    'avg_efficiency': center[1],
                    'pattern_strength': max(center),
                    'cluster_size': len(cluster_memory_ids)
                },
                created_at=datetime.now()
            )
            clusters.append(cluster)
        
        self._memory_clusters = clusters
        return clusters
    
    async def predict_usage_trends(self, memory_id: str, days_ahead: int = 30) -> Dict[str, Any]:
        """Predict future usage trends for a memory"""
        metrics = await self.calculate_usage_metrics(memory_id)
        
        # Simple trend prediction based on current patterns
        current_frequency = metrics.access_frequency
        current_efficiency = metrics.usage_efficiency
        evolution_rate = metrics.evolution_velocity
        
        # Predict trend based on lifecycle stage
        lifecycle_status = self.lifecycle_manager.get_memory_lifecycle_status(memory_id)
        lifecycle_factor = 1.0
        
        if lifecycle_status:
            stage = lifecycle_status.current_stage
            if stage in [LifecycleStage.DECLINING, LifecycleStage.DORMANT]:
                lifecycle_factor = 0.8  # Decay factor
            elif stage in [LifecycleStage.ACTIVE, LifecycleStage.PEAK]:
                lifecycle_factor = 1.1  # Growth factor
        
        # Predict future access frequency with decay/growth
        predicted_frequencies = []
        for day in range(1, days_ahead + 1):
            # Apply decay/growth with lifecycle factor
            decay_factor = 0.98 ** day  # 2% daily decay
            trend_factor = lifecycle_factor ** (day / 30)  # Apply lifecycle factor over month
            
            predicted_frequency = current_frequency * decay_factor * trend_factor
            predicted_frequencies.append({
                'day': day,
                'predicted_frequency': predicted_frequency,
                'confidence': max(0.1, 1.0 - (day / days_ahead) * 0.5)  # Decreasing confidence
            })
        
        # Detect trend direction
        final_frequency = predicted_frequencies[-1]['predicted_frequency']
        trend_direction = 'stable'
        
        if final_frequency > current_frequency * 1.1:
            trend_direction = 'increasing'
        elif final_frequency < current_frequency * 0.9:
            trend_direction = 'decreasing'
        
        return {
            'memory_id': memory_id,
            'current_metrics': {
                'access_frequency': current_frequency,
                'usage_efficiency': current_efficiency,
                'evolution_velocity': evolution_rate
            },
            'predictions': predicted_frequencies,
            'trend_direction': trend_direction,
            'confidence_score': max(0.3, min(0.9, metrics.usage_efficiency + 0.2)),
            'lifecycle_influence': lifecycle_factor,
            'predicted_at': datetime.now().isoformat()
        }
    
    async def detect_usage_anomalies(self, memory_id: str) -> List[Dict[str, Any]]:
        """Detect anomalous usage patterns for a memory"""
        metrics = await self.calculate_usage_metrics(memory_id)
        anomalies = []
        
        # Get evolution summary for detailed analysis
        evolution_summary = await self.evolution_tracker.get_memory_evolution_summary(memory_id)
        events = evolution_summary.get('events', [])
        
        if len(events) < 10:  # Need sufficient data
            return anomalies
        
        # Anomaly 1: Sudden burst of activity
        recent_events = [
            e for e in events 
            if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(days=3)
        ]
        
        if len(recent_events) > len(events) * 0.5:  # More than 50% of activity in last 3 days
            anomalies.append({
                'type': 'sudden_burst',
                'severity': 'medium',
                'description': f'Sudden burst of activity: {len(recent_events)} events in last 3 days',
                'confidence': 0.8,
                'detected_at': datetime.now().isoformat()
            })
        
        # Anomaly 2: Unusual access patterns
        if metrics.access_variance > 0:
            cv = (metrics.access_variance ** 0.5) / (metrics.access_frequency + 0.001)
            if cv > 2.0:  # High coefficient of variation
                anomalies.append({
                    'type': 'irregular_access',
                    'severity': 'low',
                    'description': f'Highly irregular access pattern (CV: {cv:.2f})',
                    'confidence': min(0.9, cv / 3.0),
                    'detected_at': datetime.now().isoformat()
                })
        
        # Anomaly 3: Low usage efficiency
        if metrics.usage_efficiency < 0.5 and metrics.total_events > 5:
            anomalies.append({
                'type': 'low_efficiency',
                'severity': 'medium',
                'description': f'Low usage efficiency: {metrics.usage_efficiency:.2f}',
                'confidence': 1.0 - metrics.usage_efficiency,
                'detected_at': datetime.now().isoformat()
            })
        
        # Anomaly 4: Rapid evolution without usage
        if metrics.evolution_velocity > 0.5 and metrics.access_frequency < 0.1:
            anomalies.append({
                'type': 'evolution_without_usage',
                'severity': 'high',
                'description': 'High evolution rate despite low usage',
                'confidence': 0.7,
                'detected_at': datetime.now().isoformat()
            })
        
        return anomalies
    
    async def generate_usage_insights(self, memory_ids: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive usage insights and recommendations"""
        if memory_ids is None:
            memory_ids = list(self._usage_metrics_cache.keys())
        
        insights = {
            'generated_at': datetime.now().isoformat(),
            'memory_count': len(memory_ids),
            'pattern_summary': {},
            'cluster_insights': {},
            'efficiency_analysis': {},
            'recommendations': []
        }
        
        # Analyze detected patterns
        pattern_counts = Counter(pattern.pattern_type for pattern in self._detected_patterns)
        insights['pattern_summary'] = {
            pattern_type.value: count 
            for pattern_type, count in pattern_counts.items()
        }
        
        # Analyze clusters
        if self._memory_clusters:
            cluster_sizes = [len(cluster.memory_ids) for cluster in self._memory_clusters]
            insights['cluster_insights'] = {
                'total_clusters': len(self._memory_clusters),
                'avg_cluster_size': statistics.mean(cluster_sizes),
                'largest_cluster': max(cluster_sizes),
                'avg_cohesion': statistics.mean(cluster.cohesion_score for cluster in self._memory_clusters)
            }
        
        # Efficiency analysis
        efficiencies = []
        frequencies = []
        
        for memory_id in memory_ids:
            if memory_id in self._usage_metrics_cache:
                metrics = self._usage_metrics_cache[memory_id]
                efficiencies.append(metrics.usage_efficiency)
                frequencies.append(metrics.access_frequency)
        
        if efficiencies:
            insights['efficiency_analysis'] = {
                'avg_efficiency': statistics.mean(efficiencies),
                'min_efficiency': min(efficiencies),
                'max_efficiency': max(efficiencies),
                'avg_frequency': statistics.mean(frequencies),
                'high_efficiency_count': sum(1 for e in efficiencies if e > 0.8),
                'low_efficiency_count': sum(1 for e in efficiencies if e < 0.5)
            }
        
        # Generate recommendations
        recommendations = []
        
        # Pattern-based recommendations
        if pattern_counts.get(UsagePatternType.BURST, 0) > 3:
            recommendations.append({
                'type': 'optimization',
                'priority': 'medium',
                'description': 'Multiple burst patterns detected - consider caching optimization',
                'action': 'implement_burst_caching'
            })
        
        if pattern_counts.get(UsagePatternType.DECAY, 0) > 2:
            recommendations.append({
                'type': 'retention',
                'priority': 'high',
                'description': 'Decay patterns detected - review memory retention policies',
                'action': 'review_retention_policies'
            })
        
        # Efficiency-based recommendations
        if insights.get('efficiency_analysis', {}).get('low_efficiency_count', 0) > len(memory_ids) * 0.2:
            recommendations.append({
                'type': 'quality',
                'priority': 'high',
                'description': 'High number of low-efficiency memories - improve memory quality',
                'action': 'enhance_memory_quality'
            })
        
        insights['recommendations'] = recommendations
        
        return insights