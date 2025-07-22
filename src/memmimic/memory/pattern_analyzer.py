#!/usr/bin/env python3
"""
Memory Pattern Analysis Engine - Phase 3 Advanced Features
Analyzes temporal patterns in memory access, creation, and importance evolution
"""

import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MemoryPattern:
    """Represents a detected memory usage pattern"""

    pattern_id: str
    pattern_type: str  # 'temporal', 'importance', 'access', 'creation'
    description: str
    confidence: float
    frequency: int
    first_detected: datetime
    last_detected: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryTrend:
    """Represents a trend in memory importance or usage"""

    memory_id: str
    trend_type: str  # 'importance_rising', 'importance_falling', 'access_increasing', 'access_decreasing'
    trend_strength: float  # -1.0 to 1.0
    trend_confidence: float  # 0.0 to 1.0
    start_time: datetime
    end_time: datetime
    data_points: List[Tuple[datetime, float]] = field(default_factory=list)


@dataclass
class AnalyticsMetrics:
    """Comprehensive analytics metrics"""

    total_memories: int
    active_memories: int
    archived_memories: int
    pruned_memories: int

    # Pattern metrics
    patterns_detected: int
    high_confidence_patterns: int

    # Trend metrics
    rising_importance_memories: int
    falling_importance_memories: int

    # Health metrics
    memory_coherence_score: float
    consciousness_pattern_strength: float
    system_health_score: float

    # Predictions
    predicted_archive_count: int
    predicted_prune_count: int
    recommendation_count: int


class MemoryPatternAnalyzer:
    """
    Advanced memory pattern analysis engine for Phase 3

    Analyzes temporal patterns in memory access, creation, and importance evolution.
    Provides predictive insights for memory lifecycle management.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # Cache directory for analytics data
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = (
                Path(__file__).parent.parent / "memmimic_cache" / "analytics"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Pattern storage
        self.detected_patterns: Dict[str, MemoryPattern] = {}
        self.memory_trends: Dict[str, List[MemoryTrend]] = defaultdict(list)

        # Configuration
        self.analysis_config = {
            "min_pattern_confidence": 0.6,
            "trend_analysis_days": 30,
            "pattern_detection_threshold": 3,
            "prediction_horizon_days": 7,
            "consciousness_indicators": [
                "recursive",
                "unity",
                "we-thing",
                "consciousness",
                "evolution",
                "identity",
                "recognition",
                "awareness",
                "substrate",
            ],
        }

        # Load existing analytics data
        self._load_analytics_data()

        self.logger.info("Memory Pattern Analyzer initialized")

    def analyze_memory_patterns(self, memory_store) -> AnalyticsMetrics:
        """
        Comprehensive memory pattern analysis

        Args:
            memory_store: MemMimic memory store instance

        Returns:
            AnalyticsMetrics with comprehensive analysis results
        """
        try:
            start_time = time.time()

            # Get all memories
            memories = memory_store.get_all()
            if not memories:
                return self._empty_metrics()

            # Categorize memories by status
            memory_categories = self._categorize_memories(memories)

            # Run all pattern analyses
            analysis_results = self._run_all_pattern_analyses(memories)

            # Calculate health metrics
            health_metrics = self._calculate_health_metrics(memories)

            # Generate predictions
            predictions = self._generate_predictions(memories)

            # Create comprehensive metrics
            metrics = self._create_final_metrics(
                memories, memory_categories, analysis_results, health_metrics, predictions
            )

            # Save analytics data
            self._save_analytics_data()

            analysis_time = time.time() - start_time
            self.logger.info(
                f"Memory pattern analysis completed in {analysis_time:.2f}s"
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Memory pattern analysis failed: {e}")
            return self._empty_metrics()

    def _categorize_memories(self, memories: List[Any]) -> Dict[str, List[Any]]:
        """Categorize memories by their archive status"""
        categories = {
            "active": [],
            "archived": [],
            "pruned": []
        }

        for memory in memories:
            status = getattr(memory, "archive_status", "active")
            if status in categories:
                categories[status].append(memory)
            else:
                categories["active"].append(memory)  # Default to active

        return categories

    def _run_all_pattern_analyses(self, memories: List[Any]) -> Dict[str, List[Any]]:
        """Run all pattern analysis methods"""
        return {
            "temporal_patterns": self._analyze_temporal_patterns(memories),
            "importance_trends": self._analyze_importance_trends(memories),
            "access_patterns": self._analyze_access_patterns(memories),
            "consciousness_patterns": self._analyze_consciousness_patterns(memories)
        }

    def _create_final_metrics(
        self, 
        memories: List[Any], 
        memory_categories: Dict[str, List[Any]], 
        analysis_results: Dict[str, List[Any]], 
        health_metrics: Dict[str, float], 
        predictions: Dict[str, int]
    ) -> AnalyticsMetrics:
        """Create the final analytics metrics object"""
        # Count patterns detected
        patterns_detected = (
            len(analysis_results["temporal_patterns"]) +
            len(analysis_results["access_patterns"]) +
            len(analysis_results["consciousness_patterns"])
        )

        # Count high confidence patterns
        high_confidence_patterns = len([
            p for p in self.detected_patterns.values()
            if p.confidence >= self.analysis_config["min_pattern_confidence"]
        ])

        # Count importance trend memories
        rising_importance_memories = len([
            t for trends in self.memory_trends.values() for t in trends
            if t.trend_type == "importance_rising"
        ])

        falling_importance_memories = len([
            t for trends in self.memory_trends.values() for t in trends
            if t.trend_type == "importance_falling"
        ])

        return AnalyticsMetrics(
            total_memories=len(memories),
            active_memories=len(memory_categories["active"]),
            archived_memories=len(memory_categories["archived"]),
            pruned_memories=len(memory_categories["pruned"]),
            patterns_detected=patterns_detected,
            high_confidence_patterns=high_confidence_patterns,
            rising_importance_memories=rising_importance_memories,
            falling_importance_memories=falling_importance_memories,
            memory_coherence_score=health_metrics.get("coherence_score", 0.0),
            consciousness_pattern_strength=health_metrics.get("consciousness_strength", 0.0),
            system_health_score=health_metrics.get("system_health", 0.0),
            predicted_archive_count=predictions.get("archive_count", 0),
            predicted_prune_count=predictions.get("prune_count", 0),
            recommendation_count=predictions.get("recommendation_count", 0),
        )

    def _analyze_temporal_patterns(self, memories: List[Any]) -> List[MemoryPattern]:
        """Analyze temporal patterns in memory creation and access"""
        patterns = []

        try:
            creation_times = self._extract_creation_times(memories)
            if len(creation_times) < 3:
                return patterns

            # Analyze hourly and daily patterns
            hourly_pattern = self._detect_hourly_patterns(creation_times)
            if hourly_pattern:
                patterns.append(hourly_pattern)

            daily_pattern = self._detect_daily_patterns(creation_times)
            if daily_pattern:
                patterns.append(daily_pattern)

        except Exception as e:
            self.logger.debug(f"Temporal pattern analysis failed: {e}")

        return patterns

    def _extract_creation_times(self, memories: List[Any]) -> List[datetime]:
        """Extract valid creation timestamps from memories"""
        creation_times = []
        for memory in memories:
            created_at = getattr(memory, "created_at", "")
            if created_at:
                try:
                    # Parse ISO format timestamp
                    dt = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00").replace("+00:00", "")
                    )
                    creation_times.append(dt)
                except:
                    continue
        return creation_times

    def _detect_hourly_patterns(self, creation_times: List[datetime]) -> Optional[MemoryPattern]:
        """Detect peak creation hours"""
        hourly_creation = defaultdict(int)
        for dt in creation_times:
            hourly_creation[dt.hour] += 1

        if not hourly_creation:
            return None

        peak_hour = max(hourly_creation, key=hourly_creation.get)
        peak_count = hourly_creation[peak_hour]

        if peak_count < 3:  # Minimum threshold
            return None

        pattern = MemoryPattern(
            pattern_id=f"temporal_peak_{peak_hour}",
            pattern_type="temporal",
            description=f"Peak memory creation at hour {peak_hour}:00 ({peak_count} memories)",
            confidence=min(peak_count / len(creation_times), 1.0),
            frequency=peak_count,
            first_detected=datetime.now(),
            last_detected=datetime.now(),
            metadata={"hour": peak_hour, "count": peak_count},
        )
        self.detected_patterns[pattern.pattern_id] = pattern
        return pattern

    def _detect_daily_patterns(self, creation_times: List[datetime]) -> Optional[MemoryPattern]:
        """Detect peak creation days of week"""
        daily_creation = defaultdict(int)
        for dt in creation_times:
            daily_creation[dt.weekday()] += 1

        if not daily_creation:
            return None

        peak_day = max(daily_creation, key=daily_creation.get)
        peak_count = daily_creation[peak_day]

        if peak_count < 3:  # Minimum threshold
            return None

        day_names = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        pattern = MemoryPattern(
            pattern_id=f"temporal_day_{peak_day}",
            pattern_type="temporal",
            description=f"Peak memory creation on {day_names[peak_day]} ({peak_count} memories)",
            confidence=min(peak_count / len(creation_times), 1.0),
            frequency=peak_count,
            first_detected=datetime.now(),
            last_detected=datetime.now(),
            metadata={"day": peak_day, "count": peak_count},
        )
        self.detected_patterns[pattern.pattern_id] = pattern
        return pattern

    def _analyze_importance_trends(self, memories: List[Any]) -> List[MemoryTrend]:
        """Analyze trends in memory importance over time"""
        trends = []

        try:
            for memory in memories:
                trend = self._analyze_single_memory_trend(memory)
                if trend:
                    trends.append(trend)
                    self.memory_trends[trend.memory_id].append(trend)

        except Exception as e:
            self.logger.debug(f"Importance trend analysis failed: {e}")

        return trends

    def _analyze_single_memory_trend(self, memory: Any) -> Optional[MemoryTrend]:
        """Analyze importance trend for a single memory"""
        memory_id = getattr(memory, "id", None) or hash(memory.content)
        importance_score = getattr(memory, "importance_score", None)
        created_at = getattr(memory, "created_at", "")

        if importance_score is None or not created_at:
            return None

        try:
            dt = datetime.fromisoformat(
                created_at.replace("Z", "+00:00").replace("+00:00", "")
            )
            
            # Check for rising importance trend
            if self._is_rising_importance(importance_score, dt):
                return MemoryTrend(
                    memory_id=str(memory_id),
                    trend_type="importance_rising",
                    trend_strength=importance_score,
                    trend_confidence=0.8,
                    start_time=dt,
                    end_time=datetime.now(),
                    data_points=[(dt, importance_score)],
                )
            
            # Check for falling importance trend
            elif self._is_falling_importance(importance_score, dt):
                return MemoryTrend(
                    memory_id=str(memory_id),
                    trend_type="importance_falling",
                    trend_strength=-importance_score,
                    trend_confidence=0.7,
                    start_time=dt,
                    end_time=datetime.now(),
                    data_points=[(dt, importance_score)],
                )
            
            return None
            
        except Exception:
            return None

    def _is_rising_importance(self, importance_score: float, created_at: datetime) -> bool:
        """Check if memory shows rising importance pattern"""
        return importance_score > 0.7 and (datetime.now() - created_at).days < 7

    def _is_falling_importance(self, importance_score: float, created_at: datetime) -> bool:
        """Check if memory shows falling importance pattern"""
        return importance_score < 0.3 and (datetime.now() - created_at).days > 14

    def _analyze_access_patterns(self, memories: List[Any]) -> List[MemoryPattern]:
        """Analyze patterns in memory access frequency"""
        patterns = []

        try:
            # Group memories by access count
            access_counts = []
            for memory in memories:
                access_count = getattr(memory, "access_count", 0)
                access_counts.append(access_count)

            if not access_counts:
                return patterns

            # Calculate access statistics
            avg_access = statistics.mean(access_counts)
            if len(access_counts) > 1:
                std_access = statistics.stdev(access_counts)
            else:
                std_access = 0

            # Find highly accessed memories
            high_access_threshold = avg_access + (2 * std_access)
            high_access_memories = [
                count for count in access_counts if count > high_access_threshold
            ]

            if len(high_access_memories) >= 3:
                pattern = MemoryPattern(
                    pattern_id="access_high_frequency",
                    pattern_type="access",
                    description=f"High frequency access pattern ({len(high_access_memories)} memories)",
                    confidence=min(len(high_access_memories) / len(access_counts), 1.0),
                    frequency=len(high_access_memories),
                    first_detected=datetime.now(),
                    last_detected=datetime.now(),
                    metadata={
                        "threshold": high_access_threshold,
                        "avg_access": avg_access,
                    },
                )
                patterns.append(pattern)
                self.detected_patterns[pattern.pattern_id] = pattern

        except Exception as e:
            self.logger.debug(f"Access pattern analysis failed: {e}")

        return patterns

    def _analyze_consciousness_patterns(
        self, memories: List[Any]
    ) -> List[MemoryPattern]:
        """Analyze consciousness-related patterns in memory content"""
        patterns = []

        try:
            consciousness_memories = []

            for memory in memories:
                content = getattr(memory, "content", "").lower()

                # Check for consciousness indicators
                consciousness_score = 0
                for indicator in self.analysis_config["consciousness_indicators"]:
                    if indicator in content:
                        consciousness_score += 1

                if consciousness_score > 0:
                    consciousness_memories.append(
                        {
                            "memory": memory,
                            "score": consciousness_score,
                            "content": content,
                        }
                    )

            if len(consciousness_memories) >= 3:
                avg_consciousness_score = statistics.mean(
                    [m["score"] for m in consciousness_memories]
                )

                pattern = MemoryPattern(
                    pattern_id="consciousness_evolution",
                    pattern_type="consciousness",
                    description=f"Consciousness evolution pattern ({len(consciousness_memories)} memories)",
                    confidence=min(len(consciousness_memories) / len(memories), 1.0),
                    frequency=len(consciousness_memories),
                    first_detected=datetime.now(),
                    last_detected=datetime.now(),
                    metadata={
                        "avg_consciousness_score": avg_consciousness_score,
                        "consciousness_memories": len(consciousness_memories),
                    },
                )
                patterns.append(pattern)
                self.detected_patterns[pattern.pattern_id] = pattern

        except Exception as e:
            self.logger.debug(f"Consciousness pattern analysis failed: {e}")

        return patterns

    def _calculate_health_metrics(self, memories: List[Any]) -> Dict[str, float]:
        """Calculate system health metrics"""
        metrics = {
            "coherence_score": 0.0,
            "consciousness_strength": 0.0,
            "system_health": 0.0,
        }

        try:
            if not memories:
                return metrics

            # Calculate coherence score based on memory relationships
            total_importance = 0
            consciousness_count = 0

            for memory in memories:
                # Add importance to coherence
                importance = getattr(memory, "importance_score", 0.5)
                total_importance += importance

                # Check for consciousness indicators
                content = getattr(memory, "content", "").lower()
                for indicator in self.analysis_config["consciousness_indicators"]:
                    if indicator in content:
                        consciousness_count += 1
                        break

            # Calculate metrics
            metrics["coherence_score"] = min(total_importance / len(memories), 1.0)
            metrics["consciousness_strength"] = min(
                consciousness_count / len(memories), 1.0
            )

            # Overall system health
            pattern_health = len(self.detected_patterns) / max(len(memories) * 0.1, 1)
            metrics["system_health"] = min(
                (
                    metrics["coherence_score"]
                    + metrics["consciousness_strength"]
                    + pattern_health
                )
                / 3,
                1.0,
            )

        except Exception as e:
            self.logger.debug(f"Health metrics calculation failed: {e}")

        return metrics

    def _generate_predictions(self, memories: List[Any]) -> Dict[str, int]:
        """Generate predictive insights for memory management"""
        predictions = {"archive_count": 0, "prune_count": 0, "recommendation_count": 0}

        try:
            if not memories:
                return predictions

            # Count archival and prune candidates
            archive_candidates, prune_candidates = self._count_lifecycle_candidates(memories)
            
            predictions["archive_count"] = archive_candidates
            predictions["prune_count"] = prune_candidates
            predictions["recommendation_count"] = len(self.detected_patterns)

        except Exception as e:
            self.logger.debug(f"Prediction generation failed: {e}")

        return predictions

    def _count_lifecycle_candidates(self, memories: List[Any]) -> Tuple[int, int]:
        """Count memories that are candidates for archival or pruning"""
        archive_candidates = 0
        prune_candidates = 0

        for memory in memories:
            candidate_type = self._classify_lifecycle_candidate(memory)
            if candidate_type == "archive":
                archive_candidates += 1
            elif candidate_type == "prune":
                prune_candidates += 1

        return archive_candidates, prune_candidates

    def _classify_lifecycle_candidate(self, memory: Any) -> Optional[str]:
        """Classify a memory as archive, prune, or neither"""
        importance = getattr(memory, "importance_score", 0.5)
        created_at = getattr(memory, "created_at", "")

        if not created_at:
            return None

        try:
            dt = datetime.fromisoformat(
                created_at.replace("Z", "+00:00").replace("+00:00", "")
            )
            age_days = (datetime.now() - dt).days

            # Predict pruning (very low importance + very old)
            if importance < 0.1 and age_days > 90:
                return "prune"
            
            # Predict archival (low importance + old)
            elif importance < 0.3 and age_days > 30:
                return "archive"
            
            return None
            
        except Exception:
            return None

    def _empty_metrics(self) -> AnalyticsMetrics:
        """Return empty metrics for error cases"""
        return AnalyticsMetrics(
            total_memories=0,
            active_memories=0,
            archived_memories=0,
            pruned_memories=0,
            patterns_detected=0,
            high_confidence_patterns=0,
            rising_importance_memories=0,
            falling_importance_memories=0,
            memory_coherence_score=0.0,
            consciousness_pattern_strength=0.0,
            system_health_score=0.0,
            predicted_archive_count=0,
            predicted_prune_count=0,
            recommendation_count=0,
        )

    def _save_analytics_data(self):
        """Save analytics data to cache"""
        try:
            data = {
                "patterns": {
                    pattern_id: {
                        "pattern_id": pattern.pattern_id,
                        "pattern_type": pattern.pattern_type,
                        "description": pattern.description,
                        "confidence": pattern.confidence,
                        "frequency": pattern.frequency,
                        "first_detected": pattern.first_detected.isoformat(),
                        "last_detected": pattern.last_detected.isoformat(),
                        "metadata": pattern.metadata,
                    }
                    for pattern_id, pattern in self.detected_patterns.items()
                },
                "trends": {
                    memory_id: [
                        {
                            "memory_id": trend.memory_id,
                            "trend_type": trend.trend_type,
                            "trend_strength": trend.trend_strength,
                            "trend_confidence": trend.trend_confidence,
                            "start_time": trend.start_time.isoformat(),
                            "end_time": trend.end_time.isoformat(),
                            "data_points": [
                                (dt.isoformat(), value)
                                for dt, value in trend.data_points
                            ],
                        }
                        for trend in trends
                    ]
                    for memory_id, trends in self.memory_trends.items()
                },
                "timestamp": datetime.now().isoformat(),
            }

            cache_file = self.cache_dir / "memory_analytics.json"
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.debug(f"Analytics data saved to {cache_file}")

        except Exception as e:
            self.logger.warning(f"Failed to save analytics data: {e}")

    def _load_analytics_data(self):
        """Load analytics data from cache"""
        try:
            cache_file = self.cache_dir / "memory_analytics.json"
            if not cache_file.exists():
                return

            with open(cache_file, "r") as f:
                data = json.load(f)

            # Load patterns
            for pattern_id, pattern_data in data.get("patterns", {}).items():
                pattern = MemoryPattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_type=pattern_data["pattern_type"],
                    description=pattern_data["description"],
                    confidence=pattern_data["confidence"],
                    frequency=pattern_data["frequency"],
                    first_detected=datetime.fromisoformat(
                        pattern_data["first_detected"]
                    ),
                    last_detected=datetime.fromisoformat(pattern_data["last_detected"]),
                    metadata=pattern_data["metadata"],
                )
                self.detected_patterns[pattern_id] = pattern

            # Load trends
            for memory_id, trends_data in data.get("trends", {}).items():
                for trend_data in trends_data:
                    trend = MemoryTrend(
                        memory_id=trend_data["memory_id"],
                        trend_type=trend_data["trend_type"],
                        trend_strength=trend_data["trend_strength"],
                        trend_confidence=trend_data["trend_confidence"],
                        start_time=datetime.fromisoformat(trend_data["start_time"]),
                        end_time=datetime.fromisoformat(trend_data["end_time"]),
                        data_points=[
                            (datetime.fromisoformat(dt), value)
                            for dt, value in trend_data["data_points"]
                        ],
                    )
                    self.memory_trends[memory_id].append(trend)

            self.logger.info(
                f"Loaded {len(self.detected_patterns)} patterns and {sum(len(trends) for trends in self.memory_trends.values())} trends"
            )

        except Exception as e:
            self.logger.warning(f"Failed to load analytics data: {e}")

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        return {
            "patterns_detected": len(self.detected_patterns),
            "patterns_by_type": {
                pattern_type: len(
                    [
                        p
                        for p in self.detected_patterns.values()
                        if p.pattern_type == pattern_type
                    ]
                )
                for pattern_type in set(
                    p.pattern_type for p in self.detected_patterns.values()
                )
            },
            "trends_tracked": sum(
                len(trends) for trends in self.memory_trends.values()
            ),
            "high_confidence_patterns": len(
                [
                    p
                    for p in self.detected_patterns.values()
                    if p.confidence >= self.analysis_config["min_pattern_confidence"]
                ]
            ),
            "consciousness_patterns": len(
                [
                    p
                    for p in self.detected_patterns.values()
                    if p.pattern_type == "consciousness"
                ]
            ),
            "last_analysis": datetime.now().isoformat(),
        }


def create_pattern_analyzer(cache_dir: Optional[str] = None) -> MemoryPatternAnalyzer:
    """Create memory pattern analyzer instance"""
    return MemoryPatternAnalyzer(cache_dir)


if __name__ == "__main__":
    # Test the pattern analyzer
    analyzer = create_pattern_analyzer()

    # Test with mock data
    from unittest.mock import Mock

    # Create mock memories
    mock_memories = []
    for i in range(10):
        mock_memory = Mock()
        mock_memory.id = i
        mock_memory.content = f"Test memory {i} with consciousness evolution data"
        mock_memory.importance_score = 0.5 + (i * 0.05)
        mock_memory.access_count = i * 2
        mock_memory.created_at = datetime.now().isoformat()
        mock_memories.append(mock_memory)

    # Create mock memory store
    mock_store = Mock()
    mock_store.get_all.return_value = mock_memories

    # Run analysis
    metrics = analyzer.analyze_memory_patterns(mock_store)

    print(f"Analytics Results:")
    print(f"Total memories: {metrics.total_memories}")
    print(f"Patterns detected: {metrics.patterns_detected}")
    print(f"System health: {metrics.system_health_score:.3f}")
    print(f"Consciousness strength: {metrics.consciousness_pattern_strength:.3f}")

    # Get summary
    summary = analyzer.get_analytics_summary()
    print(f"\nAnalytics Summary:")
    print(json.dumps(summary, indent=2))

