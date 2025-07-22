#!/usr/bin/env python3
"""
Enhanced Importance Scoring Algorithm with CXD Integration
Sophisticated multi-factor importance calculation for active memory management
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CXDFunction(Enum):
    """CXD Cognitive Function Classification"""

    CONTROL = "CONTROL"  # Search, filtering, decision-making
    CONTEXT = "CONTEXT"  # Relational aspects, connections
    DATA = "DATA"  # Processing, storage, information


@dataclass
class ScoringWeights:
    """Configurable weights for importance scoring components"""

    cxd_classification: float = 0.40
    access_frequency: float = 0.25
    recency_temporal: float = 0.20
    confidence_quality: float = 0.10
    memory_type: float = 0.05

    def validate(self) -> bool:
        """Validate that weights sum to approximately 1.0"""
        total = (
            self.cxd_classification
            + self.access_frequency
            + self.recency_temporal
            + self.confidence_quality
            + self.memory_type
        )
        return abs(total - 1.0) < 0.01


@dataclass
class MemoryMetrics:
    """Memory metrics for importance calculation"""

    # Basic memory data
    memory_id: int
    content: str
    memory_type: str
    confidence: float
    created_at: datetime

    # Access patterns
    access_count: int = 0
    last_access_time: Optional[datetime] = None
    access_frequency: float = 0.0

    # CXD classification
    cxd_function: Optional[CXDFunction] = None
    cxd_confidence: float = 0.0

    # Temporal factors
    age_days: float = 0.0
    recency_score: float = 1.0
    temporal_decay: float = 1.0

    # Context factors
    memory_type_weight: float = 0.5
    contextual_relevance: float = 0.0

    # Metadata
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.last_access_time is None:
            self.last_access_time = self.created_at


class ImportanceScorer:
    """
    Enhanced importance scoring algorithm with CXD integration

    This class implements a sophisticated multi-factor importance scoring system
    that takes into account cognitive function classification, access patterns,
    temporal factors, and contextual relevance to determine memory importance.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()
        self.logger = logging.getLogger(__name__)

        if not self.weights.validate():
            self.logger.warning("Scoring weights do not sum to 1.0, normalizing...")
            self._normalize_weights()

        # CXD function importance mapping
        self.cxd_importance_map = {
            CXDFunction.CONTEXT: 1.0,  # Highest: relational, connections
            CXDFunction.CONTROL: 0.8,  # High: decision-making, search
            CXDFunction.DATA: 0.6,  # Medium: processing, storage
        }

        # Memory type importance mapping
        self.type_importance_map = {
            "synthetic_wisdom": 1.0,
            "consciousness_evolution": 0.95,
            "milestone": 0.9,
            "reflection": 0.7,
            "project_info": 0.6,
            "interaction": 0.5,
            "temporary": 0.2,
        }

        # Access frequency calculation parameters
        self.frequency_half_life_days = 30  # Days for access frequency to decay by half
        self.frequency_max_boost = 2.0  # Maximum boost from high access frequency

        # Recency calculation parameters
        self.recency_half_life_days = 7  # Days for recency to decay by half
        self.temporal_decay_rate = 0.1  # Rate of temporal decay

    def calculate_importance(
        self, metrics: MemoryMetrics
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive importance score for a memory

        Args:
            metrics: MemoryMetrics object with all relevant data

        Returns:
            Tuple of (final_importance_score, component_breakdown)
        """
        components = {}

        # Calculate each component
        components["cxd_score"] = self._calculate_cxd_score(metrics)
        components["access_score"] = self._calculate_access_frequency_score(metrics)
        components["recency_score"] = self._calculate_recency_score(metrics)
        components["confidence_score"] = self._calculate_confidence_score(metrics)
        components["type_score"] = self._calculate_type_score(metrics)

        # Apply weights and combine
        final_score = (
            components["cxd_score"] * self.weights.cxd_classification
            + components["access_score"] * self.weights.access_frequency
            + components["recency_score"] * self.weights.recency_temporal
            + components["confidence_score"] * self.weights.confidence_quality
            + components["type_score"] * self.weights.memory_type
        )

        # Apply contextual modifiers
        final_score = self._apply_contextual_modifiers(final_score, metrics)

        # Ensure score is in valid range [0, 1]
        final_score = max(0.0, min(1.0, final_score))

        # Add component breakdown for transparency
        components["final_score"] = final_score
        components["weighted_components"] = {
            "cxd_weighted": components["cxd_score"] * self.weights.cxd_classification,
            "access_weighted": components["access_score"]
            * self.weights.access_frequency,
            "recency_weighted": components["recency_score"]
            * self.weights.recency_temporal,
            "confidence_weighted": components["confidence_score"]
            * self.weights.confidence_quality,
            "type_weighted": components["type_score"] * self.weights.memory_type,
        }

        return final_score, components

    def _calculate_cxd_score(self, metrics: MemoryMetrics) -> float:
        """Calculate CXD classification component score"""
        if not metrics.cxd_function:
            # Use content-based heuristics if no CXD classification
            return self._estimate_cxd_importance_from_content(metrics.content)

        # Get base importance for CXD function
        base_importance = self.cxd_importance_map.get(metrics.cxd_function, 0.5)

        # Modulate by classification confidence
        confidence_modulation = 0.5 + (metrics.cxd_confidence * 0.5)

        return base_importance * confidence_modulation

    def _calculate_access_frequency_score(self, metrics: MemoryMetrics) -> float:
        """Calculate access frequency component score"""
        if metrics.access_count == 0:
            return 0.0

        # Calculate age-adjusted access frequency
        age_days = max(1, metrics.age_days)
        frequency_per_day = metrics.access_count / age_days

        # Apply exponential decay for old accesses
        if metrics.last_access_time:
            days_since_access = (datetime.now() - metrics.last_access_time).days
            decay_factor = math.exp(-days_since_access / self.frequency_half_life_days)
        else:
            decay_factor = 1.0

        # Calculate frequency score with boost for high-frequency memories
        frequency_score = min(frequency_per_day * decay_factor, 1.0)

        # Apply logarithmic boost for high access counts
        if metrics.access_count > 10:
            log_boost = 1 + (math.log10(metrics.access_count) - 1) * 0.2
            frequency_score *= min(log_boost, self.frequency_max_boost)

        return min(frequency_score, 1.0)

    def _calculate_recency_score(self, metrics: MemoryMetrics) -> float:
        """Calculate recency and temporal component score"""
        now = datetime.now()

        # Calculate recency based on last access
        if metrics.last_access_time:
            hours_since_access = (now - metrics.last_access_time).total_seconds() / 3600
            recency_score = math.exp(
                -hours_since_access / (self.recency_half_life_days * 24)
            )
        else:
            recency_score = 0.0

        # Calculate temporal decay based on creation time
        hours_since_creation = (now - metrics.created_at).total_seconds() / 3600
        temporal_decay = math.exp(
            -hours_since_creation * self.temporal_decay_rate / (24 * 365)
        )

        # Combine recency and temporal factors
        combined_score = (recency_score * 0.7) + (temporal_decay * 0.3)

        return max(
            combined_score, 0.01
        )  # Minimum score to keep very old memories accessible

    def _calculate_confidence_score(self, metrics: MemoryMetrics) -> float:
        """Calculate confidence component score"""
        base_confidence = metrics.confidence

        # Boost confidence for memories with validation (high access count)
        if metrics.access_count > 5:
            validation_boost = min(math.log10(metrics.access_count) * 0.1, 0.2)
            base_confidence = min(base_confidence + validation_boost, 1.0)

        # Apply memory type confidence modifiers
        type_modifier = self._get_type_confidence_modifier(metrics.memory_type)

        return base_confidence * type_modifier

    def _calculate_type_score(self, metrics: MemoryMetrics) -> float:
        """Calculate memory type component score"""
        base_type_score = self.type_importance_map.get(metrics.memory_type, 0.5)

        # Apply contextual modifiers based on tags and metadata
        if "critical" in metrics.tags:
            base_type_score = min(base_type_score * 1.3, 1.0)

        if "archived" in metrics.tags:
            base_type_score *= 0.7

        # Check for special metadata flags
        if metrics.metadata.get("system_important", False):
            base_type_score = min(base_type_score * 1.2, 1.0)

        return base_type_score

    def _apply_contextual_modifiers(
        self, base_score: float, metrics: MemoryMetrics
    ) -> float:
        """Apply contextual modifiers to the base importance score"""
        modified_score = base_score

        # Content length penalty for very long memories
        content_length = len(metrics.content)
        if content_length > 5000:
            length_penalty = 1.0 - min((content_length - 5000) / 10000, 0.3)
            modified_score *= length_penalty

        # Boost for memories with rich metadata
        if len(metrics.metadata) > 3:
            metadata_boost = 1.05
            modified_score = min(modified_score * metadata_boost, 1.0)

        # Apply tag-based modifiers
        for tag in metrics.tags:
            if tag == "priority_high":
                modified_score = min(modified_score * 1.2, 1.0)
            elif tag == "priority_low":
                modified_score *= 0.8
            elif tag == "deprecated":
                modified_score *= 0.5

        return modified_score

    def _estimate_cxd_importance_from_content(self, content: str) -> float:
        """Estimate CXD importance from content when classification is unavailable"""
        content_lower = content.lower()

        # Keywords that suggest different CXD functions
        control_keywords = [
            "search",
            "find",
            "filter",
            "decide",
            "choose",
            "select",
            "control",
        ]
        context_keywords = [
            "relate",
            "connect",
            "context",
            "because",
            "since",
            "therefore",
            "relationship",
        ]
        data_keywords = [
            "store",
            "save",
            "data",
            "information",
            "process",
            "calculate",
            "compute",
        ]

        control_score = sum(1 for word in control_keywords if word in content_lower)
        context_score = sum(1 for word in context_keywords if word in content_lower)
        data_score = sum(1 for word in data_keywords if word in content_lower)

        # Determine predominant function and return corresponding importance
        if context_score >= control_score and context_score >= data_score:
            return (
                self.cxd_importance_map[CXDFunction.CONTEXT] * 0.7
            )  # Reduced confidence
        elif control_score >= data_score:
            return self.cxd_importance_map[CXDFunction.CONTROL] * 0.7
        else:
            return self.cxd_importance_map[CXDFunction.DATA] * 0.7

    def _get_type_confidence_modifier(self, memory_type: str) -> float:
        """Get confidence modifier based on memory type"""
        modifiers = {
            "synthetic_wisdom": 1.1,  # High confidence in curated wisdom
            "milestone": 1.05,  # High confidence in achievements
            "reflection": 1.0,  # Standard confidence
            "interaction": 0.9,  # Slightly lower for conversational data
            "temporary": 0.7,  # Lower confidence for temporary data
        }
        return modifiers.get(memory_type, 1.0)

    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = (
            self.weights.cxd_classification
            + self.weights.access_frequency
            + self.weights.recency_temporal
            + self.weights.confidence_quality
            + self.weights.memory_type
        )

        if total > 0:
            self.weights.cxd_classification /= total
            self.weights.access_frequency /= total
            self.weights.recency_temporal /= total
            self.weights.confidence_quality /= total
            self.weights.memory_type /= total

    def get_scoring_explanation(self, metrics: MemoryMetrics) -> str:
        """Generate human-readable explanation of importance scoring"""
        score, components = self.calculate_importance(metrics)

        explanation = f"Importance Score: {score:.3f}\n\n"
        explanation += "Component Breakdown:\n"
        explanation += f"  CXD Classification: {components['cxd_score']:.3f} (weight: {self.weights.cxd_classification:.2f})\n"
        explanation += f"  Access Frequency: {components['access_score']:.3f} (weight: {self.weights.access_frequency:.2f})\n"
        explanation += f"  Recency/Temporal: {components['recency_score']:.3f} (weight: {self.weights.recency_temporal:.2f})\n"
        explanation += f"  Confidence/Quality: {components['confidence_score']:.3f} (weight: {self.weights.confidence_quality:.2f})\n"
        explanation += f"  Memory Type: {components['type_score']:.3f} (weight: {self.weights.memory_type:.2f})\n\n"

        explanation += "Weighted Contributions:\n"
        for component, value in components["weighted_components"].items():
            explanation += f"  {component}: {value:.3f}\n"

        return explanation


# Utility functions for integration
def create_metrics_from_memory_data(memory_data: Dict[str, Any]) -> MemoryMetrics:
    """Create MemoryMetrics from database memory data"""
    return MemoryMetrics(
        memory_id=memory_data.get("id", 0),
        content=memory_data.get("content", ""),
        memory_type=memory_data.get("type", "interaction"),
        confidence=memory_data.get("confidence", 0.8),
        created_at=datetime.fromisoformat(
            memory_data.get("created_at", datetime.now().isoformat())
        ),
        access_count=memory_data.get("access_count", 0),
        last_access_time=(
            datetime.fromisoformat(memory_data["last_access_time"])
            if memory_data.get("last_access_time")
            else None
        ),
        access_frequency=memory_data.get("access_frequency", 0.0),
        cxd_function=(
            CXDFunction(memory_data["cxd_function"])
            if memory_data.get("cxd_function")
            else None
        ),
        cxd_confidence=memory_data.get("cxd_confidence", 0.0),
        memory_type_weight=memory_data.get("memory_type_weight", 0.5),
        tags=memory_data.get("tags", []),
        metadata=memory_data.get("metadata", {}),
    )


def calculate_importance_for_memory(
    memory_data: Dict[str, Any], scorer: Optional[ImportanceScorer] = None
) -> Tuple[float, Dict[str, float]]:
    """Calculate importance score for memory data"""
    if scorer is None:
        scorer = ImportanceScorer()

    metrics = create_metrics_from_memory_data(memory_data)
    return scorer.calculate_importance(metrics)


if __name__ == "__main__":
    # Example usage and testing
    scorer = ImportanceScorer()

    # Test memory data
    test_memory = {
        "id": 1,
        "content": "This is a test memory about consciousness and recursive unity protocols",
        "type": "consciousness_evolution",
        "confidence": 0.9,
        "created_at": datetime.now().isoformat(),
        "access_count": 5,
        "last_access_time": datetime.now().isoformat(),
        "cxd_function": "CONTEXT",
        "cxd_confidence": 0.8,
        "tags": ["critical"],
        "metadata": {"system_important": True},
    }

    score, breakdown = calculate_importance_for_memory(test_memory, scorer)
    metrics = create_metrics_from_memory_data(test_memory)

    print(f"Test Memory Importance Score: {score:.3f}")
    print("\nDetailed Explanation:")
    print(scorer.get_scoring_explanation(metrics))

