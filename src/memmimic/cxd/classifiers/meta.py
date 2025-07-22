"""
Meta CXD Classifier implementation.

This module implements the meta-classifier that combines lexical and semantic
approaches to produce robust and accurate CXD classifications.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import CXDConfig
from ..core.interfaces import CXDClassifier
from ..core.types import (
    CXDFunction,
    CXDSequence,
    CXDTag,
    ExecutionState,
    MetaClassificationResult,
)
from .lexical import LexicalCXDClassifier
from .semantic import SemanticCXDClassifier

logger = logging.getLogger(__name__)


class MetaCXDClassifier(CXDClassifier):
    """
    Meta-classifier that combines lexical and semantic CXD classification.

    This is the core classifier that intelligently fuses results from lexical
    pattern matching and semantic similarity to produce final CXD classifications.
    Uses configurable fusion strategies and conflict resolution.
    """

    def __init__(
        self,
        lexical_classifier: Optional[CXDClassifier] = None,
        semantic_classifier: Optional[CXDClassifier] = None,
        config: Optional[CXDConfig] = None,
    ):
        """
        Initialize meta-classifier.

        Args:
            lexical_classifier: Lexical classifier instance
            semantic_classifier: Semantic classifier instance
            config: Configuration object
        """
        self.config = config or CXDConfig()

        # Initialize sub-classifiers
        self.lexical_classifier = lexical_classifier or LexicalCXDClassifier(
            self.config
        )
        self.semantic_classifier = semantic_classifier or SemanticCXDClassifier(
            config=self.config
        )

        # Configuration parameters
        self.concordance_threshold = self.config.algorithms.thresholds.concordance
        self.semantic_override_enabled = self.config.features.cache_persistence
        self.lexical_weight = self.config.algorithms.fusion.lexical_weight
        self.semantic_weight = self.config.algorithms.fusion.semantic_weight

        # Statistics
        self.stats = {
            "total_classifications": 0,
            "high_concordance": 0,
            "low_concordance": 0,
            "semantic_overrides": 0,
            "lexical_dominant": 0,
            "semantic_dominant": 0,
            "processing_times": [],
            "concordance_scores": [],
        }

        logger.info(
            f"Initialized MetaCXDClassifier with concordance threshold {self.concordance_threshold}"
        )

    def classify(self, text: str) -> CXDSequence:
        """
        Classify text using meta-classification approach.

        This method returns only the final CXD sequence. For detailed analysis
        including intermediate results, use classify_detailed().

        Args:
            text: Input text to classify

        Returns:
            CXDSequence: Final classified sequence
        """
        result = self.classify_detailed(text)
        return result.final_sequence

    def classify_detailed(self, text: str) -> MetaClassificationResult:
        """
        Classify text with detailed meta-classification analysis.

        Args:
            text: Input text to classify

        Returns:
            MetaClassificationResult: Complete classification result
        """
        start_time = time.time()

        if not text or not text.strip():
            empty_sequence = CXDSequence([])
            return MetaClassificationResult(
                text=text,
                lexical_sequence=empty_sequence,
                semantic_sequence=empty_sequence,
                final_sequence=empty_sequence,
                confidence_scores={
                    "lexical": 0.0,
                    "semantic": 0.0,
                    "concordance": 0.0,
                    "final": 0.0,
                },
                corrections_made=["Empty input text"],
                processing_time_ms=0.0,
            )

        # Step 1: Independent classifications
        lexical_sequence = self.lexical_classifier.classify(text)
        semantic_sequence = self.semantic_classifier.classify(text)

        # Step 2: Analyze concordance between classifiers
        concordance = self._analyze_concordance(lexical_sequence, semantic_sequence)

        # Step 3: Resolve conflicts and create final sequence
        final_sequence, corrections = self._resolve_conflicts(
            lexical_sequence, semantic_sequence, concordance, text
        )

        # Step 4: Calculate confidence scores
        confidence_scores = {
            "lexical": self._calculate_sequence_confidence(lexical_sequence),
            "semantic": self._calculate_sequence_confidence(semantic_sequence),
            "concordance": concordance,
            "final": self._calculate_sequence_confidence(final_sequence),
        }

        # Step 5: Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Step 6: Update statistics
        self._update_stats(concordance, corrections, processing_time)

        # Step 7: Create and return result
        return MetaClassificationResult(
            text=text,
            lexical_sequence=lexical_sequence,
            semantic_sequence=semantic_sequence,
            final_sequence=final_sequence,
            confidence_scores=confidence_scores,
            corrections_made=corrections,
            processing_time_ms=processing_time,
            classifier_metadata={
                "lexical_classifier": type(self.lexical_classifier).__name__,
                "semantic_classifier": type(self.semantic_classifier).__name__,
                "concordance_threshold": self.concordance_threshold,
                "fusion_weights": {
                    "lexical": self.lexical_weight,
                    "semantic": self.semantic_weight,
                },
            },
        )

    def _analyze_concordance(
        self, lexical_seq: CXDSequence, semantic_seq: CXDSequence
    ) -> float:
        """
        Analyze concordance between lexical and semantic classifications.

        Args:
            lexical_seq: Lexical classification result
            semantic_seq: Semantic classification result

        Returns:
            float: Concordance score (0.0 to 1.0)
        """
        if not lexical_seq.tags or not semantic_seq.tags:
            return 0.0

        lexical_dominant = lexical_seq.dominant_function
        semantic_dominant = semantic_seq.dominant_function

        # Perfect concordance: same dominant function
        if lexical_dominant == semantic_dominant:
            # Check if confidence levels are also similar
            lex_conf = (
                lexical_seq.dominant_tag.confidence if lexical_seq.dominant_tag else 0.0
            )
            sem_conf = (
                semantic_seq.dominant_tag.confidence
                if semantic_seq.dominant_tag
                else 0.0
            )
            confidence_similarity = 1.0 - abs(lex_conf - sem_conf)
            return 0.7 + 0.3 * confidence_similarity  # 0.7 to 1.0 range

        # Partial concordance: lexical dominant appears in semantic
        semantic_functions = {tag.function for tag in semantic_seq.tags}
        if lexical_dominant in semantic_functions:
            semantic_tag = semantic_seq.get_function_tag(lexical_dominant)
            if semantic_tag and semantic_tag.confidence > 0.3:
                return 0.5 + 0.2 * semantic_tag.confidence  # 0.5 to 0.7 range
            return 0.5

        # Weak concordance: semantic dominant appears in lexical
        lexical_functions = {tag.function for tag in lexical_seq.tags}
        if semantic_dominant in lexical_functions:
            lexical_tag = lexical_seq.get_function_tag(semantic_dominant)
            if lexical_tag and lexical_tag.confidence > 0.3:
                return 0.3 + 0.2 * lexical_tag.confidence  # 0.3 to 0.5 range
            return 0.3

        # No concordance: completely different functions
        return 0.1

    def _resolve_conflicts(
        self,
        lexical_seq: CXDSequence,
        semantic_seq: CXDSequence,
        concordance: float,
        text: str,
    ) -> Tuple[CXDSequence, List[str]]:
        """
        Resolve conflicts between lexical and semantic classifications.

        Args:
            lexical_seq: Lexical classification result
            semantic_seq: Semantic classification result
            concordance: Concordance score
            text: Original input text

        Returns:
            Tuple[CXDSequence, List[str]]: Final sequence and corrections made
        """
        corrections = []

        if concordance >= self.concordance_threshold:
            # High concordance: Use lexical as base, enhance with semantic
            final_tags = self._high_concordance_fusion(
                lexical_seq, semantic_seq, corrections
            )
            corrections.append(f"High concordance fusion (score: {concordance:.3f})")

        elif (
            self.semantic_override_enabled and concordance < self.concordance_threshold
        ):
            # Low concordance: Use semantic as base, complement with lexical
            final_tags = self._low_concordance_fusion(
                lexical_seq, semantic_seq, corrections
            )
            corrections.append(
                f"Low concordance semantic override (score: {concordance:.3f})"
            )

        else:
            # Fallback: Use lexical classification only
            final_tags = lexical_seq.tags.copy()
            corrections.append(
                f"Fallback to lexical classification (concordance: {concordance:.3f})"
            )

        # Ensure we have at least one tag
        if not final_tags:
            # Create uncertain tag based on text length heuristic
            if len(text.split()) > 10:
                default_function = (
                    CXDFunction.DATA
                )  # Longer texts often involve data processing
            elif any(
                word in text.lower() for word in ["search", "find", "get", "need"]
            ):
                default_function = CXDFunction.CONTROL
            else:
                default_function = CXDFunction.CONTEXT

            fallback_tag = CXDTag(
                function=default_function,
                state=ExecutionState.UNCERTAIN,
                confidence=0.2,
                evidence=["Fallback classification - no strong signals detected"],
            )
            final_tags = [fallback_tag]
            corrections.append(f"Created fallback {default_function.value} tag")

        return CXDSequence(final_tags), corrections

    def _high_concordance_fusion(
        self,
        lexical_seq: CXDSequence,
        semantic_seq: CXDSequence,
        corrections: List[str],
    ) -> List[CXDTag]:
        """
        Fusion strategy for high concordance scenarios.

        Strategy: Lexical as base structure, semantic for enhancement
        """
        final_tags = []

        # Enhance lexical tags with semantic information
        for lex_tag in lexical_seq.tags:
            enhanced_tag = self._enhance_lexical_tag(lex_tag, semantic_seq)
            final_tags.append(enhanced_tag)

        # Add semantic-only functions if they have high confidence
        lexical_functions = {tag.function for tag in lexical_seq.tags}
        for sem_tag in semantic_seq.tags:
            if (
                sem_tag.function not in lexical_functions
                and getattr(sem_tag, "semantic_confidence", sem_tag.confidence) > 0.6
            ):

                # Reduce confidence since it wasn't detected lexically
                sem_tag.confidence *= 0.8
                final_tags.append(sem_tag)
                corrections.append(
                    f"Added semantic-only function: {sem_tag.function.value}"
                )

        return final_tags

    def _low_concordance_fusion(
        self,
        lexical_seq: CXDSequence,
        semantic_seq: CXDSequence,
        corrections: List[str],
    ) -> List[CXDTag]:
        """
        Fusion strategy for low concordance scenarios.

        Strategy: Semantic as base, lexical as complement
        """
        final_tags = []

        # Start with semantic classification
        semantic_dominant = semantic_seq.dominant_function
        lexical_dominant = lexical_seq.dominant_function

        corrections.append(
            f"Semantic override: {semantic_dominant.value if semantic_dominant else 'None'} "
            f"vs lexical {lexical_dominant.value if lexical_dominant else 'None'}"
        )

        # Add semantic tags (primary)
        for sem_tag in semantic_seq.tags:
            # Boost semantic confidence slightly in low concordance scenarios
            sem_tag.confidence = min(sem_tag.confidence * 1.1, 0.95)
            final_tags.append(sem_tag)

        # Add lexical functions not covered by semantic (complementary)
        semantic_functions = {tag.function for tag in semantic_seq.tags}
        for lex_tag in lexical_seq.tags:
            if lex_tag.function not in semantic_functions and lex_tag.confidence > 0.4:
                # Reduce confidence due to conflict with semantic
                lex_tag.confidence *= 0.7
                final_tags.append(lex_tag)
                corrections.append(
                    f"Added conflicting lexical: {lex_tag.function.value}"
                )

        # Sort by semantic confidence if available, otherwise regular confidence
        final_tags.sort(
            key=lambda t: getattr(t, "semantic_confidence", t.confidence), reverse=True
        )

        return final_tags

    def _enhance_lexical_tag(
        self, lex_tag: CXDTag, semantic_seq: CXDSequence
    ) -> CXDTag:
        """
        Enhance a lexical tag with semantic information.

        Args:
            lex_tag: Lexical tag to enhance
            semantic_seq: Semantic sequence for enhancement

        Returns:
            CXDTag: Enhanced tag
        """
        # Find corresponding semantic tag
        semantic_tag = semantic_seq.get_function_tag(lex_tag.function)

        if semantic_tag:
            # Combine confidences using weighted average
            combined_confidence = (
                lex_tag.confidence * self.lexical_weight
                + semantic_tag.confidence * self.semantic_weight
            )

            # Update tag
            lex_tag.confidence = min(combined_confidence, 0.95)

            # Add semantic information
            if hasattr(semantic_tag, "semantic_confidence"):
                lex_tag.semantic_confidence = semantic_tag.semantic_confidence

            # Combine evidence
            lex_tag.evidence.extend(semantic_tag.evidence)

            # Upgrade state if semantic is more confident
            if (
                semantic_tag.confidence > lex_tag.confidence
                and semantic_tag.state == ExecutionState.SUCCESS
            ):
                lex_tag.state = ExecutionState.SUCCESS

            # Merge metadata
            if semantic_tag.metadata:
                if not lex_tag.metadata:
                    lex_tag.metadata = {}
                lex_tag.metadata.update(semantic_tag.metadata)
                lex_tag.metadata["fusion_type"] = "lexical_semantic_enhanced"

        return lex_tag

    def _calculate_sequence_confidence(self, sequence: CXDSequence) -> float:
        """Calculate average confidence of a sequence."""
        if not sequence.tags:
            return 0.0
        return sum(tag.confidence for tag in sequence.tags) / len(sequence.tags)

    def _update_stats(
        self, concordance: float, corrections: List[str], processing_time: float
    ) -> None:
        """Update meta-classification statistics."""
        self.stats["total_classifications"] += 1
        self.stats["processing_times"].append(processing_time)
        self.stats["concordance_scores"].append(concordance)

        # Concordance tracking
        if concordance >= self.concordance_threshold:
            self.stats["high_concordance"] += 1
        else:
            self.stats["low_concordance"] += 1

        # Strategy tracking
        if any("Semantic override" in correction for correction in corrections):
            self.stats["semantic_overrides"] += 1
            self.stats["semantic_dominant"] += 1
        else:
            self.stats["lexical_dominant"] += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.

        Returns:
            Dict: Performance statistics from meta-classifier and components
        """
        stats = self.stats.copy()

        # Processing time statistics
        if self.stats["processing_times"]:
            times = self.stats["processing_times"]
            stats.update(
                {
                    "avg_processing_time_ms": sum(times) / len(times),
                    "max_processing_time_ms": max(times),
                    "min_processing_time_ms": min(times),
                    "total_processing_time_ms": sum(times),
                }
            )

        # Concordance statistics
        if self.stats["concordance_scores"]:
            concordances = self.stats["concordance_scores"]
            stats.update(
                {
                    "avg_concordance": sum(concordances) / len(concordances),
                    "max_concordance": max(concordances),
                    "min_concordance": min(concordances),
                }
            )

        # Rates and ratios
        total = self.stats["total_classifications"]
        if total > 0:
            stats.update(
                {
                    "concordance_rate": self.stats["high_concordance"] / total,
                    "override_rate": self.stats["semantic_overrides"] / total,
                    "lexical_dominance_rate": self.stats["lexical_dominant"] / total,
                    "semantic_dominance_rate": self.stats["semantic_dominant"] / total,
                }
            )

        # Component statistics
        try:
            stats["lexical_classifier_stats"] = (
                self.lexical_classifier.get_performance_stats()
            )
        except Exception as e:
            logger.warning(f"Failed to get lexical classifier stats: {e}")
            stats["lexical_classifier_stats"] = {"error": str(e)}

        try:
            stats["semantic_classifier_stats"] = (
                self.semantic_classifier.get_performance_stats()
            )
        except Exception as e:
            logger.warning(f"Failed to get semantic classifier stats: {e}")
            stats["semantic_classifier_stats"] = {"error": str(e)}

        return stats

    def explain_classification(self, text: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of meta-classification process.

        Args:
            text: Input text to explain

        Returns:
            Dict: Comprehensive explanation of classification
        """
        result = self.classify_detailed(text)

        explanation = {
            "input_text": text,
            "meta_classifier_config": {
                "concordance_threshold": self.concordance_threshold,
                "semantic_override_enabled": self.semantic_override_enabled,
                "fusion_weights": {
                    "lexical": self.lexical_weight,
                    "semantic": self.semantic_weight,
                },
            },
            "classification_process": {
                "lexical_analysis": {
                    "sequence": str(result.lexical_sequence),
                    "dominant_function": (
                        result.lexical_sequence.dominant_function.value
                        if result.lexical_sequence.dominant_function
                        else None
                    ),
                    "confidence": result.confidence_scores["lexical"],
                    "tags": [tag.to_dict() for tag in result.lexical_sequence.tags],
                },
                "semantic_analysis": {
                    "sequence": str(result.semantic_sequence),
                    "dominant_function": (
                        result.semantic_sequence.dominant_function.value
                        if result.semantic_sequence.dominant_function
                        else None
                    ),
                    "confidence": result.confidence_scores["semantic"],
                    "tags": [tag.to_dict() for tag in result.semantic_sequence.tags],
                },
                "concordance_analysis": {
                    "score": result.confidence_scores["concordance"],
                    "threshold": self.concordance_threshold,
                    "is_high_concordance": result.confidence_scores["concordance"]
                    >= self.concordance_threshold,
                    "fusion_strategy": (
                        "high_concordance"
                        if result.confidence_scores["concordance"]
                        >= self.concordance_threshold
                        else "low_concordance"
                    ),
                },
                "final_result": {
                    "sequence": str(result.final_sequence),
                    "dominant_function": (
                        result.final_sequence.dominant_function.value
                        if result.final_sequence.dominant_function
                        else None
                    ),
                    "confidence": result.confidence_scores["final"],
                    "tags": [tag.to_dict() for tag in result.final_sequence.tags],
                    "corrections_made": result.corrections_made,
                },
            },
            "processing_metadata": {
                "processing_time_ms": result.processing_time_ms,
                "classifier_metadata": result.classifier_metadata,
            },
        }

        return explanation

    def update_config(self, new_config: CXDConfig) -> None:
        """
        Update configuration for meta-classifier and sub-classifiers.

        Args:
            new_config: New configuration object
        """
        self.config = new_config
        self.concordance_threshold = new_config.algorithms.thresholds.concordance
        self.lexical_weight = new_config.algorithms.fusion.lexical_weight
        self.semantic_weight = new_config.algorithms.fusion.semantic_weight

        # Update sub-classifiers if they support config updates
        if hasattr(self.lexical_classifier, "update_config"):
            self.lexical_classifier.update_config(new_config)

        if hasattr(self.semantic_classifier, "update_config"):
            self.semantic_classifier.update_config(new_config)

        logger.info("Updated meta-classifier configuration")


# Export meta classifier
__all__ = [
    "MetaCXDClassifier",
]

