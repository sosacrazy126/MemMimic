"""
Optimized Meta CXD Classifier implementation.

This module implements the optimized meta-classifier that uses the enhanced
semantic classifier while inheriting all fusion logic from the base meta-classifier.
This eliminates code duplication and provides the best of both worlds.
"""

import logging
from typing import Any, Dict, Optional

from ..core.config import CXDConfig
from .lexical import LexicalCXDClassifier
from .meta import MetaCXDClassifier
from .optimized_semantic import OptimizedSemanticCXDClassifier

logger = logging.getLogger(__name__)


class OptimizedMetaCXDClassifier(MetaCXDClassifier):
    """
    Optimized meta-classifier that combines lexical and optimized semantic classification.

    This class inherits ALL fusion logic from MetaCXDClassifier but uses the
    OptimizedSemanticCXDClassifier for enhanced performance. This eliminates
    the 300+ lines of duplicated fusion code that existed in the original system.

    Key optimizations inherited:
    - FAISS-based vector indexing for sub-millisecond search
    - Persistent caching with checksum-based invalidation
    - Advanced embedding models with caching
    - Enhanced performance monitoring
    - Configurable optimization strategies

    All fusion strategies, conflict resolution, and meta-classification logic
    are inherited without duplication from the parent MetaCXDClassifier.
    """

    def __init__(self, config: Optional[CXDConfig] = None, **kwargs):
        """
        Initialize optimized meta-classifier.

        Args:
            config: Configuration object (creates default if None)
            **kwargs: Additional arguments for OptimizedSemanticCXDClassifier
                - enable_cache_persistence: Enable persistent caching (default: True)
                - rebuild_cache: Force cache rebuild (default: False)
                - Any other parameters will be passed to OptimizedSemanticCXDClassifier
        """
        # Create default config if not provided
        if config is None:
            config = CXDConfig()

        # Extract optimization-specific arguments
        enable_cache_persistence = kwargs.pop("enable_cache_persistence", True)
        rebuild_cache = kwargs.pop("rebuild_cache", False)

        # Create lexical classifier with config
        lexical_classifier = LexicalCXDClassifier(config=config)

        # Create optimized semantic classifier with enhanced features
        # OptimizedSemanticCXDClassifier handles creation of its own EmbeddingModel,
        # CanonicalExampleProvider and VectorStore internally using factory functions
        semantic_classifier = OptimizedSemanticCXDClassifier(
            config=config,
            enable_cache_persistence=enable_cache_persistence,
            rebuild_cache=rebuild_cache,
            **kwargs,  # Pass any remaining kwargs to optimized semantic classifier
        )

        # Initialize parent with optimized components
        # ALL fusion logic (classify_detailed, _analyze_concordance, _resolve_conflicts,
        # _high_concordance_fusion, _low_concordance_fusion, _enhance_lexical_tag,
        # _calculate_sequence_confidence, _update_stats) is inherited from MetaCXDClassifier
        super().__init__(
            lexical_classifier=lexical_classifier,
            semantic_classifier=semantic_classifier,
            config=config,
        )

        logger.info(
            "Initialized OptimizedMetaCXDClassifier with enhanced semantic processing"
        )

    # =========================================================================
    # OPTIMIZATION-SPECIFIC METHODS (delegate to semantic classifier)
    # =========================================================================

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics from the semantic classifier.

        Returns:
            Dict: Optimization statistics including cache performance, if available
        """
        if hasattr(self.semantic_classifier, "get_optimization_stats"):
            return self.semantic_classifier.get_optimization_stats()
        return {}

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get detailed cache information from the semantic classifier.

        Returns:
            Dict: Cache information and status, if available
        """
        if hasattr(self.semantic_classifier, "get_cache_info"):
            return self.semantic_classifier.get_cache_info()
        return {"cache_enabled": False}

    def rebuild_semantic_index(self) -> None:
        """Rebuild the semantic index with optimizations."""
        if hasattr(self.semantic_classifier, "rebuild_index"):
            self.semantic_classifier.rebuild_index()
            logger.info("Rebuilt optimized semantic index")
        else:
            logger.warning("Semantic classifier does not support index rebuilding")

    def clear_all_caches(self) -> None:
        """Clear all caches in the optimized system."""
        if hasattr(self.semantic_classifier, "clear_all_caches"):
            self.semantic_classifier.clear_all_caches()
            logger.info("Cleared all optimization caches")
        else:
            logger.warning("Semantic classifier does not support cache clearing")

    def validate_cache_integrity(self) -> Dict[str, Any]:
        """
        Validate cache integrity across the system.

        Returns:
            Dict: Cache validation results, if available
        """
        if hasattr(self.semantic_classifier, "validate_cache_integrity"):
            return self.semantic_classifier.validate_cache_integrity()
        return {"status": "not_supported"}

    # =========================================================================
    # OVERRIDDEN METHODS (extend parent functionality)
    # =========================================================================

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get enhanced performance statistics including optimizations.

        This method extends the parent's stats with optimization-specific metrics.

        Returns:
            Dict: Comprehensive performance statistics
        """
        # Get base stats from parent (includes both lexical and semantic stats)
        base_stats = super().get_performance_stats()

        # Add optimization-specific stats
        optimization_stats = self.get_optimization_stats()
        if optimization_stats:
            base_stats["optimization_stats"] = optimization_stats

        # Add cache information
        cache_info = self.get_cache_info()
        if cache_info.get("cache_enabled", False):
            base_stats["cache_info"] = cache_info

        # Calculate overall optimization impact if data is available
        if optimization_stats and "optimization_time_saved_ms" in optimization_stats:
            total_time = sum(base_stats.get("processing_times", []))
            time_saved = optimization_stats["optimization_time_saved_ms"]
            if total_time > 0:
                base_stats["optimization_impact"] = {
                    "time_saved_ms": time_saved,
                    "time_saved_percentage": (time_saved / (total_time + time_saved))
                    * 100,
                }

        return base_stats

    def explain_classification(self, text: str) -> Dict[str, Any]:
        """
        Enhanced explanation including optimization details.

        Args:
            text: Input text to explain

        Returns:
            Dict: Comprehensive explanation with optimization information
        """
        # Get base explanation from parent
        explanation = super().explain_classification(text)

        # Add optimization information section
        explanation["optimization_info"] = {
            "cache_info": self.get_cache_info(),
            "optimization_stats": self.get_optimization_stats(),
            "semantic_classifier_type": type(self.semantic_classifier).__name__,
            "optimizations_enabled": [],
        }

        # Determine which optimizations are enabled
        optimizations = []
        if self.config.features.faiss_indexing:
            optimizations.append("faiss_indexing")
        if self.get_cache_info().get("cache_enabled", False):
            optimizations.append("cache_persistence")
        if hasattr(self.semantic_classifier.embedding_model, "clear_cache"):
            optimizations.append("embedding_caching")

        explanation["optimization_info"]["optimizations_enabled"] = optimizations

        return explanation

    # =========================================================================
    # FACTORY CLASS METHODS
    # =========================================================================

    @classmethod
    def create_production_classifier(
        cls, config: Optional[CXDConfig] = None
    ) -> "OptimizedMetaCXDClassifier":
        """
        Create a production-optimized classifier with best performance settings.

        Args:
            config: Configuration object (creates production config if None)

        Returns:
            OptimizedMetaCXDClassifier: Production-optimized classifier
        """
        if config is None:
            from ..core.config import create_production_config

            config = create_production_config()

        return cls(
            config=config,
            enable_cache_persistence=True,
            rebuild_cache=False,  # Don't rebuild on startup in production
        )

    @classmethod
    def create_development_classifier(
        cls, config: Optional[CXDConfig] = None
    ) -> "OptimizedMetaCXDClassifier":
        """
        Create a development classifier with debugging features.

        Args:
            config: Configuration object (creates development config if None)

        Returns:
            OptimizedMetaCXDClassifier: Development-optimized classifier
        """
        if config is None:
            from ..core.config import create_development_config

            config = create_development_config()

        return cls(
            config=config,
            enable_cache_persistence=True,
            rebuild_cache=True,  # Always rebuild in development for fresh state
        )


# =============================================================================
# FACTORY FUNCTIONS FOR CONVENIENCE
# =============================================================================


def create_optimized_classifier(
    config: Optional[CXDConfig] = None, **kwargs
) -> OptimizedMetaCXDClassifier:
    """
    Create an optimized CXD classifier with default settings.

    Args:
        config: Configuration object
        **kwargs: Additional arguments for optimization

    Returns:
        OptimizedMetaCXDClassifier: Configured classifier
    """
    return OptimizedMetaCXDClassifier(config=config, **kwargs)


def create_fast_classifier(
    embedding_cache_size: int = 1000,
    rebuild_cache: bool = False,
    config: Optional[CXDConfig] = None,
) -> OptimizedMetaCXDClassifier:
    """
    Create a classifier optimized for speed.

    Args:
        embedding_cache_size: Size of embedding cache (passed as hint)
        rebuild_cache: Whether to rebuild cache on startup
        config: Configuration object (creates speed-optimized if None)

    Returns:
        OptimizedMetaCXDClassifier: Speed-optimized classifier
    """
    if config is None:
        config = CXDConfig()
        # Enable all speed optimizations
        config.features.faiss_indexing = True
        config.features.cache_persistence = True
        config.performance.batch_size = 64  # Larger batches for speed

    return OptimizedMetaCXDClassifier(
        config=config, enable_cache_persistence=True, rebuild_cache=rebuild_cache
    )


# Export optimized classifier and factory functions
__all__ = [
    "OptimizedMetaCXDClassifier",
    "create_optimized_classifier",
    "create_fast_classifier",
]

