"""
CXD Framework Integration

Cognitive function classification (Control/Context/Data).
"""

from .classifiers.factory import CXDClassifierFactory
from .classifiers.meta import MetaCXDClassifier
from .classifiers.optimized_meta import (
    OptimizedMetaCXDClassifier,
    create_fast_classifier,
    create_optimized_classifier,
)
from .core.config import CXDConfig
from .core.types import CXDFunction, CXDSequence, CXDTag, ExecutionState

__all__ = [
    # Core types
    "CXDFunction",
    "ExecutionState",
    "CXDTag",
    "CXDSequence",
    # Main classifiers
    "OptimizedMetaCXDClassifier",
    "MetaCXDClassifier",
    # Factory functions
    "create_optimized_classifier",
    "create_fast_classifier",
    "CXDClassifierFactory",
    # Configuration
    "CXDConfig",
]

