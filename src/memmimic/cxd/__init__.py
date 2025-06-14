"""
CXD Framework Integration

Cognitive function classification (Control/Context/Data).
"""

from .core.types import CXDFunction, ExecutionState, CXDTag, CXDSequence
from .classifiers.optimized_meta import OptimizedMetaCXDClassifier, create_optimized_classifier, create_fast_classifier
from .classifiers.meta import MetaCXDClassifier
from .classifiers.factory import CXDClassifierFactory
from .core.config import CXDConfig

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
