"""
Classifiers module for CXD Classifier.

Contains all classification implementations including lexical, semantic,
and meta-classifiers with their optimized variants.
"""

from .lexical import LexicalCXDClassifier
from .meta import MetaCXDClassifier
from .optimized_meta import (
    OptimizedMetaCXDClassifier,
    create_fast_classifier,
    create_optimized_classifier,
)
from .optimized_semantic import OptimizedSemanticCXDClassifier
from .semantic import SemanticCXDClassifier

# Try to import factory if it exists
try:
    from .factory import CXDClassifierFactory

    _has_factory = True
except ImportError:
    CXDClassifierFactory = None
    _has_factory = False

__all__ = [
    # Base classifiers
    "LexicalCXDClassifier",
    "SemanticCXDClassifier",
    "MetaCXDClassifier",
    # Optimized classifiers
    "OptimizedSemanticCXDClassifier",
    "OptimizedMetaCXDClassifier",
    # Factory functions
    "create_optimized_classifier",
    "create_fast_classifier",
]

# Add factory to exports if available
if _has_factory:
    __all__.append("CXDClassifierFactory")
