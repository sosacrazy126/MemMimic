"""
MemMimic - The Memory System That Learns You Back

A contextual memory intelligence system for AI.
"""

__version__ = "1.0.0"
__author__ = "Sprooket & Claude"

# Main API
from .api import MemMimicAPI, create_memmimic
from .cxd import (
    CXDFunction,
    CXDSequence,
    CXDTag,
    ExecutionState,
    OptimizedMetaCXDClassifier,
    create_optimized_classifier,
)

# Core components (for advanced users)
from .memory import (
    ContextualAssistant, Memory,
    AMMSStorage, create_amms_storage
)
from .tales import TaleManager

__all__ = [
    # Main API (most users)
    "MemMimicAPI",
    "create_memmimic",
    # Core components (advanced users)
    "Memory",
    "ContextualAssistant", 
    "TaleManager",
    # AMMS Storage (Post-Migration)
    "AMMSStorage",
    "create_amms_storage",
    # CXD Framework
    "CXDFunction",
    "ExecutionState",
    "CXDTag",
    "CXDSequence",
    "OptimizedMetaCXDClassifier",
    "create_optimized_classifier",
]
