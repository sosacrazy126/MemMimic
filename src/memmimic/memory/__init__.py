"""
MemMimic Memory System
Enhanced with Active Memory Management System (AMMS)
"""

# AMMS Components (available for advanced usage)
from .active_manager import ActiveMemoryConfig, ActiveMemoryPool
from .active_schema import ActiveMemorySchema
from .assistant import ContextualAssistant
from .importance_scorer import ImportanceScorer, ScoringWeights
from .memory import Memory, MemoryStore
from .socratic import SocraticDialogue, SocraticEngine
from .stale_detector import StaleDetectionConfig, StaleMemoryDetector
from .unified_store import UnifiedMemoryStore

__all__ = [
    # Core interfaces (backward compatible)
    "MemoryStore",
    "Memory",
    "ContextualAssistant",
    "UnifiedMemoryStore",
    "SocraticEngine",
    "SocraticDialogue",
    # AMMS Components (for advanced usage)
    "ActiveMemoryPool",
    "ActiveMemoryConfig",
    "ImportanceScorer",
    "ScoringWeights",
    "StaleMemoryDetector",
    "StaleDetectionConfig",
    "ActiveMemorySchema",
]
