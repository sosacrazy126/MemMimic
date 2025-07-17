"""
MemMimic Memory System
Enhanced with Active Memory Management System (AMMS)
"""

from .memory import MemoryStore, Memory
from .assistant import ContextualAssistant
from .socratic import SocraticEngine, SocraticDialogue
from .unified_store import UnifiedMemoryStore

# AMMS Components (available for advanced usage)
from .active_manager import ActiveMemoryPool, ActiveMemoryConfig
from .importance_scorer import ImportanceScorer, ScoringWeights
from .stale_detector import StaleMemoryDetector, StaleDetectionConfig
from .active_schema import ActiveMemorySchema

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
    "ActiveMemorySchema"
]
