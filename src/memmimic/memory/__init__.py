"""
MemMimic Memory System
Enhanced with Active Memory Management System (AMMS)
"""

# AMMS Components
from .active_manager import ActiveMemoryConfig, ActiveMemoryPool
from .active_schema import ActiveMemorySchema
from .assistant import ContextualAssistant
from .importance_scorer import ImportanceScorer, ScoringWeights
from .socratic import SocraticDialogue, SocraticEngine
from .stale_detector import StaleDetectionConfig, StaleMemoryDetector

# AMMS-Only Storage (Post-Migration Architecture)
from .storage import AMMSStorage, create_amms_storage, Memory

__all__ = [
    # Core interfaces
    "Memory",
    "ContextualAssistant",
    "SocraticEngine", 
    "SocraticDialogue",
    # AMMS Components
    "ActiveMemoryPool",
    "ActiveMemoryConfig", 
    "ImportanceScorer",
    "ScoringWeights",
    "StaleMemoryDetector",
    "StaleDetectionConfig",
    "ActiveMemorySchema",
    # AMMS-Only Storage (Post-Migration)
    "AMMSStorage",
    "create_amms_storage",
]

