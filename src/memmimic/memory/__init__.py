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
from .storage.amms_storage import AMMSStorage, Memory
from .storage import create_amms_storage

# Enhanced Memory Model (v2.0)
from .enhanced_memory import EnhancedMemory
from .enhanced_amms_storage import EnhancedAMMSStorage, create_enhanced_amms_storage

# Governance Framework (v2.0)
from .governance import (
    SimpleGovernance, GovernanceConfig, GovernanceValidator,
    ThresholdManager, GovernanceMetrics, GovernanceViolation, 
    GovernanceResult, GovernanceConfigError
)
from .governance_integrated_storage import (
    GovernanceIntegratedStorage, GovernanceAwareResult, 
    create_governance_integrated_storage
)

__all__ = [
    # Core interfaces
    "Memory",
    "EnhancedMemory",
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
    # Enhanced AMMS Storage (v2.0)
    "EnhancedAMMSStorage", 
    "create_enhanced_amms_storage",
    # Governance Framework (v2.0)
    "SimpleGovernance",
    "GovernanceConfig", 
    "GovernanceValidator",
    "ThresholdManager",
    "GovernanceMetrics",
    "GovernanceViolation",
    "GovernanceResult",
    "GovernanceConfigError",
    "GovernanceIntegratedStorage",
    "GovernanceAwareResult",
    "create_governance_integrated_storage",
]

