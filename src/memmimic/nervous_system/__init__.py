"""
MemMimic Nervous System - Biological Reflex Architecture

This module implements the nervous system transformation from external MCP tools
to internal biological reflex triggers with <5ms response times.

Core components:
- NervousSystemCore: Central intelligence shared by all triggers
- InternalQualityGate: Automatic quality assessment without external queues
- SemanticDuplicateDetector: Real-time duplicate detection and resolution
- InternalSocraticGuidance: Wisdom-based decision making
"""

from .core import NervousSystemCore
from .interfaces import (
    InternalIntelligenceInterface,
    QualityGateInterface,
    DuplicateDetectorInterface,
    SocraticGuidanceInterface,
    QualityAssessment,
    DuplicateAnalysis,
    SocraticGuidance
)
from .quality_gate import InternalQualityGate
from .duplicate_detector import SemanticDuplicateDetector
from .socratic_guidance import InternalSocraticGuidance

__all__ = [
    'NervousSystemCore',
    'InternalIntelligenceInterface',
    'QualityGateInterface', 
    'DuplicateDetectorInterface',
    'SocraticGuidanceInterface',
    'QualityAssessment',
    'DuplicateAnalysis', 
    'SocraticGuidance',
    'InternalQualityGate',
    'SemanticDuplicateDetector',
    'InternalSocraticGuidance'
]

__version__ = "1.0.0"