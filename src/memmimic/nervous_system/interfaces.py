"""
Nervous System Interfaces

Defines the core interfaces for internal intelligence components
that enable biological reflex processing with <5ms response times.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from ..memory.storage.amms_storage import Memory

@dataclass
class QualityAssessment:
    """Quality assessment result with multi-dimensional scoring"""
    overall_score: float  # 0.0-1.0 scale
    dimensions: Dict[str, float]  # Individual quality factors
    enhancement_suggestions: List[str]
    confidence: float
    needs_enhancement: bool
    auto_approve: bool

@dataclass
class DuplicateAnalysis:
    """Duplicate detection analysis result"""
    is_duplicate: bool
    similarity_score: float  # 0.0-1.0 scale
    duplicate_memory_id: Optional[str]
    resolution_action: str  # 'store', 'merge', 'reference', 'enhance_existing'
    relationship_strength: float
    preservation_metadata: Dict[str, Any]

@dataclass
class SocraticGuidance:
    """Socratic wisdom for memory decisions"""
    recommendation: str
    reasoning: List[str]
    questions: List[str]
    confidence: float
    alternatives: List[str]
    impact_analysis: Dict[str, Any]

class InternalIntelligenceInterface(ABC):
    """Base interface for all internal intelligence components"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the intelligence component"""
        pass
    
    @abstractmethod
    async def process_async(self, data: Any) -> Any:
        """Process data asynchronously for <5ms performance"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for monitoring"""
        pass

class QualityGateInterface(InternalIntelligenceInterface):
    """Interface for internal quality assessment without external queues"""
    
    @abstractmethod
    async def assess_quality(self, content: str, memory_type: str = "interaction") -> QualityAssessment:
        """
        Assess memory quality across 6 dimensions:
        1. Content clarity and coherence
        2. Information density and value
        3. Factual accuracy and reliability  
        4. Contextual relevance
        5. Uniqueness and novelty
        6. Long-term importance potential
        """
        pass
    
    @abstractmethod
    async def enhance_content(self, content: str, assessment: QualityAssessment) -> str:
        """Enhance content based on quality assessment"""
        pass
    
    @abstractmethod
    def get_quality_threshold(self, memory_type: str) -> float:
        """Get quality threshold for automatic approval"""
        pass

class DuplicateDetectorInterface(InternalIntelligenceInterface):
    """Interface for semantic duplicate detection and resolution"""
    
    @abstractmethod
    async def detect_duplicates(self, content: str, memory_type: str) -> DuplicateAnalysis:
        """
        Detect semantic duplicates using:
        - Vector similarity (0.85+ threshold for duplicates)
        - Content structure analysis
        - Temporal context consideration
        """
        pass
    
    @abstractmethod
    async def resolve_duplicate(self, analysis: DuplicateAnalysis, new_content: str) -> Memory:
        """Resolve duplicate with intelligent merge or reference strategy"""
        pass
    
    @abstractmethod
    async def map_relationships(self, memory: Memory, similar_memories: List[Memory]) -> Dict[str, float]:
        """Map bidirectional relationships with strength scoring"""
        pass

class SocraticGuidanceInterface(InternalIntelligenceInterface):
    """Interface for internal Socratic wisdom and decision guidance"""
    
    @abstractmethod
    async def guide_memory_decision(self, content: str, context: Dict[str, Any]) -> SocraticGuidance:
        """
        Provide Socratic guidance for memory decisions using:
        - Progressive questioning (What? Why? How? What if? So what?)
        - Contextual wisdom from memory patterns
        - Impact analysis and alternatives
        """
        pass
    
    @abstractmethod
    async def guide_memory_update(self, memory_id: str, proposed_update: str) -> SocraticGuidance:
        """Guide memory update decisions with Socratic analysis"""
        pass
    
    @abstractmethod
    async def guide_memory_deletion(self, memory_id: str, reason: Optional[str] = None) -> SocraticGuidance:
        """Guide memory deletion decisions with impact analysis"""
        pass
    
    @abstractmethod
    def get_wisdom_patterns(self) -> Dict[str, Any]:
        """Get learned wisdom patterns for decision making"""
        pass