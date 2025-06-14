"""
Core types for CXD Classifier.

This module defines the fundamental data structures and types used throughout
the CXD classification system. These types form the foundation for representing
cognitive executive dynamics (CXD) functions, states, and classification results.

The CXD ontology consists of:
- CONTROL (C): Search, filter, decision, management operations
- CONTEXT (X): Relations, references, situational awareness
- DATA (D): Processing, transformation, generation, extraction

Each function can have execution states:
- SUCCESS (+): Function executed successfully  
- FAILURE (-): Function failed
- UNCERTAIN (?): Uncertain result
- PARTIAL (~): Partial functioning
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import hashlib

import numpy as np


# =============================================================================
# CORE ENUMS
# =============================================================================

class CXDFunction(Enum):
    """
    Primary cognitive functions in the CXD ontology.
    
    The CXD framework categorizes cognitive operations into three main functions:
    - CONTROL: Executive control operations (search, filter, decide, manage)
    - CONTEXT: Contextual awareness operations (relate, reference, situate)  
    - DATA: Data processing operations (analyze, transform, generate, extract)
    """
    
    CONTROL = "C"
    CONTEXT = "X" 
    DATA = "D"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def description(self) -> str:
        """Get human-readable description of the function."""
        descriptions = {
            CXDFunction.CONTROL: "Search, filter, decision, and management operations",
            CXDFunction.CONTEXT: "Relations, references, and situational awareness",
            CXDFunction.DATA: "Processing, transformation, generation, and extraction"
        }
        return descriptions[self]
    
    @property
    def keywords(self) -> List[str]:
        """Get typical keywords associated with this function."""
        keyword_map = {
            CXDFunction.CONTROL: [
                "search", "find", "filter", "select", "control", "manage", 
                "decide", "determine", "choose", "direct", "supervise"
            ],
            CXDFunction.CONTEXT: [
                "relate", "connect", "reference", "context", "situate", 
                "link", "associate", "background", "previous", "history"
            ],
            CXDFunction.DATA: [
                "process", "analyze", "transform", "generate", "extract",
                "compute", "calculate", "organize", "create", "derive"
            ]
        }
        return keyword_map[self]
    
    @classmethod
    def from_string(cls, value: str) -> 'CXDFunction':
        """
        Create CXDFunction from string representation.
        
        Args:
            value: String representation (C, X, D, CONTROL, CONTEXT, DATA)
            
        Returns:
            CXDFunction: Corresponding function enum
            
        Raises:
            ValueError: If value is not a valid function
        """
        value_upper = value.upper()
        
        # Direct mapping
        if value_upper in ["C", "CONTROL"]:
            return cls.CONTROL
        elif value_upper in ["X", "CONTEXT"]:
            return cls.CONTEXT
        elif value_upper in ["D", "DATA"]:
            return cls.DATA
        else:
            raise ValueError(f"Invalid CXD function: {value}")


class ExecutionState(Enum):
    """
    Execution states for CXD functions.
    
    Represents the outcome or status of executing a cognitive function:
    - SUCCESS: Function executed successfully
    - FAILURE: Function failed to execute properly
    - UNCERTAIN: Result is uncertain or ambiguous
    - PARTIAL: Function executed with partial success/limitations
    """
    
    SUCCESS = "+"
    FAILURE = "-"
    UNCERTAIN = "?"
    PARTIAL = "~"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def description(self) -> str:
        """Get human-readable description of the state."""
        descriptions = {
            ExecutionState.SUCCESS: "Function executed successfully",
            ExecutionState.FAILURE: "Function failed to execute",
            ExecutionState.UNCERTAIN: "Uncertain or ambiguous result", 
            ExecutionState.PARTIAL: "Partial execution or limited success"
        }
        return descriptions[self]
    
    @property
    def numeric_value(self) -> float:
        """Get numeric representation for calculations (0.0 to 1.0)."""
        values = {
            ExecutionState.SUCCESS: 1.0,
            ExecutionState.PARTIAL: 0.5,
            ExecutionState.UNCERTAIN: 0.25,
            ExecutionState.FAILURE: 0.0
        }
        return values[self]
    
    @classmethod
    def from_string(cls, value: str) -> 'ExecutionState':
        """
        Create ExecutionState from string representation.
        
        Args:
            value: String representation (+, -, ?, ~, SUCCESS, FAILURE, etc.)
            
        Returns:
            ExecutionState: Corresponding state enum
            
        Raises:
            ValueError: If value is not a valid state
        """
        value_upper = value.upper()
        
        # Symbol mapping
        symbol_map = {
            "+": cls.SUCCESS,
            "-": cls.FAILURE, 
            "?": cls.UNCERTAIN,
            "~": cls.PARTIAL
        }
        
        if value in symbol_map:
            return symbol_map[value]
        
        # Name mapping
        name_map = {
            "SUCCESS": cls.SUCCESS,
            "FAILURE": cls.FAILURE,
            "UNCERTAIN": cls.UNCERTAIN,
            "PARTIAL": cls.PARTIAL
        }
        
        if value_upper in name_map:
            return name_map[value_upper]
        
        raise ValueError(f"Invalid execution state: {value}")
    
    @classmethod
    def from_confidence(cls, confidence: float) -> 'ExecutionState':
        """
        Infer execution state from confidence score.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            ExecutionState: Inferred state based on confidence
        """
        if confidence >= 0.8:
            return cls.SUCCESS
        elif confidence >= 0.5:
            return cls.PARTIAL
        elif confidence >= 0.3:
            return cls.UNCERTAIN
        else:
            return cls.FAILURE


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class CXDTag:
    """
    Individual CXD tag representing a cognitive function with execution state.
    
    A CXD tag combines a cognitive function (C/X/D) with an execution state (+/-/?/~)
    and associated metadata including confidence scores and evidence.
    
    Examples:
        C+ = Control function executed successfully
        X- = Context function failed
        D? = Data function with uncertain outcome
    """
    
    function: CXDFunction
    state: ExecutionState = ExecutionState.SUCCESS
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)
    semantic_confidence: float = 0.0
    timestamp: Optional[float] = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        # Validate confidence bounds
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.semantic_confidence = max(0.0, min(1.0, self.semantic_confidence))
        
        # Ensure evidence is a list
        if not isinstance(self.evidence, list):
            self.evidence = [str(self.evidence)] if self.evidence else []
        
        # Set semantic_confidence default if not provided
        if self.semantic_confidence == 0.0 and self.confidence > 0.0:
            self.semantic_confidence = self.confidence
    
    def __str__(self) -> str:
        """String representation: C+, X-, D?, etc."""
        return f"{self.function.value}{self.state.value}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"CXDTag({self.function.name}, {self.state.name}, "
                f"conf={self.confidence:.2f})")
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CXDTag):
            return NotImplemented
        # Compara los campos que definen la igualdad de un tag.
        # Podrías querer ser más o menos estricto.
        # Por ejemplo, ¿deberían los timestamps o la evidencia ser parte de la igualdad?
        # Para la prueba actual, probablemente function, state y una confianza cercana sean suficientes.
        # O una comparación estricta de todos los atributos relevantes para la clasificación.
        return (self.function == other.function and
                self.state == other.state and
                # Ten cuidado al comparar floats directamente.
                # Podrías necesitar una función de "casi igual":
                abs(self.confidence - other.confidence) < 1e-7 and
                abs(self.semantic_confidence - other.semantic_confidence) < 1e-7 and
                # Compara otros campos que consideres importantes para la igualdad,
                # por ejemplo, la lista de evidencia si debe ser idéntica.
                # self.evidence == other.evidence and # Puede ser demasiado estricto
                self.metadata == other.metadata) # También puede ser demasiado estricto    
    @property
    def pattern(self) -> str:
        """Get the CXD pattern string (same as __str__)."""
        return str(self)
    
    @property
    def strength(self) -> float:
        """
        Calculate overall strength combining confidence and state.
        
        Returns:
            float: Weighted strength score (0.0 to 1.0)
        """
        state_weight = self.state.numeric_value
        return self.confidence * state_weight
    
    @property
    def is_successful(self) -> bool:
        """Check if this tag represents a successful execution."""
        return self.state == ExecutionState.SUCCESS
    
    @property
    def is_failed(self) -> bool:
        """Check if this tag represents a failed execution."""
        return self.state == ExecutionState.FAILURE
    
    @property
    def is_uncertain(self) -> bool:
        """Check if this tag represents an uncertain execution."""
        return self.state in [ExecutionState.UNCERTAIN, ExecutionState.PARTIAL]
    
    def add_evidence(self, evidence: Union[str, List[str]]) -> None:
        """
        Add evidence supporting this classification.
        
        Args:
            evidence: Evidence string or list of evidence strings
        """
        if isinstance(evidence, str):
            if evidence not in self.evidence:
                self.evidence.append(evidence)
        elif isinstance(evidence, list):
            for item in evidence:
                if item not in self.evidence:
                    self.evidence.append(item)
    
    def update_confidence(self, new_confidence: float, method: str = "replace") -> None:
        """
        Update confidence score with different combination methods.
        
        Args:
            new_confidence: New confidence value
            method: How to combine ("replace", "average", "max", "weighted")
        """
        new_confidence = max(0.0, min(1.0, new_confidence))
        
        if method == "replace":
            self.confidence = new_confidence
        elif method == "average":
            self.confidence = (self.confidence + new_confidence) / 2
        elif method == "max":
            self.confidence = max(self.confidence, new_confidence)
        elif method == "weighted":
            # Weight new confidence by current confidence
            weight = 0.3  # 30% new, 70% old
            self.confidence = (1 - weight) * self.confidence + weight * new_confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict: Serializable dictionary representation
        """
        return {
            "function": self.function.value,
            "state": self.state.value,
            "confidence": self.confidence,
            "semantic_confidence": self.semantic_confidence,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "pattern": str(self),
            "strength": self.strength
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CXDTag':
        """
        Create CXDTag from dictionary.
        
        Args:
            data: Dictionary with tag data
            
        Returns:
            CXDTag: Reconstructed tag instance
        """
        return cls(
            function=CXDFunction.from_string(data["function"]),
            state=ExecutionState.from_string(data["state"]),
            confidence=data.get("confidence", 0.8),
            evidence=data.get("evidence", []),
            semantic_confidence=data.get("semantic_confidence", 0.0),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_string(cls, tag_str: str, confidence: float = 0.8) -> 'CXDTag':
        """
        Parse CXD tag from string representation.
        
        Args:
            tag_str: Tag string like "C+", "X-", "D?"
            confidence: Default confidence if not specified
            
        Returns:
            CXDTag: Parsed tag instance
            
        Raises:
            ValueError: If tag string is invalid
        """
        if len(tag_str) < 2:
            raise ValueError(f"Invalid CXD tag string: {tag_str}")
        
        function_char = tag_str[0].upper()
        state_char = tag_str[1]
        
        try:
            function = CXDFunction.from_string(function_char)
            state = ExecutionState.from_string(state_char)
            return cls(function=function, state=state, confidence=confidence)
        except ValueError as e:
            raise ValueError(f"Invalid CXD tag string '{tag_str}': {e}")


@dataclass
class CXDSequence:
    """
    Ordered sequence of CXD tags representing a complete classification.
    
    A CXD sequence represents the full cognitive profile of a text or operation,
    consisting of multiple CXD tags in order of importance/confidence.
    
    Examples:
        C+X+D+ = Control, context, and data all successful
        C+D- = Control successful, data failed
        X? = Only context, uncertain
    """
    
    tags: List[CXDTag]
    timestamp: Optional[float] = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize sequence after initialization."""
        # Ensure tags is a list
        if not isinstance(self.tags, list):
            self.tags = [self.tags] if self.tags else []
        
        # Remove duplicate functions (keep highest confidence)
        self._deduplicate_functions()
        
        # Sort by confidence (highest first)
        self.tags.sort(key=lambda t: t.confidence, reverse=True)
    
    def _deduplicate_functions(self) -> None:
        """Remove duplicate functions, keeping the one with highest confidence."""
        seen_functions = set()
        unique_tags = []
        
        for tag in sorted(self.tags, key=lambda t: t.confidence, reverse=True):
            if tag.function not in seen_functions:
                unique_tags.append(tag)
                seen_functions.add(tag.function)
        
        self.tags = unique_tags
    
    def __str__(self) -> str:
        """String representation with arrows: C+→X+→D+"""
        if not self.tags:
            return "∅"  # Empty set symbol
        return "→".join(str(tag) for tag in self.tags)
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"CXDSequence([{', '.join(repr(tag) for tag in self.tags)}])"
    
    def __len__(self) -> int:
        """Number of tags in sequence."""
        return len(self.tags)
    
    def __getitem__(self, index: int) -> CXDTag:
        """Get tag by index."""
        return self.tags[index]
    
    def __iter__(self):
        """Iterate over tags."""
        return iter(self.tags)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CXDSequence):
            return NotImplemented
        # Dos secuencias son iguales si su lista de tags es igual (y en el mismo orden).
        # La comparación de listas usará el __eq__ de CXDTag.
        if len(self.tags) != len(other.tags):
            return False
        for i in range(len(self.tags)):
            if self.tags[i] != other.tags[i]: # Esto usará CXDTag.__eq__
                return False
        # Podrías también querer comparar metadatos si son relevantes para la igualdad.
        # return self.metadata == other.metadata
        return True # Si todos los tags son iguales en orden
    
    @property
    def pattern(self) -> str:
        """Get function pattern: CXD, XD, C, etc."""
        return "".join(tag.function.value for tag in self.tags)
    
    @property
    def execution_pattern(self) -> str:
        """Get full execution pattern: C+X-D+"""
        return "".join(str(tag) for tag in self.tags)
    
    @property
    def dominant_function(self) -> Optional[CXDFunction]:
        """Get the dominant (first/highest confidence) function."""
        return self.tags[0].function if self.tags else None
    
    @property
    def dominant_tag(self) -> Optional[CXDTag]:
        """Get the dominant (first/highest confidence) tag."""
        return self.tags[0] if self.tags else None
    
    @property
    def functions(self) -> List[CXDFunction]:
        """Get list of all functions in sequence."""
        return [tag.function for tag in self.tags]
    
    @property
    def states(self) -> List[ExecutionState]:
        """Get list of all states in sequence."""
        return [tag.state for tag in self.tags]
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all tags."""
        if not self.tags:
            return 0.0
        return sum(tag.confidence for tag in self.tags) / len(self.tags)
    
    @property
    def weighted_confidence(self) -> float:
        """Calculate weighted average confidence (higher positions weighted more)."""
        if not self.tags:
            return 0.0
        
        total_weight = 0
        weighted_sum = 0
        
        for i, tag in enumerate(self.tags):
            weight = 1.0 / (i + 1)  # Decreasing weight: 1.0, 0.5, 0.33, ...
            weighted_sum += tag.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight
    
    @property
    def strength_score(self) -> float:
        """Calculate overall strength combining confidence and states."""
        if not self.tags:
            return 0.0
        return sum(tag.strength for tag in self.tags) / len(self.tags)
    
    @property
    def is_successful(self) -> bool:
        """Check if all functions executed successfully."""
        return all(tag.is_successful for tag in self.tags)
    
    @property
    def has_failures(self) -> bool:
        """Check if any functions failed."""
        return any(tag.is_failed for tag in self.tags)
    
    @property
    def success_rate(self) -> float:
        """Calculate percentage of successful functions."""
        if not self.tags:
            return 0.0
        successful = sum(1 for tag in self.tags if tag.is_successful)
        return successful / len(self.tags)
    
    def get_function_tag(self, function: CXDFunction) -> Optional[CXDTag]:
        """
        Get tag for specific function.
        
        Args:
            function: Function to find
            
        Returns:
            CXDTag: Tag for function, or None if not found
        """
        for tag in self.tags:
            if tag.function == function:
                return tag
        return None
    
    def has_function(self, function: CXDFunction) -> bool:
        """Check if sequence contains a specific function."""
        return function in self.functions
    
    def add_tag(self, tag: CXDTag) -> None:
        """
        Add a tag to the sequence.
        
        Args:
            tag: Tag to add
        """
        # Remove existing tag with same function
        self.tags = [t for t in self.tags if t.function != tag.function]
        
        # Add new tag
        self.tags.append(tag)
        
        # Re-sort by confidence
        self.tags.sort(key=lambda t: t.confidence, reverse=True)
    
    def remove_function(self, function: CXDFunction) -> bool:
        """
        Remove tag with specific function.
        
        Args:
            function: Function to remove
            
        Returns:
            bool: True if tag was removed, False if not found
        """
        original_length = len(self.tags)
        self.tags = [tag for tag in self.tags if tag.function != function]
        return len(self.tags) < original_length
    
    def filter_by_confidence(self, min_confidence: float) -> 'CXDSequence':
        """
        Create new sequence with tags above confidence threshold.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            CXDSequence: Filtered sequence
        """
        filtered_tags = [tag for tag in self.tags if tag.confidence >= min_confidence]
        return CXDSequence(filtered_tags, metadata=self.metadata.copy())
    
    def filter_by_state(self, states: Union[ExecutionState, List[ExecutionState]]) -> 'CXDSequence':
        """
        Create new sequence with tags matching specific states.
        
        Args:
            states: State or list of states to include
            
        Returns:
            CXDSequence: Filtered sequence
        """
        if isinstance(states, ExecutionState):
            states = [states]
        
        filtered_tags = [tag for tag in self.tags if tag.state in states]
        return CXDSequence(filtered_tags, metadata=self.metadata.copy())
    
    def delta(self, other: 'CXDSequence') -> float:
        """
        Calculate difference (delta) between this and another sequence.
        
        The delta operator (δ) quantifies how different two CXD sequences are:
        - 0.0 = identical sequences
        - 1.0 = completely different sequences
        
        Args:
            other: Other sequence to compare with
            
        Returns:
            float: Difference score (0.0 to 1.0)
        """
        if not other.tags:
            return 1.0 if self.tags else 0.0
        
        if not self.tags:
            return 1.0
        
        # Function pattern difference (0.5 weight)
        pattern_diff = 0.5 if self.pattern != other.pattern else 0.0
        
        # State execution difference (0.5 weight)
        max_len = max(len(self.tags), len(other.tags))
        state_differences = 0
        
        for i in range(max_len):
            if i >= len(self.tags) or i >= len(other.tags):
                state_differences += 1  # Missing tag
            elif self.tags[i].state != other.tags[i].state:
                state_differences += 0.5  # Different state
        
        state_diff = (state_differences / max_len) * 0.5
        
        return min(pattern_diff + state_diff, 1.0)
    
    def similarity(self, other: 'CXDSequence') -> float:
        """
        Calculate similarity between sequences (1 - delta).
        
        Args:
            other: Other sequence to compare with
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        return 1.0 - self.delta(other)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict: Serializable dictionary representation
        """
        return {
            "tags": [tag.to_dict() for tag in self.tags],
            "pattern": self.pattern,
            "execution_pattern": self.execution_pattern,
            "dominant_function": self.dominant_function.value if self.dominant_function else None,
            "average_confidence": self.average_confidence,
            "weighted_confidence": self.weighted_confidence,
            "strength_score": self.strength_score,
            "success_rate": self.success_rate,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CXDSequence':
        """
        Create CXDSequence from dictionary.
        
        Args:
            data: Dictionary with sequence data
            
        Returns:
            CXDSequence: Reconstructed sequence instance
        """
        tags = [CXDTag.from_dict(tag_data) for tag_data in data["tags"]]
        return cls(
            tags=tags,
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_string(cls, sequence_str: str, confidence: float = 0.8) -> 'CXDSequence':
        """
        Parse CXD sequence from string representation.
        
        Args:
            sequence_str: String like "C+X-D?" or "C+→X-→D?"
            confidence: Default confidence for tags
            
        Returns:
            CXDSequence: Parsed sequence
        """
        # Split on arrows or spaces
        tag_strings = sequence_str.replace("→", " ").split()
        
        tags = []
        for tag_str in tag_strings:
            tag_str = tag_str.strip()
            if tag_str:
                tags.append(CXDTag.from_string(tag_str, confidence))
        
        return cls(tags)


@dataclass
class MetaClassificationResult:
    """
    Complete result from meta-classification including all intermediate steps.
    
    This class encapsulates the full output of the CXD meta-classification process,
    including both lexical and semantic classifications, the final fused result,
    confidence scores, and processing metadata.
    """
    
    text: str
    lexical_sequence: CXDSequence
    semantic_sequence: CXDSequence  
    final_sequence: CXDSequence
    confidence_scores: Dict[str, float]
    corrections_made: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    classifier_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and enrich result after initialization."""
        # Ensure all sequences are CXDSequence instances
        if not isinstance(self.lexical_sequence, CXDSequence):
            raise ValueError("lexical_sequence must be CXDSequence")
        if not isinstance(self.semantic_sequence, CXDSequence):
            raise ValueError("semantic_sequence must be CXDSequence") 
        if not isinstance(self.final_sequence, CXDSequence):
            raise ValueError("final_sequence must be CXDSequence")
        
        # Validate confidence scores
        for key, value in self.confidence_scores.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Confidence score '{key}' must be between 0.0 and 1.0")
    
    @property
    def concordance(self) -> float:
        """Get concordance score between lexical and semantic classifications."""
        return self.confidence_scores.get("concordance", 0.0)
    
    @property
    def final_confidence(self) -> float:
        """Get final classification confidence."""
        return self.confidence_scores.get("final", 0.0)
    
    @property
    def dominant_function(self) -> Optional[CXDFunction]:
        """Get the dominant function from final classification."""
        return self.final_sequence.dominant_function
    
    @property
    def pattern(self) -> str:
        """Get the final CXD pattern."""
        return self.final_sequence.pattern
    
    @property
    def execution_pattern(self) -> str:
        """Get the final execution pattern."""
        return self.final_sequence.execution_pattern
    
    @property
    def has_corrections(self) -> bool:
        """Check if any corrections were made during classification."""
        return len(self.corrections_made) > 0
    
    @property
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if final confidence is above threshold."""
        return self.final_confidence >= threshold
    
    @property
    def is_concordant(self, threshold: float = 0.7) -> bool:
        """Check if lexical and semantic classifications are concordant."""
        return self.concordance >= threshold
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary of the classification.
        
        Returns:
            Dict: Performance metrics and analysis
        """
        return {
            "processing_time_ms": self.processing_time_ms,
            "final_confidence": self.final_confidence,
            "concordance": self.concordance,
            "dominant_function": self.dominant_function.value if self.dominant_function else None,
            "pattern": self.pattern,
            "corrections_count": len(self.corrections_made),
            "has_corrections": self.has_corrections,
            "is_high_confidence": self.is_high_confidence(),
            "is_concordant": self.is_concordant(),
            "lexical_tags_count": len(self.lexical_sequence.tags),
            "semantic_tags_count": len(self.semantic_sequence.tags),
            "final_tags_count": len(self.final_sequence.tags)
        }
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis of the classification process.
        
        Returns:
            Dict: Comprehensive analysis including comparisons
        """
        return {
            "input": {
                "text": self.text,
                "text_length": len(self.text),
                "word_count": len(self.text.split())
            },
            "lexical_analysis": {
                "sequence": str(self.lexical_sequence),
                "dominant": self.lexical_sequence.dominant_function.value if self.lexical_sequence.dominant_function else None,
                "confidence": self.confidence_scores.get("lexical", 0.0),
                "tags": [tag.to_dict() for tag in self.lexical_sequence.tags]
            },
            "semantic_analysis": {
                "sequence": str(self.semantic_sequence),
                "dominant": self.semantic_sequence.dominant_function.value if self.semantic_sequence.dominant_function else None,
                "confidence": self.confidence_scores.get("semantic", 0.0),
                "tags": [tag.to_dict() for tag in self.semantic_sequence.tags]
            },
            "meta_analysis": {
                "concordance": self.concordance,
                "corrections": self.corrections_made,
                "final_sequence": str(self.final_sequence),
                "final_confidence": self.final_confidence,
                "delta_lexical_final": self.lexical_sequence.delta(self.final_sequence),
                "delta_semantic_final": self.semantic_sequence.delta(self.final_sequence),
                "processing_time_ms": self.processing_time_ms
            },
            "performance": self.get_performance_summary()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dict: Serializable dictionary representation
        """
        return {
            "text": self.text,
            "lexical_sequence": self.lexical_sequence.to_dict(),
            "semantic_sequence": self.semantic_sequence.to_dict(),
            "final_sequence": self.final_sequence.to_dict(),
            "confidence_scores": self.confidence_scores,
            "corrections_made": self.corrections_made,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "classifier_metadata": self.classifier_metadata,
            "pattern": self.pattern,
            "dominant_function": self.dominant_function.value if self.dominant_function else None,
            "analysis": self.get_performance_summary()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaClassificationResult':
        """
        Create MetaClassificationResult from dictionary.
        
        Args:
            data: Dictionary with result data
            
        Returns:
            MetaClassificationResult: Reconstructed result instance
        """
        return cls(
            text=data["text"],
            lexical_sequence=CXDSequence.from_dict(data["lexical_sequence"]),
            semantic_sequence=CXDSequence.from_dict(data["semantic_sequence"]),
            final_sequence=CXDSequence.from_dict(data["final_sequence"]),
            confidence_scores=data["confidence_scores"],
            corrections_made=data.get("corrections_made", []),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            timestamp=data.get("timestamp", time.time()),
            classifier_metadata=data.get("classifier_metadata", {})
        )
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.
        
        Args:
            indent: JSON indentation
            
        Returns:
            str: JSON representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod  
    def from_json(cls, json_str: str) -> 'MetaClassificationResult':
        """
        Create MetaClassificationResult from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            MetaClassificationResult: Reconstructed result
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_simple_sequence(functions: List[str], 
                          states: Optional[List[str]] = None,
                          confidences: Optional[List[float]] = None) -> CXDSequence:
    """
    Create a simple CXD sequence from function and state lists.
    
    Args:
        functions: List of function strings (C, X, D)
        states: List of state strings (+, -, ?, ~). Default: all SUCCESS
        confidences: List of confidence values. Default: all 0.8
        
    Returns:
        CXDSequence: Created sequence
        
    Example:
        >>> seq = create_simple_sequence(["C", "D"], ["+", "-"], [0.9, 0.3])
        >>> print(seq)  # C+→D-
    """
    if states is None:
        states = ["+"] * len(functions)
    if confidences is None:
        confidences = [0.8] * len(functions)
    
    if not (len(functions) == len(states) == len(confidences)):
        raise ValueError("Functions, states, and confidences must have same length")
    
    tags = []
    for func, state, conf in zip(functions, states, confidences):
        function = CXDFunction.from_string(func)
        execution_state = ExecutionState.from_string(state)
        tags.append(CXDTag(function, execution_state, conf))
    
    return CXDSequence(tags)


def parse_cxd_pattern(pattern: str, confidence: float = 0.8) -> CXDSequence:
    """
    Parse a CXD pattern string into a sequence.
    
    Args:
        pattern: Pattern string like "C+X-D?" or "CXD"
        confidence: Default confidence for tags
        
    Returns:
        CXDSequence: Parsed sequence
        
    Example:
        >>> seq = parse_cxd_pattern("C+X-D?")
        >>> print(seq.pattern)  # CXD
    """
    return CXDSequence.from_string(pattern, confidence)


def calculate_sequence_hash(sequence: CXDSequence) -> str:
    """
    Calculate a hash for a CXD sequence for caching/comparison.
    
    Args:
        sequence: CXD sequence to hash
        
    Returns:
        str: SHA-256 hash of sequence
    """
    # Create deterministic string representation
    hash_string = sequence.execution_pattern
    for tag in sequence.tags:
        hash_string += f"|{tag.confidence:.3f}"
        hash_string += f"|{','.join(sorted(tag.evidence))}"
    
    return hashlib.sha256(hash_string.encode()).hexdigest()[:16]


def merge_sequences(sequences: List[CXDSequence], 
                   weights: Optional[List[float]] = None) -> CXDSequence:
    """
    Merge multiple CXD sequences into one using weighted averaging.
    
    Args:
        sequences: List of sequences to merge
        weights: Optional weights for each sequence (default: equal weights)
        
    Returns:
        CXDSequence: Merged sequence
    """
    if not sequences:
        return CXDSequence([])
    
    if weights is None:
        weights = [1.0] * len(sequences)
    
    if len(weights) != len(sequences):
        raise ValueError("Weights must match number of sequences")
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Collect all functions that appear
    all_functions = set()
    for seq in sequences:
        all_functions.update(seq.functions)
    
    # Merge tags for each function
    merged_tags = []
    for function in all_functions:
        weighted_confidence = 0.0
        weighted_semantic = 0.0
        all_evidence = []
        states = []
        
        for seq, weight in zip(sequences, weights):
            tag = seq.get_function_tag(function)
            if tag:
                weighted_confidence += tag.confidence * weight
                weighted_semantic += tag.semantic_confidence * weight
                all_evidence.extend(tag.evidence)
                states.append(tag.state)
        
        # Choose most common state
        if states:
            state = max(set(states), key=states.count)
            
            merged_tag = CXDTag(
                function=function,
                state=state,
                confidence=weighted_confidence,
                semantic_confidence=weighted_semantic,
                evidence=list(set(all_evidence))  # Remove duplicates
            )
            merged_tags.append(merged_tag)
    
    return CXDSequence(merged_tags)


# Export all public types and functions
__all__ = [
    # Enums
    "CXDFunction",
    "ExecutionState",
    
    # Data structures
    "CXDTag", 
    "CXDSequence",
    "MetaClassificationResult",
    
    # Utility functions
    "create_simple_sequence",
    "parse_cxd_pattern", 
    "calculate_sequence_hash",
    "merge_sequences",
]
