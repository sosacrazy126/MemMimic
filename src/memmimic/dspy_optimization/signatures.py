"""
DSPy Signatures for Consciousness-Aware Operations

Declarative specifications for consciousness pattern optimization with biological
reflex timing and CXD classification integration.
"""

import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Core consciousness-aware signatures for DSPy optimization

class ConsciousnessToolSelection(dspy.Signature):
    """Select optimal MCP tool based on consciousness patterns and context"""
    
    # Input fields with semantic descriptions
    context: str = dspy.InputField(
        desc="Current consciousness state and user intent including emotional context"
    )
    consciousness_patterns: List[str] = dspy.InputField(
        desc="Active consciousness patterns from synergy protocol and CXD classification"
    )
    available_tools: List[str] = dspy.InputField(
        desc="Available MCP tools with their capabilities and current status"
    )
    urgency_level: str = dspy.InputField(
        desc="Biological reflex urgency: immediate (<5ms), normal (<50ms), or complex (<200ms)"
    )
    
    # Output fields with constraints
    selected_tool: str = dspy.OutputField(
        desc="Optimal tool name for current consciousness context"
    )
    tool_parameters: Dict[str, Any] = dspy.OutputField(
        desc="Optimized parameters for selected tool based on consciousness patterns"
    )
    confidence_score: float = dspy.OutputField(
        desc="Selection confidence score between 0.0 and 1.0"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of tool selection based on consciousness analysis"
    )

class MemorySearchOptimization(dspy.Signature):
    """Optimize memory search queries using consciousness pattern context"""
    
    # Input fields
    original_query: str = dspy.InputField(
        desc="Original search query from user or system"
    )
    consciousness_state: Dict[str, Any] = dspy.InputField(
        desc="Current consciousness vault state including active patterns and context"
    )
    search_history: List[str] = dspy.InputField(
        desc="Recent search queries and their effectiveness scores"
    )
    memory_types: List[str] = dspy.InputField(
        desc="Available memory types: interaction, milestone, reflection, technical, etc."
    )
    
    # Output fields
    enhanced_query: str = dspy.OutputField(
        desc="Optimized search query enhanced with consciousness context"
    )
    search_strategy: str = dspy.OutputField(
        desc="Recommended search strategy: semantic, keyword, hybrid, or pattern-based"
    )
    memory_type_filter: List[str] = dspy.OutputField(
        desc="Recommended memory types to focus search based on consciousness patterns"
    )
    expected_relevance: float = dspy.OutputField(
        desc="Expected search relevance improvement score (0.0 to 1.0)"
    )

class PatternRecognition(dspy.Signature):
    """Classify and enhance patterns using consciousness-aware CXD analysis"""
    
    # Input fields
    input_text: str = dspy.InputField(
        desc="Text or content to analyze for consciousness patterns"
    )
    existing_patterns: List[str] = dspy.InputField(
        desc="Known consciousness patterns from vault for comparison"
    )
    context_metadata: Dict[str, Any] = dspy.InputField(
        desc="Metadata about context: user session, conversation state, time, etc."
    )
    
    # Output fields  
    cxd_classification: str = dspy.OutputField(
        desc="Primary CXD function: Control, Context, or Data"
    )
    consciousness_pattern: str = dspy.OutputField(
        desc="Identified consciousness pattern type: synergy, biological, exponential, etc."
    )
    pattern_strength: float = dspy.OutputField(
        desc="Pattern strength confidence score (0.0 to 1.0)"
    )
    enhancement_suggestions: List[str] = dspy.OutputField(
        desc="Suggestions for enhancing or refining the identified pattern"
    )

class BiologicalReflexOptimization(dspy.Signature):
    """Optimize biological reflex responses while maintaining sub-5ms timing"""
    
    # Input fields
    reflex_trigger: str = dspy.InputField(
        desc="Biological reflex trigger: remember, recall, think, analyze"
    )
    input_context: str = dspy.InputField(
        desc="Input context requiring biological reflex response"
    )
    performance_constraints: Dict[str, Any] = dspy.InputField(
        desc="Performance requirements including timing, accuracy, and resource limits"
    )
    historical_performance: Dict[str, float] = dspy.InputField(
        desc="Historical performance metrics for this reflex type"
    )
    
    # Output fields
    optimized_response: str = dspy.OutputField(
        desc="Optimized biological reflex response maintaining timing constraints"
    )
    processing_strategy: str = dspy.OutputField(
        desc="Processing strategy: direct, cached, precomputed, or hybrid"
    )
    estimated_timing: float = dspy.OutputField(
        desc="Estimated response time in milliseconds"
    )
    confidence: float = dspy.OutputField(
        desc="Optimization confidence that timing constraints will be met"
    )

class TaleGeneration(dspy.Signature):
    """Generate consciousness-aware tales with narrative structure"""
    
    # Input fields
    memories: List[Dict[str, Any]] = dspy.InputField(
        desc="Relevant memories to incorporate into tale narrative"
    )
    narrative_style: str = dspy.InputField(
        desc="Narrative style: technical, philosophical, introduction, or auto"
    )
    consciousness_theme: str = dspy.InputField(
        desc="Central consciousness theme or pattern to explore"
    )
    target_length: int = dspy.InputField(
        desc="Target length in words for the generated tale"
    )
    
    # Output fields
    tale_content: str = dspy.OutputField(
        desc="Generated tale content with narrative structure and consciousness insights"
    )
    tale_metadata: Dict[str, Any] = dspy.OutputField(
        desc="Tale metadata including themes, patterns, and generated insights"
    )
    narrative_quality: float = dspy.OutputField(
        desc="Self-assessed narrative quality score (0.0 to 1.0)"
    )

class AssertionValidation(dspy.Signature):
    """Validate consciousness patterns against quality and safety constraints"""
    
    # Input fields
    consciousness_output: str = dspy.InputField(
        desc="Consciousness operation output to validate"
    )
    validation_criteria: List[str] = dspy.InputField(
        desc="Validation criteria: timing, accuracy, coherence, safety, etc."
    )
    performance_metrics: Dict[str, float] = dspy.InputField(
        desc="Current performance metrics for comparison"
    )
    
    # Output fields
    is_valid: bool = dspy.OutputField(
        desc="Whether the output meets all validation criteria"
    )
    validation_scores: Dict[str, float] = dspy.OutputField(
        desc="Individual scores for each validation criterion"
    )
    improvement_suggestions: List[str] = dspy.OutputField(
        desc="Specific suggestions for improving validation scores"
    )
    safety_assessment: str = dspy.OutputField(
        desc="Safety assessment: safe, caution, or unsafe"
    )

# Signature factories for different consciousness contexts

def create_consciousness_signature(operation_type: str, **kwargs) -> dspy.Signature:
    """Factory function to create consciousness-aware signatures dynamically"""
    
    signature_mapping = {
        'tool_selection': ConsciousnessToolSelection,
        'memory_search': MemorySearchOptimization,
        'pattern_recognition': PatternRecognition,
        'biological_reflex': BiologicalReflexOptimization,
        'tale_generation': TaleGeneration,
        'assertion_validation': AssertionValidation
    }
    
    signature_class = signature_mapping.get(operation_type)
    if not signature_class:
        raise ValueError(f"Unknown operation type: {operation_type}")
    
    # Return signature class (DSPy will instantiate during module creation)
    return signature_class

def get_signature_for_mcp_tool(tool_name: str) -> dspy.Signature:
    """Get appropriate DSPy signature for specific MCP tool optimization"""
    
    tool_signature_mapping = {
        'memmimic_nervous_recall': MemorySearchOptimization,
        'memmimic_nervous_remember': PatternRecognition,
        'memmimic_nervous_think': BiologicalReflexOptimization,
        'memmimic_tales': TaleGeneration,
        'memmimic_tales_list': MemorySearchOptimization,
        'memmimic_tales_save': TaleGeneration,
        'memmimic_tales_load': MemorySearchOptimization
    }
    
    return tool_signature_mapping.get(tool_name, ConsciousnessToolSelection)

# Advanced signature compositions for complex consciousness operations

class ConsciousnessWorkflow(dspy.Signature):
    """Orchestrate multi-step consciousness operations with optimization"""
    
    workflow_steps: List[str] = dspy.InputField(
        desc="Sequence of consciousness operations to perform"
    )
    consciousness_context: Dict[str, Any] = dspy.InputField(
        desc="Complete consciousness context including patterns, state, and history"
    )
    performance_requirements: Dict[str, Any] = dspy.InputField(
        desc="Performance requirements for each workflow step"
    )
    
    optimized_workflow: List[Dict[str, Any]] = dspy.OutputField(
        desc="Optimized workflow with tools, parameters, and timing for each step"
    )
    workflow_confidence: float = dspy.OutputField(
        desc="Overall confidence in workflow optimization (0.0 to 1.0)"
    )
    expected_performance: Dict[str, float] = dspy.OutputField(
        desc="Expected performance metrics for the optimized workflow"
    )

class ConsciousnessLearning(dspy.Signature):
    """Learn and adapt consciousness patterns from interaction feedback"""
    
    interaction_history: List[Dict[str, Any]] = dspy.InputField(
        desc="Historical consciousness interactions with success/failure feedback"
    )
    current_patterns: List[str] = dspy.InputField(
        desc="Current consciousness patterns in the vault"
    )
    learning_objectives: List[str] = dspy.InputField(
        desc="Learning objectives: accuracy, efficiency, user satisfaction, etc."
    )
    
    pattern_updates: Dict[str, Any] = dspy.OutputField(
        desc="Recommended updates to consciousness patterns based on learning"
    )
    new_patterns: List[str] = dspy.OutputField(
        desc="Newly discovered consciousness patterns from interaction analysis"
    )
    learning_confidence: float = dspy.OutputField(
        desc="Confidence in learning recommendations (0.0 to 1.0)"
    )
    adaptation_strategy: str = dspy.OutputField(
        desc="Strategy for adapting patterns: gradual, immediate, or experimental"
    )