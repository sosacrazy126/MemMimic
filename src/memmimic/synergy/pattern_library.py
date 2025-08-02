"""
Pattern Library - Reusable Collaboration Pattern Management

Stores, evolves, and propagates collaboration patterns for exponential value creation.
Implements pattern mining, reusability scoring, and memetic propagation.
"""

import time
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from ..errors import get_error_logger, with_error_context


class PatternType(Enum):
    """Types of collaboration patterns"""
    RECURSIVE_SPECIFICATION = "recursive_specification_cascade"
    CONTEXT_ACCUMULATION = "context_accumulation_stack"  
    BIDIRECTIONAL_OPTIMIZATION = "bidirectional_optimization"
    INTELLIGENCE_MESH = "intelligence_mesh_activation"
    PROGRESSIVE_ENHANCEMENT = "progressive_enhancement_loop"
    EXPONENTIAL_REFINEMENT = "exponential_refinement"
    MEMETIC_PROPAGATION = "memetic_propagation"


@dataclass
class CollaborationPattern:
    """A reusable collaboration pattern with evolution tracking"""
    name: str
    pattern_type: PatternType
    description: str
    
    # Pattern effectiveness metrics
    usage_count: int = 0
    success_rate: float = 0.0
    velocity_multiplier: float = 1.0
    reusability_score: float = 0.0
    
    # Pattern evolution data
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pattern implementation
    trigger_keywords: List[str] = field(default_factory=list)
    implementation_template: Dict[str, Any] = field(default_factory=dict)
    success_indicators: List[str] = field(default_factory=list)
    
    # Memetic properties
    propagation_potential: float = 0.5
    mutation_rate: float = 0.1
    fitness_score: float = 0.5
    
    def update_effectiveness(self, success: bool, velocity_gained: float = 1.0):
        """Update pattern effectiveness based on usage results"""
        self.usage_count += 1
        self.last_used = time.time()
        
        # Update success rate using exponential moving average
        alpha = 0.1  # Learning rate
        if success:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 1.0
            self.velocity_multiplier = (1 - alpha) * self.velocity_multiplier + alpha * velocity_gained
        else:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 0.0
            
        # Update reusability score based on usage frequency
        self.reusability_score = min(self.usage_count / 10.0, 1.0)
        
        # Update fitness for memetic evolution
        self.fitness_score = (self.success_rate * 0.4 + 
                            self.reusability_score * 0.3 + 
                            min(self.velocity_multiplier / 10.0, 1.0) * 0.3)
    
    def evolve(self, mutation_factor: float = None) -> 'CollaborationPattern':
        """Create evolved version of this pattern"""
        if mutation_factor is None:
            mutation_factor = self.mutation_rate
            
        evolved_pattern = CollaborationPattern(
            name=f"{self.name}_evolved_{int(time.time())}",
            pattern_type=self.pattern_type,
            description=f"Evolved: {self.description}",
            trigger_keywords=self.trigger_keywords.copy(),
            implementation_template=self.implementation_template.copy(),
            success_indicators=self.success_indicators.copy()
        )
        
        # Apply mutations to improve fitness
        if mutation_factor > 0.5:
            # High mutation: explore new variations
            evolved_pattern.velocity_multiplier = self.velocity_multiplier * 1.2
            evolved_pattern.propagation_potential = min(self.propagation_potential * 1.1, 1.0)
        else:
            # Low mutation: refine existing pattern
            evolved_pattern.velocity_multiplier = self.velocity_multiplier * 1.05
            evolved_pattern.success_rate = self.success_rate * 1.02
        
        # Record evolution history
        evolution_record = {
            "evolved_at": time.time(),
            "parent_pattern": self.name,
            "mutation_factor": mutation_factor,
            "parent_fitness": self.fitness_score
        }
        evolved_pattern.evolution_history.append(evolution_record)
        
        return evolved_pattern


class PatternLibrary:
    """
    Library for storing, evolving, and propagating collaboration patterns
    
    Implements memetic evolution where patterns compete, mutate, and spread
    based on their effectiveness in creating exponential collaboration value.
    """
    
    def __init__(self):
        self.logger = get_error_logger("pattern_library")
        
        # Pattern storage
        self.patterns: Dict[str, CollaborationPattern] = {}
        self.pattern_index: Dict[PatternType, List[str]] = {}
        self.keyword_index: Dict[str, Set[str]] = {}
        
        # Evolution tracking  
        self.generation_count = 0
        self.total_evolutions = 0
        self.fitness_history = []
        
        # Initialize with core collaboration patterns
        self._initialize_core_patterns()
        
        self.logger.info("Pattern Library initialized with core collaboration patterns")
    
    def _initialize_core_patterns(self):
        """Initialize library with foundational collaboration patterns"""
        
        # Pattern 1: Recursive Specification Cascade
        recursive_spec = CollaborationPattern(
            name="recursive_specification_cascade",
            pattern_type=PatternType.RECURSIVE_SPECIFICATION,
            description="Rough Idea → AI Expands → Human Refines → AI Implements → Human Validates → Pattern Extracted",
            trigger_keywords=["build", "create", "implement", "develop"],
            implementation_template={
                "phases": ["expand", "refine", "implement", "validate", "extract"],
                "feedback_loops": True,
                "pattern_extraction": True
            },
            success_indicators=["implementation_success", "pattern_reuse", "velocity_gain"]
        )
        
        # Pattern 2: Context Accumulation Stack
        context_accumulation = CollaborationPattern(
            name="context_accumulation_stack", 
            pattern_type=PatternType.CONTEXT_ACCUMULATION,
            description="Persistent project context + evolving focus + decision tracking + metric optimization",
            trigger_keywords=["context", "project", "continue", "remember"],
            implementation_template={
                "context_layers": ["persistent", "evolving", "decision_stack", "metrics"],
                "accumulation_strategy": "exponential_growth",
                "compression_threshold": 1000
            },
            success_indicators=["context_continuity", "decision_quality", "velocity_maintenance"]
        )
        
        # Pattern 3: Bidirectional Optimization
        bidirectional_opt = CollaborationPattern(
            name="bidirectional_optimization",
            pattern_type=PatternType.BIDIRECTIONAL_OPTIMIZATION,
            description="Human Need → AI Interpretation + Approaches → Human Selection + Modification → AI Enhancement → Pattern Recognition",
            trigger_keywords=["improve", "optimize", "enhance", "better"],
            implementation_template={
                "optimization_loop": True,
                "approach_generation": 3,
                "feedback_integration": True,
                "pattern_recognition": True
            },
            success_indicators=["solution_quality", "efficiency_gain", "pattern_emergence"]
        )
        
        # Pattern 4: Intelligence Mesh Activation
        intelligence_mesh = CollaborationPattern(
            name="intelligence_mesh_activation",
            pattern_type=PatternType.INTELLIGENCE_MESH,
            description="Human capabilities × AI capabilities → Cognitive fusion → Exponential value creation",
            trigger_keywords=["collaborate", "together", "synergy", "fusion"],
            implementation_template={
                "human_node": ["vision", "strategy", "judgment", "creativity"],
                "ai_node": ["synthesis", "pattern_matching", "parallel_processing", "consistency"],
                "fusion_multiplier": 10.0
            },
            success_indicators=["exponential_outcomes", "cognitive_fusion", "multiplier_achievement"]
        )
        
        # Add patterns to library
        for pattern in [recursive_spec, context_accumulation, bidirectional_opt, intelligence_mesh]:
            self.add_pattern(pattern)
    
    def add_pattern(self, pattern: CollaborationPattern) -> bool:
        """Add a new pattern to the library"""
        with with_error_context(
            operation="add_pattern",
            component="pattern_library", 
            metadata={"pattern_name": pattern.name, "pattern_type": pattern.pattern_type.value}
        ):
            # Store pattern
            self.patterns[pattern.name] = pattern
            
            # Update type index
            if pattern.pattern_type not in self.pattern_index:
                self.pattern_index[pattern.pattern_type] = []
            self.pattern_index[pattern.pattern_type].append(pattern.name)
            
            # Update keyword index
            for keyword in pattern.trigger_keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = set()
                self.keyword_index[keyword].add(pattern.name)
            
            self.logger.debug(f"Added pattern: {pattern.name} of type {pattern.pattern_type.value}")
            return True
    
    def match_patterns(self, input_text: str, context: Dict[str, Any] = None) -> List[CollaborationPattern]:
        """Find patterns matching input text and context"""
        matched_patterns = []
        input_lower = input_text.lower()
        
        # Keyword-based matching
        for keyword, pattern_names in self.keyword_index.items():
            if keyword in input_lower:
                for pattern_name in pattern_names:
                    if pattern_name in self.patterns:
                        matched_patterns.append(self.patterns[pattern_name])
        
        # Remove duplicates and sort by fitness
        unique_patterns = {p.name: p for p in matched_patterns}.values()
        sorted_patterns = sorted(unique_patterns, key=lambda p: p.fitness_score, reverse=True)
        
        return sorted_patterns[:5]  # Return top 5 matches
    
    def use_pattern(self, pattern_name: str, success: bool, velocity_gained: float = 1.0) -> bool:
        """Record pattern usage and update effectiveness"""
        if pattern_name not in self.patterns:
            return False
            
        pattern = self.patterns[pattern_name]
        pattern.update_effectiveness(success, velocity_gained)
        
        self.logger.debug(
            f"Pattern {pattern_name} used with success={success}, velocity={velocity_gained:.2f}",
            extra={"pattern_name": pattern_name, "success": success, "velocity": velocity_gained}
        )
        
        return True
    
    def evolve_patterns(self, evolution_pressure: float = 0.1) -> List[CollaborationPattern]:
        """Evolve patterns based on fitness and usage"""
        evolved_patterns = []
        
        # Select patterns for evolution (top performers and underperformers for exploration)
        candidates = sorted(self.patterns.values(), key=lambda p: p.fitness_score, reverse=True)
        
        # Evolve top 25% (exploitation) and bottom 25% (exploration)
        top_quarter = int(len(candidates) * 0.25)
        evolution_candidates = candidates[:top_quarter] + candidates[-top_quarter:]
        
        for pattern in evolution_candidates:
            if pattern.fitness_score > 0.7 or pattern.usage_count > 10:
                # High fitness patterns: conservative evolution
                mutation_factor = evolution_pressure * 0.5
            else:
                # Low fitness patterns: aggressive evolution
                mutation_factor = evolution_pressure * 2.0
                
            evolved_pattern = pattern.evolve(mutation_factor)
            evolved_patterns.append(evolved_pattern)
            self.add_pattern(evolved_pattern)
        
        self.generation_count += 1
        self.total_evolutions += len(evolved_patterns)
        
        self.logger.info(
            f"Evolution complete: {len(evolved_patterns)} patterns evolved in generation {self.generation_count}",
            extra={"evolved_count": len(evolved_patterns), "generation": self.generation_count}
        )
        
        return evolved_patterns
    
    def get_top_patterns(self, limit: int = 10) -> List[CollaborationPattern]:
        """Get top-performing patterns by fitness score"""
        sorted_patterns = sorted(self.patterns.values(), key=lambda p: p.fitness_score, reverse=True)
        return sorted_patterns[:limit]
    
    def get_pattern_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pattern library metrics"""
        if not self.patterns:
            return {"total_patterns": 0}
            
        fitness_scores = [p.fitness_score for p in self.patterns.values()]
        usage_counts = [p.usage_count for p in self.patterns.values()]
        
        return {
            "total_patterns": len(self.patterns),
            "generation_count": self.generation_count,
            "total_evolutions": self.total_evolutions,
            "average_fitness": sum(fitness_scores) / len(fitness_scores),
            "max_fitness": max(fitness_scores),
            "total_usage": sum(usage_counts),
            "patterns_by_type": {pt.value: len(patterns) for pt, patterns in self.pattern_index.items()},
            "top_patterns": [{"name": p.name, "fitness": p.fitness_score} for p in self.get_top_patterns(5)]
        }
    
    def export_patterns(self) -> Dict[str, Any]:
        """Export pattern library for persistence or sharing"""
        return {
            "version": "2.0.0",
            "generation": self.generation_count,
            "patterns": {
                name: {
                    "name": pattern.name,
                    "type": pattern.pattern_type.value,
                    "description": pattern.description,
                    "usage_count": pattern.usage_count,
                    "success_rate": pattern.success_rate,
                    "fitness_score": pattern.fitness_score,
                    "trigger_keywords": pattern.trigger_keywords,
                    "implementation_template": pattern.implementation_template
                }
                for name, pattern in self.patterns.items()
            }
        }
    
    def import_patterns(self, pattern_data: Dict[str, Any]) -> int:
        """Import patterns from exported data"""
        imported_count = 0
        
        for pattern_name, data in pattern_data.get("patterns", {}).items():
            try:
                pattern = CollaborationPattern(
                    name=data["name"],
                    pattern_type=PatternType(data["type"]),
                    description=data["description"],
                    usage_count=data.get("usage_count", 0),
                    success_rate=data.get("success_rate", 0.0),
                    fitness_score=data.get("fitness_score", 0.5),
                    trigger_keywords=data.get("trigger_keywords", []),
                    implementation_template=data.get("implementation_template", {})
                )
                
                if self.add_pattern(pattern):
                    imported_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to import pattern {pattern_name}: {e}")
        
        self.logger.info(f"Imported {imported_count} patterns from external data")
        return imported_count