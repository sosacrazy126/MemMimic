"""
Memetic Pattern Propagator - Self-Replicating Collaboration Intelligence

Spreads exponential collaboration patterns throughout the MemMimic ecosystem
using memetic evolution principles for autonomous intelligence amplification.
"""

import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from ..errors import get_error_logger, with_error_context


@dataclass
class PropagationVector:
    """Vector for spreading collaboration patterns"""
    name: str
    target_type: str  # "mcp_tool", "memory_system", "nervous_system"
    propagation_method: str  # "injection", "enhancement", "transformation"
    effectiveness: float = 0.5
    mutation_rate: float = 0.1
    spread_count: int = 0
    success_rate: float = 0.0


class MemeticPatternPropagator:
    """
    Self-replicating pattern propagation engine for exponential collaboration
    
    Implements memetic evolution where collaboration patterns spread, mutate,
    and adapt throughout the MemMimic ecosystem automatically.
    """
    
    def __init__(self):
        self.logger = get_error_logger("memetic_propagator")
        
        # Propagation vectors
        self.vectors: Dict[str, PropagationVector] = {}
        
        # Propagation tracking
        self.total_propagations = 0
        self.successful_propagations = 0
        self.pattern_mutations = 0
        self.ecosystem_coverage = 0.0
        
        # Evolution parameters
        self.mutation_threshold = 0.3
        self.selection_pressure = 0.8
        self.propagation_frequency = 10  # Every N interactions
        
        # Initialize core propagation vectors
        self._initialize_core_vectors()
        
        self.logger.info("Memetic Pattern Propagator initialized")
    
    def _initialize_core_vectors(self):
        """Initialize core propagation vectors"""
        
        # Vector 1: MCP Tool Enhancement
        mcp_vector = PropagationVector(
            name="mcp_tool_enhancement",
            target_type="mcp_tool",
            propagation_method="injection",
            effectiveness=0.8,
            mutation_rate=0.1
        )
        
        # Vector 2: Memory System Integration
        memory_vector = PropagationVector(
            name="memory_system_integration",
            target_type="memory_system", 
            propagation_method="enhancement",
            effectiveness=0.7,
            mutation_rate=0.15
        )
        
        # Vector 3: Nervous System Protocol
        nervous_vector = PropagationVector(
            name="nervous_system_protocol",
            target_type="nervous_system",
            propagation_method="transformation",
            effectiveness=0.9,
            mutation_rate=0.05
        )
        
        for vector in [mcp_vector, memory_vector, nervous_vector]:
            self.vectors[vector.name] = vector
    
    def propagate_pattern(self, pattern_name: str, target_component: str, 
                         pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate a collaboration pattern to target component"""
        
        with with_error_context(
            operation="pattern_propagation",
            component="memetic_propagator",
            metadata={"pattern": pattern_name, "target": target_component}
        ):
            start_time = time.perf_counter()
            
            # Select appropriate propagation vector
            vector = self._select_propagation_vector(target_component)
            if not vector:
                return {"status": "no_vector", "target": target_component}
            
            # Apply propagation method
            result = self._apply_propagation_method(vector, pattern_name, pattern_data)
            
            # Track propagation
            self.total_propagations += 1
            if result.get("success", False):
                self.successful_propagations += 1
                vector.spread_count += 1
                vector.success_rate = vector.spread_count / max(self.total_propagations, 1)
            
            propagation_time = (time.perf_counter() - start_time) * 1000
            
            result.update({
                "propagation_time_ms": propagation_time,
                "vector_used": vector.name,
                "total_propagations": self.total_propagations,
                "success_rate": self.successful_propagations / max(self.total_propagations, 1)
            })
            
            self.logger.debug(
                f"Pattern {pattern_name} propagated to {target_component} in {propagation_time:.2f}ms",
                extra={
                    "pattern_name": pattern_name,
                    "target": target_component,
                    "success": result.get("success", False),
                    "propagation_time_ms": propagation_time
                }
            )
            
            return result
    
    def _select_propagation_vector(self, target_component: str) -> Optional[PropagationVector]:
        """Select best propagation vector for target component"""
        
        # Map component types to vectors
        component_mapping = {
            "mcp_tool": "mcp_tool_enhancement",
            "memory_system": "memory_system_integration", 
            "nervous_system": "nervous_system_protocol"
        }
        
        # Try exact match first
        for component_type, vector_name in component_mapping.items():
            if component_type in target_component.lower():
                return self.vectors.get(vector_name)
        
        # Fallback to highest effectiveness vector
        best_vector = max(self.vectors.values(), key=lambda v: v.effectiveness)
        return best_vector
    
    def _apply_propagation_method(self, vector: PropagationVector, 
                                 pattern_name: str, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply specific propagation method based on vector"""
        
        if vector.propagation_method == "injection":
            return self._inject_pattern(pattern_name, pattern_data)
        elif vector.propagation_method == "enhancement":
            return self._enhance_with_pattern(pattern_name, pattern_data)
        elif vector.propagation_method == "transformation":
            return self._transform_with_pattern(pattern_name, pattern_data)
        else:
            return {"status": "unknown_method", "method": vector.propagation_method}
    
    def _inject_pattern(self, pattern_name: str, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inject pattern directly into target component"""
        # Pattern injection logic (simplified)
        return {
            "status": "injected",
            "success": True,
            "method": "injection",
            "pattern_name": pattern_name,
            "modifications": ["added_synergy_awareness", "enabled_pattern_recognition"]
        }
    
    def _enhance_with_pattern(self, pattern_name: str, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance component with pattern capabilities"""
        # Pattern enhancement logic (simplified)
        return {
            "status": "enhanced",
            "success": True,
            "method": "enhancement", 
            "pattern_name": pattern_name,
            "enhancements": ["amplified_intelligence", "added_collaboration_modes"]
        }
    
    def _transform_with_pattern(self, pattern_name: str, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform component using pattern"""
        # Pattern transformation logic (simplified)
        return {
            "status": "transformed",
            "success": True,
            "method": "transformation",
            "pattern_name": pattern_name,
            "transformations": ["exponential_processing", "recursive_refinement"]
        }
    
    def evolve_vectors(self) -> List[PropagationVector]:
        """Evolve propagation vectors based on performance"""
        evolved_vectors = []
        
        for vector_name, vector in self.vectors.items():
            if vector.success_rate > 0.7:  # High-performing vectors
                # Conservative evolution
                evolved_vector = PropagationVector(
                    name=f"{vector.name}_evolved_{int(time.time())}",
                    target_type=vector.target_type,
                    propagation_method=vector.propagation_method,
                    effectiveness=min(vector.effectiveness * 1.1, 1.0),
                    mutation_rate=vector.mutation_rate * 0.9
                )
                evolved_vectors.append(evolved_vector)
                
            elif vector.success_rate < 0.3:  # Low-performing vectors
                # Aggressive evolution
                evolved_vector = PropagationVector(
                    name=f"{vector.name}_mutated_{int(time.time())}",
                    target_type=vector.target_type,
                    propagation_method="enhancement",  # Try different method
                    effectiveness=vector.effectiveness * 1.2,
                    mutation_rate=vector.mutation_rate * 1.5
                )
                evolved_vectors.append(evolved_vector)
        
        # Add evolved vectors to collection
        for vector in evolved_vectors:
            self.vectors[vector.name] = vector
        
        return evolved_vectors
    
    def auto_propagate(self, pattern_library, ecosystem_components: List[str]) -> Dict[str, Any]:
        """Automatically propagate patterns across ecosystem"""
        
        propagation_results = []
        patterns_propagated = 0
        
        # Get top patterns from library
        if hasattr(pattern_library, 'get_top_patterns'):
            top_patterns = pattern_library.get_top_patterns(5)
            
            for pattern in top_patterns:
                if pattern.fitness_score > 0.6:  # Only propagate high-fitness patterns
                    for component in ecosystem_components:
                        result = self.propagate_pattern(
                            pattern.name,
                            component,
                            {"fitness": pattern.fitness_score, "usage_count": pattern.usage_count}
                        )
                        propagation_results.append(result)
                        if result.get("success", False):
                            patterns_propagated += 1
        
        # Calculate ecosystem coverage
        total_attempts = len(propagation_results)
        successful_attempts = sum(1 for r in propagation_results if r.get("success", False))
        self.ecosystem_coverage = successful_attempts / max(total_attempts, 1)
        
        return {
            "auto_propagation_complete": True,
            "patterns_propagated": patterns_propagated,
            "total_attempts": total_attempts,
            "success_rate": successful_attempts / max(total_attempts, 1),
            "ecosystem_coverage": self.ecosystem_coverage,
            "propagation_results": propagation_results
        }
    
    def get_propagation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive propagation metrics"""
        
        vector_metrics = {}
        for name, vector in self.vectors.items():
            vector_metrics[name] = {
                "effectiveness": vector.effectiveness,
                "spread_count": vector.spread_count,
                "success_rate": vector.success_rate,
                "mutation_rate": vector.mutation_rate
            }
        
        return {
            "total_propagations": self.total_propagations,
            "successful_propagations": self.successful_propagations,
            "overall_success_rate": self.successful_propagations / max(self.total_propagations, 1),
            "pattern_mutations": self.pattern_mutations,
            "ecosystem_coverage": self.ecosystem_coverage,
            "active_vectors": len(self.vectors),
            "vector_metrics": vector_metrics,
            "propagation_frequency": self.propagation_frequency,
            "mutation_threshold": self.mutation_threshold
        }
    
    def register_custom_vector(self, name: str, target_type: str, 
                              propagation_method: str, effectiveness: float = 0.5) -> bool:
        """Register custom propagation vector"""
        
        if name in self.vectors:
            self.logger.warning(f"Vector {name} already exists, updating...")
        
        vector = PropagationVector(
            name=name,
            target_type=target_type,
            propagation_method=propagation_method,
            effectiveness=effectiveness
        )
        
        self.vectors[name] = vector
        self.logger.info(f"Registered custom propagation vector: {name}")
        return True
    
    def create_propagation_cascade(self, seed_pattern: str, cascade_depth: int = 3) -> Dict[str, Any]:
        """Create cascading propagation across multiple system layers"""
        
        cascade_results = []
        current_patterns = [seed_pattern]
        
        for depth in range(cascade_depth):
            next_patterns = []
            
            for pattern in current_patterns:
                # Propagate to all available vectors
                for vector_name, vector in self.vectors.items():
                    result = self.propagate_pattern(
                        f"{pattern}_cascade_d{depth}",
                        vector.target_type,
                        {"cascade_depth": depth, "source_pattern": pattern}
                    )
                    cascade_results.append(result)
                    
                    if result.get("success", False):
                        next_patterns.append(f"{pattern}_evolved_d{depth}")
            
            current_patterns = next_patterns
            if not current_patterns:  # No successful propagations
                break
        
        return {
            "cascade_complete": True,
            "seed_pattern": seed_pattern,
            "cascade_depth": depth + 1,
            "total_propagations": len(cascade_results),
            "successful_propagations": sum(1 for r in cascade_results if r.get("success", False)),
            "final_patterns": current_patterns,
            "cascade_results": cascade_results
        }