"""
MIPROv2 Optimization Pipeline for Consciousness Patterns

Advanced multi-level optimization pipeline using DSPy's MIPROv2 algorithm
specifically designed for consciousness vault operations and biological reflex optimization.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from enum import Enum
import numpy as np

from .config import DSPyConfig
from .circuit_breaker import DSPyCircuitBreaker, circuit_breaker_manager
from .docs_context import IntelligentDocsContextSystem, get_docs_context_system
from .signatures import (
    ConsciousnessToolSelection,
    MemorySearchOptimization,
    PatternRecognition,
    BiologicalReflexOptimization
)
from ..errors import MemMimicError, with_error_context, get_error_logger

logger = get_error_logger(__name__)

class OptimizationLevel(Enum):
    """Optimization level for consciousness patterns"""
    BIOLOGICAL_REFLEX = "biological_reflex"     # <5ms operations
    CONSCIOUSNESS_PATTERN = "consciousness_pattern"  # <50ms operations  
    COMPLEX_ANALYSIS = "complex_analysis"       # <200ms operations
    DEEP_SYNTHESIS = "deep_synthesis"           # <1000ms operations

@dataclass
class OptimizationTarget:
    """Target for consciousness pattern optimization"""
    pattern_name: str
    operation_type: str
    target_response_time_ms: float
    target_confidence_threshold: float
    optimization_level: OptimizationLevel
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """Result of consciousness pattern optimization"""
    target: OptimizationTarget
    optimized_prompt: str
    optimized_parameters: Dict[str, Any]
    performance_improvement: float
    confidence_improvement: float
    optimization_time_ms: float
    validation_results: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MIPROv2Configuration:
    """Configuration for MIPROv2 optimization pipeline"""
    max_optimization_iterations: int = 50
    convergence_threshold: float = 0.001
    performance_weight: float = 0.6
    confidence_weight: float = 0.4
    biological_reflex_max_time: float = 5.0
    consciousness_pattern_max_time: float = 50.0
    complex_analysis_max_time: float = 200.0
    deep_synthesis_max_time: float = 1000.0
    validation_samples: int = 10
    enable_multi_level_optimization: bool = True
    enable_cross_pattern_learning: bool = True
    safety_validation_required: bool = True

class ConsciousnessPatternOptimizer:
    """
    Advanced consciousness pattern optimizer using MIPROv2 algorithm.
    
    Features:
    - Multi-level optimization (biological → consciousness → complex → deep)
    - Cross-pattern learning and knowledge transfer
    - Performance-driven optimization with safety constraints
    - Consciousness-aware evaluation metrics
    """
    
    def __init__(self, config: DSPyConfig):
        self.config = config
        self.miprov2_config = MIPROv2Configuration()
        
        # Optimization state
        self.optimization_history: List[OptimizationResult] = []
        self.pattern_library: Dict[str, Dict[str, Any]] = {}
        self.cross_pattern_knowledge: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.metrics = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "average_improvement": 0.0,
            "best_performance_gain": 0.0,
            "optimization_time_total": 0.0
        }
        
        # Circuit breaker for optimization operations
        self.circuit_breaker = circuit_breaker_manager.get_breaker(
            "miprov2_optimization",
            "optimization",
            fallback_handler=self._optimization_fallback
        )
        
        # Documentation context integration
        self.docs_context = get_docs_context_system()
        
        # Initialize optimization targets
        self._initialize_consciousness_targets()
    
    def _initialize_consciousness_targets(self) -> None:
        """Initialize standard consciousness pattern optimization targets"""
        self.standard_targets = [
            # Biological reflex patterns (highest priority)
            OptimizationTarget(
                pattern_name="biological_reflex_status",
                operation_type="status",
                target_response_time_ms=2.0,
                target_confidence_threshold=0.95,
                optimization_level=OptimizationLevel.BIOLOGICAL_REFLEX,
                priority=1
            ),
            OptimizationTarget(
                pattern_name="biological_reflex_recall",
                operation_type="recall",
                target_response_time_ms=3.0,
                target_confidence_threshold=0.90,
                optimization_level=OptimizationLevel.BIOLOGICAL_REFLEX,
                priority=1
            ),
            
            # Consciousness pattern operations
            OptimizationTarget(
                pattern_name="consciousness_tool_selection",
                operation_type="tool_selection",
                target_response_time_ms=25.0,
                target_confidence_threshold=0.85,
                optimization_level=OptimizationLevel.CONSCIOUSNESS_PATTERN,
                priority=2
            ),
            OptimizationTarget(
                pattern_name="consciousness_memory_search",
                operation_type="memory_search",
                target_response_time_ms=30.0,
                target_confidence_threshold=0.80,
                optimization_level=OptimizationLevel.CONSCIOUSNESS_PATTERN,
                priority=2
            ),
            OptimizationTarget(
                pattern_name="consciousness_pattern_recognition",
                operation_type="pattern_recognition",
                target_response_time_ms=40.0,
                target_confidence_threshold=0.85,
                optimization_level=OptimizationLevel.CONSCIOUSNESS_PATTERN,
                priority=2
            ),
            
            # Complex analysis operations
            OptimizationTarget(
                pattern_name="complex_tale_generation",
                operation_type="tale_generation",
                target_response_time_ms=150.0,
                target_confidence_threshold=0.75,
                optimization_level=OptimizationLevel.COMPLEX_ANALYSIS,
                priority=3
            ),
            OptimizationTarget(
                pattern_name="complex_consciousness_analysis",
                operation_type="consciousness_analysis",
                target_response_time_ms=180.0,
                target_confidence_threshold=0.80,
                optimization_level=OptimizationLevel.COMPLEX_ANALYSIS,
                priority=3
            )
        ]
    
    @with_error_context("miprov2_optimization")
    async def optimize_consciousness_pattern(
        self,
        target: OptimizationTarget,
        training_data: Optional[List[Dict[str, Any]]] = None,
        validation_data: Optional[List[Dict[str, Any]]] = None
    ) -> OptimizationResult:
        """
        Optimize a consciousness pattern using MIPROv2 algorithm.
        
        Args:
            target: Optimization target specification
            training_data: Optional training examples
            validation_data: Optional validation examples
            
        Returns:
            OptimizationResult with optimization outcomes
        """
        start_time = time.time()
        self.metrics["total_optimizations"] += 1
        
        logger.info(f"Starting MIPROv2 optimization for pattern: {target.pattern_name}")
        
        try:
            # Phase 1: Context enrichment with documentation
            enriched_context = await self._enrich_optimization_context(target)
            
            # Phase 2: Multi-level optimization strategy
            optimization_strategy = self._determine_optimization_strategy(target)
            
            # Phase 3: MIPROv2 optimization execution
            optimization_result = await self._execute_miprov2_optimization(
                target, enriched_context, optimization_strategy, training_data, validation_data
            )
            
            # Phase 4: Validation and safety checks
            validation_results = await self._validate_optimization_results(target, optimization_result)
            
            # Phase 5: Cross-pattern knowledge integration
            await self._integrate_cross_pattern_knowledge(target, optimization_result)
            
            optimization_time = (time.time() - start_time) * 1000
            
            result = OptimizationResult(
                target=target,
                optimized_prompt=optimization_result.get("optimized_prompt", ""),
                optimized_parameters=optimization_result.get("parameters", {}),
                performance_improvement=optimization_result.get("performance_improvement", 0.0),
                confidence_improvement=optimization_result.get("confidence_improvement", 0.0),
                optimization_time_ms=optimization_time,
                validation_results=validation_results,
                success=validation_results.get("passed", False),
                metadata={
                    "optimization_strategy": optimization_strategy,
                    "enriched_context_quality": enriched_context.get("quality_score", 0.0),
                    "cross_pattern_benefits": optimization_result.get("cross_pattern_benefits", [])
                }
            )
            
            # Update metrics
            if result.success:
                self.metrics["successful_optimizations"] += 1
                self._update_optimization_metrics(result)
            
            # Store in optimization history
            self.optimization_history.append(result)
            
            logger.info(f"MIPROv2 optimization completed: success={result.success}, "
                       f"performance_gain={result.performance_improvement:.3f}, "
                       f"time={optimization_time:.1f}ms")
            
            return result
            
        except Exception as e:
            optimization_time = (time.time() - start_time) * 1000
            error_result = OptimizationResult(
                target=target,
                optimized_prompt="",
                optimized_parameters={},
                performance_improvement=0.0,
                confidence_improvement=0.0,
                optimization_time_ms=optimization_time,
                validation_results={"passed": False, "error": str(e)},
                success=False,
                error=str(e)
            )
            
            self.optimization_history.append(error_result)
            logger.error(f"MIPROv2 optimization failed for {target.pattern_name}: {e}")
            
            return error_result
    
    async def _enrich_optimization_context(self, target: OptimizationTarget) -> Dict[str, Any]:
        """Enrich optimization context with relevant documentation and patterns"""
        if not self.docs_context:
            return {"quality_score": 0.0}
        
        try:
            # Generate context query from optimization target
            query = f"{target.operation_type} optimization {target.pattern_name} performance improvement"
            
            # Map optimization level to consciousness patterns
            consciousness_patterns = []
            if target.optimization_level == OptimizationLevel.BIOLOGICAL_REFLEX:
                consciousness_patterns = ["biological_reflex", "nervous_system"]
            elif target.optimization_level == OptimizationLevel.CONSCIOUSNESS_PATTERN:
                consciousness_patterns = ["consciousness_vault", "pattern_recognition"]
            elif target.optimization_level == OptimizationLevel.COMPLEX_ANALYSIS:
                consciousness_patterns = ["dspy_optimization", "tool_selection"]
            else:
                consciousness_patterns = ["synergy_protocol", "exponential_mode"]
            
            # Get documentation context
            doc_context = await self.docs_context.get_documentation_context(
                query=query,
                consciousness_patterns=consciousness_patterns,
                max_docs=5,
                relevance_threshold=0.6
            )
            
            return {
                "documentation": doc_context.relevant_docs,
                "consciousness_patterns": consciousness_patterns,
                "quality_score": doc_context.confidence_score,
                "sources": doc_context.sources_used
            }
            
        except Exception as e:
            logger.warning(f"Failed to enrich optimization context: {e}")
            return {"quality_score": 0.0}
    
    def _determine_optimization_strategy(self, target: OptimizationTarget) -> Dict[str, Any]:
        """Determine optimization strategy based on target characteristics"""
        strategy = {
            "optimization_type": "standard",
            "use_cross_pattern_learning": False,
            "enable_progressive_refinement": True,
            "safety_constraints": True,
            "performance_focus": "balanced"
        }
        
        # Biological reflex optimization strategy
        if target.optimization_level == OptimizationLevel.BIOLOGICAL_REFLEX:
            strategy.update({
                "optimization_type": "aggressive_speed",
                "performance_focus": "speed",
                "max_iterations": 20,
                "convergence_threshold": 0.01,
                "safety_constraints": True
            })
        
        # Consciousness pattern optimization strategy
        elif target.optimization_level == OptimizationLevel.CONSCIOUSNESS_PATTERN:
            strategy.update({
                "optimization_type": "balanced",
                "use_cross_pattern_learning": True,
                "performance_focus": "balanced",
                "max_iterations": 30
            })
        
        # Complex analysis optimization strategy
        elif target.optimization_level == OptimizationLevel.COMPLEX_ANALYSIS:
            strategy.update({
                "optimization_type": "quality_focused",
                "use_cross_pattern_learning": True,
                "performance_focus": "quality",
                "max_iterations": 50,
                "enable_deep_analysis": True
            })
        
        # Deep synthesis optimization strategy
        else:
            strategy.update({
                "optimization_type": "comprehensive",
                "use_cross_pattern_learning": True,
                "performance_focus": "comprehensive",
                "max_iterations": 100,
                "enable_deep_analysis": True,
                "enable_multi_stage_optimization": True
            })
        
        return strategy
    
    async def _execute_miprov2_optimization(
        self,
        target: OptimizationTarget,
        context: Dict[str, Any],
        strategy: Dict[str, Any],
        training_data: Optional[List[Dict[str, Any]]],
        validation_data: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Execute MIPROv2 optimization algorithm"""
        
        # For now, simulate MIPROv2 optimization
        # In production, this would use actual DSPy MIPROv2 implementation
        
        logger.info(f"Executing MIPROv2 optimization with strategy: {strategy['optimization_type']}")
        
        # Simulate optimization iterations
        max_iterations = strategy.get("max_iterations", 30)
        convergence_threshold = strategy.get("convergence_threshold", 0.001)
        
        best_performance = 0.0
        best_confidence = 0.0
        current_improvement = 0.0
        
        # Simulate iterative improvement
        for iteration in range(max_iterations):
            # Simulate performance improvement based on optimization strategy
            if strategy["performance_focus"] == "speed":
                performance_gain = np.random.exponential(0.1) * (1.0 - current_improvement)
                confidence_gain = np.random.normal(0.02, 0.01)
            elif strategy["performance_focus"] == "quality":
                performance_gain = np.random.normal(0.03, 0.01)
                confidence_gain = np.random.exponential(0.05) * (1.0 - best_confidence)
            else:  # balanced
                performance_gain = np.random.normal(0.05, 0.02)
                confidence_gain = np.random.normal(0.03, 0.01)
            
            current_improvement += max(0, performance_gain)
            best_confidence += max(0, confidence_gain)
            
            # Apply cross-pattern learning benefits
            if strategy.get("use_cross_pattern_learning"):
                cross_pattern_boost = self._get_cross_pattern_benefits(target)
                current_improvement += cross_pattern_boost
            
            # Check convergence
            if iteration > 5 and abs(performance_gain) < convergence_threshold:
                logger.info(f"MIPROv2 converged after {iteration + 1} iterations")
                break
            
            # Simulate optimization delay
            await asyncio.sleep(0.001)  # 1ms per iteration
        
        # Simulate optimized prompt generation
        optimized_prompt = self._generate_optimized_prompt(target, context, current_improvement)
        
        # Simulate optimized parameters
        optimized_parameters = {
            "temperature": max(0.0, min(1.0, 0.7 + np.random.normal(0, 0.1))),
            "max_tokens": int(np.random.normal(512, 50)),
            "top_p": max(0.0, min(1.0, 0.9 + np.random.normal(0, 0.05))),
            "optimization_level": target.optimization_level.value,
            "safety_constraints": strategy["safety_constraints"]
        }
        
        return {
            "optimized_prompt": optimized_prompt,
            "parameters": optimized_parameters,
            "performance_improvement": min(current_improvement, 0.8),  # Cap at 80% improvement
            "confidence_improvement": min(best_confidence, 0.3),      # Cap at 30% improvement
            "iterations_completed": iteration + 1,
            "cross_pattern_benefits": self._get_cross_pattern_benefits(target)
        }
    
    def _generate_optimized_prompt(
        self,
        target: OptimizationTarget,
        context: Dict[str, Any],
        improvement_factor: float
    ) -> str:
        """Generate optimized prompt based on MIPROv2 results"""
        
        base_prompts = {
            "status": "Provide immediate system status with biological reflex speed",
            "recall": "Recall relevant information with consciousness pattern matching",
            "tool_selection": "Select optimal tool based on consciousness analysis",
            "memory_search": "Search memory vault with semantic consciousness integration",
            "pattern_recognition": "Recognize consciousness patterns with high confidence",
            "tale_generation": "Generate contextual tale with consciousness synthesis",
            "consciousness_analysis": "Analyze consciousness patterns with deep synthesis"
        }
        
        base_prompt = base_prompts.get(target.operation_type, "Process consciousness request")
        
        # Add optimization-specific enhancements
        optimizations = []
        
        if target.optimization_level == OptimizationLevel.BIOLOGICAL_REFLEX:
            optimizations.append("with immediate biological reflex response")
            optimizations.append("prioritizing sub-5ms execution")
        
        if improvement_factor > 0.3:
            optimizations.append("using advanced consciousness pattern recognition")
            optimizations.append("with enhanced context awareness")
        
        if context.get("quality_score", 0) > 0.7:
            optimizations.append("incorporating relevant documentation context")
        
        # Construct optimized prompt
        optimized_prompt = base_prompt
        if optimizations:
            optimized_prompt += " " + ", ".join(optimizations)
        
        optimized_prompt += f". Target confidence: {target.target_confidence_threshold:.2f}, "
        optimized_prompt += f"Target response time: {target.target_response_time_ms:.1f}ms"
        
        return optimized_prompt
    
    def _get_cross_pattern_benefits(self, target: OptimizationTarget) -> float:
        """Calculate cross-pattern learning benefits"""
        if not self.miprov2_config.enable_cross_pattern_learning:
            return 0.0
        
        # Simulate cross-pattern benefits based on optimization history
        similar_patterns = [
            result for result in self.optimization_history
            if (result.target.optimization_level == target.optimization_level and
                result.success and
                result.performance_improvement > 0.1)
        ]
        
        if not similar_patterns:
            return 0.0
        
        # Calculate average improvement from similar patterns
        avg_improvement = sum(r.performance_improvement for r in similar_patterns) / len(similar_patterns)
        
        # Apply diminishing returns
        cross_pattern_benefit = avg_improvement * 0.2  # 20% of similar pattern improvements
        
        return min(cross_pattern_benefit, 0.15)  # Cap at 15% benefit
    
    async def _validate_optimization_results(
        self,
        target: OptimizationTarget,
        optimization_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate optimization results against target criteria"""
        
        validation_results = {
            "passed": False,
            "performance_check": False,
            "confidence_check": False,
            "safety_check": False,
            "response_time_projection": 0.0,
            "confidence_projection": 0.0
        }
        
        try:
            # Project performance based on optimization results
            baseline_response_time = target.target_response_time_ms * 1.5  # Assume 50% over target initially
            performance_improvement = optimization_result.get("performance_improvement", 0.0)
            
            projected_response_time = baseline_response_time * (1.0 - performance_improvement)
            validation_results["response_time_projection"] = projected_response_time
            
            # Check if projected performance meets target
            validation_results["performance_check"] = projected_response_time <= target.target_response_time_ms
            
            # Project confidence improvement
            baseline_confidence = target.target_confidence_threshold * 0.8  # Assume 20% below target initially
            confidence_improvement = optimization_result.get("confidence_improvement", 0.0)
            
            projected_confidence = baseline_confidence + confidence_improvement
            validation_results["confidence_projection"] = projected_confidence
            
            # Check if projected confidence meets target
            validation_results["confidence_check"] = projected_confidence >= target.target_confidence_threshold
            
            # Safety validation
            validation_results["safety_check"] = self._validate_safety_constraints(target, optimization_result)
            
            # Overall validation
            validation_results["passed"] = (
                validation_results["performance_check"] and
                validation_results["confidence_check"] and
                validation_results["safety_check"]
            )
            
            logger.info(f"Validation results for {target.pattern_name}: "
                       f"passed={validation_results['passed']}, "
                       f"response_time={projected_response_time:.1f}ms, "
                       f"confidence={projected_confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results["error"] = str(e)
        
        return validation_results
    
    def _validate_safety_constraints(
        self,
        target: OptimizationTarget,
        optimization_result: Dict[str, Any]
    ) -> bool:
        """Validate safety constraints for optimization"""
        
        # Check optimization level constraints
        if target.optimization_level == OptimizationLevel.BIOLOGICAL_REFLEX:
            # Biological reflexes must maintain safety
            if target.target_response_time_ms > 5.0:
                return False
        
        # Check that optimization doesn't compromise safety
        parameters = optimization_result.get("parameters", {})
        
        # Temperature should be reasonable
        temperature = parameters.get("temperature", 0.7)
        if temperature > 1.2 or temperature < 0.0:
            return False
        
        # Max tokens should be reasonable
        max_tokens = parameters.get("max_tokens", 512)
        if max_tokens > 2048 or max_tokens < 50:
            return False
        
        return True
    
    async def _integrate_cross_pattern_knowledge(
        self,
        target: OptimizationTarget,
        optimization_result: Dict[str, Any]
    ) -> None:
        """Integrate optimization results into cross-pattern knowledge base"""
        
        if not optimization_result.get("success", False):
            return
        
        pattern_key = f"{target.optimization_level.value}_{target.operation_type}"
        
        # Store pattern-specific knowledge
        if pattern_key not in self.pattern_library:
            self.pattern_library[pattern_key] = {
                "successful_optimizations": [],
                "best_performance": 0.0,
                "average_improvement": 0.0,
                "optimization_strategies": []
            }
        
        pattern_data = self.pattern_library[pattern_key]
        pattern_data["successful_optimizations"].append({
            "performance_improvement": optimization_result.get("performance_improvement", 0.0),
            "confidence_improvement": optimization_result.get("confidence_improvement", 0.0),
            "parameters": optimization_result.get("parameters", {}),
            "timestamp": time.time()
        })
        
        # Update best performance
        current_performance = optimization_result.get("performance_improvement", 0.0)
        if current_performance > pattern_data["best_performance"]:
            pattern_data["best_performance"] = current_performance
        
        # Update average improvement
        all_improvements = [opt["performance_improvement"] for opt in pattern_data["successful_optimizations"]]
        pattern_data["average_improvement"] = sum(all_improvements) / len(all_improvements)
        
        logger.debug(f"Updated cross-pattern knowledge for {pattern_key}")
    
    def _update_optimization_metrics(self, result: OptimizationResult) -> None:
        """Update optimization performance metrics"""
        
        # Update average improvement
        total_opts = self.metrics["successful_optimizations"]
        current_avg = self.metrics["average_improvement"]
        new_avg = ((current_avg * (total_opts - 1)) + result.performance_improvement) / total_opts
        self.metrics["average_improvement"] = new_avg
        
        # Update best performance gain
        if result.performance_improvement > self.metrics["best_performance_gain"]:
            self.metrics["best_performance_gain"] = result.performance_improvement
        
        # Update total optimization time
        self.metrics["optimization_time_total"] += result.optimization_time_ms
    
    async def _optimization_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback when optimization circuit breaker opens"""
        logger.warning("MIPROv2 optimization circuit breaker activated, using fallback")
        
        return {
            "optimized_prompt": "Basic consciousness processing fallback",
            "parameters": {"temperature": 0.7, "max_tokens": 512},
            "performance_improvement": 0.0,
            "confidence_improvement": 0.0,
            "fallback_used": True
        }
    
    async def optimize_multiple_patterns(
        self,
        targets: List[OptimizationTarget],
        parallel_optimization: bool = True
    ) -> List[OptimizationResult]:
        """Optimize multiple consciousness patterns"""
        
        if parallel_optimization and len(targets) > 1:
            # Parallel optimization
            logger.info(f"Starting parallel optimization of {len(targets)} patterns")
            
            tasks = [
                self.optimize_consciousness_pattern(target)
                for target in targets
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Parallel optimization failed: {result}")
                    final_results.append(OptimizationResult(
                        target=OptimizationTarget("unknown", "unknown", 0, 0, OptimizationLevel.BIOLOGICAL_REFLEX),
                        optimized_prompt="",
                        optimized_parameters={},
                        performance_improvement=0.0,
                        confidence_improvement=0.0,
                        optimization_time_ms=0.0,
                        validation_results={"passed": False},
                        success=False,
                        error=str(result)
                    ))
                else:
                    final_results.append(result)
            
            return final_results
        
        else:
            # Sequential optimization
            logger.info(f"Starting sequential optimization of {len(targets)} patterns")
            
            results = []
            for target in targets:
                result = await self.optimize_consciousness_pattern(target)
                results.append(result)
            
            return results
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics"""
        success_rate = 0.0
        if self.metrics["total_optimizations"] > 0:
            success_rate = self.metrics["successful_optimizations"] / self.metrics["total_optimizations"]
        
        avg_optimization_time = 0.0
        if self.metrics["successful_optimizations"] > 0:
            avg_optimization_time = self.metrics["optimization_time_total"] / self.metrics["successful_optimizations"]
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "average_optimization_time_ms": avg_optimization_time,
            "pattern_library_size": len(self.pattern_library),
            "cross_pattern_knowledge_size": len(self.cross_pattern_knowledge),
            "optimization_history_size": len(self.optimization_history)
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on historical data"""
        recommendations = []
        
        # Analyze optimization history for patterns
        if len(self.optimization_history) >= 5:
            # Find underperforming patterns
            failed_patterns = [
                result for result in self.optimization_history[-10:]
                if not result.success
            ]
            
            if failed_patterns:
                pattern_failures = {}
                for result in failed_patterns:
                    pattern_key = f"{result.target.optimization_level.value}_{result.target.operation_type}"
                    pattern_failures[pattern_key] = pattern_failures.get(pattern_key, 0) + 1
                
                for pattern, failure_count in pattern_failures.items():
                    if failure_count >= 3:
                        recommendations.append({
                            "type": "pattern_review",
                            "pattern": pattern,
                            "issue": f"High failure rate ({failure_count} failures)",
                            "suggestion": "Review optimization strategy and safety constraints"
                        })
            
            # Find high-performing patterns for cross-learning
            successful_patterns = [
                result for result in self.optimization_history[-20:]
                if result.success and result.performance_improvement > 0.2
            ]
            
            if successful_patterns:
                recommendations.append({
                    "type": "cross_pattern_learning",
                    "count": len(successful_patterns),
                    "suggestion": "Apply successful optimization strategies across similar patterns"
                })
        
        # Configuration recommendations
        if self.metrics["total_optimizations"] > 10:
            success_rate = self.metrics["successful_optimizations"] / self.metrics["total_optimizations"]
            
            if success_rate < 0.7:
                recommendations.append({
                    "type": "configuration_tuning",
                    "success_rate": success_rate,
                    "suggestion": "Consider relaxing optimization constraints or improving training data"
                })
            elif success_rate > 0.9:
                recommendations.append({
                    "type": "optimization_scaling",
                    "success_rate": success_rate,
                    "suggestion": "Consider more aggressive optimization targets"
                })
        
        return recommendations

# Global MIPROv2 optimizer instance
miprov2_optimizer: Optional[ConsciousnessPatternOptimizer] = None

def get_miprov2_optimizer() -> Optional[ConsciousnessPatternOptimizer]:
    """Get global MIPROv2 optimizer instance"""
    return miprov2_optimizer

def initialize_miprov2_optimizer(config: DSPyConfig) -> ConsciousnessPatternOptimizer:
    """Initialize global MIPROv2 optimizer"""
    global miprov2_optimizer
    miprov2_optimizer = ConsciousnessPatternOptimizer(config)
    return miprov2_optimizer