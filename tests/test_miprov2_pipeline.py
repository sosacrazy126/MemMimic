"""
Test Suite for MIPROv2 Optimization Pipeline

Comprehensive testing for consciousness pattern optimization using MIPROv2 algorithm
with validation of multi-level optimization and cross-pattern learning capabilities.
"""

import pytest
import asyncio
import time
import numpy as np
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memmimic.dspy_optimization.miprov2_pipeline import (
    ConsciousnessPatternOptimizer,
    OptimizationTarget,
    OptimizationResult,
    OptimizationLevel,
    MIPROv2Configuration,
    initialize_miprov2_optimizer
)
from memmimic.dspy_optimization.config import create_default_config

class TestMIPROv2Pipeline:
    """Test suite for MIPROv2 optimization pipeline"""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return create_default_config()
    
    @pytest.fixture
    def miprov2_optimizer(self, test_config):
        """Create MIPROv2 optimizer for testing"""
        return ConsciousnessPatternOptimizer(test_config)
    
    @pytest.fixture
    def sample_optimization_targets(self):
        """Create sample optimization targets"""
        return [
            OptimizationTarget(
                pattern_name="test_biological_reflex",
                operation_type="status",
                target_response_time_ms=3.0,
                target_confidence_threshold=0.90,
                optimization_level=OptimizationLevel.BIOLOGICAL_REFLEX,
                priority=1
            ),
            OptimizationTarget(
                pattern_name="test_consciousness_pattern",
                operation_type="tool_selection",
                target_response_time_ms=25.0,
                target_confidence_threshold=0.85,
                optimization_level=OptimizationLevel.CONSCIOUSNESS_PATTERN,
                priority=2
            ),
            OptimizationTarget(
                pattern_name="test_complex_analysis",
                operation_type="tale_generation",
                target_response_time_ms=150.0,
                target_confidence_threshold=0.75,
                optimization_level=OptimizationLevel.COMPLEX_ANALYSIS,
                priority=3
            )
        ]
    
    def test_optimizer_initialization(self, miprov2_optimizer):
        """Test MIPROv2 optimizer initialization"""
        assert miprov2_optimizer.config is not None
        assert isinstance(miprov2_optimizer.miprov2_config, MIPROv2Configuration)
        assert len(miprov2_optimizer.standard_targets) > 0
        assert miprov2_optimizer.metrics["total_optimizations"] == 0
        
        # Check standard targets are properly configured
        biological_targets = [
            target for target in miprov2_optimizer.standard_targets
            if target.optimization_level == OptimizationLevel.BIOLOGICAL_REFLEX
        ]
        assert len(biological_targets) >= 2
        
        consciousness_targets = [
            target for target in miprov2_optimizer.standard_targets
            if target.optimization_level == OptimizationLevel.CONSCIOUSNESS_PATTERN
        ]
        assert len(consciousness_targets) >= 3
    
    def test_optimization_target_creation(self):
        """Test optimization target creation and validation"""
        target = OptimizationTarget(
            pattern_name="test_pattern",
            operation_type="test_operation",
            target_response_time_ms=10.0,
            target_confidence_threshold=0.8,
            optimization_level=OptimizationLevel.BIOLOGICAL_REFLEX
        )
        
        assert target.pattern_name == "test_pattern"
        assert target.operation_type == "test_operation"
        assert target.target_response_time_ms == 10.0
        assert target.target_confidence_threshold == 0.8
        assert target.optimization_level == OptimizationLevel.BIOLOGICAL_REFLEX
        assert target.priority == 1  # Default priority
    
    def test_optimization_strategy_determination(self, miprov2_optimizer, sample_optimization_targets):
        """Test optimization strategy determination for different levels"""
        
        # Test biological reflex strategy
        bio_target = sample_optimization_targets[0]
        bio_strategy = miprov2_optimizer._determine_optimization_strategy(bio_target)
        
        assert bio_strategy["optimization_type"] == "aggressive_speed"
        assert bio_strategy["performance_focus"] == "speed"
        assert bio_strategy["safety_constraints"] is True
        assert bio_strategy["max_iterations"] == 20
        
        # Test consciousness pattern strategy
        consciousness_target = sample_optimization_targets[1]
        consciousness_strategy = miprov2_optimizer._determine_optimization_strategy(consciousness_target)
        
        assert consciousness_strategy["optimization_type"] == "balanced"
        assert consciousness_strategy["use_cross_pattern_learning"] is True
        assert consciousness_strategy["performance_focus"] == "balanced"
        
        # Test complex analysis strategy
        complex_target = sample_optimization_targets[2]
        complex_strategy = miprov2_optimizer._determine_optimization_strategy(complex_target)
        
        assert complex_strategy["optimization_type"] == "quality_focused"
        assert complex_strategy["performance_focus"] == "quality"
        assert complex_strategy["enable_deep_analysis"] is True
    
    @pytest.mark.asyncio
    async def test_context_enrichment(self, miprov2_optimizer, sample_optimization_targets):
        """Test optimization context enrichment with documentation"""
        target = sample_optimization_targets[0]
        
        enriched_context = await miprov2_optimizer._enrich_optimization_context(target)
        
        assert isinstance(enriched_context, dict)
        assert "quality_score" in enriched_context
        assert enriched_context["quality_score"] >= 0.0
        
        # If documentation context is available, check structure
        if "documentation" in enriched_context:
            assert isinstance(enriched_context["documentation"], list)
            assert "consciousness_patterns" in enriched_context
            assert "sources" in enriched_context
    
    def test_optimized_prompt_generation(self, miprov2_optimizer, sample_optimization_targets):
        """Test optimized prompt generation"""
        target = sample_optimization_targets[0]
        context = {"quality_score": 0.8}
        improvement_factor = 0.3
        
        prompt = miprov2_optimizer._generate_optimized_prompt(target, context, improvement_factor)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "status" in prompt.lower()  # Should contain operation type
        assert str(target.target_confidence_threshold) in prompt
        assert str(target.target_response_time_ms) in prompt
        
        # For biological reflex, should mention sub-5ms
        if target.optimization_level == OptimizationLevel.BIOLOGICAL_REFLEX:
            assert "sub-5ms" in prompt or "biological reflex" in prompt.lower()
    
    def test_cross_pattern_benefits_calculation(self, miprov2_optimizer, sample_optimization_targets):
        """Test cross-pattern learning benefits calculation"""
        target = sample_optimization_targets[1]
        
        # Initially no benefits (no history)
        benefits = miprov2_optimizer._get_cross_pattern_benefits(target)
        assert benefits == 0.0
        
        # Add some successful optimization history
        for i in range(3):
            miprov2_optimizer.optimization_history.append(
                OptimizationResult(
                    target=OptimizationTarget(
                        f"test_pattern_{i}",
                        "tool_selection",
                        25.0,
                        0.85,
                        OptimizationLevel.CONSCIOUSNESS_PATTERN
                    ),
                    optimized_prompt="test",
                    optimized_parameters={},
                    performance_improvement=0.2 + i * 0.1,
                    confidence_improvement=0.1,
                    optimization_time_ms=50.0,
                    validation_results={"passed": True},
                    success=True
                )
            )
        
        # Now should have some benefits
        benefits = miprov2_optimizer._get_cross_pattern_benefits(target)
        assert benefits > 0.0
        assert benefits <= 0.15  # Should be capped at 15%
    
    def test_safety_constraint_validation(self, miprov2_optimizer, sample_optimization_targets):
        """Test safety constraint validation"""
        target = sample_optimization_targets[0]  # Biological reflex target
        
        # Valid optimization result
        valid_result = {
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 0.9
            }
        }
        
        assert miprov2_optimizer._validate_safety_constraints(target, valid_result) is True
        
        # Invalid temperature
        invalid_temp_result = {
            "parameters": {
                "temperature": 1.5,  # Too high
                "max_tokens": 512
            }
        }
        
        assert miprov2_optimizer._validate_safety_constraints(target, invalid_temp_result) is False
        
        # Invalid max_tokens
        invalid_tokens_result = {
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 3000  # Too high
            }
        }
        
        assert miprov2_optimizer._validate_safety_constraints(target, invalid_tokens_result) is False
        
        # Biological reflex with too slow target time
        slow_target = OptimizationTarget(
            "slow_reflex",
            "status",
            target_response_time_ms=10.0,  # Too slow for biological reflex
            target_confidence_threshold=0.9,
            optimization_level=OptimizationLevel.BIOLOGICAL_REFLEX
        )
        
        assert miprov2_optimizer._validate_safety_constraints(slow_target, valid_result) is False
    
    @pytest.mark.asyncio
    async def test_optimization_result_validation(self, miprov2_optimizer, sample_optimization_targets):
        """Test optimization result validation"""
        target = sample_optimization_targets[0]
        
        # Good optimization result
        good_result = {
            "performance_improvement": 0.4,  # 40% improvement
            "confidence_improvement": 0.15,  # 15% improvement
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 512
            }
        }
        
        validation = await miprov2_optimizer._validate_optimization_results(target, good_result)
        
        assert isinstance(validation, dict)
        assert "passed" in validation
        assert "performance_check" in validation
        assert "confidence_check" in validation
        assert "safety_check" in validation
        assert "response_time_projection" in validation
        assert "confidence_projection" in validation
        
        # Performance projection should be reasonable
        assert validation["response_time_projection"] > 0
        assert validation["confidence_projection"] > 0
    
    @pytest.mark.asyncio
    async def test_single_pattern_optimization(self, miprov2_optimizer, sample_optimization_targets):
        """Test single pattern optimization end-to-end"""
        target = sample_optimization_targets[0]
        
        result = await miprov2_optimizer.optimize_consciousness_pattern(target)
        
        assert isinstance(result, OptimizationResult)
        assert result.target == target
        assert isinstance(result.optimized_prompt, str)
        assert len(result.optimized_prompt) > 0
        assert isinstance(result.optimized_parameters, dict)
        assert result.optimization_time_ms > 0
        assert isinstance(result.validation_results, dict)
        assert isinstance(result.success, bool)
        
        # Check that metrics were updated
        assert miprov2_optimizer.metrics["total_optimizations"] == 1
        
        if result.success:
            assert miprov2_optimizer.metrics["successful_optimizations"] == 1
            assert result.performance_improvement >= 0.0
            assert result.confidence_improvement >= 0.0
        
        # Check optimization history
        assert len(miprov2_optimizer.optimization_history) == 1
        assert miprov2_optimizer.optimization_history[0] == result
    
    @pytest.mark.asyncio
    async def test_multiple_pattern_optimization_sequential(self, miprov2_optimizer, sample_optimization_targets):
        """Test multiple pattern optimization in sequential mode"""
        
        results = await miprov2_optimizer.optimize_multiple_patterns(
            sample_optimization_targets,
            parallel_optimization=False
        )
        
        assert len(results) == len(sample_optimization_targets)
        assert all(isinstance(result, OptimizationResult) for result in results)
        
        # Check that all targets were processed
        target_names = {result.target.pattern_name for result in results}
        expected_names = {target.pattern_name for target in sample_optimization_targets}
        assert target_names == expected_names
        
        # Check metrics updated correctly
        assert miprov2_optimizer.metrics["total_optimizations"] == len(sample_optimization_targets)
    
    @pytest.mark.asyncio
    async def test_multiple_pattern_optimization_parallel(self, miprov2_optimizer, sample_optimization_targets):
        """Test multiple pattern optimization in parallel mode"""
        
        # Use a fresh optimizer to avoid interference from previous tests
        fresh_optimizer = ConsciousnessPatternOptimizer(miprov2_optimizer.config)
        
        start_time = time.time()
        results = await fresh_optimizer.optimize_multiple_patterns(
            sample_optimization_targets[:2],  # Use first 2 targets for speed
            parallel_optimization=True
        )
        total_time = time.time() - start_time
        
        assert len(results) == 2
        assert all(isinstance(result, OptimizationResult) for result in results)
        
        # Parallel should be faster than sum of individual times
        # (This is a rough check since we're simulating)
        individual_times = [result.optimization_time_ms for result in results]
        if all(time > 0 for time in individual_times):
            # Allow some overhead, but should be significantly faster than sequential
            assert total_time * 1000 < sum(individual_times) * 1.5
    
    def test_metrics_tracking(self, miprov2_optimizer):
        """Test optimization metrics tracking"""
        
        # Initial metrics
        initial_metrics = miprov2_optimizer.get_optimization_metrics()
        
        assert "total_optimizations" in initial_metrics
        assert "successful_optimizations" in initial_metrics
        assert "average_improvement" in initial_metrics
        assert "best_performance_gain" in initial_metrics
        assert "success_rate" in initial_metrics
        assert "pattern_library_size" in initial_metrics
        
        assert initial_metrics["total_optimizations"] == 0
        assert initial_metrics["success_rate"] == 0.0
        
        # Simulate some optimizations by adding to history
        for i in range(5):
            miprov2_optimizer.optimization_history.append(
                OptimizationResult(
                    target=OptimizationTarget(f"test_{i}", "test", 10.0, 0.8, OptimizationLevel.BIOLOGICAL_REFLEX),
                    optimized_prompt="test",
                    optimized_parameters={},
                    performance_improvement=0.1 + i * 0.05,
                    confidence_improvement=0.05,
                    optimization_time_ms=50.0,
                    validation_results={"passed": True},
                    success=i < 4  # 4 successful, 1 failed
                )
            )
            
            miprov2_optimizer.metrics["total_optimizations"] += 1
            if i < 4:
                miprov2_optimizer.metrics["successful_optimizations"] += 1
                miprov2_optimizer._update_optimization_metrics(miprov2_optimizer.optimization_history[-1])
        
        updated_metrics = miprov2_optimizer.get_optimization_metrics()
        
        assert updated_metrics["total_optimizations"] == 5
        assert updated_metrics["successful_optimizations"] == 4
        assert updated_metrics["success_rate"] == 0.8
        assert updated_metrics["average_improvement"] > 0
        assert updated_metrics["best_performance_gain"] > 0
    
    @pytest.mark.asyncio
    async def test_cross_pattern_knowledge_integration(self, miprov2_optimizer, sample_optimization_targets):
        """Test cross-pattern knowledge integration"""
        
        target = sample_optimization_targets[1]  # Consciousness pattern target
        
        optimization_result = {
            "success": True,
            "performance_improvement": 0.25,
            "confidence_improvement": 0.1,
            "parameters": {"temperature": 0.7}
        }
        
        # Initial pattern library should be empty
        pattern_key = f"{target.optimization_level.value}_{target.operation_type}"
        assert pattern_key not in miprov2_optimizer.pattern_library
        
        # Integrate knowledge
        await miprov2_optimizer._integrate_cross_pattern_knowledge(target, optimization_result)
        
        # Pattern library should now contain the pattern
        assert pattern_key in miprov2_optimizer.pattern_library
        
        pattern_data = miprov2_optimizer.pattern_library[pattern_key]
        assert len(pattern_data["successful_optimizations"]) == 1
        assert pattern_data["best_performance"] == 0.25
        assert pattern_data["average_improvement"] == 0.25
        
        # Add another optimization for the same pattern
        optimization_result2 = {
            "success": True,
            "performance_improvement": 0.15,
            "confidence_improvement": 0.08,
            "parameters": {"temperature": 0.8}
        }
        
        await miprov2_optimizer._integrate_cross_pattern_knowledge(target, optimization_result2)
        
        # Should have updated averages
        updated_pattern_data = miprov2_optimizer.pattern_library[pattern_key]
        assert len(updated_pattern_data["successful_optimizations"]) == 2
        assert updated_pattern_data["best_performance"] == 0.25  # Still the best
        assert updated_pattern_data["average_improvement"] == 0.2  # (0.25 + 0.15) / 2
    
    def test_optimization_recommendations(self, miprov2_optimizer):
        """Test optimization recommendation generation"""
        
        # Initially should have no recommendations
        recommendations = miprov2_optimizer.get_optimization_recommendations()
        assert isinstance(recommendations, list)
        
        # Add some optimization history to generate recommendations
        
        # Add failed optimizations for pattern review recommendation
        for i in range(4):
            miprov2_optimizer.optimization_history.append(
                OptimizationResult(
                    target=OptimizationTarget("failed_pattern", "test_op", 10.0, 0.8, OptimizationLevel.BIOLOGICAL_REFLEX),
                    optimized_prompt="",
                    optimized_parameters={},
                    performance_improvement=0.0,
                    confidence_improvement=0.0,
                    optimization_time_ms=50.0,
                    validation_results={"passed": False},
                    success=False,
                    error="Test failure"
                )
            )
        
        # Add successful optimizations for cross-pattern learning recommendation
        for i in range(5):
            miprov2_optimizer.optimization_history.append(
                OptimizationResult(
                    target=OptimizationTarget(f"success_{i}", "test_op", 10.0, 0.8, OptimizationLevel.CONSCIOUSNESS_PATTERN),
                    optimized_prompt="test",
                    optimized_parameters={},
                    performance_improvement=0.3,  # High improvement
                    confidence_improvement=0.1,
                    optimization_time_ms=50.0,
                    validation_results={"passed": True},
                    success=True
                )
            )
        
        # Update metrics
        miprov2_optimizer.metrics["total_optimizations"] = 20
        miprov2_optimizer.metrics["successful_optimizations"] = 15
        
        recommendations = miprov2_optimizer.get_optimization_recommendations()
        
        # Should have recommendations now
        assert len(recommendations) > 0
        
        # Check for expected recommendation types
        recommendation_types = {rec["type"] for rec in recommendations}
        assert "pattern_review" in recommendation_types  # Due to failed patterns
        assert "cross_pattern_learning" in recommendation_types  # Due to successful patterns
    
    def test_configuration_validation(self):
        """Test MIPROv2 configuration validation"""
        config = MIPROv2Configuration()
        
        # Check default values are reasonable
        assert config.max_optimization_iterations > 0
        assert 0.0 < config.convergence_threshold < 1.0
        assert 0.0 < config.performance_weight < 1.0
        assert 0.0 < config.confidence_weight < 1.0
        assert config.performance_weight + config.confidence_weight == 1.0
        assert config.biological_reflex_max_time == 5.0
        assert config.validation_samples > 0
        assert config.safety_validation_required is True

if __name__ == "__main__":
    # Run basic tests for development
    
    async def run_development_tests():
        """Run tests for development/debugging"""
        print("🧪 Running MIPROv2 Optimization Pipeline Tests\n")
        
        # Create test instance
        config = create_default_config()
        optimizer = ConsciousnessPatternOptimizer(config)
        
        print("✅ MIPROv2 Optimizer Initialized")
        print(f"   • Standard targets: {len(optimizer.standard_targets)}")
        print(f"   • Pattern library: {len(optimizer.pattern_library)}")
        
        # Test optimization strategy determination
        print("\n🎯 Testing Optimization Strategies:")
        
        test_targets = [
            OptimizationTarget("bio_test", "status", 3.0, 0.9, OptimizationLevel.BIOLOGICAL_REFLEX),
            OptimizationTarget("consciousness_test", "tool_selection", 25.0, 0.85, OptimizationLevel.CONSCIOUSNESS_PATTERN),
            OptimizationTarget("complex_test", "tale_generation", 150.0, 0.75, OptimizationLevel.COMPLEX_ANALYSIS)
        ]
        
        for target in test_targets:
            strategy = optimizer._determine_optimization_strategy(target)
            print(f"   • {target.optimization_level.value}: {strategy['optimization_type']}")
        
        # Test single optimization
        print(f"\n🚀 Testing Single Pattern Optimization:")
        
        target = test_targets[0]  # Biological reflex
        print(f"   Target: {target.pattern_name} ({target.optimization_level.value})")
        
        result = await optimizer.optimize_consciousness_pattern(target)
        print(f"   • Success: {result.success}")
        print(f"   • Performance improvement: {result.performance_improvement:.3f}")
        print(f"   • Confidence improvement: {result.confidence_improvement:.3f}")
        print(f"   • Optimization time: {result.optimization_time_ms:.1f}ms")
        print(f"   • Validation passed: {result.validation_results.get('passed', False)}")
        
        # Test multiple optimizations
        print(f"\n⚡ Testing Multiple Pattern Optimization:")
        
        results = await optimizer.optimize_multiple_patterns(test_targets[:2], parallel_optimization=True)
        print(f"   • Optimized {len(results)} patterns")
        print(f"   • Success rate: {sum(1 for r in results if r.success)}/{len(results)}")
        
        # Test metrics
        print(f"\n📊 Performance Metrics:")
        metrics = optimizer.get_optimization_metrics()
        print(f"   • Total optimizations: {metrics['total_optimizations']}")
        print(f"   • Success rate: {metrics['success_rate']:.2%}")
        print(f"   • Average improvement: {metrics['average_improvement']:.3f}")
        print(f"   • Best performance gain: {metrics['best_performance_gain']:.3f}")
        
        # Test recommendations
        recommendations = optimizer.get_optimization_recommendations()
        if recommendations:
            print(f"\n💡 Optimization Recommendations:")
            for rec in recommendations:
                print(f"   • {rec['type']}: {rec.get('suggestion', 'No suggestion')}")
        
        print(f"\n🎯 MIPROv2 Pipeline Testing Complete!")
    
    # Run development tests
    asyncio.run(run_development_tests())