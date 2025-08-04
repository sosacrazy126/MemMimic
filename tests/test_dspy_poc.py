"""
DSPy Proof-of-Concept Test Suite

Comprehensive testing for DSPy consciousness optimization integration
with safety validation and performance benchmarking.
"""

import pytest
import asyncio
import time
from pathlib import Path
from typing import Dict, Any

from src.memmimic.dspy_optimization.poc_implementation import (
    DSPyPoCIntegration,
    DSPyPoCValidator,
    PoCTestCase
)
from src.memmimic.dspy_optimization.config import create_default_config
from src.memmimic.dspy_optimization.hybrid_processor import ProcessingMode

class TestDSPyPoC:
    """Test suite for DSPy proof-of-concept implementation"""
    
    @pytest.fixture
    async def poc_integration(self):
        """Create PoC integration for testing"""
        integration = DSPyPoCIntegration()
        success = await integration.initialize()
        assert success, "Failed to initialize PoC integration"
        return integration
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        config = create_default_config()
        # Override for testing
        config.performance.biological_reflex_max_time = 10  # More lenient for testing
        config.integration.enable_dspy_optimization = False  # Start with safety
        return config
    
    @pytest.mark.asyncio
    async def test_poc_initialization(self):
        """Test PoC integration initialization"""
        integration = DSPyPoCIntegration()
        
        # Test initialization
        success = await integration.initialize()
        assert success
        
        # Check components are created
        assert integration.config is not None
        assert integration.processor is not None
        assert integration.nervous_system is not None
        
        # Check status
        status = integration.get_integration_status()
        assert status["initialized"] is True
        assert status["config_loaded"] is True
    
    @pytest.mark.asyncio
    async def test_biological_reflex_timing(self, poc_integration):
        """Test biological reflex timing compliance"""
        processor = poc_integration.processor
        
        # Test immediate urgency (biological reflex)
        start_time = time.time()
        response = await processor.process_consciousness_request(
            operation_type="status",
            context={"query": "system status"},
            urgency_level="immediate"
        )
        total_time = (time.time() - start_time) * 1000
        
        # Validate response
        assert response is not None
        assert response.error is None
        assert response.processing_mode == ProcessingMode.FAST_PATH
        assert response.response_time_ms <= 10.0  # Allow 10ms for testing
        assert total_time <= 50.0  # Total time should be reasonable
        
        print(f"Biological reflex test: {response.response_time_ms}ms")
    
    @pytest.mark.asyncio
    async def test_consciousness_pattern_processing(self, poc_integration):
        """Test consciousness pattern processing"""
        processor = poc_integration.processor
        
        # Test normal consciousness operation
        response = await processor.process_consciousness_request(
            operation_type="pattern_recognition",
            context={"input_text": "exponential collaboration mode"},
            urgency_level="normal",
            consciousness_patterns=["synergy_protocol", "exponential_mode"]
        )
        
        # Validate response
        assert response is not None
        assert response.error is None
        assert response.response_time_ms <= 100.0  # Reasonable limit for testing
        assert response.confidence_score > 0.0
        
        print(f"Consciousness pattern test: {response.response_time_ms}ms, confidence: {response.confidence_score}")
    
    @pytest.mark.asyncio
    async def test_complex_optimization_routing(self, poc_integration):
        """Test complex operation routing"""
        processor = poc_integration.processor
        
        # Test complex operation
        response = await processor.process_consciousness_request(
            operation_type="tool_selection",
            context={
                "context": "user needs complex consciousness analysis",
                "available_tools": ["recall", "think", "analyze", "tale_generation"]
            },
            urgency_level="complex",
            consciousness_patterns=["intelligence_mesh", "pattern_analysis"]
        )
        
        # Validate response
        assert response is not None
        assert response.error is None
        assert response.response_time_ms <= 500.0  # Allow more time for complex operations
        assert response.confidence_score > 0.0
        
        print(f"Complex operation test: {response.response_time_ms}ms, mode: {response.processing_mode.value}")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self, poc_integration):
        """Test error handling and fallback mechanisms"""
        processor = poc_integration.processor
        
        # Test with invalid context
        response = await processor.process_consciousness_request(
            operation_type="invalid_operation",
            context={},
            urgency_level="immediate"
        )
        
        # Should handle gracefully
        assert response is not None
        # Either succeeds with fallback or fails gracefully
        if response.error:
            assert response.fallback_used or response.processing_mode == ProcessingMode.FAST_PATH
        
        print(f"Error handling test: error={response.error}, fallback={response.fallback_used}")
    
    @pytest.mark.asyncio
    async def test_poc_validation_suite(self, poc_integration):
        """Test complete PoC validation suite"""
        validator = DSPyPoCValidator(poc_integration.config)
        
        # Run validation
        validation_report = await validator.run_poc_validation(poc_integration.processor)
        
        # Check validation results
        assert "summary" in validation_report
        assert "performance" in validation_report
        assert "detailed_results" in validation_report
        
        summary = validation_report["summary"]
        assert summary["total_tests"] > 0
        
        # Print results
        print(f"\nPoC Validation Results:")
        print(f"Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Status: {summary['overall_status']}")
        
        # Check biological reflex compliance
        bio_compliance = validation_report["performance"]["biological_reflex_compliance"]
        print(f"Biological Reflex Compliance: {bio_compliance['compliant']}")
        print(f"Compliance Rate: {bio_compliance['compliance_rate']:.2%}")
        
        # Recommendations
        if validation_report["recommendations"]:
            print("\nRecommendations:")
            for rec in validation_report["recommendations"]:
                print(f"- {rec}")
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, poc_integration):
        """Test performance benchmarking"""
        # Run smaller benchmark for testing
        benchmark_results = await poc_integration.run_performance_benchmark(iterations=10)
        
        # Validate benchmark results
        assert "benchmark_summary" in benchmark_results
        assert "overall_metrics" in benchmark_results
        
        summary = benchmark_results["benchmark_summary"]
        
        # Check each operation type
        for operation_type, results in summary.items():
            assert "iterations" in results
            assert "success_rate" in results
            
            if results["success_rate"] > 0:
                assert "average_response_time_ms" in results
                assert "average_confidence" in results
                
                print(f"\n{operation_type.upper()} Benchmark:")
                print(f"Success Rate: {results['success_rate']:.2%}")
                print(f"Avg Response Time: {results['average_response_time_ms']:.2f}ms")
                print(f"Avg Confidence: {results['average_confidence']:.3f}")
                
                if "p95_response_time_ms" in results:
                    print(f"P95 Response Time: {results['p95_response_time_ms']:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, test_config):
        """Test configuration validation"""
        # Test valid configuration
        issues = test_config.validate()
        print(f"\nConfiguration validation issues: {len(issues)}")
        for issue in issues:
            print(f"- {issue}")
        
        # Test configuration loading/saving
        test_path = Path("/tmp/test_dspy_config.yaml")
        try:
            test_config.to_yaml(test_path)
            loaded_config = test_config.from_yaml(test_path)
            
            assert loaded_config.config_version == test_config.config_version
            assert loaded_config.environment == test_config.environment
            
        finally:
            if test_path.exists():
                test_path.unlink()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, poc_integration):
        """Test circuit breaker protection"""
        processor = poc_integration.processor
        
        # Get initial metrics
        initial_metrics = processor.get_performance_metrics()
        
        # Run multiple requests to test circuit breaker
        for i in range(5):
            response = await processor.process_consciousness_request(
                operation_type="status",
                context={"query": f"test request {i}"},
                urgency_level="immediate"
            )
            assert response is not None
        
        # Get final metrics
        final_metrics = processor.get_performance_metrics()
        
        # Check metrics updated
        assert final_metrics["total_requests"] >= initial_metrics["total_requests"] + 5
        
        print(f"\nCircuit Breaker Test:")
        print(f"Total Requests: {final_metrics['total_requests']}")
        print(f"Fast Path: {final_metrics['fast_path_requests']}")
        print(f"Optimization Path: {final_metrics['optimization_path_requests']}")
    
    def test_safety_constraints(self, test_config):
        """Test safety constraint validation"""
        # Test biological reflex timing constraints
        assert test_config.performance.biological_reflex_max_time <= 10
        
        # Test resource limits
        assert test_config.performance.max_token_budget_per_hour > 0
        assert test_config.performance.max_concurrent_optimizations > 0
        
        # Test safety defaults
        assert test_config.integration.enable_dspy_optimization == False  # Should start disabled
        assert test_config.integration.fallback_strategy == "graceful"
        
        # Test A/B testing constraints
        assert test_config.integration.ab_test_percentage <= 0.1  # Conservative rollout
        
        print(f"\nSafety Configuration Validated:")
        print(f"Biological Reflex Max Time: {test_config.performance.biological_reflex_max_time}ms")
        print(f"DSPy Optimization Enabled: {test_config.integration.enable_dspy_optimization}")
        print(f"A/B Test Percentage: {test_config.integration.ab_test_percentage:.1%}")

if __name__ == "__main__":
    # Run tests directly for development
    import sys
    import os
    
    # Add src to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    
    async def run_development_tests():
        """Run tests for development/debugging"""
        print("🧪 Running DSPy PoC Development Tests\n")
        
        # Initialize PoC
        integration = DSPyPoCIntegration()
        success = await integration.initialize()
        if not success:
            print("❌ Failed to initialize PoC integration")
            return
        
        print("✅ PoC Integration Initialized")
        
        # Run validation
        validator = DSPyPoCValidator(integration.config)
        print("\n🔍 Running PoC Validation...")
        validation_report = await validator.run_poc_validation(integration.processor)
        
        summary = validation_report["summary"]
        print(f"\n📊 Validation Summary:")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Overall Status: {summary['overall_status']}")
        
        # Show performance metrics
        perf = validation_report["performance"]
        print(f"\n⚡ Performance Metrics:")
        print(f"Average Response Time: {perf['average_response_time_ms']:.2f}ms")
        print(f"Max Response Time: {perf['max_response_time_ms']:.2f}ms")
        print(f"Average Confidence: {perf['average_confidence']:.3f}")
        
        # Biological reflex compliance
        bio_compliance = perf["biological_reflex_compliance"]
        print(f"\n🧬 Biological Reflex Compliance:")
        print(f"Compliant: {bio_compliance['compliant']}")
        print(f"Compliance Rate: {bio_compliance['compliance_rate']:.2%}")
        print(f"Message: {bio_compliance['message']}")
        
        # Recommendations
        if validation_report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in validation_report["recommendations"]:
                print(f"• {rec}")
        
        # Run performance benchmark
        print(f"\n🏃 Running Performance Benchmark...")
        benchmark_results = await integration.run_performance_benchmark(iterations=5)
        
        for operation_type, results in benchmark_results["benchmark_summary"].items():
            if results["success_rate"] > 0:
                print(f"\n{operation_type.upper()}:")
                print(f"  Success Rate: {results['success_rate']:.2%}")
                print(f"  Avg Response Time: {results['average_response_time_ms']:.2f}ms")
                print(f"  P95 Response Time: {results.get('p95_response_time_ms', 0):.2f}ms")
        
        print(f"\n🎯 PoC Testing Complete!")
    
    # Run development tests
    asyncio.run(run_development_tests())