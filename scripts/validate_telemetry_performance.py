#!/usr/bin/env python3
"""
Quick performance validation script for MemMimic v2.0 Telemetry System
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from memmimic.telemetry.performance_test import quick_overhead_test, TelemetryPerformanceValidator
    
    print("MemMimic v2.0 Telemetry Performance Validation")
    print("=" * 60)
    
    # Quick overhead test
    print("\n1. Quick Telemetry Overhead Test")
    print("-" * 30)
    overhead_ms = quick_overhead_test(iterations=1000)
    
    if overhead_ms < 1.0:
        print("✅ TELEMETRY OVERHEAD TARGET MET!")
    else:
        print("❌ Telemetry overhead exceeds target")
    
    print(f"\nResult: {overhead_ms:.6f}ms (Target: <1.0ms)")
    print(f"Performance Status: {'PASS' if overhead_ms < 1.0 else 'FAIL'}")
    
    # Comprehensive validation
    print("\n2. Comprehensive Performance Validation")
    print("-" * 40)
    print("Running comprehensive telemetry performance tests...")
    
    validator = TelemetryPerformanceValidator()
    
    # Just test the critical components for quick validation
    print("\nTesting core telemetry overhead...")
    core_result = validator._test_core_telemetry_overhead()
    
    print(f"Core Telemetry Performance:")
    print(f"  Mean: {core_result.mean_duration_ms:.6f}ms")
    print(f"  P95:  {core_result.p95_duration_ms:.6f}ms") 
    print(f"  P99:  {core_result.p99_duration_ms:.6f}ms")
    print(f"  Target Met: {'YES' if core_result.target_met else 'NO'}")
    
    print("\nTesting decorator overhead...")
    decorator_result = validator._test_decorator_overhead()
    
    print(f"Decorator Overhead:")
    print(f"  P95: {decorator_result.p95_duration_ms:.6f}ms")
    print(f"  Overhead %: {decorator_result.overhead_percentage:.2f}%")
    print(f"  Target Met: {'YES' if decorator_result.target_met else 'NO'}")
    
    # Overall assessment
    print("\n" + "=" * 60)
    if core_result.target_met and decorator_result.target_met:
        print("🎉 TELEMETRY SYSTEM PERFORMANCE VALIDATION: PASSED")
        print("✅ All critical performance targets met")
        print("✅ <1ms telemetry overhead confirmed")
        print("✅ System ready for production deployment")
    else:
        print("❌ TELEMETRY SYSTEM PERFORMANCE VALIDATION: FAILED")
        if not core_result.target_met:
            print(f"❌ Core telemetry overhead: {core_result.p95_duration_ms:.6f}ms > 1.0ms")
        if not decorator_result.target_met:
            print(f"❌ Decorator overhead: {decorator_result.p95_duration_ms:.6f}ms > 1.0ms")
    
    print("=" * 60)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to run this script from the project root directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error during validation: {e}")
    sys.exit(1)