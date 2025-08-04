#!/usr/bin/env python3
"""
DSPy Proof-of-Concept Runner

Command-line interface for running DSPy consciousness optimization
proof-of-concept validation and benchmarking.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memmimic.dspy_optimization.poc_implementation import DSPyPoCIntegration
from memmimic.dspy_optimization.config import create_default_config, create_production_config

def print_banner():
    """Print PoC runner banner"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                   DSPy Consciousness PoC Runner                ║
║                                                               ║
║  Testing DSPy optimization integration with MemMimic          ║
║  consciousness vault and biological reflex systems            ║
╚═══════════════════════════════════════════════════════════════╝
    """)

def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_subsection(title: str):
    """Print subsection header"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

def format_time(ms: float) -> str:
    """Format time with appropriate units"""
    if ms < 1:
        return f"{ms*1000:.1f}μs"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"

def format_percentage(value: float) -> str:
    """Format percentage with color coding"""
    percentage = value * 100
    if percentage >= 90:
        return f"🟢 {percentage:.1f}%"
    elif percentage >= 70:
        return f"🟡 {percentage:.1f}%"
    else:
        return f"🔴 {percentage:.1f}%"

async def run_initialization_test(integration: DSPyPoCIntegration) -> bool:
    """Test PoC initialization"""
    print_subsection("Initialization Test")
    
    try:
        start_time = time.time()
        success = await integration.initialize()
        init_time = (time.time() - start_time) * 1000
        
        if success:
            print(f"✅ Initialization successful ({format_time(init_time)})")
            
            # Show status
            status = integration.get_integration_status()
            print(f"   • Config loaded: {status['config_loaded']}")
            print(f"   • DSPy enabled: {status['dspy_enabled']}")
            print(f"   • Optimization mode: {status['optimization_mode']}")
            print(f"   • Biological reflex enabled: {status['biological_reflex_enabled']}")
            
            return True
        else:
            print("❌ Initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

async def run_validation_tests(integration: DSPyPoCIntegration) -> Dict[str, Any]:
    """Run comprehensive validation tests"""
    print_subsection("Validation Test Suite")
    
    from memmimic.dspy_optimization.poc_implementation import DSPyPoCValidator
    
    validator = DSPyPoCValidator(integration.config)
    
    print("Running validation test suite...")
    start_time = time.time()
    validation_report = await validator.run_poc_validation(integration.processor)
    total_time = (time.time() - start_time) * 1000
    
    # Print summary
    summary = validation_report["summary"]
    print(f"\n📊 Test Results ({format_time(total_time)}):")
    print(f"   • Total tests: {summary['total_tests']}")
    print(f"   • Passed: 🟢 {summary['passed_tests']}")
    print(f"   • Failed: 🔴 {summary['failed_tests']}")
    print(f"   • Success rate: {format_percentage(summary['success_rate'])}")
    print(f"   • Overall status: {'✅' if summary['overall_status'] == 'PASSED' else '❌'} {summary['overall_status']}")
    
    # Print performance metrics
    perf = validation_report["performance"]
    print(f"\n⚡ Performance Metrics:")
    print(f"   • Average response time: {format_time(perf['average_response_time_ms'])}")
    print(f"   • Max response time: {format_time(perf['max_response_time_ms'])}")
    print(f"   • Min response time: {format_time(perf['min_response_time_ms'])}")
    print(f"   • Average confidence: {perf['average_confidence']:.3f}")
    
    # Biological reflex compliance
    bio_compliance = perf["biological_reflex_compliance"]
    compliance_icon = "✅" if bio_compliance["compliant"] else "❌"
    print(f"\n🧬 Biological Reflex Compliance:")
    print(f"   • Status: {compliance_icon} {bio_compliance['compliant']}")
    print(f"   • Compliance rate: {format_percentage(bio_compliance['compliance_rate'])}")
    print(f"   • Tests: {bio_compliance['compliant_tests']}/{bio_compliance['total_biological_tests']}")
    
    # Processing mode breakdown
    if validation_report["processing_modes"]:
        print(f"\n🔄 Processing Mode Usage:")
        for mode, count in validation_report["processing_modes"].items():
            print(f"   • {mode}: {count} tests")
    
    # Show failed tests if any
    failed_tests = [
        result for result in validation_report["detailed_results"]
        if not result["passed"]
    ]
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"   • {test['test_name']}: {test['failure_reason']}")
    
    # Recommendations
    if validation_report["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in validation_report["recommendations"]:
            print(f"   • {rec}")
    
    return validation_report

async def run_performance_benchmark(integration: DSPyPoCIntegration, iterations: int = 100) -> Dict[str, Any]:
    """Run performance benchmarking"""
    print_subsection(f"Performance Benchmark ({iterations} iterations)")
    
    print("Running performance benchmark...")
    start_time = time.time()
    benchmark_results = await integration.run_performance_benchmark(iterations=iterations)
    total_time = (time.time() - start_time) * 1000
    
    print(f"\n🏃 Benchmark Results ({format_time(total_time)} total):")
    
    summary = benchmark_results["benchmark_summary"]
    
    for operation_type, results in summary.items():
        print(f"\n{operation_type.upper()}:")
        
        if "error" in results:
            print(f"   ❌ {results['error']}")
            continue
        
        success_rate = results["success_rate"]
        print(f"   • Success rate: {format_percentage(success_rate)}")
        
        if success_rate > 0:
            print(f"   • Average response time: {format_time(results['average_response_time_ms'])}")
            print(f"   • Min response time: {format_time(results['min_response_time_ms'])}")
            print(f"   • Max response time: {format_time(results['max_response_time_ms'])}")
            
            if "p95_response_time_ms" in results:
                print(f"   • P95 response time: {format_time(results['p95_response_time_ms'])}")
            if "p99_response_time_ms" in results:
                print(f"   • P99 response time: {format_time(results['p99_response_time_ms'])}")
            
            print(f"   • Average confidence: {results['average_confidence']:.3f}")
    
    # Overall system metrics
    if "overall_metrics" in benchmark_results:
        metrics = benchmark_results["overall_metrics"]
        if metrics:
            print(f"\n📈 System Metrics:")
            print(f"   • Total requests: {metrics.get('total_requests', 0)}")
            print(f"   • Fast path requests: {metrics.get('fast_path_requests', 0)}")
            print(f"   • Optimization requests: {metrics.get('optimization_path_requests', 0)}")
            print(f"   • Average response time: {format_time(metrics.get('average_response_time', 0))}")
    
    return benchmark_results

async def run_stress_test(integration: DSPyPoCIntegration, concurrent_requests: int = 10, duration_seconds: int = 30):
    """Run stress testing"""
    print_subsection(f"Stress Test ({concurrent_requests} concurrent, {duration_seconds}s)")
    
    async def stress_worker(worker_id: int, results: list):
        """Worker function for stress testing"""
        request_count = 0
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            try:
                response = await integration.processor.process_consciousness_request(
                    operation_type="status",
                    context={"query": f"stress test {worker_id}-{request_count}"},
                    urgency_level="immediate"
                )
                
                results.append({
                    "worker_id": worker_id,
                    "request_id": request_count,
                    "response_time_ms": response.response_time_ms,
                    "success": response.error is None,
                    "confidence": response.confidence_score
                })
                
                request_count += 1
                
            except Exception as e:
                results.append({
                    "worker_id": worker_id,
                    "request_id": request_count,
                    "error": str(e),
                    "success": False
                })
        
        return request_count
    
    print(f"Starting stress test with {concurrent_requests} workers...")
    
    # Run concurrent workers
    results = []
    tasks = [
        stress_worker(i, results)
        for i in range(concurrent_requests)
    ]
    
    start_time = time.time()
    worker_results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Analyze results
    total_requests = sum(worker_results)
    successful_requests = sum(1 for r in results if r.get("success", False))
    failed_requests = total_requests - successful_requests
    
    response_times = [r["response_time_ms"] for r in results if "response_time_ms" in r]
    
    print(f"\n🔥 Stress Test Results:")
    print(f"   • Duration: {total_time:.1f}s")
    print(f"   • Total requests: {total_requests}")
    print(f"   • Successful requests: 🟢 {successful_requests}")
    print(f"   • Failed requests: 🔴 {failed_requests}")
    print(f"   • Success rate: {format_percentage(successful_requests / total_requests if total_requests > 0 else 0)}")
    print(f"   • Requests per second: {total_requests / total_time:.1f}")
    
    if response_times:
        print(f"   • Average response time: {format_time(sum(response_times) / len(response_times))}")
        print(f"   • Min response time: {format_time(min(response_times))}")
        print(f"   • Max response time: {format_time(max(response_times))}")
        
        sorted_times = sorted(response_times)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)
        print(f"   • P95 response time: {format_time(sorted_times[p95_idx])}")
        print(f"   • P99 response time: {format_time(sorted_times[p99_idx])}")

def save_results(results: Dict[str, Any], output_file: Path):
    """Save results to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")

async def main():
    """Main PoC runner function"""
    parser = argparse.ArgumentParser(description="DSPy Consciousness PoC Runner")
    parser.add_argument("--validation", action="store_true", help="Run validation tests")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--stress", action="store_true", help="Run stress test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--concurrent", type=int, default=10, help="Stress test concurrent requests")
    parser.add_argument("--duration", type=int, default=30, help="Stress test duration (seconds)")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Default to all tests if none specified
    if not any([args.validation, args.benchmark, args.stress, args.all]):
        args.all = True
    
    print_banner()
    
    # Initialize PoC integration
    print_section("Initialization")
    
    config_path = Path(args.config) if args.config else None
    integration = DSPyPoCIntegration(config_path)
    
    if not await run_initialization_test(integration):
        print("\n❌ Cannot continue - initialization failed")
        sys.exit(1)
    
    # Store all results
    all_results = {
        "timestamp": time.time(),
        "config": {
            "iterations": args.iterations,
            "concurrent": args.concurrent,
            "duration": args.duration
        }
    }
    
    try:
        # Run validation tests
        if args.validation or args.all:
            print_section("Validation Tests")
            validation_results = await run_validation_tests(integration)
            all_results["validation"] = validation_results
        
        # Run performance benchmark
        if args.benchmark or args.all:
            print_section("Performance Benchmark")
            benchmark_results = await run_performance_benchmark(integration, args.iterations)
            all_results["benchmark"] = benchmark_results
        
        # Run stress test
        if args.stress or args.all:
            print_section("Stress Test")
            await run_stress_test(integration, args.concurrent, args.duration)
        
        # Final summary
        print_section("Summary")
        
        if "validation" in all_results:
            val_summary = all_results["validation"]["summary"]
            print(f"Validation: {format_percentage(val_summary['success_rate'])} ({val_summary['passed_tests']}/{val_summary['total_tests']} passed)")
        
        if "benchmark" in all_results:
            bench_summary = all_results["benchmark"]["benchmark_summary"]
            avg_success = sum(r.get("success_rate", 0) for r in bench_summary.values()) / len(bench_summary)
            print(f"Benchmark: {format_percentage(avg_success)} average success rate")
        
        print(f"\n✅ PoC testing completed successfully!")
        
        # Save results if requested
        if args.output:
            save_results(all_results, Path(args.output))
        
    except Exception as e:
        print(f"\n❌ PoC testing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())