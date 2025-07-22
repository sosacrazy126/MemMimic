"""
Performance Validation Test for MemMimic v2.0 Telemetry System

Comprehensive performance testing to validate <1ms telemetry overhead target
and ensure all v2.0 performance requirements are met.
"""

import time
import statistics
import asyncio
import threading
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from ..errors import get_error_logger
from .collector import TelemetryCollector, get_telemetry_collector, telemetry_timer
from .aggregator import MetricsAggregator, get_metrics_aggregator
from .monitor import PerformanceMonitor, get_performance_monitor
from .alerts import AlertingSystem, get_alerting_system
from .integration import telemetry_enabled, storage_telemetry, TelemetryContext

logger = get_error_logger("telemetry.performance_test")


@dataclass
class PerformanceTestResult:
    """Performance test result structure"""
    test_name: str
    iterations: int
    mean_duration_ms: float
    median_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    std_deviation_ms: float
    target_met: bool
    target_threshold_ms: float
    overhead_percentage: float = 0.0
    success_rate: float = 100.0


class TelemetryPerformanceValidator:
    """
    Comprehensive performance validator for telemetry system.
    
    Tests:
    1. Core telemetry collection overhead
    2. Integration decorator overhead  
    3. Context manager overhead
    4. Aggregation performance
    5. Monitoring system performance
    6. Alert processing performance
    7. End-to-end system performance
    """
    
    def __init__(self):
        self.collector = get_telemetry_collector()
        self.aggregator = get_metrics_aggregator()
        self.monitor = get_performance_monitor()
        self.alerting = get_alerting_system()
        
        # Test configuration
        self.iterations = 10000  # High iteration count for statistical significance
        self.warmup_iterations = 1000
        self.target_overhead_ms = 1.0  # <1ms target
        
        # Results storage
        self.test_results: Dict[str, PerformanceTestResult] = {}
    
    def run_comprehensive_performance_validation(self) -> Dict[str, Any]:
        """Run complete performance validation suite"""
        logger.info("Starting comprehensive telemetry performance validation")
        
        # Warm up the telemetry system
        self._warmup_telemetry_system()
        
        # Run individual performance tests
        tests = [
            ("core_telemetry_overhead", self._test_core_telemetry_overhead),
            ("decorator_overhead", self._test_decorator_overhead),
            ("async_decorator_overhead", self._test_async_decorator_overhead),
            ("context_manager_overhead", self._test_context_manager_overhead),
            ("storage_telemetry_overhead", self._test_storage_telemetry_overhead),
            ("integration_overhead", self._test_integration_overhead),
            ("aggregation_performance", self._test_aggregation_performance),
            ("monitoring_performance", self._test_monitoring_performance),
            ("alert_processing_performance", self._test_alert_processing_performance),
            ("concurrent_telemetry_overhead", self._test_concurrent_telemetry_overhead),
            ("memory_usage_impact", self._test_memory_usage_impact)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = result
                
                status = "PASS" if result.target_met else "FAIL"
                logger.info(f"Test {test_name}: {status} - P95: {result.p95_duration_ms:.3f}ms")
                
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}")
                self.test_results[test_name] = PerformanceTestResult(
                    test_name=test_name,
                    iterations=0,
                    mean_duration_ms=float('inf'),
                    median_duration_ms=float('inf'),
                    p95_duration_ms=float('inf'),
                    p99_duration_ms=float('inf'),
                    min_duration_ms=float('inf'),
                    max_duration_ms=float('inf'),
                    std_deviation_ms=float('inf'),
                    target_met=False,
                    target_threshold_ms=self.target_overhead_ms
                )
        
        # Generate comprehensive report
        return self._generate_performance_report()
    
    def _warmup_telemetry_system(self):
        """Warm up telemetry system to ensure stable measurements"""
        logger.info("Warming up telemetry system...")
        
        for _ in range(self.warmup_iterations):
            # Exercise all telemetry paths
            self.collector.record_operation("warmup_test", 1.0, {"warmup": True}, True)
            self.collector.record_storage_metrics("warmup_storage", 1.0, 100, False, True)
            self.collector.record_memory_metrics("warmup_memory", 1.0, "enhanced", 5, True)
            self.collector.record_governance_metrics("warmup_governance", 1.0, 0, 0, "approved")
        
        # Let background processing catch up
        time.sleep(1.0)
        
        logger.info("Telemetry system warmed up")
    
    def _test_core_telemetry_overhead(self) -> PerformanceTestResult:
        """Test core telemetry collection overhead"""
        durations = []
        
        for _ in range(self.iterations):
            # Measure time for minimal telemetry record
            start_time = time.perf_counter()
            self.collector.record_operation("test_operation", 1.0, {}, True)
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)
        
        return self._calculate_performance_result("core_telemetry_overhead", durations, self.target_overhead_ms)
    
    def _test_decorator_overhead(self) -> PerformanceTestResult:
        """Test decorator overhead"""
        
        @telemetry_enabled("test_decorator_operation")
        def test_function():
            # Minimal function to test decorator overhead
            return 42
        
        # Baseline without decorator
        baseline_durations = []
        for _ in range(self.iterations):
            start_time = time.perf_counter()
            result = 42  # Same operation without decorator
            end_time = time.perf_counter()
            baseline_durations.append((end_time - start_time) * 1000)
        
        # With decorator
        decorated_durations = []
        for _ in range(self.iterations):
            start_time = time.perf_counter()
            result = test_function()
            end_time = time.perf_counter()
            decorated_durations.append((end_time - start_time) * 1000)
        
        # Calculate overhead
        baseline_mean = statistics.mean(baseline_durations)
        decorated_mean = statistics.mean(decorated_durations)
        overhead_durations = [d - baseline_mean for d in decorated_durations]
        
        result = self._calculate_performance_result("decorator_overhead", overhead_durations, self.target_overhead_ms)
        result.overhead_percentage = ((decorated_mean - baseline_mean) / baseline_mean) * 100
        
        return result
    
    def _test_async_decorator_overhead(self) -> PerformanceTestResult:
        """Test async decorator overhead"""
        from .integration import telemetry_enabled_async
        
        @telemetry_enabled_async("test_async_decorator_operation")
        async def async_test_function():
            return 42
        
        async def run_async_test():
            # Baseline without decorator
            baseline_durations = []
            for _ in range(self.iterations):
                start_time = time.perf_counter()
                result = 42
                end_time = time.perf_counter()
                baseline_durations.append((end_time - start_time) * 1000)
            
            # With async decorator
            decorated_durations = []
            for _ in range(self.iterations):
                start_time = time.perf_counter()
                result = await async_test_function()
                end_time = time.perf_counter()
                decorated_durations.append((end_time - start_time) * 1000)
            
            return baseline_durations, decorated_durations
        
        # Run async test
        baseline_durations, decorated_durations = asyncio.run(run_async_test())
        
        # Calculate overhead
        baseline_mean = statistics.mean(baseline_durations)
        decorated_mean = statistics.mean(decorated_durations)
        overhead_durations = [d - baseline_mean for d in decorated_durations]
        
        result = self._calculate_performance_result("async_decorator_overhead", overhead_durations, self.target_overhead_ms)
        result.overhead_percentage = ((decorated_mean - baseline_mean) / baseline_mean) * 100
        
        return result
    
    def _test_context_manager_overhead(self) -> PerformanceTestResult:
        """Test context manager overhead"""
        durations = []
        
        for _ in range(self.iterations):
            start_time = time.perf_counter()
            
            with TelemetryContext("test_context_operation"):
                # Minimal operation inside context
                result = 42
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)
        
        return self._calculate_performance_result("context_manager_overhead", durations, self.target_overhead_ms)
    
    def _test_storage_telemetry_overhead(self) -> PerformanceTestResult:
        """Test storage-specific telemetry overhead"""
        durations = []
        
        for _ in range(self.iterations):
            start_time = time.perf_counter()
            self.collector.record_storage_metrics(
                operation="test_storage_operation",
                duration_ms=1.0,
                context_size=1000,
                cache_hit=False,
                success=True
            )
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)
        
        return self._calculate_performance_result("storage_telemetry_overhead", durations, self.target_overhead_ms)
    
    def _test_integration_overhead(self) -> PerformanceTestResult:
        """Test full integration overhead"""
        from .integration import integrate_telemetry_with_component
        
        class TestComponent:
            def test_method(self):
                return 42
        
        # Create test component with telemetry integration
        component = TestComponent()
        integrate_telemetry_with_component(component, "test_component")
        
        durations = []
        
        for _ in range(self.iterations):
            start_time = time.perf_counter()
            
            # Use integrated telemetry method
            component._record_operation("integrated_test", 1.0, True)
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)
        
        return self._calculate_performance_result("integration_overhead", durations, self.target_overhead_ms)
    
    def _test_aggregation_performance(self) -> PerformanceTestResult:
        """Test aggregation system performance"""
        durations = []
        
        # Generate some test data first
        for i in range(100):
            self.collector.record_operation(f"aggregation_test_op_{i % 5}", float(i), success=True)
        
        for _ in range(min(self.iterations // 10, 1000)):  # Fewer iterations for aggregation
            start_time = time.perf_counter()
            
            # Test aggregation operation
            summary = self.aggregator.get_aggregation_summary()
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)
        
        # Aggregation has higher overhead tolerance (10ms)
        return self._calculate_performance_result("aggregation_performance", durations, 10.0)
    
    def _test_monitoring_performance(self) -> PerformanceTestResult:
        """Test monitoring system performance"""
        durations = []
        
        for _ in range(min(self.iterations // 10, 1000)):  # Fewer iterations for monitoring
            start_time = time.perf_counter()
            
            # Test monitoring operation
            snapshot = self.monitor.get_performance_snapshot()
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)
        
        # Monitoring has higher overhead tolerance (50ms)
        return self._calculate_performance_result("monitoring_performance", durations, 50.0)
    
    def _test_alert_processing_performance(self) -> PerformanceTestResult:
        """Test alert processing performance"""
        durations = []
        
        for _ in range(min(self.iterations // 10, 1000)):  # Fewer iterations for alerting
            start_time = time.perf_counter()
            
            # Create and process a test alert
            alert = self.alerting.create_custom_alert(
                title="Performance Test Alert",
                message="Test alert for performance validation",
                severity=self.alerting.AlertSeverity.INFO
            )
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)
        
        # Alert processing has higher overhead tolerance (10ms)
        return self._calculate_performance_result("alert_processing_performance", durations, 10.0)
    
    def _test_concurrent_telemetry_overhead(self) -> PerformanceTestResult:
        """Test telemetry overhead under concurrent load"""
        import concurrent.futures
        
        def record_telemetry_batch():
            durations = []
            for _ in range(100):  # 100 operations per thread
                start_time = time.perf_counter()
                self.collector.record_operation("concurrent_test", 1.0, {"thread_id": threading.current_thread().ident}, True)
                end_time = time.perf_counter()
                durations.append((end_time - start_time) * 1000)
            return durations
        
        all_durations = []
        
        # Run with 10 concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(record_telemetry_batch) for _ in range(10)]
            
            for future in concurrent.futures.as_completed(futures):
                all_durations.extend(future.result())
        
        return self._calculate_performance_result("concurrent_telemetry_overhead", all_durations, self.target_overhead_ms * 2)  # Allow 2x overhead for concurrency
    
    def _test_memory_usage_impact(self) -> PerformanceTestResult:
        """Test memory usage impact of telemetry system"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate significant telemetry load
        operations_per_batch = 10000
        batches = 5
        
        memory_measurements = []
        
        for batch in range(batches):
            # Generate telemetry data
            for i in range(operations_per_batch):
                self.collector.record_operation(f"memory_test_op_{i % 10}", float(i), {"batch": batch}, True)
            
            # Measure memory after each batch
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(current_memory - initial_memory)
            
            time.sleep(0.1)  # Brief pause
        
        # Calculate memory growth rate
        if len(memory_measurements) > 1:
            memory_growth_mb = memory_measurements[-1] - memory_measurements[0]
            operations_total = operations_per_batch * batches
            memory_per_operation_kb = (memory_growth_mb * 1024) / operations_total
        else:
            memory_per_operation_kb = 0
        
        # Create dummy performance result (memory test doesn't use durations)
        result = PerformanceTestResult(
            test_name="memory_usage_impact",
            iterations=operations_per_batch * batches,
            mean_duration_ms=memory_per_operation_kb,  # Repurpose for memory per operation
            median_duration_ms=memory_measurements[-1] if memory_measurements else 0,
            p95_duration_ms=max(memory_measurements) if memory_measurements else 0,
            p99_duration_ms=memory_growth_mb,
            min_duration_ms=min(memory_measurements) if memory_measurements else 0,
            max_duration_ms=max(memory_measurements) if memory_measurements else 0,
            std_deviation_ms=statistics.stdev(memory_measurements) if len(memory_measurements) > 1 else 0,
            target_met=memory_per_operation_kb < 0.1,  # <0.1KB per operation target
            target_threshold_ms=0.1,  # KB threshold
            overhead_percentage=0.0,
            success_rate=100.0
        )
        
        logger.info(f"Memory impact: {memory_per_operation_kb:.6f} KB per operation, Total growth: {memory_growth_mb:.2f} MB")
        
        return result
    
    def _calculate_performance_result(
        self, 
        test_name: str, 
        durations: List[float], 
        target_ms: float
    ) -> PerformanceTestResult:
        """Calculate performance test result from duration measurements"""
        if not durations:
            return PerformanceTestResult(
                test_name=test_name,
                iterations=0,
                mean_duration_ms=float('inf'),
                median_duration_ms=float('inf'),
                p95_duration_ms=float('inf'),
                p99_duration_ms=float('inf'),
                min_duration_ms=float('inf'),
                max_duration_ms=float('inf'),
                std_deviation_ms=float('inf'),
                target_met=False,
                target_threshold_ms=target_ms
            )
        
        durations.sort()
        n = len(durations)
        
        result = PerformanceTestResult(
            test_name=test_name,
            iterations=n,
            mean_duration_ms=statistics.mean(durations),
            median_duration_ms=statistics.median(durations),
            p95_duration_ms=durations[int(n * 0.95)] if n > 20 else durations[-1],
            p99_duration_ms=durations[int(n * 0.99)] if n > 100 else durations[-1],
            min_duration_ms=min(durations),
            max_duration_ms=max(durations),
            std_deviation_ms=statistics.stdev(durations) if n > 1 else 0,
            target_met=durations[int(n * 0.95)] <= target_ms if n > 20 else durations[-1] <= target_ms,
            target_threshold_ms=target_ms,
            success_rate=100.0  # All tests succeed unless exception occurs
        )
        
        return result
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance validation report"""
        overall_target_met = all(result.target_met for result in self.test_results.values())
        
        # Key performance metrics
        core_overhead = self.test_results.get('core_telemetry_overhead')
        critical_tests_passed = 0
        total_critical_tests = 0
        
        critical_tests = ['core_telemetry_overhead', 'decorator_overhead', 'context_manager_overhead', 'storage_telemetry_overhead']
        
        for test_name in critical_tests:
            total_critical_tests += 1
            if test_name in self.test_results and self.test_results[test_name].target_met:
                critical_tests_passed += 1
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': 'PASS' if overall_target_met else 'FAIL',
            'telemetry_overhead_target_met': core_overhead.target_met if core_overhead else False,
            'core_telemetry_p95_ms': core_overhead.p95_duration_ms if core_overhead else float('inf'),
            'critical_tests_passed': f"{critical_tests_passed}/{total_critical_tests}",
            'total_tests_run': len(self.test_results),
            'tests_passed': len([r for r in self.test_results.values() if r.target_met]),
            'performance_summary': {
                'telemetry_overhead_target': f"<{self.target_overhead_ms}ms",
                'core_telemetry_performance': {
                    'p95_ms': core_overhead.p95_duration_ms if core_overhead else float('inf'),
                    'p99_ms': core_overhead.p99_duration_ms if core_overhead else float('inf'),
                    'mean_ms': core_overhead.mean_duration_ms if core_overhead else float('inf'),
                    'target_met': core_overhead.target_met if core_overhead else False
                },
                'decorator_overhead': {
                    test_name: {
                        'p95_ms': result.p95_duration_ms,
                        'overhead_percentage': result.overhead_percentage,
                        'target_met': result.target_met
                    }
                    for test_name, result in self.test_results.items()
                    if 'decorator' in test_name
                }
            },
            'detailed_results': {
                test_name: {
                    'iterations': result.iterations,
                    'mean_ms': round(result.mean_duration_ms, 6),
                    'median_ms': round(result.median_duration_ms, 6),
                    'p95_ms': round(result.p95_duration_ms, 6),
                    'p99_ms': round(result.p99_duration_ms, 6),
                    'min_ms': round(result.min_duration_ms, 6),
                    'max_ms': round(result.max_duration_ms, 6),
                    'std_dev_ms': round(result.std_deviation_ms, 6),
                    'target_threshold_ms': result.target_threshold_ms,
                    'target_met': result.target_met,
                    'overhead_percentage': round(result.overhead_percentage, 2) if result.overhead_percentage > 0 else None,
                    'success_rate': result.success_rate
                }
                for test_name, result in self.test_results.items()
            },
            'recommendations': self._generate_performance_recommendations()
        }
        
        return report
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check core telemetry overhead
        core_result = self.test_results.get('core_telemetry_overhead')
        if core_result and not core_result.target_met:
            recommendations.append(f"Core telemetry overhead ({core_result.p95_duration_ms:.3f}ms) exceeds target ({self.target_overhead_ms}ms) - consider optimizing record_operation method")
        
        # Check decorator overhead
        decorator_result = self.test_results.get('decorator_overhead')
        if decorator_result and decorator_result.overhead_percentage > 20:
            recommendations.append(f"Decorator overhead ({decorator_result.overhead_percentage:.1f}%) is high - consider lightweight decorator implementation")
        
        # Check memory usage
        memory_result = self.test_results.get('memory_usage_impact')
        if memory_result and not memory_result.target_met:
            recommendations.append("Memory usage per operation is high - implement more aggressive buffer management")
        
        # Check concurrent performance
        concurrent_result = self.test_results.get('concurrent_telemetry_overhead')
        if concurrent_result and not concurrent_result.target_met:
            recommendations.append("Concurrent telemetry performance is degraded - consider lock-free data structures")
        
        if not recommendations:
            recommendations.append("All performance targets met - system is operating within specifications")
        
        return recommendations
    
    def print_performance_summary(self):
        """Print performance validation summary to console"""
        report = self._generate_performance_report()
        
        print("\n" + "="*80)
        print("MEMMIMIC v2.0 TELEMETRY PERFORMANCE VALIDATION REPORT")
        print("="*80)
        print(f"Validation Time: {report['validation_timestamp']}")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Core Telemetry P95: {report['core_telemetry_p95_ms']:.6f}ms (Target: <{self.target_overhead_ms}ms)")
        print(f"Tests Passed: {report['tests_passed']}/{report['total_tests_run']}")
        print(f"Critical Tests: {report['critical_tests_passed']}")
        print()
        
        print("DETAILED RESULTS:")
        print("-" * 80)
        for test_name, result in report['detailed_results'].items():
            status = "PASS" if result['target_met'] else "FAIL"
            print(f"{test_name:35} | {status:4} | P95: {result['p95_ms']:8.3f}ms | Target: <{result['target_threshold_ms']:6.1f}ms")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 80)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("="*80)


def run_telemetry_performance_validation() -> Dict[str, Any]:
    """Run comprehensive telemetry performance validation"""
    validator = TelemetryPerformanceValidator()
    return validator.run_comprehensive_performance_validation()


def quick_overhead_test(iterations: int = 1000) -> float:
    """Quick telemetry overhead test"""
    collector = get_telemetry_collector()
    durations = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        collector.record_operation("quick_test", 1.0, {}, True)
        end_time = time.perf_counter()
        durations.append((end_time - start_time) * 1000)
    
    durations.sort()
    p95_duration = durations[int(len(durations) * 0.95)]
    
    print(f"Quick telemetry overhead test ({iterations} iterations):")
    print(f"P95 Duration: {p95_duration:.6f}ms")
    print(f"Target (<1ms): {'PASS' if p95_duration < 1.0 else 'FAIL'}")
    
    return p95_duration


if __name__ == "__main__":
    # Run performance validation
    validator = TelemetryPerformanceValidator()
    report = validator.run_comprehensive_performance_validation()
    validator.print_performance_summary()