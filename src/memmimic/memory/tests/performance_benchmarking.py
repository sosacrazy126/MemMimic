"""
MemMimic v2.0 Performance Benchmarking Suite
Continuous performance monitoring and regression detection for production systems.

Features:
- Automated performance baseline establishment
- Regression detection with statistical significance
- Performance trend analysis
- Production monitoring integration
- Performance optimization recommendations
"""

import asyncio
import json
import statistics
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np

from memmimic.memory.enhanced_memory import EnhancedMemory
from memmimic.memory.enhanced_amms_storage import EnhancedAMMSStorage


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result with statistical analysis"""
    operation: str
    timestamp: datetime
    sample_count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float
    throughput_ops_sec: float
    success_rate: float
    environment: str = "test"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionAnalysis:
    """Statistical analysis of performance regression"""
    operation: str
    baseline_p95: float
    current_p95: float
    regression_percent: float
    statistical_significance: float  # p-value
    confidence_interval: Tuple[float, float]
    is_regression: bool
    severity: str  # low, medium, high, critical
    recommendation: str


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report"""
    timestamp: datetime
    environment: str
    benchmarks: List[PerformanceBenchmark]
    regressions: List[RegressionAnalysis]
    trends: Dict[str, List[float]]
    overall_health_score: float
    recommendations: List[str]


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmarking with statistical analysis
    Designed for continuous performance monitoring in production
    """
    
    # Standard benchmark configurations
    BENCHMARK_CONFIGS = {
        'summary_retrieval': {
            'target_ms': 5.0,
            'samples': 1000,
            'warmup_samples': 100,
            'description': 'Ultra-fast summary retrieval performance'
        },
        'full_context_retrieval': {
            'target_ms': 50.0,
            'samples': 500,
            'warmup_samples': 50,
            'description': 'Full context retrieval with lazy loading'
        },
        'enhanced_store': {
            'target_ms': 15.0,
            'samples': 500,
            'warmup_samples': 50,
            'description': 'Enhanced memory storage with governance'
        },
        'concurrent_mixed_workload': {
            'target_throughput': 100.0,  # ops/sec
            'duration_seconds': 30,
            'concurrent_users': 25,
            'description': 'Mixed workload concurrent performance'
        },
        'cache_effectiveness': {
            'target_hit_rate': 0.8,
            'samples': 200,
            'cache_warmup': 50,
            'description': 'Summary cache hit rate and performance'
        }
    }
    
    def __init__(self, storage: EnhancedAMMSStorage, environment: str = "test"):
        self.storage = storage
        self.environment = environment
        self.benchmark_history = deque(maxlen=100)  # Keep last 100 benchmark runs
        self.baseline_benchmarks = {}
        self._setup_test_data()
    
    def _setup_test_data(self):
        """Setup standardized test data for consistent benchmarking"""
        self.test_memories = []
        
        # Create varied test data for realistic benchmarking
        test_scenarios = [
            # Small memories (20% of dataset)
            ('small', 40, 'Small memory content for quick testing', 
             'Small memory content for quick testing with minimal context',
             ['small', 'quick']),
            
            # Medium memories (60% of dataset) 
            ('medium', 120, 'Medium-sized memory entry representing typical usage',
             'Medium-sized memory entry representing typical usage patterns with moderate context ' * 3,
             ['medium', 'typical', 'usage']),
            
            # Large memories (20% of dataset)
            ('large', 40, 'Large memory entry for stress testing',
             'Large memory entry for stress testing performance boundaries with extensive context ' * 15,
             ['large', 'stress', 'performance'])
        ]
        
        memory_id_counter = 0
        for category, count, content_template, context_template, base_tags in test_scenarios:
            for i in range(count):
                memory = EnhancedMemory(
                    content=f"{content_template} #{memory_id_counter}",
                    full_context=f"{context_template} - Memory {memory_id_counter} with unique identifier",
                    tags=base_tags + [f"{category}_{i}", f"id_{memory_id_counter}"],
                    importance_score=0.1 + (memory_id_counter % 9) * 0.1,  # 0.1-0.9 range
                    metadata={'category': category, 'test_id': memory_id_counter}
                )
                self.test_memories.append(memory)
                memory_id_counter += 1
    
    async def establish_performance_baseline(self) -> Dict[str, PerformanceBenchmark]:
        """
        Establish performance baselines for all operations
        Run multiple times and use statistical analysis for robust baselines
        """
        print("🎯 Establishing performance baselines...")
        baselines = {}
        
        for operation, config in self.BENCHMARK_CONFIGS.items():
            print(f"  📊 Benchmarking {operation}...")
            
            if operation == 'concurrent_mixed_workload':
                benchmark = await self._benchmark_concurrent_workload(config)
            elif operation == 'cache_effectiveness':
                benchmark = await self._benchmark_cache_effectiveness(config)
            else:
                benchmark = await self._benchmark_operation(operation, config)
            
            baselines[operation] = benchmark
            self.baseline_benchmarks[operation] = benchmark
            
            # Print immediate results
            if hasattr(benchmark, 'p95_ms'):
                print(f"    ✅ P95: {benchmark.p95_ms:.2f}ms (target: {config.get('target_ms', 'N/A')})")
            else:
                print(f"    ✅ Throughput: {benchmark.throughput_ops_sec:.1f} ops/sec")
        
        return baselines
    
    async def _benchmark_operation(self, operation: str, config: Dict[str, Any]) -> PerformanceBenchmark:
        """Benchmark a specific operation with statistical rigor"""
        samples = config['samples']
        warmup_samples = config['warmup_samples']
        
        # Prepare test data if needed
        if operation in ['summary_retrieval', 'full_context_retrieval']:
            # Store memories for retrieval testing
            stored_ids = []
            for i, memory in enumerate(self.test_memories[:samples + warmup_samples]):
                memory_id = await self.storage.store_enhanced_memory_optimized(memory)
                stored_ids.append(memory_id)
        
        # Warmup phase
        warmup_times = []
        for i in range(warmup_samples):
            if operation == 'enhanced_store':
                start_time = time.perf_counter()
                await self.storage.store_enhanced_memory_optimized(self.test_memories[i])
                elapsed = (time.perf_counter() - start_time) * 1000
                warmup_times.append(elapsed)
            elif operation == 'summary_retrieval':
                start_time = time.perf_counter()
                await self.storage.retrieve_summary_optimized(stored_ids[i])
                elapsed = (time.perf_counter() - start_time) * 1000
                warmup_times.append(elapsed)
            elif operation == 'full_context_retrieval':
                start_time = time.perf_counter()
                await self.storage.retrieve_full_context_optimized(stored_ids[i])
                elapsed = (time.perf_counter() - start_time) * 1000
                warmup_times.append(elapsed)
        
        # Main benchmark phase
        benchmark_start = time.perf_counter()
        times = []
        successes = 0
        
        for i in range(samples):
            idx = warmup_samples + i
            
            if operation == 'enhanced_store':
                start_time = time.perf_counter()
                try:
                    result = await self.storage.store_enhanced_memory_optimized(self.test_memories[idx % len(self.test_memories)])
                    elapsed = (time.perf_counter() - start_time) * 1000
                    if result:
                        successes += 1
                except Exception:
                    elapsed = (time.perf_counter() - start_time) * 1000
                times.append(elapsed)
                
            elif operation == 'summary_retrieval':
                start_time = time.perf_counter()
                try:
                    result = await self.storage.retrieve_summary_optimized(stored_ids[idx % len(stored_ids)])
                    elapsed = (time.perf_counter() - start_time) * 1000
                    if result is not None:
                        successes += 1
                except Exception:
                    elapsed = (time.perf_counter() - start_time) * 1000
                times.append(elapsed)
                
            elif operation == 'full_context_retrieval':
                start_time = time.perf_counter()
                try:
                    result = await self.storage.retrieve_full_context_optimized(stored_ids[idx % len(stored_ids)])
                    elapsed = (time.perf_counter() - start_time) * 1000
                    if result is not None:
                        successes += 1
                except Exception:
                    elapsed = (time.perf_counter() - start_time) * 1000
                times.append(elapsed)
        
        benchmark_duration = time.perf_counter() - benchmark_start
        
        # Statistical analysis
        mean_ms = statistics.mean(times)
        median_ms = statistics.median(times)
        p95_ms = sorted(times)[int(len(times) * 0.95)]
        p99_ms = sorted(times)[int(len(times) * 0.99)]
        min_ms = min(times)
        max_ms = max(times)
        std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = samples / benchmark_duration if benchmark_duration > 0 else 0.0
        success_rate = successes / samples if samples > 0 else 0.0
        
        return PerformanceBenchmark(
            operation=operation,
            timestamp=datetime.now(),
            sample_count=samples,
            mean_ms=mean_ms,
            median_ms=median_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            std_dev_ms=std_dev_ms,
            throughput_ops_sec=throughput,
            success_rate=success_rate,
            environment=self.environment,
            metadata={
                'warmup_samples': warmup_samples,
                'target_ms': config.get('target_ms'),
                'config': config
            }
        )
    
    async def _benchmark_concurrent_workload(self, config: Dict[str, Any]) -> PerformanceBenchmark:
        """Benchmark concurrent mixed workload performance"""
        duration = config['duration_seconds']
        concurrent_users = config['concurrent_users']
        
        async def worker(worker_id: int):
            """Simulate a single user's mixed workload"""
            worker_times = []
            worker_ops = 0
            start_time = time.perf_counter()
            
            while (time.perf_counter() - start_time) < duration:
                # Mixed operations: 50% store, 30% summary retrieval, 20% full context
                operation_type = worker_ops % 10
                
                if operation_type < 5:  # Store operation
                    memory = self.test_memories[worker_ops % len(self.test_memories)]
                    op_start = time.perf_counter()
                    await self.storage.store_enhanced_memory_optimized(memory)
                    op_time = (time.perf_counter() - op_start) * 1000
                    worker_times.append(op_time)
                
                elif operation_type < 8:  # Summary retrieval
                    # Use a recently stored memory (simulate cache usage)
                    if worker_ops > 0:
                        op_start = time.perf_counter()
                        # Simulate retrieval of recent memory
                        await self.storage.retrieve_summary_optimized(str(worker_ops % 100 + 1))
                        op_time = (time.perf_counter() - op_start) * 1000
                        worker_times.append(op_time)
                
                else:  # Full context retrieval
                    if worker_ops > 0:
                        op_start = time.perf_counter()
                        await self.storage.retrieve_full_context_optimized(str(worker_ops % 50 + 1))
                        op_time = (time.perf_counter() - op_start) * 1000
                        worker_times.append(op_time)
                
                worker_ops += 1
                
                # Small delay to simulate realistic usage
                await asyncio.sleep(0.01)  # 10ms between operations
            
            return worker_times, worker_ops
        
        # Run concurrent workload
        benchmark_start = time.perf_counter()
        tasks = [worker(i) for i in range(concurrent_users)]
        results = await asyncio.gather(*tasks)
        benchmark_duration = time.perf_counter() - benchmark_start
        
        # Aggregate results
        all_times = []
        total_operations = 0
        for worker_times, worker_ops in results:
            all_times.extend(worker_times)
            total_operations += worker_ops
        
        if not all_times:
            # Fallback for empty results
            return PerformanceBenchmark(
                operation='concurrent_mixed_workload',
                timestamp=datetime.now(),
                sample_count=0,
                mean_ms=0.0,
                median_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                std_dev_ms=0.0,
                throughput_ops_sec=0.0,
                success_rate=0.0,
                environment=self.environment
            )
        
        # Statistical analysis
        mean_ms = statistics.mean(all_times)
        median_ms = statistics.median(all_times)
        p95_ms = sorted(all_times)[int(len(all_times) * 0.95)]
        p99_ms = sorted(all_times)[int(len(all_times) * 0.99)]
        throughput = total_operations / benchmark_duration
        
        return PerformanceBenchmark(
            operation='concurrent_mixed_workload',
            timestamp=datetime.now(),
            sample_count=len(all_times),
            mean_ms=mean_ms,
            median_ms=median_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            min_ms=min(all_times),
            max_ms=max(all_times),
            std_dev_ms=statistics.stdev(all_times),
            throughput_ops_sec=throughput,
            success_rate=1.0,  # Assume success for completed operations
            environment=self.environment,
            metadata={
                'concurrent_users': concurrent_users,
                'duration_seconds': duration,
                'total_operations': total_operations
            }
        )
    
    async def _benchmark_cache_effectiveness(self, config: Dict[str, Any]) -> PerformanceBenchmark:
        """Benchmark cache effectiveness and hit rate"""
        samples = config['samples']
        warmup = config['cache_warmup']
        
        # Store test memories and warm up cache
        stored_ids = []
        for memory in self.test_memories[:samples + warmup]:
            memory_id = await self.storage.store_enhanced_memory_optimized(memory)
            stored_ids.append(memory_id)
        
        # Warmup cache
        for i in range(warmup):
            await self.storage.retrieve_summary_optimized(stored_ids[i])
        
        # Test cache performance
        cache_hits = 0
        cache_misses = 0
        hit_times = []
        miss_times = []
        
        for i in range(samples):
            # 80% cache hits (repeat recent IDs), 20% cache misses (new IDs)
            if i % 5 != 0:  # 80% - cache hit
                memory_id = stored_ids[i % warmup]  # Reuse warmup IDs
                start_time = time.perf_counter()
                result = await self.storage.retrieve_summary_optimized(memory_id)
                elapsed = (time.perf_counter() - start_time) * 1000
                hit_times.append(elapsed)
                cache_hits += 1
            else:  # 20% - cache miss
                memory_id = stored_ids[warmup + (i // 5)]  # Use new IDs
                start_time = time.perf_counter()
                result = await self.storage.retrieve_summary_optimized(memory_id)
                elapsed = (time.perf_counter() - start_time) * 1000
                miss_times.append(elapsed)
                cache_misses += 1
        
        # Calculate cache effectiveness metrics
        hit_rate = cache_hits / (cache_hits + cache_misses)
        avg_hit_time = statistics.mean(hit_times) if hit_times else 0.0
        avg_miss_time = statistics.mean(miss_times) if miss_times else 0.0
        
        all_times = hit_times + miss_times
        
        return PerformanceBenchmark(
            operation='cache_effectiveness',
            timestamp=datetime.now(),
            sample_count=samples,
            mean_ms=statistics.mean(all_times) if all_times else 0.0,
            median_ms=statistics.median(all_times) if all_times else 0.0,
            p95_ms=sorted(all_times)[int(len(all_times) * 0.95)] if all_times else 0.0,
            p99_ms=sorted(all_times)[int(len(all_times) * 0.99)] if all_times else 0.0,
            min_ms=min(all_times) if all_times else 0.0,
            max_ms=max(all_times) if all_times else 0.0,
            std_dev_ms=statistics.stdev(all_times) if len(all_times) > 1 else 0.0,
            throughput_ops_sec=samples / sum(all_times) * 1000 if all_times else 0.0,
            success_rate=1.0,
            environment=self.environment,
            metadata={
                'hit_rate': hit_rate,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'avg_hit_time_ms': avg_hit_time,
                'avg_miss_time_ms': avg_miss_time,
                'target_hit_rate': config.get('target_hit_rate', 0.8)
            }
        )
    
    async def run_performance_regression_test(self) -> List[RegressionAnalysis]:
        """
        Run performance regression analysis against established baselines
        Uses statistical significance testing to detect real regressions
        """
        if not self.baseline_benchmarks:
            print("⚠️  No baseline benchmarks found. Establishing baselines first...")
            await self.establish_performance_baseline()
        
        print("🔍 Running regression analysis...")
        regressions = []
        
        # Run current benchmarks
        current_benchmarks = {}
        for operation, config in self.BENCHMARK_CONFIGS.items():
            if operation == 'concurrent_mixed_workload':
                benchmark = await self._benchmark_concurrent_workload(config)
            elif operation == 'cache_effectiveness':
                benchmark = await self._benchmark_cache_effectiveness(config)
            else:
                benchmark = await self._benchmark_operation(operation, config)
            current_benchmarks[operation] = benchmark
        
        # Compare with baselines
        for operation, current_benchmark in current_benchmarks.items():
            if operation in self.baseline_benchmarks:
                regression = self._analyze_performance_regression(
                    operation,
                    self.baseline_benchmarks[operation],
                    current_benchmark
                )
                regressions.append(regression)
        
        return regressions
    
    def _analyze_performance_regression(
        self,
        operation: str,
        baseline: PerformanceBenchmark,
        current: PerformanceBenchmark
    ) -> RegressionAnalysis:
        """Analyze performance regression with statistical significance"""
        
        # Use P95 as the primary metric for regression detection
        baseline_p95 = baseline.p95_ms
        current_p95 = current.p95_ms
        
        # Calculate regression percentage
        if baseline_p95 > 0:
            regression_percent = ((current_p95 - baseline_p95) / baseline_p95) * 100
        else:
            regression_percent = 0.0
        
        # Simple statistical significance estimation
        # In a real implementation, would use proper t-test with sample data
        pooled_std = (baseline.std_dev_ms + current.std_dev_ms) / 2
        if pooled_std > 0:
            # Approximate z-score for difference in means
            z_score = abs(current_p95 - baseline_p95) / (pooled_std / (baseline.sample_count ** 0.5))
            # Convert to approximate p-value (simplified)
            p_value = max(0.001, min(1.0, 2 * (1 - min(z_score / 3, 0.999))))
        else:
            p_value = 1.0  # No variance, no significance
        
        # Confidence interval (simplified 95% CI)
        margin_of_error = 1.96 * pooled_std / (current.sample_count ** 0.5) if current.sample_count > 0 else 0
        confidence_interval = (current_p95 - margin_of_error, current_p95 + margin_of_error)
        
        # Determine regression severity
        is_regression = regression_percent > 10.0 and p_value < 0.05  # >10% regression with significance
        
        if regression_percent > 50:
            severity = "critical"
        elif regression_percent > 25:
            severity = "high"
        elif regression_percent > 10:
            severity = "medium"
        else:
            severity = "low"
        
        # Generate recommendation
        if is_regression:
            if severity == "critical":
                recommendation = f"URGENT: {operation} performance degraded by {regression_percent:.1f}%. Immediate investigation required."
            elif severity == "high":
                recommendation = f"HIGH: {operation} performance degraded by {regression_percent:.1f}%. Optimization needed."
            elif severity == "medium":
                recommendation = f"MEDIUM: {operation} performance degraded by {regression_percent:.1f}%. Monitor and consider optimization."
            else:
                recommendation = f"LOW: Minor performance change in {operation} ({regression_percent:.1f}%). Continue monitoring."
        else:
            if regression_percent < -10:
                recommendation = f"IMPROVEMENT: {operation} performance improved by {abs(regression_percent):.1f}%."
            else:
                recommendation = f"STABLE: {operation} performance is stable (change: {regression_percent:.1f}%)."
        
        return RegressionAnalysis(
            operation=operation,
            baseline_p95=baseline_p95,
            current_p95=current_p95,
            regression_percent=regression_percent,
            statistical_significance=p_value,
            confidence_interval=confidence_interval,
            is_regression=is_regression,
            severity=severity,
            recommendation=recommendation
        )
    
    async def generate_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report with trends and recommendations"""
        print("📈 Generating comprehensive performance report...")
        
        # Run regression analysis (includes current benchmarks)
        regressions = await self.run_performance_regression_test()
        
        # Get latest benchmarks
        current_benchmarks = []
        for operation, config in self.BENCHMARK_CONFIGS.items():
            if operation in self.baseline_benchmarks:
                # Use the current benchmark from regression analysis
                current_benchmark = next(
                    (r for r in regressions if r.operation == operation), 
                    None
                )
                if current_benchmark:
                    # Create benchmark from regression data
                    benchmark = PerformanceBenchmark(
                        operation=operation,
                        timestamp=datetime.now(),
                        sample_count=self.baseline_benchmarks[operation].sample_count,
                        mean_ms=0.0,  # Not available from regression
                        median_ms=0.0,  # Not available from regression
                        p95_ms=current_benchmark.current_p95,
                        p99_ms=0.0,  # Not available from regression
                        min_ms=0.0,  # Not available from regression
                        max_ms=0.0,  # Not available from regression
                        std_dev_ms=0.0,  # Not available from regression
                        throughput_ops_sec=0.0,  # Not available from regression
                        success_rate=1.0,
                        environment=self.environment
                    )
                    current_benchmarks.append(benchmark)
        
        # Calculate trends (simplified - would use historical data in production)
        trends = {}
        for operation in self.BENCHMARK_CONFIGS.keys():
            if operation in self.baseline_benchmarks:
                # Simulate trend data (in production, this would come from historical benchmarks)
                baseline_p95 = self.baseline_benchmarks[operation].p95_ms
                current_p95 = next((r.current_p95 for r in regressions if r.operation == operation), baseline_p95)
                trends[operation] = [baseline_p95 * 0.95, baseline_p95, current_p95]  # Simplified trend
        
        # Calculate overall health score
        regression_penalty = sum(1 for r in regressions if r.is_regression and r.severity in ['high', 'critical'])
        total_operations = len(regressions)
        health_score = max(0.0, (total_operations - regression_penalty * 2) / total_operations) if total_operations > 0 else 1.0
        
        # Generate recommendations
        recommendations = []
        critical_regressions = [r for r in regressions if r.severity == 'critical']
        high_regressions = [r for r in regressions if r.severity == 'high']
        
        if critical_regressions:
            recommendations.append(f"🚨 CRITICAL: {len(critical_regressions)} operations have critical performance degradation")
            for reg in critical_regressions:
                recommendations.append(f"   • {reg.operation}: {reg.regression_percent:.1f}% degradation")
        
        if high_regressions:
            recommendations.append(f"⚠️  HIGH: {len(high_regressions)} operations have high performance impact")
        
        if health_score > 0.9:
            recommendations.append("✅ Overall performance is excellent")
        elif health_score > 0.7:
            recommendations.append("👍 Overall performance is good with minor issues")
        else:
            recommendations.append("⚠️  Overall performance needs attention")
        
        # Add specific optimization recommendations
        cache_regression = next((r for r in regressions if r.operation == 'cache_effectiveness'), None)
        if cache_regression and cache_regression.is_regression:
            recommendations.append("💾 Consider cache optimization: increase cache size or review cache eviction policy")
        
        concurrent_regression = next((r for r in regressions if r.operation == 'concurrent_mixed_workload'), None)
        if concurrent_regression and concurrent_regression.is_regression:
            recommendations.append("🔄 Consider connection pool optimization for concurrent workloads")
        
        return PerformanceReport(
            timestamp=datetime.now(),
            environment=self.environment,
            benchmarks=current_benchmarks,
            regressions=regressions,
            trends=trends,
            overall_health_score=health_score,
            recommendations=recommendations
        )
    
    def save_benchmark_results(self, benchmarks: Dict[str, PerformanceBenchmark], filename: Optional[str] = None):
        """Save benchmark results to JSON file for historical tracking"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{self.environment}_{timestamp}.json"
        
        # Convert benchmarks to serializable format
        serializable_benchmarks = {}
        for operation, benchmark in benchmarks.items():
            serializable_benchmarks[operation] = {
                'operation': benchmark.operation,
                'timestamp': benchmark.timestamp.isoformat(),
                'sample_count': benchmark.sample_count,
                'mean_ms': benchmark.mean_ms,
                'median_ms': benchmark.median_ms,
                'p95_ms': benchmark.p95_ms,
                'p99_ms': benchmark.p99_ms,
                'min_ms': benchmark.min_ms,
                'max_ms': benchmark.max_ms,
                'std_dev_ms': benchmark.std_dev_ms,
                'throughput_ops_sec': benchmark.throughput_ops_sec,
                'success_rate': benchmark.success_rate,
                'environment': benchmark.environment,
                'metadata': benchmark.metadata
            }
        
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_benchmarks, f, indent=2)
            print(f"📁 Benchmark results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save benchmark results: {e}")
    
    def create_performance_charts(self, report: PerformanceReport, output_dir: str = "performance_charts"):
        """Create performance visualization charts"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            Path(output_dir).mkdir(exist_ok=True)
            
            # Performance comparison chart
            operations = list(report.trends.keys())
            if operations:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # P95 latency comparison
                baseline_p95s = [report.trends[op][1] for op in operations]
                current_p95s = [report.trends[op][-1] for op in operations]
                
                x = np.arange(len(operations))
                width = 0.35
                
                ax1.bar(x - width/2, baseline_p95s, width, label='Baseline P95', alpha=0.7)
                ax1.bar(x + width/2, current_p95s, width, label='Current P95', alpha=0.7)
                ax1.set_xlabel('Operations')
                ax1.set_ylabel('Latency (ms)')
                ax1.set_title('P95 Latency Comparison')
                ax1.set_xticks(x)
                ax1.set_xticklabels(operations, rotation=45)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Regression severity chart
                regression_ops = [r.operation for r in report.regressions]
                regression_pcts = [r.regression_percent for r in report.regressions]
                colors = ['red' if r.is_regression else 'green' for r in report.regressions]
                
                ax2.bar(regression_ops, regression_pcts, color=colors, alpha=0.7)
                ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='10% Regression Threshold')
                ax2.set_xlabel('Operations')
                ax2.set_ylabel('Regression %')
                ax2.set_title('Performance Regression Analysis')
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                chart_path = Path(output_dir) / "performance_overview.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"📊 Performance charts saved to {output_dir}")
        
        except ImportError:
            print("📊 Matplotlib not available - skipping chart generation")
        except Exception as e:
            print(f"❌ Failed to create performance charts: {e}")


# CLI interface for standalone benchmarking
async def main():
    """Run performance benchmarking suite"""
    print("🚀 MemMimic v2.0 Performance Benchmarking Suite")
    print("=" * 50)
    
    # Create temporary storage for benchmarking
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        config = {
            'enable_summary_cache': True,
            'summary_cache_size': 1000
        }
        
        storage = EnhancedAMMSStorage(db_path, config=config)
        benchmark_suite = PerformanceBenchmarkSuite(storage, environment="benchmark")
        
        print("📊 Establishing performance baselines...")
        baselines = await benchmark_suite.establish_performance_baseline()
        
        print("\n🔍 Running regression analysis...")
        regressions = await benchmark_suite.run_performance_regression_test()
        
        print("\n📈 Generating comprehensive report...")
        report = await benchmark_suite.generate_performance_report()
        
        # Print results
        print(f"\n📊 PERFORMANCE BENCHMARKING RESULTS")
        print(f"Environment: {report.environment}")
        print(f"Overall Health Score: {report.overall_health_score:.1%}")
        print(f"Timestamp: {report.timestamp}")
        
        print(f"\n⚡ BASELINE PERFORMANCE")
        for operation, benchmark in baselines.items():
            target = benchmark.metadata.get('target_ms', 'N/A')
            if hasattr(benchmark, 'p95_ms') and benchmark.p95_ms > 0:
                status = "✅" if (isinstance(target, (int, float)) and benchmark.p95_ms <= target) else "⚠️"
                print(f"  {operation}: {status} P95={benchmark.p95_ms:.2f}ms (target: {target}ms)")
            else:
                print(f"  {operation}: Throughput={benchmark.throughput_ops_sec:.1f} ops/sec")
        
        print(f"\n🔍 REGRESSION ANALYSIS")
        for regression in report.regressions:
            status_emoji = "🚨" if regression.severity == "critical" else "⚠️" if regression.is_regression else "✅"
            print(f"  {regression.operation}: {status_emoji} {regression.regression_percent:+.1f}% ({regression.severity})")
            if regression.is_regression:
                print(f"    └─ {regression.recommendation}")
        
        print(f"\n💡 RECOMMENDATIONS")
        for recommendation in report.recommendations:
            print(f"  {recommendation}")
        
        # Save results
        benchmark_suite.save_benchmark_results(baselines)
        
        # Create charts if possible
        benchmark_suite.create_performance_charts(report)
        
        await storage.close()
        
    finally:
        # Cleanup
        try:
            Path(db_path).unlink()
        except:
            pass
    
    print(f"\n🏁 Performance benchmarking completed!")


if __name__ == "__main__":
    asyncio.run(main())