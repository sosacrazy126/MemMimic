#!/usr/bin/env python3
"""
MemMimic v2.0 Integration Test Runner
Comprehensive integration testing orchestration with production readiness validation.

This is the primary entry point for Task #10: Comprehensive Performance Validation Suite.

Features:
- Orchestrates all integration testing components
- Validates all v2.0 performance targets
- Provides deployment readiness assessment
- Generates comprehensive reports
- Supports CI/CD integration
"""

import argparse
import asyncio
import json
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from test_integration_validation import (
    ProductionReadinessChecker,
    PerformanceValidator,
    IntegrationTestSuite,
    LoadTestRunner,
    QualityGateValidator
)
from performance_benchmarking import PerformanceBenchmarkSuite
from production_monitoring import ProductionMonitor
from memmimic.memory.enhanced_amms_storage import EnhancedAMMSStorage


class IntegrationTestOrchestrator:
    """
    Master orchestrator for all MemMimic v2.0 integration testing
    Coordinates performance validation, integration testing, and production readiness
    """
    
    def __init__(self, config: Optional[Dict] = None, environment: str = "test"):
        self.config = config or self._get_default_config()
        self.environment = environment
        self.storage = None
        self.test_results = {}
        self.start_time = datetime.now()
    
    def _get_default_config(self) -> Dict:
        """Get default test configuration"""
        return {
            'database': {
                'enable_summary_cache': True,
                'summary_cache_size': 1000,
                'pool_size': 5
            },
            'testing': {
                'run_performance_tests': True,
                'run_integration_tests': True,
                'run_load_tests': True,
                'run_production_readiness': True,
                'run_benchmarking': True,
                'run_monitoring_demo': False
            },
            'performance_targets': {
                'summary_retrieval_ms': 5.0,
                'full_context_retrieval_ms': 50.0,
                'enhanced_remember_ms': 15.0,
                'governance_overhead_ms': 10.0,
                'telemetry_overhead_ms': 1.0
            },
            'quality_gates': {
                'min_success_rate': 0.95,
                'max_failure_rate': 0.05,
                'min_health_score': 0.8
            }
        }
    
    async def setup(self):
        """Setup test environment and storage"""
        print("🔧 Setting up integration test environment...")
        
        # Create temporary database
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        db_path = self.temp_file.name
        
        # Initialize enhanced storage
        self.storage = EnhancedAMMSStorage(
            db_path=db_path,
            pool_size=self.config['database'].get('pool_size', 5),
            config=self.config['database']
        )
        
        print(f"✅ Test environment ready (database: {db_path})")
        print(f"   Cache enabled: {self.config['database']['enable_summary_cache']}")
        print(f"   Cache size: {self.config['database']['summary_cache_size']}")
    
    async def teardown(self):
        """Clean up test environment"""
        print("🧹 Cleaning up test environment...")
        
        if self.storage:
            await self.storage.close()
        
        if hasattr(self, 'temp_file'):
            try:
                Path(self.temp_file.name).unlink()
                print("✅ Test database cleaned up")
            except Exception as e:
                print(f"⚠️  Database cleanup warning: {e}")
    
    async def run_performance_validation(self) -> Dict:
        """Run comprehensive performance validation"""
        print("\n" + "="*60)
        print("⚡ PERFORMANCE VALIDATION")
        print("="*60)
        
        validator = PerformanceValidator(self.storage)
        performance_results = await validator.validate_all_performance_targets()
        
        # Print detailed results
        all_targets_met = True
        for test_name, result in performance_results.items():
            status = "✅ PASS" if result.target_met else "❌ FAIL"
            print(f"  {test_name}: {status}")
            print(f"    Target: {result.target_value}ms | Actual: {result.actual_value:.2f}ms")
            print(f"    Message: {result.message}")
            
            if not result.target_met:
                all_targets_met = False
        
        success_rate = sum(1 for r in performance_results.values() if r.target_met) / len(performance_results)
        print(f"\n📊 Performance Summary: {len(performance_results)} tests, {success_rate:.1%} success rate")
        
        return {
            'success': all_targets_met,
            'results': performance_results,
            'success_rate': success_rate,
            'total_tests': len(performance_results)
        }
    
    async def run_integration_testing(self) -> Dict:
        """Run integration testing across all components"""
        print("\n" + "="*60) 
        print("🔗 INTEGRATION TESTING")
        print("="*60)
        
        integration_tester = IntegrationTestSuite(self.storage)
        integration_results = await integration_tester.run_integration_tests()
        
        # Print results
        all_passed = True
        for test_name, result in integration_results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
            print(f"    Message: {result.message}")
            
            if not result.passed:
                all_passed = False
        
        success_rate = sum(1 for r in integration_results.values() if r.passed) / len(integration_results)
        print(f"\n📊 Integration Summary: {len(integration_results)} tests, {success_rate:.1%} success rate")
        
        return {
            'success': all_passed,
            'results': integration_results,
            'success_rate': success_rate,
            'total_tests': len(integration_results)
        }
    
    async def run_load_testing(self) -> Dict:
        """Run load testing and stress testing"""
        print("\n" + "="*60)
        print("📈 LOAD TESTING")
        print("="*60)
        
        load_tester = LoadTestRunner(self.storage)
        load_results = await load_tester.run_load_tests()
        
        # Print results
        all_passed = True
        for test_name, result in load_results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
            print(f"    Target: {result.target_value} | Actual: {result.actual_value:.2f}")
            print(f"    Message: {result.message}")
            
            if not result.passed:
                all_passed = False
        
        success_rate = sum(1 for r in load_results.values() if r.passed) / len(load_results)
        print(f"\n📊 Load Testing Summary: {len(load_results)} tests, {success_rate:.1%} success rate")
        
        return {
            'success': all_passed,
            'results': load_results,
            'success_rate': success_rate,
            'total_tests': len(load_results)
        }
    
    async def run_benchmarking(self) -> Dict:
        """Run performance benchmarking with statistical analysis"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE BENCHMARKING")
        print("="*60)
        
        benchmark_suite = PerformanceBenchmarkSuite(self.storage, self.environment)
        
        # Establish baselines
        print("🎯 Establishing performance baselines...")
        baselines = await benchmark_suite.establish_performance_baseline()
        
        # Run regression analysis
        print("🔍 Running regression analysis...")
        regressions = await benchmark_suite.run_performance_regression_test()
        
        # Generate comprehensive report
        print("📈 Generating performance report...")
        report = await benchmark_suite.generate_performance_report()
        
        # Print benchmark results
        print("\n⚡ BASELINE PERFORMANCE:")
        for operation, benchmark in baselines.items():
            target = benchmark.metadata.get('target_ms', 'N/A')
            if hasattr(benchmark, 'p95_ms') and benchmark.p95_ms > 0:
                status = "✅" if (isinstance(target, (int, float)) and benchmark.p95_ms <= target) else "⚠️"
                print(f"  {operation}: {status} P95={benchmark.p95_ms:.2f}ms (target: {target}ms)")
            else:
                print(f"  {operation}: Throughput={benchmark.throughput_ops_sec:.1f} ops/sec")
        
        print(f"\n🔍 REGRESSION ANALYSIS:")
        regression_found = False
        for regression in regressions:
            if regression.is_regression:
                regression_found = True
                severity_emoji = "🚨" if regression.severity == "critical" else "⚠️"
                print(f"  {regression.operation}: {severity_emoji} {regression.regression_percent:+.1f}% ({regression.severity})")
        
        if not regression_found:
            print("  ✅ No significant regressions detected")
        
        print(f"\n📈 Overall Health Score: {report.overall_health_score:.1%}")
        
        # Save results
        benchmark_suite.save_benchmark_results(baselines, f"baselines_{self.environment}.json")
        
        return {
            'success': report.overall_health_score >= 0.8,
            'health_score': report.overall_health_score,
            'baselines': baselines,
            'regressions': regressions,
            'report': report
        }
    
    async def run_production_readiness(self) -> Dict:
        """Run comprehensive production readiness validation"""
        print("\n" + "="*60)
        print("🚀 PRODUCTION READINESS VALIDATION")
        print("="*60)
        
        checker = ProductionReadinessChecker(self.storage)
        result = await checker.run_comprehensive_validation()
        
        # Print comprehensive results
        print(f"📊 OVERALL STATUS: {result.overall_status}")
        print(f"   Total Tests: {result.total_tests}")
        print(f"   Passed: {result.passed_tests}")
        print(f"   Failed: {result.failed_tests}")
        print(f"   Execution Time: {result.execution_time_ms:.0f}ms")
        
        print(f"\n🎯 QUALITY GATES:")
        for gate, passed in result.quality_gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {gate}: {status}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for recommendation in result.recommendations:
            print(f"   {recommendation}")
        
        return {
            'success': result.overall_status == "PASSED",
            'status': result.overall_status,
            'result': result
        }
    
    async def run_monitoring_demo(self) -> Dict:
        """Run production monitoring demonstration"""
        print("\n" + "="*60)
        print("📡 PRODUCTION MONITORING DEMO")
        print("="*60)
        
        monitor = ProductionMonitor(self.storage, self.environment)
        
        print("🎯 Setting up monitoring system...")
        
        # Run monitoring for a short demo period
        try:
            await asyncio.wait_for(monitor.start_monitoring(), timeout=30)  # 30 seconds demo
        except asyncio.TimeoutError:
            print("⏱️  Demo monitoring period completed")
        
        await monitor.stop_monitoring()
        
        # Get dashboard data
        dashboard_data = monitor.get_monitoring_dashboard_data()
        health_score = dashboard_data['health_score']['overall']
        
        print(f"📊 Monitoring Results:")
        print(f"   Health Score: {health_score:.1%}")
        print(f"   Active Alerts: {dashboard_data['alerts']['active']}")
        
        return {
            'success': health_score >= 0.8,
            'health_score': health_score,
            'dashboard_data': dashboard_data
        }
    
    async def run_all_tests(self) -> Dict:
        """Run all integration tests and return comprehensive results"""
        print(f"🚀 MemMimic v2.0 Comprehensive Integration Testing")
        print(f"Environment: {self.environment}")
        print(f"Started at: {self.start_time}")
        print("=" * 80)
        
        all_results = {}
        overall_success = True
        
        try:
            # Performance Validation
            if self.config['testing']['run_performance_tests']:
                perf_results = await self.run_performance_validation()
                all_results['performance'] = perf_results
                if not perf_results['success']:
                    overall_success = False
            
            # Integration Testing
            if self.config['testing']['run_integration_tests']:
                integration_results = await self.run_integration_testing()
                all_results['integration'] = integration_results
                if not integration_results['success']:
                    overall_success = False
            
            # Load Testing
            if self.config['testing']['run_load_tests']:
                load_results = await self.run_load_testing()
                all_results['load_testing'] = load_results
                if not load_results['success']:
                    overall_success = False
            
            # Benchmarking
            if self.config['testing']['run_benchmarking']:
                benchmark_results = await self.run_benchmarking()
                all_results['benchmarking'] = benchmark_results
                if not benchmark_results['success']:
                    overall_success = False
            
            # Production Readiness
            if self.config['testing']['run_production_readiness']:
                readiness_results = await self.run_production_readiness()
                all_results['production_readiness'] = readiness_results
                if not readiness_results['success']:
                    overall_success = False
            
            # Monitoring Demo (optional)
            if self.config['testing']['run_monitoring_demo']:
                monitoring_results = await self.run_monitoring_demo()
                all_results['monitoring'] = monitoring_results
                # Don't fail overall on monitoring demo
        
        except Exception as e:
            print(f"❌ Critical error during testing: {e}")
            traceback.print_exc()
            overall_success = False
            all_results['error'] = str(e)
        
        # Calculate final statistics
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        final_results = {
            'overall_success': overall_success,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_seconds': total_duration,
            'environment': self.environment,
            'test_results': all_results,
            'summary': self._generate_summary(all_results, overall_success)
        }
        
        return final_results
    
    def _generate_summary(self, results: Dict, overall_success: bool) -> Dict:
        """Generate test execution summary"""
        total_tests = 0
        passed_tests = 0
        
        for category, category_results in results.items():
            if isinstance(category_results, dict) and 'total_tests' in category_results:
                total_tests += category_results['total_tests']
                if 'success_rate' in category_results:
                    passed_tests += int(category_results['total_tests'] * category_results['success_rate'])
        
        return {
            'overall_status': "PASSED" if overall_success else "FAILED",
            'total_test_categories': len(results),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'deployment_recommendation': (
                "✅ APPROVED FOR PRODUCTION DEPLOYMENT" if overall_success
                else "❌ NOT READY FOR PRODUCTION - ADDRESS FAILURES"
            )
        }
    
    def print_final_report(self, results: Dict):
        """Print comprehensive final report"""
        print("\n" + "="*80)
        print("📋 FINAL INTEGRATION TEST REPORT")
        print("="*80)
        
        summary = results['summary']
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Test Duration: {results['total_duration_seconds']:.1f} seconds")
        print(f"Total Test Categories: {summary['total_test_categories']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        
        print(f"\n🎯 DEPLOYMENT RECOMMENDATION:")
        print(f"   {summary['deployment_recommendation']}")
        
        # Print category summaries
        print(f"\n📊 CATEGORY RESULTS:")
        for category, category_results in results['test_results'].items():
            if isinstance(category_results, dict) and 'success' in category_results:
                status = "✅ PASS" if category_results['success'] else "❌ FAIL"
                print(f"   {category}: {status}")
        
        # Performance highlights
        if 'performance' in results['test_results']:
            perf_results = results['test_results']['performance']['results']
            print(f"\n⚡ PERFORMANCE HIGHLIGHTS:")
            for operation, result in perf_results.items():
                if result.target_met:
                    print(f"   ✅ {operation}: {result.actual_value:.2f}ms (target: {result.target_value}ms)")
                else:
                    print(f"   ❌ {operation}: {result.actual_value:.2f}ms (target: {result.target_value}ms)")
        
        print(f"\n⏰ Test completed at: {results['end_time']}")
        print("="*80)
    
    def save_results(self, results: Dict, filename: Optional[str] = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integration_test_results_{self.environment}_{timestamp}.json"
        
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_results = json.loads(json.dumps(results, default=str))
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"📁 Test results saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="MemMimic v2.0 Integration Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_integration_tests.py
  
  # Run only performance tests
  python run_integration_tests.py --performance-only
  
  # Run with custom environment
  python run_integration_tests.py --environment production
  
  # Save results to specific file
  python run_integration_tests.py --output results.json
        """
    )
    
    parser.add_argument(
        '--environment', 
        default='test',
        help='Test environment name (default: test)'
    )
    
    parser.add_argument(
        '--performance-only',
        action='store_true',
        help='Run only performance validation tests'
    )
    
    parser.add_argument(
        '--integration-only',
        action='store_true', 
        help='Run only integration tests'
    )
    
    parser.add_argument(
        '--load-only',
        action='store_true',
        help='Run only load tests'
    )
    
    parser.add_argument(
        '--benchmarking-only', 
        action='store_true',
        help='Run only benchmarking tests'
    )
    
    parser.add_argument(
        '--monitoring-demo',
        action='store_true',
        help='Include monitoring demonstration'
    )
    
    parser.add_argument(
        '--output',
        help='Output file for test results (JSON format)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to custom configuration file (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load config file: {e}")
            sys.exit(1)
    
    # Override test selection based on arguments
    if config is None:
        config = {}
    
    if 'testing' not in config:
        config['testing'] = {}
    
    # Set test selection based on arguments
    if args.performance_only:
        config['testing'].update({
            'run_performance_tests': True,
            'run_integration_tests': False,
            'run_load_tests': False,
            'run_production_readiness': False,
            'run_benchmarking': False
        })
    elif args.integration_only:
        config['testing'].update({
            'run_performance_tests': False,
            'run_integration_tests': True,
            'run_load_tests': False,
            'run_production_readiness': False,
            'run_benchmarking': False
        })
    elif args.load_only:
        config['testing'].update({
            'run_performance_tests': False,
            'run_integration_tests': False,
            'run_load_tests': True,
            'run_production_readiness': False,
            'run_benchmarking': False
        })
    elif args.benchmarking_only:
        config['testing'].update({
            'run_performance_tests': False,
            'run_integration_tests': False,
            'run_load_tests': False,
            'run_production_readiness': False,
            'run_benchmarking': True
        })
    
    if args.monitoring_demo:
        config['testing']['run_monitoring_demo'] = True
    
    # Create orchestrator and run tests
    orchestrator = IntegrationTestOrchestrator(config, args.environment)
    
    try:
        await orchestrator.setup()
        results = await orchestrator.run_all_tests()
        
        # Print final report
        orchestrator.print_final_report(results)
        
        # Save results
        orchestrator.save_results(results, args.output)
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_success'] else 1
        print(f"\n🏁 Integration testing completed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        await orchestrator.teardown()


if __name__ == "__main__":
    asyncio.run(main())