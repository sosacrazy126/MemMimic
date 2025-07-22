#!/usr/bin/env python3
"""
MemMimic Comprehensive Test Suite Runner

Master test runner for Quality Agent Gamma's comprehensive test coverage:
1. Phase 1 Security Regression Tests
2. Phase 2 Performance Optimization Tests  
3. Modular Architecture Integration Tests
4. Coverage Analysis and Reporting
"""

import asyncio
import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class ComprehensiveTestRunner:
    """Master test runner for all MemMimic test suites."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_dir = Path(__file__).parent
        self.results = {}
        self.start_time = time.time()
        
    def print_header(self):
        """Print the test suite header."""
        print("🛡️" + "=" * 78 + "🛡️")
        print("🚀 MEMMIMIC COMPREHENSIVE TEST SUITE - QUALITY AGENT GAMMA")
        print("🛡️" + "=" * 78 + "🛡️")
        print()
        print("📋 Test Suite Components:")
        print("   🔒 Phase 1: Security Regression Tests")
        print("   ⚡ Phase 2: Performance Optimization Tests")
        print("   🏗️ Modular Architecture Integration Tests")
        print("   📊 Test Coverage Analysis")
        print()
        print("🎯 Target: >95% Overall Test Coverage")
        print()
        
    async def run_security_regression_tests(self) -> Dict[str, Any]:
        """Run Phase 1 security regression tests."""
        print("🔒 Running Phase 1 Security Regression Tests...")
        print("-" * 50)
        
        security_test_file = self.tests_dir / 'security' / 'test_phase1_security_regression.py'
        
        if not security_test_file.exists():
            return {
                'status': 'failed',
                'error': 'Security test file not found',
                'exit_code': 1,
                'duration': 0
            }
        
        start_time = time.time()
        
        try:
            # Import and run the security tests
            sys.path.insert(0, str(security_test_file.parent))
            
            # Dynamic import to avoid circular dependencies
            spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location(
                'security_tests', security_test_file
            )
            security_module = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
            spec.loader.exec_module(security_module)
            
            # Run the security tests
            if hasattr(security_module, 'run_phase1_security_tests'):
                exit_code = await security_module.run_phase1_security_tests()
            else:
                exit_code = 1
                
            duration = time.time() - start_time
            
            return {
                'status': 'passed' if exit_code == 0 else 'failed',
                'exit_code': exit_code,
                'duration': duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Security tests failed with exception: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'exit_code': 2,
                'duration': duration
            }
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run Phase 2 performance optimization tests."""
        print("\n⚡ Running Phase 2 Performance Optimization Tests...")
        print("-" * 50)
        
        performance_test_file = self.tests_dir / 'performance' / 'test_phase2_performance_optimization.py'
        
        if not performance_test_file.exists():
            return {
                'status': 'failed',
                'error': 'Performance test file not found',
                'exit_code': 1,
                'duration': 0
            }
        
        start_time = time.time()
        
        try:
            # Import and run the performance tests
            sys.path.insert(0, str(performance_test_file.parent))
            
            spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location(
                'performance_tests', performance_test_file
            )
            performance_module = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
            spec.loader.exec_module(performance_module)
            
            # Run the performance tests
            if hasattr(performance_module, 'run_phase2_performance_tests'):
                exit_code = await performance_module.run_phase2_performance_tests()
            else:
                exit_code = 1
                
            duration = time.time() - start_time
            
            return {
                'status': 'passed' if exit_code == 0 else 'failed',
                'exit_code': exit_code,
                'duration': duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Performance tests failed with exception: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'exit_code': 2,
                'duration': duration
            }
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run modular architecture integration tests."""
        print("\n🏗️ Running Modular Architecture Integration Tests...")
        print("-" * 50)
        
        integration_test_file = self.tests_dir / 'integration' / 'test_modular_architecture_integration.py'
        
        if not integration_test_file.exists():
            return {
                'status': 'failed',
                'error': 'Integration test file not found',
                'exit_code': 1,
                'duration': 0
            }
        
        start_time = time.time()
        
        try:
            # Import and run the integration tests
            sys.path.insert(0, str(integration_test_file.parent))
            
            spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location(
                'integration_tests', integration_test_file
            )
            integration_module = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
            spec.loader.exec_module(integration_module)
            
            # Run the integration tests
            if hasattr(integration_module, 'run_modular_architecture_integration_tests'):
                exit_code = await integration_module.run_modular_architecture_integration_tests()
            else:
                exit_code = 1
                
            duration = time.time() - start_time
            
            return {
                'status': 'passed' if exit_code == 0 else 'failed',
                'exit_code': exit_code,
                'duration': duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Integration tests failed with exception: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'exit_code': 2,
                'duration': duration
            }
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run comprehensive coverage analysis."""
        print("\n📊 Running Comprehensive Coverage Analysis...")
        print("-" * 50)
        
        coverage_script = self.tests_dir / 'test_coverage_analysis.py'
        
        if not coverage_script.exists():
            return {
                'status': 'failed',
                'error': 'Coverage analysis script not found',
                'exit_code': 1,
                'duration': 0
            }
        
        start_time = time.time()
        
        try:
            # Run coverage analysis as subprocess to avoid import issues
            result = subprocess.run([
                sys.executable, str(coverage_script)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            duration = time.time() - start_time
            
            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'exit_code': result.returncode,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Coverage analysis failed with exception: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'exit_code': 2,
                'duration': duration
            }
    
    def run_existing_tests(self) -> Dict[str, Any]:
        """Run existing test suite for baseline."""
        print("\n🧪 Running Existing Test Suite for Baseline...")
        print("-" * 50)
        
        existing_test_runner = self.tests_dir / 'run_all_tests.py'
        
        if not existing_test_runner.exists():
            return {
                'status': 'skipped',
                'reason': 'Existing test runner not found',
                'exit_code': 0,
                'duration': 0
            }
        
        start_time = time.time()
        
        try:
            # Run existing tests
            result = subprocess.run([
                sys.executable, str(existing_test_runner)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            duration = time.time() - start_time
            
            # Print abbreviated output
            if result.stdout:
                lines = result.stdout.split('\n')
                # Show first 10 and last 10 lines
                if len(lines) > 20:
                    print('\n'.join(lines[:10]))
                    print("... [output truncated] ...")
                    print('\n'.join(lines[-10:]))
                else:
                    print(result.stdout)
            
            if result.stderr:
                print("STDERR:", result.stderr)
            
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'exit_code': result.returncode,
                'duration': duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Existing tests failed with exception: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'exit_code': 2,
                'duration': duration
            }
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final comprehensive test report."""
        total_duration = time.time() - self.start_time
        
        # Calculate overall statistics
        test_suites = ['security_tests', 'performance_tests', 'integration_tests', 'coverage_analysis', 'existing_tests']
        total_suites = len(test_suites)
        passed_suites = sum(1 for suite in test_suites if self.results.get(suite, {}).get('status') == 'passed')
        failed_suites = sum(1 for suite in test_suites if self.results.get(suite, {}).get('status') == 'failed')
        skipped_suites = sum(1 for suite in test_suites if self.results.get(suite, {}).get('status') == 'skipped')
        
        success_rate = (passed_suites / total_suites * 100) if total_suites > 0 else 0
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration': total_duration,
            'test_suite_results': self.results,
            'summary': {
                'total_suites': total_suites,
                'passed_suites': passed_suites,
                'failed_suites': failed_suites,
                'skipped_suites': skipped_suites,
                'success_rate': success_rate
            },
            'quality_assessment': self._assess_quality(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _assess_quality(self) -> Dict[str, Any]:
        """Assess overall quality based on test results."""
        security_passed = self.results.get('security_tests', {}).get('status') == 'passed'
        performance_passed = self.results.get('performance_tests', {}).get('status') == 'passed'
        integration_passed = self.results.get('integration_tests', {}).get('status') == 'passed'
        coverage_passed = self.results.get('coverage_analysis', {}).get('status') == 'passed'
        
        # Quality scoring
        quality_score = 0
        max_score = 100
        
        if security_passed:
            quality_score += 30  # Security is critical
        if performance_passed:
            quality_score += 25  # Performance is important
        if integration_passed:
            quality_score += 25  # Integration is important
        if coverage_passed:
            quality_score += 20  # Coverage is important
        
        # Quality levels
        if quality_score >= 95:
            quality_level = "EXCELLENT"
            status = "✅"
        elif quality_score >= 80:
            quality_level = "GOOD"
            status = "⚠️"
        elif quality_score >= 60:
            quality_level = "FAIR"
            status = "⚠️"
        else:
            quality_level = "POOR"
            status = "❌"
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'status': status,
            'components': {
                'security': security_passed,
                'performance': performance_passed,
                'integration': integration_passed,
                'coverage': coverage_passed
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check failed test suites
        if self.results.get('security_tests', {}).get('status') == 'failed':
            recommendations.append("🚨 CRITICAL: Fix security regression test failures before deployment")
        
        if self.results.get('performance_tests', {}).get('status') == 'failed':
            recommendations.append("⚡ HIGH: Address performance optimization test failures")
        
        if self.results.get('integration_tests', {}).get('status') == 'failed':
            recommendations.append("🏗️ HIGH: Fix modular architecture integration issues")
        
        if self.results.get('coverage_analysis', {}).get('status') == 'failed':
            recommendations.append("📊 MEDIUM: Improve test coverage to meet >95% target")
        
        if self.results.get('existing_tests', {}).get('status') == 'failed':
            recommendations.append("🧪 MEDIUM: Fix regressions in existing test suite")
        
        # Performance recommendations
        total_duration = time.time() - self.start_time
        if total_duration > 300:  # 5 minutes
            recommendations.append("⏱️ Consider optimizing test execution time (currently {:.1f}s)".format(total_duration))
        
        # Success recommendations
        quality_assessment = self._assess_quality()
        if quality_assessment['quality_score'] >= 95:
            recommendations.append("🎉 Excellent test coverage! Ready for production deployment")
        elif quality_assessment['quality_score'] >= 80:
            recommendations.append("👍 Good test coverage. Address remaining issues for production readiness")
        
        return recommendations
    
    def print_final_report(self, report: Dict[str, Any]):
        """Print the final comprehensive report."""
        print("\n" + "🛡️" + "=" * 78 + "🛡️")
        print("📊 COMPREHENSIVE TEST SUITE FINAL REPORT")
        print("🛡️" + "=" * 78 + "🛡️")
        
        summary = report['summary']
        quality = report['quality_assessment']
        
        print(f"\n⏱️ Total Duration: {report['total_duration']:.1f} seconds")
        print(f"📅 Completed: {report['timestamp']}")
        
        print(f"\n📈 Test Suite Summary:")
        print(f"   Total Suites: {summary['total_suites']}")
        print(f"   ✅ Passed: {summary['passed_suites']}")
        print(f"   ❌ Failed: {summary['failed_suites']}")
        print(f"   ⏭️ Skipped: {summary['skipped_suites']}")
        print(f"   📊 Success Rate: {summary['success_rate']:.1f}%")
        
        print(f"\n🎯 Quality Assessment:")
        print(f"   {quality['status']} Overall Quality: {quality['quality_level']} ({quality['quality_score']}/100)")
        
        print(f"\n📋 Component Status:")
        for component, status in quality['components'].items():
            icon = "✅" if status else "❌"
            print(f"   {icon} {component.title()}: {'PASSED' if status else 'FAILED'}")
        
        print(f"\n📝 Individual Test Results:")
        for suite_name, result in report['test_suite_results'].items():
            status_icon = "✅" if result['status'] == 'passed' else "⏭️" if result['status'] == 'skipped' else "❌"
            suite_title = suite_name.replace('_', ' ').title()
            duration = result.get('duration', 0)
            
            print(f"   {status_icon} {suite_title}: {result['status'].upper()} ({duration:.1f}s)")
            
            if 'error' in result:
                print(f"      Error: {result['error']}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        # Final verdict
        print(f"\n🎯 FINAL VERDICT:")
        if quality['quality_score'] >= 95:
            print("   🎉 COMPREHENSIVE TEST SUITE PASSED!")
            print("   ✅ Quality target achieved - Ready for deployment")
        elif quality['quality_score'] >= 80:
            print("   ⚠️ MOSTLY SUCCESSFUL")
            print("   🔧 Minor issues need attention before deployment")
        else:
            print("   ❌ SIGNIFICANT ISSUES DETECTED")
            print("   🚨 Major fixes required before deployment")
        
        print("\n🛡️" + "=" * 78 + "🛡️")
    
    async def run_all_tests(self) -> int:
        """Run all comprehensive tests and return exit code."""
        self.print_header()
        
        # Run all test suites
        self.results['security_tests'] = await self.run_security_regression_tests()
        self.results['performance_tests'] = await self.run_performance_tests()
        self.results['integration_tests'] = await self.run_integration_tests()
        self.results['coverage_analysis'] = self.run_coverage_analysis()
        self.results['existing_tests'] = self.run_existing_tests()
        
        # Generate and print final report
        report = self.generate_final_report()
        self.print_final_report(report)
        
        # Save report to file
        report_file = self.project_root / 'comprehensive_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Return appropriate exit code
        quality_score = report['quality_assessment']['quality_score']
        
        if quality_score >= 95:
            return 0  # Success
        elif quality_score >= 80:
            return 1  # Warning
        else:
            return 2  # Failure


async def main():
    """Main entry point."""
    runner = ComprehensiveTestRunner()
    exit_code = await runner.run_all_tests()
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))