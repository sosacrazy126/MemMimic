#!/usr/bin/env python3
"""
MemMimic Comprehensive Test Runner

Runs all test suites and provides comprehensive reporting.
This script executes all our improved test coverage including:
- Updated API tests
- Comprehensive AMMS storage tests  
- Performance configuration tests
- Error handling and recovery tests
- Critical integration tests
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def run_test_file(test_file, description):
    """Run a single test file and capture results."""
    print(f"\n{'=' * 60}")
    print(f"🧪 Running: {description}")
    print(f"📁 File: {test_file}")
    print('=' * 60)
    
    start_time = time.time()
    
    try:
        # Run the test file
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        
        print(f"\n⏱️ Duration: {duration:.2f}s")
        print(f"📊 Result: {'✅ PASSED' if success else '❌ FAILED'}")
        
        return {
            'name': description,
            'file': test_file,
            'success': success,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ Test execution failed: {e}")
        
        return {
            'name': description,
            'file': test_file,
            'success': False,
            'duration': duration,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }


def check_dependencies():
    """Check that required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_modules = [
        'sqlite3',
        'asyncio', 
        'tempfile',
        'yaml',
        'json',
        'pathlib'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"   ❌ {module} (missing)")
    
    if missing:
        print(f"\n❌ Missing required modules: {', '.join(missing)}")
        return False
    
    # Check for MemMimic modules
    try:
        import memmimic
        print("   ✅ memmimic")
    except ImportError as e:
        print(f"   ❌ memmimic (missing): {e}")
        return False
    
    print("✅ All dependencies available")
    return True


def main():
    """Run all tests and generate comprehensive report."""
    print("🚀 MemMimic Comprehensive Test Suite")
    print("=" * 60)
    print("Running all improved test coverage...")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n💥 Dependency check failed! Please install missing modules.")
        return 1
    
    # Define test files and descriptions
    test_files = [
        ("test_unified_api.py", "Unified API (Updated)"),
        ("test_unified_api_improved.py", "Unified API with Improvements"),
        ("test_amms_storage_comprehensive.py", "AMMS Storage Comprehensive"),
        ("test_performance_config.py", "Performance Configuration"),
        ("test_error_handling.py", "Error Handling and Recovery"),
        ("test_amms_critical.py", "AMMS Critical Integration"),
    ]
    
    # Verify all test files exist
    test_dir = Path(__file__).parent
    missing_files = []
    
    for test_file, _ in test_files:
        test_path = test_dir / test_file
        if not test_path.exists():
            missing_files.append(test_file)
    
    if missing_files:
        print(f"\n❌ Missing test files: {', '.join(missing_files)}")
        return 1
    
    # Run all tests
    results = []
    start_time = time.time()
    
    for test_file, description in test_files:
        test_path = test_dir / test_file
        result = run_test_file(str(test_path), description)
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Generate comprehensive report
    print("\n\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    passed = 0
    failed = 0
    total_tests = len(results)
    
    # Individual results
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {result['name']:.<50} {result['duration']:>6.2f}s")
        
        if result['success']:
            passed += 1
        else:
            failed += 1
    
    print("-" * 80)
    
    # Summary statistics
    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}% ({passed}/{total_tests})")
    print(f"⏱️ Total Time: {total_time:.2f}s")
    print(f"🏃 Average Time: {total_time/total_tests:.2f}s per test")
    
    # Performance analysis
    fastest = min(results, key=lambda x: x['duration'])
    slowest = max(results, key=lambda x: x['duration'])
    
    print(f"🏃‍♂️ Fastest: {fastest['name']} ({fastest['duration']:.2f}s)")
    print(f"🐌 Slowest: {slowest['name']} ({slowest['duration']:.2f}s)")
    
    # Failed tests details
    if failed > 0:
        print(f"\n❌ FAILED TESTS ({failed}):")
        for result in results:
            if not result['success']:
                print(f"   • {result['name']} (exit code: {result['returncode']})")
                if result['stderr']:
                    stderr_lines = result['stderr'].split('\n')[:3]  # First 3 lines
                    for line in stderr_lines:
                        if line.strip():
                            print(f"     {line}")
    
    # Coverage analysis
    print(f"\n📋 TEST COVERAGE ANALYSIS:")
    coverage_areas = [
        ("API Functionality", ["Unified API (Updated)", "Unified API with Improvements"]),
        ("Storage Systems", ["AMMS Storage Comprehensive"]),
        ("Configuration", ["Performance Configuration"]), 
        ("Error Handling", ["Error Handling and Recovery"]),
        ("Integration", ["AMMS Critical Integration"])
    ]
    
    for area, test_names in coverage_areas:
        area_results = [r for r in results if r['name'] in test_names]
        area_passed = sum(1 for r in area_results if r['success'])
        area_total = len(area_results)
        area_rate = (area_passed / area_total * 100) if area_total > 0 else 0
        
        status = "✅" if area_rate == 100 else "⚠️" if area_rate >= 80 else "❌"
        print(f"   {status} {area:.<25} {area_rate:>5.1f}% ({area_passed}/{area_total})")
    
    # Final verdict
    print("\n" + "=" * 80)
    if passed == total_tests:
        print("🎉 ALL TESTS PASSED! MemMimic improvements are working perfectly!")
        print("✅ Connection pooling")
        print("✅ Performance monitoring") 
        print("✅ JSON safety (no eval vulnerabilities)")
        print("✅ Enhanced error handling")
        print("✅ Configuration system")
        print("✅ Type safety")
        print("✅ Async/sync bridge improvements")
        return 0
    elif success_rate >= 80:
        print(f"⚠️  MOSTLY SUCCESSFUL ({success_rate:.1f}% pass rate)")
        print("Most improvements are working, but some issues need attention.")
        return 1
    else:
        print(f"❌ SIGNIFICANT FAILURES ({success_rate:.1f}% pass rate)")
        print("Major issues detected. Review failed tests before deployment.")
        return 2


if __name__ == "__main__":
    sys.exit(main())