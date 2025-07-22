#!/usr/bin/env python3
"""
Comprehensive Test Coverage Analysis

This script analyzes test coverage across all MemMimic components to identify
gaps and ensure >95% overall coverage target is met.
"""

import sys
import os
import subprocess
import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import re

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class CoverageAnalyzer:
    """Analyzes test coverage for MemMimic codebase."""
    
    def __init__(self, src_dir: Path, tests_dir: Path):
        self.src_dir = src_dir
        self.tests_dir = tests_dir
        self.coverage_data = {}
        self.source_modules = {}
        self.test_modules = {}
        
    def discover_source_modules(self) -> Dict[str, Dict[str, Any]]:
        """Discover all source modules and their functions/classes."""
        print("🔍 Discovering source modules...")
        
        source_modules = {}
        
        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            relative_path = py_file.relative_to(self.src_dir)
            module_name = str(relative_path).replace('/', '.').replace('.py', '')
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find classes and functions
                tree = ast.parse(content)
                
                classes = []
                functions = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Get class methods
                        methods = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                methods.append(item.name)
                        classes.append({
                            'name': node.name,
                            'methods': methods,
                            'line': node.lineno
                        })
                    elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
                        # Only top-level functions
                        if isinstance(getattr(node, 'parent', None), ast.Module) or not hasattr(node, 'parent'):
                            functions.append({
                                'name': node.name,
                                'line': node.lineno
                            })
                    elif isinstance(node, ast.AsyncFunctionDef):
                        if isinstance(getattr(node, 'parent', None), ast.Module) or not hasattr(node, 'parent'):
                            functions.append({
                                'name': node.name,
                                'line': node.lineno,
                                'async': True
                            })
                
                source_modules[module_name] = {
                    'file_path': py_file,
                    'classes': classes,
                    'functions': functions,
                    'line_count': len(content.splitlines())
                }
                
            except Exception as e:
                print(f"   ⚠️ Error parsing {py_file}: {e}")
                source_modules[module_name] = {
                    'file_path': py_file,
                    'classes': [],
                    'functions': [],
                    'line_count': 0,
                    'error': str(e)
                }
        
        self.source_modules = source_modules
        print(f"   ✅ Discovered {len(source_modules)} source modules")
        return source_modules
    
    def discover_test_modules(self) -> Dict[str, Dict[str, Any]]:
        """Discover all test modules and what they test."""
        print("🧪 Discovering test modules...")
        
        test_modules = {}
        
        for py_file in self.tests_dir.rglob("*.py"):
            if not py_file.name.startswith("test_"):
                continue
                
            relative_path = py_file.relative_to(self.tests_dir)
            module_name = str(relative_path).replace('/', '.').replace('.py', '')
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find imports to determine what modules are being tested
                imported_modules = self._extract_imports(content)
                
                # Find test classes and methods
                tree = ast.parse(content)
                test_classes = []
                test_functions = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and 'test' in node.name.lower():
                        methods = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                                methods.append(item.name)
                        test_classes.append({
                            'name': node.name,
                            'methods': methods,
                            'line': node.lineno
                        })
                    elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        test_functions.append({
                            'name': node.name,
                            'line': node.lineno
                        })
                
                test_modules[module_name] = {
                    'file_path': py_file,
                    'imports': imported_modules,
                    'test_classes': test_classes,
                    'test_functions': test_functions,
                    'line_count': len(content.splitlines())
                }
                
            except Exception as e:
                print(f"   ⚠️ Error parsing test {py_file}: {e}")
                test_modules[module_name] = {
                    'file_path': py_file,
                    'imports': [],
                    'test_classes': [],
                    'test_functions': [],
                    'line_count': 0,
                    'error': str(e)
                }
        
        self.test_modules = test_modules
        print(f"   ✅ Discovered {len(test_modules)} test modules")
        return test_modules
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract imported modules from test content."""
        imports = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                        
        except Exception:
            # Fallback to regex if AST fails
            import_patterns = [
                r'^from\s+(memmimic\.[^\s]+)',
                r'^import\s+(memmimic\.[^\s,]+)',
            ]
            
            for line in content.splitlines():
                for pattern in import_patterns:
                    match = re.search(pattern, line.strip())
                    if match:
                        imports.append(match.group(1))
        
        return imports
    
    def analyze_coverage_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Analyze which source modules have test coverage."""
        print("📊 Analyzing coverage mapping...")
        
        coverage_mapping = {}
        
        for src_module, src_info in self.source_modules.items():
            coverage_mapping[src_module] = {
                'source_info': src_info,
                'test_files': [],
                'test_coverage': {
                    'classes': {},
                    'functions': {},
                    'overall': 0
                },
                'gaps': []
            }
            
            # Find test files that import this source module
            for test_module, test_info in self.test_modules.items():
                if any(src_module in imp or imp in src_module for imp in test_info['imports']):
                    coverage_mapping[src_module]['test_files'].append(test_module)
            
            # Analyze class coverage
            for src_class in src_info['classes']:
                class_name = src_class['name']
                coverage_mapping[src_module]['test_coverage']['classes'][class_name] = {
                    'tested': False,
                    'methods_tested': 0,
                    'total_methods': len(src_class['methods']),
                    'method_coverage': {}
                }
                
                # Check if any test mentions this class
                for test_module in coverage_mapping[src_module]['test_files']:
                    test_info = self.test_modules[test_module]
                    test_content = test_info['file_path'].read_text()
                    
                    if class_name in test_content:
                        coverage_mapping[src_module]['test_coverage']['classes'][class_name]['tested'] = True
                    
                    # Check method coverage
                    for method_name in src_class['methods']:
                        if method_name in test_content:
                            coverage_mapping[src_module]['test_coverage']['classes'][class_name]['method_coverage'][method_name] = True
                        else:
                            coverage_mapping[src_module]['test_coverage']['classes'][class_name]['method_coverage'][method_name] = False
                
                # Count tested methods
                tested_methods = sum(coverage_mapping[src_module]['test_coverage']['classes'][class_name]['method_coverage'].values())
                coverage_mapping[src_module]['test_coverage']['classes'][class_name]['methods_tested'] = tested_methods
            
            # Analyze function coverage
            for src_function in src_info['functions']:
                func_name = src_function['name']
                coverage_mapping[src_module]['test_coverage']['functions'][func_name] = False
                
                # Check if any test mentions this function
                for test_module in coverage_mapping[src_module]['test_files']:
                    test_info = self.test_modules[test_module]
                    test_content = test_info['file_path'].read_text()
                    
                    if func_name in test_content:
                        coverage_mapping[src_module]['test_coverage']['functions'][func_name] = True
                        break
            
            # Calculate overall coverage
            total_items = len(src_info['classes']) + len(src_info['functions'])
            if total_items > 0:
                tested_classes = sum(1 for c in coverage_mapping[src_module]['test_coverage']['classes'].values() if c['tested'])
                tested_functions = sum(coverage_mapping[src_module]['test_coverage']['functions'].values())
                
                overall_coverage = (tested_classes + tested_functions) / total_items
                coverage_mapping[src_module]['test_coverage']['overall'] = overall_coverage
            
            # Identify gaps
            gaps = []
            for class_name, class_info in coverage_mapping[src_module]['test_coverage']['classes'].items():
                if not class_info['tested']:
                    gaps.append(f"Class {class_name} not tested")
                elif class_info['methods_tested'] < class_info['total_methods']:
                    untested_methods = [
                        method for method, tested in class_info['method_coverage'].items() 
                        if not tested
                    ]
                    gaps.append(f"Class {class_name} methods not tested: {untested_methods}")
            
            for func_name, tested in coverage_mapping[src_module]['test_coverage']['functions'].items():
                if not tested:
                    gaps.append(f"Function {func_name} not tested")
            
            coverage_mapping[src_module]['gaps'] = gaps
        
        self.coverage_data = coverage_mapping
        print(f"   ✅ Coverage mapping complete for {len(coverage_mapping)} modules")
        return coverage_mapping
    
    def run_pytest_coverage(self) -> Optional[Dict[str, Any]]:
        """Run pytest with coverage if available."""
        print("🔬 Running pytest coverage analysis...")
        
        try:
            # Try to run pytest with coverage
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                str(self.tests_dir),
                '--cov=' + str(self.src_dir / 'memmimic'),
                '--cov-report=json:coverage.json',
                '--cov-report=term-missing',
                '--quiet'
            ], capture_output=True, text=True, cwd=self.tests_dir.parent)
            
            if result.returncode == 0:
                print("   ✅ Pytest coverage completed")
                
                # Try to read coverage.json if it was created
                coverage_file = self.tests_dir.parent / 'coverage.json'
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                    print(f"   ✅ Coverage data loaded from {coverage_file}")
                    return coverage_data
            else:
                print(f"   ⚠️ Pytest coverage failed: {result.stderr}")
                
        except FileNotFoundError:
            print("   ⚠️ Pytest not available")
        except Exception as e:
            print(f"   ⚠️ Pytest coverage error: {e}")
        
        return None
    
    def generate_coverage_report(self) -> Dict[str, Any]:
        """Generate comprehensive coverage report."""
        print("📈 Generating coverage report...")
        
        # Get pytest coverage if available
        pytest_coverage = self.run_pytest_coverage()
        
        # Calculate overall statistics
        total_modules = len(self.source_modules)
        modules_with_tests = sum(1 for m in self.coverage_data.values() if m['test_files'])
        modules_without_tests = total_modules - modules_with_tests
        
        total_classes = sum(len(info['classes']) for info in self.source_modules.values())
        total_functions = sum(len(info['functions']) for info in self.source_modules.values())
        
        tested_classes = sum(
            sum(1 for c in module_data['test_coverage']['classes'].values() if c['tested'])
            for module_data in self.coverage_data.values()
        )
        tested_functions = sum(
            sum(module_data['test_coverage']['functions'].values())
            for module_data in self.coverage_data.values()
        )
        
        overall_coverage = ((tested_classes + tested_functions) / (total_classes + total_functions)) * 100 if (total_classes + total_functions) > 0 else 0
        
        # Identify critical gaps
        critical_gaps = []
        major_gaps = []
        
        for module_name, module_data in self.coverage_data.items():
            if not module_data['test_files']:
                if 'core' in module_name or 'api' in module_name or 'storage' in module_name:
                    critical_gaps.append(f"Critical module has no tests: {module_name}")
                else:
                    major_gaps.append(f"Module has no tests: {module_name}")
            elif module_data['test_coverage']['overall'] < 0.5:
                if 'core' in module_name or 'api' in module_name or 'storage' in module_name:
                    critical_gaps.append(f"Critical module low coverage ({module_data['test_coverage']['overall']:.1%}): {module_name}")
                else:
                    major_gaps.append(f"Module low coverage ({module_data['test_coverage']['overall']:.1%}): {module_name}")
        
        # Phase-specific analysis
        phase1_modules = [m for m in self.source_modules.keys() if 'security' in m]
        phase2_modules = [m for m in self.source_modules.keys() if any(x in m for x in ['caching', 'search', 'performance'])]
        
        phase1_coverage = self._calculate_phase_coverage(phase1_modules)
        phase2_coverage = self._calculate_phase_coverage(phase2_modules)
        
        report = {
            'overall_statistics': {
                'total_modules': total_modules,
                'modules_with_tests': modules_with_tests,
                'modules_without_tests': modules_without_tests,
                'total_classes': total_classes,
                'total_functions': total_functions,
                'tested_classes': tested_classes,
                'tested_functions': tested_functions,
                'overall_coverage_percentage': overall_coverage
            },
            'phase_analysis': {
                'phase1_security': {
                    'modules': len(phase1_modules),
                    'coverage': phase1_coverage
                },
                'phase2_performance': {
                    'modules': len(phase2_modules),
                    'coverage': phase2_coverage
                }
            },
            'gaps_analysis': {
                'critical_gaps': critical_gaps,
                'major_gaps': major_gaps,
                'total_gaps': len(critical_gaps) + len(major_gaps)
            },
            'pytest_coverage': pytest_coverage,
            'detailed_coverage': self.coverage_data,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_phase_coverage(self, module_names: List[str]) -> float:
        """Calculate coverage for specific phase modules."""
        if not module_names:
            return 0.0
            
        total_items = 0
        tested_items = 0
        
        for module_name in module_names:
            if module_name in self.coverage_data:
                module_data = self.coverage_data[module_name]
                source_info = module_data['source_info']
                
                total_items += len(source_info['classes']) + len(source_info['functions'])
                
                tested_classes = sum(1 for c in module_data['test_coverage']['classes'].values() if c['tested'])
                tested_functions = sum(module_data['test_coverage']['functions'].values())
                
                tested_items += tested_classes + tested_functions
        
        return (tested_items / total_items * 100) if total_items > 0 else 0.0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving coverage."""
        recommendations = []
        
        # Check if target coverage is met
        total_classes = sum(len(info['classes']) for info in self.source_modules.values())
        total_functions = sum(len(info['functions']) for info in self.source_modules.values())
        
        tested_classes = sum(
            sum(1 for c in module_data['test_coverage']['classes'].values() if c['tested'])
            for module_data in self.coverage_data.values()
        )
        tested_functions = sum(
            sum(module_data['test_coverage']['functions'].values())
            for module_data in self.coverage_data.values()
        )
        
        overall_coverage = ((tested_classes + tested_functions) / (total_classes + total_functions)) * 100 if (total_classes + total_functions) > 0 else 0
        
        if overall_coverage < 95:
            recommendations.append(f"Overall coverage {overall_coverage:.1f}% is below 95% target. Need {95 - overall_coverage:.1f}% improvement.")
        
        # Module-specific recommendations
        untested_modules = [
            name for name, data in self.coverage_data.items()
            if not data['test_files']
        ]
        
        if untested_modules:
            recommendations.append(f"Create tests for {len(untested_modules)} untested modules: {untested_modules[:5]}")
        
        # Critical module recommendations
        critical_modules = [name for name in self.source_modules.keys() if any(x in name for x in ['api', 'storage', 'security', 'core'])]
        critical_needing_tests = [
            name for name in critical_modules
            if name in self.coverage_data and not self.coverage_data[name]['test_files']
        ]
        
        if critical_needing_tests:
            recommendations.append(f"PRIORITY: Create tests for critical modules: {critical_needing_tests}")
        
        # Phase-specific recommendations
        phase1_modules = [m for m in self.source_modules.keys() if 'security' in m]
        phase1_coverage = self._calculate_phase_coverage(phase1_modules)
        
        if phase1_coverage < 90:
            recommendations.append(f"Phase 1 (Security) coverage {phase1_coverage:.1f}% needs improvement")
        
        phase2_modules = [m for m in self.source_modules.keys() if any(x in m for x in ['caching', 'search', 'performance'])]
        phase2_coverage = self._calculate_phase_coverage(phase2_modules)
        
        if phase2_coverage < 90:
            recommendations.append(f"Phase 2 (Performance) coverage {phase2_coverage:.1f}% needs improvement")
        
        return recommendations
    
    def print_coverage_summary(self, report: Dict[str, Any]):
        """Print a comprehensive coverage summary."""
        print("\n" + "=" * 80)
        print("📊 MEMMIMIC TEST COVERAGE ANALYSIS REPORT")
        print("=" * 80)
        
        stats = report['overall_statistics']
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Modules: {stats['total_modules']}")
        print(f"   Modules with Tests: {stats['modules_with_tests']}")
        print(f"   Modules without Tests: {stats['modules_without_tests']}")
        print(f"   Total Classes: {stats['total_classes']}")
        print(f"   Total Functions: {stats['total_functions']}")
        print(f"   Tested Classes: {stats['tested_classes']}")
        print(f"   Tested Functions: {stats['tested_functions']}")
        
        overall_pct = stats['overall_coverage_percentage']
        status = "✅" if overall_pct >= 95 else "⚠️" if overall_pct >= 80 else "❌"
        print(f"\n{status} Overall Coverage: {overall_pct:.1f}%")
        
        # Phase analysis
        print(f"\n🔒 Phase Analysis:")
        phase_analysis = report['phase_analysis']
        
        phase1_pct = phase_analysis['phase1_security']['coverage']
        phase1_status = "✅" if phase1_pct >= 90 else "⚠️" if phase1_pct >= 70 else "❌"
        print(f"   {phase1_status} Phase 1 (Security): {phase1_pct:.1f}% ({phase_analysis['phase1_security']['modules']} modules)")
        
        phase2_pct = phase_analysis['phase2_performance']['coverage']
        phase2_status = "✅" if phase2_pct >= 90 else "⚠️" if phase2_pct >= 70 else "❌"
        print(f"   {phase2_status} Phase 2 (Performance): {phase2_pct:.1f}% ({phase_analysis['phase2_performance']['modules']} modules)")
        
        # Gaps analysis
        gaps = report['gaps_analysis']
        print(f"\n🔍 Coverage Gaps:")
        print(f"   Critical Gaps: {len(gaps['critical_gaps'])}")
        print(f"   Major Gaps: {len(gaps['major_gaps'])}")
        
        if gaps['critical_gaps']:
            print("\n   🚨 Critical Gaps:")
            for gap in gaps['critical_gaps'][:5]:
                print(f"      • {gap}")
        
        if gaps['major_gaps']:
            print("\n   ⚠️ Major Gaps:")
            for gap in gaps['major_gaps'][:5]:
                print(f"      • {gap}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
        
        # Target assessment
        print(f"\n🎯 Coverage Target Assessment:")
        if overall_pct >= 95:
            print("   ✅ COVERAGE TARGET ACHIEVED (>95%)")
        elif overall_pct >= 90:
            print(f"   ⚠️ Close to target - need {95 - overall_pct:.1f}% more coverage")
        elif overall_pct >= 80:
            print(f"   ⚠️ Moderate coverage gap - need {95 - overall_pct:.1f}% more coverage")
        else:
            print(f"   ❌ SIGNIFICANT COVERAGE GAP - need {95 - overall_pct:.1f}% more coverage")


def main():
    """Run comprehensive coverage analysis."""
    print("🚀 Starting MemMimic Comprehensive Test Coverage Analysis")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    src_dir = project_root / 'src'
    tests_dir = project_root / 'tests'
    
    if not src_dir.exists() or not tests_dir.exists():
        print(f"❌ Required directories not found:")
        print(f"   src: {src_dir.exists()} ({src_dir})")
        print(f"   tests: {tests_dir.exists()} ({tests_dir})")
        return 1
    
    # Initialize analyzer
    analyzer = CoverageAnalyzer(src_dir, tests_dir)
    
    # Discover modules
    analyzer.discover_source_modules()
    analyzer.discover_test_modules()
    
    # Analyze coverage
    analyzer.analyze_coverage_mapping()
    
    # Generate report
    report = analyzer.generate_coverage_report()
    
    # Print summary
    analyzer.print_coverage_summary(report)
    
    # Save detailed report
    report_file = project_root / 'coverage_analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    overall_coverage = report['overall_statistics']['overall_coverage_percentage']
    
    if overall_coverage >= 95:
        print("\n🎉 COVERAGE ANALYSIS COMPLETE - TARGET ACHIEVED!")
        return 0
    elif overall_coverage >= 80:
        print("\n⚠️ COVERAGE ANALYSIS COMPLETE - IMPROVEMENTS NEEDED")
        return 1
    else:
        print("\n❌ COVERAGE ANALYSIS COMPLETE - SIGNIFICANT GAPS")
        return 2


if __name__ == "__main__":
    sys.exit(main())