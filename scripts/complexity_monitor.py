#!/usr/bin/env python3
"""
Code Complexity Monitor - Automated complexity analysis and monitoring
Analyzes Python files for complexity metrics and generates reports
"""

import ast
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a function or class"""
    name: str
    type: str  # 'function' or 'class'
    file_path: str
    line_start: int
    line_end: int
    lines_of_code: int
    cyclomatic_complexity: int
    nesting_depth: int
    parameter_count: int
    return_count: int
    branch_count: int
    loop_count: int
    
    @property
    def complexity_score(self) -> float:
        """Calculate overall complexity score (0-100)"""
        # Weighted complexity calculation
        cc_weight = 0.4
        loc_weight = 0.2
        nesting_weight = 0.2
        param_weight = 0.1
        branch_weight = 0.1
        
        # Normalize metrics
        cc_norm = min(self.cyclomatic_complexity / 10.0, 3.0)  # Cap at 30
        loc_norm = min(self.lines_of_code / 50.0, 2.0)  # Cap at 100
        nest_norm = min(self.nesting_depth / 5.0, 2.0)  # Cap at 10
        param_norm = min(self.parameter_count / 5.0, 2.0)  # Cap at 10
        branch_norm = min(self.branch_count / 10.0, 2.0)  # Cap at 20
        
        score = (
            cc_norm * cc_weight +
            loc_norm * loc_weight +
            nest_norm * nesting_weight +
            param_norm * param_weight +
            branch_norm * branch_weight
        ) * 100 / 3.0  # Scale to 0-100
        
        return min(score, 100.0)
    
    @property
    def complexity_level(self) -> str:
        """Get complexity level classification"""
        score = self.complexity_score
        if score < 20:
            return "LOW"
        elif score < 40:
            return "MODERATE"
        elif score < 60:
            return "HIGH"
        elif score < 80:
            return "VERY_HIGH"
        else:
            return "CRITICAL"


@dataclass
class FileComplexity:
    """Complexity metrics for an entire file"""
    file_path: str
    total_lines: int
    function_count: int
    class_count: int
    functions: List[ComplexityMetrics]
    classes: List[ComplexityMetrics]
    average_complexity: float
    max_complexity: float
    high_complexity_count: int
    
    @property
    def file_complexity_score(self) -> float:
        """Calculate overall file complexity score"""
        if not self.functions and not self.classes:
            return 0.0
        
        all_items = self.functions + self.classes
        return sum(item.complexity_score for item in all_items) / len(all_items)


class ComplexityAnalyzer(ast.NodeVisitor):
    """AST visitor for analyzing code complexity"""
    
    def __init__(self):
        self.current_function = None
        self.current_class = None
        self.complexity_metrics = []
        self.nesting_level = 0
        self.function_stack = []
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition"""
        self._analyze_function(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition"""
        self._analyze_function(node)
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition"""
        self._analyze_class(node)
        
    def _analyze_function(self, node: ast.FunctionDef):
        """Analyze function complexity"""
        # Calculate function metrics
        lines_of_code = node.end_lineno - node.lineno + 1
        parameter_count = len(node.args.args)
        
        # Calculate cyclomatic complexity
        complexity_visitor = CyclomaticComplexityVisitor()
        complexity_visitor.visit(node)
        
        # Calculate nesting depth
        nesting_visitor = NestingDepthVisitor()
        nesting_visitor.visit(node)
        
        # Count returns and branches
        return_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))
        branch_count = sum(1 for _ in ast.walk(node) if isinstance(_, (ast.If, ast.While, ast.For, ast.With, ast.Try)))
        loop_count = sum(1 for _ in ast.walk(node) if isinstance(_, (ast.While, ast.For)))
        
        metrics = ComplexityMetrics(
            name=node.name,
            type='function',
            file_path='',  # Will be set by caller
            line_start=node.lineno,
            line_end=node.end_lineno,
            lines_of_code=lines_of_code,
            cyclomatic_complexity=complexity_visitor.complexity,
            nesting_depth=nesting_visitor.max_depth,
            parameter_count=parameter_count,
            return_count=return_count,
            branch_count=branch_count,
            loop_count=loop_count
        )
        
        self.complexity_metrics.append(metrics)
        
        # Continue visiting child nodes
        self.generic_visit(node)
    
    def _analyze_class(self, node: ast.ClassDef):
        """Analyze class complexity"""
        lines_of_code = node.end_lineno - node.lineno + 1
        
        # Count methods in class
        method_count = sum(1 for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))
        
        # Estimate class complexity based on methods and inheritance
        base_count = len(node.bases)
        decorator_count = len(node.decorator_list)
        
        # Simple heuristic for class complexity
        estimated_complexity = max(1, method_count // 2 + base_count + decorator_count)
        
        metrics = ComplexityMetrics(
            name=node.name,
            type='class',
            file_path='',  # Will be set by caller
            line_start=node.lineno,
            line_end=node.end_lineno,
            lines_of_code=lines_of_code,
            cyclomatic_complexity=estimated_complexity,
            nesting_depth=1,  # Classes have base nesting of 1
            parameter_count=base_count,  # Use base classes as parameters
            return_count=0,
            branch_count=0,
            loop_count=0
        )
        
        self.complexity_metrics.append(metrics)
        
        # Continue visiting child nodes
        self.generic_visit(node)


class CyclomaticComplexityVisitor(ast.NodeVisitor):
    """Calculate cyclomatic complexity"""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
    
    def visit_If(self, node):
        self.complexity += 1
        if node.orelse:
            # Check if orelse contains another If (elif)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                pass  # Don't count elif as separate complexity
            else:
                pass  # else clause doesn't add complexity
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_With(self, node):
        self.complexity += 1
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        # Add complexity for each additional boolean condition
        self.complexity += len(node.values) - 1
        self.generic_visit(node)
    
    def visit_ListComp(self, node):
        # List comprehensions add complexity
        self.complexity += 1
        for generator in node.generators:
            if generator.ifs:
                self.complexity += len(generator.ifs)
        self.generic_visit(node)
    
    def visit_SetComp(self, node):
        self.complexity += 1
        for generator in node.generators:
            if generator.ifs:
                self.complexity += len(generator.ifs)
        self.generic_visit(node)
    
    def visit_DictComp(self, node):
        self.complexity += 1
        for generator in node.generators:
            if generator.ifs:
                self.complexity += len(generator.ifs)
        self.generic_visit(node)


class NestingDepthVisitor(ast.NodeVisitor):
    """Calculate maximum nesting depth"""
    
    def __init__(self):
        self.current_depth = 0
        self.max_depth = 0
    
    def _visit_nesting_node(self, node):
        """Visit node that increases nesting"""
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1
    
    def visit_If(self, node):
        self._visit_nesting_node(node)
    
    def visit_While(self, node):
        self._visit_nesting_node(node)
    
    def visit_For(self, node):
        self._visit_nesting_node(node)
    
    def visit_With(self, node):
        self._visit_nesting_node(node)
    
    def visit_Try(self, node):
        self._visit_nesting_node(node)
    
    def visit_FunctionDef(self, node):
        self._visit_nesting_node(node)
    
    def visit_AsyncFunctionDef(self, node):
        self._visit_nesting_node(node)
    
    def visit_ClassDef(self, node):
        self._visit_nesting_node(node)


class ComplexityMonitor:
    """Main complexity monitoring system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_file)
        self.results = {}
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            "thresholds": {
                "cyclomatic_complexity": 10,
                "lines_of_code": 50,
                "nesting_depth": 4,
                "parameter_count": 5,
                "complexity_score": 50
            },
            "exclude_patterns": [
                "__pycache__",
                ".git",
                ".pytest_cache",
                "test_",
                "_test.py"
            ],
            "include_patterns": [
                "*.py"
            ],
            "output_formats": ["json", "text"],
            "report_all": False
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config {config_file}: {e}")
        
        return default_config
    
    def analyze_file(self, file_path: str) -> Optional[FileComplexity]:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Analyze complexity
            analyzer = ComplexityAnalyzer()
            analyzer.visit(tree)
            
            # Set file path for all metrics
            for metric in analyzer.complexity_metrics:
                metric.file_path = file_path
            
            # Separate functions and classes
            functions = [m for m in analyzer.complexity_metrics if m.type == 'function']
            classes = [m for m in analyzer.complexity_metrics if m.type == 'class']
            
            # Calculate file metrics
            total_lines = len(content.split('\n'))
            all_items = functions + classes
            
            if all_items:
                average_complexity = sum(item.complexity_score for item in all_items) / len(all_items)
                max_complexity = max(item.complexity_score for item in all_items)
                high_complexity_count = sum(1 for item in all_items if item.complexity_score > self.config['thresholds']['complexity_score'])
            else:
                average_complexity = 0.0
                max_complexity = 0.0
                high_complexity_count = 0
            
            return FileComplexity(
                file_path=file_path,
                total_lines=total_lines,
                function_count=len(functions),
                class_count=len(classes),
                functions=functions,
                classes=classes,
                average_complexity=average_complexity,
                max_complexity=max_complexity,
                high_complexity_count=high_complexity_count
            )
        
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def analyze_directory(self, directory: str, recursive: bool = True) -> Dict[str, FileComplexity]:
        """Analyze all Python files in a directory"""
        results = {}
        directory_path = Path(directory)
        
        if recursive:
            pattern = "**/*.py"
        else:
            pattern = "*.py"
        
        for file_path in directory_path.glob(pattern):
            # Skip excluded patterns
            if any(pattern in str(file_path) for pattern in self.config['exclude_patterns']):
                continue
            
            file_complexity = self.analyze_file(str(file_path))
            if file_complexity:
                results[str(file_path)] = file_complexity
        
        return results
    
    def generate_report(self, results: Dict[str, FileComplexity], output_format: str = "text") -> str:
        """Generate complexity report"""
        if output_format == "json":
            return self._generate_json_report(results)
        else:
            return self._generate_text_report(results)
    
    def _generate_text_report(self, results: Dict[str, FileComplexity]) -> str:
        """Generate human-readable text report"""
        report = []
        report.append("=" * 80)
        report.append("CODE COMPLEXITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Files analyzed: {len(results)}")
        report.append("")
        
        # Summary statistics
        all_functions = []
        all_classes = []
        high_complexity_items = []
        
        for file_complexity in results.values():
            all_functions.extend(file_complexity.functions)
            all_classes.extend(file_complexity.classes)
            
            for item in file_complexity.functions + file_complexity.classes:
                if item.complexity_score > self.config['thresholds']['complexity_score']:
                    high_complexity_items.append(item)
        
        report.append("SUMMARY:")
        report.append(f"  Total functions: {len(all_functions)}")
        report.append(f"  Total classes: {len(all_classes)}")
        report.append(f"  High complexity items: {len(high_complexity_items)}")
        
        if all_functions + all_classes:
            avg_complexity = sum(item.complexity_score for item in all_functions + all_classes) / len(all_functions + all_classes)
            report.append(f"  Average complexity score: {avg_complexity:.1f}")
        
        report.append("")
        
        # High complexity items
        if high_complexity_items:
            report.append("HIGH COMPLEXITY ITEMS (requiring refactoring):")
            report.append("-" * 50)
            
            # Sort by complexity score
            high_complexity_items.sort(key=lambda x: x.complexity_score, reverse=True)
            
            for item in high_complexity_items:
                report.append(f"{item.name} ({item.type})")
                report.append(f"  File: {item.file_path}")
                report.append(f"  Lines: {item.line_start}-{item.line_end} ({item.lines_of_code} LOC)")
                report.append(f"  Complexity Score: {item.complexity_score:.1f} ({item.complexity_level})")
                report.append(f"  Cyclomatic Complexity: {item.cyclomatic_complexity}")
                report.append(f"  Nesting Depth: {item.nesting_depth}")
                report.append(f"  Parameters: {item.parameter_count}")
                report.append("")
        
        # File-by-file breakdown
        if self.config.get('report_all', False):
            report.append("FILE-BY-FILE ANALYSIS:")
            report.append("-" * 50)
            
            for file_path, file_complexity in sorted(results.items()):
                report.append(f"File: {file_path}")
                report.append(f"  Lines: {file_complexity.total_lines}")
                report.append(f"  Functions: {file_complexity.function_count}")
                report.append(f"  Classes: {file_complexity.class_count}")
                report.append(f"  Average Complexity: {file_complexity.average_complexity:.1f}")
                report.append(f"  Max Complexity: {file_complexity.max_complexity:.1f}")
                report.append(f"  High Complexity Items: {file_complexity.high_complexity_count}")
                report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 30)
        if high_complexity_items:
            report.append("1. Refactor functions/classes with complexity score > 50")
            report.append("2. Break down large functions using Extract Method pattern")
            report.append("3. Reduce nesting depth using Guard Clauses")
            report.append("4. Consider splitting classes with too many responsibilities")
        else:
            report.append("✓ All code complexity is within acceptable limits!")
        
        return "\n".join(report)
    
    def _generate_json_report(self, results: Dict[str, FileComplexity]) -> str:
        """Generate JSON report for programmatic use"""
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "summary": {
                "files_analyzed": len(results),
                "total_functions": sum(len(fc.functions) for fc in results.values()),
                "total_classes": sum(len(fc.classes) for fc in results.values()),
            },
            "files": {}
        }
        
        # Add high complexity items to summary
        high_complexity_items = []
        for file_complexity in results.values():
            for item in file_complexity.functions + file_complexity.classes:
                if item.complexity_score > self.config['thresholds']['complexity_score']:
                    high_complexity_items.append(asdict(item))
        
        json_data["summary"]["high_complexity_count"] = len(high_complexity_items)
        json_data["high_complexity_items"] = high_complexity_items
        
        # Add file details
        for file_path, file_complexity in results.items():
            json_data["files"][file_path] = {
                "total_lines": file_complexity.total_lines,
                "function_count": file_complexity.function_count,
                "class_count": file_complexity.class_count,
                "average_complexity": file_complexity.average_complexity,
                "max_complexity": file_complexity.max_complexity,
                "high_complexity_count": file_complexity.high_complexity_count,
                "functions": [asdict(func) for func in file_complexity.functions],
                "classes": [asdict(cls) for cls in file_complexity.classes]
            }
        
        return json.dumps(json_data, indent=2)
    
    def save_report(self, report: str, output_file: str):
        """Save report to file"""
        try:
            with open(output_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def check_thresholds(self, results: Dict[str, FileComplexity]) -> bool:
        """Check if any items exceed complexity thresholds"""
        thresholds = self.config['thresholds']
        violations = []
        
        for file_complexity in results.values():
            for item in file_complexity.functions + file_complexity.classes:
                if item.cyclomatic_complexity > thresholds['cyclomatic_complexity']:
                    violations.append(f"{item.name}: CC={item.cyclomatic_complexity}")
                if item.lines_of_code > thresholds['lines_of_code']:
                    violations.append(f"{item.name}: LOC={item.lines_of_code}")
                if item.nesting_depth > thresholds['nesting_depth']:
                    violations.append(f"{item.name}: Nesting={item.nesting_depth}")
        
        if violations:
            self.logger.warning(f"Complexity threshold violations: {violations}")
            return False
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Complexity Monitor")
    parser.add_argument("path", help="File or directory to analyze")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--recursive", action="store_true", help="Recursively analyze directories")
    parser.add_argument("--fail-on-threshold", action="store_true", help="Exit with error code if thresholds exceeded")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Create monitor
    monitor = ComplexityMonitor(args.config)
    
    # Analyze path
    if os.path.isfile(args.path):
        file_complexity = monitor.analyze_file(args.path)
        if file_complexity:
            results = {args.path: file_complexity}
        else:
            print(f"Failed to analyze {args.path}")
            sys.exit(1)
    elif os.path.isdir(args.path):
        results = monitor.analyze_directory(args.path, args.recursive)
        if not results:
            print(f"No Python files found in {args.path}")
            sys.exit(1)
    else:
        print(f"Path not found: {args.path}")
        sys.exit(1)
    
    # Generate report
    report = monitor.generate_report(results, args.format)
    
    # Output report
    if args.output:
        monitor.save_report(report, args.output)
    else:
        print(report)
    
    # Check thresholds
    if args.fail_on_threshold:
        if not monitor.check_thresholds(results):
            sys.exit(1)


if __name__ == "__main__":
    main()