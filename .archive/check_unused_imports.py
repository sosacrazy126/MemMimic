#!/usr/bin/env python3
"""
Unused Import Checker for MemMimic
Analyzes Python files to find potentially unused imports.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze imports and their usage."""
    
    def __init__(self):
        self.imports = {}  # {name: (module, line_no)}
        self.used_names = set()
        self.from_imports = {}  # {name: (module, line_no)}
        
    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.imports[name] = (alias.name, node.lineno)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        module = node.module or ''
        for alias in node.names:
            if alias.name == '*':
                # Star imports are complex to analyze, skip them
                continue
            name = alias.asname if alias.asname else alias.name
            self.from_imports[name] = (module, node.lineno)
        self.generic_visit(node)
    
    def visit_Name(self, node):
        self.used_names.add(node.id)
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        # Handle module.attribute access
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)

def analyze_file(file_path: Path) -> Dict[str, List[Tuple[str, int]]]:
    """
    Analyze a Python file for unused imports.
    
    Returns:
        Dict with 'unused_imports' and 'unused_from_imports' lists
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        
        # Additional usage checks for common patterns
        used_in_strings = find_usage_in_strings(content)
        used_in_comments = find_usage_in_comments(content)
        used_in_type_hints = find_usage_in_type_hints(content)
        
        # Combine all usage sources
        all_used = (analyzer.used_names | 
                   used_in_strings | 
                   used_in_comments | 
                   used_in_type_hints)
        
        # Check for unused imports
        unused_imports = []
        unused_from_imports = []
        
        for name, (module, line_no) in analyzer.imports.items():
            if name not in all_used:
                # Additional conservative checks
                if not is_likely_used(name, content):
                    unused_imports.append((name, module, line_no))
        
        for name, (module, line_no) in analyzer.from_imports.items():
            if name not in all_used:
                # Additional conservative checks
                if not is_likely_used(name, content):
                    unused_from_imports.append((name, module, line_no))
        
        return {
            'unused_imports': unused_imports,
            'unused_from_imports': unused_from_imports
        }
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return {'unused_imports': [], 'unused_from_imports': []}

def find_usage_in_strings(content: str) -> Set[str]:
    """Find potential usage of imports in string literals."""
    used = set()
    
    # Find single and double quoted strings
    string_patterns = [
        r'"[^"]*"',  # Double quotes
        r"'[^']*'",  # Single quotes
        r'""".*?"""',  # Triple double quotes
        r"'''.*?'''",  # Triple single quotes
    ]
    
    for pattern in string_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            # Look for potential module names in strings
            words = re.findall(r'\b[a-zA-Z_]\w*\b', match)
            used.update(words)
    
    return used

def find_usage_in_comments(content: str) -> Set[str]:
    """Find potential usage of imports in comments."""
    used = set()
    
    # Find comments
    comment_pattern = r'#.*$'
    matches = re.findall(comment_pattern, content, re.MULTILINE)
    
    for match in matches:
        words = re.findall(r'\b[a-zA-Z_]\w*\b', match)
        used.update(words)
    
    return used

def find_usage_in_type_hints(content: str) -> Set[str]:
    """Find usage in type hints and annotations."""
    used = set()
    
    # Find type annotations
    patterns = [
        r':\s*([A-Za-z_]\w*(?:\[[^\]]*\])?)',  # Variable annotations
        r'->\s*([A-Za-z_]\w*(?:\[[^\]]*\])?)',  # Return annotations
        r'@(\w+)',  # Decorators
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            # Extract type names
            type_parts = re.findall(r'\b[A-Za-z_]\w*\b', match)
            used.update(type_parts)
    
    return used

def is_likely_used(name: str, content: str) -> bool:
    """
    More aggressive check for actual usage that might be missed by AST analysis.
    """
    # First check: simple name occurrence (but be more selective)
    # Only return True if the name is used in actual code, not just mentioned
    
    # Skip very common names that appear everywhere
    if name in ['Any', 'Dict', 'List', 'Optional', 'Union', 'Tuple']:
        # These typing imports are often used in type hints
        if re.search(rf'\b{re.escape(name)}\s*\[', content):  # Used with brackets
            return True
        if re.search(rf':\s*{re.escape(name)}\b', content):  # Used in type annotations
            return True
        if re.search(rf'->\s*{re.escape(name)}\b', content):  # Used in return annotations
            return True
        return False
    
    # Check for module usage patterns
    if '.' in content and f'{name}.' in content:
        return True
    
    # Check for function calls
    if f'{name}(' in content:
        return True
    
    # Special cases for common usage patterns that might be missed
    special_patterns = {
        'sys': [r'sys\.path', r'sys\.argv', r'sys\.exit', r'sys\.stdout', r'sys\.stderr', r'sys\.platform'],
        'os': [r'os\.path', r'os\.environ', r'os\.getcwd', r'os\.makedirs', r'os\.dirname', r'os\.path\.'],
        'json': [r'json\.dumps', r'json\.loads', r'json\.load', r'json\.dump'],
        'logging': [r'logging\.getLogger', r'logging\.info', r'logging\.error', r'logging\.basicConfig'],
        'datetime': [r'datetime\.now', r'datetime\.fromisoformat'],
        'time': [r'time\.time', r'time\.sleep'],
        'argparse': [r'argparse\.ArgumentParser'],
        'pathlib': [r'Path\('],
        'hashlib': [r'hashlib\.'],
        'pickle': [r'pickle\.'],
        'statistics': [r'statistics\.'],
        'threading': [r'threading\.'],
        'sqlite3': [r'sqlite3\.'],
        'contextlib': [r'contextmanager', r'@contextmanager'],
        'enum': [r'Enum\b'],
        'dataclasses': [r'@dataclass', r'dataclass\b', r'field\('],
        'collections': [r'defaultdict', r'Counter'],
        'functools': [r'@wraps', r'wraps\('],
        'math': [r'math\.'],
        're': [r're\.'],
    }
    
    if name in special_patterns:
        for pattern in special_patterns[name]:
            if re.search(pattern, content):
                return True
    
    # Check if it's used in exception handling
    if f'except {name}' in content or f'raise {name}' in content:
        return True
    
    # Check if it's used in isinstance checks
    if f'isinstance(' in content and name in content:
        return True
    
    # If we get here and the name appears in the content, it might be used
    # But be more conservative - only count as used if it appears in non-comment, non-string context
    lines = content.split('\n')
    for line in lines:
        # Skip comments and docstrings
        if line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''"):
            continue
        # Check if name appears in this line
        if re.search(rf'\b{re.escape(name)}\b', line):
            return True
    
    return False

def check_unused_imports_in_directory(directory: Path) -> Dict[str, Dict]:
    """Check all Python files in a directory for unused imports."""
    results = {}
    
    for py_file in directory.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        analysis = analyze_file(py_file)
        if analysis['unused_imports'] or analysis['unused_from_imports']:
            results[str(py_file)] = analysis
    
    return results

def main():
    """Main function to run the unused import analysis."""
    src_dir = Path("/home/evilbastardxd/Desktop/tools/memmimic/src/memmimic")
    
    print("🔍 Analyzing Python files for unused imports...")
    print(f"Directory: {src_dir}")
    print("=" * 80)
    
    # Count total files first
    py_files = list(src_dir.rglob('*.py'))
    py_files = [f for f in py_files if '__pycache__' not in str(f)]
    print(f"Found {len(py_files)} Python files to analyze...")
    
    results = check_unused_imports_in_directory(src_dir)
    
    if not results:
        print("✅ No unused imports found!")
        return
    
    total_unused = 0
    
    for file_path, analysis in results.items():
        rel_path = Path(file_path).relative_to(src_dir.parent.parent)
        
        print(f"\n📄 {rel_path}")
        print("-" * 60)
        
        if analysis['unused_imports']:
            print("❌ Unused imports:")
            for name, module, line_no in analysis['unused_imports']:
                print(f"  Line {line_no:3d}: import {module} (as {name})")
                total_unused += 1
        
        if analysis['unused_from_imports']:
            print("❌ Unused from imports:")
            for name, module, line_no in analysis['unused_from_imports']:
                print(f"  Line {line_no:3d}: from {module} import {name}")
                total_unused += 1
    
    print("\n" + "=" * 80)
    print(f"📊 Summary: {total_unused} potentially unused imports found in {len(results)} files")
    print("\n⚠️  Note: This analysis is conservative. Manual verification recommended.")
    print("   Some imports might be used in ways not detected by static analysis.")

if __name__ == "__main__":
    main()