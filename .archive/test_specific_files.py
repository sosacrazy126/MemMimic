#!/usr/bin/env python3
"""
Test specific files for unused imports
"""

import ast
import re
from pathlib import Path

def analyze_specific_file(file_path: str):
    """Analyze a specific file for unused imports with detailed output."""
    print(f"\n🔍 Analyzing: {file_path}")
    print("-" * 60)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to get imports
        tree = ast.parse(content)
        
        imports = []
        from_imports = []
        
        class ImportCollector(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split('.')[0]
                    imports.append((name, alias.name, node.lineno))
            
            def visit_ImportFrom(self, node):
                module = node.module or ''
                for alias in node.names:
                    if alias.name != '*':
                        name = alias.asname if alias.asname else alias.name
                        from_imports.append((name, module, node.lineno))
        
        collector = ImportCollector()
        collector.visit(tree)
        
        print(f"Found {len(imports)} regular imports and {len(from_imports)} from imports")
        
        # Check each import
        for name, module, line_no in imports:
            used = check_usage_in_content(name, content)
            status = "✅ USED" if used else "❌ UNUSED"
            print(f"  Line {line_no:3d}: import {module} (as {name}) - {status}")
            
        for name, module, line_no in from_imports:
            used = check_usage_in_content(name, content)
            status = "✅ USED" if used else "❌ UNUSED"
            print(f"  Line {line_no:3d}: from {module} import {name} - {status}")
        
    except Exception as e:
        print(f"Error: {e}")

def check_usage_in_content(name: str, content: str) -> bool:
    """Check if a name is used in the content."""
    
    # Check for direct usage patterns
    patterns = [
        rf'\b{re.escape(name)}\.',  # module.something
        rf'\b{re.escape(name)}\(',  # function call
        rf'@{re.escape(name)}\b',   # decorator
        rf'except\s+{re.escape(name)}\b',  # exception handling
        rf'raise\s+{re.escape(name)}\b',   # raising exception
        rf'isinstance\([^,]+,\s*{re.escape(name)}\b',  # isinstance check
    ]
    
    for pattern in patterns:
        if re.search(pattern, content):
            return True
    
    # Special case for typing imports
    if name in ['Dict', 'List', 'Optional', 'Any', 'Union', 'Tuple', 'Set']:
        type_patterns = [
            rf'\b{re.escape(name)}\s*\[',  # Type[something]
            rf':\s*{re.escape(name)}\b',   # variable: Type
            rf'->\s*{re.escape(name)}\b',  # -> Type
        ]
        for pattern in type_patterns:
            if re.search(pattern, content):
                return True
    
    # Check if name appears as a standalone identifier (be more strict)
    lines = content.split('\n')
    for line_no, line in enumerate(lines, 1):
        # Skip comments and strings
        if line.strip().startswith('#'):
            continue
        
        # Remove string literals to avoid false positives
        line_no_strings = re.sub(r'"[^"]*"', '', line)
        line_no_strings = re.sub(r"'[^']*'", '', line_no_strings)
        
        if re.search(rf'\b{re.escape(name)}\b', line_no_strings):
            return True
    
    return False

def main():
    """Test specific files that likely have unused imports."""
    
    test_files = [
        "/home/evilbastardxd/Desktop/tools/memmimic/src/memmimic/mcp/memmimic_save_tale.py",
        "/home/evilbastardxd/Desktop/tools/memmimic/src/memmimic/memory/stale_detector.py",
        "/home/evilbastardxd/Desktop/tools/memmimic/src/memmimic/memory/pattern_analyzer.py",
        "/home/evilbastardxd/Desktop/tools/memmimic/src/memmimic/memory/memory_consolidator.py",
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            analyze_specific_file(file_path)
        else:
            print(f"❌ File not found: {file_path}")

if __name__ == "__main__":
    main()