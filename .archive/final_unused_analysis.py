#!/usr/bin/env python3
"""
Final comprehensive unused import analysis for MemMimic
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set

def check_specific_unused_patterns(file_path: Path) -> List[str]:
    """Check for specific patterns that are commonly unused."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return []
    
    unused_imports = []
    
    # Parse imports
    try:
        tree = ast.parse(content)
    except:
        return []
    
    imports = {}
    from_imports = {}
    
    class ImportCollector(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split('.')[0]
                imports[name] = (alias.name, node.lineno)
        
        def visit_ImportFrom(self, node):
            module = node.module or ''
            for alias in node.names:
                if alias.name != '*':
                    name = alias.asname if alias.asname else alias.name
                    from_imports[name] = (module, node.lineno)
    
    collector = ImportCollector()
    collector.visit(tree)
    
    # Check each import with aggressive detection
    all_imports = list(imports.items()) + list(from_imports.items())
    
    for name, (module_info, line_no) in all_imports:
        is_used = False
        
        # Comprehensive usage checks
        usage_patterns = [
            rf'\b{re.escape(name)}\.',  # Module usage
            rf'\b{re.escape(name)}\(',  # Function call
            rf'@{re.escape(name)}\b',   # Decorator
            rf'except\s+{re.escape(name)}\b',  # Exception
            rf'raise\s+{re.escape(name)}\b',   # Raise
            rf'isinstance\([^,)]+,\s*{re.escape(name)}\b',  # isinstance
            rf'\b{re.escape(name)}\s*\[',  # Type annotation with brackets
            rf':\s*{re.escape(name)}\b',   # Type annotation
            rf'->\s*{re.escape(name)}\b',  # Return type
            rf'=\s*{re.escape(name)}\b',   # Assignment
            rf'\b{re.escape(name)}\s*=',   # Assignment target
        ]
        
        for pattern in usage_patterns:
            if re.search(pattern, content):
                is_used = True
                break
        
        # Special case checks
        if not is_used:
            # Check if it's used in string formatting or f-strings
            if re.search(rf'f["\'].*\b{re.escape(name)}\b.*["\']', content):
                is_used = True
            
            # Check if it's used in exec/eval statements
            if f'exec(' in content or f'eval(' in content:
                if name in content:
                    is_used = True
            
            # Check for indirect usage through getattr, hasattr, etc.
            if re.search(rf'(getattr|hasattr|setattr|delattr)\([^,)]*,\s*["\']?{re.escape(name)}["\']?', content):
                is_used = True
        
        if not is_used:
            if name in from_imports:
                module, line_no = from_imports[name]
                unused_imports.append(f"Line {line_no:3d}: from {module} import {name}")
            else:
                module, line_no = imports[name]
                unused_imports.append(f"Line {line_no:3d}: import {module} (as {name})")
    
    return unused_imports

def analyze_all_files():
    """Analyze all Python files in the MemMimic codebase."""
    src_dir = Path("/home/evilbastardxd/Desktop/tools/memmimic/src/memmimic")
    
    print("🔍 Final Analysis: Searching for unused imports in MemMimic codebase")
    print("=" * 80)
    
    files_with_unused = {}
    total_unused = 0
    
    for py_file in src_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        
        unused = check_specific_unused_patterns(py_file)
        
        if unused:
            rel_path = py_file.relative_to(src_dir.parent.parent)
            files_with_unused[str(rel_path)] = unused
            total_unused += len(unused)
    
    if not files_with_unused:
        print("✅ No unused imports found in the codebase!")
        print("\n📊 Analysis Summary:")
        print("   - All imports appear to be used")
        print("   - This indicates good code hygiene")
        print("   - The codebase has minimal import bloat")
        return
    
    print(f"❌ Found {total_unused} potentially unused imports across {len(files_with_unused)} files:")
    print()
    
    for file_path, unused_list in files_with_unused.items():
        print(f"📄 {file_path}")
        print("-" * 60)
        for unused_import in unused_list:
            print(f"  {unused_import}")
        print()
    
    print("=" * 80)
    print(f"📊 Summary: {total_unused} unused imports in {len(files_with_unused)} files")
    print()
    print("⚠️  Recommendations:")
    print("   1. Review each unused import manually")
    print("   2. Remove confirmed unused imports to reduce code bloat")
    print("   3. Some imports might be used in ways not detected by static analysis")
    print("   4. Consider using tools like 'autoflake' for automatic removal")

if __name__ == "__main__":
    analyze_all_files()