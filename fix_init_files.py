#!/usr/bin/env python3
"""
Fix missing __init__.py files in MemMimic project
Run this script from the project root to add all missing __init__.py files
"""

import os
from pathlib import Path

# Directories that should have __init__.py files
PYTHON_PACKAGE_DIRS = [
    "src",
    "src/memmimic/cxd/config",  # Already added, but kept for completeness
    "tests",
]

# Directories to skip (not Python packages)
SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    "*.egg-info",
    "logs",
    "models",
    "cxd_cache",
    "tales",  # Data directories, not Python packages
}

def should_skip(dir_path):
    """Check if directory should be skipped"""
    dir_name = os.path.basename(dir_path)
    
    # Skip if in skip list
    if dir_name in SKIP_DIRS:
        return True
    
    # Skip if matches patterns
    for pattern in SKIP_DIRS:
        if "*" in pattern and dir_name.endswith(pattern.replace("*", "")):
            return True
    
    # Skip if under node_modules or other non-Python directories
    if "node_modules" in dir_path or "__pycache__" in dir_path:
        return True
    
    return False

def find_python_directories(root_dir):
    """Find all directories that should be Python packages"""
    python_dirs = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip non-Python directories
        if should_skip(dirpath):
            continue
        
        # Check if directory contains Python files
        has_python_files = any(f.endswith('.py') for f in filenames)
        
        # If it has Python files or is in our explicit list, it should have __init__.py
        if has_python_files or dirpath in PYTHON_PACKAGE_DIRS:
            init_file = os.path.join(dirpath, "__init__.py")
            if not os.path.exists(init_file):
                python_dirs.append(dirpath)
    
    return python_dirs

def create_init_file(dir_path):
    """Create an __init__.py file in the given directory"""
    init_file = os.path.join(dir_path, "__init__.py")
    
    # Generate a simple docstring based on directory name
    dir_name = os.path.basename(dir_path)
    module_name = dir_name.replace("_", " ").title()
    
    content = f'''"""
{module_name} Module
"""
'''
    
    with open(init_file, 'w') as f:
        f.write(content)
    
    return init_file

def main():
    """Main function to fix missing __init__.py files"""
    print("🔍 Scanning for missing __init__.py files...")
    
    # Find project root (where this script is located)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Focus on src directory
    src_dir = os.path.join(project_root, "src")
    
    if not os.path.exists(src_dir):
        print("❌ src/ directory not found!")
        return
    
    # Find directories missing __init__.py
    missing_dirs = find_python_directories(src_dir)
    
    if not missing_dirs:
        print("✅ All Python packages already have __init__.py files!")
        return
    
    print(f"📦 Found {len(missing_dirs)} directories missing __init__.py files")
    
    # Create __init__.py files
    created_files = []
    for dir_path in missing_dirs:
        try:
            init_file = create_init_file(dir_path)
            created_files.append(init_file)
            print(f"✅ Created: {init_file}")
        except Exception as e:
            print(f"❌ Failed to create __init__.py in {dir_path}: {e}")
    
    print(f"\n🎉 Successfully created {len(created_files)} __init__.py files!")
    
    # Show the most important ones
    important_dirs = [
        "src",
        "src/memmimic/cxd/config",
    ]
    
    print("\n📌 Important directories fixed:")
    for dir_path in important_dirs:
        init_file = os.path.join(project_root, dir_path, "__init__.py")
        if os.path.exists(init_file):
            print(f"  ✅ {dir_path}")

if __name__ == "__main__":
    main()