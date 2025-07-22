#!/usr/bin/env python3
"""
Root folder cleanup and organization script for MemMimic project
Removes temporary files, test artifacts, and organizes structure
"""

import os
import shutil
import glob
import json
from pathlib import Path
from datetime import datetime

def organize_root_folder():
    """Organize root folder by moving temp files and organizing structure"""
    root = Path("/home/evilbastardxd/Desktop/tools/memmimic")
    
    # Files to remove (temporary/test artifacts)
    temp_files = [
        "test.db", "test_api.db", "memmimic.db", "memmimic_memories.db", "memory_queue.db",
        "test_specific_files.py", "final_unused_analysis.py", "verify_implementation.py",
        "check_unused_imports.py", "migrate_to_amms.py", "ml_integration_example.py",
        "*.json", "*.log", "async_vs_sync_benchmark_report.json", "cache_performance_report.json",
        "coverage.json", "coverage_analysis_report.json", "import_analysis_report.json",
        "optimized_cache_performance_report.json", "performance_dashboard.json"
    ]
    
    # Directories to remove/clean
    temp_dirs = [
        "cxd_cache", "logs", "models", "venv", 
        "src/cxd_cache", "src/logs", "src/models", "src/memmimic_cache"
    ]
    
    # Create archive directory for moved files
    archive_dir = root / ".archive"
    archive_dir.mkdir(exist_ok=True)
    
    # Move temporary files to archive
    moved_files = []
    for pattern in temp_files:
        for file_path in root.glob(pattern):
            if file_path.is_file():
                try:
                    dest = archive_dir / file_path.name
                    shutil.move(str(file_path), str(dest))
                    moved_files.append(str(file_path.relative_to(root)))
                except Exception as e:
                    print(f"Error moving {file_path}: {e}")
    
    # Remove/move temp directories
    removed_dirs = []
    for temp_dir in temp_dirs:
        dir_path = root / temp_dir
        if dir_path.exists():
            try:
                if temp_dir in ["venv", "models"]:
                    # Remove completely
                    shutil.rmtree(dir_path)
                    removed_dirs.append(temp_dir + " (removed)")
                else:
                    # Move to archive
                    dest = archive_dir / temp_dir
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(dir_path), str(dest))
                    removed_dirs.append(temp_dir + " (archived)")
            except Exception as e:
                print(f"Error handling {dir_path}: {e}")
    
    # Clean duplicate src/ subdirectories
    src_duplicates = ["src/memmimic_memories.db", "src/memory_queue.db", "src/tales"]
    for dup in src_duplicates:
        dup_path = root / dup
        if dup_path.exists():
            try:
                if dup_path.is_file():
                    dup_path.unlink()
                else:
                    shutil.rmtree(dup_path)
                removed_dirs.append(f"{dup} (duplicate removed)")
            except Exception as e:
                print(f"Error removing duplicate {dup_path}: {e}")
    
    # Create organized structure
    organized_dirs = {
        "archive": "Historical and temporary files",
        "config": "Configuration files", 
        "docs": "Documentation",
        "examples": "Usage examples",
        "infrastructure": "Deployment infrastructure",
        "scripts": "Utility scripts",
        "src": "Source code",
        "tests": "Test suite",
        "tales": "Memory narratives"
    }
    
    # Create .gitignore if missing
    gitignore_path = root / ".gitignore"
    gitignore_content = """
# Temporary files
*.pyc
__pycache__/
*.db
*.log
*.tmp
.pytest_cache/
.coverage
.env
.venv/
venv/

# Cache directories
*cache*/
models/
logs/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Build artifacts
build/
dist/
*.egg-info/
"""
    
    if not gitignore_path.exists():
        gitignore_path.write_text(gitignore_content.strip())
    
    # Generate cleanup report
    report = {
        "timestamp": datetime.now().isoformat(),
        "moved_files": moved_files,
        "handled_directories": removed_dirs,
        "organized_structure": organized_dirs,
        "root_files_remaining": [f.name for f in root.iterdir() if f.is_file()],
        "root_dirs_remaining": [d.name for d in root.iterdir() if d.is_dir()]
    }
    
    # Save report
    report_path = root / "cleanup_organization_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Root folder organized successfully!")
    print(f"📁 Moved {len(moved_files)} files to .archive/")
    print(f"📂 Handled {len(removed_dirs)} directories")
    print(f"📋 Report saved to: {report_path}")
    
    return report

if __name__ == "__main__":
    organize_root_folder()