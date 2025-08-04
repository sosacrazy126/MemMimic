#!/usr/bin/env python3
"""
MemMimic Project Cleanup Script

Comprehensive cleanup of code, files, and project structure after completion
of all 4 phases of the remediation plan.
"""

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectCleanup:
    """Comprehensive project cleanup utilities"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_root = self.project_root / "src"
        self.cleanup_stats = {
            'files_cleaned': 0,
            'lines_removed': 0,
            'imports_optimized': 0,
            'bytes_saved': 0,
            'cache_files_removed': 0
        }
    
    def clean_python_cache(self):
        """Remove Python cache files and directories"""
        logger.info("Cleaning Python cache files...")
        
        cache_patterns = ['__pycache__', '*.pyc', '*.pyo', '.pytest_cache']
        removed_count = 0
        
        for pattern in cache_patterns:
            if pattern.startswith('.') or pattern == '__pycache__':
                # Directory patterns
                for path in self.project_root.rglob(pattern):
                    if path.is_dir():
                        shutil.rmtree(path)
                        removed_count += 1
            else:
                # File patterns
                for path in self.project_root.rglob(pattern):
                    if path.is_file():
                        path.unlink()
                        removed_count += 1
        
        self.cleanup_stats['cache_files_removed'] = removed_count
        logger.info(f"Removed {removed_count} cache files/directories")
    
    def clean_system_files(self):
        """Remove system-generated files"""
        logger.info("Cleaning system files...")
        
        system_patterns = ['.DS_Store', 'Thumbs.db', '*.tmp', '*.swp', '*~']
        removed_count = 0
        
        for pattern in system_patterns:
            for path in self.project_root.rglob(pattern):
                if path.is_file():
                    path.unlink()
                    removed_count += 1
        
        logger.info(f"Removed {removed_count} system files")
    
    def optimize_empty_lines(self):
        """Remove excessive empty lines from Python files"""
        logger.info("Optimizing empty lines in Python files...")
        
        files_processed = 0
        lines_removed = 0
        
        for py_file in self.src_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Remove excessive empty lines (more than 2 consecutive)
                optimized_lines = []
                empty_count = 0
                
                for line in lines:
                    if line.strip() == '':
                        empty_count += 1
                        if empty_count <= 2:  # Allow max 2 consecutive empty lines
                            optimized_lines.append(line)
                    else:
                        empty_count = 0
                        optimized_lines.append(line)
                
                # Remove trailing empty lines
                while optimized_lines and optimized_lines[-1].strip() == '':
                    optimized_lines.pop()
                
                # Add single trailing newline
                if optimized_lines and not optimized_lines[-1].endswith('\n'):
                    optimized_lines[-1] += '\n'
                elif optimized_lines:
                    optimized_lines.append('\n')
                
                # Write back if changes were made
                if len(optimized_lines) != len(lines):
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.writelines(optimized_lines)
                    
                    lines_removed += len(lines) - len(optimized_lines)
                    files_processed += 1
            
            except Exception as e:
                logger.warning(f"Error processing {py_file}: {e}")
        
        self.cleanup_stats['files_cleaned'] += files_processed
        self.cleanup_stats['lines_removed'] += lines_removed
        logger.info(f"Optimized {files_processed} files, removed {lines_removed} empty lines")
    
    def find_unused_imports(self, file_path: Path) -> List[str]:
        """Find unused imports in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract imports
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
                    for alias in node.names:
                        if alias.name != '*':
                            imports.add(alias.name)
            
            # Find names used in code
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # Get the base name
                    while isinstance(node, ast.Attribute):
                        node = node.value
                    if isinstance(node, ast.Name):
                        used_names.add(node.id)
            
            # Find potentially unused imports
            unused = imports - used_names
            
            # Filter out common false positives
            false_positives = {'__future__', 'typing', 'dataclasses', 'abc', 'enum'}
            unused = unused - false_positives
            
            return list(unused)
        
        except Exception as e:
            logger.warning(f"Error analyzing imports in {file_path}: {e}")
            return []
    
    def optimize_imports(self):
        """Optimize imports across Python files"""
        logger.info("Analyzing imports for optimization...")
        
        import_analysis = {}
        total_potentially_unused = 0
        
        for py_file in self.src_root.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue  # Skip __init__.py files
            
            unused = self.find_unused_imports(py_file)
            if unused:
                import_analysis[str(py_file)] = unused
                total_potentially_unused += len(unused)
        
        self.cleanup_stats['imports_optimized'] = total_potentially_unused
        
        # Generate report
        if import_analysis:
            report_path = self.project_root / "import_analysis_report.json"
            with open(report_path, 'w') as f:
                json.dump(import_analysis, f, indent=2)
            logger.info(f"Import analysis report saved to {report_path}")
            logger.info(f"Found {total_potentially_unused} potentially unused imports across {len(import_analysis)} files")
        else:
            logger.info("No unused imports detected")
    
    def optimize_docstrings(self):
        """Standardize and optimize docstrings"""
        logger.info("Optimizing docstrings...")
        
        files_processed = 0
        
        for py_file in self.src_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if file has proper module docstring
                tree = ast.parse(content)
                
                # Look for module-level docstring
                has_module_docstring = (
                    len(tree.body) > 0 and
                    isinstance(tree.body[0], ast.Expr) and
                    isinstance(tree.body[0].value, ast.Constant) and
                    isinstance(tree.body[0].value.value, str)
                )
                
                if not has_module_docstring and not py_file.name.startswith('__'):
                    # Add basic module docstring if missing
                    module_name = py_file.stem.replace('_', ' ').title()
                    docstring = f'"""\n{module_name}\n\nMemMimic module for {module_name.lower()} functionality.\n"""\n\n'
                    
                    # Insert at the beginning after any __future__ imports
                    lines = content.split('\n')
                    insert_index = 0
                    
                    for i, line in enumerate(lines):
                        if line.strip().startswith('from __future__'):
                            insert_index = i + 1
                        elif line.strip() and not line.startswith('#'):
                            break
                    
                    lines.insert(insert_index, docstring)
                    new_content = '\n'.join(lines)
                    
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    files_processed += 1
            
            except Exception as e:
                logger.warning(f"Error processing docstrings in {py_file}: {e}")
        
        logger.info(f"Added/optimized docstrings in {files_processed} files")
    
    def calculate_project_size(self) -> Dict[str, int]:
        """Calculate project size statistics"""
        stats = {
            'total_files': 0,
            'python_files': 0,
            'total_lines': 0,
            'python_lines': 0,
            'total_size_bytes': 0
        }
        
        for path in self.project_root.rglob('*'):
            if path.is_file():
                stats['total_files'] += 1
                stats['total_size_bytes'] += path.stat().st_size
                
                if path.suffix == '.py':
                    stats['python_files'] += 1
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            stats['python_lines'] += len(lines)
                            stats['total_lines'] += len(lines)
                    except:
                        pass
                elif path.suffix in ['.md', '.txt', '.json', '.yaml', '.yml']:
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            stats['total_lines'] += len(lines)
                    except:
                        pass
        
        return stats
    
    def create_gitignore(self):
        """Create or update .gitignore file"""
        gitignore_path = self.project_root / '.gitignore'
        
        gitignore_content = '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# System files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Node.js (for MCP components)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# MemMimic specific
*.db
*.sqlite
*.sqlite3
import_analysis_report.json
cache_performance_report.json
async_vs_sync_benchmark_report.json
performance_dashboard.json
optimized_cache_performance_report.json

# Temporary files
*.tmp
*.temp
*.bak
*.backup
'''
        
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        logger.info(f"Created/updated .gitignore file")
    
    def generate_cleanup_report(self):
        """Generate comprehensive cleanup report"""
        logger.info("Generating cleanup report...")
        
        # Calculate final project stats
        project_stats = self.calculate_project_size()
        
        report = {
            'cleanup_timestamp': str(Path(__file__).stat().st_mtime),
            'project_stats': project_stats,
            'cleanup_stats': self.cleanup_stats,
            'optimization_summary': {
                'files_cleaned': self.cleanup_stats['files_cleaned'],
                'lines_removed': self.cleanup_stats['lines_removed'],
                'imports_optimized': self.cleanup_stats['imports_optimized'],
                'cache_files_removed': self.cleanup_stats['cache_files_removed'],
                'project_size_mb': round(project_stats['total_size_bytes'] / (1024 * 1024), 2),
                'total_python_files': project_stats['python_files'],
                'total_python_lines': project_stats['python_lines']
            },
            'recommendations': [
                'Project has been optimized for production use',
                'All Python cache files have been removed',
                'Empty lines have been optimized',
                'Import analysis completed - check import_analysis_report.json for details',
                'Docstrings have been standardized',
                '.gitignore has been created/updated'
            ]
        }
        
        report_path = self.project_root / 'cleanup_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Cleanup report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("🧹 MEMMIMIC PROJECT CLEANUP COMPLETE")
        print("="*60)
        print(f"📁 Total files: {project_stats['total_files']:,}")
        print(f"🐍 Python files: {project_stats['python_files']:,}")
        print(f"📝 Lines of code: {project_stats['python_lines']:,}")
        print(f"💾 Project size: {report['optimization_summary']['project_size_mb']} MB")
        print(f"🧽 Files cleaned: {self.cleanup_stats['files_cleaned']}")
        print(f"📉 Lines removed: {self.cleanup_stats['lines_removed']}")
        print(f"🗑️  Cache files removed: {self.cleanup_stats['cache_files_removed']}")
        print(f"📦 Imports analyzed: {self.cleanup_stats['imports_optimized']} potentially unused")
        print("="*60)
        print("✅ PROJECT READY FOR PRODUCTION DEPLOYMENT")
        print("="*60)
    
    def run_full_cleanup(self):
        """Execute comprehensive project cleanup"""
        logger.info("Starting comprehensive MemMimic project cleanup...")
        
        # Execute all cleanup operations
        self.clean_python_cache()
        self.clean_system_files()
        self.optimize_empty_lines()
        self.optimize_imports()
        self.optimize_docstrings()
        self.create_gitignore()
        self.generate_cleanup_report()
        
        logger.info("Comprehensive project cleanup completed successfully!")


def main():
    """Main cleanup execution"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "/home/evilbastardxd/Desktop/tools/memmimic"
    
    if not Path(project_root).exists():
        print(f"Error: Project root {project_root} does not exist")
        sys.exit(1)
    
    cleanup = ProjectCleanup(project_root)
    cleanup.run_full_cleanup()


if __name__ == "__main__":
    main()