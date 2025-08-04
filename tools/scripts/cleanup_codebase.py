#!/usr/bin/env python3
"""
MemMimic Codebase Cleanup Script

This script performs routine maintenance on the MemMimic codebase:
- Removes Python cache files and directories
- Cleans up temporary files
- Removes duplicate cache directories
- Validates configuration files
- Reports on unused imports (optional)
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Set
import tempfile
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def find_and_remove_python_cache(root_dir: Path) -> int:
    """Remove all Python cache files and directories"""
    removed_count = 0
    
    # Remove __pycache__ directories
    for pycache_dir in root_dir.rglob("__pycache__"):
        if pycache_dir.is_dir():
            logger.debug(f"Removing {pycache_dir}")
            shutil.rmtree(pycache_dir)
            removed_count += 1
    
    # Remove .pyc, .pyo files
    for pyc_file in root_dir.rglob("*.pyc"):
        logger.debug(f"Removing {pyc_file}")
        pyc_file.unlink()
        removed_count += 1
    
    for pyo_file in root_dir.rglob("*.pyo"):
        logger.debug(f"Removing {pyo_file}")
        pyo_file.unlink()
        removed_count += 1
    
    return removed_count


def find_and_remove_temp_files(root_dir: Path) -> int:
    """Remove temporary and backup files"""
    removed_count = 0
    temp_patterns = ["*~", "*.bak", "*.orig", "*.tmp", ".DS_Store", "Thumbs.db"]
    
    for pattern in temp_patterns:
        for temp_file in root_dir.rglob(pattern):
            logger.debug(f"Removing temp file: {temp_file}")
            if temp_file.is_file():
                temp_file.unlink()
                removed_count += 1
            elif temp_file.is_dir():
                shutil.rmtree(temp_file)
                removed_count += 1
    
    return removed_count


def find_duplicate_cache_dirs(root_dir: Path) -> List[Path]:
    """Find duplicate cache directories that can be consolidated"""
    cache_dirs = []
    
    # Find all cxd_cache directories
    for cache_dir in root_dir.rglob("cxd_cache"):
        if cache_dir.is_dir():
            cache_dirs.append(cache_dir)
    
    # Return all but the main one (keep ./cxd_cache)
    main_cache = root_dir / "cxd_cache"
    duplicates = [d for d in cache_dirs if d != main_cache and not str(d).startswith('.archive')]
    
    return duplicates


def validate_config_files(root_dir: Path) -> List[str]:
    """Validate configuration files"""
    issues = []
    
    # Check for required config files
    required_configs = [
        "config/memmimic_config.yaml",
        "config/optimized_tales_config.yaml",
        "src/memmimic/cxd/config/cxd_config.yaml"
    ]
    
    for config_path in required_configs:
        full_path = root_dir / config_path
        if not full_path.exists():
            issues.append(f"Missing required config: {config_path}")
        else:
            try:
                import yaml
                with open(full_path, 'r') as f:
                    yaml.safe_load(f)
            except Exception as e:
                issues.append(f"Invalid YAML in {config_path}: {e}")
    
    return issues


def find_empty_files(root_dir: Path) -> List[Path]:
    """Find empty Python files that might be unnecessary"""
    empty_files = []
    
    for py_file in root_dir.rglob("*.py"):
        if py_file.stat().st_size == 0:
            # Skip __init__.py files as they're often intentionally empty
            if py_file.name != "__init__.py":
                empty_files.append(py_file)
    
    return empty_files


def cleanup_codebase(root_dir: Path, remove_duplicates: bool = False, dry_run: bool = False) -> dict:
    """Perform comprehensive codebase cleanup"""
    results = {
        'python_cache_removed': 0,
        'temp_files_removed': 0,
        'duplicate_caches_found': 0,
        'duplicate_caches_removed': 0,
        'config_issues': [],
        'empty_files': [],
        'warnings': []
    }
    
    logger.info("Starting MemMimic codebase cleanup...")
    
    # Clean Python cache files
    if not dry_run:
        results['python_cache_removed'] = find_and_remove_python_cache(root_dir)
        results['temp_files_removed'] = find_and_remove_temp_files(root_dir)
    else:
        logger.info("DRY RUN: Would remove Python cache and temp files")
    
    # Find duplicate cache directories
    duplicate_caches = find_duplicate_cache_dirs(root_dir)
    results['duplicate_caches_found'] = len(duplicate_caches)
    
    if remove_duplicates and duplicate_caches:
        if not dry_run:
            for dup_cache in duplicate_caches:
                logger.info(f"Removing duplicate cache: {dup_cache}")
                shutil.rmtree(dup_cache)
                results['duplicate_caches_removed'] += 1
        else:
            logger.info(f"DRY RUN: Would remove {len(duplicate_caches)} duplicate cache directories")
    
    # Validate configuration files
    results['config_issues'] = validate_config_files(root_dir)
    
    # Find empty files
    results['empty_files'] = find_empty_files(root_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Clean up MemMimic codebase",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--root", 
        type=str, 
        default=".", 
        help="Root directory of the project"
    )
    parser.add_argument(
        "--remove-duplicates", 
        action="store_true",
        help="Remove duplicate cache directories"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    root_dir = Path(args.root).resolve()
    
    if not root_dir.exists():
        logger.error(f"Root directory does not exist: {root_dir}")
        return 1
    
    # Perform cleanup
    results = cleanup_codebase(
        root_dir, 
        remove_duplicates=args.remove_duplicates,
        dry_run=args.dry_run
    )
    
    # Report results
    logger.info("Cleanup completed!")
    logger.info(f"Python cache files removed: {results['python_cache_removed']}")
    logger.info(f"Temporary files removed: {results['temp_files_removed']}")
    logger.info(f"Duplicate cache directories found: {results['duplicate_caches_found']}")
    logger.info(f"Duplicate cache directories removed: {results['duplicate_caches_removed']}")
    
    if results['config_issues']:
        logger.warning("Configuration issues found:")
        for issue in results['config_issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All configuration files are valid")
    
    if results['empty_files']:
        logger.info(f"Empty Python files found: {len(results['empty_files'])}")
        if args.verbose:
            for empty_file in results['empty_files']:
                logger.info(f"  - {empty_file}")
    
    logger.info("MemMimic codebase cleanup completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())