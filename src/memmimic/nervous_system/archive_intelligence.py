"""
ArchiveIntelligence - Legacy Code Evolution System

Transforms patterns from the .archive directory into active intelligence components
for the nervous system. Implements automated legacy code evolution, cleanup reflexes,
and unused code detection based on archived migration patterns.

This component bridges the gap between historical code patterns and active system
intelligence, enabling continuous evolution and optimization.
"""

import asyncio
import os
import shutil
import json
import ast
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..errors import MemMimicError, with_error_context, get_error_logger
from ..memory.storage.amms_storage import create_amms_storage


@dataclass
class MigrationPattern:
    """Represents a migration pattern extracted from archive"""
    name: str
    source_file: str
    pattern_type: str  # 'database', 'cleanup', 'analysis', 'organization'
    description: str
    implementation_hints: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    extracted_functions: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvolutionMetrics:
    """Metrics for tracking code evolution and optimization"""
    patterns_extracted: int = 0
    patterns_applied: int = 0
    cleanup_operations: int = 0
    unused_components_detected: int = 0
    database_migrations: int = 0
    performance_improvements: float = 0.0
    last_evolution_timestamp: float = field(default_factory=time.time)


class ArchiveIntelligence:
    """
    Archive Intelligence System for Legacy Code Evolution
    
    Extracts patterns from .archive directory and transforms them into active
    intelligence components for continuous system evolution and optimization.
    """
    
    def __init__(self, archive_path: str = ".archive", db_path: str = None):
        self.archive_path = Path(archive_path)
        self.db_path = db_path or "./src/memmimic/mcp/memmimic.db"
        self.logger = get_error_logger("archive_intelligence")
        
        # Pattern storage
        self.migration_patterns: Dict[str, MigrationPattern] = {}
        self.evolution_metrics = EvolutionMetrics()
        
        # Active intelligence components
        self._storage = None
        self._initialized = False
        
        # Pattern extraction cache
        self._pattern_cache: Dict[str, Any] = {}
        self._cache_ttl = 3600  # 1 hour
        
    async def initialize(self) -> None:
        """Initialize archive intelligence system"""
        if self._initialized:
            return
            
        try:
            # Initialize storage
            self._storage = create_amms_storage(self.db_path)
            
            # Extract patterns from archive
            await self._extract_archive_patterns()
            
            # Initialize active intelligence components
            await self._initialize_intelligence_components()
            
            self._initialized = True
            self.logger.info("Archive intelligence system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize archive intelligence: {e}")
            raise MemMimicError(f"Archive intelligence initialization failed: {e}")
    
    async def _extract_archive_patterns(self) -> None:
        """Extract migration patterns from archive directory"""
        if not self.archive_path.exists():
            self.logger.warning(f"Archive directory not found: {self.archive_path}")
            return
            
        with with_error_context(
            operation="extract_archive_patterns",
            component="archive_intelligence"
        ):
            # Extract database migration patterns
            await self._extract_migration_patterns()
            
            # Extract cleanup patterns
            await self._extract_cleanup_patterns()
            
            # Extract analysis patterns
            await self._extract_analysis_patterns()
            
            self.evolution_metrics.patterns_extracted = len(self.migration_patterns)
            self.logger.info(f"Extracted {self.evolution_metrics.patterns_extracted} patterns from archive")
    
    async def _extract_migration_patterns(self) -> None:
        """Extract database migration patterns from migrate_to_amms.py"""
        migrate_file = self.archive_path / "migrate_to_amms.py"
        if not migrate_file.exists():
            return
            
        try:
            with open(migrate_file, 'r') as f:
                content = f.read()
            
            # Parse AST to extract functions
            tree = ast.parse(content)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_start = node.lineno
                    func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 10
                    func_content = '\n'.join(content.split('\n')[func_start-1:func_end])
                    functions[node.name] = func_content
            
            # Create migration pattern
            pattern = MigrationPattern(
                name="database_migration",
                source_file="migrate_to_amms.py",
                pattern_type="database",
                description="Database migration and compatibility checking patterns",
                implementation_hints=[
                    "Create backup before migration",
                    "Check database compatibility",
                    "Test functionality after migration",
                    "Provide rollback mechanism"
                ],
                dependencies=["memmimic.memory", "memmimic.memory.active_schema"],
                success_criteria=[
                    "Migration completes without errors",
                    "All legacy memories preserved",
                    "Enhanced schema active",
                    "Functionality tests pass"
                ],
                extracted_functions=functions
            )
            
            self.migration_patterns["database_migration"] = pattern
            
        except Exception as e:
            self.logger.error(f"Failed to extract migration patterns: {e}")
    
    async def _extract_cleanup_patterns(self) -> None:
        """Extract cleanup patterns from organize_root.py"""
        cleanup_file = self.archive_path / "organize_root.py"
        if not cleanup_file.exists():
            return
            
        try:
            with open(cleanup_file, 'r') as f:
                content = f.read()
            
            # Extract cleanup logic
            tree = ast.parse(content)
            functions = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_start = node.lineno
                    func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 10
                    func_content = '\n'.join(content.split('\n')[func_start-1:func_end])
                    functions[node.name] = func_content
            
            # Create cleanup pattern
            pattern = MigrationPattern(
                name="cleanup_reflexes",
                source_file="organize_root.py",
                pattern_type="cleanup",
                description="Automated cleanup and organization reflexes",
                implementation_hints=[
                    "Archive temporary files safely",
                    "Maintain organized directory structure",
                    "Generate cleanup reports",
                    "Preserve important artifacts"
                ],
                dependencies=["shutil", "pathlib", "json"],
                success_criteria=[
                    "Temporary files archived",
                    "Directory structure organized",
                    "Cleanup report generated",
                    "No important files lost"
                ],
                extracted_functions=functions
            )
            
            self.migration_patterns["cleanup_reflexes"] = pattern
            
        except Exception as e:
            self.logger.error(f"Failed to extract cleanup patterns: {e}")
    
    async def _extract_analysis_patterns(self) -> None:
        """Extract analysis patterns from check_unused_imports.py"""
        analysis_file = self.archive_path / "check_unused_imports.py"
        if not analysis_file.exists():
            return
            
        try:
            with open(analysis_file, 'r') as f:
                content = f.read()
            
            # Extract analysis logic
            tree = ast.parse(content)
            functions = {}
            classes = {}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_start = node.lineno
                    func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 10
                    func_content = '\n'.join(content.split('\n')[func_start-1:func_end])
                    functions[node.name] = func_content
                elif isinstance(node, ast.ClassDef):
                    class_start = node.lineno
                    class_end = node.end_lineno if hasattr(node, 'end_lineno') else class_start + 20
                    class_content = '\n'.join(content.split('\n')[class_start-1:class_end])
                    classes[node.name] = class_content
            
            # Create analysis pattern
            pattern = MigrationPattern(
                name="unused_code_detection",
                source_file="check_unused_imports.py",
                pattern_type="analysis",
                description="Intelligent unused code detection and evolution",
                implementation_hints=[
                    "Use AST analysis for accurate detection",
                    "Check multiple usage patterns",
                    "Be conservative with removal decisions",
                    "Provide detailed analysis reports"
                ],
                dependencies=["ast", "re", "pathlib"],
                success_criteria=[
                    "Accurate unused code detection",
                    "Minimal false positives",
                    "Detailed analysis reports",
                    "Safe evolution recommendations"
                ],
                extracted_functions={**functions, **classes}
            )
            
            self.migration_patterns["unused_code_detection"] = pattern
            
        except Exception as e:
            self.logger.error(f"Failed to extract analysis patterns: {e}")
    
    async def _initialize_intelligence_components(self) -> None:
        """Initialize active intelligence components based on extracted patterns"""
        # This will be expanded in subsequent implementations
        pass
    
    async def apply_migration_pattern(self, pattern_name: str, target_path: str, **kwargs) -> Dict[str, Any]:
        """Apply a migration pattern to a target path"""
        if not self._initialized:
            await self.initialize()
            
        if pattern_name not in self.migration_patterns:
            raise MemMimicError(f"Migration pattern not found: {pattern_name}")
        
        pattern = self.migration_patterns[pattern_name]
        
        with with_error_context(
            operation="apply_migration_pattern",
            component="archive_intelligence",
            metadata={"pattern": pattern_name, "target": target_path}
        ):
            try:
                result = await self._execute_pattern(pattern, target_path, **kwargs)
                self.evolution_metrics.patterns_applied += 1
                return result
                
            except Exception as e:
                self.logger.error(f"Failed to apply pattern {pattern_name}: {e}")
                raise
    
    async def _execute_pattern(self, pattern: MigrationPattern, target_path: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific migration pattern"""
        # Implementation will be pattern-specific
        # This is a framework for pattern execution
        
        execution_result = {
            "pattern_name": pattern.name,
            "pattern_type": pattern.pattern_type,
            "target_path": target_path,
            "execution_time": time.time(),
            "success": False,
            "details": {},
            "metrics": {}
        }
        
        try:
            if pattern.pattern_type == "database":
                execution_result.update(await self._execute_database_pattern(pattern, target_path, **kwargs))
            elif pattern.pattern_type == "cleanup":
                execution_result.update(await self._execute_cleanup_pattern(pattern, target_path, **kwargs))
            elif pattern.pattern_type == "analysis":
                execution_result.update(await self._execute_analysis_pattern(pattern, target_path, **kwargs))
            
            execution_result["success"] = True
            
        except Exception as e:
            execution_result["error"] = str(e)
            raise
        
        return execution_result
    
    async def _execute_database_pattern(self, pattern: MigrationPattern, target_path: str, **kwargs) -> Dict[str, Any]:
        """Execute database migration pattern"""
        # Placeholder for database migration logic
        return {"type": "database_migration", "status": "executed"}
    
    async def _execute_cleanup_pattern(self, pattern: MigrationPattern, target_path: str, **kwargs) -> Dict[str, Any]:
        """Execute cleanup pattern"""
        # Placeholder for cleanup logic
        return {"type": "cleanup", "status": "executed"}
    
    async def _execute_analysis_pattern(self, pattern: MigrationPattern, target_path: str, **kwargs) -> Dict[str, Any]:
        """Execute analysis pattern"""
        # Placeholder for analysis logic
        return {"type": "analysis", "status": "executed"}
    
    def get_evolution_metrics(self) -> EvolutionMetrics:
        """Get current evolution metrics"""
        return self.evolution_metrics
    
    def get_available_patterns(self) -> List[str]:
        """Get list of available migration patterns"""
        return list(self.migration_patterns.keys())
    
    def get_pattern_details(self, pattern_name: str) -> Optional[MigrationPattern]:
        """Get details for a specific pattern"""
        return self.migration_patterns.get(pattern_name)
