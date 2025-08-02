#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tale System Manager - Unified Configuration-Driven Interface

This module provides a unified interface for tale management that can switch
between the legacy file-based system and the optimized SQLite system based
on configuration. It ensures backward compatibility while providing dramatic
performance improvements when the optimized system is enabled.

Features:
- Configuration-driven backend selection
- Seamless migration between systems
- Performance monitoring and comparison
- Backward compatibility guarantee
- Gradual transition support
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import os

# Import both tale systems
from .tale_manager import TaleManager as LegacyTaleManager
from .optimized_tale_manager import OptimizedTaleManager
from .optimized_mcp_integration import OptimizedMCPTaleHandler

logger = logging.getLogger(__name__)


@dataclass
class TaleSystemConfig:
    """Configuration for tale system behavior"""
    use_optimized_backend: bool = True
    enable_performance_monitoring: bool = True
    enable_gradual_migration: bool = True
    migration_threshold_ms: float = 100.0  # Switch to optimized if legacy > 100ms
    cache_size: int = 1000
    enable_compression: bool = True
    enable_search_index: bool = True
    legacy_tales_path: str = "tales"
    optimized_db_path: str = "tales_optimized.db"
    performance_log_path: str = "tale_performance.log"


class TaleSystemManager:
    """
    Unified tale system manager that provides configuration-driven backend selection
    with seamless migration and performance monitoring.
    """
    
    def __init__(self, config: TaleSystemConfig = None, base_dir: str = None):
        self.config = config or TaleSystemConfig()
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        
        # Initialize performance tracking
        self.performance_metrics = {
            'legacy_total_time': 0.0,
            'legacy_operation_count': 0,
            'optimized_total_time': 0.0,
            'optimized_operation_count': 0,
            'migration_events': 0,
            'performance_improvement_factor': 0.0
        }
        
        # Initialize both systems
        self._legacy_manager = None
        self._optimized_manager = None
        self._optimized_handler = None
        self._current_backend = "optimized" if self.config.use_optimized_backend else "legacy"
        
        logger.info(f"TaleSystemManager initialized with {self._current_backend} backend")
    
    async def initialize(self):
        """Initialize the tale system components"""
        if self.config.use_optimized_backend or self.config.enable_gradual_migration:
            await self._initialize_optimized_system()
        
        if not self.config.use_optimized_backend or self.config.enable_gradual_migration:
            self._initialize_legacy_system()
        
        logger.info("Tale system components initialized")
    
    def _initialize_legacy_system(self):
        """Initialize the legacy file-based tale system"""
        legacy_path = self.base_dir / self.config.legacy_tales_path
        self._legacy_manager = LegacyTaleManager(str(legacy_path))
        logger.info(f"Legacy tale system initialized at {legacy_path}")
    
    async def _initialize_optimized_system(self):
        """Initialize the optimized SQLite tale system"""
        optimized_db_path = self.base_dir / self.config.optimized_db_path
        
        self._optimized_manager = OptimizedTaleManager(
            db_path=str(optimized_db_path),
            cache_size=self.config.cache_size,
            enable_compression=self.config.enable_compression,
            enable_search_index=self.config.enable_search_index
        )
        
        self._optimized_handler = OptimizedMCPTaleHandler(
            db_path=str(optimized_db_path),
            cache_size=self.config.cache_size
        )
        
        logger.info(f"Optimized tale system initialized at {optimized_db_path}")
    
    def _get_active_backend(self, operation_name: str = None) -> str:
        """Determine which backend to use for the current operation"""
        if not self.config.enable_gradual_migration:
            return self._current_backend
        
        # Adaptive backend selection based on performance
        legacy_avg = (self.performance_metrics['legacy_total_time'] / 
                     max(self.performance_metrics['legacy_operation_count'], 1))
        
        if (legacy_avg > self.config.migration_threshold_ms and 
            self._optimized_manager is not None):
            if self._current_backend == "legacy":
                logger.info(f"Switching to optimized backend due to performance (avg: {legacy_avg:.2f}ms)")
                self._current_backend = "optimized"
                self.performance_metrics['migration_events'] += 1
        
        return self._current_backend
    
    def _log_performance(self, backend: str, operation: str, duration_ms: float):
        """Log performance metrics for comparison"""
        if not self.config.enable_performance_monitoring:
            return
        
        if backend == "legacy":
            self.performance_metrics['legacy_total_time'] += duration_ms
            self.performance_metrics['legacy_operation_count'] += 1
        else:
            self.performance_metrics['optimized_total_time'] += duration_ms
            self.performance_metrics['optimized_operation_count'] += 1
        
        # Update performance improvement factor
        if (self.performance_metrics['legacy_operation_count'] > 0 and 
            self.performance_metrics['optimized_operation_count'] > 0):
            
            legacy_avg = (self.performance_metrics['legacy_total_time'] / 
                         self.performance_metrics['legacy_operation_count'])
            optimized_avg = (self.performance_metrics['optimized_total_time'] / 
                           self.performance_metrics['optimized_operation_count'])
            
            if optimized_avg > 0:
                self.performance_metrics['performance_improvement_factor'] = legacy_avg / optimized_avg
        
        # Log to file if configured
        if self.config.performance_log_path:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'backend': backend,
                'operation': operation,
                'duration_ms': duration_ms,
                'metrics': self.performance_metrics.copy()
            }
            
            log_file = self.base_dir / self.config.performance_log_path
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    # Unified API methods that delegate to the appropriate backend
    
    async def list_tales(self, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List tales using the active backend"""
        backend = self._get_active_backend("list_tales")
        start_time = time.perf_counter()
        
        try:
            if backend == "optimized" and self._optimized_handler:
                result = await self._optimized_handler.mcp_tales_list(
                    category=category, limit=limit
                )
                # Convert optimized format to legacy format for compatibility
                if 'tales' in result:
                    return result['tales']
                return []
            else:
                # Use legacy system
                if not self._legacy_manager:
                    self._initialize_legacy_system()
                return self._legacy_manager.list_tales(category=category)[:limit]
        
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log_performance(backend, "list_tales", duration_ms)
    
    async def search_tales(self, query: str, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search tales using the active backend"""
        backend = self._get_active_backend("search_tales")
        start_time = time.perf_counter()
        
        try:
            if backend == "optimized" and self._optimized_handler:
                result = await self._optimized_handler.mcp_tales_list(
                    query=query, category=category, limit=limit
                )
                # Convert optimized format to legacy format for compatibility
                if 'search_results' in result:
                    return result['search_results']
                return []
            else:
                # Use legacy system
                if not self._legacy_manager:
                    self._initialize_legacy_system()
                return self._legacy_manager.search_tales(
                    query=query, category=category
                )[:limit]
        
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log_performance(backend, "search_tales", duration_ms)
    
    async def load_tale(self, name: str, category: str = None) -> Optional[Dict[str, Any]]:
        """Load a tale using the active backend"""
        backend = self._get_active_backend("load_tale")
        start_time = time.perf_counter()
        
        try:
            if backend == "optimized" and self._optimized_handler:
                result = await self._optimized_handler.mcp_load_tale(
                    name=name, category=category
                )
                if result.get('status') == 'success':
                    return {
                        'name': result['name'],
                        'category': result['category'],
                        'content': result['content'],
                        'tags': result.get('tags', []),
                        'metadata': result.get('metadata', {}),
                        'size': result.get('size', 0)
                    }
                return None
            else:
                # Use legacy system
                if not self._legacy_manager:
                    self._initialize_legacy_system()
                tale = self._legacy_manager.load_tale(name, category)
                if tale:
                    return {
                        'name': tale.name,
                        'category': tale.category,
                        'content': tale.content,
                        'tags': tale.tags,
                        'metadata': tale.metadata,
                        'size': tale.metadata.get('size_chars', len(tale.content))
                    }
                return None
        
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log_performance(backend, "load_tale", duration_ms)
    
    async def save_tale(self, name: str, content: str, category: str = "claude/core", 
                       tags: List[str] = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Save a tale using the active backend"""
        backend = self._get_active_backend("save_tale")
        start_time = time.perf_counter()
        
        try:
            if backend == "optimized" and self._optimized_handler:
                tags_str = ",".join(tags) if tags else ""
                result = await self._optimized_handler.mcp_save_tale(
                    name=name, content=content, category=category, 
                    tags=tags_str, metadata=metadata
                )
                return result
            else:
                # Use legacy system
                if not self._legacy_manager:
                    self._initialize_legacy_system()
                tale = self._legacy_manager.create_tale(
                    name=name, content=content, category=category, tags=tags
                )
                return {
                    'status': 'success',
                    'name': tale.name,
                    'category': tale.category,
                    'size': len(tale.content)
                }
        
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log_performance(backend, "save_tale", duration_ms)
    
    async def delete_tale(self, name: str, category: str = None, confirm: bool = False) -> Dict[str, Any]:
        """Delete a tale using the active backend"""
        backend = self._get_active_backend("delete_tale")
        start_time = time.perf_counter()
        
        try:
            if backend == "optimized" and self._optimized_handler:
                result = await self._optimized_handler.mcp_delete_tale(
                    name=name, category=category, confirm=confirm
                )
                return result
            else:
                # Use legacy system
                if not self._legacy_manager:
                    self._initialize_legacy_system()
                success = self._legacy_manager.delete_tale(name, category)
                return {
                    'status': 'success' if success else 'error',
                    'message': f"Tale '{name}' {'deleted' if success else 'not found'}"
                }
        
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log_performance(backend, "delete_tale", duration_ms)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from both systems"""
        backend = self._get_active_backend("get_statistics")
        start_time = time.perf_counter()
        
        try:
            stats = {'system_manager': True, 'active_backend': backend}
            
            if backend == "optimized" and self._optimized_manager:
                optimized_stats = await self._optimized_manager.get_statistics()
                stats.update(optimized_stats)
                
                if self._optimized_handler:
                    performance_report = await self._optimized_handler.get_performance_report()
                    stats['performance_report'] = performance_report
            else:
                # Use legacy system
                if not self._legacy_manager:
                    self._initialize_legacy_system()
                legacy_stats = self._legacy_manager.get_statistics()
                stats.update(legacy_stats)
            
            # Add system manager performance metrics
            stats['system_manager_metrics'] = self.performance_metrics.copy()
            
            return stats
        
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log_performance(backend, "get_statistics", duration_ms)
    
    async def migrate_to_optimized(self) -> Dict[str, Any]:
        """Migrate all tales from legacy system to optimized system"""
        if not self._legacy_manager:
            return {'status': 'error', 'message': 'Legacy system not initialized'}
        
        if not self._optimized_manager:
            await self._initialize_optimized_system()
        
        migration_report = {
            'status': 'in_progress',
            'tales_migrated': 0,
            'tales_failed': 0,
            'errors': []
        }
        
        try:
            # Get all tales from legacy system
            legacy_tales = self._legacy_manager.list_tales()
            
            for tale_info in legacy_tales:
                try:
                    # Load full tale from legacy system
                    tale = self._legacy_manager.load_tale(
                        tale_info['name'], tale_info['category']
                    )
                    
                    if tale:
                        # Create in optimized system
                        await self._optimized_manager.create_tale(
                            name=tale.name,
                            content=tale.content,
                            category=tale.category,
                            tags=list(tale.tags),
                            metadata=tale.metadata
                        )
                        migration_report['tales_migrated'] += 1
                    
                except Exception as e:
                    migration_report['tales_failed'] += 1
                    migration_report['errors'].append(
                        f"Failed to migrate {tale_info['name']}: {str(e)}"
                    )
            
            migration_report['status'] = 'completed'
            
            # Switch to optimized backend after successful migration
            if migration_report['tales_failed'] == 0:
                self._current_backend = "optimized"
                self.config.use_optimized_backend = True
                logger.info("Migration completed successfully, switched to optimized backend")
            
            return migration_report
        
        except Exception as e:
            migration_report['status'] = 'failed'
            migration_report['error'] = str(e)
            return migration_report
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get detailed performance comparison between backends"""
        metrics = self.performance_metrics.copy()
        
        if metrics['legacy_operation_count'] > 0:
            metrics['legacy_avg_ms'] = (
                metrics['legacy_total_time'] / metrics['legacy_operation_count']
            )
        else:
            metrics['legacy_avg_ms'] = 0
        
        if metrics['optimized_operation_count'] > 0:
            metrics['optimized_avg_ms'] = (
                metrics['optimized_total_time'] / metrics['optimized_operation_count']
            )
        else:
            metrics['optimized_avg_ms'] = 0
        
        return {
            'performance_metrics': metrics,
            'current_backend': self._current_backend,
            'config': {
                'use_optimized_backend': self.config.use_optimized_backend,
                'enable_gradual_migration': self.config.enable_gradual_migration,
                'migration_threshold_ms': self.config.migration_threshold_ms
            },
            'recommendation': self._get_performance_recommendation()
        }
    
    def _get_performance_recommendation(self) -> str:
        """Generate performance-based recommendations"""
        improvement_factor = self.performance_metrics.get('performance_improvement_factor', 0)
        
        if improvement_factor > 10:
            return "🚀 Optimized system shows 10x+ improvement. Highly recommended!"
        elif improvement_factor > 5:
            return "⚡ Optimized system shows 5x+ improvement. Recommended for adoption."
        elif improvement_factor > 2:
            return "📈 Optimized system shows 2x+ improvement. Consider migration."
        elif improvement_factor > 1:
            return "✅ Optimized system shows modest improvement."
        else:
            return "⚠️  Need more data to compare performance."


# Global instance for easy access
_global_tale_system = None

def get_tale_system_manager(config: TaleSystemConfig = None) -> TaleSystemManager:
    """Get or create the global tale system manager"""
    global _global_tale_system
    if _global_tale_system is None:
        _global_tale_system = TaleSystemManager(config)
    return _global_tale_system

async def initialize_tale_system(config: TaleSystemConfig = None):
    """Initialize the global tale system"""
    manager = get_tale_system_manager(config)
    await manager.initialize()
    return manager