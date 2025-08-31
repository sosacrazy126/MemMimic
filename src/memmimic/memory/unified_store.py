"""
UnifiedMemoryStore - Orchestrates multiple memory backends.

This coordinator manages different storage backends and routes operations
to the most appropriate backend based on capabilities and configuration.
"""

import logging
import threading
from typing import List, Optional, Dict, Any, Set
from pathlib import Path

from .backends.base import MemoryBackend
from .backends.sqlite_backend import SQLiteBackend
from .backends.filesystem_backend import FileSystemBackend
from .memory import Memory

logger = logging.getLogger(__name__)


class UnifiedMemoryStore:
    """
    Unified interface for memory operations across multiple backends.
    
    Orchestrates different storage backends to leverage their unique strengths:
    - SQLite for structured queries and metadata
    - FileSystem for human-readable backup (future)
    - FAISS for vector search (future)
    - Redis for caching (future)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize UnifiedMemoryStore with configuration.
        
        Args:
            config: Configuration dictionary with backend settings
                Example:
                {
                    'primary_backend': 'sqlite',
                    'backends': {
                        'sqlite': {'db_path': 'memories.db'},
                        'filesystem': {'base_path': '~/.memmimic/memories'},
                        'faiss': {'index_path': 'vectors.index'}
                    },
                    'features': {
                        'auto_backup': True,
                        'vector_search': False,
                        'caching': False
                    }
                }
        """
        self.config = config or self._default_config()
        self.backends: Dict[str, MemoryBackend] = {}
        self.primary_backend: Optional[MemoryBackend] = None
        self.lock = threading.Lock()  # Thread safety for orchestrator
        
        self._initialize_backends()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'primary_backend': 'sqlite',
            'backends': {
                'sqlite': {'db_path': 'memories.db'}
            },
            'features': {
                'auto_backup': False,
                'vector_search': False,
                'caching': False
            }
        }
    
    def _initialize_backends(self) -> None:
        """Initialize configured backends."""
        backend_configs = self.config.get('backends', {})
        
        # Initialize SQLite backend (always present as fallback)
        if 'sqlite' in backend_configs:
            sqlite_config = backend_configs['sqlite']
            self.backends['sqlite'] = SQLiteBackend(
                db_path=sqlite_config.get('db_path', 'memories.db')
            )
        
        # Initialize FileSystem backend for human-readable storage
        if 'filesystem' in backend_configs:
            fs_config = backend_configs['filesystem']
            self.backends['filesystem'] = FileSystemBackend(
                base_path=fs_config.get('base_path', '~/.memmimic/memories')
            )
            logger.info("FileSystem backend initialized for human-readable storage")
        
        # Future: Initialize other backends
        # if 'faiss' in backend_configs and self.config['features']['vector_search']:
        #     self.backends['faiss'] = FAISSBackend(...)
        
        # Set primary backend
        primary_name = self.config.get('primary_backend', 'sqlite')
        if primary_name in self.backends:
            self.primary_backend = self.backends[primary_name]
        else:
            # Fallback to SQLite if configured primary not available
            self.primary_backend = self.backends.get('sqlite')
            if self.primary_backend:
                logger.warning(f"Primary backend '{primary_name}' not available, using SQLite")
        
        if not self.primary_backend:
            raise RuntimeError("No primary backend available")
    
    def add(self, memory: Memory) -> int:
        """
        Add a memory using the primary backend.
        
        Args:
            memory: Memory object to store
            
        Returns:
            ID of the created memory
        """
        with self.lock:  # Thread safety
            # Extract only additional metadata (not content, type, confidence)
            metadata = {}
            if hasattr(memory, '_embedding') and memory._embedding is not None:
                metadata['has_embedding'] = True
                metadata['embedding_model'] = memory._embedding_model
            if hasattr(memory, 'tags'):
                metadata['tags'] = memory.tags
            
            memory_id = self.primary_backend.add(
                content=memory.content,
                memory_type=memory.type,
                confidence=memory.confidence,
                metadata=metadata if metadata else None
            )
            
            # Auto-backup to filesystem if enabled
            if self.config.get('features', {}).get('auto_backup') and 'filesystem' in self.backends:
                if self.primary_backend.backend_type != 'filesystem':  # Don't double-store
                    try:
                        self.backends['filesystem'].add(
                            content=memory.content,
                            memory_type=memory.type,
                            confidence=memory.confidence,
                            metadata=metadata
                        )
                        logger.debug(f"Memory {memory_id} backed up to filesystem")
                    except Exception as e:
                        logger.warning(f"Failed to backup memory to filesystem: {e}")
            
            return memory_id
    
    def search(self, query: str, limit: int = 5, backend_hint: Optional[str] = None) -> List[Memory]:
        """
        Search for memories, potentially across multiple backends.
        
        Args:
            query: Search query
            limit: Maximum results
            backend_hint: Optional hint for which backend to use
            
        Returns:
            List of Memory objects
        """
        # If specific backend requested and available, use it
        if backend_hint and backend_hint in self.backends:
            return self.backends[backend_hint].search(query, limit)
        
        # For now, use primary backend
        # Future: Could combine results from multiple backends
        results = self.primary_backend.search(query, limit)
        
        # Future enhancement: If vector search is enabled, also search FAISS
        # if self.config['features']['vector_search'] and 'faiss' in self.backends:
        #     vector_results = self.backends['faiss'].search(query, limit)
        #     results = self._merge_results(results, vector_results, limit)
        
        return results
    
    def get(self, memory_id: int) -> Optional[Memory]:
        """Get a specific memory by ID."""
        # Try primary backend first
        memory = self.primary_backend.get(memory_id)
        
        # If not found and we have other backends, try them
        if not memory:
            for name, backend in self.backends.items():
                if backend != self.primary_backend:
                    memory = backend.get(memory_id)
                    if memory:
                        break
        
        return memory
    
    def update(self, memory_id: int, content: Optional[str] = None,
               confidence: Optional[float] = None) -> bool:
        """Update a memory."""
        success = self.primary_backend.update(memory_id, content, confidence)
        
        # Future: Update in other backends if present
        # for name, backend in self.backends.items():
        #     if backend != self.primary_backend:
        #         backend.update(memory_id, content, confidence)
        
        return success
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory from all backends."""
        success = self.primary_backend.delete(memory_id)
        
        # Delete from all other backends too
        for name, backend in self.backends.items():
            if backend != self.primary_backend:
                try:
                    backend.delete(memory_id)
                except Exception as e:
                    logger.warning(f"Failed to delete from {name}: {e}")
        
        return success
    
    def get_all(self) -> List[Memory]:
        """Get all memories from primary backend."""
        return self.primary_backend.get_all()
    
    def export(self, format: str = "json", backend: Optional[str] = None) -> str:
        """
        Export memories from specified or primary backend.
        
        Args:
            format: Export format (json, markdown)
            backend: Specific backend to export from
            
        Returns:
            Exported data as string
        """
        export_backend = self.backends.get(backend, self.primary_backend)
        return export_backend.export(format)
    
    def import_data(self, data: str, format: str = "json") -> int:
        """Import memories into primary backend."""
        return self.primary_backend.import_data(data, format)
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about available backends."""
        info = {
            'primary': self.primary_backend.backend_type,
            'available_backends': list(self.backends.keys()),
            'capabilities': {}
        }
        
        for name, backend in self.backends.items():
            info['capabilities'][name] = backend.capabilities
        
        return info
    
    def switch_primary(self, backend_name: str) -> bool:
        """
        Switch the primary backend.
        
        Args:
            backend_name: Name of backend to make primary
            
        Returns:
            True if successful
        """
        with self.lock:  # Thread safety for backend switching
            if backend_name in self.backends:
                self.primary_backend = self.backends[backend_name]
                self.config['primary_backend'] = backend_name
                logger.info(f"Switched primary backend to {backend_name}")
                return True
            return False
    
    def _merge_results(self, results1: List[Memory], results2: List[Memory],
                      limit: int) -> List[Memory]:
        """
        Merge and deduplicate results from multiple backends.
        
        Future enhancement for combining results from different backends.
        """
        seen_ids: Set[int] = set()
        merged = []
        
        for memory in results1 + results2:
            if memory.id not in seen_ids:
                seen_ids.add(memory.id)
                merged.append(memory)
                if len(merged) >= limit:
                    break
        
        return merged
    
    def close(self) -> None:
        """Close all backend connections."""
        for backend in self.backends.values():
            try:
                backend.close()
            except Exception as e:
                logger.warning(f"Error closing backend: {e}")
    
    # Compatibility methods for existing code
    def add_memory(self, memory: Memory) -> int:
        """Compatibility wrapper for add()."""
        return self.add(memory)