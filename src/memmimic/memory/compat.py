"""
Compatibility layer for transitioning from MemoryStore to UnifiedMemoryStore.

This module provides a wrapper that makes UnifiedMemoryStore work exactly
like the original MemoryStore for backward compatibility.
"""

from typing import List, Optional
from .unified_store import UnifiedMemoryStore
from .memory import Memory


class MemoryStoreCompat:
    """
    Compatibility wrapper that makes UnifiedMemoryStore behave like MemoryStore.
    
    This allows existing code to work without modification while using the
    new unified architecture under the hood.
    """
    
    def __init__(self, db_path: str = "memories.db", use_unified: bool = True):
        """
        Initialize compatibility wrapper.
        
        Args:
            db_path: Path to SQLite database (for compatibility)
            use_unified: If True, use UnifiedMemoryStore, else use legacy MemoryStore
        """
        self.db_path = db_path
        
        if use_unified:
            # Configure UnifiedMemoryStore to use SQLite backend
            config = {
                'primary_backend': 'sqlite',
                'backends': {
                    'sqlite': {'db_path': db_path}
                },
                'features': {
                    'auto_backup': False,
                    'vector_search': False,
                    'caching': False
                }
            }
            self.store = UnifiedMemoryStore(config)
        else:
            # Fall back to legacy MemoryStore if needed
            from .memory import MemoryStore
            self.store = MemoryStore(db_path)
    
    # Delegate all methods to the underlying store
    
    def add(self, memory: Memory) -> int:
        """Add a memory (compatible with both stores)."""
        if isinstance(self.store, UnifiedMemoryStore):
            return self.store.add(memory)
        else:
            return self.store.add(memory)
    
    def search(self, query: str, limit: int = 5) -> List[Memory]:
        """Search for memories."""
        return self.store.search(query, limit)
    
    def get_all(self) -> List[Memory]:
        """Get all memories."""
        return self.store.get_all()
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory."""
        return self.store.delete(memory_id)
    
    def update(self, memory_id: int, content: str = None, confidence: float = None) -> bool:
        """Update a memory."""
        return self.store.update(memory_id, content, confidence)
    
    def _init_db(self):
        """Legacy method for compatibility."""
        pass  # Already initialized in constructor
    
    def _load_semantic_expansions(self):
        """Legacy method for compatibility."""
        pass  # Handled by backend
    
    def _row_to_memory(self, row):
        """Legacy method for compatibility."""
        # This shouldn't be called in new code
        return Memory(
            content=row.get('content', ''),
            memory_type=row.get('type', 'interaction'),
            confidence=row.get('confidence', 0.8),
            id=row.get('id')
        )
    
    # Properties for compatibility
    @property
    def conn(self):
        """Legacy property for SQLite connection."""
        if hasattr(self.store, 'primary_backend'):
            backend = self.store.primary_backend
            if hasattr(backend, 'conn'):
                return backend.conn
        return None
    
    @property
    def semantic_expansions(self):
        """Legacy property for semantic expansions."""
        if hasattr(self.store, 'primary_backend'):
            backend = self.store.primary_backend
            if hasattr(backend, 'semantic_expansions'):
                return backend.semantic_expansions
        return {}


def create_compatible_store(db_path: str = "memories.db", 
                           use_unified: bool = True) -> MemoryStoreCompat:
    """
    Factory function to create a compatible memory store.
    
    Args:
        db_path: Database path for SQLite
        use_unified: Whether to use UnifiedMemoryStore (True) or legacy (False)
        
    Returns:
        MemoryStoreCompat instance that works with existing code
    """
    return MemoryStoreCompat(db_path, use_unified)