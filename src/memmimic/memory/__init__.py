"""
MemMimic Memory System

Provides unified memory storage with multiple backend support.
"""

from .memory import MemoryStore, Memory
from .assistant import ContextualAssistant
from .socratic import SocraticEngine, SocraticDialogue
from .unified_store import UnifiedMemoryStore

# For backward compatibility, create a factory function
def create_memory_store(unified: bool = False, **kwargs):
    """
    Create a memory store instance.
    
    Args:
        unified: If True, create UnifiedMemoryStore, otherwise MemoryStore
        **kwargs: Arguments passed to the store constructor
        
    Returns:
        MemoryStore or UnifiedMemoryStore instance
    """
    if unified:
        return UnifiedMemoryStore(kwargs)
    else:
        db_path = kwargs.get('db_path', 'memories.db')
        return MemoryStore(db_path)

__all__ = [
    "MemoryStore", 
    "Memory", 
    "ContextualAssistant", 
    "UnifiedMemoryStore", 
    "SocraticEngine", 
    "SocraticDialogue",
    "create_memory_store"
]
