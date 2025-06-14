"""
MemMimic Memory System
"""

from .memory import MemoryStore, Memory
from .assistant import ContextualAssistant
from .socratic import SocraticEngine, SocraticDialogue

# TODO: Implement UnifiedMemoryStore that combines both
UnifiedMemoryStore = MemoryStore  # Temporary alias

__all__ = ["MemoryStore", "Memory", "ContextualAssistant", "UnifiedMemoryStore", "SocraticEngine", "SocraticDialogue"]
