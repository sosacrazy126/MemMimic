"""
Abstract base class for memory storage backends.

Each backend must implement these core methods to work with UnifiedMemoryStore.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime


class MemoryBackend(ABC):
    """Abstract interface for memory storage backends."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend storage (create tables, directories, etc.)."""
        pass
    
    @abstractmethod
    def add(self, content: str, memory_type: str = "interaction", 
            confidence: float = 0.8, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a new memory to the backend.
        
        Args:
            content: The memory content
            memory_type: Type of memory (interaction, milestone, socratic, etc.)
            confidence: Confidence score (0.0 to 1.0)
            metadata: Optional metadata dictionary
            
        Returns:
            The ID of the created memory
        """
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Search for memories matching the query.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            filters: Optional filters (type, confidence range, date range, etc.)
            
        Returns:
            List of Memory objects matching the query
        """
        pass
    
    @abstractmethod
    def get(self, memory_id: int) -> Optional[Any]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: The memory ID
            
        Returns:
            Memory object or None if not found
        """
        pass
    
    @abstractmethod
    def update(self, memory_id: int, content: Optional[str] = None, 
               confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: The memory ID to update
            content: New content (if provided)
            confidence: New confidence score (if provided)
            metadata: New or updated metadata (if provided)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, memory_id: int) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: The memory ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Get all memories, optionally filtered.
        
        Args:
            filters: Optional filters (type, confidence range, date range, etc.)
            
        Returns:
            List of all Memory objects matching filters
        """
        pass
    
    @abstractmethod
    def export(self, format: str = "json") -> str:
        """
        Export all memories in the specified format.
        
        Args:
            format: Export format (json, markdown, yaml)
            
        Returns:
            Exported data as string
        """
        pass
    
    @abstractmethod
    def import_data(self, data: str, format: str = "json") -> int:
        """
        Import memories from the specified format.
        
        Args:
            data: Data to import as string
            format: Import format (json, markdown, yaml)
            
        Returns:
            Number of memories imported
        """
        pass
    
    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, bool]:
        """
        Return backend capabilities.
        
        Example:
            {
                'semantic_search': True,
                'exact_search': True,
                'regex_search': False,
                'vector_operations': True,
                'transactions': True,
                'concurrent_access': True
            }
        """
        pass
    
    def close(self) -> None:
        """Clean up resources (close connections, files, etc.)."""
        pass