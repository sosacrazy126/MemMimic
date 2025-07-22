"""
MemMimic Unified API - 11 Core Tools

The main interface combining all MemMimic functionality with comprehensive security validation.
"""

from typing import Any, Dict, List, Optional, Union

from .cxd import create_optimized_classifier
from .memory import ContextualAssistant, create_amms_storage
from .memory.storage.amms_storage import Memory
from .tales import TaleManager
from .errors import (
    MemMimicError, InitializationError, ExternalServiceError, 
    handle_errors, log_errors, with_error_context, get_error_logger
)
from .security import (
    validate_input, sanitize_output, rate_limit, audit_security,
    validate_memory_content, validate_tale_input, validate_query_input
)


class MemMimicAPI:
    """Unified API for MemMimic - 11 core tools with AMMS integration."""

    def __init__(self, db_path: str = "memmimic.db") -> None:
        self.logger = get_error_logger("api")
        
        with with_error_context(
            operation="memmimic_initialization",
            component="api",
            metadata={"db_path": db_path}
        ):
            # Use AMMS storage (post-migration, clean architecture)
            self.memory = create_amms_storage(db_path)
            self.logger.info("Using AMMS storage (post-migration architecture)")
                
            self.assistant = ContextualAssistant("memmimic", db_path)
            self.tales_manager = TaleManager()
            
            # Initialize CXD classifier with proper error handling
            try:
                self.cxd = create_optimized_classifier()
                self.logger.info("CXD classifier initialized successfully")
            except Exception as e:
                # Convert to structured error with context
                error = ExternalServiceError(
                    "Failed to initialize CXD classifier",
                    service_name="cxd_classifier",
                    operation="initialization",
                    context={"db_path": db_path}
                )
                
                # Log the error but continue with graceful fallback
                self.logger.warning(
                    "CXD classifier initialization failed, continuing without classification",
                    extra={"error_id": error.error_id, "original_error": str(e)}
                )
                self.cxd = None

    # === MEMORY CORE (4 tools) ===
    @validate_memory_content(strict=True)
    @rate_limit(max_calls=100, window_seconds=60)
    @audit_security(event_type="memory_storage", log_inputs=False, 
                    sensitive_params=["content"])
    async def remember(self, content: str, memory_type: str = "interaction") -> str:
        """Store + auto-classify in one step with security validation."""
        from .memory import Memory

        memory = Memory(content)

        # Auto-classify with CXD if available
        if self.cxd:
            try:
                classification = self.cxd.classify(content)
                memory.metadata = {"cxd": getattr(classification, "pattern", "unknown"), "type": memory_type}
            except Exception as e:
                # Log classification failure but continue
                memory.metadata = {"type": memory_type}
                self.logger.debug(
                    "CXD classification failed, storing memory without classification",
                    extra={"content_length": len(content), "error": str(e)}
                )

        return await self.memory.store_memory(memory)

    @validate_query_input(strict=True)
    @rate_limit(max_calls=200, window_seconds=60)
    @audit_security(event_type="memory_search")
    @sanitize_output(sanitization_type="memory")
    async def recall_cxd(self, query: str, limit: int = 10) -> List[Memory]:
        """Hybrid semantic search with security validation.""" 
        return await self.memory.search_memories(query, limit)

    @validate_input(validation_type="auto", strict=True)
    @rate_limit(max_calls=50, window_seconds=60)
    @audit_security(event_type="memory_processing", log_inputs=False, 
                    sensitive_params=["input_text"])
    def think_with_memory(self, input_text: str) -> Any:
        """Contextual processing with security validation."""
        return self.assistant.think(input_text)

    async def status(self) -> Dict[str, Any]:
        """System status with AMMS information."""
        memory_count = await self.memory.count_memories()
        all_tales = self.tales_manager.list_tales()
        stats = self.memory.get_stats()

        return {
            "memories": memory_count,
            "tales": len(all_tales),
            "cxd_available": self.cxd is not None,
            "storage_type": stats.get("storage_type", "amms_only"),
            "performance": stats.get("metrics", {}),
            "status": "operational",
        }

    # === TALES SYSTEM (5 tools) ===
    @validate_input(validation_type="auto", strict=False)
    @rate_limit(max_calls=100, window_seconds=60)
    @audit_security(event_type="tale_access")
    @sanitize_output(sanitization_type="json")
    def tales(
        self, 
        query: Optional[str] = None, 
        stats: bool = False, 
        load: bool = False, 
        category: Optional[str] = None, 
        limit: int = 10
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Unified list/search/stats/load with security validation."""
        if load and query:
            return self.tales_manager.load_tale(query, category)
        elif stats:
            return self.tales_manager.get_statistics()
        elif query:
            return self.tales_manager.search_tales(query, category)[:limit]
        else:
            return self.tales_manager.list_tales(category)[:limit]

    @validate_tale_input(strict=True)
    @rate_limit(max_calls=50, window_seconds=60)
    @audit_security(event_type="tale_modification", log_inputs=False,
                    sensitive_params=["content"])
    def save_tale(
        self, 
        name: str, 
        content: str, 
        category: str = "claude/core", 
        tags: Optional[List[str]] = None
    ) -> Any:
        """Auto create/update with security validation."""
        # Check if tale exists to determine create vs update
        existing = self.tales_manager.load_tale(name, category)
        if existing:
            return self.tales_manager.update_tale(name, content, category)
        else:
            return self.tales_manager.create_tale(name, content, category, tags)

    @validate_input(validation_type="auto", strict=True)
    @rate_limit(max_calls=100, window_seconds=60)
    @audit_security(event_type="tale_access")
    @sanitize_output(sanitization_type="json")
    def load_tale(self, name: str, category: Optional[str] = None) -> Optional[Any]:
        """Load specific tale with security validation."""
        return self.tales_manager.load_tale(name, category)

    @validate_input(validation_type="auto", strict=True)
    @rate_limit(max_calls=20, window_seconds=60)
    @audit_security(event_type="tale_deletion", log_inputs=True)
    def delete_tale(
        self, 
        name: str, 
        category: Optional[str] = None, 
        confirm: bool = False
    ) -> Dict[str, str]:
        """Delete tale with security validation (explicit confirmation required)."""
        if not confirm:
            return {"error": "Confirmation required for deletion"}
        return self.tales_manager.delete_tale(name, category)

    async def context_tale(
        self, 
        query: str, 
        style: str = "auto", 
        max_memories: int = 15
    ) -> str:
        """Generate narrative from memories."""
        # Get relevant memories
        relevant_memories = await self.memory.search_memories(query, limit=max_memories)

        if not relevant_memories:
            return f"No memories found for: {query}"

        # Create a narrative from the memories
        narrative_parts = [f"📖 Context Tale: {query}\n"]

        for i, memory in enumerate(relevant_memories, 1):
            memory_type = memory.metadata.get("type", "unknown")
            content = (
                memory.content[:200] + "..."
                if len(memory.content) > 200
                else memory.content
            )
            narrative_parts.append(f"{i}. [{memory_type.upper()}] {content}")

        narrative_parts.append(f"\n✨ Generated from {len(relevant_memories)} memories")

        return "\n".join(narrative_parts)

    # === MEMORY MANAGEMENT (3 tools) ===
    def update_memory_guided(self, memory_id: int) -> Dict[str, Any]:
        """Socratic memory update."""
        # Basic implementation - can be enhanced with Socratic questioning
        all_memories = self.memory.get_all()

        # Find memory by ID
        target_memory = None
        for memory in all_memories:
            if getattr(memory, "id", None) == memory_id:
                target_memory = memory
                break

        if not target_memory:
            return {"error": f"Memory {memory_id} not found"}

        return {
            "memory_id": memory_id,
            "current_content": target_memory.content,
            "guidance": "Memory found. Update process would begin here.",
        }

    def delete_memory_guided(self, memory_id: int, confirm: bool = False) -> Dict[str, str]:
        """Guided memory deletion."""
        if not confirm:
            return {"error": "Confirmation required for deletion"}

        return {"result": f"Memory {memory_id} deletion would be processed here"}

    async def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Pattern analysis."""
        all_memories = await self.memory.list_memories(limit=1000)  # Get recent memories

        if not all_memories:
            return {"error": "No memories to analyze"}

        # Basic pattern analysis
        type_counts = {}
        total_chars = 0

        for memory in all_memories:
            memory_type = memory.metadata.get("type", "unknown")
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
            total_chars += len(memory.content)

        return {
            "total_memories": len(all_memories),
            "type_distribution": type_counts,
            "average_length": total_chars / len(all_memories),
            "analysis": "Basic pattern analysis completed",
        }

    # === COGNITIVE (1 tool) ===
    async def socratic_dialogue(self, topic: str, depth: int = 3) -> Dict[str, Any]:
        """Self-questioning."""
        # Basic implementation using the SocraticEngine
        if hasattr(self.assistant, "socratic_engine"):
            try:
                # Create a mock dialogue
                relevant_memories = await self.memory.search_memories(topic, limit=5)
                dialogue = self.assistant.socratic_engine.conduct_dialogue(
                    topic, f"Initial analysis of: {topic}", relevant_memories
                )

                return {
                    "topic": topic,
                    "depth": depth,
                    "questions": dialogue.questions,
                    "insights": dialogue.insights,
                    "synthesis": dialogue.final_synthesis,
                }
            except Exception as e:
                return {"error": f"Socratic dialogue failed: {e}"}

        return {"error": "Socratic engine not available"}


# Factory function
def create_memmimic(db_path: str = "memmimic.db") -> MemMimicAPI:
    """
    Create MemMimic instance with AMMS storage.
    
    Args:
        db_path: Database file path
    """
    return MemMimicAPI(db_path)

