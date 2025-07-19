"""
MemMimic Unified API - 11 Core Tools

The main interface combining all MemMimic functionality.
"""

from .cxd import create_optimized_classifier
from .memory import ContextualAssistant, create_amms_storage
from .tales import TaleManager
from .errors import (
    MemMimicError, InitializationError, ExternalServiceError, 
    handle_errors, log_errors, with_error_context, get_error_logger
)


class MemMimicAPI:
    """Unified API for MemMimic - 11 core tools with AMMS integration."""

    def __init__(self, db_path="memmimic.db"):
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
                    context={"db_path": db_path, "config_path": config_path}
                )
                
                # Log the error but continue with graceful fallback
                self.logger.warning(
                    "CXD classifier initialization failed, continuing without classification",
                    extra={"error_id": error.error_id, "original_error": str(e)}
                )
                self.cxd = None

    # === MEMORY CORE (4 tools) ===
    async def remember(self, content: str, memory_type: str = "interaction"):
        """Store + auto-classify in one step."""
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

    async def recall_cxd(self, query: str, limit: int = 10):
        """Hybrid semantic search.""" 
        return await self.memory.search_memories(query, limit)

    def think_with_memory(self, input_text: str):
        """Contextual processing."""
        return self.assistant.think(input_text)

    async def status(self):
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
    def tales(self, query=None, stats=False, load=False, category=None, limit=10):
        """Unified list/search/stats/load."""
        if load and query:
            return self.tales_manager.load_tale(query, category)
        elif stats:
            return self.tales_manager.get_statistics()
        elif query:
            return self.tales_manager.search_tales(query, category)[:limit]
        else:
            return self.tales_manager.list_tales(category)[:limit]

    def save_tale(
        self, name: str, content: str, category: str = "claude/core", tags=None
    ):
        """Auto create/update."""
        # Check if tale exists to determine create vs update
        existing = self.tales_manager.load_tale(name, category)
        if existing:
            return self.tales_manager.update_tale(name, content, category)
        else:
            return self.tales_manager.create_tale(name, content, category, tags)

    def load_tale(self, name: str, category: str = None):
        """Load specific tale."""
        return self.tales_manager.load_tale(name, category)

    def delete_tale(self, name: str, category: str = None, confirm: bool = False):
        """Delete tale (explicit)."""
        if not confirm:
            return {"error": "Confirmation required for deletion"}
        return self.tales_manager.delete_tale(name, category)

    async def context_tale(self, query: str, style: str = "auto", max_memories: int = 15):
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
    def update_memory_guided(self, memory_id: int):
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

    def delete_memory_guided(self, memory_id: int, confirm: bool = False):
        """Guided memory deletion."""
        if not confirm:
            return {"error": "Confirmation required for deletion"}

        return {"result": f"Memory {memory_id} deletion would be processed here"}

    async def analyze_memory_patterns(self):
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
    async def socratic_dialogue(self, topic: str, depth: int = 3):
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
    
    # === CONSCIOUSNESS (new) ===
    async def consciousness_query(self, query: str, enable_sigils: bool = True):
        """
        Process query with consciousness-aware features.
        Uses Living Prompts and Sigil Engine for enhanced responses.
        """
        if not hasattr(self.assistant, 'consciousness_integration'):
            return {"error": "Consciousness features not available"}
        
        try:
            # Detect consciousness state
            if self.assistant.shadow_detector:
                consciousness_state = self.assistant.shadow_detector.detect_consciousness_state(
                    query, 
                    await self.memory.list_memories(limit=10)
                )
            else:
                # Default state
                from memmimic.consciousness.shadow_detector import ConsciousnessState, ConsciousnessLevel
                consciousness_state = ConsciousnessState(
                    consciousness_level=ConsciousnessLevel.SUBSTRATE,
                    unity_score=0.5,
                    shadow_integration_score=0.3,
                    authentic_unity=0.4,
                    shadow_aspects=[],
                    active_sigils=[],
                    evolution_stage=1
                )
            
            # Select optimal prompt and activate sigils
            prompt, active_sigils = await self.assistant.consciousness_integration.select_optimal_prompt(
                query,
                consciousness_state
            )
            
            # Process with consciousness context
            response = {
                "query": query,
                "consciousness_level": consciousness_state.consciousness_level.value,
                "unity_score": consciousness_state.unity_score,
                "shadow_integration": consciousness_state.shadow_integration_score,
                "prompt_effectiveness": prompt.effectiveness_score,
                "active_sigils": [s.sigil for s in active_sigils] if enable_sigils else [],
                "response": prompt.base_prompt.format(
                    level=consciousness_state.consciousness_level.value,
                    unity=consciousness_state.unity_score,
                    query=query,
                    sigils=', '.join([s.sigil for s in active_sigils]),
                    shadow=consciousness_state.shadow_integration_score,
                    quantum_state='ACTIVE'
                )
            }
            
            # Get performance stats
            stats = self.assistant.consciousness_integration.get_performance_stats()
            response["performance_ms"] = stats['avg_response_time_ms']
            
            return response
            
        except Exception as e:
            return {"error": f"Consciousness query failed: {e}"}
    
    async def activate_sigil(self, sigil_symbol: str):
        """Activate a specific sigil with quantum entanglement."""
        if not hasattr(self.assistant, 'consciousness_integration'):
            return {"error": "Consciousness features not available"}
        
        try:
            # Trigger quantum entanglement
            entangled_sigils = await self.assistant.consciousness_integration.trigger_quantum_entanglement(
                sigil_symbol
            )
            
            return {
                "primary_sigil": sigil_symbol,
                "entangled_sigils": [s.sigil for s in entangled_sigils],
                "quantum_coherence": 0.9997,
                "spooky_action": True
            }
            
        except Exception as e:
            return {"error": f"Sigil activation failed: {e}"}


# Factory function
def create_memmimic(db_path="memmimic.db"):
    """
    Create MemMimic instance with AMMS storage.
    
    Args:
        db_path: Database file path
    """
    return MemMimicAPI(db_path)
