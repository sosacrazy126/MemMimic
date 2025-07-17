"""
MemMimic Unified API - 11 Core Tools

The main interface combining all MemMimic functionality.
"""

from .memory import UnifiedMemoryStore, ContextualAssistant
from .tales import TaleManager
from .cxd import create_optimized_classifier

class MemMimicAPI:
    """Unified API for MemMimic - 11 core tools with AMMS integration."""
    
    def __init__(self, db_path="memmimic.db", config_path=None):
        self.memory = UnifiedMemoryStore(db_path, config_path)
        self.assistant = ContextualAssistant("memmimic", db_path)
        self.tales_manager = TaleManager()
        try:
            self.cxd = create_optimized_classifier()
        except Exception:
            self.cxd = None  # Graceful fallback
    
    # === MEMORY CORE (4 tools) ===
    def remember(self, content: str, memory_type: str = "interaction"):
        """Store + auto-classify in one step."""
        from .memory import Memory
        memory = Memory(content, memory_type)
        
        # Auto-classify with CXD if available
        if self.cxd:
            try:
                classification = self.cxd.classify(content)
                memory.metadata = {"cxd": getattr(classification, 'pattern', 'unknown')}
            except Exception:
                pass  # Continue without classification
        
        return self.memory.add(memory)
    
    def recall_cxd(self, query: str, limit: int = 10):
        """Hybrid semantic search."""
        # For now, use basic search - can be enhanced later
        return self.memory.search(query, limit=limit)
    
    def think_with_memory(self, input_text: str):
        """Contextual processing."""
        return self.assistant.think(input_text)
    
    def status(self):
        """System status with AMMS information."""
        all_memories = self.memory.get_all()
        all_tales = self.tales_manager.list_tales()
        
        # Get AMMS status if available
        amms_status = {}
        if hasattr(self.memory, 'get_active_pool_status'):
            try:
                amms_status = self.memory.get_active_pool_status()
            except Exception as e:
                amms_status = {"error": str(e)}
        
        base_status = {
            "memories": len(all_memories),
            "tales": len(all_tales),
            "cxd_available": self.cxd is not None,
            "amms_active": getattr(self.memory, 'is_amms_active', False),
            "status": "operational"
        }
        
        # Include AMMS details if available
        if amms_status and 'error' not in amms_status:
            base_status.update({
                "amms_pool_status": amms_status.get('status_stats', {}),
                "amms_performance": amms_status.get('performance', {}),
                "memory_types": amms_status.get('memory_type_distribution', {})
            })
        
        return base_status
    
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
    
    def save_tale(self, name: str, content: str, category: str = "claude/core", tags=None):
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
    
    def context_tale(self, query: str, style: str = "auto", max_memories: int = 15):
        """Generate narrative from memories."""
        # Get relevant memories
        relevant_memories = self.memory.search(query, limit=max_memories)
        
        if not relevant_memories:
            return f"No memories found for: {query}"
        
        # Create a narrative from the memories
        narrative_parts = [f"📖 Context Tale: {query}\n"]
        
        for i, memory in enumerate(relevant_memories, 1):
            memory_type = getattr(memory, 'memory_type', getattr(memory, 'type', 'unknown'))
            content = memory.content[:200] + "..." if len(memory.content) > 200 else memory.content
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
            if getattr(memory, 'id', None) == memory_id:
                target_memory = memory
                break
        
        if not target_memory:
            return {"error": f"Memory {memory_id} not found"}
        
        return {
            "memory_id": memory_id,
            "current_content": target_memory.content,
            "guidance": "Memory found. Update process would begin here."
        }
    
    def delete_memory_guided(self, memory_id: int, confirm: bool = False):
        """Guided memory deletion."""
        if not confirm:
            return {"error": "Confirmation required for deletion"}
        
        return {"result": f"Memory {memory_id} deletion would be processed here"}
    
    def analyze_memory_patterns(self):
        """Pattern analysis."""
        all_memories = self.memory.get_all()
        
        if not all_memories:
            return {"error": "No memories to analyze"}
        
        # Basic pattern analysis
        type_counts = {}
        total_chars = 0
        
        for memory in all_memories:
            memory_type = getattr(memory, 'memory_type', getattr(memory, 'type', 'unknown'))
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
            total_chars += len(memory.content)
        
        return {
            "total_memories": len(all_memories),
            "type_distribution": type_counts,
            "average_length": total_chars / len(all_memories),
            "analysis": "Basic pattern analysis completed"
        }
    
    # === COGNITIVE (1 tool) ===
    def socratic_dialogue(self, topic: str, depth: int = 3):
        """Self-questioning."""
        # Basic implementation using the SocraticEngine
        if hasattr(self.assistant, 'socratic_engine'):
            try:
                # Create a mock dialogue
                relevant_memories = self.memory.search(topic, limit=5)
                dialogue = self.assistant.socratic_engine.conduct_dialogue(
                    topic, f"Initial analysis of: {topic}", relevant_memories
                )
                
                return {
                    "topic": topic,
                    "depth": depth,
                    "questions": dialogue.questions,
                    "insights": dialogue.insights,
                    "synthesis": dialogue.final_synthesis
                }
            except Exception as e:
                return {"error": f"Socratic dialogue failed: {e}"}
        
        return {"error": "Socratic engine not available"}

# Factory function
def create_memmimic(db_path="memmimic.db"):
    """Create MemMimic instance."""
    return MemMimicAPI(db_path)
