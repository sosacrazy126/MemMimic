"""
MemMimic Unified API - 11 Core Tools

The main interface combining all MemMimic functionality.
"""

from .memory import MemoryStore, ContextualAssistant
from .tales import TaleManager
from .cxd import create_optimized_classifier

class MemMimicAPI:
    """Unified API for MemMimic - 11 core tools."""
    
    def __init__(self, db_path="memmimic.db"):
        self.memory = MemoryStore(db_path)
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
    
    def recall_cxd(self, query: str, limit: int = 10, function_filter: str = "ALL"):
        """Hybrid semantic + WordNet search with CXD filtering.
        
        Args:
            query: Search query
            limit: Maximum results to return
            function_filter: CXD function filter (CONTROL, CONTEXT, DATA, ALL)
        """
        import nltk
        from nltk.corpus import wordnet
        
        # Ensure WordNet is downloaded
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        # Step 1: Get base search results
        base_results = self.memory.search(query, limit=limit * 3)  # Get more for filtering
        
        # Step 2: Expand query with WordNet synonyms
        expanded_terms = set([query.lower()])
        for word in query.split():
            if len(word) > 3:  # Skip short words
                try:
                    for syn in wordnet.synsets(word):
                        for lemma in syn.lemmas()[:3]:  # Limit synonyms
                            expanded_terms.add(lemma.name().lower().replace('_', ' '))
                except:
                    pass
        
        # Step 3: Search with expanded terms
        expanded_results = []
        for term in expanded_terms:
            if term != query.lower():
                term_results = self.memory.search(term, limit=5)
                expanded_results.extend(term_results)
        
        # Step 4: Combine and deduplicate results
        all_results = base_results + expanded_results
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
        
        # Step 5: Apply CXD filtering if requested
        if function_filter != "ALL" and self.cxd:
            filtered_results = []
            for memory in unique_results:
                try:
                    classification = self.cxd.classify(memory.content)
                    # Check if the classification matches the filter
                    if hasattr(classification, 'dominant_function'):
                        func_name = classification.dominant_function.value if hasattr(classification.dominant_function, 'value') else str(classification.dominant_function)
                        if func_name == function_filter:
                            filtered_results.append(memory)
                except:
                    pass  # Skip if classification fails
            unique_results = filtered_results if filtered_results else unique_results[:limit]
        
        # Step 6: Score and rank results (fusion scoring)
        scored_results = []
        for memory in unique_results:
            score = 0
            content_lower = memory.content.lower()
            
            # Semantic score (based on direct match)
            if query.lower() in content_lower:
                score += 10
            
            # Lexical expansion score
            for term in expanded_terms:
                if term in content_lower:
                    score += 5
            
            # Confidence boost
            score += memory.confidence * 5
            
            # Type boost for synthetic memories
            if hasattr(memory, 'type') and 'synthetic' in memory.type:
                score += 3
            
            scored_results.append((memory, score))
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, score in scored_results[:limit]]
    
    def think_with_memory(self, input_text: str):
        """Contextual processing."""
        return self.assistant.think(input_text)
    
    def status(self):
        """System status."""
        all_memories = self.memory.get_all()
        all_tales = self.tales_manager.list_tales()
        
        return {
            "memories": len(all_memories),
            "tales": len(all_tales),
            "cxd_available": self.cxd is not None,
            "status": "operational"
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
        """Socratic-guided memory update with deep reflection."""
        from .memory.socratic import SocraticEngine
        
        all_memories = self.memory.get_all()
        
        # Find memory by ID
        target_memory = None
        for memory in all_memories:
            if getattr(memory, 'id', None) == memory_id:
                target_memory = memory
                break
        
        if not target_memory:
            return {"error": f"Memory {memory_id} not found"}
        
        # Initialize Socratic engine
        socratic_engine = SocraticEngine(self.memory)
        
        # Conduct Socratic dialogue about the memory
        dialogue = socratic_engine.conduct_dialogue(
            user_input=f"Should I update memory {memory_id}?",
            initial_response=target_memory.content,
            memories_used=[target_memory]
        )
        
        # Generate update suggestions based on insights
        update_suggestions = []
        
        # Check for uncertainty indicators
        if any("uncertainty" in insight.lower() for insight in dialogue.insights):
            update_suggestions.append("Add clarity qualifiers (e.g., 'possibly', 'likely')")
        
        # Check for missing context
        if any("context" in insight.lower() for insight in dialogue.insights):
            update_suggestions.append("Add more contextual information")
        
        # Check for low confidence
        if target_memory.confidence < 0.7:
            update_suggestions.append("Consider marking as 'needs review' or updating confidence")
        
        # Store the Socratic dialogue as a new memory
        dialogue_memory = dialogue.to_memory()
        dialogue_id = self.memory.add(dialogue_memory)
        
        return {
            "memory_id": memory_id,
            "current_content": target_memory.content,
            "current_confidence": target_memory.confidence,
            "socratic_questions": dialogue.questions[:3],  # Top 3 questions
            "key_insights": dialogue.insights[:3],  # Top 3 insights
            "update_suggestions": update_suggestions,
            "synthesis": dialogue.final_synthesis,
            "dialogue_memory_id": dialogue_id,
            "guidance": "Review the Socratic analysis above and update the memory content accordingly."
        }
    
    def delete_memory_guided(self, memory_id: int, confirm: bool = False):
        """Guided memory deletion with relationship analysis."""
        if not confirm:
            all_memories = self.memory.get_all()
            
            # Find the target memory
            target_memory = None
            for memory in all_memories:
                if getattr(memory, 'id', None) == memory_id:
                    target_memory = memory
                    break
            
            if not target_memory:
                return {"error": f"Memory {memory_id} not found"}
            
            # Analyze relationships
            related_memories = []
            target_words = set(target_memory.content.lower().split())
            
            for memory in all_memories:
                if memory.id != memory_id:
                    memory_words = set(memory.content.lower().split())
                    overlap = len(target_words & memory_words) / max(len(target_words), 1)
                    if overlap > 0.3:  # 30% word overlap indicates relationship
                        related_memories.append({
                            "id": memory.id,
                            "type": getattr(memory, 'type', 'unknown'),
                            "preview": memory.content[:100]
                        })
            
            return {
                "memory_id": memory_id,
                "content_preview": target_memory.content[:200],
                "type": getattr(target_memory, 'type', 'unknown'),
                "confidence": target_memory.confidence,
                "related_memories_count": len(related_memories),
                "related_memories": related_memories[:5],  # Show first 5
                "warning": f"This memory has {len(related_memories)} related memories that might be affected.",
                "guidance": "Add confirm=True to proceed with deletion"
            }
        
        # Actual deletion with confirmation
        success = self.memory.delete(memory_id)
        
        if success:
            return {
                "result": f"Memory {memory_id} successfully deleted",
                "status": "success"
            }
        else:
            return {
                "error": f"Failed to delete memory {memory_id}",
                "status": "failed"
            }
    
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
