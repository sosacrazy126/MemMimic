# memmimic/assistant.py - With integrated Socratic dialogues
"""
The assistant that remembers and self-questions.
Not to impress, but to persist and deepen understanding.
"""
import logging
from typing import Dict, List, Optional, Any, Union
from .memory import Memory, MemoryStore
from .memory.socratic import SocraticEngine

class ContextualAssistant:
    """An assistant that preserves context and self-questions with improved error handling"""
    
    def __init__(self, name: str, db_path: Optional[str] = None) -> None:
        if not name or not name.strip():
            raise ValueError("Assistant name cannot be empty")
        
        self.name = name.strip()
        self.db_path = db_path or f"{self.name}_memories.db"
        self.logger = logging.getLogger(__name__)
        
        try:
            self.memory_store = MemoryStore(self.db_path)
            self.socratic_engine = SocraticEngine(self.memory_store)
            self.current_context: Dict[str, Any] = {}
            self.logger.info(f"ContextualAssistant '{self.name}' initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize assistant: {e}")
            raise
        
    def think(self, user_input: str) -> Dict[str, Any]:
        """
        Thinking now includes Socratic self-questioning.
        Remember → Reason → Respond → Self-question → Refine → Learn
        
        Args:
            user_input: The user's input to process
            
        Returns:
            Dictionary containing response, analysis, and metadata
            
        Raises:
            ValueError: If user_input is empty
            RuntimeError: If thinking process fails
        """
        if not user_input or not user_input.strip():
            raise ValueError("User input cannot be empty")
            
        try:
            user_input = user_input.strip()
            self.logger.debug(f"Processing user input: {user_input[:100]}...")
            # 1. REMEMBER - Search for relevant memories
            relevant_memories = self.memory_store.search(user_input)
            self.logger.debug(f"Found {len(relevant_memories)} relevant memories")
            
            # 2. REASON - Build context with memories
            thought_process = self._build_thought_process(user_input, relevant_memories)
            
            # 3. RESPOND - Generate initial response
            initial_response = self._generate_response(user_input, thought_process, relevant_memories)
            
            # 4. SELF-QUESTION - Socratic dialogue if appropriate
            socratic_result = self._conduct_socratic_analysis(user_input, initial_response, relevant_memories)
            
            # 5. REFINE - Use Socratic insights to improve response
            final_response = self._refine_response(initial_response, socratic_result)
            
            # 6. LEARN - Save interaction and Socratic dialogue
            self._save_learning(user_input, final_response, socratic_result)
            
            result = {
                "response": final_response,
                "memories_used": len(relevant_memories),
                "thought_process": thought_process,
                "socratic_analysis": socratic_result,
                "confidence": self._calculate_confidence(relevant_memories, socratic_result)
            }
            
            self.logger.debug(f"Think process completed with confidence: {result['confidence']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Think process failed: {e}")
            raise RuntimeError(f"Failed to process user input: {e}") from e
    
    def _conduct_socratic_analysis(self, user_input: str, initial_response: str, memories: List[Memory]) -> Optional[Dict[str, Any]]:
        """Perform Socratic analysis if appropriate with improved error handling"""
        
        try:
            # Decide if it's worth self-questioning
            if not self.socratic_engine.should_trigger_dialogue(user_input, initial_response, memories):
                self.logger.debug("Socratic dialogue not triggered")
                return None
            
            self.logger.debug("Conducting Socratic dialogue")
            
            # Perform complete Socratic dialogue
            dialogue = self.socratic_engine.conduct_dialogue(user_input, initial_response, memories)
            
            # Save dialogue as memory
            dialogue_memory = dialogue.to_memory()
            memory_id = self.memory_store.add(dialogue_memory)
            
            result = {
                "triggered": True,
                "questions_asked": len(dialogue.questions),
                "insights_generated": len(dialogue.insights),
                "synthesis": dialogue.final_synthesis,
                "dialogue_saved": True,
                "memory_id": memory_id
            }
            
            self.logger.debug(f"Socratic analysis completed: {result['questions_asked']} questions, {result['insights_generated']} insights")
            return result
            
        except Exception as e:
            self.logger.error(f"Socratic analysis failed: {e}")
            return None
    
    def _refine_response(self, initial_response: str, socratic_result: Optional[Dict[str, Any]]) -> str:
        """Refine response using Socratic insights with better error handling"""
        
        if not socratic_result:
            return initial_response
        
        # Extract recommendations from synthesis
        synthesis = socratic_result["synthesis"]
        
        # Apply refinements according to synthesis
        if "Reformulate response" in synthesis:
            # Major reformulation
            refined = f"{initial_response}\n\n[After self-reflection]: {self._extract_synthesis_insight(synthesis)}"
        elif "more explicit about limitations" in synthesis:
            # Add transparency about uncertainty
            refined = f"{initial_response}\n\n[Transparency note]: I must admit that my confidence in this response is limited by the available context."
        elif "greater process transparency" in synthesis:
            # Explain thinking process
            refined = f"{initial_response}\n\n[Transparency]: I arrived at this response by considering {socratic_result['questions_asked']} internal questions and generating {socratic_result['insights_generated']} insights about my own reasoning."
        else:
            # Keep response but indicate self-reflection occurred
            refined = f"{initial_response}\n\n[Self-reflection applied]: I have questioned my internal reasoning to offer a more nuanced perspective."
        
        return refined
    
    def _extract_synthesis_insight(self, synthesis: str) -> str:
        """Extract key insight from Socratic synthesis"""
        lines = synthesis.split('\n')
        for line in lines:
            if "RECOMMENDATION:" in line:
                return line.replace("• RECOMMENDATION:", "").strip()
        return "I have applied self-questioning to deepen my understanding."
    
    def _save_learning(self, user_input: str, final_response: str, socratic_result: Optional[Dict[str, Any]]) -> None:
        """Save interaction and learning with improved error handling"""
        
        try:
            # Save main interaction
            interaction_content = f"User: {user_input}\nResponse: {final_response}"
            if socratic_result:
                interaction_content += f"\n[Socratic dialogue applied: {socratic_result['questions_asked']} questions, {socratic_result['insights_generated']} insights]"
            
            interaction_memory = Memory(
                content=interaction_content,
                memory_type="interaction",
                confidence=0.8 if not socratic_result else 0.85  # Higher confidence with self-questioning
            )
            memory_id = self.memory_store.add(interaction_memory)
            self.logger.debug(f"Saved interaction memory with ID: {memory_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save learning: {e}")
            # Don't raise, as this is not critical for the main response
    
    def _build_thought_process(self, user_input: str, memories: List[Memory]) -> Dict[str, Any]:
        """Build thought process transparently with improved analysis"""
        process = {
            "input_understood": user_input,
            "memories_activated": len(memories),
            "memory_types": [m.type for m in memories],
            "connections_made": [],
            "context_strength": "high" if len(memories) >= 3 else "medium" if memories else "none",
            "socratic_potential": False  # Will be updated later
        }
        
        # Analyze connections between input and memories
        input_lower = user_input.lower()
        for memory in memories:
            memory_lower = memory.content.lower()
            
            # Detect continuity patterns
            if any(word in memory_lower for word in input_lower.split() if len(word) > 3):
                process["connections_made"].append(f"Continuity detected with previous memory: {memory.type}")
                
            # Detect deepening questions
            if any(trigger in input_lower for trigger in ["more", "details", "explain", "how", "why"]):
                process["connections_made"].append("User requests deepening")
                process["socratic_potential"] = True
                
        return process
    
    def _generate_response(self, user_input: str, thought_process: Dict[str, Any], memories: List[Memory]) -> str:
        """Generate initial response (will be refined by Socratic process) with improved error handling"""
        input_lower = user_input.lower()
        
        # Detect responses that directly cite synthetic memories
        synthetic_memories = [m for m in memories if m.type.startswith('synthetic')]
        if synthetic_memories:
            # Prioritize most relevant synthetic memories
            best_synthetic = synthetic_memories[0]
            if any(keyword in input_lower for keyword in ["philosophy", "principle", "architecture", "origin"]):
                return f"Based on what I remember: {best_synthetic.content}\n\nMemMimic is operational with persistent memory functioning."
        
        # CASE 1: User requests information about the MemMimic project
        if any(term in input_lower for term in ["memmimic", "project", "status", "memory", "assistant"]):
            return self._respond_about_project(memories)
            
        # CASE 2: User requests continuity/deepening
        if any(term in input_lower for term in ["more", "details", "continue", "follow", "explain better"]):
            return self._respond_with_continuity(memories, user_input)
            
        # CASE 3: Response with memory context
        if memories:
            return self._respond_with_context(user_input, memories)
            
        # CASE 4: First interaction or no relevant memories
        return self._respond_without_context(user_input)
    
    def _respond_about_project(self, memories: List[Memory]) -> str:
        """Respond about MemMimic project status using memories with validation"""
        project_memories = [m for m in memories if any(term in m.content.lower() 
                                                      for term in ["memmimic", "project", "memory", "status"])]
        
        if project_memories:
            latest = project_memories[0].content
            return f"Based on what I remember: {latest}\n\nMemMimic is operational with persistent memory functioning."
        
        return "MemMimic is our persistent memory system. It's functioning and storing our interactions."
    
    def _respond_with_continuity(self, memories: List[Memory], user_input: str) -> str:
        """Respond asking for more details about previous interactions"""
        if not memories:
            return "I don't have enough previous context to deepen. Could you give me more information?"
            
        recent_interactions = [m for m in memories if m.type == "interaction"]
        if recent_interactions:
            last_interaction = recent_interactions[0].content
            return f"Continuing with what we were discussing: {last_interaction}\nWhat specific aspect would you like me to deepen?"
            
        return "I remember our previous conversations. What specific topic would you like more details about?"
    
    def _respond_with_context(self, user_input: str, memories: List[Memory]) -> str:
        """Respond using context from retrieved memories"""
        # Extract main themes from memories
        memory_content = " ".join([m.content for m in memories[:3]])  # Top 3 most relevant memories
        
        # Build contextual response
        context_summary = self._extract_key_context(memory_content)
        
        response = f"Considering our previous conversations about {context_summary}, "
        
        # Add specific response to current input
        if "how" in user_input.lower():
            response += "I can explain the process step by step."
        elif "why" in user_input.lower():
            response += "the reasons are important to understand."
        elif "what" in user_input.lower():
            response += "let me clarify that concept."
        else:
            response += "I can give you an informed perspective."
            
        return response
    
    def _respond_without_context(self, user_input: str) -> str:
        """Respond without previous memories, but helpfully"""
        if "hello" in user_input.lower() or "hi" in user_input.lower():
            return "Hello! I'm MemMimic, your assistant with persistent memory. How can I help you?"
            
        return f"I understand you're asking about '{user_input}'. Although I don't have previous context about this topic, I can help you. Can you give me more details?"
    
    def _extract_key_context(self, memory_content: str) -> str:
        """Extract key concepts from memory content"""
        # Common technical keywords
        key_terms = []
        content_lower = memory_content.lower()
        
        # Search for relevant terms
        technical_terms = ["memmimic", "memory", "assistant", "project", "database", 
                          "persistent", "context", "mcp", "server"]
        
        for term in technical_terms:
            if term in content_lower:
                key_terms.append(term)
                
        if key_terms:
            return ", ".join(key_terms[:3])  # Top 3 terms
        else:
            return "our conversation topics"
    
    def _calculate_confidence(self, memories: List[Memory], socratic_result: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence based on memories and Socratic analysis with bounds checking"""
        try:
            if not memories:
                base_confidence = 0.3  # Low confidence without context
            else:
                # More relevant memories = more confidence
                base_confidence = min(0.3 + (len(memories) * 0.15), 0.9)
                
                # Boost for high-confidence memories
                if memories:
                    avg_memory_confidence = sum(m.confidence for m in memories) / len(memories)
                    base_confidence = min(base_confidence + (avg_memory_confidence * 0.1), 0.95)
            
            # Boost for Socratic analysis
            if socratic_result and socratic_result.get("insights_generated", 0) > 0:
                # Self-questioning increases confidence in final response
                socratic_boost = 0.1 if socratic_result["insights_generated"] >= 3 else 0.05
                base_confidence = min(base_confidence + socratic_boost, 0.98)
            
            # Ensure confidence is within bounds
            confidence = max(0.0, min(1.0, base_confidence))
            self.logger.debug(f"Calculated confidence: {confidence:.3f}")
            return confidence
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence
