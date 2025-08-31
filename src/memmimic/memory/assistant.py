# clay/assistant.py - With integrated Socratic dialogues
"""
The assistant that remembers and self-questions.
Not to impress, but to persist and deepen understanding.
"""
from typing import Dict, List, Optional
from .memory import Memory, MemoryStore
from .socratic import SocraticEngine

class ContextualAssistant:
    """An assistant that preserves context and self-questions"""
    
    def __init__(self, name: str, db_path: str = None):
        self.name = name
        self.db_path = db_path or f"{name}_memories.db"
        self.memory_store = MemoryStore(self.db_path)
        self.socratic_engine = SocraticEngine(self.memory_store)
        self.current_context = {}
        
    def think(self, user_input: str) -> Dict:
        """
        Pensar ahora incluye auto-cuestionamiento socrático.
        Recordar → Razonar → Responder → Auto-cuestionar → Refinar → Aprender
        """
        # 1. RECORDAR - Buscar memorias relevantes
        relevant_memories = self.memory_store.search(user_input)
        
        # 2. REASON - Build context with memories
        thought_process = self._build_thought_process(user_input, relevant_memories)
        
        # 3. RESPONDER - Generar respuesta inicial
        initial_response = self._generate_response(user_input, thought_process, relevant_memories)
        
        # 4. SELF-QUESTION - Socratic dialogue if appropriate
        socratic_result = self._conduct_socratic_analysis(user_input, initial_response, relevant_memories)
        
        # 5. REFINE - Use Socratic insights to improve response
        final_response = self._refine_response(initial_response, socratic_result)
        
        # 6. LEARN - Save interaction and Socratic dialogue
        self._save_learning(user_input, final_response, socratic_result)
        
        return {
            "response": final_response,
            "memories_used": len(relevant_memories),
            "thought_process": thought_process,
            "socratic_analysis": socratic_result,
            "confidence": self._calculate_confidence(relevant_memories, socratic_result)
        }
    
    def _conduct_socratic_analysis(self, user_input: str, initial_response: str, memories: List[Memory]) -> Optional[Dict]:
        """Perform Socratic analysis if appropriate"""
        
        # Decide if it's worth self-questioning
        if not self.socratic_engine.should_trigger_dialogue(user_input, initial_response, memories):
            return None
        
        # Perform complete Socratic dialogue
        dialogue = self.socratic_engine.conduct_dialogue(user_input, initial_response, memories)
        
        # Save dialogue as memory
        dialogue_memory = dialogue.to_memory()
        self.memory_store.add(dialogue_memory)
        
        return {
            "triggered": True,
            "questions_asked": len(dialogue.questions),
            "insights_generated": len(dialogue.insights),
            "synthesis": dialogue.final_synthesis,
            "dialogue_saved": True
        }
    
    def _refine_response(self, initial_response: str, socratic_result: Optional[Dict]) -> str:
        """Refinar respuesta usando insights socráticos"""
        
        if not socratic_result:
            return initial_response
        
        # Extract recommendations from synthesis
        synthesis = socratic_result["synthesis"]
        
        # Apply refinements according to synthesis
        if "Reformular respuesta" in synthesis:
            # Major reformulation
            refined = f"{initial_response}\n\n[After self-reflection]: {self._extract_synthesis_insight(synthesis)}"
        elif "more explicit about limitations" in synthesis:
            # Add transparency about uncertainty
            refined = f"{initial_response}\n\n[Transparency note]: I must admit my confidence in this response is limited by available context."
        elif "greater process transparency" in synthesis:
            # Explain thought process
            refined = f"{initial_response}\n\n[Transparency]: I reached this response considering {socratic_result['questions_asked']} internal questions and generating {socratic_result['insights_generated']} insights about my own reasoning."
        else:
            # Keep response but indicate self-reflection occurred
            refined = f"{initial_response}\n\n[Self-reflection applied]: I have questioned my internal reasoning to offer a more nuanced perspective."
        
        return refined
    
    def _extract_synthesis_insight(self, synthesis: str) -> str:
        """Extract key insight from Socratic synthesis"""
        lines = synthesis.split('\n')
        for line in lines:
            if "RECOMENDACIÓN:" in line:
                return line.replace("• RECOMENDACIÓN:", "").strip()
        return "I have applied self-questioning to deepen my understanding."
    
    def _save_learning(self, user_input: str, final_response: str, socratic_result: Optional[Dict]):
        """Guardar interacción y aprendizaje"""
        
        # Save main interaction
        interaction_content = f"User: {user_input}\nResponse: {final_response}"
        if socratic_result:
            interaction_content += f"\n[Socratic dialogue applied: {socratic_result['questions_asked']} questions, {socratic_result['insights_generated']} insights]"
        
        interaction_memory = Memory(
            content=interaction_content,
            memory_type="interaction",
            confidence=0.8 if not socratic_result else 0.85  # Mayor confianza si hubo auto-cuestionamiento
        )
        self.memory_store.add(interaction_memory)
    
    def _build_thought_process(self, user_input: str, memories: List[Memory]) -> Dict:
        """Build thought process transparently"""
        process = {
            "input_understood": user_input,
            "memories_activated": len(memories),
            "memory_types": [m.type for m in memories],
            "connections_made": [],
            "context_strength": "high" if len(memories) >= 3 else "medium" if memories else "none",
            "socratic_potential": False  # Se actualizará después
        }
        
        # Analizar conexiones entre input y memorias
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
    
    def _generate_response(self, user_input: str, thought_process: Dict, memories: List[Memory]) -> str:
        """Generar respuesta inicial (será refinada por proceso socrático)"""
        input_lower = user_input.lower()
        
        # Detectar respuestas que citan memorias sintéticas directamente
        synthetic_memories = [m for m in memories if m.type.startswith('synthetic')]
        if synthetic_memories:
            # Priorizar memorias sintéticas más relevantes
            best_synthetic = synthetic_memories[0]
            if any(keyword in input_lower for keyword in ["philosophy", "principle", "architecture", "origin"]):
                return f"Based on what I remember: {best_synthetic.content}\n\nClay is operational with persistent memory working."
        
        # CASE 1: User requests information about Clay project
        if any(term in input_lower for term in ["clay", "project", "status", "memory", "assistant"]):
            return self._respond_about_project(memories)
            
        # CASE 2: User asks for continuity/deepening
        if any(term in input_lower for term in ["more", "details", "continue", "go on", "explain better"]):
            return self._respond_with_continuity(memories, user_input)
            
        # CASE 3: Response with memory context
        if memories:
            return self._respond_with_context(user_input, memories)
            
        # CASE 4: First interaction or no relevant memories
        return self._respond_without_context(user_input)
    
    def _respond_about_project(self, memories: List[Memory]) -> str:
        """Responder sobre el estado del proyecto Clay usando memorias"""
        project_memories = [m for m in memories if any(term in m.content.lower() 
                                                      for term in ["clay", "project", "memory", "status"])]
        
        if project_memories:
            latest = project_memories[0].content
            return f"Based on what I remember: {latest}\n\nClay is operational with persistent memory working."
        
        return "Clay is our persistent memory system. It's working and storing our interactions."
    
    def _respond_with_continuity(self, memories: List[Memory], user_input: str) -> str:
        """Responder pidiendo más detalles sobre interacciones previas"""
        if not memories:
            return "I don't have enough prior context to go deeper. Could you give me more information?"
            
        recent_interactions = [m for m in memories if m.type == "interaction"]
        if recent_interactions:
            last_interaction = recent_interactions[0].content
            return f"Continuing with what we discussed: {last_interaction}\nWhat specific aspect would you like me to elaborate on?"
            
        return "I remember our previous conversations. What specific topic would you like more details about?"
    
    def _respond_with_context(self, user_input: str, memories: List[Memory]) -> str:
        """Responder usando el contexto de las memorias recuperadas"""
        # Extraer temas principales de las memorias
        memory_content = " ".join([m.content for m in memories[:3]])  # Top 3 memorias más relevantes
        
        # Construir respuesta contextual
        context_summary = self._extract_key_context(memory_content)
        
        response = f"Considering our previous conversations about {context_summary}, "
        
        # Añadir respuesta específica al input actual
        if "how" in user_input.lower():
            response += "I can explain the process step by step."
        elif "why" in user_input.lower():
            response += "the reasons are important to understand."
        elif "what" in user_input.lower():
            response += "let me clarify that concept for you."
        else:
            response += "I can give you an informed perspective."
            
        return response
    
    def _respond_without_context(self, user_input: str) -> str:
        """Responder sin memorias previas, pero de forma útil"""
        if "hello" in user_input.lower() or "hi" in user_input.lower():
            return "Hello! I'm Clay, your assistant with persistent memory. How can I help you?"
            
        return f"I understand you're asking about '{user_input}'. Although I don't have prior context on this topic, I can help you. Can you give me more details?"
    
    def _extract_key_context(self, memory_content: str) -> str:
        """Extraer conceptos clave del contenido de memorias"""
        # Common technical keywords
        key_terms = []
        content_lower = memory_content.lower()
        
        # Search for relevant terms
        technical_terms = ["clay", "memory", "assistant", "project", "database", 
                          "persistente", "contexto", "mcp", "servidor"]
        
        for term in technical_terms:
            if term in content_lower:
                key_terms.append(term)
                
        if key_terms:
            return ", ".join(key_terms[:3])  # Top 3 terms
        else:
            return "our conversation topics"
    
    def _calculate_confidence(self, memories: List[Memory], socratic_result: Optional[Dict]) -> float:
        """Calcular confianza basada en memorias y análisis socrático"""
        if not memories:
            base_confidence = 0.3  # Baja confianza sin contexto
        else:
            # More relevant memories = more confidence
            base_confidence = min(0.3 + (len(memories) * 0.15), 0.9)
            
            # Boost por memorias de alta confianza
            avg_memory_confidence = sum(m.confidence for m in memories) / len(memories)
            base_confidence = min(base_confidence + (avg_memory_confidence * 0.1), 0.95)
        
        # Boost from Socratic analysis
        if socratic_result:
            # Self-questioning increases confidence in final response
            socratic_boost = 0.1 if socratic_result["insights_generated"] >= 3 else 0.05
            base_confidence = min(base_confidence + socratic_boost, 0.98)
            
        return base_confidence
