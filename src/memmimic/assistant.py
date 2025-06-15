# clay/assistant.py - Con diálogos socráticos integrados
"""
El asistente que recuerda y se auto-cuestiona.
No para impresionar, sino para persistir y profundizar.
"""
from typing import Dict, List, Optional
from .memory import Memory, MemoryStore
from .socratic import SocraticEngine

class ContextualAssistant:
    """Un asistente que preserva contexto y se auto-cuestiona"""
    
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
        
        # 2. RAZONAR - Construir contexto con memorias
        thought_process = self._build_thought_process(user_input, relevant_memories)
        
        # 3. RESPONDER - Generar respuesta inicial
        initial_response = self._generate_response(user_input, thought_process, relevant_memories)
        
        # 4. AUTO-CUESTIONARSE - Diálogo socrático si es apropiado
        socratic_result = self._conduct_socratic_analysis(user_input, initial_response, relevant_memories)
        
        # 5. REFINAR - Usar insights socráticos para mejorar respuesta
        final_response = self._refine_response(initial_response, socratic_result)
        
        # 6. APRENDER - Guardar interacción y diálogo socrático
        self._save_learning(user_input, final_response, socratic_result)
        
        return {
            "response": final_response,
            "memories_used": len(relevant_memories),
            "thought_process": thought_process,
            "socratic_analysis": socratic_result,
            "confidence": self._calculate_confidence(relevant_memories, socratic_result)
        }
    
    def _conduct_socratic_analysis(self, user_input: str, initial_response: str, memories: List[Memory]) -> Optional[Dict]:
        """Realizar análisis socrático si es apropiado"""
        
        # Decidir si vale la pena auto-cuestionarse
        if not self.socratic_engine.should_trigger_dialogue(user_input, initial_response, memories):
            return None
        
        # Realizar diálogo socrático completo
        dialogue = self.socratic_engine.conduct_dialogue(user_input, initial_response, memories)
        
        # Guardar diálogo como memoria
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
        
        # Extraer recomendaciones de la síntesis
        synthesis = socratic_result["synthesis"]
        
        # Aplicar refinamientos según síntesis
        if "Reformular respuesta" in synthesis:
            # Reformulación mayor
            refined = f"{initial_response}\n\n[Tras auto-reflexión]: {self._extract_synthesis_insight(synthesis)}"
        elif "más explícito sobre limitaciones" in synthesis:
            # Añadir transparencia sobre incertidumbre
            refined = f"{initial_response}\n\n[Nota de transparencia]: Debo admitir que mi confianza en esta respuesta es limitada por el contexto disponible."
        elif "mayor transparencia del proceso" in synthesis:
            # Explicar proceso de pensamiento
            refined = f"{initial_response}\n\n[Transparencia]: Llegué a esta respuesta considerando {socratic_result['questions_asked']} preguntas internas y generando {socratic_result['insights_generated']} insights sobre mi propio razonamiento."
        else:
            # Mantener respuesta pero indicar que hubo auto-reflexión
            refined = f"{initial_response}\n\n[Auto-reflexión aplicada]: He cuestionado mi razonamiento interno para ofrecer una perspectiva más matizada."
        
        return refined
    
    def _extract_synthesis_insight(self, synthesis: str) -> str:
        """Extraer insight clave de la síntesis socrática"""
        lines = synthesis.split('\n')
        for line in lines:
            if "RECOMENDACIÓN:" in line:
                return line.replace("• RECOMENDACIÓN:", "").strip()
        return "He aplicado auto-cuestionamiento para profundizar mi comprensión."
    
    def _save_learning(self, user_input: str, final_response: str, socratic_result: Optional[Dict]):
        """Guardar interacción y aprendizaje"""
        
        # Guardar interacción principal
        interaction_content = f"Usuario: {user_input}\nRespuesta: {final_response}"
        if socratic_result:
            interaction_content += f"\n[Diálogo socrático aplicado: {socratic_result['questions_asked']} preguntas, {socratic_result['insights_generated']} insights]"
        
        interaction_memory = Memory(
            content=interaction_content,
            memory_type="interaction",
            confidence=0.8 if not socratic_result else 0.85  # Mayor confianza si hubo auto-cuestionamiento
        )
        self.memory_store.add(interaction_memory)
    
    def _build_thought_process(self, user_input: str, memories: List[Memory]) -> Dict:
        """Construir el proceso de pensamiento de forma transparente"""
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
            
            # Detectar patrones de continuidad
            if any(word in memory_lower for word in input_lower.split() if len(word) > 3):
                process["connections_made"].append(f"Continuidad detectada con memoria previa: {memory.type}")
                
            # Detectar preguntas de profundización
            if any(trigger in input_lower for trigger in ["más", "detalles", "explica", "cómo", "por qué"]):
                process["connections_made"].append("Usuario solicita profundización")
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
            if any(keyword in input_lower for keyword in ["filosofía", "principio", "arquitectura", "origen"]):
                return f"Basándome en lo que recuerdo: {best_synthetic.content}\n\nClay está operativo con memoria persistente funcionando."
        
        # CASO 1: Usuario solicita información sobre el proyecto Clay
        if any(term in input_lower for term in ["clay", "proyecto", "estado", "memoria", "asistente"]):
            return self._respond_about_project(memories)
            
        # CASO 2: Usuario pide continuidad/profundización
        if any(term in input_lower for term in ["más", "detalles", "continúa", "sigue", "explica mejor"]):
            return self._respond_with_continuity(memories, user_input)
            
        # CASO 3: Respuesta con contexto de memorias
        if memories:
            return self._respond_with_context(user_input, memories)
            
        # CASO 4: Primera interacción o sin memorias relevantes
        return self._respond_without_context(user_input)
    
    def _respond_about_project(self, memories: List[Memory]) -> str:
        """Responder sobre el estado del proyecto Clay usando memorias"""
        project_memories = [m for m in memories if any(term in m.content.lower() 
                                                      for term in ["clay", "proyecto", "memoria", "estado"])]
        
        if project_memories:
            latest = project_memories[0].content
            return f"Basándome en lo que recuerdo: {latest}\n\nClay está operativo con memoria persistente funcionando."
        
        return "Clay es nuestro sistema de memoria persistente. Está funcionando y almacenando nuestras interacciones."
    
    def _respond_with_continuity(self, memories: List[Memory], user_input: str) -> str:
        """Responder pidiendo más detalles sobre interacciones previas"""
        if not memories:
            return "No tengo suficiente contexto previo para profundizar. ¿Podrías darme más información?"
            
        recent_interactions = [m for m in memories if m.type == "interaction"]
        if recent_interactions:
            last_interaction = recent_interactions[0].content
            return f"Continuando con lo que discutíamos: {last_interaction}\n¿En qué aspecto específico quieres que profundice?"
            
        return "Recuerdo nuestras conversaciones anteriores. ¿Sobre qué tema específico quieres más detalles?"
    
    def _respond_with_context(self, user_input: str, memories: List[Memory]) -> str:
        """Responder usando el contexto de las memorias recuperadas"""
        # Extraer temas principales de las memorias
        memory_content = " ".join([m.content for m in memories[:3]])  # Top 3 memorias más relevantes
        
        # Construir respuesta contextual
        context_summary = self._extract_key_context(memory_content)
        
        response = f"Considerando nuestras conversaciones anteriores sobre {context_summary}, "
        
        # Añadir respuesta específica al input actual
        if "cómo" in user_input.lower():
            response += "puedo explicarte el proceso paso a paso."
        elif "por qué" in user_input.lower():
            response += "las razones son importantes de entender."
        elif "qué" in user_input.lower():
            response += "déjame clarificarte ese concepto."
        else:
            response += "puedo darte una perspectiva informada."
            
        return response
    
    def _respond_without_context(self, user_input: str) -> str:
        """Responder sin memorias previas, pero de forma útil"""
        if "hola" in user_input.lower() or "hi" in user_input.lower():
            return "¡Hola! Soy Clay, tu asistente con memoria persistente. ¿En qué puedo ayudarte?"
            
        return f"Entiendo que preguntas sobre '{user_input}'. Aunque no tengo contexto previo sobre este tema, puedo ayudarte. ¿Puedes darme más detalles?"
    
    def _extract_key_context(self, memory_content: str) -> str:
        """Extraer conceptos clave del contenido de memorias"""
        # Palabras clave técnicas comunes
        key_terms = []
        content_lower = memory_content.lower()
        
        # Buscar términos relevantes
        technical_terms = ["clay", "memoria", "asistente", "proyecto", "base de datos", 
                          "persistente", "contexto", "mcp", "servidor"]
        
        for term in technical_terms:
            if term in content_lower:
                key_terms.append(term)
                
        if key_terms:
            return ", ".join(key_terms[:3])  # Top 3 términos
        else:
            return "nuestros temas de conversación"
    
    def _calculate_confidence(self, memories: List[Memory], socratic_result: Optional[Dict]) -> float:
        """Calcular confianza basada en memorias y análisis socrático"""
        if not memories:
            base_confidence = 0.3  # Baja confianza sin contexto
        else:
            # Más memorias relevantes = más confianza
            base_confidence = min(0.3 + (len(memories) * 0.15), 0.9)
            
            # Boost por memorias de alta confianza
            avg_memory_confidence = sum(m.confidence for m in memories) / len(memories)
            base_confidence = min(base_confidence + (avg_memory_confidence * 0.1), 0.95)
        
        # Boost por análisis socrático
        if socratic_result:
            # Auto-cuestionamiento aumenta confianza en la respuesta final
            socratic_boost = 0.1 if socratic_result["insights_generated"] >= 3 else 0.05
            base_confidence = min(base_confidence + socratic_boost, 0.98)
            
        return base_confidence
