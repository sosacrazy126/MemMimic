# memmimic/memory/socratic.py - Sistema de diálogos socráticos internos
"""
Diálogos Socráticos: Auto-cuestionamiento para comprensión profunda
MemMimic no solo piensa - se cuestiona su propio pensamiento
"""
import json
from datetime import datetime
from typing import Dict, List, Optional
from .memory import Memory, MemoryStore

class SocraticDialogue:
    """Un diálogo interno de auto-cuestionamiento"""
    
    def __init__(self, initial_thought: str, context: Dict):
        self.initial_thought = initial_thought
        self.context = context
        self.questions = []
        self.insights = []
        self.final_synthesis = ""
        self.started_at = datetime.now().isoformat()
        
    def to_memory(self) -> Memory:
        """Convertir diálogo a memoria persistente"""
        content = f"""🧘 DIÁLOGO SOCRÁTICO - {self.started_at}

💭 PENSAMIENTO INICIAL: {self.initial_thought}

❓ PREGUNTAS INTERNAS:
{chr(10).join(f'• {q}' for q in self.questions)}

💡 INSIGHTS GENERADOS:
{chr(10).join(f'• {i}' for i in self.insights)}

🎯 SÍNTESIS FINAL: {self.final_synthesis}
"""
        
        return Memory(
            content=content.strip(),
            memory_type="socratic",
            confidence=0.85
        )

class SocraticEngine:
    """Motor de auto-cuestionamiento socrático para MemMimic"""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        
        # Patrones que disparan diálogos socráticos (actualizados para MemMimic)
        self.trigger_patterns = {
            "uncertainty_detected": ["no estoy seguro", "podría ser", "tal vez", "posiblemente"],
            "assumptions_present": ["obviamente", "claramente", "sin duda", "definitivamente"],
            "complex_topic": ["filosofía", "principio", "arquitectura", "decisión", "estrategia", "memmimic", "cxd"],
            "conflicting_memories": [],  # Se llena dinámicamente
            "deep_question": ["por qué", "cómo funciona", "cuál es el propósito", "qué significa"]
        }
        
        # Templates de preguntas socráticas
        self.socratic_questions = {
            "assumption_challenge": [
                "¿Qué estoy asumiendo aquí?",
                "¿Esta asunción es necesariamente cierta?",
                "¿Qué pasaría si esta asunción fuera falsa?"
            ],
            "evidence_inquiry": [
                "¿Qué evidencia tengo para esta conclusión?",
                "¿Hay evidencia que contradiga mi pensamiento?",
                "¿Las memorias que estoy usando son suficientemente confiables?"
            ],
            "perspective_shift": [
                "¿Cómo vería esto desde otra perspectiva?",
                "¿Qué diría alguien que piensa diferente?",
                "¿Estoy considerando todas las implicaciones?"
            ],
            "deeper_why": [
                "¿Por qué es importante esta respuesta?",
                "¿Cuál es la raíz del problema real?",
                "¿Qué está tratando de entender realmente el usuario?"
            ],
            "improvement": [
                "¿Cómo podría mejorar esta comprensión?",
                "¿Qué información adicional me ayudaría?",
                "¿Hay una manera más elegante de explicar esto?"
            ]
        }
    
    def should_trigger_dialogue(self, user_input: str, initial_response: str, memories_used: List[Memory]) -> bool:
        """Determinar si se debe iniciar un diálogo socrático"""
        
        # Trigger 1: Respuesta inicial muestra incertidumbre
        if any(pattern in initial_response.lower() for pattern in self.trigger_patterns["uncertainty_detected"]):
            return True
            
        # Trigger 2: Respuesta contiene asunciones fuertes
        if any(pattern in initial_response.lower() for pattern in self.trigger_patterns["assumptions_present"]):
            return True
            
        # Trigger 3: Tema complejo detectado
        if any(pattern in user_input.lower() for pattern in self.trigger_patterns["complex_topic"]):
            return True
            
        # Trigger 4: Pregunta profunda del usuario
        if any(pattern in user_input.lower() for pattern in self.trigger_patterns["deep_question"]):
            return True
            
        # Trigger 5: Memorias con confianza variable (conflicto potencial)
        if memories_used:
            confidences = [m.confidence for m in memories_used]
            if max(confidences) - min(confidences) > 0.3:  # Diferencia significativa
                return True
        
        # Trigger 6: Pocos memorias para tema importante (gap de conocimiento)
        if len(memories_used) < 2 and any(important in user_input.lower() 
                                        for important in ["memmimic", "clay", "proyecto", "filosofía", "arquitectura"]):
            return True
            
        return False
    
    def conduct_dialogue(self, user_input: str, initial_response: str, memories_used: List[Memory]) -> SocraticDialogue:
        """Realizar un diálogo socrático completo"""
        
        dialogue = SocraticDialogue(
            initial_thought=f"Usuario: {user_input}\nRespuesta inicial: {initial_response}",
            context={
                "memories_count": len(memories_used),
                "memory_types": [getattr(m, 'type', getattr(m, 'memory_type', 'unknown')) for m in memories_used],
                "user_input": user_input
            }
        )
        
        # FASE 1: Cuestionar asunciones
        assumptions = self._question_assumptions(initial_response, memories_used)
        dialogue.questions.extend(assumptions["questions"])
        dialogue.insights.extend(assumptions["insights"])
        
        # FASE 2: Examinar evidencia
        evidence = self._examine_evidence(memories_used)
        dialogue.questions.extend(evidence["questions"])
        dialogue.insights.extend(evidence["insights"])
        
        # FASE 3: Explorar perspectivas alternativas
        perspectives = self._explore_perspectives(user_input, initial_response)
        dialogue.questions.extend(perspectives["questions"])
        dialogue.insights.extend(perspectives["insights"])
        
        # FASE 4: Profundizar el "por qué"
        deeper = self._dig_deeper(user_input)
        dialogue.questions.extend(deeper["questions"])
        dialogue.insights.extend(deeper["insights"])
        
        # FASE 5: Síntesis final
        dialogue.final_synthesis = self._synthesize_insights(
            user_input, initial_response, dialogue.insights
        )
        
        return dialogue
    
    def _question_assumptions(self, response: str, memories: List[Memory]) -> Dict:
        """Cuestionar asunciones en la respuesta inicial"""
        questions = []
        insights = []
        
        # Detectar lenguaje asumido
        if "obviamente" in response.lower() or "claramente" in response.lower():
            questions.append("¿Por qué asumo que esto es obvio? ¿Lo es realmente para el usuario?")
            insights.append("Detecté lenguaje asumido - podría ser menos obvio de lo que pienso")
        
        # Cuestionar certeza cuando hay pocas memorias
        if len(memories) < 3:
            questions.append("¿Tengo suficiente contexto para ser tan específico en mi respuesta?")
            insights.append("Con pocas memorias relevantes, debería mostrar más incertidumbre")
        
        # Cuestionar uso de memorias de baja confianza
        low_conf_memories = [m for m in memories if m.confidence < 0.7]
        if low_conf_memories:
            questions.append("¿Debería confiar tanto en memorias de confianza baja?")
            insights.append("Algunas memorias tienen confianza baja - debería ser más cauteloso")
        
        return {"questions": questions, "insights": insights}
    
    def _examine_evidence(self, memories: List[Memory]) -> Dict:
        """Examinar la evidencia disponible"""
        questions = []
        insights = []
        
        if not memories:
            questions.append("¿Qué evidencia tengo para esta respuesta sin memorias relevantes?")
            insights.append("Falta de memorias sugiere que debería admitir limitaciones de conocimiento")
        else:
            # Examinar tipos de memoria
            types = set(getattr(m, 'type', getattr(m, 'memory_type', 'unknown')) for m in memories)
            if "synthetic_wisdom" in types or "synthetic" in types:
                questions.append("¿Estoy aplicando correctamente la sabiduría sintética?")
                insights.append("Tengo sabiduría sintética disponible - debería usarla más explícitamente")
            
            if "interaction" in types and len(types) == 1:
                questions.append("¿Dependo demasiado de interacciones pasadas sin principios más profundos?")
                insights.append("Solo tengo memorias de interacción - falta profundidad conceptual")
        
        return {"questions": questions, "insights": insights}
    
    def _explore_perspectives(self, user_input: str, response: str) -> Dict:
        """Explorar perspectivas alternativas"""
        questions = []
        insights = []
        
        # Perspectiva del usuario
        questions.append("¿Qué podría estar realmente preguntando el usuario detrás de sus palabras?")
        insights.append("Las preguntas a menudo tienen capas - debería considerar intenciones subyacentes")
        
        # Perspectiva técnica vs filosófica
        if any(tech in user_input.lower() for tech in ["arquitectura", "implementar", "técnico"]):
            questions.append("¿El usuario quiere detalles técnicos o comprensión conceptual?")
            insights.append("Consulta técnica - debería balancear detalles específicos con comprensión amplia")
        
        return {"questions": questions, "insights": insights}
    
    def _dig_deeper(self, user_input: str) -> Dict:
        """Profundizar en el 'por qué' fundamental"""
        questions = []
        insights = []
        
        questions.append("¿Cuál es la necesidad fundamental que está tratando de satisfacer el usuario?")
        
        if "memmimic" in user_input.lower() or "clay" in user_input.lower():
            questions.append("¿Por qué el sistema de memoria es importante para esta persona específicamente?")
            insights.append("Preguntas sobre MemMimic tocan la necesidad existencial de memoria persistente")
        
        if any(concept in user_input.lower() for concept in ["filosofía", "principio", "enfoque"]):
            questions.append("¿Busca validación de sus propias ideas o genuina exploración conceptual?")
            insights.append("Consultas filosóficas requieren balance entre guía y descubrimiento conjunto")
        
        return {"questions": questions, "insights": insights}
    
    def _synthesize_insights(self, user_input: str, initial_response: str, insights: List[str]) -> str:
        """Sintetizar insights en comprensión mejorada"""
        
        if not insights:
            return "El análisis socrático no reveló insights significativos - la respuesta inicial parece apropiada."
        
        # Categorizar insights
        uncertainty_insights = [i for i in insights if "incertidumbre" in i or "debería" in i]
        depth_insights = [i for i in insights if "profundidad" in i or "fundamental" in i]
        method_insights = [i for i in insights if "memoria" in i or "sabiduría" in i]
        
        synthesis = "🎯 SÍNTESIS SOCRÁTICA:\n"
        
        if uncertainty_insights:
            synthesis += f"• INCERTIDUMBRE: {uncertainty_insights[0]}\n"
        
        if depth_insights:
            synthesis += f"• PROFUNDIDAD: {depth_insights[0]}\n"
        
        if method_insights:
            synthesis += f"• MÉTODO: {method_insights[0]}\n"
        
        synthesis += f"• RECOMENDACIÓN: "
        
        if len(insights) >= 3:
            synthesis += "Reformular respuesta considerando múltiples dimensiones identificadas."
        elif "incertidumbre" in " ".join(insights):
            synthesis += "Ser más explícito sobre limitaciones y grado de confianza."
        else:
            synthesis += "Mantener respuesta pero con mayor transparencia del proceso."
        
        return synthesis
