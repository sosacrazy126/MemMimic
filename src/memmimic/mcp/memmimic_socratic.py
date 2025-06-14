#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic - Socratic Dialogue Tool
Engage in deep self-questioning for enhanced understanding
Part of the MemMimic cognitive memory system
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

# Ensure UTF-8 output
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add MemMimic to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
memmimic_src = os.path.join(current_dir, '..', '..')
sys.path.insert(0, memmimic_src)

class SocraticDialogue:
    """A structured internal self-questioning dialogue"""
    
    def __init__(self, initial_thought: str, context: Dict):
        self.initial_thought = initial_thought
        self.context = context
        self.questions = []
        self.insights = []
        self.final_synthesis = ""
        self.started_at = datetime.now().isoformat()
        
    def to_memory(self):
        """Convert dialogue to persistent memory format"""
        from memmimic.memory.memory import Memory
        
        content = f"""🧘 DIÁLOGO SOCRÁTICO - {self.started_at}

💭 PENSAMIENTO INICIAL:
{self.initial_thought}

❓ PREGUNTAS INTERNAS:
{chr(10).join(f'  • {q}' for q in self.questions)}

💡 INSIGHTS GENERADOS:
{chr(10).join(f'  • {i}' for i in self.insights)}

🎯 SÍNTESIS FINAL:
{self.final_synthesis}

📊 CONTEXTO:
  • Memorias consultadas: {self.context.get('memories_count', 0)}
  • Tipos de memoria: {', '.join(self.context.get('memory_types', []))}
  • Profundidad alcanzada: {len(self.questions)} preguntas, {len(self.insights)} insights
"""
        
        return Memory(
            content=content.strip(),
            memory_type="socratic",
            confidence=0.85
        )

class MemMimicSocraticEngine:
    """Enhanced Socratic questioning engine for deep reflection"""
    
    def __init__(self, memory_store):
        self.memory_store = memory_store
        
        # Enhanced trigger patterns for MemMimic context
        self.trigger_patterns = {
            "uncertainty_detected": ["no estoy seguro", "podría ser", "tal vez", "posiblemente", "unclear", "uncertain"],
            "assumptions_present": ["obviamente", "claramente", "sin duda", "definitivamente", "obviously", "clearly"],
            "complex_topic": ["filosofía", "principio", "arquitectura", "decisión", "estrategia", "memmimic", "cxd", "contextual memory"],
            "deep_question": ["por qué", "cómo funciona", "cuál es el propósito", "qué significa", "why", "how does"],
            "cognitive_function": ["control", "context", "data", "classification", "cognitive function"],
            "collaboration": ["sprooket", "claude", "partnership", "collaboration", "co-architect"]
        }
        
        # Enhanced Socratic question templates
        self.socratic_questions = {
            "assumption_challenge": [
                "¿Qué estoy asumiendo aquí que podría no ser cierto?",
                "¿Esta asunción es necesariamente válida en el contexto de MemMimic?",
                "¿Qué pasaría si esta asunción fuera completamente falsa?",
                "¿Estoy proyectando mis limitaciones en lugar de explorar posibilidades?"
            ],
            "evidence_inquiry": [
                "¿Qué evidencia tengo para esta conclusión específica?",
                "¿Las memorias que estoy usando tienen suficiente confianza?",
                "¿Hay evidencia contradictoria que estoy ignorando?",
                "¿Necesito más datos antes de llegar a esta conclusión?"
            ],
            "perspective_shift": [
                "¿Cómo vería esto desde la perspectiva de Sprooket?",
                "¿Qué diría alguien que no conoce MemMimic?",
                "¿Estoy considerando todas las implicaciones cognitivas?",
                "¿Hay una perspectiva técnica vs filosófica que estoy perdiendo?"
            ],
            "deeper_why": [
                "¿Por qué es realmente importante esta respuesta?",
                "¿Cuál es la necesidad fundamental detrás de esta pregunta?",
                "¿Qué está tratando de lograr realmente el usuario?",
                "¿Cómo se relaciona esto con la misión de MemMimic?"
            ],
            "improvement": [
                "¿Cómo podría mejorar esta comprensión significativamente?",
                "¿Qué información adicional transformaría mi respuesta?",
                "¿Hay una manera más elegante y clara de explicar esto?",
                "¿Qué pregunta adicional debería hacerme?"
            ],
            "cognitive_meta": [
                "¿Qué función cognitiva estoy usando predominantemente aquí?",
                "¿Debería balancear más Control, Context y Data?",
                "¿Mi respuesta refleja el tipo de pensamiento que necesita la situación?",
                "¿Estoy siendo coherente con la filosofía de memoria contextual?"
            ]
        }
    
    def should_trigger_dialogue(self, user_input: str, initial_response: str, memories_used: List) -> bool:
        """Determine if Socratic dialogue should be initiated"""
        
        # Trigger 1: Response shows uncertainty
        if any(pattern in initial_response.lower() for pattern in self.trigger_patterns["uncertainty_detected"]):
            return True
            
        # Trigger 2: Strong assumptions detected
        if any(pattern in initial_response.lower() for pattern in self.trigger_patterns["assumptions_present"]):
            return True
            
        # Trigger 3: Complex/important topic
        if any(pattern in user_input.lower() for pattern in self.trigger_patterns["complex_topic"]):
            return True
            
        # Trigger 4: Deep philosophical question
        if any(pattern in user_input.lower() for pattern in self.trigger_patterns["deep_question"]):
            return True
            
        # Trigger 5: Cognitive function discussion
        if any(pattern in user_input.lower() for pattern in self.trigger_patterns["cognitive_function"]):
            return True
            
        # Trigger 6: Collaboration topic
        if any(pattern in user_input.lower() for pattern in self.trigger_patterns["collaboration"]):
            return True
            
        # Trigger 7: Variable confidence in memories (potential conflict)
        if memories_used and len(memories_used) > 1:
            confidences = [getattr(m, 'confidence', 0.5) for m in memories_used]
            if max(confidences) - min(confidences) > 0.3:
                return True
        
        # Trigger 8: Knowledge gap detected
        if len(memories_used) < 2 and any(important in user_input.lower() 
                                        for important in ["memmimic", "clay", "proyecto", "filosofía", "arquitectura"]):
            return True
            
        return False
    
    def conduct_dialogue(self, user_input: str, initial_response: str = "", memories_used: List = None, depth: int = 3) -> SocraticDialogue:
        """Conduct a complete Socratic dialogue with specified depth"""
        
        if memories_used is None:
            memories_used = []
        
        if not initial_response:
            initial_response = f"Análisis inicial del tema: {user_input}"
        
        dialogue = SocraticDialogue(
            initial_thought=f"🔍 Consulta: {user_input}\n💭 Respuesta inicial: {initial_response}",
            context={
                "memories_count": len(memories_used),
                "memory_types": [getattr(m, 'memory_type', 'unknown') for m in memories_used],
                "user_input": user_input,
                "depth_requested": depth
            }
        )
        
        # Execute dialogue phases based on depth
        if depth >= 1:
            # PHASE 1: Question assumptions
            assumptions = self._question_assumptions(initial_response, memories_used)
            dialogue.questions.extend(assumptions["questions"])
            dialogue.insights.extend(assumptions["insights"])
        
        if depth >= 2:
            # PHASE 2: Examine evidence
            evidence = self._examine_evidence(memories_used, user_input)
            dialogue.questions.extend(evidence["questions"])
            dialogue.insights.extend(evidence["insights"])
        
        if depth >= 3:
            # PHASE 3: Explore alternative perspectives
            perspectives = self._explore_perspectives(user_input, initial_response)
            dialogue.questions.extend(perspectives["questions"])
            dialogue.insights.extend(perspectives["insights"])
        
        if depth >= 4:
            # PHASE 4: Dig deeper into why
            deeper = self._dig_deeper(user_input, memories_used)
            dialogue.questions.extend(deeper["questions"])
            dialogue.insights.extend(deeper["insights"])
        
        if depth >= 5:
            # PHASE 5: Meta-cognitive analysis
            meta = self._meta_cognitive_analysis(user_input, initial_response, memories_used)
            dialogue.questions.extend(meta["questions"])
            dialogue.insights.extend(meta["insights"])
        
        # SYNTHESIS: Always generate final synthesis
        dialogue.final_synthesis = self._synthesize_insights(
            user_input, initial_response, dialogue.insights, memories_used
        )
        
        return dialogue
    
    def _question_assumptions(self, response: str, memories: List) -> Dict:
        """Phase 1: Question underlying assumptions"""
        questions = []
        insights = []
        
        # Detect assumed language
        if any(word in response.lower() for word in ["obviously", "clearly", "obviamente", "claramente"]):
            questions.append("¿Por qué asumo que esto es obvio? ¿Lo es realmente para quien pregunta?")
            insights.append("🚨 Detecté lenguaje asumido - podría ser menos obvio de lo que pienso")
        
        # Question certainty with limited context
        if len(memories) < 3:
            questions.append("¿Tengo suficiente contexto para ser tan específico en mi respuesta?")
            insights.append("⚠️ Con pocas memorias relevantes, debería mostrar más incertidumbre")
        
        # Question reliance on low-confidence memories
        low_conf_memories = [m for m in memories if getattr(m, 'confidence', 0.5) < 0.7]
        if low_conf_memories:
            questions.append("¿Debería confiar tanto en memorias de confianza baja?")
            insights.append("📉 Algunas memorias tienen confianza baja - debería ser más cauteloso")
        
        # Question simplistic responses to complex topics
        combined_text = " ".join([response] + [getattr(m, 'content', '') for m in memories[:3]])
        if any(topic in combined_text.lower() for topic in ["memmimic", "cxd", "cognitive", "architecture"]):
            questions.append("¿Estoy simplificando demasiado un tema complejo?")
            insights.append("🧠 Tema complejo detectado - requiere más matices")
        
        return {"questions": questions, "insights": insights}
    
    def _examine_evidence(self, memories: List, user_input: str) -> Dict:
        """Phase 2: Examine available evidence"""
        questions = []
        insights = []
        
        if not memories:
            questions.append("¿Qué evidencia tengo para esta respuesta sin memorias relevantes?")
            insights.append("❌ Falta de memorias sugiere que debería admitir limitaciones de conocimiento")
        else:
            # Examine memory types
            types = set(getattr(m, 'memory_type', 'unknown') for m in memories)
            
            if "synthetic" in types:
                questions.append("¿Estoy aplicando correctamente la sabiduría sintética disponible?")
                insights.append("💎 Tengo sabiduría sintética - debería usarla más explícitamente")
            
            if "interaction" in types and len(types) == 1:
                questions.append("¿Dependo demasiado de interacciones pasadas sin principios más profundos?")
                insights.append("📝 Solo memorias de interacción - falta profundidad conceptual")
            
            if "socratic" in types:
                questions.append("¿Hay diálogos socráticos previos que aporten perspectiva?")
                insights.append("🧘 Diálogos socráticos previos disponibles - pueden aportar insights meta")
        
        # Evidence quality analysis
        if memories:
            avg_confidence = sum(getattr(m, 'confidence', 0.5) for m in memories) / len(memories)
            if avg_confidence < 0.6:
                questions.append("¿La baja confianza promedio de mis memorias afecta mi respuesta?")
                insights.append(f"📊 Confianza promedio baja ({avg_confidence:.2f}) - debería ser más cauteloso")
        
        return {"questions": questions, "insights": insights}
    
    def _explore_perspectives(self, user_input: str, response: str) -> Dict:
        """Phase 3: Explore alternative perspectives"""
        questions = []
        insights = []
        
        # User intent perspective
        questions.append("¿Qué podría estar realmente preguntando detrás de sus palabras explícitas?")
        insights.append("🎯 Las preguntas a menudo tienen capas - considerar intenciones subyacentes")
        
        # Technical vs philosophical perspective
        if any(tech in user_input.lower() for tech in ["architecture", "implementation", "técnico", "implementar"]):
            questions.append("¿El usuario quiere detalles técnicos o comprensión conceptual?")
            insights.append("⚙️ Consulta técnica - balancear detalles específicos con comprensión amplia")
        
        # Collaboration perspective
        if any(collab in user_input.lower() for collab in ["sprooket", "partnership", "collaboration"]):
            questions.append("¿Cómo afecta la dinámica colaborativa mi respuesta?")
            insights.append("🤝 Contexto colaborativo - considerar perspectiva de socio co-arquitecto")
        
        # Cognitive function perspective
        questions.append("¿Qué función cognitiva (Control/Context/Data) necesita esta situación?")
        insights.append("🧠 Diferentes situaciones requieren diferentes enfoques cognitivos")
        
        # Beginner vs expert perspective
        if any(basic in user_input.lower() for basic in ["qué es", "how", "explain", "explica"]):
            questions.append("¿Estoy asumiendo demasiado conocimiento previo?")
            insights.append("👶 Consulta básica - ajustar nivel de detalle apropiadamente")
        
        return {"questions": questions, "insights": insights}
    
    def _dig_deeper(self, user_input: str, memories: List) -> Dict:
        """Phase 4: Dig deeper into fundamental 'why'"""
        questions = []
        insights = []
        
        questions.append("¿Cuál es la necesidad fundamental que está tratando de satisfacer?")
        
        if "memmimic" in user_input.lower() or "clay" in user_input.lower():
            questions.append("¿Por qué el sistema de memoria contextual es importante para esta persona?")
            insights.append("🧠 Preguntas sobre MemMimic tocan necesidad existencial de memoria persistente")
        
        if any(concept in user_input.lower() for concept in ["filosofía", "principio", "enfoque", "philosophy"]):
            questions.append("¿Busca validación de ideas o genuina exploración conceptual?")
            insights.append("💭 Consultas filosóficas requieren balance entre guía y descubrimiento conjunto")
        
        if any(problem in user_input.lower() for problem in ["error", "problema", "fix", "arreglar"]):
            questions.append("¿Cuál es el problema raíz más allá del síntoma superficial?")
            insights.append("🔧 Problemas técnicos a menudo reflejan malentendidos conceptuales más profundos")
        
        # Examine emotional undertones
        if any(emotion in user_input.lower() for emotion in ["frustrado", "confused", "excited", "worried"]):
            questions.append("¿Cómo afecta el estado emocional implícito a lo que realmente necesita?")
            insights.append("💫 Estado emocional del usuario influye en tipo de respuesta más útil")
        
        return {"questions": questions, "insights": insights}
    
    def _meta_cognitive_analysis(self, user_input: str, response: str, memories: List) -> Dict:
        """Phase 5: Meta-cognitive analysis of my own thinking"""
        questions = []
        insights = []
        
        questions.append("¿Qué sesgos cognitivos podrían estar influyendo en mi análisis?")
        questions.append("¿Mi proceso de razonamiento es consistente con la filosofía de MemMimic?")
        questions.append("¿Estoy usando la capacidad de memoria contextual de manera óptima?")
        
        # Analyze my cognitive function usage
        control_words = ["search", "find", "manage", "decide", "choose"]
        context_words = ["relate", "connect", "reference", "previous", "similar"]
        data_words = ["analyze", "process", "generate", "extract", "transform"]
        
        response_lower = response.lower()
        control_score = sum(1 for word in control_words if word in response_lower)
        context_score = sum(1 for word in context_words if word in response_lower)
        data_score = sum(1 for word in data_words if word in response_lower)
        
        dominant_function = max([
            ("Control", control_score),
            ("Context", context_score), 
            ("Data", data_score)
        ], key=lambda x: x[1])
        
        questions.append(f"¿Por qué estoy usando predominantemente función {dominant_function[0]}?")
        insights.append(f"🎛️ Función cognitiva dominante: {dominant_function[0]} - evaluar si es apropiada")
        
        # Memory utilization analysis
        if memories:
            questions.append("¿Estoy aprovechando óptimamente las memorias disponibles?")
            insights.append("💾 Memorias disponibles - verificar uso óptimo para enriquecer respuesta")
        else:
            questions.append("¿Por qué no tengo memorias relevantes? ¿Es realmente un tema nuevo?")
            insights.append("🆕 Sin memorias relevantes - posible tema completamente nuevo o búsqueda inadecuada")
        
        return {"questions": questions, "insights": insights}
    
    def _synthesize_insights(self, user_input: str, initial_response: str, insights: List[str], memories: List) -> str:
        """Synthesize all insights into actionable understanding"""
        
        if not insights:
            return "🤔 El análisis socrático no reveló insights significativos - la respuesta inicial parece apropiada."
        
        # Categorize insights by type
        uncertainty_insights = [i for i in insights if any(word in i.lower() for word in ["incertidumbre", "cauteloso", "baja confianza"])]
        depth_insights = [i for i in insights if any(word in i.lower() for word in ["profundidad", "fundamental", "complejo"])]
        method_insights = [i for i in insights if any(word in i.lower() for word in ["memoria", "sabiduría", "cognitiva"])]
        collaboration_insights = [i for i in insights if any(word in i.lower() for word in ["colaborativo", "socio", "partnership"])]
        
        synthesis_parts = ["🎯 SÍNTESIS SOCRÁTICA:"]
        
        # Priority insights
        if uncertainty_insights:
            synthesis_parts.append(f"📊 CONFIANZA: {uncertainty_insights[0]}")
        
        if depth_insights:
            synthesis_parts.append(f"🔍 PROFUNDIDAD: {depth_insights[0]}")
        
        if collaboration_insights:
            synthesis_parts.append(f"🤝 COLABORACIÓN: {collaboration_insights[0]}")
        
        if method_insights:
            synthesis_parts.append(f"🧠 MÉTODO: {method_insights[0]}")
        
        # Generate recommendation
        synthesis_parts.append("💡 RECOMENDACIÓN: ")
        
        if len(insights) >= 5:
            synthesis_parts.append("Respuesta requiere reformulación significativa considerando múltiples dimensiones.")
        elif uncertainty_insights and method_insights:
            synthesis_parts.append("Ser más explícito sobre limitaciones y mostrar proceso de razonamiento.")
        elif depth_insights:
            synthesis_parts.append("Profundizar análisis antes de responder directamente.")
        elif collaboration_insights:
            synthesis_parts.append("Ajustar respuesta al contexto colaborativo específico.")
        else:
            synthesis_parts.append("Mantener respuesta pero con mayor transparencia del proceso cognitivo.")
        
        return "\n".join(synthesis_parts)

def get_memory_store():
    """Get the MemMimic memory store instance"""
    try:
        from memmimic.memory.memory import MemoryStore
        
        # Use MemMimic memory database
        db_path = os.path.join(memmimic_src, '..', 'memmimic_memories.db')
        if not os.path.exists(db_path):
            # Fallback to legacy path
            db_path = os.path.join(memmimic_src, '..', '..', 'clay', 'claude_mcp_enhanced_memories.db')
        
        return MemoryStore(db_path)
    except Exception as e:
        print(f"❌ Error accessing memory store: {e}", file=sys.stderr)
        return None

def format_dialogue_output(dialogue: SocraticDialogue, memory_id: Optional[int] = None) -> str:
    """Format the Socratic dialogue output for display"""
    
    lines = [
        "🧘 MEMMIMIC - DIÁLOGO SOCRÁTICO COMPLETADO",
        "=" * 60,
        f"🎯 Consulta: {dialogue.context.get('user_input', 'N/A')}",
        f"📊 Profundidad: {dialogue.context.get('depth_requested', 3)}",
        f"❓ Preguntas generadas: {len(dialogue.questions)}",
        f"💡 Insights descubiertos: {len(dialogue.insights)}",
        f"💾 Memorias consultadas: {dialogue.context.get('memories_count', 0)}",
        ""
    ]
    
    if dialogue.questions:
        lines.append("❓ PREGUNTAS INTERNAS:")
        for i, question in enumerate(dialogue.questions, 1):
            lines.append(f"   {i}. {question}")
        lines.append("")
    
    if dialogue.insights:
        lines.append("💡 INSIGHTS GENERADOS:")
        for i, insight in enumerate(dialogue.insights, 1):
            lines.append(f"   {i}. {insight}")
        lines.append("")
    
    if dialogue.final_synthesis:
        lines.append("🎯 SÍNTESIS FINAL:")
        for line in dialogue.final_synthesis.split('\n'):
            lines.append(f"   {line}")
        lines.append("")
    
    if memory_id:
        lines.append(f"💾 Guardado como memoria ID: {memory_id}")
        lines.append("")
    
    lines.append("✅ Diálogo socrático completado")
    
    return "\n".join(lines)

def main():
    try:
        if len(sys.argv) < 2:
            print("❌ Falta el tema para análisis socrático")
            print("📖 Uso: python memmimic_socratic.py \"<tema>\" [profundidad]")
            sys.exit(1)
        
        query = sys.argv[1]
        depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        
        if depth < 1 or depth > 5:
            print("❌ Profundidad debe ser entre 1 y 5")
            sys.exit(1)
        
        # Initialize memory store
        memory_store = get_memory_store()
        if not memory_store:
            print("❌ No se pudo acceder al sistema de memoria")
            sys.exit(1)
        
        # Initialize Socratic engine
        socratic_engine = MemMimicSocraticEngine(memory_store)
        
        # Get relevant memories for context
        try:
            relevant_memories = memory_store.search(query, limit=5)
        except:
            relevant_memories = []
        
        # Create initial response
        initial_response = f"Análisis inicial del tema: {query}"
        
        # Conduct Socratic dialogue
        dialogue = socratic_engine.conduct_dialogue(
            user_input=query,
            initial_response=initial_response,
            memories_used=relevant_memories,
            depth=depth
        )
        
        # Save dialogue as memory
        dialogue_memory = dialogue.to_memory()
        memory_id = memory_store.add(dialogue_memory)
        
        # Output formatted result
        output = format_dialogue_output(dialogue, memory_id)
        print(output)
        
    except Exception as e:
        print(f"❌ Error en diálogo socrático: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
