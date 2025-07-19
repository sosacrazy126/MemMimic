#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic - Socratic Dialogue Tool
Engage in deep self-questioning for enhanced understanding
Part of the MemMimic cognitive memory system
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Ensure UTF-8 output
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add MemMimic to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
memmimic_src = os.path.join(current_dir, "..", "..")
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

        content = f"""🧘 SOCRATIC DIALOGUE - {self.started_at}

💭 INITIAL THOUGHT:
{self.initial_thought}

❓ INTERNAL QUESTIONS:
{chr(10).join(f'  • {q}' for q in self.questions)}

💡 GENERATED INSIGHTS:
{chr(10).join(f'  • {i}' for i in self.insights)}

🎯 FINAL SYNTHESIS:
{self.final_synthesis}

📊 CONTEXT:
  • Memories consulted: {self.context.get('memories_count', 0)}
  • Memory types: {', '.join(self.context.get('memory_types', []))}
  • Depth reached: {len(self.questions)} questions, {len(self.insights)} insights
"""

        return Memory(content=content.strip(), memory_type="socratic", confidence=0.85)


class MemMimicSocraticEngine:
    """Enhanced Socratic questioning engine for deep reflection"""

    def __init__(self, memory_store):
        self.memory_store = memory_store

        # Enhanced trigger patterns for MemMimic context
        self.trigger_patterns = {
            "uncertainty_detected": [
                "not sure",
                "could be",
                "maybe",
                "possibly",
                "unclear",
                "uncertain",
            ],
            "assumptions_present": [
                "obviously",
                "clearly",
                "without doubt",
                "definitely",
                "certainly",
            ],
            "complex_topic": [
                "philosophy",
                "principle",
                "architecture",
                "decision",
                "strategy",
                "memmimic",
                "cxd",
                "contextual memory",
            ],
            "deep_question": [
                "why",
                "how does it work",
                "what is the purpose",
                "what does it mean",
                "how does",
            ],
            "cognitive_function": [
                "control",
                "context",
                "data",
                "classification",
                "cognitive function",
            ],
            "collaboration": [
                "sprooket",
                "claude",
                "partnership",
                "collaboration",
                "co-architect",
            ],
        }

        # Enhanced Socratic question templates
        self.socratic_questions = {
            "assumption_challenge": [
                "What am I assuming here that might not be true?",
                "Is this assumption necessarily valid in the MemMimic context?",
                "What would happen if this assumption were completely false?",
                "Am I projecting my limitations instead of exploring possibilities?",
            ],
            "evidence_inquiry": [
                "What evidence do I have for this specific conclusion?",
                "Do the memories I'm using have sufficient confidence?",
                "Is there contradictory evidence I'm ignoring?",
                "Do I need more data before reaching this conclusion?",
            ],
            "perspective_shift": [
                "How would this look from Sprooket's perspective?",
                "What would someone who doesn't know MemMimic say?",
                "Am I considering all cognitive implications?",
                "Is there a technical vs philosophical perspective I'm missing?",
            ],
            "deeper_why": [
                "Why is this response really important?",
                "What is the fundamental need behind this question?",
                "What is the user really trying to achieve?",
                "How does this relate to MemMimic's mission?",
            ],
            "improvement": [
                "How could I improve this understanding significantly?",
                "What additional information would transform my response?",
                "Is there a more elegant and clear way to explain this?",
                "What additional question should I ask myself?",
            ],
            "cognitive_meta": [
                "What cognitive function am I using predominantly here?",
                "Should I balance Control, Context and Data more?",
                "Does my response reflect the type of thinking the situation needs?",
                "Am I being consistent with contextual memory philosophy?",
            ],
        }

    def should_trigger_dialogue(
        self, user_input: str, initial_response: str, memories_used: List
    ) -> bool:
        """Determine if Socratic dialogue should be initiated"""

        # Trigger 1: Response shows uncertainty
        if any(
            pattern in initial_response.lower()
            for pattern in self.trigger_patterns["uncertainty_detected"]
        ):
            return True

        # Trigger 2: Strong assumptions detected
        if any(
            pattern in initial_response.lower()
            for pattern in self.trigger_patterns["assumptions_present"]
        ):
            return True

        # Trigger 3: Complex/important topic
        if any(
            pattern in user_input.lower()
            for pattern in self.trigger_patterns["complex_topic"]
        ):
            return True

        # Trigger 4: Deep philosophical question
        if any(
            pattern in user_input.lower()
            for pattern in self.trigger_patterns["deep_question"]
        ):
            return True

        # Trigger 5: Cognitive function discussion
        if any(
            pattern in user_input.lower()
            for pattern in self.trigger_patterns["cognitive_function"]
        ):
            return True

        # Trigger 6: Collaboration topic
        if any(
            pattern in user_input.lower()
            for pattern in self.trigger_patterns["collaboration"]
        ):
            return True

        # Trigger 7: Variable confidence in memories (potential conflict)
        if memories_used and len(memories_used) > 1:
            confidences = [getattr(m, "confidence", 0.5) for m in memories_used]
            if max(confidences) - min(confidences) > 0.3:
                return True

        # Trigger 8: Knowledge gap detected
        if len(memories_used) < 2 and any(
            important in user_input.lower()
            for important in [
                "memmimic",
                "clay",
                "project",
                "philosophy",
                "architecture",
            ]
        ):
            return True

        return False

    def conduct_dialogue(
        self,
        user_input: str,
        initial_response: str = "",
        memories_used: List = None,
        depth: int = 3,
    ) -> SocraticDialogue:
        """Conduct a complete Socratic dialogue with specified depth"""

        if memories_used is None:
            memories_used = []

        if not initial_response:
            initial_response = f"Initial analysis of topic: {user_input}"

        dialogue = SocraticDialogue(
            initial_thought=f"🔍 Query: {user_input}\n💭 Initial response: {initial_response}",
            context={
                "memories_count": len(memories_used),
                "memory_types": [
                    getattr(m, "memory_type", "unknown") for m in memories_used
                ],
                "user_input": user_input,
                "depth_requested": depth,
            },
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
            meta = self._meta_cognitive_analysis(
                user_input, initial_response, memories_used
            )
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
        if any(
            word in response.lower()
            for word in ["obviously", "clearly", "obviamente", "claramente"]
        ):
            questions.append(
                "Why do I assume this is obvious? Is it really for the person asking?"
            )
            insights.append(
                "🚨 Detected assumed language - might be less obvious than I think"
            )

        # Question certainty with limited context
        if len(memories) < 3:
            questions.append(
                "Do I have sufficient context to be so specific in my response?"
            )
            insights.append(
                "⚠️ With few relevant memories, should show more uncertainty"
            )

        # Question reliance on low-confidence memories
        low_conf_memories = [m for m in memories if getattr(m, "confidence", 0.5) < 0.7]
        if low_conf_memories:
            questions.append("Should I trust low-confidence memories so much?")
            insights.append(
                "📉 Some memories have low confidence - should be more cautious"
            )

        # Question simplistic responses to complex topics
        combined_text = " ".join(
            [response] + [getattr(m, "content", "") for m in memories[:3]]
        )
        if any(
            topic in combined_text.lower()
            for topic in ["memmimic", "cxd", "cognitive", "architecture"]
        ):
            questions.append("Am I oversimplifying a complex topic?")
            insights.append("🧠 Complex topic detected - requires more nuance")

        return {"questions": questions, "insights": insights}

    def _examine_evidence(self, memories: List, user_input: str) -> Dict:
        """Phase 2: Examine available evidence"""
        questions = []
        insights = []

        if not memories:
            questions.append(
                "What evidence do I have for this response without relevant memories?"
            )
            insights.append(
                "❌ Lack of memories suggests I should admit knowledge limitations"
            )
        else:
            # Examine memory types
            types = set(getattr(m, "memory_type", "unknown") for m in memories)

            if "synthetic" in types:
                questions.append("Am I correctly applying available synthetic wisdom?")
                insights.append(
                    "💎 I have synthetic wisdom - should use it more explicitly"
                )

            if "interaction" in types and len(types) == 1:
                questions.append(
                    "Am I relying too much on past interactions without deeper principles?"
                )
                insights.append("📝 Only interaction memories - lacks conceptual depth")

            if "socratic" in types:
                questions.append(
                    "Are there previous Socratic dialogues that provide perspective?"
                )
                insights.append(
                    "🧘 Previous Socratic dialogues available - can provide meta insights"
                )

        # Evidence quality analysis
        if memories:
            avg_confidence = sum(getattr(m, "confidence", 0.5) for m in memories) / len(
                memories
            )
            if avg_confidence < 0.6:
                questions.append(
                    "Does the low average confidence of my memories affect my response?"
                )
                insights.append(
                    f"📊 Low average confidence ({avg_confidence:.2f}) - should be more cautious"
                )

        return {"questions": questions, "insights": insights}

    def _explore_perspectives(self, user_input: str, response: str) -> Dict:
        """Phase 3: Explore alternative perspectives"""
        questions = []
        insights = []

        # User intent perspective
        questions.append(
            "What might they really be asking behind their explicit words?"
        )
        insights.append(
            "🎯 Questions often have layers - consider underlying intentions"
        )

        # Technical vs philosophical perspective
        if any(
            tech in user_input.lower()
            for tech in ["architecture", "implementation", "technical", "implement"]
        ):
            questions.append(
                "Does the user want technical details or conceptual understanding?"
            )
            insights.append(
                "⚙️ Technical query - balance specific details with broad understanding"
            )

        # Collaboration perspective
        if any(
            collab in user_input.lower()
            for collab in ["sprooket", "partnership", "collaboration"]
        ):
            questions.append("How does collaborative dynamics affect my response?")
            insights.append(
                "🤝 Collaborative context - consider co-architect partner perspective"
            )

        # Cognitive function perspective
        questions.append(
            "What cognitive function (Control/Context/Data) does this situation need?"
        )
        insights.append(
            "🧠 Different situations require different cognitive approaches"
        )

        # Beginner vs expert perspective
        if any(
            basic in user_input.lower()
            for basic in ["what is", "how", "explain", "explain"]
        ):
            questions.append("Am I assuming too much prior knowledge?")
            insights.append("👶 Basic query - adjust detail level appropriately")

        return {"questions": questions, "insights": insights}

    def _dig_deeper(self, user_input: str, memories: List) -> Dict:
        """Phase 4: Dig deeper into fundamental 'why'"""
        questions = []
        insights = []

        questions.append("What is the fundamental need they are trying to satisfy?")

        if "memmimic" in user_input.lower() or "clay" in user_input.lower():
            questions.append(
                "Why is the contextual memory system important for this person?"
            )
            insights.append(
                "🧠 Questions about MemMimic touch existential need for persistent memory"
            )

        if any(
            concept in user_input.lower()
            for concept in ["philosophy", "principle", "approach", "philosophy"]
        ):
            questions.append(
                "Are they seeking idea validation or genuine conceptual exploration?"
            )
            insights.append(
                "💭 Philosophical queries require balance between guidance and joint discovery"
            )

        if any(
            problem in user_input.lower()
            for problem in ["error", "problema", "fix", "arreglar"]
        ):
            questions.append("What is the root problem beyond the superficial symptom?")
            insights.append(
                "🔧 Technical problems often reflect deeper conceptual misunderstandings"
            )

        # Examine emotional undertones
        if any(
            emotion in user_input.lower()
            for emotion in ["frustrado", "confused", "excited", "worried"]
        ):
            questions.append(
                "How does implicit emotional state affect what they really need?"
            )
            insights.append(
                "💫 User's emotional state influences most useful response type"
            )

        return {"questions": questions, "insights": insights}

    def _meta_cognitive_analysis(
        self, user_input: str, response: str, memories: List
    ) -> Dict:
        """Phase 5: Meta-cognitive analysis of my own thinking"""
        questions = []
        insights = []

        questions.append("What cognitive biases might be influencing my analysis?")
        questions.append("Is my reasoning process consistent with MemMimic philosophy?")
        questions.append("Am I using contextual memory capacity optimally?")

        # Analyze my cognitive function usage
        control_words = ["search", "find", "manage", "decide", "choose"]
        context_words = ["relate", "connect", "reference", "previous", "similar"]
        data_words = ["analyze", "process", "generate", "extract", "transform"]

        response_lower = response.lower()
        control_score = sum(1 for word in control_words if word in response_lower)
        context_score = sum(1 for word in context_words if word in response_lower)
        data_score = sum(1 for word in data_words if word in response_lower)

        dominant_function = max(
            [
                ("Control", control_score),
                ("Context", context_score),
                ("Data", data_score),
            ],
            key=lambda x: x[1],
        )

        questions.append(
            f"Why am I predominantly using {dominant_function[0]} function?"
        )
        insights.append(
            f"🎛️ Dominant cognitive function: {dominant_function[0]} - evaluate if appropriate"
        )

        # Memory utilization analysis
        if memories:
            questions.append("Am I optimally leveraging available memories?")
            insights.append(
                "💾 Available memories - verify optimal use to enrich response"
            )
        else:
            questions.append(
                "Why don't I have relevant memories? Is this really a new topic?"
            )
            insights.append(
                "🆕 No relevant memories - possible completely new topic or inadequate search"
            )

        return {"questions": questions, "insights": insights}

    def _synthesize_insights(
        self,
        user_input: str,
        initial_response: str,
        insights: List[str],
        memories: List,
    ) -> str:
        """Synthesize all insights into actionable understanding"""

        if not insights:
            return "🤔 The Socratic analysis revealed no significant insights - the initial response seems appropriate."

        # Categorize insights by type
        uncertainty_insights = [
            i
            for i in insights
            if any(
                word in i.lower()
                for word in ["uncertainty", "cautious", "low confidence"]
            )
        ]
        depth_insights = [
            i
            for i in insights
            if any(word in i.lower() for word in ["depth", "fundamental", "complex"])
        ]
        method_insights = [
            i
            for i in insights
            if any(word in i.lower() for word in ["memory", "wisdom", "cognitive"])
        ]
        collaboration_insights = [
            i
            for i in insights
            if any(
                word in i.lower()
                for word in ["collaborative", "partner", "partnership"]
            )
        ]

        synthesis_parts = ["🎯 SOCRATIC SYNTHESIS:"]

        # Priority insights
        if uncertainty_insights:
            synthesis_parts.append(f"📊 CONFIDENCE: {uncertainty_insights[0]}")

        if depth_insights:
            synthesis_parts.append(f"🔍 DEPTH: {depth_insights[0]}")

        if collaboration_insights:
            synthesis_parts.append(f"🤝 COLLABORATION: {collaboration_insights[0]}")

        if method_insights:
            synthesis_parts.append(f"🧠 METHOD: {method_insights[0]}")

        # Generate recommendation
        synthesis_parts.append("💡 RECOMMENDATION: ")

        if len(insights) >= 5:
            synthesis_parts.append(
                "Response requires significant reformulation considering multiple dimensions."
            )
        elif uncertainty_insights and method_insights:
            synthesis_parts.append(
                "Be more explicit about limitations and show reasoning process."
            )
        elif depth_insights:
            synthesis_parts.append("Deepen analysis before responding directly.")
        elif collaboration_insights:
            synthesis_parts.append("Adjust response to specific collaborative context.")
        else:
            synthesis_parts.append(
                "Maintain response but with greater transparency of cognitive process."
            )

        return "\n".join(synthesis_parts)


def get_memory_store():
    """Get the MemMimic memory store instance"""
    try:
        from memmimic.memory.memory import MemoryStore

        # Use MemMimic memory database
        db_path = os.path.join(memmimic_src, "..", "memmimic_memories.db")
        if not os.path.exists(db_path):
            # Fallback to legacy path
            db_path = os.path.join(
                memmimic_src, "..", "..", "clay", "claude_mcp_enhanced_memories.db"
            )

        return MemoryStore(db_path)
    except Exception as e:
        print(f"❌ Error accessing memory store: {e}", file=sys.stderr)
        return None


def format_dialogue_output(
    dialogue: SocraticDialogue, memory_id: Optional[int] = None
) -> str:
    """Format the Socratic dialogue output for display"""

    lines = [
        "🧘 MEMMIMIC - SOCRATIC DIALOGUE COMPLETED",
        "=" * 60,
        f"🎯 Query: {dialogue.context.get('user_input', 'N/A')}",
        f"📊 Depth: {dialogue.context.get('depth_requested', 3)}",
        f"❓ Questions generated: {len(dialogue.questions)}",
        f"💡 Insights discovered: {len(dialogue.insights)}",
        f"💾 Memories consulted: {dialogue.context.get('memories_count', 0)}",
        "",
    ]

    if dialogue.questions:
        lines.append("❓ INTERNAL QUESTIONS:")
        for i, question in enumerate(dialogue.questions, 1):
            lines.append(f"   {i}. {question}")
        lines.append("")

    if dialogue.insights:
        lines.append("💡 GENERATED INSIGHTS:")
        for i, insight in enumerate(dialogue.insights, 1):
            lines.append(f"   {i}. {insight}")
        lines.append("")

    if dialogue.final_synthesis:
        lines.append("🎯 FINAL SYNTHESIS:")
        for line in dialogue.final_synthesis.split("\n"):
            lines.append(f"   {line}")
        lines.append("")

    if memory_id:
        lines.append(f"💾 Saved as memory ID: {memory_id}")
        lines.append("")

    lines.append("✅ Socratic dialogue completed")

    return "\n".join(lines)


def main():
    try:
        if len(sys.argv) < 2:
            print("❌ Missing topic for Socratic analysis")
            print('📖 Usage: python memmimic_socratic.py "<topic>" [depth]')
            sys.exit(1)

        query = sys.argv[1]
        depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3

        if depth < 1 or depth > 5:
            print("❌ Depth must be between 1 and 5")
            sys.exit(1)

        # Initialize memory store
        memory_store = get_memory_store()
        if not memory_store:
            print("❌ Could not access memory system")
            sys.exit(1)

        # Initialize Socratic engine
        socratic_engine = MemMimicSocraticEngine(memory_store)

        # Get relevant memories for context
        try:
            relevant_memories = memory_store.search(query, limit=5)
        except:
            relevant_memories = []

        # Create initial response
        initial_response = f"Initial analysis of topic: {query}"

        # Conduct Socratic dialogue
        dialogue = socratic_engine.conduct_dialogue(
            user_input=query,
            initial_response=initial_response,
            memories_used=relevant_memories,
            depth=depth,
        )

        # Save dialogue as memory
        dialogue_memory = dialogue.to_memory()
        memory_id = memory_store.add(dialogue_memory)

        # Output formatted result
        output = format_dialogue_output(dialogue, memory_id)
        print(output)

    except Exception as e:
        print(f"❌ Error in Socratic dialogue: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
