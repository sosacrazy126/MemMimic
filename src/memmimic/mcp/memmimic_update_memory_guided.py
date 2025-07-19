#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Update Memory Guided Tool - Socratic Memory Enhancement
Professional memory updating with Socratic questioning guidance
"""

import argparse
import os
import sys

# Force UTF-8 I/O for cross-platform compatibility
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from memmimic.assistant import ContextualAssistant
except ImportError as e:
    print(f"❌ Error importing MemMimic: {e}", file=sys.stderr)
    print("❌ Error: Cannot import MemMimic core")
    sys.exit(1)


def find_memory_by_id(memory_store, memory_id):
    """Find memory by ID with error handling"""
    try:
        all_memories = memory_store.get_all()

        for memory in all_memories:
            if hasattr(memory, "id") and memory.id == memory_id:
                return memory

        return None

    except Exception as e:
        print(f"⚠️ Error searching for memory: {e}", file=sys.stderr)
        return None


def analyze_memory_context(memory, assistant):
    """Analyze memory for Socratic questioning"""
    try:
        content = getattr(memory, "content", "")
        mem_type = getattr(memory, "memory_type", "unknown")
        confidence = getattr(memory, "confidence", 0.0)
        created_at = getattr(memory, "created_at", "unknown")

        # Generate Socratic questions based on memory type and content
        questions = []

        if mem_type == "interaction":
            questions.extend(
                [
                    "What was the key insight from this interaction?",
                    "How does this connect to previous conversations?",
                    "What could be improved or clarified?",
                ]
            )
        elif mem_type == "reflection":
            questions.extend(
                [
                    "What patterns does this reflection reveal?",
                    "How has my understanding evolved since this?",
                    "What deeper questions does this raise?",
                ]
            )
        elif mem_type == "milestone":
            questions.extend(
                [
                    "What made this moment significant?",
                    "What did I learn from this achievement?",
                    "How can this inform future decisions?",
                ]
            )
        else:
            questions.extend(
                [
                    "What is the core value of this memory?",
                    "How does this relate to my overall development?",
                    "What context is missing?",
                ]
            )

        # Content-based questions
        if len(content) < 50:
            questions.append("Could this memory be expanded with more detail?")

        if confidence < 0.7:
            questions.append("What would increase the confidence in this memory?")

        return {
            "questions": questions,
            "suggestions": [
                "Consider adding more context or detail",
                "Link to related memories or concepts",
                "Clarify the significance or impact",
                "Update confidence level if appropriate",
            ],
        }

    except Exception as e:
        return {
            "questions": ["What is the core meaning of this memory?"],
            "suggestions": ["Consider basic improvements"],
            "error": str(e),
        }


def main():
    """Guided memory update with Socratic questioning"""

    parser = argparse.ArgumentParser(description="MemMimic Update Memory Guided")
    parser.add_argument("memory_id", type=int, help="ID of memory to update")

    args = parser.parse_args()

    try:
        # Initialize MemMimic assistant
        assistant = ContextualAssistant("memmimic")
        memory_store = assistant.memory_store

        # Find the memory
        memory = find_memory_by_id(memory_store, args.memory_id)

        if not memory:
            print(f"❌ Memory not found: ID #{args.memory_id}")
            print("")
            print("💡 SUGGESTIONS:")
            print("  status()                  # Check total memories")
            print("  analyze_memory_patterns() # Find memory IDs")
            print("  recall('search_term')     # Search for specific memories")
            sys.exit(1)

        # Extract memory details
        content = getattr(memory, "content", "")
        mem_type = getattr(memory, "memory_type", "unknown")
        confidence = getattr(memory, "confidence", 0.0)
        created_at = getattr(memory, "created_at", "unknown")

        # Analyze memory for guidance
        analysis = analyze_memory_context(memory, assistant)

        # Display memory and analysis
        result_parts = []

        # Header
        result_parts.append(f"🔍 GUIDED MEMORY UPDATE: #{args.memory_id}")
        result_parts.append("=" * 60)
        result_parts.append("")

        # Current memory details
        result_parts.append("📝 CURRENT MEMORY:")
        result_parts.append(f"  Type: {mem_type}")
        result_parts.append(f"  Confidence: {confidence:.2f}")
        result_parts.append(f"  Created: {created_at}")
        result_parts.append(f"  Content: {content}")
        result_parts.append("")

        # Socratic questions
        result_parts.append("🤔 SOCRATIC QUESTIONS FOR REFLECTION:")
        for i, question in enumerate(analysis["questions"], 1):
            result_parts.append(f"  {i}. {question}")
        result_parts.append("")

        # Improvement suggestions
        result_parts.append("💡 IMPROVEMENT SUGGESTIONS:")
        for suggestion in analysis["suggestions"]:
            result_parts.append(f"  • {suggestion}")
        result_parts.append("")

        # Update instructions
        result_parts.append("🔧 TO UPDATE THIS MEMORY:")
        result_parts.append("This tool provides analysis only. To actually update:")
        result_parts.append("")
        result_parts.append("1. Consider the Socratic questions above")
        result_parts.append("2. Reflect on potential improvements")
        result_parts.append("3. Use remember() to create an updated version")
        result_parts.append(
            "4. Use delete_memory_guided() to remove old version if needed"
        )
        result_parts.append("")
        result_parts.append("📝 EXAMPLE UPDATE PROCESS:")
        result_parts.append(
            "  remember('Enhanced: [your improved content]', 'interaction')"
        )
        result_parts.append(f"  delete_memory_guided({args.memory_id}, confirm=True)")
        result_parts.append("")

        # Related actions
        result_parts.append("🔗 RELATED ACTIONS:")
        result_parts.append(
            "  socratic_dialogue('memory_topic')     # Deep questioning"
        )
        result_parts.append(
            "  analyze_memory_patterns()             # Pattern analysis"
        )
        result_parts.append(
            "  context_tale('memory_context')        # Generate narrative"
        )
        result_parts.append("")
        result_parts.append("🧠 MemMimic - Socratic memory enhancement")

        print("\n".join(result_parts))

    except Exception as e:
        print(f"❌ Critical error in guided update: {str(e)}", file=sys.stderr)
        print(f"❌ Guided update failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
