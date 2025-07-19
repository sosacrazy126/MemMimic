#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Context Tale Tool - Memory Narrative Generation
Professional narrative generation from memory fragments
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
    from memmimic import ContextualAssistant
except ImportError as e:
    print(f"❌ Error importing MemMimic: {e}", file=sys.stderr)
    print("❌ Error: Cannot import MemMimic core")
    sys.exit(1)


def generate_narrative(assistant, query, style="auto", max_memories=15):
    """Generate narrative from relevant memories"""
    try:
        # Search for relevant memories
        relevant_memories = assistant.memory_store.search(query, limit=max_memories)

        if not relevant_memories:
            return [
                f"📝 No relevant memories found for: '{query}'",
                "",
                "💡 Try:",
                "  • Different keywords",
                "  • Broader search terms",
                "  • Check that memories exist with: status()",
            ]

        # Style-based narrative generation
        if style == "auto":
            # Auto-detect style based on query
            if any(
                word in query.lower() for word in ["introduction", "who", "what am i"]
            ):
                style = "introduction"
            elif any(
                word in query.lower()
                for word in ["technical", "architecture", "system"]
            ):
                style = "technical"
            elif any(
                word in query.lower() for word in ["philosophy", "principle", "belief"]
            ):
                style = "philosophical"
            else:
                style = "general"

        # Generate narrative based on style
        narrative_parts = []

        # Header
        if style == "introduction":
            narrative_parts.append(f"🧠 IDENTITY NARRATIVE: {query}")
            narrative_parts.append("Who I am based on persistent memories")
        elif style == "technical":
            narrative_parts.append(f"🔧 TECHNICAL NARRATIVE: {query}")
            narrative_parts.append("System architecture and implementation details")
        elif style == "philosophical":
            narrative_parts.append(f"💭 PHILOSOPHICAL NARRATIVE: {query}")
            narrative_parts.append("Core principles and beliefs")
        else:
            narrative_parts.append(f"📖 NARRATIVE: {query}")
            narrative_parts.append("Story compiled from memory fragments")

        narrative_parts.append("=" * 60)
        narrative_parts.append(
            f"📊 Based on {len(relevant_memories)} relevant memories"
        )
        narrative_parts.append("")

        # Content generation
        narrative_parts.append("📝 NARRATIVE CONTENT:")
        narrative_parts.append("-" * 40)

        # Organize memories by type and importance
        memory_by_type = {}
        for memory in relevant_memories:
            mem_type = getattr(memory, "memory_type", "unknown")
            if mem_type not in memory_by_type:
                memory_by_type[mem_type] = []
            memory_by_type[mem_type].append(memory)

        # Generate narrative sections
        narrative_content = []

        if style == "introduction":
            narrative_content.append(
                "Based on my persistent memories, here's who I am:"
            )
            narrative_content.append("")

        elif style == "technical":
            narrative_content.append("Technical overview from implementation memories:")
            narrative_content.append("")

        elif style == "philosophical":
            narrative_content.append(
                "My core principles and philosophical foundations:"
            )
            narrative_content.append("")

        # Add memory content organized by relevance
        for i, memory in enumerate(relevant_memories[:10], 1):  # Top 10 most relevant
            content = getattr(memory, "content", "")
            mem_type = getattr(memory, "memory_type", "unknown")
            confidence = getattr(memory, "confidence", 0.0)

            # Truncate very long memories
            if len(content) > 300:
                content = content[:300] + "..."

            narrative_content.append(f"{i}. {content}")
            narrative_content.append(
                f"   [Type: {mem_type}, Confidence: {confidence:.2f}]"
            )
            narrative_content.append("")

        # Add generated narrative content
        narrative_parts.extend(narrative_content)

        narrative_parts.append("-" * 40)
        narrative_parts.append("")

        # Memory references
        narrative_parts.append("🔗 MEMORY REFERENCES:")
        narrative_parts.append("To explore specific memories:")
        for i, memory in enumerate(relevant_memories[:5], 1):
            mem_id = getattr(memory, "id", f"hash-{hash(memory.content)}")
            narrative_parts.append(f"  Memory {i}: ID #{mem_id}")

        narrative_parts.append("")
        narrative_parts.append("💡 NEXT STEPS:")
        narrative_parts.append(
            "  save_tale('narrative_name', 'content')  # Save this narrative"
        )
        narrative_parts.append(
            "  remember('new_insight')                 # Add new memories"
        )
        narrative_parts.append("")
        narrative_parts.append("🧠 MemMimic - Contextual narrative generation")

        return narrative_parts

    except Exception as e:
        return [f"❌ Error generating narrative: {str(e)}"]


def main():
    """Generate contextual narrative from memory fragments"""

    parser = argparse.ArgumentParser(
        description="MemMimic Context Tale - Narrative Generation"
    )
    parser.add_argument("query", help="Topic for narrative generation")
    parser.add_argument(
        "--style",
        default="auto",
        choices=["auto", "introduction", "technical", "philosophical", "general"],
        help="Narrative style",
    )
    parser.add_argument(
        "--max-memories", type=int, default=15, help="Maximum memories to include"
    )

    args = parser.parse_args()

    try:
        # Initialize MemMimic assistant
        assistant = ContextualAssistant("memmimic")

        # Generate narrative
        narrative_parts = generate_narrative(
            assistant=assistant,
            query=args.query,
            style=args.style,
            max_memories=args.max_memories,
        )

        # Output narrative
        print("\n".join(narrative_parts))

    except Exception as e:
        print(f"❌ Critical error in narrative generation: {str(e)}", file=sys.stderr)
        print(f"❌ Context tale generation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
