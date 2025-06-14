#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Think Tool - Contextual Memory Processing
Professional-grade thinking with full memory context
"""

import sys
import os
import json

# Force UTF-8 I/O for cross-platform compatibility
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from memmimic.assistant import ContextualAssistant
except ImportError as e:
    print(f"❌ Error importing MemMimic: {e}", file=sys.stderr)
    print("❌ Error: Cannot import MemMimic assistant")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("❌ Error: Missing input text")
        sys.exit(1)

    try:
        input_text = sys.argv[1]

        # Initialize MemMimic assistant
        assistant = ContextualAssistant("memmimic")

        # Process with memory context
        result = assistant.think(input_text)

        # Format professional response
        response_parts = []

        # Header
        response_parts.append("🧠 MEMMIMIC CONTEXTUAL THINKING")
        response_parts.append("=" * 50)
        response_parts.append(f"📝 Input: {input_text}")
        response_parts.append("")

        # Main response
        if isinstance(result, dict) and 'response' in result:
            response_parts.append("💭 Response:")
            response_parts.append(result['response'])
            response_parts.append("")

            # Memory usage statistics
            if result.get('memories_used', 0) > 0:
                response_parts.append(f"🧮 Memories utilized: {result['memories_used']}")
                response_parts.append("")

            # Thought process details
            if result.get('thought_process'):
                response_parts.append("🔍 Reasoning process:")
                if isinstance(result['thought_process'], dict):
                    for key, value in result['thought_process'].items():
                        response_parts.append(f"  • {key}: {value}")
                else:
                    response_parts.append(f"  {result['thought_process']}")
                response_parts.append("")

            # Socratic analysis if available
            if result.get('socratic_analysis'):
                socratic = result['socratic_analysis']
                response_parts.append("🤔 Socratic analysis:")
                response_parts.append(f"  • Internal questions: {socratic.get('questions_asked', 0)}")
                response_parts.append(f"  • Insights generated: {socratic.get('insights_generated', 0)}")
                if socratic.get('synthesis'):
                    response_parts.append(f"  • Synthesis: {socratic['synthesis'][:200]}...")
                response_parts.append("")
        else:
            # Fallback for simple string response
            response_parts.append("💭 Response:")
            response_parts.append(str(result))
            response_parts.append("")

        # Footer
        response_parts.append("🎯 MemMimic - Contextual memory processing")

        print("\n".join(response_parts))

    except Exception as e:
        print(f"❌ Error in contextual processing: {str(e)}", file=sys.stderr)
        print(f"❌ Failed to process with memory: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
