#!/usr/bin/env python3
"""
MemMimic Socratic Dialogue Tool - Self-questioning for deeper understanding
"""

import sys
import os
sys.path.insert(0, '/home/evilbastardxd/Desktop/tools/memmimicc')
os.environ['MEMMIMIC_STORAGE'] = 'markdown'
os.environ['MEMMIMIC_MD_DIR'] = '/home/evilbastardxd/Desktop/tools/memmimicc'

try:
    from updated_mcp_tools import MemMimicMCP
except ImportError as e:
    print(f"❌ Error importing MemMimicMCP: {e}")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("❌ Error: Missing query argument")
        sys.exit(1)
    
    try:
        query = sys.argv[1]
        depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        
        mcp = MemMimicMCP()
        
        response = []
        response.append("🤔 SOCRATIC DIALOGUE")
        response.append("=" * 50)
        response.append(f"Topic: {query}")
        response.append(f"Depth: {depth} levels")
        response.append("")
        
        # Generate questions at different depths
        questions = [
            f"What do we know about: {query}?",
            f"Why is {query} important?",
            f"What assumptions are we making about {query}?",
            f"How does {query} connect to other concepts?",
            f"What evidence supports our understanding of {query}?",
            f"What questions remain unanswered about {query}?"
        ]
        
        for i, question in enumerate(questions[:depth], 1):
            response.append(f"Question {i}: {question}")
            response.append("")
            
            # Use think_with_memory to explore each question
            result = mcp.think_with_memory(question, mode='enhanced', max_thoughts=3)
            
            if result['status'] == 'success':
                response.append(f"Analysis: {result['final_analysis'][:200]}...")
                response.append(f"Memories examined: {result['memories_examined']}")
                response.append("")
            else:
                response.append("Unable to find relevant memories")
                response.append("")
        
        response.append("🎯 SYNTHESIS:")
        response.append("-" * 30)
        response.append(f"Through {depth} levels of questioning, we explored {query}")
        response.append("from multiple perspectives using memory-based reasoning.")
        
        print('\n'.join(response))
        
    except Exception as e:
        print(f"❌ Error in socratic dialogue: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()