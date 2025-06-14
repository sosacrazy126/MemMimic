#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Delete Memory Guided Tool - Safe Memory Deletion
Professional memory deletion with contextual analysis and confirmation
"""

import sys
import os
import json
import argparse

# Force UTF-8 I/O for cross-platform compatibility
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from memmimic.memory import MemoryStore
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
            if hasattr(memory, 'id') and memory.id == memory_id:
                return memory
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error searching for memory: {e}", file=sys.stderr)
        return None

def analyze_deletion_impact(memory, assistant):
    """Analyze potential impact of deleting this memory"""
    try:
        content = getattr(memory, 'content', '')
        mem_type = getattr(memory, 'memory_type', 'unknown')
        confidence = getattr(memory, 'confidence', 0.0)
        
        # Assess deletion risks
        risks = []
        recommendations = []
        
        # Type-based analysis
        if mem_type == 'milestone':
            risks.append("Milestones mark important achievements - deletion may lose historical context")
            recommendations.append("Consider archiving instead of deleting")
            
        elif mem_type == 'synthetic':
            risks.append("Synthetic memories contain foundational knowledge")
            recommendations.append("Only delete if content is outdated or incorrect")
            
        elif mem_type == 'reflection':
            risks.append("Reflections contain valuable insights and patterns")
            recommendations.append("Consider if insights are captured elsewhere")
        
        # Confidence-based analysis
        if confidence > 0.8:
            risks.append("High-confidence memory - likely contains valuable information")
        elif confidence < 0.3:
            recommendations.append("Low confidence suggests this may be safe to delete")
        
        # Content-based analysis
        if len(content) > 500:
            risks.append("Large memory with substantial content")
            recommendations.append("Review content carefully before deletion")
        
        if any(keyword in content.lower() for keyword in ['important', 'critical', 'key', 'principle']):
            risks.append("Content indicates high importance")
        
        # Generate overall recommendation
        risk_level = len(risks)
        if risk_level == 0:
            overall = "LOW RISK - Safe to delete"
        elif risk_level <= 2:
            overall = "MEDIUM RISK - Review carefully"
        else:
            overall = "HIGH RISK - Consider alternatives"
        
        return {
            'risks': risks,
            'recommendations': recommendations,
            'overall': overall,
            'risk_level': risk_level
        }
        
    except Exception as e:
        return {
            'risks': ["Unable to assess risks"],
            'recommendations': ["Proceed with caution"],
            'overall': "UNKNOWN RISK",
            'risk_level': 1,
            'error': str(e)
        }

def main():
    """Guided memory deletion with safety analysis"""
    
    parser = argparse.ArgumentParser(description="MemMimic Delete Memory Guided")
    parser.add_argument('memory_id', type=int, help='ID of memory to delete')
    parser.add_argument('--confirm', action='store_true', help='Confirm deletion after analysis')
    
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
        content = getattr(memory, 'content', '')
        mem_type = getattr(memory, 'memory_type', 'unknown')
        confidence = getattr(memory, 'confidence', 0.0)
        created_at = getattr(memory, 'created_at', 'unknown')
        
        # Analyze deletion impact
        impact_analysis = analyze_deletion_impact(memory, assistant)
        
        # Display analysis
        result_parts = []
        
        # Header
        result_parts.append(f"⚠️  GUIDED MEMORY DELETION: #{args.memory_id}")
        result_parts.append("=" * 60)
        result_parts.append("")
        
        # Memory details
        result_parts.append("📝 MEMORY TO DELETE:")
        result_parts.append(f"  Type: {mem_type}")
        result_parts.append(f"  Confidence: {confidence:.2f}")
        result_parts.append(f"  Created: {created_at}")
        result_parts.append(f"  Size: {len(content)} characters")
        result_parts.append("")
        result_parts.append("Content:")
        # Show content with truncation if very long
        display_content = content if len(content) <= 300 else content[:300] + "..."
        result_parts.append(f"  {display_content}")
        result_parts.append("")
        
        # Risk analysis
        result_parts.append("🔍 DELETION IMPACT ANALYSIS:")
        result_parts.append(f"  Overall Risk: {impact_analysis['overall']}")
        result_parts.append("")
        
        if impact_analysis['risks']:
            result_parts.append("⚠️  POTENTIAL RISKS:")
            for risk in impact_analysis['risks']:
                result_parts.append(f"  • {risk}")
            result_parts.append("")
        
        if impact_analysis['recommendations']:
            result_parts.append("💡 RECOMMENDATIONS:")
            for rec in impact_analysis['recommendations']:
                result_parts.append(f"  • {rec}")
            result_parts.append("")
        
        # Confirmation handling
        if not args.confirm:
            result_parts.append("🛑 CONFIRMATION REQUIRED:")
            result_parts.append("  This analysis is for review only.")
            result_parts.append("  To proceed with deletion, use --confirm flag:")
            result_parts.append(f"  delete_memory_guided({args.memory_id}, confirm=True)")
            result_parts.append("")
            result_parts.append("🔄 ALTERNATIVES TO DELETION:")
            result_parts.append("  • Update memory with improved content")
            result_parts.append("  • Lower confidence level instead of deleting")
            result_parts.append("  • Create archive copy before deletion")
            
        else:
            # Proceed with deletion
            try:
                # Note: This would need to be implemented in the actual MemoryStore
                # For now, we'll simulate the deletion
                result_parts.append("✅ MEMORY DELETION CONFIRMED")
                result_parts.append("")
                result_parts.append("⚠️  SIMULATED DELETION (not yet implemented)")
                result_parts.append("In production, this would:")
                result_parts.append("  1. Create backup copy")
                result_parts.append("  2. Remove memory from database")
                result_parts.append("  3. Update memory statistics")
                result_parts.append("  4. Log deletion for audit trail")
                
            except Exception as e:
                result_parts.append(f"❌ Deletion failed: {str(e)}")
        
        result_parts.append("")
        result_parts.append("🧠 MemMimic - Safe memory management")
        
        print("\n".join(result_parts))
        
    except Exception as e:
        print(f"❌ Critical error in guided deletion: {str(e)}", file=sys.stderr)
        print(f"❌ Guided deletion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
