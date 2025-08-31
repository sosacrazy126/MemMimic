#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Load Tale Tool - Professional Tale Loading
Clean, direct tale loading with beautiful formatting
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
    from memmimic.tales.tale_manager import TaleManager
except ImportError as e:
    print(f"❌ Error importing MemMimic: {e}", file=sys.stderr)
    print("❌ Error: Cannot import MemMimic tales")
    sys.exit(1)

def main():
    """Load and display specific tale with professional formatting"""
    
    parser = argparse.ArgumentParser(description="MemMimic Load Tale")
    parser.add_argument('name', help='Name of tale to load')
    parser.add_argument('--category', help='Specific category to search in')
    
    args = parser.parse_args()
    
    try:
        # Initialize tales manager
        tale_manager = TaleManager()
        
        # Load the tale
        tale = tale_manager.load_tale(args.name, category=args.category)
        
        if not tale:
            print(f"❌ Tale not found: '{args.name}'")
            print("")
            print("💡 SUGGESTIONS:")
            print("  tales()                    # List all available tales")
            print(f"  tales('{args.name}')        # Search for similar names")
            print("  tales(category='specific') # Browse by category")
            sys.exit(1)
        
        # Extract tale information from Tale object
        tale_name = tale.name
        tale_category = tale.category
        tale_size = tale.metadata.get('size_chars', len(tale.content))
        tale_updated = tale.metadata.get('updated', tale.metadata.get('created', 'unknown'))
        tale_created = tale.metadata.get('created', 'unknown')
        tale_version = tale.metadata.get('version', 1)
        tale_usage_count = tale.metadata.get('usage_count', 0)  # Fixed: usage_count is in metadata
        tale_tags = tale.metadata.get('tags', [])
        tale_content = tale.content
        
        # Format category emoji
        category_emoji = "🧠" if tale_category.startswith('claude') else "🔧" if tale_category.startswith('projects') else "📦"
        
        # Build beautiful output
        result_parts = []
        
        # Header
        result_parts.append(f"📖 TALE LOADED: {tale_name}")
        result_parts.append("=" * 60)
        
        # Metadata section
        result_parts.append(f"{category_emoji} Category: {tale_category}")
        result_parts.append(f"📏 Size: {tale_size:,} characters")
        result_parts.append(f"🆔 Version: {tale_version}")
        
        if tale_usage_count > 0:
            result_parts.append(f"📈 Usage: {tale_usage_count}x")
        
        if tale_tags:
            result_parts.append(f"🏷️  Tags: {', '.join(tale_tags)}")
        
        result_parts.append(f"📅 Created: {tale_created}")
        result_parts.append(f"🕐 Updated: {tale_updated}")
        result_parts.append("")
        
        # Content section
        result_parts.append("📝 CONTENT:")
        result_parts.append("-" * 40)
        result_parts.append(tale_content)
        result_parts.append("-" * 40)
        result_parts.append("")
        
        # Action suggestions
        result_parts.append("💡 AVAILABLE ACTIONS:")
        result_parts.append(f"  save_tale('{tale_name}', 'new_content')  # Update this tale")
        result_parts.append(f"  delete_tale('{tale_name}')               # Delete this tale")
        result_parts.append("  tales()                                  # Browse all tales")
        result_parts.append("")
        
        # Footer
        result_parts.append("📖 MemMimic Tales - Professional narrative management")
        
        # Output
        print("\n".join(result_parts))
        
        # Update usage count
        try:
            tale_manager.increment_usage_count(args.name, args.category)
        except:
            pass  # Non-critical operation
        
    except Exception as e:
        print(f"❌ Critical error loading tale: {str(e)}", file=sys.stderr)
        print(f"❌ Load operation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
