#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Load Tale Tool - Professional Tale Loading
Clean, direct tale loading with beautiful formatting
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
    from memmimic.tales.tale_manager import TaleManager
except ImportError as e:
    print(f"❌ Error importing MemMimic: {e}", file=sys.stderr)
    print("❌ Error: Cannot import MemMimic tales")
    sys.exit(1)


def main():
    """Load and display specific tale with professional formatting"""

    parser = argparse.ArgumentParser(description="MemMimic Load Tale")
    parser.add_argument("name", help="Name of tale to load")
    parser.add_argument("--category", help="Specific category to search in")

    args = parser.parse_args()

    try:
        # Initialize tales manager
        tale_manager = TaleManager()

        # Load the tale
        tale_data = tale_manager.load_tale(args.name, category=args.category)

        if not tale_data:
            print(f"❌ Tale not found: '{args.name}'")
            print("")
            print("💡 SUGGESTIONS:")
            print("  tales()                    # List all available tales")
            print(f"  tales('{args.name}')        # Search for similar names")
            print("  tales(category='specific') # Browse by category")
            sys.exit(1)

        # Extract tale information from Tale object
        tale_name = tale_data.name
        tale_category = tale_data.category
        tale_size = tale_data.metadata.get("size_chars", 0)
        tale_updated = tale_data.metadata.get("updated", "unknown")
        tale_created = tale_data.metadata.get("created", "unknown")
        tale_version = tale_data.metadata.get("version", 1)
        tale_usage_count = tale_data.metadata.get("usage_count", 0)
        tale_tags = tale_data.tags
        tale_content = tale_data.content

        # Format category emoji
        category_emoji = (
            "🧠"
            if tale_category.startswith("claude")
            else "🔧" if tale_category.startswith("projects") else "📦"
        )

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
        result_parts.append(
            f"  save_tale('{tale_name}', 'new_content')  # Update this tale"
        )
        result_parts.append(
            f"  delete_tale('{tale_name}')               # Delete this tale"
        )
        result_parts.append(
            "  tales()                                  # Browse all tales"
        )
        result_parts.append("")

        # Footer
        result_parts.append("📖 MemMimic Tales - Professional narrative management")

        # Output
        print("\n".join(result_parts))

        # Usage count is already incremented by load_tale method

    except Exception as e:
        print(f"❌ Critical error loading tale: {str(e)}", file=sys.stderr)
        print(f"❌ Load operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

