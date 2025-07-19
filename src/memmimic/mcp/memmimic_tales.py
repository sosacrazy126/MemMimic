#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Tales Tool - Unified Interface
FUSION: list + search + stats + load in one intelligent tool
Professional-grade tales management with smart auto-detection
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


def list_tales(tale_manager, category=None, limit=10):
    """List tales with beautiful formatting"""
    try:
        tales = tale_manager.list_tales(category=category)

        if not tales:
            return [
                "📚 No tales found.",
                "",
                "💡 Create your first tale with save_tale()",
            ]

        result_parts = []
        result_parts.append("📚 MEMMIMIC TALES COLLECTION")
        result_parts.append("=" * 50)

        if category:
            result_parts.append(f"📂 Category: {category}")

        result_parts.append(f"📊 Showing {len(tales)} tales")
        result_parts.append("")

        for i, tale in enumerate(tales, 1):
            name = tale.get("name", "untitled")
            size = tale.get("size", 0)
            category_display = tale.get("category", "unknown")
            updated = tale.get("updated", "unknown")
            usage_count = tale.get("usage_count", 0)

            # Format size
            if size < 1000:
                size_str = f"{size}B"
            elif size < 1000000:
                size_str = f"{size//1000}K"
            else:
                size_str = f"{size//1000000}M"

            # Format category emoji
            category_emoji = (
                "🧠"
                if category_display.startswith("claude")
                else "🔧" if category_display.startswith("projects") else "📦"
            )

            result_parts.append(f"{i:2d}. {category_emoji} {name}")
            result_parts.append(
                f"    📂 {category_display} | 📏 {size_str} | 🕐 {updated}"
            )
            if usage_count > 0:
                result_parts.append(f"    📈 Used {usage_count}x")
            result_parts.append("")

        result_parts.append("💡 COMMANDS:")
        result_parts.append("  tales('tale_name', load=True)  # Load specific tale")
        result_parts.append("  tales('search_term')           # Search tales")
        result_parts.append("  tales(stats=True)              # Collection statistics")

        return result_parts

    except Exception as e:
        return [f"❌ Error listing tales: {str(e)}"]


def search_tales(tale_manager, query, category=None, limit=10):
    """Search tales with intelligent highlighting"""
    try:
        results = tale_manager.search_tales(query=query, category=category)

        if not results:
            return [
                f"🔍 No tales found for: '{query}'",
                "",
                "💡 Try:",
                "  • Different keywords",
                "  • Remove category filter",
                "  • Check spelling",
            ]

        result_parts = []
        result_parts.append(f"🔍 TALES SEARCH: '{query}'")
        result_parts.append("=" * 50)

        if category:
            result_parts.append(f"📂 Category filter: {category}")

        result_parts.append(f"📊 Found {len(results)} matches")
        result_parts.append("")

        for i, result in enumerate(results, 1):
            name = result.get("name", "untitled")
            category_display = result.get("category", "unknown")
            preview = result.get("preview", "")  # Use preview instead of context

            # Format category emoji
            category_emoji = (
                "🧠"
                if category_display.startswith("claude")
                else "🔧" if category_display.startswith("projects") else "📦"
            )

            result_parts.append(f"{i}. {category_emoji} {name}")
            result_parts.append(f"   📂 {category_display}")

            if preview:
                # Truncate preview if too long
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                result_parts.append(f"   💭 ...{preview}...")

            result_parts.append("")

        result_parts.append("💡 NEXT STEPS:")
        result_parts.append("  tales('tale_name', load=True)  # Load specific tale")
        result_parts.append("  save_tale('name', 'content')   # Create new tale")

        return result_parts

    except Exception as e:
        return [f"❌ Error searching tales: {str(e)}"]


def load_tale(tale_manager, name, category=None):
    """Load and display specific tale"""
    try:
        tale = tale_manager.load_tale(name, category=category)

        if not tale:
            return [
                f"❌ Tale not found: '{name}'",
                "",
                "💡 Try:",
                "  tales()                  # List all tales",
                f"  tales('{name}')          # Search for similar names",
            ]

        result_parts = []

        # Tale header
        tale_name = tale.name
        tale_category = tale.category
        tale_size = tale.metadata.get("size_chars", len(tale.content))
        tale_updated = tale.metadata.get("updated", "unknown")
        tale_content = tale.content

        # Format category emoji
        category_emoji = (
            "🧠"
            if tale_category.startswith("claude")
            else "🔧" if tale_category.startswith("projects") else "📦"
        )

        result_parts.append(f"📖 TALE LOADED: {tale_name}")
        result_parts.append("=" * 60)
        result_parts.append(f"{category_emoji} Category: {tale_category}")
        result_parts.append(f"📏 Size: {tale_size:,} characters")
        result_parts.append(f"🕐 Updated: {tale_updated}")
        result_parts.append("")
        result_parts.append("📝 CONTENT:")
        result_parts.append("-" * 40)
        result_parts.append(tale_content)
        result_parts.append("-" * 40)
        result_parts.append("")
        result_parts.append("💡 ACTIONS:")
        result_parts.append(
            f"  save_tale('{tale_name}', 'new_content')  # Update this tale"
        )
        result_parts.append(
            f"  delete_tale('{tale_name}')               # Delete this tale"
        )

        return result_parts

    except Exception as e:
        return [f"❌ Error loading tale: {str(e)}"]


def show_stats(tale_manager):
    """Show detailed collection statistics"""
    try:
        stats = tale_manager.get_statistics()

        result_parts = []
        result_parts.append("📊 TALES COLLECTION STATISTICS")
        result_parts.append("=" * 50)

        # Basic stats
        total_tales = stats.get("total_tales", 0)
        total_chars = stats.get("total_chars", 0)
        avg_size = stats.get("avg_tale_size", 0)

        result_parts.append(f"📚 Total tales: {total_tales}")
        result_parts.append(f"💾 Total size: {total_chars:,} characters")
        if total_tales > 0:
            result_parts.append(f"📏 Average size: {avg_size:.0f} characters")
        result_parts.append("")

        # Category distribution
        category_stats = stats.get("by_category", {})
        if category_stats:
            result_parts.append("📂 BY CATEGORY:")
            for category, info in sorted(category_stats.items()):
                count = info.get("count", 0) if isinstance(info, dict) else info
                percentage = (count / total_tales * 100) if total_tales > 0 else 0
                category_emoji = (
                    "🧠"
                    if category.startswith("claude")
                    else "🔧" if category.startswith("projects") else "📦"
                )
                result_parts.append(
                    f"  {category_emoji} {category}: {count} ({percentage:.1f}%)"
                )
            result_parts.append("")

        # System info
        result_parts.append("🔧 SYSTEM INFO:")
        result_parts.append(f"  📦 Cache size: {stats.get('cache_size', 0)}")
        result_parts.append(
            f"  🔄 Structure version: {stats.get('structure_version', 'unknown')}"
        )
        result_parts.append("")

        result_parts.append("🚀 MemMimic Tales - Professional narrative management")

        return result_parts

    except Exception as e:
        return [f"❌ Error getting statistics: {str(e)}"]


def main():
    """Unified tales interface with intelligent behavior detection"""

    # Parse arguments intelligently
    parser = argparse.ArgumentParser(description="MemMimic Tales - Unified Interface")
    parser.add_argument("query", nargs="?", help="Search query or tale name")
    parser.add_argument(
        "--stats", action="store_true", help="Show collection statistics"
    )
    parser.add_argument("--load", action="store_true", help="Load tale by name")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--limit", type=int, default=10, help="Maximum results")

    args = parser.parse_args()

    try:
        # Initialize tales manager
        tale_manager = TaleManager()

        # Intelligent behavior detection
        if args.stats:
            # Show statistics
            result_parts = show_stats(tale_manager)

        elif args.load and args.query:
            # Load specific tale
            result_parts = load_tale(tale_manager, args.query, args.category)

        elif args.query:
            # Search tales (query provided)
            result_parts = search_tales(
                tale_manager, args.query, args.category, args.limit
            )

        else:
            # List tales (no query)
            result_parts = list_tales(tale_manager, args.category, args.limit)

        # Output results
        print("\n".join(result_parts))

    except Exception as e:
        print(f"❌ Critical error in tales interface: {str(e)}", file=sys.stderr)
        print(f"❌ Tales operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
