#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Delete Tale Tool - Safe Tale Deletion
Professional tale deletion with confirmation and backup
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
    """Safe tale deletion with confirmation"""

    parser = argparse.ArgumentParser(description="MemMimic Delete Tale")
    parser.add_argument("name", help="Name of tale to delete")
    parser.add_argument("--category", help="Specific category to search in")
    parser.add_argument(
        "--confirm", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    try:
        # Initialize tales manager
        tale_manager = TaleManager()

        # Check if tale exists
        tale_data = tale_manager.load_tale(args.name, category=args.category)

        if not tale_data:
            print(f"❌ Tale not found: '{args.name}'")
            print("")
            print("💡 SUGGESTIONS:")
            print("  tales()                    # List all available tales")
            print(f"  tales('{args.name}')        # Search for similar names")
            sys.exit(1)

        # Extract tale information for confirmation
        tale_name = tale_data.get("name", args.name)
        tale_category = tale_data.get("category", "unknown")
        tale_size = tale_data.get("size", 0)
        tale_content = tale_data.get("content", "")

        # Show tale info and confirmation
        if not args.confirm:
            print("⚠️  TALE DELETION CONFIRMATION")
            print("=" * 50)
            print(f"📖 Name: {tale_name}")
            print(f"📂 Category: {tale_category}")
            print(f"📏 Size: {tale_size:,} characters")
            print("")
            print("📝 Content preview:")
            preview = (
                tale_content[:200] + "..." if len(tale_content) > 200 else tale_content
            )
            print(f"   {preview}")
            print("")
            print("❌ This action cannot be undone!")
            print("   Use --confirm flag to proceed with deletion")
            sys.exit(0)

        # Perform deletion
        result = tale_manager.delete_tale(
            name=args.name, category=args.category, hard=False  # Soft delete for safety
        )

        # Format success message
        success_parts = []
        success_parts.append("✅ TALE DELETED SUCCESSFULLY")
        success_parts.append("=" * 50)
        success_parts.append(f"📖 Deleted: {tale_name}")
        success_parts.append(f"📂 From category: {tale_category}")
        success_parts.append(f"📏 Size was: {tale_size:,} characters")
        success_parts.append("")
        success_parts.append("🗑️  Tale moved to trash (soft delete)")
        success_parts.append("💡 Recovery may be possible - contact administrator")
        success_parts.append("")
        success_parts.append("REMAINING ACTIONS:")
        success_parts.append("  tales()                       # Browse remaining tales")
        success_parts.append(
            f"  save_tale('{tale_name}', 'content')  # Recreate if needed"
        )

        print("\n".join(success_parts))

    except Exception as e:
        print(f"❌ Critical error deleting tale: {str(e)}", file=sys.stderr)
        print(f"❌ Delete operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

