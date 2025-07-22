#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Save Tale Tool - Auto-Detect Create/Update
FUSION: Intelligent auto-detection of create vs update operations
Professional-grade tale management with zero user decision overhead
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


def auto_detect_save_operation(tale_manager, name, category=None):
    """
    Intelligently detect whether this should be create or update operation
    Returns: ('create', None) or ('update', existing_tale_data)
    """
    try:
        # Try to load existing tale
        existing_tale = tale_manager.load_tale(name, category=category)

        if existing_tale:
            return ("update", existing_tale)
        else:
            return ("create", None)

    except Exception:
        # If load fails, assume it doesn't exist
        return ("create", None)


def create_new_tale(tale_manager, name, content, category, tags=None):
    """Create a new tale with professional formatting"""
    try:
        # Prepare tale data
        tale_data = {
            "name": name,
            "content": content,
            "category": category,
            "tags": tags.split(",") if tags else [],
        }

        # Create the tale
        result = tale_manager.create_tale(
            name=name,
            content=content,
            category=category,
            tags=tags,
            overwrite=False,  # Fail if exists (shouldn't happen due to auto-detect)
        )

        # Format success message
        success_parts = []
        success_parts.append("✅ TALE CREATED SUCCESSFULLY")
        success_parts.append("=" * 50)
        success_parts.append(f"📖 Name: {name}")
        success_parts.append(f"📂 Category: {category}")
        success_parts.append(f"📏 Size: {len(content):,} characters")

        if tags:
            success_parts.append(f"🏷️  Tags: {tags}")

        success_parts.append("")
        success_parts.append("💡 NEXT STEPS:")
        success_parts.append(f"  tales('{name}', load=True)    # View your new tale")
        success_parts.append(f"  save_tale('{name}', 'text')   # Update content")
        success_parts.append("  tales()                       # List all tales")
        success_parts.append("")
        success_parts.append("🚀 Tale ready for use!")

        return success_parts

    except Exception as e:
        return [f"❌ Error creating tale: {str(e)}"]


def update_existing_tale(
    tale_manager, name, content, category, existing_tale, tags=None
):
    """Update an existing tale with change detection"""
    try:
        # Get existing content for comparison
        old_content = existing_tale.get("content", "")
        old_size = len(old_content)
        new_size = len(content)

        # Detect changes
        content_changed = old_content != content
        size_change = new_size - old_size

        if not content_changed:
            return [
                f"ℹ️ Tale '{name}' unchanged",
                f"📏 Content identical ({new_size:,} characters)",
                "",
                "💡 No update needed - content is the same",
            ]

        # Update the tale
        result = tale_manager.update_tale(
            name=name, content=content, category=category, tags=tags
        )

        # Format update message
        update_parts = []
        update_parts.append("✅ TALE UPDATED SUCCESSFULLY")
        update_parts.append("=" * 50)
        update_parts.append(f"📖 Name: {name}")
        update_parts.append(f"📂 Category: {category}")
        update_parts.append(f"📏 Size: {old_size:,} → {new_size:,} characters")

        # Show size change with direction
        if size_change > 0:
            update_parts.append(f"📈 Change: +{size_change:,} characters (expanded)")
        elif size_change < 0:
            update_parts.append(f"📉 Change: {size_change:,} characters (condensed)")
        else:
            update_parts.append(f"📊 Change: Content modified (same length)")

        if tags:
            update_parts.append(f"🏷️  Tags: {tags}")

        update_parts.append("")
        update_parts.append("💡 NEXT STEPS:")
        update_parts.append(f"  tales('{name}', load=True)    # View updated tale")
        update_parts.append("  tales()                       # List all tales")
        update_parts.append("")
        update_parts.append("🔄 Tale successfully updated!")

        return update_parts

    except Exception as e:
        return [f"❌ Error updating tale: {str(e)}"]


def main():
    """Smart save tale with auto-detection"""

    parser = argparse.ArgumentParser(
        description="MemMimic Save Tale - Auto-Detect Create/Update"
    )
    parser.add_argument("name", help="Tale name")
    parser.add_argument("content", help="Tale content")
    parser.add_argument("--category", default="claude/core", help="Tale category")
    parser.add_argument("--tags", help="Comma-separated tags")

    args = parser.parse_args()

    try:
        # Initialize tales manager
        tale_manager = TaleManager()

        # Auto-detect operation type
        operation, existing_tale = auto_detect_save_operation(
            tale_manager, args.name, args.category
        )

        # Execute appropriate operation
        if operation == "create":
            result_parts = create_new_tale(
                tale_manager, args.name, args.content, args.category, args.tags
            )
        else:  # operation == 'update'
            result_parts = update_existing_tale(
                tale_manager,
                args.name,
                args.content,
                args.category,
                existing_tale,
                args.tags,
            )

        # Output results
        print("\n".join(result_parts))

    except Exception as e:
        print(f"❌ Critical error in save tale: {str(e)}", file=sys.stderr)
        print(f"❌ Save operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

