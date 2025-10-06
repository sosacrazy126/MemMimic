#!/usr/bin/env python3
"""
Enhance existing memories with new intelligence features.

This script retroactively adds:
- Quality scoring
- Relationship tracking
- Auto-tagging
to memories that were created before these features existed.
"""

import os
import yaml
from pathlib import Path
from datetime import datetime
from enhanced_cxd_classifier import EnhancedCXDClassifier
from updated_mcp_tools import MemMimicMCP


def load_memory_file(filepath: Path) -> tuple[dict, str]:
    """Load memory markdown file and split frontmatter from content."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split frontmatter and content
    if content.startswith('---\n'):
        parts = content.split('---\n', 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return frontmatter, body

    return {}, content.strip()


def save_memory_file(filepath: Path, frontmatter: dict, content: str):
    """Save memory file with updated frontmatter."""
    yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
    full_content = f"---\n{yaml_str}---\n\n{content}\n"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)


def enhance_memories(memories_dir: str = "memories", dry_run: bool = False):
    """
    Enhance all existing memories with new intelligence features.

    Args:
        memories_dir: Base directory containing memory files
        dry_run: If True, only report changes without saving
    """
    mcp = MemMimicMCP()
    memories_path = Path(memories_dir)

    if not memories_path.exists():
        print(f"❌ Memories directory not found: {memories_dir}")
        return

    # Find all memory markdown files
    memory_files = list(memories_path.rglob("*.md"))

    if not memory_files:
        print(f"❌ No memory files found in {memories_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Enhancing {len(memory_files)} memories with new intelligence features...")
    print(f"Mode: {'DRY RUN (no changes saved)' if dry_run else 'LIVE (will update files)'}")
    print(f"{'='*80}\n")

    # Statistics
    stats = {
        'total': len(memory_files),
        'enhanced': 0,
        'skipped': 0,
        'errors': 0,
        'features_added': {
            'quality': 0,
            'tags': 0,
            'relationships': 0
        }
    }

    for i, filepath in enumerate(memory_files, 1):
        try:
            # Load memory
            frontmatter, content = load_memory_file(filepath)

            # Check if already enhanced (has all features)
            has_quality = 'quality' in frontmatter
            has_tags = 'tags' in frontmatter
            has_relationships = 'relationships' in frontmatter

            if has_quality and has_tags and has_relationships:
                stats['skipped'] += 1
                if i % 20 == 0:
                    print(f"  Progress: {i}/{len(memory_files)} files processed, {stats['enhanced']} enhanced...")
                continue

            enhanced = False

            # Add quality scoring if missing
            if not has_quality:
                metadata = frontmatter.copy()
                quality = mcp._calculate_quality_score(content, metadata)
                frontmatter['quality'] = quality
                stats['features_added']['quality'] += 1
                enhanced = True

            # Add tags if missing
            if not has_tags:
                cxd = frontmatter.get('cxd', 'unknown')
                tags = mcp._extract_tags(content, cxd)
                frontmatter['tags'] = tags
                stats['features_added']['tags'] += 1
                enhanced = True

            # Add relationships if missing (limited to avoid too much processing)
            if not has_relationships:
                # Only find relationships for first 100 files to avoid excessive processing
                if i <= 100:
                    similar = mcp._find_similar_memories(content, limit=5)
                    relationships = {
                        'similar_memories': [s['id'] for s in similar],
                        'relationship_strength': similar[0]['similarity'] if similar else 0.0
                    }
                    frontmatter['relationships'] = relationships
                    stats['features_added']['relationships'] += 1
                else:
                    # Set empty relationships for others
                    frontmatter['relationships'] = {
                        'similar_memories': [],
                        'relationship_strength': 0.0
                    }
                enhanced = True

            if enhanced:
                stats['enhanced'] += 1

                # Save if not dry run
                if not dry_run:
                    save_memory_file(filepath, frontmatter, content)

                # Progress indicator
                if stats['enhanced'] % 10 == 0:
                    print(f"  Progress: {i}/{len(memory_files)} files processed, {stats['enhanced']} enhanced...")

        except Exception as e:
            stats['errors'] += 1
            print(f"  ⚠️  Error processing {filepath.name}: {e}")

    # Print report
    print(f"\n{'='*80}")
    print("ENHANCEMENT REPORT")
    print(f"{'='*80}\n")

    print(f"Total Files: {stats['total']}")
    print(f"  ✅ Enhanced: {stats['enhanced']}")
    print(f"  ➖ Skipped (already enhanced): {stats['skipped']}")
    print(f"  ❌ Errors: {stats['errors']}\n")

    print("Features Added:")
    for feature, count in stats['features_added'].items():
        print(f"  {feature:15} {count:3} memories")

    if dry_run:
        print(f"\n{'='*80}")
        print("⚠️  DRY RUN MODE - No changes were saved")
        print("Run with --live to apply changes")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("✅ All enhancements saved successfully")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhance existing MemMimic memories with new intelligence features"
    )
    parser.add_argument(
        "--memories-dir",
        default="memories",
        help="Base directory containing memory files (default: memories)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Apply changes (default is dry-run)"
    )

    args = parser.parse_args()

    # Default to dry run unless --live specified
    dry_run = not args.live

    enhance_memories(
        memories_dir=args.memories_dir,
        dry_run=dry_run
    )
