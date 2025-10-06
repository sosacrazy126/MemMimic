#!/usr/bin/env python3
"""
Re-classify existing memories using the enhanced CXD classifier.

This script:
1. Scans all memory markdown files
2. Re-classifies each using the enhanced classifier
3. Updates the YAML frontmatter with new classification + confidence
4. Generates a detailed report of changes
"""

import os
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from enhanced_cxd_classifier import EnhancedCXDClassifier


def load_memory_file(filepath: Path) -> tuple[dict, str]:
    """
    Load a memory markdown file and split frontmatter from content.

    Returns:
        (frontmatter_dict, content_text)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split frontmatter and content
    if content.startswith('---\n'):
        parts = content.split('---\n', 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return frontmatter, body

    # No frontmatter found
    return {}, content.strip()


def save_memory_file(filepath: Path, frontmatter: dict, content: str):
    """Save memory file with updated frontmatter."""
    # Convert frontmatter to YAML
    yaml_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)

    # Combine with content
    full_content = f"---\n{yaml_str}---\n\n{content}\n"

    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)


def reclassify_all_memories(memories_dir: str = "memories", dry_run: bool = False):
    """
    Re-classify all memory files in the memories directory.

    Args:
        memories_dir: Base directory containing memory files
        dry_run: If True, only report changes without saving
    """
    classifier = EnhancedCXDClassifier()
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
    print(f"Re-classifying {len(memory_files)} memories...")
    print(f"Mode: {'DRY RUN (no changes saved)' if dry_run else 'LIVE (will update files)'}")
    print(f"{'='*80}\n")

    # Statistics tracking
    stats = {
        'total': len(memory_files),
        'changed': 0,
        'unchanged': 0,
        'errors': 0,
        'by_old_cxd': defaultdict(int),
        'by_new_cxd': defaultdict(int),
        'transitions': defaultdict(int)
    }

    changes = []
    errors = []

    # Process each memory file
    for i, filepath in enumerate(memory_files, 1):
        try:
            # Load memory
            frontmatter, content = load_memory_file(filepath)

            # Get old classification
            old_cxd = frontmatter.get('cxd', 'unknown')
            old_confidence = frontmatter.get('cxd_confidence', 0.0)

            # Re-classify
            result = classifier.classify(content)
            new_cxd = result.function
            new_confidence = result.confidence

            # Track statistics
            stats['by_old_cxd'][old_cxd] += 1
            stats['by_new_cxd'][new_cxd] += 1

            # Check if changed
            if old_cxd != new_cxd or abs(old_confidence - new_confidence) > 0.05:
                stats['changed'] += 1
                transition = f"{old_cxd} → {new_cxd}"
                stats['transitions'][transition] += 1

                changes.append({
                    'file': filepath.name,
                    'old_cxd': old_cxd,
                    'new_cxd': new_cxd,
                    'old_confidence': old_confidence,
                    'new_confidence': new_confidence,
                    'evidence': result.evidence[:3]
                })

                # Update frontmatter
                frontmatter['cxd'] = new_cxd
                frontmatter['cxd_confidence'] = round(new_confidence, 2)

                # Save if not dry run
                if not dry_run:
                    save_memory_file(filepath, frontmatter, content)

                # Progress indicator
                if stats['changed'] % 10 == 0:
                    print(f"  Progress: {i}/{len(memory_files)} files processed, {stats['changed']} changed...")
            else:
                stats['unchanged'] += 1

        except Exception as e:
            stats['errors'] += 1
            errors.append({
                'file': filepath.name,
                'error': str(e)
            })
            print(f"  ⚠️  Error processing {filepath.name}: {e}")

    # Print report
    print(f"\n{'='*80}")
    print("RECLASSIFICATION REPORT")
    print(f"{'='*80}\n")

    print(f"Total Files Processed: {stats['total']}")
    print(f"  ✅ Changed: {stats['changed']}")
    print(f"  ➖ Unchanged: {stats['unchanged']}")
    print(f"  ❌ Errors: {stats['errors']}\n")

    print("ORIGINAL CLASSIFICATIONS:")
    for cxd, count in sorted(stats['by_old_cxd'].items()):
        pct = (count / stats['total']) * 100
        print(f"  {cxd:15} {count:3} ({pct:5.1f}%)")

    print("\nNEW CLASSIFICATIONS:")
    for cxd, count in sorted(stats['by_new_cxd'].items()):
        pct = (count / stats['total']) * 100
        print(f"  {cxd:15} {count:3} ({pct:5.1f}%)")

    if stats['transitions']:
        print("\nTOP TRANSITIONS:")
        sorted_transitions = sorted(stats['transitions'].items(),
                                   key=lambda x: x[1], reverse=True)
        for transition, count in sorted_transitions[:10]:
            print(f"  {transition:30} {count:3} files")

    if changes:
        print(f"\n{'='*80}")
        print(f"DETAILED CHANGES (showing first 10 of {len(changes)}):")
        print(f"{'='*80}\n")
        for change in changes[:10]:
            print(f"📄 {change['file']}")
            print(f"   Old: {change['old_cxd']} (conf: {change['old_confidence']:.2f})")
            print(f"   New: {change['new_cxd']} (conf: {change['new_confidence']:.2f})")
            print(f"   Evidence: {', '.join(change['evidence'][:2])}")
            print()

    if errors:
        print(f"\n{'='*80}")
        print(f"ERRORS ({len(errors)}):")
        print(f"{'='*80}\n")
        for error in errors:
            print(f"  ❌ {error['file']}: {error['error']}")

    if dry_run:
        print(f"\n{'='*80}")
        print("⚠️  DRY RUN MODE - No changes were saved")
        print("Run with --live to apply changes")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("✅ All changes saved successfully")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-classify MemMimic memories with enhanced CXD classifier"
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

    reclassify_all_memories(
        memories_dir=args.memories_dir,
        dry_run=dry_run
    )
