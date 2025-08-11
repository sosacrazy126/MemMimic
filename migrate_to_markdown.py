#!/usr/bin/env python3
"""
Migrate MemMimic SQLite database to Markdown files
"""

import sqlite3
import json
import yaml
from pathlib import Path
from datetime import datetime
import re

def read_all_memories(db_path):
    """Read all memories from SQLite database"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    memories = conn.execute("""
        SELECT id, content, metadata, importance_score, created_at, updated_at
        FROM memories
        ORDER BY created_at DESC
    """).fetchall()
    
    conn.close()
    return memories

def parse_metadata(metadata_json):
    """Parse metadata JSON string"""
    if not metadata_json:
        return {}
    try:
        return json.loads(metadata_json)
    except:
        return {}

def generate_file_path(memory, output_dir):
    """Generate organized file path based on date and content"""
    # Parse date
    created = datetime.fromisoformat(memory['created_at'])
    
    # Create directory structure: YYYY/MM/DD/
    dir_path = Path(output_dir) / 'memories' / f"{created.year:04d}" / f"{created.month:02d}" / f"{created.day:02d}"
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Extract CXD type and keywords for filename
    metadata = parse_metadata(memory['metadata'])
    # Handle nested CXD structure
    if isinstance(metadata.get('cxd'), dict):
        cxd_type = metadata['cxd'].get('function', 'unknown').lower()
    else:
        cxd_type = str(metadata.get('cxd', 'unknown')).lower()
    
    # Extract first few meaningful words for filename
    content_preview = re.sub(r'[^\w\s]', '', memory['content'][:50])
    keywords = '_'.join(content_preview.split()[:3]).lower()
    
    # Create filename: mem_ID_CXD_keywords.md
    filename = f"mem_{memory['id']}_{cxd_type}_{keywords}.md"
    
    return dir_path / filename

def create_markdown_content(memory):
    """Convert database memory to markdown with frontmatter"""
    metadata = parse_metadata(memory['metadata'])
    
    # Build frontmatter
    frontmatter = {
        'id': f"mem_{memory['id']}",
        'importance': memory['importance_score'],
        'created': memory['created_at'],
        'updated': memory['updated_at'],
    }
    
    # Add metadata fields
    if 'cxd' in metadata:
        # Handle nested CXD structure
        if isinstance(metadata['cxd'], dict):
            frontmatter['cxd'] = metadata['cxd'].get('function', 'unknown')
            frontmatter['cxd_pattern'] = metadata['cxd'].get('pattern', '')
            frontmatter['cxd_confidence'] = metadata['cxd'].get('confidence', 0.0)
        else:
            frontmatter['cxd'] = metadata['cxd']
    if 'memory_type' in metadata:
        frontmatter['type'] = metadata['memory_type']
    elif 'type' in metadata:
        frontmatter['type'] = metadata['type']
    if 'tags' in metadata:
        frontmatter['tags'] = metadata['tags']
    
    # Add any other metadata
    for key, value in metadata.items():
        if key not in ['cxd', 'type', 'tags']:
            frontmatter[key] = value
    
    # Create markdown content
    yaml_front = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    
    markdown = f"---\n{yaml_front}---\n\n"
    markdown += f"# Memory {memory['id']}\n\n"
    markdown += memory['content']
    
    return markdown

def save_markdown_file(file_path, content):
    """Save markdown content to file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_search_index(memories, output_dir):
    """Create JSON index for fast searching"""
    index = []
    
    for memory in memories:
        metadata = parse_metadata(memory['metadata'])
        created = datetime.fromisoformat(memory['created_at'])
        
        # Handle nested CXD structure
        if isinstance(metadata.get('cxd'), dict):
            cxd_value = metadata['cxd'].get('function', 'unknown')
        else:
            cxd_value = metadata.get('cxd', 'unknown')
        
        index_entry = {
            'id': memory['id'],
            'path': f"{created.year:04d}/{created.month:02d}/{created.day:02d}",
            'importance': memory['importance_score'],
            'created': memory['created_at'],
            'cxd': cxd_value,
            'preview': memory['content'][:100],
        }
        index.append(index_entry)
    
    # Save index
    index_path = Path(output_dir) / 'memories' / 'index.json'
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created search index with {len(index)} entries")

def migrate_relationships(db_path, output_dir):
    """Migrate memory relationships to JSON"""
    conn = sqlite3.connect(db_path)
    
    relationships = conn.execute("""
        SELECT source_memory_id, target_memory_id, relationship_type, strength
        FROM memory_relationships
    """).fetchall()
    
    conn.close()
    
    if relationships:
        rel_data = []
        for rel in relationships:
            rel_data.append({
                'source': rel[0],
                'target': rel[1],
                'type': rel[2],
                'strength': rel[3]
            })
        
        rel_path = Path(output_dir) / 'memories' / 'relationships.json'
        with open(rel_path, 'w', encoding='utf-8') as f:
            json.dump(rel_data, f, indent=2)
        
        print(f"✅ Migrated {len(relationships)} relationships")

def migrate_database_to_markdown(db_path, output_dir):
    """Main migration function"""
    print(f"🔄 Starting migration from {db_path} to {output_dir}")
    
    # 1. Read all memories
    memories = read_all_memories(db_path)
    print(f"📚 Found {len(memories)} memories to migrate")
    
    # 2. For each memory, create markdown file
    for i, memory in enumerate(memories, 1):
        md_content = create_markdown_content(memory)
        file_path = generate_file_path(memory, output_dir)
        save_markdown_file(file_path, md_content)
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(memories)} memories...")
    
    # 3. Create index for fast search
    create_search_index(memories, output_dir)
    
    # 4. Migrate relationships
    migrate_relationships(db_path, output_dir)
    
    print(f"✅ Successfully migrated {len(memories)} memories to markdown")
    print(f"📁 Output directory: {output_dir}/memories/")
    
    return len(memories)

def main():
    """Run migration"""
    import sys
    
    # Default paths
    db_path = "/home/evilbastardxd/Desktop/tools/memmimicc/data/databases/memmimic.db"
    output_dir = "/home/evilbastardxd/Desktop/tools/memmimicc"
    
    # Override with command line args if provided
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return 1
    
    # Run migration
    try:
        count = migrate_database_to_markdown(db_path, output_dir)
        return 0
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())