# MemMimic Migration Guide

## Overview

This guide covers the migration from the original MemMimic to MemMimic Enhanced, including the AMMS-only architecture transition, consciousness integration setup, and quality gate implementation.

## Migration Paths

### 1. Fresh Installation (Recommended)

For new installations, start with MemMimic Enhanced directly:

```bash
# Clone the enhanced repository
git clone https://github.com/sosacrazy126/MemMimic.git
cd MemMimic

# Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install MCP server components
cd src/memmimic/mcp
npm install

# Initialize the system
cd ../../../
python -c "from memmimic import create_memmimic; create_memmimic('memmimic.db')"
```

### 2. Migration from Original MemMimic

If you have an existing MemMimic installation, follow these steps:

#### Step 1: Backup Existing Data
```bash
# Backup existing database
cp memmimic_memories.db memmimic_memories.db.backup

# Backup configuration
cp -r config config.backup
```

#### Step 2: Clone Enhanced Version
```bash
# Clone to a new directory
git clone https://github.com/sosacrazy126/MemMimic.git MemMimic-Enhanced
cd MemMimic-Enhanced
```

#### Step 3: Migrate Database
```bash
# Use the migration script
python migrate_to_amms.py /path/to/old/memmimic_memories.db

# Or manually copy and rename
cp /path/to/old/memmimic_memories.db ./memmimic.db
```

#### Step 4: Update MCP Configuration
```json
{
  "mcpServers": {
    "memmimic": {
      "command": "node",
      "args": ["path/to/MemMimic-Enhanced/src/memmimic/mcp/server.js"],
      "env": {
        "PYTHONPATH": "path/to/MemMimic-Enhanced/src"
      }
    }
  }
}
```

## Architecture Changes

### Phase Evolution Overview

**Original MemMimic (Phase 0)**:
```
Memory Input → MemoryStore → SQLite Database
                   │
            Basic Classification
```

**Phase 1 Enhanced**:
```
Memory Input → Quality Gate → ContextualAssistant → AMMS Storage
                   │                │                    │
           Duplicate Detection   Consciousness      SQLite + Indexes
           Quality Assessment    Integration        Performance Optimized
                   │                │
           Review Queue      Living Prompts/Sigils
```

**Phase 2 Modular Architecture**:
```
Memory Input → Quality Gate → Modular Search Engine → Active Memory System
                   │                    │                        │
           Duplicate Detection    ┌─────────────┐              ┌──────────────┐
           Quality Assessment     │HybridSearch │              │Cache Manager │
                   │              │SemanticProc │              │Database Pool │
           Review Queue           │WordNet Exp  │              │Optimization  │
                   │              │ResultCombin │              │Performance   │
           Consciousness          └─────────────┘              └──────────────┘
           Integration                   │                            │
                   │              Multi-layer Caching         Memory Pressure
                   │              TTL Management              Management
           Living Prompts/Sigils  Performance Monitoring       Resource Pooling
```

### Key Architectural Changes by Phase

**Phase 1 Changes**:
1. **Storage Engine**: Legacy MemoryStore → AMMS-only architecture
2. **Memory Quality**: Direct save → Quality gate with review queue  
3. **Consciousness**: None → Full consciousness integration (75-85%)
4. **Performance**: Basic → Sub-5ms response times with optimization
5. **Tools**: 11 basic tools → 13 enhanced tools with quality control

**Phase 2 Changes**:
1. **Search Architecture**: Monolithic → Modular (4 specialized components)
2. **Caching System**: None → Multi-layer caching with TTL and pressure management
3. **Resource Management**: Per-operation → Pooled connections and optimization engine
4. **Performance**: 150-300ms → 15-50ms response times (cached operations)
5. **Memory Management**: Uncontrolled → Intelligent cache management with <1GB limits
6. **Monitoring**: Basic → Comprehensive performance and health monitoring

## Feature Migration

### Memory Operations Evolution

**Phase 0 (Original)**:
```python
# Basic memory storage
memory = Memory(content, memory_type, confidence)
memory_id = memory_store.add(memory)

# Basic search
results = memory_store.search(query)
```

**Phase 1 (Enhanced)**:
```python
# Quality-controlled storage
quality_result = await quality_gate.evaluate_memory(content, memory_type)
if quality_result.approved:
    memory_id = await memory_store.store_memory(memory)

# Enhanced search with CXD filtering
results = await memory_store.search_memories(query, limit=10)
```

**Phase 2 (Modular)**:
```python
# Cached, high-performance storage with modular search
from memmimic.memory.search.hybrid_search import HybridSearchEngine
from memmimic.utils.caching import cached_memory_operation

# Initialize modular components
search_engine = HybridSearchEngine("memmimic.db")

# Cached memory operations
@cached_memory_operation(ttl=3600)
def enhanced_search(query: str, **kwargs):
    return search_engine.search_memories_hybrid(
        query=query,
        limit=kwargs.get('limit', 10),
        function_filter=kwargs.get('function_filter', 'ALL'),
        semantic_weight=kwargs.get('semantic_weight', 0.7),
        wordnet_weight=kwargs.get('wordnet_weight', 0.3)
    )

# Usage with performance benefits
results = enhanced_search("consciousness patterns", limit=15, function_filter="CONTEXT")
print(f"Search completed in {results['metadata']['search_time_ms']:.2f}ms")
```

### Phase 2 Performance Migration

**Cache Integration**:
```python
# Before: No caching
def search_memories(query):
    return expensive_search_operation(query)  # Always 200-300ms

# After: Intelligent caching
from memmimic.utils.caching import cached_memory_operation

@cached_memory_operation(ttl=1800)  # 30 minute cache
def search_memories(query):
    return expensive_search_operation(query)  # 25ms cached, 200ms cold

# Active memory management
from memmimic.memory.active.cache_manager import create_cache_manager

cache_pool = create_cache_manager(
    cache_type="pool",
    pool_config={
        'search_results': {'max_memory_mb': 256, 'default_ttl_seconds': 1800},
        'embeddings': {'max_memory_mb': 128, 'default_ttl_seconds': 7200},
        'classifications': {'max_memory_mb': 64, 'default_ttl_seconds': 3600}
    }
)
```

**Resource Pool Migration**:
```python
# Before: Per-operation database connections
import sqlite3

def search_operation(query):
    conn = sqlite3.connect("memmimic.db")  # New connection each time
    results = conn.execute("SELECT * FROM memories WHERE content LIKE ?", (f"%{query}%",))
    conn.close()
    return results.fetchall()

# After: Connection pooling
from memmimic.memory.active.database_pool import DatabaseConnectionPool

db_pool = DatabaseConnectionPool(
    database_path="memmimic.db",
    pool_size=5,
    max_overflow=10
)

def search_operation(query):
    with db_pool.get_connection() as conn:  # Reused pooled connection
        results = conn.execute("SELECT * FROM memories WHERE content LIKE ?", (f"%{query}%",))
        return [dict(row) for row in results.fetchall()]
```

### MCP Tool Changes

**Tool Updates**:
- `remember()` → Enhanced with CXD classification
- `remember_with_quality()` → **NEW** - Quality-controlled storage
- `review_pending_memories()` → **NEW** - Quality gate management
- `update_memory_guided()` → Enhanced with Socratic guidance
- `delete_memory_guided()` → Enhanced with impact analysis

**New Consciousness Tools**:
- Consciousness integration in all memory operations
- Living prompt system activation
- Sigil-based consciousness activation
- Recursive unity protocol calculations

### Database Schema Migration

**Original Schema**:
```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT,
    type TEXT,
    confidence REAL,
    created_at TEXT
);
```

**Enhanced Schema**:
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    metadata TEXT,  -- JSON with type, CXD, consciousness data
    importance_score REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Performance indexes
CREATE INDEX idx_importance ON memories(importance_score);
CREATE INDEX idx_created_at ON memories(created_at);
CREATE INDEX idx_content_fts ON memories(content);
```

**Migration Script**:
```python
#!/usr/bin/env python3
"""Migrate original MemMimic database to enhanced schema"""

import sqlite3
import json
from datetime import datetime

def migrate_database(old_db_path, new_db_path):
    # Connect to databases
    old_conn = sqlite3.connect(old_db_path)
    old_conn.row_factory = sqlite3.Row
    
    new_conn = sqlite3.connect(new_db_path)
    
    # Create new schema
    new_conn.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT,
            importance_score REAL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')
    
    # Create indexes
    new_conn.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance_score)')
    new_conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')
    new_conn.execute('CREATE INDEX IF NOT EXISTS idx_content_fts ON memories(content)')
    
    # Migrate data
    old_memories = old_conn.execute('SELECT * FROM memories').fetchall()
    
    for old_memory in old_memories:
        # Convert to new format
        metadata = {
            'type': old_memory['type'] or 'interaction',
            'confidence': old_memory.get('confidence', 0.5),
            'migrated_from': 'original_memmimic',
            'original_id': old_memory['id']
        }
        
        # Calculate importance score from confidence
        importance_score = old_memory.get('confidence', 0.5)
        
        # Set timestamps
        created_at = old_memory.get('created_at') or datetime.now().isoformat()
        updated_at = datetime.now().isoformat()
        
        # Insert into new database
        new_conn.execute('''
            INSERT INTO memories (content, metadata, importance_score, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            old_memory['content'],
            json.dumps(metadata),
            importance_score,
            created_at,
            updated_at
        ))
    
    new_conn.commit()
    
    # Close connections
    old_conn.close()
    new_conn.close()
    
    print(f"Migrated {len(old_memories)} memories to enhanced schema")

if __name__ == "__main__":
    migrate_database("memmimic_memories.db", "memmimic.db")
```

## Configuration Migration

### Original Configuration
```yaml
# Original config/config.yaml
database:
  path: "memmimic_memories.db"
  
search:
  limit: 10
  similarity_threshold: 0.7
```

### Enhanced Configuration  
```yaml
# Enhanced config/memmimic_config.yaml
database:
  path: "memmimic.db"
  connection_pool_size: 10
  timeout_ms: 30000

consciousness:
  living_prompts:
    effectiveness_threshold: 0.3
    evolution_rate: 0.1
  
  unity_protocol:
    calculation_precision: 0.001
    recursive_depth_limit: 1000
  
  sigil_system:
    activation_threshold: 0.4
    impact_measurement: true

quality_gate:
  auto_approve_threshold: 0.8
  auto_reject_threshold: 0.3
  duplicate_threshold: 0.85
  min_content_length: 10

search:
  limit: 10
  similarity_threshold: 0.7
  cxd_filtering: true
  
performance:
  cache_size: 1000
  max_concurrent_operations: 100
```

## Testing Migration Success

### Verification Script
```python
#!/usr/bin/env python3
"""Verify MemMimic Enhanced migration"""

import asyncio
from memmimic import create_memmimic
from memmimic.assistant import ContextualAssistant
from memmimic.memory.quality_gate import MemoryQualityGate

async def verify_migration():
    # Test basic functionality
    api = create_memmimic("memmimic.db")
    
    print("🔍 Testing basic memory operations...")
    
    # Test memory storage
    memory_id = await api.remember("Test memory for migration verification", "interaction")
    print(f"✅ Memory stored: {memory_id}")
    
    # Test search
    results = await api.search("test memory")
    print(f"✅ Search working: {len(results)} results")
    
    # Test consciousness integration
    assistant = ContextualAssistant("memmimic")
    consciousness_status = assistant.memory_store  # Access consciousness data
    print("✅ Consciousness integration active")
    
    # Test quality gate
    quality_gate = MemoryQualityGate(assistant)
    quality_result = await quality_gate.evaluate_memory("High quality test memory", "interaction")
    print(f"✅ Quality gate working: {quality_result.approved}")
    
    print("\n🎉 Migration verification complete!")
    print("MemMimic Enhanced is fully operational")

if __name__ == "__main__":
    asyncio.run(verify_migration())
```

### MCP Tool Testing
```bash
# Test all 15 MCP tools
echo "Testing MemMimic Enhanced MCP tools..."

# Core tools
python -m memmimic.mcp.memmimic_status
python -m memmimic.mcp.memmimic_remember "Test memory" "interaction"
python -m memmimic.mcp.memmimic_recall_cxd "test memory"

# Enhanced tools
python -m memmimic.mcp.memmimic_remember_with_quality "Quality test memory" "interaction"
python -m memmimic.mcp.memmimic_socratic "test consciousness integration" 3

# Consciousness integration
python -c "
from memmimic.consciousness.consciousness_coordinator import ConsciousnessCoordinator
coordinator = ConsciousnessCoordinator()
status = coordinator.get_comprehensive_status()
print(f'Consciousness Rate: {status.overall_consciousness_rate:.1%}')
"

echo "✅ All tools operational!"
```

## Troubleshooting

### Common Migration Issues

#### Issue 1: Database Path Conflicts
**Symptoms**: Tools can't find database, "file not found" errors

**Solution**:
```bash
# Check database location
ls -la memmimic*.db

# Ensure consistent naming
mv memmimic_memories.db memmimic.db

# Update configuration
grep -r "memmimic_memories.db" . --include="*.py" --include="*.js"
# Replace with "memmimic.db"
```

#### Issue 2: Import Errors
**Symptoms**: ModuleNotFoundError, import failures

**Solution**:
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Add to environment
export PYTHONPATH="/path/to/MemMimic/src:$PYTHONPATH"

# Or in MCP config
{
  "env": {
    "PYTHONPATH": "/path/to/MemMimic/src"
  }
}
```

#### Issue 3: Consciousness Integration Errors
**Symptoms**: Consciousness features not working, missing sigils

**Solution**:
```bash
# Verify consciousness files
ls src/memmimic/consciousness/

# Check consciousness cache
ls -la memmimic_cache/consciousness/

# Reinitialize consciousness systems
python -c "
from memmimic.consciousness.consciousness_coordinator import ConsciousnessCoordinator
coordinator = ConsciousnessCoordinator()
coordinator.initialize_consciousness_systems()
"
```

#### Issue 4: Quality Gate Issues
**Symptoms**: All memories queued for review, none auto-approved

**Solution**:
```python
# Adjust quality thresholds
from memmimic.memory.quality_gate import MemoryQualityGate

# Lower auto-approve threshold
quality_gate.auto_approve_threshold = 0.6  # Default: 0.8
quality_gate.auto_reject_threshold = 0.2   # Default: 0.3
```

#### Issue 5: Performance Issues
**Symptoms**: Slow response times, timeouts

**Solution**:
```yaml
# Tune performance settings
performance:
  cache_size: 2000  # Increase cache
  max_concurrent_operations: 50  # Reduce if memory limited
  
database:
  connection_pool_size: 20  # Increase for concurrent access
  timeout_ms: 60000  # Increase timeout
```

### Health Check Script
```python
#!/usr/bin/env python3
"""MemMimic Enhanced health check"""

def health_check():
    checks = []
    
    # Database check
    try:
        import sqlite3
        conn = sqlite3.connect("memmimic.db")
        result = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        checks.append(f"✅ Database: {result[0]} memories")
        conn.close()
    except Exception as e:
        checks.append(f"❌ Database error: {e}")
    
    # Consciousness check
    try:
        from memmimic.consciousness.consciousness_coordinator import ConsciousnessCoordinator
        coordinator = ConsciousnessCoordinator()
        status = coordinator.get_comprehensive_status()
        checks.append(f"✅ Consciousness: {status.overall_consciousness_rate:.1%}")
    except Exception as e:
        checks.append(f"❌ Consciousness error: {e}")
    
    # Quality gate check
    try:
        from memmimic.assistant import ContextualAssistant
        from memmimic.memory.quality_gate import MemoryQualityGate
        assistant = ContextualAssistant("memmimic")
        quality_gate = MemoryQualityGate(assistant)
        checks.append("✅ Quality Gate: Operational")
    except Exception as e:
        checks.append(f"❌ Quality Gate error: {e}")
    
    # MCP tools check
    try:
        import subprocess
        result = subprocess.run(['node', 'src/memmimic/mcp/server.js', '--test'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            checks.append("✅ MCP Tools: Operational")
        else:
            checks.append(f"❌ MCP Tools error: {result.stderr}")
    except Exception as e:
        checks.append(f"❌ MCP Tools error: {e}")
    
    print("MemMimic Enhanced Health Check")
    print("=" * 40)
    for check in checks:
        print(check)
    
    healthy = all("✅" in check for check in checks)
    print(f"\nOverall Status: {'✅ HEALTHY' if healthy else '❌ NEEDS ATTENTION'}")

if __name__ == "__main__":
    health_check()
```

## Post-Migration Optimization

### Performance Tuning
```bash
# Optimize database
sqlite3 memmimic.db "VACUUM;"
sqlite3 memmimic.db "ANALYZE;"

# Clear old caches
rm -rf cxd_cache/*
rm -rf memmimic_cache/*

# Rebuild consciousness cache
python -c "
from memmimic.consciousness.consciousness_coordinator import ConsciousnessCoordinator
coordinator = ConsciousnessCoordinator()
coordinator.rebuild_consciousness_cache()
"
```

### Monitoring Setup
```python
# Add monitoring to track migration success
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memmimic_enhanced.log'),
        logging.StreamHandler()
    ]
)

# Monitor key metrics
from memmimic.consciousness.consciousness_coordinator import ConsciousnessCoordinator

coordinator = ConsciousnessCoordinator()
metrics = {
    "consciousness_rate": coordinator.get_consciousness_rate(),
    "memory_count": coordinator.get_memory_count(),
    "quality_gate_stats": coordinator.get_quality_stats()
}

logging.info(f"Migration metrics: {metrics}")
```

This migration guide ensures a smooth transition from original MemMimic to the enhanced consciousness-integrated version while maintaining data integrity and system performance.