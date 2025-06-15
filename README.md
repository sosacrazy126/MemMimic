# MemMimic

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/xprooket/memmimic)

<div align="center">

![MemMimic Logo](docs/images/MemMimic.png)

</div>

Persistent contextual memory system for AI assistants via Model Context Protocol (MCP).

## What It Does

MemMimic provides AI assistants with persistent memory that survives across conversations. It combines semantic search, cognitive classification, and narrative management to maintain context over time.

**Core capabilities:**
- Store and retrieve memories with semantic + keyword search
- Automatic cognitive function classification (Control/Context/Data)
- Generate coherent narratives from memory fragments
- Self-reflective analysis and pattern recognition

![MemMimic Architecture](docs/images/bluePrint.png)

## Installation

### Prerequisites
- Python 3.10+
- Node.js 16+

### Setup
```bash
git clone https://github.com/xprooket/memmimic.git
cd memmimic

# Install Python dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Node.js dependencies for MCP server
cd src/memmimic/mcp
npm install
```

### Claude Desktop Integration
Add to your Claude Desktop MCP settings:

```json
{
  "mcpServers": {
    "memmimic": {
      "command": "node",
      "args": ["path/to/memmimic/src/memmimic/mcp/server.js"],
      "env": {
        "PYTHONPATH": "path/to/memmimic/src"
      }
    }
  }
}
```

## API Reference

MemMimic provides 11 essential tools organized by function:

### 🔍 Search
**`recall_cxd(query, function_filter?, limit?, db_name?)`**
Hybrid semantic + keyword memory search with cognitive filtering.

```
recall_cxd("project architecture decisions")
recall_cxd("error handling", function_filter="CONTROL", limit=3)
```

### 🧠 Memory Management
**`remember(content, memory_type?)`**
Store information with automatic cognitive classification.

```
remember("User prefers technical documentation over tutorials", "interaction")
remember("Project completed successfully", "milestone")
```

**`think_with_memory(input_text)`**
Process input with full memory context.

```
think_with_memory("How should we approach the database migration?")
```

**`status()`**
System health and memory statistics.

### 📖 Narrative Management
**`tales(query?, stats?, load?, category?, limit?)`**
Unified interface for tale management.

```
tales()                                    # List all tales
tales("project history")                   # Search tales
tales(stats=true)                         # Collection statistics
tales("intro", load=true)                 # Load specific tale
```

**`save_tale(name, content, category?, tags?)`**
Create or update narrative tales.

```
save_tale("project_overview", "Brief project description", "projects/main")
```

**`load_tale(name, category?)`**
Load specific tale by name.

**`delete_tale(name, category?, confirm?)`**
Delete tale with optional confirmation.

**`context_tale(query, style?, max_memories?)`**
Generate narrative from memory fragments.

```
context_tale("project introduction", "technical", 10)
```

### 🔧 Advanced Memory Operations
**`update_memory_guided(memory_id)`**
Update memory with Socratic guidance.

**`delete_memory_guided(memory_id, confirm?)`**
Delete memory with guided analysis.

**`analyze_memory_patterns()`**
Analyze usage patterns and content relationships.

### 🧘 Cognitive Tools
**`socratic_dialogue(query, depth?)`**
Self-questioning for deeper understanding.

```
socratic_dialogue("Why did this approach fail?", 3)
```

## Architecture

```
src/memmimic/
├── memory/           # Core memory management
├── cxd/             # Cognitive classification system
├── tales/           # Narrative management
├── mcp/             # Model Context Protocol tools
└── api.py           # Main API interface
```

**Key components:**
- **Memory Store**: SQLite-based persistent storage
- **CXD Classifier**: Cognitive function detection (Control/Context/Data)
- **Tale Manager**: Narrative organization with v2.0 structure
- **Semantic Search**: Sentence transformers + FAISS vector store
- **MCP Bridge**: JavaScript-Python integration

## Configuration

MemMimic works out of the box with sensible defaults. Advanced configuration available via:

- `src/memmimic/cxd/config/cxd_config.yaml` - Classification settings
- Environment variables: `CXD_CONFIG`, `CXD_CACHE_DIR`, `CXD_MODE`

## Memory Types

- `interaction` - Conversational exchanges
- `milestone` - Important project events
- `reflection` - Analysis and insights
- `synthetic` - Pre-loaded knowledge
- `socratic` - Self-questioning dialogues

## Tale Categories

- `claude/core` - Personal identity and principles
- `claude/contexts` - Collaboration contexts
- `claude/insights` - Accumulated wisdom
- `projects/*` - Technical documentation by project
- `misc/*` - General content

## Development

```bash
# Run tests
cd src && python -m pytest tests/

# Test MCP integration
node src/memmimic/mcp/server.js

# Verify installation
python -c "from memmimic.api import create_memmimic; mm = create_memmimic(':memory:'); print(mm.status())"
```

## Performance

- **Memory retrieval**: Sub-second for most queries
- **Semantic indexing**: ~1 second for 200+ memories
- **Storage**: SQLite with automatic optimization
- **Caching**: Persistent embeddings and vector indexes

## ⚠️ Usage Considerations

**Claude API Rate Limits**: MemMimic's conversational and memory operations may approach Anthropic's usage limits during intensive sessions. The system performs multiple API calls for:
- Memory classification and storage
- Semantic search operations
- Socratic dialogue generation
- Memory pattern analysis

Consider this for production deployments and monitor your usage accordingly.

## Limitations

- Requires Model Context Protocol support
- Memory grows over time (no automatic cleanup)
- Semantic search quality depends on content similarity
- CXD classification optimized for English text
- May consume significant API quota during heavy usage

## License

Apache License 2.0

## Support

This is research-grade software. It works reliably for its intended use cases but isn't enterprise production-ready. Use as foundation for more robust implementations.

For technical questions, see source code documentation.
