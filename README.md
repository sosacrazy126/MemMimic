# MemMimic - Enhanced AI Memory System

**Revolutionary AI memory system that combines sequential thinking with iterative memory retrieval**

## What It Does

MemMimic provides AI assistants with **cognitive memory capabilities** that go far beyond simple storage and retrieval. It thinks with memories as cognitive substrate, mimicking human memory processes.

**🧠 Core Innovation**: Enhanced sequential thinking that explores, refines, synthesizes, and validates understanding through iterative memory discovery.

### Key Features
- **Enhanced Think-with-Memory**: Sequential thinking patterns combined with memory retrieval
- **Markdown Storage**: Human-readable memory files with YAML frontmatter
- **CXD Classification**: Automatic cognitive categorization (CONTROL/CONTEXT/DATA)
- **Storage Adapter Pattern**: Flexible backend support (SQLite/Markdown/Hybrid)
- **MCP Integration**: Seamless Claude Code integration with 11 specialized tools
- **Narrative Tales**: Weave memories into coherent stories

## Architecture Evolution

**From Fragmented → Clean Cognitive System**

1. **Original**: 6 fragmented SQLite databases, opaque storage, complex unused modules
2. **Migration**: Complete transition to transparent Markdown storage
3. **Enhancement**: Sequential thinking integration with iterative memory retrieval
4. **Optimization**: Removed dead code, streamlined to essential components (~1600 lines)
5. **Result**: Clean, focused cognitive memory that thinks, not just stores

## Installation

### Prerequisites
- Python 3.10+
- Node.js 16+

### Setup
```bash
git clone https://github.com/user/memmimicc.git
cd memmimicc

# Configure storage
export MEMMIMIC_STORAGE=markdown
export MEMMIMIC_MD_DIR=/path/to/memmimicc

# Install dependencies (Node.js handles Python via spawn)
cd src/memmimic/mcp
npm install
```

### Claude Desktop Integration
```json
{
  "mcpServers": {
    "memmimic": {
      "command": "node",
      "args": ["/path/to/memmimicc/src/memmimic/mcp/server.js"],
      "env": {
        "PYTHONPATH": "/path/to/memmimicc/src"
      }
    }
  }
}
```

## Enhanced Think-with-Memory System

The revolutionary **think_with_memory** tool combines sequential thinking with memory:

```
🔍 EXPLORATION → 🎯 REFINEMENT → 🔗 SYNTHESIS → ✓ VALIDATION
```

### How It Works
1. **Broad Exploration**: Multiple searches to discover relevant memories
2. **Iterative Refinement**: Each search informed by previous discoveries  
3. **Progressive Synthesis**: Connects insights across memory fragments
4. **Confidence Validation**: Stops when sufficient understanding achieved

### Example Usage
```javascript
// Simple query
think_with_memory("How does the memory system work?")

// Complex analysis  
think_with_memory("What architectural improvements make MemMimic superior?")
```

## MCP Tools (11 Essential Tools)

### 🧠 Enhanced Thinking
- **`think_with_memory`**: Sequential thinking with iterative memory retrieval
- **`remember`**: Store memories with automatic CXD classification
- **`recall_cxd`**: Search memories with cognitive filtering
- **`status`**: System health and memory statistics

### 📖 Narrative Management
- **`tales`**: List and search narrative stories
- **`save_tale`**: Create coherent narratives from memories
- **`load_tale`**: Retrieve specific tales
- **`delete_tale`**: Remove tales with confirmation
- **`context_tale`**: Generate narratives from memory fragments

### 🔧 Advanced Features
- **`analyze_memory_patterns`**: Analyze usage and content relationships
- **`socratic_dialogue`**: Self-questioning for deeper understanding

## Storage Architecture

### Markdown Format
```
memories/
├── 2025/01/12/
│   ├── mem_1754962003621863.md
│   └── mem_1754961775015866.md
├── index.json                    # Fast lookup index
└── tales/
    ├── projects/memmimic/
    └── claude/core/
```

### Memory Structure
```markdown
---
id: mem_1754962003621863
timestamp: 2025-01-12T18:26:43.623491
type: milestone
cxd: CONTROL
---

Successfully fixed all MCP tools! The enhanced think_with_memory system 
is now fully operational through MCP.
```

## Development Journey

**🚀 Major Milestones:**
1. **Discovery**: Found 6 fragmented SQLite databases causing issues
2. **Migration**: Complete SQLite → Markdown transition (76 memories migrated)
3. **Architecture**: Storage adapter pattern with dual backend support
4. **Enhancement**: Sequential thinking integration with iterative retrieval
5. **Integration**: Full MCP tool ecosystem (11 tools operational)
6. **Optimization**: Removed dead code, streamlined architecture (~1600 core lines)
7. **Validation**: Clean, focused system with enhanced cognitive capabilities

## Performance

- **Memory Storage**: 94+ memories in Markdown format
- **Search Speed**: Sub-second keyword matching
- **Thinking Process**: 5-10 sequential thoughts per query
- **Confidence Building**: Progressive understanding to 80-100%
- **MCP Integration**: Seamless Claude Code tool access

## File Structure
```
src/
├── memmimic/
│   ├── mcp/                     # MCP server and Python wrappers
│   │   ├── server.js            # Node.js MCP server
│   │   ├── memmimic_think.py    # Enhanced thinking wrapper
│   │   ├── memmimic_remember.py # Memory storage wrapper
│   │   └── memmimic_*.py        # Other tool wrappers
│   └── tales/                   # Narrative management
├── updated_mcp_tools.py         # Core MemMimic implementation  
├── enhanced_think_with_memory.py # Sequential thinking engine
└── storage_adapter.py           # Storage abstraction layer
```

## Usage Examples

### Basic Memory Operations
```python
# Store a memory
remember("Project milestone completed", "milestone")

# Search memories  
recall_cxd("project completion", limit=5)

# Check system status
status()
```

### Enhanced Thinking
```python
# Simple query
think_with_memory("What did we accomplish?")

# Complex analysis
think_with_memory("How does sequential thinking improve AI cognition?")
```

### Tale Management
```python
# Create a narrative
save_tale("project_journey", "Story of MemMimic development...", "projects/memmimic")

# List all tales
tales()
```

## What Makes It Revolutionary

**Traditional Systems**: Store → Search → Retrieve  
**MemMimic**: Store → Think → Explore → Refine → Synthesize → Validate

The system doesn't just remember - **it thinks with memories as cognitive substrate**, creating true AI cognition that mirrors human memory processes.

## License

GPLv3 - See LICENSE file

## Support

Research-grade software with production-quality implementation. All core functionality tested and operational.

For questions about the sequential thinking architecture or MCP integration, see the source code documentation.