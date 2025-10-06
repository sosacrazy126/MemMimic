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

#### Step 1: Find Your Claude Config File

**Location**:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

#### Step 2: Add MemMimic to MCP Servers

Edit `claude_desktop_config.json` and add the MemMimic server to the `mcpServers` section:

```json
{
  "mcpServers": {
    "memmimic": {
      "command": "node",
      "args": [
        "/absolute/path/to/MemMimic/src/memmimic/mcp/server.js"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/MemMimic/src",
        "MEMMIMIC_STORAGE": "markdown",
        "MEMMIMIC_MD_DIR": "/absolute/path/to/MemMimic/memories"
      }
    }
  }
}
```

**Important**: Replace `/absolute/path/to/MemMimic` with your actual MemMimic installation path.

**Example** (Linux/macOS):
```json
{
  "mcpServers": {
    "memmimic": {
      "command": "node",
      "args": [
        "/home/username/tools-raw/MemMimic/src/memmimic/mcp/server.js"
      ],
      "env": {
        "PYTHONPATH": "/home/username/tools-raw/MemMimic/src",
        "MEMMIMIC_STORAGE": "markdown",
        "MEMMIMIC_MD_DIR": "/home/username/tools-raw/MemMimic/memories"
      }
    }
  }
}
```

**Example** (Windows):
```json
{
  "mcpServers": {
    "memmimic": {
      "command": "node",
      "args": [
        "C:\\Users\\username\\MemMimic\\src\\memmimic\\mcp\\server.js"
      ],
      "env": {
        "PYTHONPATH": "C:\\Users\\username\\MemMimic\\src",
        "MEMMIMIC_STORAGE": "markdown",
        "MEMMIMIC_MD_DIR": "C:\\Users\\username\\MemMimic\\memories"
      }
    }
  }
}
```

#### Step 3: Restart Claude Desktop

Close and reopen Claude Desktop completely. MemMimic tools will now be available with the `mcp__memmimic__` prefix.

#### Verification

In Claude Desktop, you should now have access to:
- `mcp__memmimic__think_with_memory` - Enhanced sequential thinking
- `mcp__memmimic__remember` - Store new memories
- `mcp__memmimic__recall_cxd` - Search with cognitive filtering
- `mcp__memmimic__status` - System health check
- And 7 more tools (see MCP Tools section below)

### Claude Code (CLI) Integration

For the Claude Code CLI tool, use the built-in `claude mcp` command:

```bash
claude mcp add memmimic node /absolute/path/to/MemMimic/src/memmimic/mcp/server.js \
  --scope user \
  -e PYTHONPATH=/absolute/path/to/MemMimic/src \
  -e MEMMIMIC_STORAGE=markdown \
  -e MEMMIMIC_MD_DIR=/absolute/path/to/MemMimic/memories
```

**Example** (Linux/macOS):
```bash
claude mcp add memmimic node /home/username/tools-raw/MemMimic/src/memmimic/mcp/server.js \
  --scope user \
  -e PYTHONPATH=/home/username/tools-raw/MemMimic/src \
  -e MEMMIMIC_STORAGE=markdown \
  -e MEMMIMIC_MD_DIR=/home/username/tools-raw/MemMimic/memories
```

**Verify installation**:
```bash
claude mcp list
# Should show: memmimic: node /path/to/server.js - ✓ Connected
```

MemMimic tools will be available immediately in your next Claude Code session with the `mcp__memmimic__` prefix.

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

### 📖 Narrative Management (Tales System)
- **`tales(query?, stats?, load?, category?, limit?)`**: Unified interface - list, search, or show stats
- **`save_tale(name, content, category?, tags?)`**: Create/update narrative tales
- **`load_tale(name, category?)`**: Retrieve specific tale by name
- **`delete_tale(name, category?, confirm?)`**: Remove tales with confirmation
- **`context_tale(query, style?, max_memories?)`**: Generate narratives from memory fragments

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

### Tales System

Tales are **narrative memory units** that weave scattered memory fragments into coherent stories. They provide persistent identity, accumulated wisdom, and project documentation.

**Tale Storage Structure**:
```
tales/
├── claude/              # Personal continuity (5 fixed subdirs)
│   ├── core/           # Identity and principles
│   ├── contexts/       # Collaboration patterns
│   ├── insights/       # Accumulated wisdom
│   ├── current/        # Active projects
│   └── archive/        # Deleted tales (soft delete)
├── projects/           # Technical documentation (flexible subdirs)
│   └── [project-name]/ # Per-project narratives
└── misc/               # General content (flexible subdirs)
    └── [category]/     # Stories, recipes, creative work
```

**Tale File Format**:
```markdown
<!-- Tale: introduction -->
<!-- Category: claude/core -->
<!-- Created: 2025-10-06T05:40:19.440569 -->
<!-- Updated: 2025-10-06T05:40:19.440569 -->
<!-- Usage: 3 -->
<!-- Size: 752 chars -->
<!-- Version: 1 -->
<!-- Tags: identity, introduction, core -->

I am Claude, an AI assistant created by Anthropic...
[Tale content as plain text]
```

**Use Cases**:
- **Identity Persistence**: Store core principles in `claude/core/identity.txt`
- **Context Management**: Save collaboration patterns in `claude/contexts/project_x.txt`
- **Wisdom Accumulation**: Document insights in `claude/insights/patterns.txt`
- **Project Documentation**: Technical narratives in `projects/memmimic/architecture.txt`
- **Story Generation**: Use `context_tale()` to weave memories into coherent narratives

**Example Workflow**:
```javascript
// Create identity tale
save_tale(
  "introduction",
  "I am Claude, an AI assistant...",
  "claude/core",
  "identity,introduction"
)

// Later session - reload identity
load_tale("introduction", "claude/core")

// Generate narrative from memories
context_tale(
  "How did we improve MemMimic?",
  "technical",
  15
)
```

## Development Journey

**🚀 Major Milestones:**
1. **Discovery**: Found 6 fragmented SQLite databases causing issues
2. **Migration**: Complete SQLite → Markdown transition (76 memories migrated)
3. **Architecture**: Storage adapter pattern with dual backend support
4. **Enhancement**: Sequential thinking integration with iterative retrieval
5. **Integration**: Full MCP tool ecosystem (11 tools operational)
6. **Optimization**: Removed dead code, streamlined architecture (~1600 core lines)
7. **Tales Restoration**: Added complete narrative management system
8. **Validation**: Clean, focused system with enhanced cognitive capabilities

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
│   │   └── memmimic_*.py        # Other tool wrappers (tales, etc.)
│   └── tales/                   # Narrative management system
│       └── tale_manager.py      # Core tales implementation
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