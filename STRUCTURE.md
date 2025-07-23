# MemMimic Project Structure

## Overview
This document describes the streamlined structure of the MemMimic project after architecture cleanup, focusing on core memory management functionality.

## Root Directory Structure

```
memmimic/
├── .archive/                    # Archived temporary and historical files
├── .github/                     # GitHub workflows and templates
├── config/                      # Configuration files
│   ├── memmimic_config.yaml    # Main configuration
│   └── performance_config.yaml  # Performance settings
├── docs/                        # Comprehensive documentation
│   ├── analysis/               # Technical analysis documents
│   ├── images/                 # Documentation assets
│   ├── prd/                    # Product requirements documents
│   ├── reports/                # Generated reports and summaries
│   └── *.md                    # Core documentation files
├── examples/                    # Usage examples and integrations
├── scripts/                     # Utility and maintenance scripts
├── src/                         # Source code
│   └── memmimic/               # Main Python package
│       ├── cxd/                # CXD classification system
│       ├── errors/             # Error handling framework
│       ├── local/              # Local client
│       ├── mcp/                # MCP server tools (13 tools)
│       ├── memory/             # Memory management system (AMMS)
│       ├── tales/              # Narrative management
│       └── utils/              # Utility functions
├── tales/                       # Memory narratives and stories
├── tests/                       # Comprehensive test suite
└── Core Files
    ├── CHANGELOG.md            # Version history
    ├── CLAUDE.md               # Claude integration instructions
    ├── LICENSE                 # MIT License
    ├── PROJECT_COMPLETION_SUMMARY.md  # Transformation summary
    ├── README.md               # Main project documentation
    ├── pyproject.toml          # Python project configuration
    └── requirements.txt        # Python dependencies
```

## Key Components

### Source Code (`src/memmimic/`)
- **CXD**: Control/Context/Data classification system
- **Memory**: AMMS (Active Memory Management System) with quality gates
- **MCP**: 13 Model Context Protocol tools for memory operations
- **Tales**: Narrative management and context generation
- **Errors**: Structured error handling and logging
- **Utils**: Caching and utility functions

### Documentation (`docs/`)
- **Core Docs**: README, Architecture, API reference, Quick start
- **Quality Gates**: Comprehensive quality control documentation
- **Analysis**: Technical deep-dive documents
- **Images**: Documentation assets and diagrams

### MCP Integration (`src/memmimic/mcp/`)
- **13 Core Tools**: Memory management, search, and analytics
- **Node.js Server**: MCP protocol bridge for Claude Desktop
- **Python Handlers**: Async/sync compatibility layer
- **Performance Monitoring**: Real-time metrics and optimization

### Testing (`tests/`)
- Memory system unit tests
- CXD classification tests
- MCP integration tests
- Performance benchmarks
- Quality gate validation

## Cache Directories
Runtimegenerated:
- `cxd_cache/`: CXD classification model cache
- `memmimic_cache/`: Memory operation cache
- `*.db`: SQLite database files

## Core Features
✅ AMMS memory management system  
✅ Quality gate with semantic similarity  
✅ CXD cognitive classification  
✅ 13 production-ready MCP tools  
✅ High-performance caching  
✅ Streamlined architecture  
✅ Complete documentation  

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Configure Claude Desktop MCP integration
3. Initialize database: `python -c "from memmimic import create_memmimic; create_memmimic('memmimic.db')"`
4. Test with: `python -m memmimic.mcp.memmimic_status`

---
*Updated after architecture cleanup - July 2025*