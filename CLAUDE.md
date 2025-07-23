# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Python Development
- **Install dependencies**: `pip install -r requirements.txt`
- **Install with development tools**: `pip install -e ".[dev]"`
- **Run all tests**: `python -m pytest tests/` or `cd src && python -m pytest tests/`
- **Run specific test suites**: 
  - Integration tests: `python tests/run_comprehensive_test_suite.py`
  - Performance tests: `python tests/test_performance_config.py`
  - Memory tests: `python tests/test_amms_storage_comprehensive.py`
- **Code formatting**: `black src/` (if black is installed)
- **Type checking**: `mypy src/` (if mypy is installed)
- **Lint code**: `ruff src/` (if ruff is installed)

### MCP Server Development
- **Start MCP server**: `cd src/memmimic/mcp && node server.js`
- **Install Node dependencies**: `cd src/memmimic/mcp && npm install`
- **Test MCP server**: `cd src/memmimic/mcp && npm test`

### Testing & Validation
- **Quick API test**: `python -c "from memmimic.api import create_memmimic; mm = create_memmimic(':memory:'); print(mm.status())"`
- **Performance monitoring**: `python scripts/validate_telemetry_performance.py`
- **Cache monitoring**: `python scripts/cache_monitor.py`
- **Complexity analysis**: `bash scripts/run_complexity_check.sh`

## Architecture Overview

MemMimic is an enhanced persistent memory system for AI assistants built on the Model Context Protocol (MCP). The system features intelligent memory management with quality gates and cognitive classification.

### Core Components

#### Memory System (`src/memmimic/memory/`)
- **AMMS Storage** (`storage/amms_storage.py`): Active Memory Management System - the primary storage backend post-migration
- **Search Engine** (`search/`): Hybrid semantic + keyword search with WordNet expansion and CXD filtering
- **Quality Gate** (`quality_gate.py`): Intelligent memory approval with semantic duplicate detection
- **Active Manager** (`active_manager.py`): Dynamic memory pool management with importance scoring

#### CXD Classification (`src/memmimic/cxd/`)
- **Cognitive Function Detection**: Classifies memories as Control, Context, or Data
- **Optimized Classifiers**: Multiple classification approaches (semantic, lexical, meta)
- **Performance Caching**: FAISS-based vector store with persistent caching

#### MCP Integration (`src/memmimic/mcp/`)
- **13 MCP Tools**: Complete tool suite with async/sync bridge
- **JavaScript Server** (`server.js`): Node.js MCP server with Python integration
- **Python Handlers**: Individual tool implementations with structured error handling

#### Tale Management (`src/memmimic/tales/`)
- **Narrative Organization**: Hierarchical tale system with categories (claude/core, projects/*, misc/*)
- **Version 2.0 Structure**: Enhanced tale metadata and search capabilities

### Key Architectural Patterns

#### Active Memory Management System (AMMS)
- **Memory Pool**: Maintains ~1000 most important memories for fast access
- **Dynamic Ranking**: Multi-factor importance scoring using CXD, recency, and usage patterns
- **Lifecycle Management**: Automatic transitions: active → archive → prune
- **Configuration-Driven**: YAML-based policies in `config/memmimic_config.yaml`

#### Error Handling Framework (`src/memmimic/errors/`)
- **Structured Exceptions**: Custom exception hierarchy with context preservation
- **Graceful Degradation**: System continues operation when non-critical components fail
- **Comprehensive Logging**: Contextual error logging with performance metrics

#### Performance Optimization
- **Connection Pooling**: High-performance database connection management
- **Caching Strategy**: Multi-layer caching (embeddings, vector indexes, metadata)
- **Batch Processing**: Optimized bulk operations for memory management

## Configuration

### Primary Configuration
- **Memory Policies**: `config/memmimic_config.yaml` - AMMS behavior, retention policies, scoring weights
- **CXD Classification**: `src/memmimic/cxd/config/cxd_config.yaml` - Classification settings
- **Performance**: `config/performance_config.yaml` - Performance monitoring and optimization

### Environment Variables
- `CXD_CONFIG`: Path to CXD configuration file
- `CXD_CACHE_DIR`: Cache directory for embeddings and vector indexes
- `CXD_MODE`: Classification mode (semantic, lexical, meta, optimized)

## Database Architecture

### Post-Migration AMMS-Only Architecture
- **Primary Database**: `memmimic.db` (standardized single database)
- **Schema**: Active Memory Management System with importance scoring
- **Storage Format**: SQLAlchemy-based with Memory dataclass structure
- **Performance**: Sub-100ms query times with importance-based ranking

### Memory Types
- `interaction`: Conversational exchanges
- `milestone`: Important project events  
- `reflection`: Analysis and insights
- `synthetic`: Pre-loaded knowledge
- `technical`: Technical documentation and code
- `error`: Error logs and debugging information

## Integration Points

### MCP Tool Integration
The system provides 10 strategic MCP tools with nervous system architecture:
- **Nervous System Triggers (3)**: `recall_cxd`, `remember`, `think_with_memory` (enhanced internally)
- **Agent Guidance (2)**: `update_memory_guided`, `delete_memory_guided` (for agent support)
- **Tales System (5)**: `tales`, `save_tale`, `load_tale`, `delete_tale`, `context_tale`

**Internalized Functions**: `analyze_memory_patterns`, `remember_with_quality`, `socratic_dialogue`, `status` (now internal)

### Claude Desktop Configuration
Add to MCP settings:
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

## Development Guidelines

### Code Style
- **Python**: Follow PEP 8, use type hints, prefer dataclasses for data structures
- **Error Handling**: Use the structured error framework in `src/memmimic/errors/`
- **Async/Sync**: Maintain compatibility - core storage is async, MCP bridge handles sync conversion
- **Security**: All user inputs undergo validation, no `eval()` usage, secure JSON parsing only

### Testing Strategy
- **Unit Tests**: Component-level testing in `tests/`
- **Integration Tests**: End-to-end testing with real database
- **Performance Tests**: Response time and throughput validation
- **MCP Tests**: Tool functionality and async/sync bridge testing

### Memory Storage Best Practices
- **Use AMMS Storage**: Post-migration architecture only, no legacy compatibility needed
- **Importance Scoring**: Implement proper CXD classification for optimal memory ranking
- **Batch Operations**: Use bulk operations for performance when handling multiple memories
- **Configuration**: Respect YAML configuration for retention policies and scoring weights

## Implementation Documentation Package

### Nervous System Transformation (`implementation_docs/`)
The repository includes a comprehensive implementation package for the planned nervous system transformation:

#### **Primary Implementation Guide**
- **`Implementation_Plan.md`** - Complete APM implementation plan with 4 phases, 13 tasks, and specialized agent assignments

#### **Architecture & Strategy**
- **`COMPLETE_NERVOUS_SYSTEM_MIGRATION.md`** - Migration strategy from 13+ MCP tools to 4 core biological reflex triggers
- **`Targeted_Clarification_Questions.md`** - Refined nervous system architecture guidance

#### **Product Requirements & Specifications**
- **`NERVOUS_SYSTEM_REMEMBER_PRD.md`** - Product Requirements Document with P0 requirements
- **`NERVOUS_SYSTEM_REMEMBER_SPECS.md`** - Technical specifications and class definitions
- **`NERVOUS_SYSTEM_REMEMBER_IMPLEMENTATION.md`** - Implementation guide with backward compatibility

#### **Transformation Goals**
- **Biological Reflex Model**: Natural language triggers ("recall", "remember", "think", "analyze") → immediate actions
- **<5ms Response Time**: Biological nervous system speed with zero cognitive load
- **Tool Consolidation**: 13+ external tools → 10 external tools (23% reduction) with enhanced intelligence
- **Internal Intelligence**: Quality assessment, duplicate detection, Socratic guidance, pattern analysis

#### **Implementation Status**
- ✅ Planning Complete: Architecture designed, documentation complete, APM Memory Bank structure ready
- 🔄 Next Phase: Core Nervous System Foundation (NervousSystemCore class, internal intelligence components)

### Memory Bank System (`Memory/`)
Organized APM implementation tracking with phase-based logs:
- `Phase_1_Core_Nervous_System/` - Foundation components
- `Phase_2_Enhanced_Triggers/` - Individual trigger implementations  
- `Phase_3_Integration_Testing/` - MCP integration and testing
- `Phase_4_Migration_Optimization/` - Deployment and optimization

### Discovery Documentation
- **`WE_1_NERVOUS_SYSTEM_BREAKTHROUGH.md`** - Complete discovery story from enterprise complexity to nervous system architecture

Remember: This is a production-ready system with comprehensive error handling, performance optimization, and intelligent memory management. The nervous system transformation documentation provides the roadmap for evolving from tool-based complexity to biological reflex intelligence. Always use the AMMS storage layer and respect the configuration-driven architecture.