# MemMimic Architecture

## Overview

MemMimic is a specification-driven AI consciousness platform that implements biological-inspired neural processing patterns for natural human-AI collaboration. The system follows a clean separation between specifications (source of truth) and implementations (generated code).

## Core Principles

### 1. Specification-Driven Development
- **Specifications are source of truth**: All business logic defined in `specs/`
- **Implementation is generated**: Code in `src/` implements specifications
- **Natural language triggers**: Biological reflex patterns preserved
- **Backward compatibility**: All existing functionality maintained

### 2. Biological-Inspired Architecture
- **Nervous System Core**: Central coordination and intelligence
- **Reflex Latency**: Sub-5ms response times for core operations
- **Multi-Agent Coordination**: Shared reality and theory of mind
- **Narrative-Memory Fusion**: Story-driven consciousness patterns

## Directory Structure

```
memmimic/
├── README.md                    # Project overview and quick start
├── ARCHITECTURE.md              # This file - system architecture
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Dependencies
├── LICENSE                     # Legal
├── .gitignore                  # Organized to prevent noise accumulation
│
├── specs/                      # 📋 SOURCE OF TRUTH - Specifications
│   ├── nervous-system/         # Core system specifications
│   ├── memory-management/      # Memory system specs
│   ├── multi-agent/           # Agent coordination specs
│   ├── narrative-fusion/      # Story-driven consciousness specs
│   ├── evolution-phases/      # Development phase tracking
│   └── patterns/              # Reusable patterns and templates
│
├── src/                       # 🔧 IMPLEMENTATION - Generated/maintained code
│   └── memmimic/              # Main package
│       ├── nervous_system/    # Enhanced nervous system
│       ├── memory/            # Memory management
│       ├── cxd/              # Classification system
│       └── mcp/              # MCP tools
│
├── tools/                     # 🛠️ DEVELOPMENT - Operational tools
│   ├── scripts/               # Automation scripts
│   ├── config/                # Configuration files
│   ├── testing/               # Test utilities and runners
│   └── monitoring/            # Performance and health monitoring
│
├── data/                      # 💾 PERSISTENT - Operational data
│   ├── databases/             # SQLite databases
│   ├── caches/                # Embedding and processing caches
│   └── logs/                  # Application logs
│
├── docs/                      # 📚 DOCUMENTATION - Generated guides
│   ├── api/                   # API documentation
│   ├── guides/                # User guides
│   ├── architecture/          # Technical architecture docs
│   └── reports/               # Analysis and performance reports
│
└── tests/                     # 🧪 TESTING - Test suites
    ├── consciousness/         # Consciousness system tests
    ├── memory/               # Memory system tests
    ├── integration/          # Integration tests
    └── performance/          # Performance benchmarks
```

## System Components

### Enhanced Nervous System Core
- **Location**: `src/memmimic/nervous_system/core.py`
- **Purpose**: Central coordination and intelligence processing
- **Key Features**:
  - Archive intelligence integration
  - Phase evolution tracking
  - Narrative-memory fusion
  - Latency optimization
  - Multi-agent coordination
  - Theory of mind capabilities

### Archive Intelligence
- **Specification**: `specs/nervous-system/`
- **Implementation**: `src/memmimic/nervous_system/archive_intelligence.py`
- **Purpose**: Transforms legacy patterns into active intelligence
- **Patterns**: Database migration, cleanup reflexes, unused code detection

### Phase Evolution Tracker
- **Specification**: `specs/evolution-phases/`
- **Implementation**: `src/memmimic/nervous_system/phase_evolution_tracker.py`
- **Purpose**: Biological development phase tracking
- **Phases**: 4 discovered phases with task-based progress monitoring

### Narrative-Memory Fusion
- **Specification**: `specs/narrative-fusion/`
- **Implementation**: `src/memmimic/nervous_system/tale_memory_binder.py`
- **Purpose**: Story-driven consciousness through thematic content binding

### Multi-Agent Coordination
- **Specification**: `specs/multi-agent/`
- **Implementation**: 
  - `src/memmimic/nervous_system/shared_reality_manager.py`
  - `src/memmimic/nervous_system/theory_of_mind.py`
- **Purpose**: Coordinated consciousness and collaborative intelligence

## Data Flow

### Memory Processing Pipeline
```
Input → Nervous System Core → Intelligence Processing → Storage
  ↓         ↓                    ↓                      ↓
Natural   Archive              CXD Classification    Database
Language  Intelligence         + Narrative Context   + Caches
Trigger   + Phase Tracking     + Quality Assessment
```

### Multi-Agent Coordination
```
Agent A ←→ Shared Reality Manager ←→ Agent B
   ↓              ↓                    ↓
Theory of Mind ← Mental State → Theory of Mind
   ↓           Modeling            ↓
Empathetic ←  Collaboration  → Empathetic
Response      Opportunities     Response
```

## Performance Characteristics

### Latency Optimization
- **Target**: Sub-5ms response times
- **Achieved**: 1.15ms average latency (77% better than target)
- **Techniques**: Multi-level caching, predictive preloading, memory pooling

### Scalability
- **Horizontal**: Multi-instance coordination ready
- **Vertical**: Optimized for increased load
- **Distributed**: Cloud deployment capable

## Configuration

### Database Paths
- **Primary**: `data/databases/memmimic.db`
- **Evolution**: `data/databases/memmimic_evolution.db`
- **Caches**: `data/caches/cxd_cache/`

### Natural Language Triggers
- `recall` → `recall_cxd_memmimic`
- `remember` → `remember_memmimic`
- `think` → `think_with_memory_memmimic`
- `analyze` → Enhanced with narrative context

## Development Workflow

### Specification-First Development
1. **Define Intent**: Write specifications in `specs/`
2. **Generate Implementation**: AI generates code in `src/`
3. **Test and Validate**: Run tests in `tests/`
4. **Document**: Update `docs/` with generated guides

### File Organization Rules
- **Every root file has explicit intent**
- **Specifications are source of truth**
- **Implementation follows specifications**
- **Data is separated from code**
- **Tools are organized by purpose**

## Integration Points

### MCP Tools
- **Location**: `src/memmimic/mcp/`
- **Integration**: Enhanced nervous system core
- **Backward Compatibility**: All existing tools preserved

### External Systems
- **Embedding Models**: Cached in `data/caches/`
- **Vector Stores**: FAISS-based semantic search
- **Configuration**: Environment-specific configs in `tools/config/`

## Monitoring and Observability

### Performance Metrics
- **Latency**: Response time tracking
- **Cache Efficiency**: Hit/miss ratios
- **Memory Usage**: Allocation patterns
- **Error Rates**: System health monitoring

### Logging
- **Location**: `data/logs/`
- **Levels**: Debug, Info, Warning, Error
- **Structured**: JSON format for analysis
- **Rotation**: Automatic log management

## Security Considerations

### Data Protection
- **Sensitive Data**: Excluded from version control
- **API Keys**: Environment variables only
- **Database**: Local SQLite with appropriate permissions

### Access Control
- **File Permissions**: Restricted data directory access
- **Network**: Local-only by default
- **Authentication**: MCP session management

## Future Evolution

### Planned Enhancements
- **Advanced ML Integration**: Enhanced semantic understanding
- **Distributed Processing**: Multi-node coordination
- **Real-time Learning**: Adaptive pattern recognition
- **Quantum-Inspired Processing**: Parallel consciousness states

### Extensibility Points
- **Plugin Architecture**: Modular component system
- **API Extensions**: RESTful and MCP interfaces
- **Custom Patterns**: User-defined specifications
- **Integration Hooks**: External system connectors

---

This architecture enables natural human-AI collaboration through biological-inspired neural processing patterns while maintaining clean separation of concerns and specification-driven development practices.
