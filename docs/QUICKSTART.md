# MemMimic Enhanced - Quick Start Guide

## Installation (5 minutes)

### Prerequisites
- Python 3.10+
- Node.js 16+
- Git

### 1. Clone & Setup
```bash
# Clone the enhanced repository
git clone https://github.com/sosacrazy126/MemMimic.git
cd MemMimic

# Create Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install MCP server
cd src/memmimic/mcp
npm install
cd ../../..
```

### 2. Initialize System
```bash
# Initialize MemMimic database
python -c "
from memmimic import create_memmimic
api = create_memmimic('memmimic.db')
print('✅ MemMimic Enhanced initialized!')
"
```

### 3. Configure Claude Desktop
Add to your Claude Desktop MCP settings (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memmimic": {
      "command": "node",
      "args": ["/absolute/path/to/MemMimic/src/memmimic/mcp/server.js"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/MemMimic/src"
      }
    }
  }
}
```

**Replace** `/absolute/path/to/MemMimic` with your actual installation path.

### 4. Verify Installation
```bash
# Test MCP tools
python -m memmimic.mcp.memmimic_status

# Should show:
# 🎯 MEMMIMIC SYSTEM STATUS (POST-MIGRATION)
# ✅ Storage Type: amms_only
# ✅ Total Memories: 0
# 🟢 SYSTEM HEALTH: All systems operational
```

## First Usage (2 minutes)

### 1. Store Your First Memory
```bash
# Basic memory storage
python -m memmimic.mcp.memmimic_remember "MemMimic Enhanced is now installed and ready to use!" "milestone"

# Quality-controlled storage
python -m memmimic.mcp.memmimic_remember_with_quality "This is a test of the quality control system" "interaction"
```

### 2. Search Memories
```bash
# Search for your memory
python -m memmimic.mcp.memmimic_recall_cxd "MemMimic Enhanced"

# Should return your stored memory with details
```

### 3. Test Memory Analytics
```bash
# Test analytical features
python -m memmimic.mcp.memmimic_socratic "How effective is MemMimic Enhanced?" 3

# Should show:
# 🧘 MEMMIMIC - SOCRATIC DIALOGUE COMPLETED
# 🎯 Query: How effective is MemMimic Enhanced?
# 📊 Depth: 3
# ❓ Questions generated: 5
# 💡 Insights discovered: 5
```

## Core Features Demo (10 minutes)

### Memory Management with Quality Control

```bash
# 1. Store high-quality memory (auto-approved)
python -m memmimic.mcp.memmimic_remember_with_quality "MemMimic Enhanced provides intelligent memory quality control with duplicate detection, semantic similarity analysis, and human review workflows for borderline memories. This system prevents memory pollution while maintaining high-quality persistent storage." "milestone"

# Expected output:
# ✅ MEMORY APPROVED AND SAVED
# 💡 Quality: AUTO-APPROVED (confidence: 0.85)

# 2. Store borderline memory (queued for review)
python -m memmimic.mcp.memmimic_remember_with_quality "test memory" "interaction"

# Expected output:
# ⏳ MEMORY QUEUED FOR REVIEW
# 💡 Quality: BORDERLINE (confidence: 0.50)
# 🔍 Queue ID: pending_20250720_231248

# 3. Review pending memories
python -m memmimic.mcp.memmimic_remember_with_quality "" "" --review

# 4. Force save low-quality memory (bypassing quality gate)
python -m memmimic.mcp.memmimic_remember_with_quality "quick note" "interaction" --force

# Expected output:
# ✅ MEMORY FORCE SAVED
# ⚠️ Quality: BYPASSED (forced save)
```

### Advanced Search & Analysis

```bash
# 1. Hybrid semantic + keyword search
python -m memmimic.mcp.memmimic_recall_cxd "quality control system" "CONTEXT" 5

# 2. Memory pattern analysis
python -m memmimic.mcp.memmimic_analyze_patterns

# 3. Socratic self-questioning
python -m memmimic.mcp.memmimic_socratic "What patterns exist in my memory usage?" 4
```

### Memory Analytics

```python
# Test memory analytics (Python)
from memmimic.memory.analytics_dashboard import MemoryAnalyticsDashboard

analytics = MemoryAnalyticsDashboard()
status = analytics.get_comprehensive_status()

print(f"Memory Quality Rate: {status.overall_quality_rate:.1%}")
print(f"Active Memories: {status.active_memory_count}")
print(f"CXD Accuracy: {status.cxd_classification_accuracy:.3f}")
print(f"Performance Level: {status.performance_level}")
```

### Tale Management

```bash
# 1. Create a tale
python -m memmimic.mcp.memmimic_save_tale "getting_started" "MemMimic Enhanced Quick Start: Successfully installed and configured MemMimic Enhanced with consciousness integration, quality control, and 15 operational MCP tools." "projects/memmimic" "quickstart,setup,tutorial"

# 2. List all tales
python -m memmimic.mcp.memmimic_tales

# 3. Load a tale
python -m memmimic.mcp.memmimic_load_tale "getting_started"

# 4. Generate contextual narrative
python -m memmimic.mcp.memmimic_context_tale "getting started with MemMimic" "technical" 10
```

## Using with Claude Desktop

Once configured, you can use MemMimic Enhanced directly in Claude Desktop:

### Basic Commands
```
# Store memories
remember_with_quality("Important insight about memory optimization", "reflection")

# Search with filtering
recall_cxd("memory optimization", "CONTEXT", 5)

# Get system status
status()

# Socratic questioning
socratic_dialogue("How can I improve my memory usage patterns?", 3)
```

### Quality Control Workflow
```
# Store memory (may be queued for review)
remember_with_quality("Borderline quality memory content", "interaction")

# Review pending memories
review_pending_memories()

# The system will show pending memories with queue IDs
# Human review happens through approval/rejection
```

### Intelligence-Enhanced Usage
```
# The system automatically integrates intelligent features:
# - Quality gates evaluate memory content
# - CXD classification categorizes cognitive functions
# - Pattern analysis discovers usage insights
# - Memory operations include analytics metadata
```

## Configuration Customization

### Basic Configuration
Create `config/memmimic_config.yaml`:

```yaml
# Quality Gate Settings
quality_gate:
  auto_approve_threshold: 0.8    # Lower = more auto-approvals
  auto_reject_threshold: 0.3     # Higher = fewer auto-rejections
  duplicate_threshold: 0.85      # Similarity threshold for duplicates
  min_content_length: 10         # Minimum characters required

# Memory Analytics Settings  
memory_analytics:
  pattern_analysis:
    frequency_threshold: 0.3
    pattern_detection_rate: 0.1
  
  quality_metrics:
    calculation_precision: 0.001
    analysis_depth_limit: 1000

# Performance Settings
performance:
  cache_size: 1000
  max_concurrent_operations: 100
```

### Advanced CXD Configuration
Edit `src/memmimic/cxd/config/cxd_config.yaml`:

```yaml
classifiers:
  lexical:
    enabled: true
    confidence_threshold: 0.6
  
  semantic:
    enabled: true
    model: "all-MiniLM-L6-v2"
    cache_size: 1000
  
  meta:
    enabled: true
    concordance_threshold: 0.6
```

## Common Use Cases

### 1. Personal Knowledge Management
```bash
# Store insights
remember_with_quality("Key insight about productivity: Focus on outcomes rather than activities", "reflection")

# Retrieve related insights
recall_cxd("productivity insights", "CONTEXT", 5)

# Generate summary
context_tale("my productivity insights", "technical", 15)
```

### 2. Project Documentation  
```bash
# Document project milestones
remember_with_quality("Project Phase 1 completed successfully with all features implemented and tested", "milestone")

# Create project tale
save_tale("project_phase1", "Comprehensive documentation of Phase 1 completion...", "projects/current")

# Search project history
recall_cxd("project phase", "DATA", 10)
```

### 3. Learning & Reflection
```bash
# Capture learning
remember_with_quality("Understanding: Memory optimization requires both technical implementation and intelligent analysis", "reflection")

# Self-questioning
socratic_dialogue("What have I learned about memory optimization?", 4)

# Pattern analysis
analyze_memory_patterns()
```

### 4. Collaborative Memory
```bash
# Store team insights
remember_with_quality("Team discussion revealed that quality control reduces memory noise by 80%", "interaction")

# Create team narrative
context_tale("team collaboration insights", "technical", 20)
```

## Troubleshooting Quick Fixes

### Issue: MCP tools not working
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Should include /path/to/MemMimic/src
# If not, add to your shell profile:
export PYTHONPATH="/path/to/MemMimic/src:$PYTHONPATH"
```

### Issue: Database errors
```bash
# Check database exists
ls -la memmimic.db

# If missing, reinitialize:
python -c "from memmimic import create_memmimic; create_memmimic('memmimic.db')"
```

### Issue: Analytics features not working
```bash
# Check analytics cache
ls -la memmimic_cache/analytics/

# If missing, reinitialize:
python -c "
from memmimic.memory.analytics_dashboard import MemoryAnalyticsDashboard
analytics = MemoryAnalyticsDashboard()
analytics.initialize_analytics_systems()
print('✅ Analytics systems initialized')
"
```

### Issue: Performance problems
```bash
# Optimize database
sqlite3 memmimic.db "VACUUM;"
sqlite3 memmimic.db "ANALYZE;"

# Clear caches
rm -rf cxd_cache/*
rm -rf memmimic_cache/*
```

## Next Steps

### Explore Advanced Features
1. **Memory Analytics**: Monitor your memory system performance over time
2. **Quality Analytics**: Analyze memory quality patterns
3. **Advanced Search**: Use CXD filtering for precise results
4. **Narrative Generation**: Create contextual tales from your memories
5. **Pattern Recognition**: Discover insights through memory pattern analysis

### Integration Options
1. **API Integration**: Use Python API for custom applications
2. **Webhook Integration**: Set up real-time memory notifications  
3. **Dashboard Creation**: Build custom memory analytics dashboards
4. **Export/Import**: Implement memory backup and synchronization

### Community & Support
- **Documentation**: Full docs in `/docs` directory
- **Issues**: Report issues on GitHub repository
- **Enhancements**: Fork and contribute improvements
- **Examples**: Check `/examples` for advanced usage patterns

## Success Indicators

You've successfully set up MemMimic Enhanced when:

✅ **System Status**: `status()` shows "All systems operational"  
✅ **Memory Storage**: `remember_with_quality()` works with quality feedback  
✅ **Search Functionality**: `recall_cxd()` returns relevant results  
✅ **Memory Analytics**: Quality gate shows 85-95% effectiveness  
✅ **Quality Control**: Memory approval/rejection workflow functions  
✅ **MCP Integration**: All 13 tools accessible in Claude Desktop

**Congratulations! MemMimic Enhanced is ready to enhance your AI interactions with persistent, intelligent memory management.**