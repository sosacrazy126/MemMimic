# MemMimic Quality Gate System

## Overview

The MemMimic Quality Gate System provides intelligent memory quality control with advanced semantic duplicate detection. It prevents memory pollution from AI agents while preserving valuable memories through automated approval and human oversight for borderline cases.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Agent      │───▶│   Quality Gate   │───▶│  Memory Store   │
│ (remember tool) │    │                  │    │    (AMMS)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Review Queue    │
                       │   (SQLite)       │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Human Reviewer   │
                       │ (approve/reject) │
                       └──────────────────┘
```

## Core Components

### 1. MemoryQualityGate
Central quality control engine that evaluates memory content using:
- **Semantic Similarity Detection**: Advanced duplicate detection using SentenceTransformer embeddings
- **Quality Assessment**: Multi-factor scoring based on content characteristics
- **Automatic Decision Making**: Configurable thresholds for auto-approval/rejection
- **Human Review Queue**: Persistent storage for borderline cases

### 2. SemanticSimilarityDetector
Advanced duplicate detection using machine learning:
- **Model**: `all-MiniLM-L6-v2` SentenceTransformer
- **Similarity Scoring**: Cosine similarity between embeddings
- **Fallback**: Word overlap similarity if embeddings unavailable
- **Performance**: Handles batch similarity analysis efficiently

### 3. PersistentMemoryQueue
SQLite-based review queue for cross-session persistence:
- **Storage**: Durable storage for pending memories
- **Status Tracking**: pending_review, approved, rejected
- **Metadata**: Quality scores, timestamps, reviewer notes
- **Cleanup**: Automatic removal of old processed entries

## Quality Assessment Algorithm

### Confidence Scoring
The system calculates confidence scores (0.0-1.0) based on:

#### Base Quality Factors
- **Length**: Minimum 10 characters, bonus for 50+ characters
- **Structure**: Sentence punctuation, word count
- **Information Density**: Meaningful content vs noise
- **Memory Type**: Milestones and reflections get quality boost
- **Keywords**: Quality indicators ("important", "learned", "discovered")
- **Noise Detection**: Penalties for test/debug content

#### Semantic Enhancement
- **Novelty Bonus**: +0.05 for unique content
- **Context Bonus**: +0.1 for good related context (50-70% similarity)
- **Redundancy Penalty**: -0.1 to -0.2 for high similarity to existing memories

### Decision Thresholds
- **Auto-Approve**: ≥0.8 confidence (high quality, unique content)
- **Auto-Reject**: ≤0.3 confidence (low quality or duplicates)
- **Human Review**: 0.3-0.8 confidence (borderline cases)

### Duplicate Detection
- **Primary**: Semantic similarity using embeddings (threshold: 0.85)
- **Fallback**: Word overlap similarity (threshold: 0.85)
- **Context**: Considers top 10 similar memories from AMMS search

## Usage Guide

### Enhanced Remember Tool

#### Basic Usage
```bash
# Standard memory storage with quality control
python -m memmimic.mcp.memmimic_remember_with_quality "Learned how to implement async patterns in Python" interaction

# Force save without quality check
python -m memmimic.mcp.memmimic_remember_with_quality "debug test" interaction --force
```

#### Review Workflow
```bash
# View pending memories
python -m memmimic.mcp.memmimic_remember_with_quality --review

# Approve a memory
python -m memmimic.mcp.memmimic_remember_with_quality --approve pending_20240120_143052_123456 --note "Good technical insight"

# Reject a memory
python -m memmimic.mcp.memmimic_remember_with_quality --reject pending_20240120_143052_789012 --reason "Too vague, lacks context"
```

### MCP Server Integration

The quality gate is fully integrated with the MCP server, providing these tools:

#### Core Tools
- `remember_with_quality`: Store memory with quality control
- `review_pending_memories`: List memories awaiting approval
- `approve_memory`: Approve pending memory
- `reject_memory`: Reject pending memory

#### Example MCP Usage
```javascript
// Store memory with quality control
await mcp.callTool('remember_with_quality', {
  content: 'Successfully implemented rate limiting with Redis',
  memory_type: 'milestone'
});

// Review pending memories
await mcp.callTool('review_pending_memories', {});

// Approve a memory
await mcp.callTool('approve_memory', {
  queue_id: 'pending_20240120_143052_123456',
  note: 'Important architectural decision'
});
```

## Configuration

### Quality Thresholds
```python
class MemoryQualityGate:
    def __init__(self, assistant, queue_db_path="memory_queue.db"):
        # Configurable thresholds
        self.auto_approve_threshold = 0.8    # Auto-approve threshold
        self.auto_reject_threshold = 0.3     # Auto-reject threshold
        self.duplicate_threshold = 0.85      # Duplicate detection threshold
        self.min_content_length = 10         # Minimum content length
```

### Semantic Similarity
```python
class SemanticSimilarityDetector:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Model configuration
        self.model_name = model_name  # SentenceTransformer model
        
    def find_similar_memories(self, content, memories, threshold=0.75):
        # Similarity threshold for related content detection
```

## Database Schema

### Review Queue (memory_queue.db)
```sql
CREATE TABLE memory_review_queue (
    id TEXT PRIMARY KEY,                 -- Unique queue ID
    content TEXT NOT NULL,               -- Memory content
    memory_type TEXT NOT NULL,           -- Type (interaction, reflection, milestone)
    quality_result_json TEXT NOT NULL,   -- Serialized quality assessment
    queued_at TEXT NOT NULL,             -- Timestamp when queued
    status TEXT DEFAULT 'pending_review', -- Status (pending_review, approved, rejected)
    reviewer_note TEXT,                  -- Human reviewer note
    processed_at TEXT,                   -- Timestamp when processed
    created_at TEXT NOT NULL,            -- Creation timestamp
    updated_at TEXT NOT NULL             -- Last update timestamp
);

CREATE INDEX idx_status ON memory_review_queue(status);
CREATE INDEX idx_queued_at ON memory_review_queue(queued_at);
```

## Quality Assessment Examples

### Auto-Approved (High Quality)
```
Content: "Discovered an elegant solution to the memory management problem using weak references"
Type: milestone
Confidence: 0.900
Reason: High quality content with good confidence
Decision: ✅ AUTO-APPROVED
```

### Auto-Rejected (Low Quality)
```
Content: "test"
Type: interaction
Confidence: 0.000
Reason: Content too short (minimum 10 characters)
Decision: ❌ AUTO-REJECTED
```

### Human Review Required (Borderline)
```
Content: "Learning Python programming fundamentals"
Type: interaction
Confidence: 0.500
Reason: Borderline quality - requires human review
Decision: ⏳ NEEDS REVIEW
Queue ID: pending_20240120_143052_123456
```

### Duplicate Detected
```
Content: "Python programming fundamentals for beginners"
Type: interaction
Confidence: 0.200
Reason: High similarity to existing memory (0.89)
Decision: ❌ AUTO-REJECTED
Similar memories: 1 found
```

## Integration with Existing Systems

### AMMS Compatibility
- Fully compatible with Active Memory Management System (AMMS)
- Uses existing `AMMSStorage` for memory search and storage
- Maintains consciousness integration features
- Preserves CXD classification metadata

### ContextualAssistant Integration
- Leverages existing `ContextualAssistant` infrastructure
- Uses standardized database path (`memmimic.db`)
- Maintains assistant thinking and analysis capabilities

## Performance Considerations

### Semantic Similarity
- **Model Loading**: SentenceTransformer loads once per session
- **Embedding Computation**: ~100ms per similarity comparison
- **Batch Processing**: Optimized for multiple comparisons
- **Memory Usage**: ~200MB for model in memory

### Database Performance
- **SQLite Indexes**: Optimized queries on status and timestamp
- **Connection Pooling**: Reuses connections for better performance
- **Queue Cleanup**: Automatic cleanup of old entries (30 days default)

## Monitoring and Statistics

### Queue Statistics
```python
# Get queue statistics
queue_stats = quality_gate.persistent_queue.get_queue_stats()
# Returns: {'pending_review': 5, 'approved': 120, 'rejected': 15}
```

### Quality Metrics
- **Approval Rate**: Percentage of memories auto-approved
- **Review Rate**: Percentage requiring human review
- **Duplicate Rate**: Percentage rejected as duplicates
- **Processing Time**: Average time from queue to decision

## Troubleshooting

### Common Issues

#### SentenceTransformer Not Loading
```
ERROR: Failed to initialize SentenceTransformer
SOLUTION: Install sentence-transformers: pip install sentence-transformers>=2.2
FALLBACK: System automatically uses word overlap similarity
```

#### Queue Database Locked
```
ERROR: Database is locked
SOLUTION: Ensure only one process accesses queue at a time
WORKAROUND: Delete memory_queue.db and restart (loses pending reviews)
```

#### High Memory Usage
```
ISSUE: SentenceTransformer uses ~200MB RAM
SOLUTION: Use smaller model: all-MiniLM-L12-v2 → all-MiniLM-L6-v2
ALTERNATIVE: Disable semantic similarity and use word overlap only
```

### Debug Mode
```python
import logging
logging.getLogger('memmimic.memory.quality_gate').setLevel(logging.DEBUG)
logging.getLogger('memmimic.memory.semantic_similarity').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features
1. **Configuration Management**: YAML/JSON config files for thresholds
2. **Queue Cleanup Automation**: Scheduled cleanup of old entries
3. **Usage Statistics**: Dashboard for quality gate performance
4. **Custom Similarity Models**: Support for domain-specific embeddings
5. **Batch Processing**: Bulk approval/rejection tools
6. **API Integration**: REST API for external quality management

### Advanced Configurations
1. **Dynamic Thresholds**: Adaptive thresholds based on memory history
2. **Context-Aware Scoring**: Quality assessment based on conversation context
3. **User-Specific Rules**: Personalized quality criteria per user
4. **Integration Webhooks**: External system notifications for reviews

## Security Considerations

### Data Privacy
- **Local Processing**: All similarity computation happens locally
- **No External Calls**: SentenceTransformer runs offline
- **Secure Storage**: SQLite database with file-level security

### Access Control
- **File Permissions**: Queue database respects filesystem permissions
- **Process Isolation**: Each assistant instance has separate queues
- **Audit Trail**: Complete logging of all approval/rejection decisions

---

*This documentation covers the complete MemMimic Quality Gate System. For implementation details, see the source code in `/src/memmimic/memory/`.*