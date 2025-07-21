# Quality Gate API Reference

## Core Classes

### MemoryQualityGate

Central quality control engine for memory evaluation and management.

```python
from memmimic.memory.quality_gate import MemoryQualityGate, create_quality_gate
from memmimic.assistant import ContextualAssistant
```

#### Constructor

```python
MemoryQualityGate(assistant: ContextualAssistant, queue_db_path: str = "memory_queue.db")
```

**Parameters:**
- `assistant`: ContextualAssistant instance with memory store
- `queue_db_path`: Path to SQLite queue database

**Configuration Attributes:**
```python
# Quality thresholds (configurable)
self.auto_approve_threshold = 0.8    # Auto-approve if confidence ≥ 0.8
self.auto_reject_threshold = 0.3     # Auto-reject if confidence ≤ 0.3
self.duplicate_threshold = 0.85      # Duplicate detection threshold
self.min_content_length = 10         # Minimum content length in characters
```

#### Methods

##### evaluate_memory()
```python
async def evaluate_memory(content: str, memory_type: str = "interaction") -> MemoryQualityResult
```

Evaluate memory quality using semantic analysis and content assessment.

**Parameters:**
- `content`: Memory content to evaluate
- `memory_type`: Type of memory ("interaction", "reflection", "milestone")

**Returns:** `MemoryQualityResult` with approval decision

**Example:**
```python
quality_gate = create_quality_gate()
result = await quality_gate.evaluate_memory(
    "Implemented rate limiting using Redis for API protection", 
    "milestone"
)

print(f"Approved: {result.approved}")
print(f"Confidence: {result.confidence}")
print(f"Reason: {result.reason}")
```

##### queue_for_review()
```python
async def queue_for_review(content: str, memory_type: str, quality_result: MemoryQualityResult) -> str
```

Queue memory for human review.

**Returns:** Unique queue ID string

##### get_pending_reviews()
```python
def get_pending_reviews() -> List[Dict[str, Any]]
```

Get all memories awaiting human review.

**Returns:** List of review dictionaries with keys:
- `id`: Queue ID
- `content`: Memory content  
- `memory_type`: Type of memory
- `quality_result`: MemoryQualityResult object
- `queued_at`: Timestamp when queued
- `status`: Current status

##### approve_pending()
```python
async def approve_pending(queue_id: str, reviewer_note: str = "") -> bool
```

Approve a pending memory and store it.

**Returns:** `True` if successful, `False` if failed

##### reject_pending()
```python
async def reject_pending(queue_id: str, reason: str) -> bool
```

Reject a pending memory.

**Returns:** `True` if successful, `False` if failed

### MemoryQualityResult

Data class representing quality evaluation results.

```python
from memmimic.memory.quality_types import MemoryQualityResult
```

#### Attributes

```python
@dataclass
class MemoryQualityResult:
    approved: bool                           # Whether memory is approved
    reason: str                             # Explanation of decision
    confidence: float                       # Quality confidence (0.0-1.0)
    auto_decision: bool                     # If decision was automatic
    suggested_content: Optional[str] = None # Improvement suggestions
    duplicates: Optional[List[Memory]] = None # Similar existing memories
    timestamp: datetime = field(default_factory=datetime.now) # Evaluation time
```

#### Example Usage

```python
if result.approved and result.auto_decision:
    print("✅ Auto-approved")
elif not result.approved and result.auto_decision:
    print("❌ Auto-rejected")
else:
    print("⏳ Needs human review")

if result.duplicates:
    print(f"Found {len(result.duplicates)} similar memories")

if result.suggested_content:
    print(f"Suggestion: {result.suggested_content}")
```

### SemanticSimilarityDetector

Advanced semantic similarity detection using SentenceTransformer embeddings.

```python
from memmimic.memory.semantic_similarity import SemanticSimilarityDetector, get_semantic_detector
```

#### Constructor

```python
SemanticSimilarityDetector(model_name: str = "all-MiniLM-L6-v2")
```

**Parameters:**
- `model_name`: SentenceTransformer model name

#### Methods

##### compute_similarity()
```python
def compute_similarity(text1: str, text2: str) -> float
```

Compute semantic similarity between two texts.

**Returns:** Similarity score between 0.0 and 1.0

##### find_similar_memories()
```python
async def find_similar_memories(
    content: str, 
    memories: List[Memory], 
    threshold: float = 0.75,
    max_results: int = 5
) -> List[Tuple[Memory, float]]
```

Find memories similar to given content.

**Returns:** List of (Memory, similarity_score) tuples, sorted by similarity

##### is_likely_duplicate()
```python
def is_likely_duplicate(content: str, existing_content: str, threshold: float = 0.85) -> bool
```

Check if content is likely a duplicate.

**Returns:** `True` if similarity exceeds threshold

##### batch_similarity_check()
```python
async def batch_similarity_check(
    content: str, 
    memories: List[Memory], 
    duplicate_threshold: float = 0.85,
    similar_threshold: float = 0.75
) -> dict
```

Perform batch similarity analysis.

**Returns:** Dictionary with analysis results:
```python
{
    "duplicates": [(Memory, float), ...],
    "similar": [(Memory, float), ...], 
    "max_similarity": float,
    "analysis_summary": str
}
```

#### Global Instance

```python
# Get or create global instance (recommended)
detector = get_semantic_detector()
similarity = detector.compute_similarity("text1", "text2")
```

### PersistentMemoryQueue

SQLite-based persistent review queue.

```python
from memmimic.memory.persistent_queue import PersistentMemoryQueue
```

#### Constructor

```python
PersistentMemoryQueue(db_path: str = "memory_queue.db")
```

#### Methods

##### add_to_queue()
```python
def add_to_queue(content: str, memory_type: str, quality_result: MemoryQualityResult) -> str
```

Add memory to review queue.

**Returns:** Unique queue ID

##### get_pending_reviews()
```python
def get_pending_reviews() -> List[Dict[str, Any]]
```

Get all memories awaiting review.

##### approve_memory()
```python
def approve_memory(queue_id: str, reviewer_note: str = "") -> bool
```

Mark queued memory as approved.

##### reject_memory()
```python
def reject_memory(queue_id: str, reason: str = "") -> bool
```

Mark queued memory as rejected.

##### cleanup_old_entries()
```python
def cleanup_old_entries(days: int = 30) -> int
```

Clean up old processed entries.

**Returns:** Number of entries deleted

##### get_queue_stats()
```python
def get_queue_stats() -> Dict[str, int]
```

Get queue statistics.

**Returns:** Dictionary with status counts:
```python
{
    "pending_review": 5,
    "approved": 120, 
    "rejected": 15
}
```

## Convenience Functions

### create_quality_gate()
```python
def create_quality_gate(assistant_name: str = "memmimic") -> MemoryQualityGate
```

Create a MemoryQualityGate with default assistant.

### get_semantic_detector()
```python
def get_semantic_detector() -> SemanticSimilarityDetector
```

Get or create global semantic similarity detector instance.

## MCP Server Tools

The quality gate system provides these MCP tools:

### remember_with_quality
Store memory with quality control.

**Input Schema:**
```json
{
  "content": "string (required)",
  "memory_type": "string (default: interaction)", 
  "force": "boolean (default: false)"
}
```

### review_pending_memories
Show memories awaiting quality approval.

**Input Schema:** `{}` (no parameters)

### approve_memory
Approve a pending memory for storage.

**Input Schema:**
```json
{
  "queue_id": "string (required)",
  "note": "string (optional)"
}
```

### reject_memory
Reject a pending memory.

**Input Schema:**
```json
{
  "queue_id": "string (required)",
  "reason": "string (default: 'Rejected by reviewer')"
}
```

## Quality Assessment Algorithm

### Base Quality Score Calculation

```python
def _calculate_simple_quality_score(content: str, memory_type: str) -> float:
    score = 0.5  # Base score
    
    # Length-based scoring
    if len(content) > 100: score += 0.15
    elif len(content) > 50: score += 0.1
    elif len(content) < 20: score -= 0.2
    
    # Structure-based scoring  
    if any(punct in content for punct in '.!?'): score += 0.1
    
    # Word count
    if len(content.split()) > 10: score += 0.1
    
    # Memory type bonus
    if memory_type == "milestone": score += 0.1
    elif memory_type == "reflection": score += 0.05
    
    # Quality keywords
    quality_keywords = ["important", "key", "critical", "learned", "discovered", "achieved"]
    if any(keyword in content.lower() for keyword in quality_keywords):
        score += 0.1
    
    # Noise detection
    noise_indicators = ["test", "testing", "tmp", "temporary", "debug"]
    if any(noise in content.lower() for noise in noise_indicators):
        score -= 0.2
    
    return max(0.0, min(1.0, score))
```

### Semantic Enhancement

```python
def enhance_with_semantic_context(base_score: float, similar_memories: List[Tuple[Memory, float]]) -> float:
    semantic_boost = 0.0
    
    if similar_memories:
        max_similarity = max([sim for _, sim in similar_memories[:3]])
        
        if max_similarity > 0.9:
            semantic_boost = -0.2    # Likely redundant
        elif max_similarity > 0.8:
            semantic_boost = -0.1    # Somewhat redundant  
        elif 0.5 <= max_similarity <= 0.7:
            semantic_boost = 0.1     # Good related context
        else:
            semantic_boost = 0.05    # Novel content
    
    return max(0.0, min(1.0, base_score + semantic_boost))
```

## Error Handling

### Common Exceptions

#### SentenceTransformerError
```python
try:
    detector = SemanticSimilarityDetector()
except Exception as e:
    print(f"Fallback to word overlap: {e}")
    # System automatically falls back to word overlap similarity
```

#### QueueDatabaseError
```python
try:
    queue_id = quality_gate.queue_for_review(content, memory_type, result)
except sqlite3.Error as e:
    print(f"Queue database error: {e}")
    # Handle database connectivity issues
```

### Graceful Degradation

The system is designed to gracefully handle failures:

1. **Semantic Similarity Unavailable**: Falls back to word overlap similarity
2. **Queue Database Issues**: Continues with direct memory storage
3. **Quality Assessment Errors**: Defaults to human review (conservative approach)

## Performance Considerations

### Semantic Similarity
- **Model Loading**: ~2-3 seconds initial load
- **Embedding Computation**: ~50-100ms per text
- **Memory Usage**: ~200MB for model
- **Batch Processing**: Recommended for multiple comparisons

### Database Operations
- **SQLite Performance**: Optimized with indexes
- **Connection Reuse**: Connections pooled for efficiency  
- **Queue Size**: Performance optimal under 1000 pending entries

### Optimization Tips

```python
# Reuse detector instance
detector = get_semantic_detector()  # Global singleton

# Batch similarity checks
results = await detector.batch_similarity_check(content, memories)

# Use appropriate thresholds
similarity = detector.compute_similarity(text1, text2)
if similarity > 0.85:  # High threshold for duplicates
    handle_duplicate()
```

## Example Integration

### Complete Workflow Example

```python
import asyncio
from memmimic.memory.quality_gate import create_quality_gate
from memmimic.memory.storage.amms_storage import Memory

async def intelligent_memory_storage():
    # Initialize quality gate
    quality_gate = create_quality_gate("my_assistant")
    
    # Content to store
    content = "Implemented OAuth2 with PKCE for mobile app security"
    memory_type = "milestone"
    
    # Evaluate quality
    result = await quality_gate.evaluate_memory(content, memory_type)
    
    if result.approved and result.auto_decision:
        # Auto-approved - store directly
        memory = Memory(content=content, metadata={"type": memory_type})
        memory_id = await quality_gate.memory_store.store_memory(memory)
        print(f"✅ Stored as memory {memory_id}")
        
    elif not result.approved and result.auto_decision:
        # Auto-rejected
        print(f"❌ Rejected: {result.reason}")
        if result.suggested_content:
            print(f"💡 Suggestion: {result.suggested_content}")
            
    else:
        # Needs human review
        queue_id = await quality_gate.queue_for_review(content, memory_type, result)
        print(f"⏳ Queued for review: {queue_id}")
        
        # Later: human reviewer approves
        success = await quality_gate.approve_pending(queue_id, "Good security practice")
        if success:
            print(f"✅ Approved and stored")

# Run the example
asyncio.run(intelligent_memory_storage())
```

---

*For complete implementation details and advanced usage patterns, see the source code in `/src/memmimic/memory/`.*