# Quality Gate Quick Start Guide

## 🚀 Getting Started with MemMimic Quality Gate

The Quality Gate prevents memory pollution from AI agents while preserving valuable memories through intelligent filtering and human oversight.

## Quick Setup

### 1. Verify Dependencies
```bash
# Check if sentence-transformers is installed
python -c "import sentence_transformers; print('✅ sentence-transformers available')"

# If not installed:
pip install sentence-transformers>=2.2
```

### 2. Test the System
```bash
cd /path/to/memmimic

# Test quality gate with a high-quality memory
python -m memmimic.mcp.memmimic_remember_with_quality "Successfully implemented OAuth2 authentication with JWT tokens for secure API access" milestone

# Test with low-quality content (should be rejected)
python -m memmimic.mcp.memmimic_remember_with_quality "test" interaction

# Test with borderline content (should need review)
python -m memmimic.mcp.memmimic_remember_with_quality "working on the project" interaction
```

## Basic Workflow

### 📝 Storing Memories
```bash
# Standard usage - quality control enabled
python -m memmimic.mcp.memmimic_remember_with_quality "Learned advanced async patterns in Python using asyncio and aiohttp" interaction

# Force save without quality check
python -m memmimic.mcp.memmimic_remember_with_quality "debug message" interaction --force
```

**Expected Results:**
- ✅ **High quality** → Auto-approved and stored
- ❌ **Low quality/duplicates** → Auto-rejected  
- ⏳ **Borderline quality** → Queued for human review

### 📋 Managing Reviews
```bash
# View pending memories
python -m memmimic.mcp.memmimic_remember_with_quality --review

# Approve a memory
python -m memmimic.mcp.memmimic_remember_with_quality --approve pending_20240120_143052_123456

# Reject a memory  
python -m memmimic.mcp.memmimic_remember_with_quality --reject pending_20240120_143052_789012 --reason "Too vague"
```

## MCP Server Usage

### Available Tools
- `remember_with_quality` - Store memory with quality control
- `review_pending_memories` - Show pending reviews
- `approve_memory` - Approve pending memory
- `reject_memory` - Reject pending memory

### Example Usage
```javascript
// Store high-quality memory
await mcp.callTool('remember_with_quality', {
  content: 'Implemented Redis caching layer reducing API response time by 70%',
  memory_type: 'milestone'
});

// Check for pending reviews
await mcp.callTool('review_pending_memories', {});

// Approve a memory
await mcp.callTool('approve_memory', {
  queue_id: 'pending_20240120_143052_123456',
  note: 'Good technical achievement'
});
```

## Understanding Quality Scores

### Quality Factors
- **Length**: Minimum 10 chars, bonus for detailed content
- **Structure**: Complete sentences, proper punctuation  
- **Information**: Meaningful content vs test/debug messages
- **Context**: Relationship to existing memories
- **Type**: Milestones and reflections get priority

### Score Ranges
- **0.8-1.0**: ✅ Auto-approved (excellent quality)
- **0.3-0.8**: ⏳ Human review needed (borderline)
- **0.0-0.3**: ❌ Auto-rejected (poor quality/duplicates)

## Common Scenarios

### ✅ Auto-Approved Examples
```
"Discovered memory leak in WebSocket connections - fixed by implementing proper cleanup in disconnect handler"
→ Confidence: 0.92 - Detailed technical insight with solution

"Completed migration to microservices architecture - reduced deployment time from 2 hours to 15 minutes"  
→ Confidence: 0.85 - Quantified achievement with clear impact
```

### ❌ Auto-Rejected Examples
```
"test"
→ Confidence: 0.0 - Too short (minimum 10 characters)

"debug message for testing purposes"
→ Confidence: 0.1 - Contains noise indicators (debug, test)

"Python is a programming language" (when similar memory exists)
→ Confidence: 0.2 - High similarity to existing memory (0.89)
```

### ⏳ Human Review Examples
```
"working on authentication system"
→ Confidence: 0.45 - Vague but potentially valuable

"learned about react hooks today"
→ Confidence: 0.6 - Basic but could be meaningful in context
```

## Best Practices

### ✨ Writing Quality Memories
1. **Be Specific**: Include technical details, solutions, or outcomes
2. **Add Context**: Explain why something was important or challenging  
3. **Use Complete Sentences**: Proper grammar and punctuation
4. **Include Metrics**: Quantify improvements or results when possible
5. **Avoid Test Content**: Don't use "test", "debug", "tmp" in production

### 🔄 Review Management
1. **Regular Reviews**: Check pending queue daily
2. **Clear Decisions**: Approve valuable content, reject noise
3. **Add Notes**: Explain approval/rejection reasoning
4. **Be Consistent**: Develop consistent quality standards

### ⚡ Performance Tips
1. **Batch Operations**: Review multiple memories at once
2. **Use Force Sparingly**: Only bypass quality gate when necessary
3. **Monitor Queue Size**: Keep pending reviews manageable
4. **Clean Old Entries**: System auto-cleans after 30 days

## Troubleshooting

### Issue: Semantic Similarity Not Working
```bash
# Check if model is loaded
python -c "from memmimic.memory.semantic_similarity import get_semantic_detector; d = get_semantic_detector(); print('Model loaded:', d.model is not None)"

# If False, install dependencies:
pip install sentence-transformers torch
```

### Issue: All Memories Need Review
**Cause**: Default thresholds might be too strict for your content  
**Solution**: Consider adjusting thresholds in quality_gate.py:
```python
self.auto_approve_threshold = 0.7  # Lower from 0.8
self.auto_reject_threshold = 0.2   # Lower from 0.3
```

### Issue: Queue Database Errors
**Cause**: Database corruption or permission issues  
**Solution**: 
```bash
# Reset queue (loses pending reviews)
rm memory_queue.db

# Or check permissions
ls -la memory_queue.db
```

## Integration Examples

### With AI Agents
```python
# In your AI agent code
from memmimic.memory.quality_gate import create_quality_gate

async def remember_with_quality(content, memory_type="interaction"):
    quality_gate = create_quality_gate()
    result = await quality_gate.evaluate_memory(content, memory_type)
    
    if result.approved and result.auto_decision:
        # Store directly
        await store_memory(content, memory_type)
        return f"Stored: {content}"
    elif result.auto_decision:
        # Rejected
        return f"Rejected: {result.reason}"
    else:
        # Queue for review
        queue_id = await quality_gate.queue_for_review(content, memory_type, result)
        return f"Queued for review: {queue_id}"
```

### With Existing Code
```python
# Replace basic remember() calls
# OLD:
# await assistant.remember(content)

# NEW:
await assistant.remember_with_quality(content, memory_type)
```

## Next Steps

1. **Read Full Documentation**: [quality_gate_system.md](quality_gate_system.md)
2. **Configure Thresholds**: Adjust for your specific use case
3. **Monitor Performance**: Track approval/rejection rates
4. **Integrate with Workflows**: Add to existing AI agent pipelines
5. **Set Up Automation**: Consider automated queue management

---

🎯 **Goal**: Maintain high-quality memory database while reducing manual oversight through intelligent automation.

📈 **Success Metrics**: 
- High auto-approval rate for quality content
- Low false positive rate (good content rejected)
- Manageable review queue size
- Improved memory database quality over time