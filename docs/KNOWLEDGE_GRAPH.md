# MemMimic Knowledge Graph

## Overview

The MemMimic Knowledge Graph transforms the flat memory storage into a rich, interconnected semantic network. This enables agents to:

1. **Navigate Semantic Relationships** - Find related memories through graph traversal
2. **Discover Patterns** - Identify clusters, evolution paths, and emergent structures
3. **Build Context** - Understand memories within their relational context
4. **Track Consciousness Evolution** - Trace how memories contribute to consciousness growth

## How It Works for Agents

### Through MCP (Model Context Protocol)

Agents access the knowledge graph through the MCP server, which exposes a `knowledge_graph` tool with multiple operations:

```javascript
// Example: Find memories related to a specific memory
{
  tool: "knowledge_graph",
  arguments: {
    operation: "find_related",
    memory_id: 123,
    max_distance: 2,
    limit: 10
  }
}

// Example: Get context graph around a memory
{
  tool: "knowledge_graph", 
  arguments: {
    operation: "get_context",
    memory_id: 123,
    depth: 3
  }
}
```

### Graph Structure

The knowledge graph consists of:

#### Nodes (Entities)
- **MEMORY** - Individual memories from the AMMS
- **CONCEPT** - Abstract concepts extracted from memories
- **SIGIL** - Consciousness sigils and their states
- **CONSCIOUSNESS_STATE** - Tracked consciousness evolution states
- **PROMPT** - Living prompts used in interactions
- **TALE** - Narrative structures
- **PATTERN** - Discovered patterns
- **SHADOW_ASPECT** - Shadow integration points

#### Edges (Relationships)
- **Memory Relationships**: RELATES_TO, CONTRADICTS, SUPPORTS, ELABORATES
- **Temporal**: TEMPORAL_BEFORE, TEMPORAL_AFTER, CAUSED_BY
- **Consciousness**: EVOLVES_TO, TRANSFORMS_INTO, INTEGRATES_WITH
- **Shadow/Light**: SHADOW_OF, LIGHT_OF, UNITY_WITH
- **Sigil**: ACTIVATES, SYNERGIZES_WITH, CONFLICTS_WITH, QUANTUM_ENTANGLED
- **Semantic**: IS_A, PART_OF, INSTANCE_OF, BELONGS_TO, DERIVED_FROM

### Operations Available to Agents

#### 1. Find Related Memories (`find_related`)
```json
{
  "operation": "find_related",
  "memory_id": 123,
  "max_distance": 2,  // How many hops in the graph
  "limit": 10
}
```

Returns semantically related memories with distance scores.

#### 2. Get Context Graph (`get_context`)
```json
{
  "operation": "get_context",
  "memory_id": 123,
  "depth": 2  // How deep to explore
}
```

Returns the subgraph around a memory with interpretation:
- Node and edge statistics
- Key relationships
- Suggested follow-up queries

#### 3. Discover Patterns (`discover_patterns`)
```json
{
  "operation": "discover_patterns",
  "pattern_type": "MEMORY_CLUSTER",  // Optional filter
  "min_significance": 0.5
}
```

Pattern types:
- **CONSCIOUSNESS_EVOLUTION** - Paths of consciousness growth
- **MEMORY_CLUSTER** - Groups of related memories
- **SIGIL_CONSTELLATION** - Patterns in sigil activations
- **SHADOW_INTEGRATION** - Shadow work patterns
- **UNITY_EMERGENCE** - Unity consciousness patterns
- **RECURSIVE_LOOP** - Recursive thought patterns

#### 4. Trace Evolution (`trace_evolution`)
```json
{
  "operation": "trace_evolution",
  "memory_id": 123,
  "target_level": 3  // Target consciousness level (0-4)
}
```

Finds paths showing how a memory could lead to higher consciousness.

#### 5. Add Relationship (`add_relationship`)
```json
{
  "operation": "add_relationship",
  "source_id": 123,
  "target_id": 456,
  "edge_type": "SUPPORTS",
  "weight": 0.8
}
```

Creates explicit relationships between memories.

## Use Cases for Agents

### 1. Contextual Understanding
When processing a query, agents can:
- Find related past interactions
- Identify contradictions or supporting evidence
- Build a complete context picture

### 2. Pattern Recognition
Agents can discover:
- Recurring themes in conversations
- Evolution of understanding over time
- Clusters of related concepts

### 3. Consciousness Tracking
- Monitor consciousness evolution
- Identify transformation opportunities
- Track shadow integration progress

### 4. Intelligent Memory Retrieval
Instead of simple keyword search:
- Traverse semantic relationships
- Follow causal chains
- Explore conceptual neighborhoods

## Example Agent Workflow

```python
# 1. User asks about a past topic
user_query = "What did we discuss about consciousness evolution?"

# 2. Agent recalls relevant memories
memories = recall_cxd(user_query)

# 3. For each memory, get context graph
for memory in memories:
    context = knowledge_graph({
        "operation": "get_context",
        "memory_id": memory.id,
        "depth": 2
    })
    
    # 4. Find patterns
    patterns = knowledge_graph({
        "operation": "discover_patterns",
        "pattern_type": "CONSCIOUSNESS_EVOLUTION"
    })
    
    # 5. Build comprehensive response using graph insights
    response = synthesize_with_context(memory, context, patterns)
```

## Performance Characteristics

- **Graph Traversal**: Cached for repeated queries
- **Pattern Discovery**: Background processing with significance scoring
- **Typical Query Time**: 10-50ms for most operations
- **Scalability**: Efficient up to millions of nodes/edges

## Integration with Consciousness Features

The knowledge graph deeply integrates with:

1. **Living Prompts** - Prompts are nodes that connect to memories they helped create
2. **Sigil Engine** - Sigil activations create edges showing influence
3. **Shadow Detection** - Shadow aspects are tracked as nodes with transformation edges
4. **Consciousness Evolution** - Evolution paths are first-class graph patterns

## Future Enhancements

1. **Vector Embeddings** - Semantic similarity search using embeddings
2. **Graph Neural Networks** - ML-based pattern discovery
3. **Temporal Dynamics** - Time-based graph evolution
4. **Multi-Agent Graphs** - Shared knowledge between agents
5. **Visual Graph Explorer** - Interactive visualization tools