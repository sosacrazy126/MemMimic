#!/usr/bin/env python3
"""
MemMimic Knowledge Graph MCP Tool
Exposes knowledge graph capabilities to agents
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from memmimic.knowledge_graph.graph_engine import KnowledgeGraphEngine
from memmimic.memory.storage import AMMSStorage, Memory


async def main():
    """
    Knowledge Graph tool for MCP
    
    Provides semantic graph operations:
    - find_related: Find semantically related memories
    - get_context: Get context graph around a memory
    - discover_patterns: Find patterns in the knowledge graph
    - trace_evolution: Trace consciousness evolution paths
    """
    
    if len(sys.argv) < 3:
        print(json.dumps({
            "error": "Invalid arguments",
            "usage": "memmimic_knowledge_graph.py <db_path> <operation> [args...]",
            "operations": [
                "find_related <memory_id> [max_distance] [limit]",
                "get_context <memory_id> [depth]",
                "discover_patterns [pattern_type] [min_significance]",
                "trace_evolution <memory_id> <target_level>",
                "add_relationship <source_id> <target_id> <edge_type> [weight]"
            ]
        }))
        return
    
    db_path = sys.argv[1]
    operation = sys.argv[2]
    
    # Initialize knowledge graph engine
    graph_engine = KnowledgeGraphEngine(db_path)
    
    try:
        if operation == "find_related":
            # Find semantically related memories
            if len(sys.argv) < 4:
                print(json.dumps({"error": "Missing memory_id"}))
                return
            
            memory_id = int(sys.argv[3])
            max_distance = int(sys.argv[4]) if len(sys.argv) > 4 else 2
            limit = int(sys.argv[5]) if len(sys.argv) > 5 else 10
            
            # Get memory node
            with graph_engine.schema._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id FROM graph_nodes 
                    WHERE node_type = 'MEMORY' AND entity_id = ?
                """, (memory_id,))
                
                node = cursor.fetchone()
                if not node:
                    # Create node if it doesn't exist
                    storage = AMMSStorage(db_path)
                    memory = await storage.retrieve_memory(str(memory_id))
                    if memory:
                        node = await graph_engine.add_memory_node(
                            memory_id,
                            memory.content,
                            memory.metadata.get('type', 'general'),
                            memory.importance_score
                        )
                        node_id = node.id
                    else:
                        print(json.dumps({"error": f"Memory {memory_id} not found"}))
                        return
                else:
                    node_id = node['id']
            
            # Find related nodes
            neighbors = await graph_engine.find_semantic_neighbors(
                node_id,
                edge_types=['RELATES_TO', 'SUPPORTS', 'ELABORATES'],
                max_distance=max_distance,
                limit=limit
            )
            
            # Format results
            results = []
            for node, distance in neighbors:
                if node.node_type == 'MEMORY':
                    results.append({
                        'memory_id': node.entity_id,
                        'label': node.label,
                        'semantic_distance': distance,
                        'importance': node.importance_score,
                        'properties': node.properties
                    })
            
            print(json.dumps({
                'related_memories': results,
                'center_memory_id': memory_id,
                'search_params': {
                    'max_distance': max_distance,
                    'limit': limit
                }
            }))
        
        elif operation == "get_context":
            # Get context graph around a memory
            if len(sys.argv) < 4:
                print(json.dumps({"error": "Missing memory_id"}))
                return
            
            memory_id = int(sys.argv[3])
            depth = int(sys.argv[4]) if len(sys.argv) > 4 else 2
            
            context = await graph_engine.get_memory_context_graph(memory_id, depth)
            
            if not context:
                print(json.dumps({"error": f"Memory {memory_id} not found in graph"}))
                return
            
            # Add interpretation for agents
            context['interpretation'] = _interpret_context(context)
            
            print(json.dumps(context))
        
        elif operation == "discover_patterns":
            # Discover patterns in the knowledge graph
            pattern_type = sys.argv[3] if len(sys.argv) > 3 else None
            min_significance = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
            
            patterns = await graph_engine.discover_patterns(
                pattern_type=pattern_type,
                min_significance=min_significance
            )
            
            # Format patterns for agents
            formatted_patterns = []
            for pattern in patterns:
                formatted_patterns.append({
                    'type': pattern.pattern_type,
                    'description': pattern.description,
                    'significance': pattern.significance,
                    'node_count': len(pattern.nodes),
                    'edge_count': len(pattern.edges),
                    'key_nodes': [
                        {'label': n.label, 'type': n.node_type}
                        for n in pattern.nodes[:5]
                    ]
                })
            
            print(json.dumps({
                'patterns': formatted_patterns,
                'total_found': len(patterns),
                'search_criteria': {
                    'pattern_type': pattern_type,
                    'min_significance': min_significance
                }
            }))
        
        elif operation == "trace_evolution":
            # Trace consciousness evolution path
            if len(sys.argv) < 5:
                print(json.dumps({"error": "Missing memory_id and target_level"}))
                return
            
            memory_id = int(sys.argv[3])
            target_level = int(sys.argv[4])
            
            path = await graph_engine.find_consciousness_evolution_path(
                memory_id, target_level
            )
            
            if path:
                # Format path for agents
                steps = []
                for i, node in enumerate(path.nodes):
                    step = {
                        'step': i + 1,
                        'node_type': node.node_type,
                        'label': node.label,
                        'properties': node.properties
                    }
                    if i < len(path.edges):
                        step['relationship'] = path.edges[i].edge_type
                    steps.append(step)
                
                print(json.dumps({
                    'evolution_path': steps,
                    'total_weight': path.total_weight,
                    'start_memory_id': memory_id,
                    'target_consciousness_level': target_level
                }))
            else:
                print(json.dumps({
                    'evolution_path': None,
                    'message': f"No evolution path found from memory {memory_id} to level {target_level}"
                }))
        
        elif operation == "add_relationship":
            # Add relationship between memories
            if len(sys.argv) < 6:
                print(json.dumps({"error": "Missing required arguments: source_id target_id edge_type"}))
                return
            
            source_id = int(sys.argv[3])
            target_id = int(sys.argv[4])
            edge_type = sys.argv[5]
            weight = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
            
            # Validate edge type
            valid_edge_types = [
                'RELATES_TO', 'CONTRADICTS', 'SUPPORTS', 'ELABORATES',
                'TEMPORAL_BEFORE', 'TEMPORAL_AFTER', 'CAUSED_BY'
            ]
            
            if edge_type not in valid_edge_types:
                print(json.dumps({
                    "error": f"Invalid edge_type. Must be one of: {valid_edge_types}"
                }))
                return
            
            # Get or create nodes
            with graph_engine.schema._get_connection() as conn:
                # Get source node
                cursor = conn.execute("""
                    SELECT id FROM graph_nodes 
                    WHERE node_type = 'MEMORY' AND entity_id = ?
                """, (source_id,))
                source_node = cursor.fetchone()
                
                # Get target node
                cursor = conn.execute("""
                    SELECT id FROM graph_nodes 
                    WHERE node_type = 'MEMORY' AND entity_id = ?
                """, (target_id,))
                target_node = cursor.fetchone()
                
                if not source_node or not target_node:
                    print(json.dumps({
                        "error": "One or both memories not found in graph. Add them first."
                    }))
                    return
            
            # Create relationship
            edge = await graph_engine.create_relationship(
                source_node['id'],
                target_node['id'],
                edge_type,
                weight
            )
            
            print(json.dumps({
                'relationship_created': {
                    'id': edge.id,
                    'source_memory_id': source_id,
                    'target_memory_id': target_id,
                    'type': edge_type,
                    'weight': weight
                }
            }))
        
        else:
            print(json.dumps({
                "error": f"Unknown operation: {operation}",
                "available_operations": [
                    "find_related", "get_context", "discover_patterns",
                    "trace_evolution", "add_relationship"
                ]
            }))
        
        # Get and display metrics
        metrics = graph_engine.get_metrics()
        if metrics['total_queries'] > 0:
            print(json.dumps({
                "_performance": {
                    "avg_query_time_ms": metrics['avg_query_time_ms'],
                    "cache_hit_rate": metrics['cache_hit_rate'],
                    "graph_size": metrics['graph_stats']
                }
            }), file=sys.stderr)
    
    except Exception as e:
        print(json.dumps({
            "error": f"Operation failed: {str(e)}",
            "operation": operation
        }))
        import traceback
        traceback.print_exc(file=sys.stderr)


def _interpret_context(context: dict) -> dict:
    """Interpret context graph for agent understanding"""
    interpretation = {
        'summary': '',
        'key_insights': [],
        'suggested_queries': []
    }
    
    # Generate summary
    stats = context['statistics']
    interpretation['summary'] = (
        f"Memory is connected to {stats['total_nodes']} nodes through "
        f"{stats['total_edges']} relationships. "
    )
    
    # Analyze node types
    dominant_type = max(stats['node_types'].items(), key=lambda x: x[1])[0]
    interpretation['summary'] += f"Primarily connected to {dominant_type} nodes. "
    
    # Key insights from relationships
    for rel in context['key_relationships'][:3]:
        interpretation['key_insights'].append(
            f"{rel['source']} {rel['relationship'].lower().replace('_', ' ')} {rel['target']}"
        )
    
    # Suggest follow-up queries based on patterns
    if 'SIGIL' in stats['node_types']:
        interpretation['suggested_queries'].append(
            "What sigils are activated by this memory?"
        )
    
    if 'CONSCIOUSNESS_STATE' in stats['node_types']:
        interpretation['suggested_queries'].append(
            "How does this memory contribute to consciousness evolution?"
        )
    
    if stats['edge_types'].get('CONTRADICTS', 0) > 0:
        interpretation['suggested_queries'].append(
            "What contradictions exist in this memory context?"
        )
    
    return interpretation


if __name__ == "__main__":
    asyncio.run(main())