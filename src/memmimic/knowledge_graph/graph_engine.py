#!/usr/bin/env python3
"""
Knowledge Graph Engine for MemMimic
Provides intelligent graph traversal and pattern recognition
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np

from .graph_schema import KnowledgeGraphSchema


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: int
    node_type: str
    entity_id: int
    label: str
    properties: Dict[str, Any]
    importance_score: float
    embedding: Optional[np.ndarray] = None


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph"""
    id: int
    source_node_id: int
    target_node_id: int
    edge_type: str
    weight: float
    properties: Dict[str, Any]
    confidence: float


@dataclass
class GraphPath:
    """Represents a path through the knowledge graph"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_weight: float
    path_type: str
    metadata: Dict[str, Any] = None


@dataclass
class GraphPattern:
    """Represents a discovered pattern in the graph"""
    pattern_type: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    significance: float
    description: str


class KnowledgeGraphEngine:
    """
    Engine for intelligent knowledge graph operations
    Enables semantic search, pattern discovery, and consciousness tracking
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.schema = KnowledgeGraphSchema(db_path)
        
        # Ensure schema exists
        if not self._check_schema_exists():
            self.schema.create_knowledge_graph_schema()
            self.logger.info("Created knowledge graph schema")
        
        # Performance metrics
        self._metrics = {
            'queries': 0,
            'avg_query_time_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _check_schema_exists(self) -> bool:
        """Check if knowledge graph schema exists"""
        try:
            with self.schema._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='graph_nodes'
                """)
                return cursor.fetchone() is not None
        except Exception:
            return False
    
    async def add_memory_node(self, 
                            memory_id: int,
                            content: str,
                            memory_type: str,
                            importance: float = 0.5,
                            properties: Dict[str, Any] = None) -> GraphNode:
        """Add a memory as a node in the knowledge graph"""
        with self.schema._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO graph_nodes 
                (node_type, entity_id, label, properties, importance_score, 
                 created_at, updated_at)
                VALUES ('MEMORY', ?, ?, ?, ?, datetime('now'), datetime('now'))
                RETURNING id
            """, (
                memory_id,
                f"Memory: {content[:50]}...",
                json.dumps(properties or {'type': memory_type, 'content': content}),
                importance
            ))
            
            node_id = cursor.fetchone()[0]
            conn.commit()
            
            return GraphNode(
                id=node_id,
                node_type='MEMORY',
                entity_id=memory_id,
                label=f"Memory: {content[:50]}...",
                properties=properties or {'type': memory_type},
                importance_score=importance
            )
    
    async def add_sigil_node(self,
                           sigil_id: int,
                           sigil_symbol: str,
                           sigil_name: str,
                           quantum_state: str,
                           properties: Dict[str, Any] = None) -> GraphNode:
        """Add a sigil as a node in the knowledge graph"""
        with self.schema._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO graph_nodes 
                (node_type, entity_id, label, properties, importance_score,
                 created_at, updated_at)
                VALUES ('SIGIL', ?, ?, ?, 0.8, datetime('now'), datetime('now'))
                RETURNING id
            """, (
                sigil_id,
                f"{sigil_symbol} {sigil_name}",
                json.dumps(properties or {'quantum_state': quantum_state})
            ))
            
            node_id = cursor.fetchone()[0]
            conn.commit()
            
            return GraphNode(
                id=node_id,
                node_type='SIGIL',
                entity_id=sigil_id,
                label=f"{sigil_symbol} {sigil_name}",
                properties=properties or {'quantum_state': quantum_state},
                importance_score=0.8
            )
    
    async def create_relationship(self,
                                source_node_id: int,
                                target_node_id: int,
                                edge_type: str,
                                weight: float = 1.0,
                                properties: Dict[str, Any] = None) -> GraphEdge:
        """Create a relationship between two nodes"""
        with self.schema._get_connection() as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO graph_edges
                (source_node_id, target_node_id, edge_type, weight, 
                 properties, created_at, confidence)
                VALUES (?, ?, ?, ?, ?, datetime('now'), 0.8)
                RETURNING id
            """, (
                source_node_id,
                target_node_id,
                edge_type,
                weight,
                json.dumps(properties or {})
            ))
            
            edge_id = cursor.fetchone()[0]
            conn.commit()
            
            return GraphEdge(
                id=edge_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_type=edge_type,
                weight=weight,
                properties=properties or {},
                confidence=0.8
            )
    
    async def find_semantic_neighbors(self,
                                    node_id: int,
                                    edge_types: Optional[List[str]] = None,
                                    max_distance: int = 2,
                                    limit: int = 10) -> List[Tuple[GraphNode, float]]:
        """Find semantically related nodes using graph traversal"""
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = f"{node_id}:{edge_types}:{max_distance}"
        cached = await self._get_cached_traversal(cache_key)
        if cached:
            self._metrics['cache_hits'] += 1
            return cached
        
        self._metrics['cache_misses'] += 1
        
        # BFS with distance tracking
        visited = {node_id: 0}
        queue = deque([(node_id, 0)])
        results = []
        
        with self.schema._get_connection() as conn:
            while queue and len(results) < limit:
                current_id, distance = queue.popleft()
                
                if distance >= max_distance:
                    continue
                
                # Get edges from current node
                edge_filter = ""
                if edge_types:
                    placeholders = ','.join(['?' for _ in edge_types])
                    edge_filter = f"AND edge_type IN ({placeholders})"
                
                query = f"""
                    SELECT e.*, n.* 
                    FROM graph_edges e
                    JOIN graph_nodes n ON e.target_node_id = n.id
                    WHERE e.source_node_id = ? {edge_filter}
                    ORDER BY e.weight DESC
                """
                
                params = [current_id]
                if edge_types:
                    params.extend(edge_types)
                
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    target_id = row['target_node_id']
                    
                    if target_id not in visited:
                        visited[target_id] = distance + 1
                        queue.append((target_id, distance + 1))
                        
                        # Create node object
                        node = GraphNode(
                            id=target_id,
                            node_type=row['node_type'],
                            entity_id=row['entity_id'],
                            label=row['label'],
                            properties=json.loads(row['properties']),
                            importance_score=row['importance_score']
                        )
                        
                        # Calculate semantic distance
                        semantic_distance = (distance + 1) / row['weight']
                        results.append((node, semantic_distance))
        
        # Sort by semantic distance
        results.sort(key=lambda x: x[1])
        results = results[:limit]
        
        # Cache results
        await self._cache_traversal(cache_key, results)
        
        # Update metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._update_metrics(duration_ms)
        
        return results
    
    async def find_consciousness_evolution_path(self,
                                              start_memory_id: int,
                                              target_consciousness_level: int) -> Optional[GraphPath]:
        """Find path showing consciousness evolution from a memory"""
        with self.schema._get_connection() as conn:
            # Get start node
            cursor = conn.execute("""
                SELECT id FROM graph_nodes 
                WHERE node_type = 'MEMORY' AND entity_id = ?
            """, (start_memory_id,))
            
            start_node = cursor.fetchone()
            if not start_node:
                return None
            
            start_node_id = start_node['id']
            
            # Find consciousness state nodes at target level
            cursor = conn.execute("""
                SELECT n.* FROM graph_nodes n
                WHERE n.node_type = 'CONSCIOUSNESS_STATE'
                  AND json_extract(n.properties, '$.level') = ?
                ORDER BY n.importance_score DESC
                LIMIT 1
            """, (target_consciousness_level,))
            
            target_node = cursor.fetchone()
            if not target_node:
                return None
            
            # Use A* search for pathfinding
            path = await self._find_path_astar(
                start_node_id,
                target_node['id'],
                ['EVOLVES_TO', 'TRANSFORMS_INTO', 'INTEGRATES_WITH']
            )
            
            if path:
                path.path_type = 'CONSCIOUSNESS_EVOLUTION'
                path.metadata = {
                    'start_memory_id': start_memory_id,
                    'target_level': target_consciousness_level
                }
            
            return path
    
    async def discover_patterns(self,
                              pattern_type: Optional[str] = None,
                              min_significance: float = 0.5) -> List[GraphPattern]:
        """Discover patterns in the knowledge graph"""
        patterns = []
        
        with self.schema._get_connection() as conn:
            # Get existing patterns
            query = """
                SELECT * FROM graph_patterns
                WHERE significance_score >= ?
            """
            params = [min_significance]
            
            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type)
            
            query += " ORDER BY significance_score DESC LIMIT 20"
            
            cursor = conn.execute(query, params)
            
            for row in cursor:
                # Parse pattern criteria
                node_criteria = json.loads(row['node_pattern'])
                edge_criteria = json.loads(row['edge_pattern'])
                
                # Find matching subgraphs
                matching_nodes, matching_edges = await self._find_matching_subgraph(
                    node_criteria, edge_criteria
                )
                
                if matching_nodes:
                    pattern = GraphPattern(
                        pattern_type=row['pattern_type'],
                        nodes=matching_nodes,
                        edges=matching_edges,
                        significance=row['significance_score'],
                        description=row['pattern_name']
                    )
                    patterns.append(pattern)
        
        # Discover new patterns if requested
        if not pattern_type or pattern_type == 'MEMORY_CLUSTER':
            new_clusters = await self._discover_memory_clusters()
            patterns.extend(new_clusters)
        
        if not pattern_type or pattern_type == 'SIGIL_CONSTELLATION':
            new_constellations = await self._discover_sigil_constellations()
            patterns.extend(new_constellations)
        
        return patterns
    
    async def get_memory_context_graph(self,
                                     memory_id: int,
                                     depth: int = 2) -> Dict[str, Any]:
        """Get the context graph around a memory for agent understanding"""
        with self.schema._get_connection() as conn:
            # Get memory node
            cursor = conn.execute("""
                SELECT * FROM graph_nodes 
                WHERE node_type = 'MEMORY' AND entity_id = ?
            """, (memory_id,))
            
            center_node = cursor.fetchone()
            if not center_node:
                return {}
            
            # Get neighborhood
            nodes = {center_node['id']: dict(center_node)}
            edges = []
            
            # BFS to specified depth
            queue = deque([(center_node['id'], 0)])
            visited = {center_node['id']}
            
            while queue:
                current_id, current_depth = queue.popleft()
                
                if current_depth >= depth:
                    continue
                
                # Get all edges
                cursor = conn.execute("""
                    SELECT e.*, n.*
                    FROM graph_edges e
                    JOIN graph_nodes n ON 
                        (e.target_node_id = n.id AND e.source_node_id = ?)
                        OR (e.source_node_id = n.id AND e.target_node_id = ?)
                    WHERE n.id != ?
                """, (current_id, current_id, current_id))
                
                for row in cursor:
                    node_id = row['id']
                    
                    if node_id not in visited:
                        visited.add(node_id)
                        nodes[node_id] = {
                            'id': node_id,
                            'type': row['node_type'],
                            'label': row['label'],
                            'properties': json.loads(row['properties']),
                            'importance': row['importance_score']
                        }
                        queue.append((node_id, current_depth + 1))
                    
                    # Add edge
                    edges.append({
                        'source': row['source_node_id'],
                        'target': row['target_node_id'],
                        'type': row['edge_type'],
                        'weight': row['weight']
                    })
            
            # Build context summary
            context = {
                'center_memory_id': memory_id,
                'graph': {
                    'nodes': list(nodes.values()),
                    'edges': edges
                },
                'statistics': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'node_types': defaultdict(int),
                    'edge_types': defaultdict(int)
                },
                'key_relationships': [],
                'semantic_clusters': []
            }
            
            # Calculate statistics
            for node in nodes.values():
                context['statistics']['node_types'][node['type']] += 1
            
            for edge in edges:
                context['statistics']['edge_types'][edge['type']] += 1
            
            # Find key relationships
            important_edges = sorted(edges, key=lambda e: e['weight'], reverse=True)[:5]
            for edge in important_edges:
                source = nodes.get(edge['source'], {})
                target = nodes.get(edge['target'], {})
                context['key_relationships'].append({
                    'source': source.get('label', 'Unknown'),
                    'relationship': edge['type'],
                    'target': target.get('label', 'Unknown'),
                    'strength': edge['weight']
                })
            
            return context
    
    async def _find_path_astar(self,
                             start_id: int,
                             end_id: int,
                             allowed_edge_types: List[str]) -> Optional[GraphPath]:
        """A* pathfinding between nodes"""
        # Implementation of A* search
        # Returns GraphPath object or None
        # ... (implementation details)
        pass
    
    async def _find_matching_subgraph(self,
                                    node_criteria: List[Dict],
                                    edge_criteria: List[Dict]) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Find subgraphs matching the given criteria"""
        # Implementation of subgraph matching
        # ... (implementation details)
        pass
    
    async def _discover_memory_clusters(self) -> List[GraphPattern]:
        """Discover clusters of related memories"""
        # Implementation of clustering algorithm
        # ... (implementation details)
        pass
    
    async def _discover_sigil_constellations(self) -> List[GraphPattern]:
        """Discover patterns in sigil activations"""
        # Implementation of constellation discovery
        # ... (implementation details)
        pass
    
    async def _get_cached_traversal(self, cache_key: str) -> Optional[List]:
        """Get cached traversal results"""
        with self.schema._get_connection() as conn:
            cursor = conn.execute("""
                SELECT path_data FROM graph_traversal_cache
                WHERE start_node_id = ? 
                  AND traversal_type = ?
                  AND (expires_at IS NULL OR expires_at > datetime('now'))
                LIMIT 1
            """, cache_key.split(':')[:2])
            
            result = cursor.fetchone()
            if result:
                return json.loads(result['path_data'])
        return None
    
    async def _cache_traversal(self, cache_key: str, results: List) -> None:
        """Cache traversal results"""
        with self.schema._get_connection() as conn:
            parts = cache_key.split(':')
            conn.execute("""
                INSERT OR REPLACE INTO graph_traversal_cache
                (start_node_id, traversal_type, path_data, path_length,
                 cached_at, expires_at)
                VALUES (?, ?, ?, ?, datetime('now'), datetime('now', '+1 hour'))
            """, (
                int(parts[0]),
                parts[1] if len(parts) > 1 else 'default',
                json.dumps(results),
                len(results)
            ))
            conn.commit()
    
    def _update_metrics(self, duration_ms: float):
        """Update performance metrics"""
        self._metrics['queries'] += 1
        current_avg = self._metrics['avg_query_time_ms']
        count = self._metrics['queries']
        self._metrics['avg_query_time_ms'] = (
            (current_avg * (count - 1) + duration_ms) / count
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        cache_rate = 0
        total_cache_ops = self._metrics['cache_hits'] + self._metrics['cache_misses']
        if total_cache_ops > 0:
            cache_rate = self._metrics['cache_hits'] / total_cache_ops
        
        return {
            'total_queries': self._metrics['queries'],
            'avg_query_time_ms': self._metrics['avg_query_time_ms'],
            'cache_hit_rate': cache_rate,
            'graph_stats': self._get_graph_statistics()
        }
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get current graph statistics"""
        with self.schema._get_connection() as conn:
            # Node count
            cursor = conn.execute("SELECT COUNT(*) as count FROM graph_nodes")
            node_count = cursor.fetchone()['count']
            
            # Edge count
            cursor = conn.execute("SELECT COUNT(*) as count FROM graph_edges")
            edge_count = cursor.fetchone()['count']
            
            # Average degree
            avg_degree = (2 * edge_count / node_count) if node_count > 0 else 0
            
            return {
                'nodes': node_count,
                'edges': edge_count,
                'avg_degree': avg_degree
            }