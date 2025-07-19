#!/usr/bin/env python3
"""
Knowledge Graph Schema for MemMimic
Enables semantic relationships between memories, consciousness states, and sigils
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging
import json


class KnowledgeGraphSchema:
    """Database schema for knowledge graph features"""
    
    SCHEMA_VERSION = "3.0.0"  # AMMS + Consciousness + Knowledge Graph
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA encoding = 'UTF-8'")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_knowledge_graph_schema(self) -> None:
        """Create knowledge graph schema extensions"""
        with self._get_connection() as conn:
            # Node types in the knowledge graph
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_type TEXT NOT NULL CHECK (node_type IN (
                        'MEMORY', 'CONCEPT', 'SIGIL', 'CONSCIOUSNESS_STATE', 
                        'PROMPT', 'TALE', 'PATTERN', 'SHADOW_ASPECT'
                    )),
                    entity_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    properties TEXT NOT NULL, -- JSON properties
                    embedding BLOB, -- Vector embedding for semantic search
                    importance_score REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    
                    UNIQUE(node_type, entity_id)
                )
            """)
            
            # Edge types representing relationships
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_node_id INTEGER NOT NULL,
                    target_node_id INTEGER NOT NULL,
                    edge_type TEXT NOT NULL CHECK (edge_type IN (
                        -- Memory relationships
                        'RELATES_TO', 'CONTRADICTS', 'SUPPORTS', 'ELABORATES',
                        'TEMPORAL_BEFORE', 'TEMPORAL_AFTER', 'CAUSED_BY',
                        
                        -- Consciousness relationships
                        'EVOLVES_TO', 'TRANSFORMS_INTO', 'INTEGRATES_WITH',
                        'SHADOW_OF', 'LIGHT_OF', 'UNITY_WITH',
                        
                        -- Sigil relationships
                        'ACTIVATES', 'SYNERGIZES_WITH', 'CONFLICTS_WITH',
                        'QUANTUM_ENTANGLED', 'AMPLIFIES', 'NULLIFIES',
                        
                        -- Semantic relationships
                        'IS_A', 'PART_OF', 'INSTANCE_OF', 'BELONGS_TO',
                        'DERIVED_FROM', 'INFLUENCES', 'MANIFESTS_AS'
                    )),
                    weight REAL DEFAULT 1.0,
                    properties TEXT, -- JSON for edge-specific data
                    created_at TEXT NOT NULL,
                    confidence REAL DEFAULT 0.8,
                    
                    FOREIGN KEY (source_node_id) REFERENCES graph_nodes(id),
                    FOREIGN KEY (target_node_id) REFERENCES graph_nodes(id),
                    UNIQUE(source_node_id, target_node_id, edge_type)
                )
            """)
            
            # Concept hierarchy for semantic organization
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concept_hierarchy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_name TEXT NOT NULL UNIQUE,
                    parent_concept_id INTEGER,
                    level INTEGER NOT NULL DEFAULT 0,
                    description TEXT,
                    properties TEXT, -- JSON metadata
                    
                    FOREIGN KEY (parent_concept_id) REFERENCES concept_hierarchy(id)
                )
            """)
            
            # Graph patterns for consciousness evolution
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    pattern_type TEXT NOT NULL CHECK (pattern_type IN (
                        'CONSCIOUSNESS_EVOLUTION', 'MEMORY_CLUSTER',
                        'SIGIL_CONSTELLATION', 'SHADOW_INTEGRATION',
                        'UNITY_EMERGENCE', 'RECURSIVE_LOOP'
                    )),
                    node_pattern TEXT NOT NULL, -- JSON array of node criteria
                    edge_pattern TEXT NOT NULL, -- JSON array of edge criteria
                    significance_score REAL DEFAULT 0.5,
                    occurrence_count INTEGER DEFAULT 0,
                    first_detected TEXT NOT NULL,
                    last_observed TEXT NOT NULL,
                    metadata TEXT -- JSON for pattern-specific data
                )
            """)
            
            # Semantic clusters for memory organization
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_name TEXT NOT NULL,
                    centroid_node_id INTEGER,
                    cluster_type TEXT CHECK (cluster_type IN (
                        'TOPIC', 'EMOTION', 'CONTEXT', 'TEMPORAL',
                        'CONSCIOUSNESS', 'ARCHETYPAL'
                    )),
                    coherence_score REAL DEFAULT 0.0,
                    member_count INTEGER DEFAULT 0,
                    properties TEXT, -- JSON cluster properties
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    
                    FOREIGN KEY (centroid_node_id) REFERENCES graph_nodes(id)
                )
            """)
            
            # Cluster membership
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cluster_members (
                    cluster_id INTEGER NOT NULL,
                    node_id INTEGER NOT NULL,
                    membership_strength REAL DEFAULT 1.0,
                    joined_at TEXT NOT NULL,
                    
                    PRIMARY KEY (cluster_id, node_id),
                    FOREIGN KEY (cluster_id) REFERENCES semantic_clusters(id),
                    FOREIGN KEY (node_id) REFERENCES graph_nodes(id)
                )
            """)
            
            # Graph traversal cache for performance
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_traversal_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_node_id INTEGER NOT NULL,
                    end_node_id INTEGER,
                    traversal_type TEXT NOT NULL,
                    path_data TEXT NOT NULL, -- JSON array of node IDs
                    path_length INTEGER NOT NULL,
                    total_weight REAL,
                    cached_at TEXT NOT NULL,
                    expires_at TEXT,
                    
                    FOREIGN KEY (start_node_id) REFERENCES graph_nodes(id),
                    FOREIGN KEY (end_node_id) REFERENCES graph_nodes(id)
                )
            """)
            
            # Knowledge graph metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    node_count INTEGER,
                    edge_count INTEGER,
                    avg_degree REAL,
                    clustering_coefficient REAL,
                    calculated_at TEXT NOT NULL,
                    metadata TEXT -- JSON for additional metrics
                )
            """)
            
            self._create_knowledge_indices(conn)
            self._insert_base_concepts(conn)
            conn.commit()
            
            # Update schema version
            conn.execute("""
                INSERT OR REPLACE INTO schema_version (version, applied_at, description)
                VALUES (?, ?, ?)
            """, (
                self.SCHEMA_VERSION,
                datetime.now().isoformat(),
                "Knowledge Graph schema for semantic memory organization"
            ))
            conn.commit()
    
    def _create_knowledge_indices(self, conn) -> None:
        """Create performance indices for knowledge graph queries"""
        indices = [
            # Node indices
            "CREATE INDEX IF NOT EXISTS idx_nodes_type ON graph_nodes(node_type)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_importance ON graph_nodes(importance_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_label ON graph_nodes(label)",
            
            # Edge indices
            "CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges(edge_type)",
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_node_id)",
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_node_id)",
            "CREATE INDEX IF NOT EXISTS idx_edges_weight ON graph_edges(weight DESC)",
            
            # Pattern indices
            "CREATE INDEX IF NOT EXISTS idx_patterns_type ON graph_patterns(pattern_type)",
            "CREATE INDEX IF NOT EXISTS idx_patterns_significance ON graph_patterns(significance_score DESC)",
            
            # Cluster indices
            "CREATE INDEX IF NOT EXISTS idx_clusters_type ON semantic_clusters(cluster_type)",
            "CREATE INDEX IF NOT EXISTS idx_clusters_coherence ON semantic_clusters(coherence_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_cluster_members ON cluster_members(cluster_id, membership_strength DESC)",
            
            # Cache indices
            "CREATE INDEX IF NOT EXISTS idx_cache_traversal ON graph_traversal_cache(start_node_id, traversal_type)",
            "CREATE INDEX IF NOT EXISTS idx_cache_expiry ON graph_traversal_cache(expires_at)"
        ]
        
        for index_sql in indices:
            conn.execute(index_sql)
    
    def _insert_base_concepts(self, conn) -> None:
        """Insert foundational concept hierarchy"""
        base_concepts = [
            # Root concepts
            (None, "CONSCIOUSNESS", 0, "Root concept for consciousness-related nodes"),
            (None, "KNOWLEDGE", 0, "Root concept for knowledge and information"),
            (None, "EMOTION", 0, "Root concept for emotional states"),
            (None, "TIME", 0, "Root concept for temporal relationships"),
            
            # Consciousness subconcepts
            (1, "UNITY", 1, "States of unified consciousness"),
            (1, "SHADOW", 1, "Shadow aspects and integration"),
            (1, "EVOLUTION", 1, "Consciousness evolution stages"),
            (1, "AWARENESS", 1, "Levels of awareness"),
            
            # Knowledge subconcepts
            (2, "PATTERN", 1, "Recognized patterns and structures"),
            (2, "INSIGHT", 1, "Derived insights and understanding"),
            (2, "WISDOM", 1, "Synthesized wisdom"),
            (2, "MEMORY", 1, "Stored memories and experiences"),
            
            # Emotion subconcepts
            (3, "POSITIVE", 1, "Positive emotional states"),
            (3, "NEGATIVE", 1, "Negative emotional states"),
            (3, "NEUTRAL", 1, "Neutral or balanced states"),
            (3, "TRANSFORMATIVE", 1, "Emotions driving change")
        ]
        
        concept_map = {None: None}  # Maps old IDs to new IDs
        
        for parent_idx, name, level, desc in base_concepts:
            parent_id = concept_map.get(parent_idx)
            cursor = conn.execute("""
                INSERT OR IGNORE INTO concept_hierarchy 
                (concept_name, parent_concept_id, level, description, properties)
                VALUES (?, ?, ?, ?, '{}')
            """, (name, parent_id, level, desc))
            
            # Store the mapping for child concepts
            if parent_idx is None:
                concept_map[len(concept_map)] = cursor.lastrowid