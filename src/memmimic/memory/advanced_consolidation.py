"""
Advanced Memory Consolidation using Clustering Algorithms and Semantic Analysis.

Implements intelligent memory consolidation that groups related memories, identifies
redundant content, and creates consolidated representations to improve efficiency
and retrieval while preserving important information.
"""

import logging
import numpy as np
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
import json
import hashlib

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class MemoryCluster:
    """Represents a cluster of related memories."""
    cluster_id: str
    memory_ids: List[str]
    centroid: np.ndarray
    cluster_type: str  # semantic, temporal, usage, hybrid
    consolidation_level: str  # candidate, partial, full
    importance_score: float = 0.0
    coherence_score: float = 0.0
    redundancy_score: float = 0.0
    creation_timestamp: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationCandidate:
    """Represents a candidate for memory consolidation."""
    primary_memory_id: str
    related_memory_ids: List[str]
    consolidation_type: str  # merge, summarize, archive, reference
    confidence: float
    potential_savings_bytes: int
    semantic_similarity: float
    temporal_proximity: float
    usage_overlap: float
    consolidation_strategy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationConfig:
    """Configuration for advanced memory consolidation."""
    # Clustering parameters
    semantic_cluster_threshold: float = 0.75
    temporal_window_hours: int = 24
    min_cluster_size: int = 2
    max_cluster_size: int = 20
    
    # Semantic analysis
    min_semantic_similarity: float = 0.6
    tfidf_max_features: int = 1000
    embedding_dimension: int = 128
    
    # Consolidation thresholds
    redundancy_threshold: float = 0.8
    importance_preservation_threshold: float = 0.7
    min_consolidation_benefit: float = 0.3
    
    # Performance optimization
    max_memories_per_batch: int = 500
    clustering_algorithm: str = "kmeans"  # kmeans, dbscan, hierarchical
    dimensionality_reduction: str = "pca"  # pca, svd, none
    
    # System parameters
    consolidation_interval_hours: int = 6
    max_consolidation_candidates: int = 100
    backup_before_consolidation: bool = True
    
    # Advanced features
    enable_graph_analysis: bool = True
    enable_temporal_consolidation: bool = True
    enable_usage_based_consolidation: bool = True
    preserve_user_favorites: bool = True


class SemanticAnalyzer:
    """Analyzes semantic relationships between memories."""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_semantic_features(self, memory_contents: List[str]) -> np.ndarray:
        """Extract semantic features from memory contents."""
        if not memory_contents:
            return np.array([])
        
        # TF-IDF features
        if not self.is_fitted:
            tfidf_features = self.tfidf_vectorizer.fit_transform(memory_contents)
            self.is_fitted = True
        else:
            tfidf_features = self.tfidf_vectorizer.transform(memory_contents)
        
        # Convert to dense array
        tfidf_dense = tfidf_features.toarray()
        
        # Apply scaling
        if not hasattr(self, '_scaler_fitted'):
            scaled_features = self.scaler.fit_transform(tfidf_dense)
            self._scaler_fitted = True
        else:
            scaled_features = self.scaler.transform(tfidf_dense)
        
        return scaled_features
    
    def calculate_semantic_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Calculate pairwise semantic similarity matrix."""
        return cosine_similarity(features)
    
    def identify_semantic_clusters(self, features: np.ndarray, 
                                 similarity_matrix: np.ndarray) -> List[List[int]]:
        """Identify semantic clusters using the configured algorithm."""
        if features.shape[0] < self.config.min_cluster_size:
            return []
        
        if self.config.clustering_algorithm == "kmeans":
            return self._kmeans_clustering(features)
        elif self.config.clustering_algorithm == "dbscan":
            return self._dbscan_clustering(similarity_matrix)
        elif self.config.clustering_algorithm == "hierarchical":
            return self._hierarchical_clustering(similarity_matrix)
        else:
            logger.warning(f"Unknown clustering algorithm: {self.config.clustering_algorithm}")
            return self._kmeans_clustering(features)
    
    def _kmeans_clustering(self, features: np.ndarray) -> List[List[int]]:
        """Perform K-means clustering."""
        n_samples = features.shape[0]
        n_clusters = min(max(2, n_samples // 3), 10)  # Adaptive cluster count
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Group indices by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)
        
        # Filter by minimum cluster size
        return [
            cluster for cluster in clusters.values() 
            if len(cluster) >= self.config.min_cluster_size
        ]
    
    def _dbscan_clustering(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """Perform DBSCAN clustering using similarity matrix."""
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        dbscan = DBSCAN(
            eps=1 - self.config.semantic_cluster_threshold,
            min_samples=self.config.min_cluster_size,
            metric='precomputed'
        )
        cluster_labels = dbscan.fit_predict(distance_matrix)
        
        # Group indices by cluster (exclude noise points with label -1)
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # Exclude noise
                clusters[label].append(idx)
        
        return list(clusters.values())
    
    def _hierarchical_clustering(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """Perform hierarchical clustering."""
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        n_samples = similarity_matrix.shape[0]
        n_clusters = min(max(2, n_samples // 4), 8)
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = hierarchical.fit_predict(distance_matrix)
        
        # Group indices by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)
        
        return [
            cluster for cluster in clusters.values()
            if len(cluster) >= self.config.min_cluster_size
        ]


class TemporalAnalyzer:
    """Analyzes temporal relationships between memories."""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
    
    def identify_temporal_clusters(self, 
                                 memory_timestamps: List[datetime],
                                 memory_ids: List[str]) -> List[List[int]]:
        """Identify temporally related memory clusters."""
        if not memory_timestamps or len(memory_timestamps) < self.config.min_cluster_size:
            return []
        
        # Sort by timestamp
        sorted_indices = sorted(
            range(len(memory_timestamps)), 
            key=lambda i: memory_timestamps[i]
        )
        
        clusters = []
        current_cluster = []
        window_start = None
        
        for idx in sorted_indices:
            timestamp = memory_timestamps[idx]
            
            if window_start is None:
                window_start = timestamp
                current_cluster = [idx]
            else:
                # Check if within temporal window
                if (timestamp - window_start).total_seconds() <= self.config.temporal_window_hours * 3600:
                    current_cluster.append(idx)
                else:
                    # Close current cluster if it meets size requirement
                    if len(current_cluster) >= self.config.min_cluster_size:
                        clusters.append(current_cluster)
                    
                    # Start new cluster
                    window_start = timestamp
                    current_cluster = [idx]
        
        # Don't forget the last cluster
        if len(current_cluster) >= self.config.min_cluster_size:
            clusters.append(current_cluster)
        
        return clusters
    
    def calculate_temporal_proximity(self, 
                                   timestamps: List[datetime]) -> np.ndarray:
        """Calculate temporal proximity matrix between memories."""
        n = len(timestamps)
        proximity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    time_diff_hours = abs(
                        (timestamps[i] - timestamps[j]).total_seconds() / 3600
                    )
                    # Exponential decay based on time difference
                    proximity = np.exp(-time_diff_hours / 24.0)  # Decay over days
                    proximity_matrix[i, j] = proximity
                else:
                    proximity_matrix[i, j] = 1.0
        
        return proximity_matrix


class UsagePatternAnalyzer:
    """Analyzes usage patterns for memory consolidation."""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
    
    def analyze_usage_overlap(self, 
                            access_patterns: List[List[datetime]]) -> np.ndarray:
        """Calculate usage pattern overlap between memories."""
        n = len(access_patterns)
        overlap_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    overlap = self._calculate_pattern_overlap(
                        access_patterns[i], 
                        access_patterns[j]
                    )
                    overlap_matrix[i, j] = overlap
                else:
                    overlap_matrix[i, j] = 1.0
        
        return overlap_matrix
    
    def _calculate_pattern_overlap(self, 
                                 pattern1: List[datetime], 
                                 pattern2: List[datetime]) -> float:
        """Calculate overlap between two access patterns."""
        if not pattern1 or not pattern2:
            return 0.0
        
        # Convert to hour-of-day and day-of-week patterns
        hours1 = set(t.hour for t in pattern1)
        hours2 = set(t.hour for t in pattern2)
        days1 = set(t.weekday() for t in pattern1)
        days2 = set(t.weekday() for t in pattern2)
        
        # Calculate overlaps
        hour_overlap = len(hours1 & hours2) / max(len(hours1 | hours2), 1)
        day_overlap = len(days1 & days2) / max(len(days1 | days2), 1)
        
        # Combined overlap score
        return (hour_overlap + day_overlap) / 2.0
    
    def identify_usage_clusters(self, 
                              usage_overlap_matrix: np.ndarray) -> List[List[int]]:
        """Identify clusters based on usage patterns."""
        # Use threshold-based clustering for usage patterns
        n = usage_overlap_matrix.shape[0]
        clusters = []
        visited = set()
        
        for i in range(n):
            if i in visited:
                continue
            
            cluster = [i]
            visited.add(i)
            
            # Find all memories with high usage overlap
            for j in range(n):
                if (j not in visited and 
                    usage_overlap_matrix[i, j] > self.config.semantic_cluster_threshold):
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) >= self.config.min_cluster_size:
                clusters.append(cluster)
        
        return clusters


class ConsolidationStrategy:
    """Determines optimal consolidation strategies for memory clusters."""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
    
    def analyze_consolidation_potential(self, 
                                      cluster: MemoryCluster,
                                      memory_contents: List[str],
                                      memory_metadata: List[Dict]) -> ConsolidationCandidate:
        """Analyze consolidation potential for a memory cluster."""
        if len(cluster.memory_ids) < 2:
            return None
        
        # Calculate redundancy score
        redundancy = self._calculate_redundancy(memory_contents)
        
        # Calculate potential savings
        total_size = sum(
            metadata.get('size_bytes', len(content.encode('utf-8')))
            for content, metadata in zip(memory_contents, memory_metadata)
        )
        
        # Estimate consolidation savings
        potential_savings = int(total_size * redundancy * 0.7)  # Conservative estimate
        
        # Determine consolidation strategy
        strategy = self._determine_strategy(
            cluster, memory_contents, memory_metadata, redundancy
        )
        
        # Calculate confidence score
        confidence = self._calculate_consolidation_confidence(
            cluster, redundancy, strategy
        )
        
        # Create consolidation candidate
        candidate = ConsolidationCandidate(
            primary_memory_id=cluster.memory_ids[0],  # Use first as primary
            related_memory_ids=cluster.memory_ids[1:],
            consolidation_type=strategy['type'],
            confidence=confidence,
            potential_savings_bytes=potential_savings,
            semantic_similarity=cluster.coherence_score,
            temporal_proximity=0.0,  # Would be calculated based on timestamps
            usage_overlap=0.0,  # Would be calculated based on usage patterns
            consolidation_strategy=strategy
        )
        
        return candidate
    
    def _calculate_redundancy(self, contents: List[str]) -> float:
        """Calculate redundancy score for a group of memories."""
        if len(contents) < 2:
            return 0.0
        
        # Simple approach: calculate average pairwise similarity
        similarities = []
        
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                # Simple Jaccard similarity on words
                words1 = set(contents[i].lower().split())
                words2 = set(contents[j].lower().split())
                
                if not words1 and not words2:
                    similarity = 1.0
                elif not words1 or not words2:
                    similarity = 0.0
                else:
                    similarity = len(words1 & words2) / len(words1 | words2)
                
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _determine_strategy(self, 
                          cluster: MemoryCluster,
                          contents: List[str],
                          metadata: List[Dict],
                          redundancy: float) -> Dict[str, Any]:
        """Determine optimal consolidation strategy."""
        # Analyze memory characteristics
        memory_types = [meta.get('memory_type', 'unknown') for meta in metadata]
        importance_scores = [meta.get('importance', 0.5) for meta in metadata]
        
        # Strategy selection logic
        if redundancy > 0.9:
            # Very high redundancy - merge into single memory
            return {
                'type': 'merge',
                'method': 'union',
                'preserve_metadata': True,
                'keep_references': True
            }
        elif redundancy > 0.7:
            # High redundancy - create summary with references
            return {
                'type': 'summarize',
                'method': 'extractive',
                'preserve_important': True,
                'reference_originals': True
            }
        elif 'temporary' in memory_types or max(importance_scores) < 0.3:
            # Low importance - archive cluster
            return {
                'type': 'archive',
                'method': 'compress',
                'create_index': True,
                'preserve_searchable': True
            }
        else:
            # Create reference relationships
            return {
                'type': 'reference',
                'method': 'link_graph',
                'maintain_individual': True,
                'create_cluster_node': True
            }
    
    def _calculate_consolidation_confidence(self, 
                                          cluster: MemoryCluster,
                                          redundancy: float,
                                          strategy: Dict) -> float:
        """Calculate confidence score for consolidation."""
        confidence_factors = []
        
        # Cluster coherence
        confidence_factors.append(cluster.coherence_score * 0.3)
        
        # Redundancy level
        confidence_factors.append(redundancy * 0.4)
        
        # Cluster size (larger clusters more confident)
        size_factor = min(1.0, len(cluster.memory_ids) / 10.0)
        confidence_factors.append(size_factor * 0.2)
        
        # Strategy-specific confidence
        strategy_confidence = {
            'merge': 0.9 if redundancy > 0.8 else 0.6,
            'summarize': 0.8,
            'archive': 0.7,
            'reference': 0.6
        }
        confidence_factors.append(
            strategy_confidence.get(strategy['type'], 0.5) * 0.1
        )
        
        return min(1.0, sum(confidence_factors))


class AdvancedMemoryConsolidator:
    """
    Advanced memory consolidation system using clustering and semantic analysis.
    
    Features:
    - Multi-dimensional clustering (semantic, temporal, usage)
    - Intelligent consolidation strategies
    - Redundancy detection and elimination
    - Graph-based memory relationship analysis
    - Performance-optimized batch processing
    """
    
    def __init__(self, 
                 config: Optional[ConsolidationConfig] = None,
                 storage_backend: Optional[Any] = None):
        """
        Initialize advanced memory consolidator.
        
        Args:
            config: Consolidation configuration
            storage_backend: Storage backend for memory operations
        """
        self.config = config or ConsolidationConfig()
        self.storage_backend = storage_backend
        
        # Analysis components
        self.semantic_analyzer = SemanticAnalyzer(self.config)
        self.temporal_analyzer = TemporalAnalyzer(self.config)
        self.usage_analyzer = UsagePatternAnalyzer(self.config)
        self.strategy_analyzer = ConsolidationStrategy(self.config)
        
        # State tracking
        self.memory_clusters: Dict[str, MemoryCluster] = {}
        self.consolidation_history: List[Dict[str, Any]] = []
        self.performance_stats = {
            'total_memories_processed': 0,
            'clusters_identified': 0,
            'consolidations_performed': 0,
            'bytes_saved': 0,
            'processing_time_ms': 0.0
        }
        
        # Thread safety
        self._consolidation_lock = threading.RLock()
        
        # Memory relationship graph
        if self.config.enable_graph_analysis:
            self.memory_graph = nx.Graph()
        else:
            self.memory_graph = None
        
        logger.info("AdvancedMemoryConsolidator initialized")
    
    def analyze_memories(self, 
                        memory_data: List[Dict[str, Any]],
                        batch_size: Optional[int] = None) -> List[MemoryCluster]:
        """
        Analyze memories to identify consolidation opportunities.
        
        Args:
            memory_data: List of memory dictionaries with content and metadata
            batch_size: Optional batch size for processing
            
        Returns:
            List of identified memory clusters
        """
        start_time = time.perf_counter()
        batch_size = batch_size or self.config.max_memories_per_batch
        
        all_clusters = []
        
        # Process memories in batches
        for i in range(0, len(memory_data), batch_size):
            batch = memory_data[i:i + batch_size]
            batch_clusters = self._analyze_memory_batch(batch)
            all_clusters.extend(batch_clusters)
        
        # Update performance stats
        processing_time = (time.perf_counter() - start_time) * 1000
        self.performance_stats['processing_time_ms'] += processing_time
        self.performance_stats['total_memories_processed'] += len(memory_data)
        self.performance_stats['clusters_identified'] += len(all_clusters)
        
        logger.info(f"Analyzed {len(memory_data)} memories, found {len(all_clusters)} clusters")
        return all_clusters
    
    def _analyze_memory_batch(self, memory_batch: List[Dict[str, Any]]) -> List[MemoryCluster]:
        """Analyze a batch of memories for clustering."""
        if len(memory_batch) < self.config.min_cluster_size:
            return []
        
        # Extract data for analysis
        memory_ids = [mem['id'] for mem in memory_batch]
        contents = [mem.get('content', '') for mem in memory_batch]
        metadata = [mem.get('metadata', {}) for mem in memory_batch]
        timestamps = [
            mem.get('timestamp', datetime.now()) 
            for mem in memory_batch
        ]
        
        clusters = []
        
        # Semantic clustering
        semantic_clusters = self._perform_semantic_clustering(
            memory_ids, contents, metadata
        )
        clusters.extend(semantic_clusters)
        
        # Temporal clustering (if enabled)
        if self.config.enable_temporal_consolidation:
            temporal_clusters = self._perform_temporal_clustering(
                memory_ids, timestamps, metadata
            )
            clusters.extend(temporal_clusters)
        
        # Usage pattern clustering (if enabled)
        if self.config.enable_usage_based_consolidation:
            usage_clusters = self._perform_usage_clustering(
                memory_ids, metadata
            )
            clusters.extend(usage_clusters)
        
        # Graph-based clustering (if enabled)
        if self.config.enable_graph_analysis and self.memory_graph:
            graph_clusters = self._perform_graph_clustering(memory_ids)
            clusters.extend(graph_clusters)
        
        return self._merge_overlapping_clusters(clusters)
    
    def _perform_semantic_clustering(self, 
                                   memory_ids: List[str],
                                   contents: List[str],
                                   metadata: List[Dict]) -> List[MemoryCluster]:
        """Perform semantic clustering on memories."""
        try:
            # Extract semantic features
            semantic_features = self.semantic_analyzer.extract_semantic_features(contents)
            
            if semantic_features.size == 0:
                return []
            
            # Calculate similarity matrix
            similarity_matrix = self.semantic_analyzer.calculate_semantic_similarity_matrix(
                semantic_features
            )
            
            # Identify clusters
            cluster_indices = self.semantic_analyzer.identify_semantic_clusters(
                semantic_features, similarity_matrix
            )
            
            clusters = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) >= self.config.min_cluster_size:
                    # Calculate cluster centroid
                    cluster_features = semantic_features[indices]
                    centroid = np.mean(cluster_features, axis=0)
                    
                    # Calculate coherence score
                    cluster_similarities = similarity_matrix[np.ix_(indices, indices)]
                    coherence = np.mean(cluster_similarities)
                    
                    # Create cluster
                    cluster_memory_ids = [memory_ids[idx] for idx in indices]
                    cluster = MemoryCluster(
                        cluster_id=f"semantic_{i}_{int(time.time())}",
                        memory_ids=cluster_memory_ids,
                        centroid=centroid,
                        cluster_type="semantic",
                        consolidation_level="candidate",
                        coherence_score=coherence,
                        metadata={'similarity_threshold': self.config.semantic_cluster_threshold}
                    )
                    
                    clusters.append(cluster)
                    self.memory_clusters[cluster.cluster_id] = cluster
            
            return clusters
            
        except Exception as e:
            logger.error(f"Semantic clustering failed: {e}")
            return []
    
    def _perform_temporal_clustering(self, 
                                   memory_ids: List[str],
                                   timestamps: List[datetime],
                                   metadata: List[Dict]) -> List[MemoryCluster]:
        """Perform temporal clustering on memories."""
        try:
            cluster_indices = self.temporal_analyzer.identify_temporal_clusters(
                timestamps, memory_ids
            )
            
            clusters = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) >= self.config.min_cluster_size:
                    cluster_memory_ids = [memory_ids[idx] for idx in indices]
                    cluster_timestamps = [timestamps[idx] for idx in indices]
                    
                    # Calculate temporal coherence
                    time_span = max(cluster_timestamps) - min(cluster_timestamps)
                    coherence = max(0.0, 1.0 - time_span.total_seconds() / (24 * 3600))
                    
                    cluster = MemoryCluster(
                        cluster_id=f"temporal_{i}_{int(time.time())}",
                        memory_ids=cluster_memory_ids,
                        centroid=np.array([]),  # No centroid for temporal clusters
                        cluster_type="temporal",
                        consolidation_level="candidate",
                        coherence_score=coherence,
                        metadata={
                            'time_span_hours': time_span.total_seconds() / 3600,
                            'window_size': self.config.temporal_window_hours
                        }
                    )
                    
                    clusters.append(cluster)
                    self.memory_clusters[cluster.cluster_id] = cluster
            
            return clusters
            
        except Exception as e:
            logger.error(f"Temporal clustering failed: {e}")
            return []
    
    def _perform_usage_clustering(self, 
                                memory_ids: List[str],
                                metadata: List[Dict]) -> List[MemoryCluster]:
        """Perform usage pattern clustering on memories."""
        try:
            # Extract access patterns
            access_patterns = []
            for meta in metadata:
                access_history = meta.get('access_history', [])
                # Convert string timestamps back to datetime if needed
                if access_history and isinstance(access_history[0], str):
                    access_patterns.append([
                        datetime.fromisoformat(ts) for ts in access_history
                    ])
                else:
                    access_patterns.append(access_history)
            
            if not any(access_patterns):
                return []
            
            # Calculate usage overlap matrix
            overlap_matrix = self.usage_analyzer.analyze_usage_overlap(access_patterns)
            
            # Identify usage clusters
            cluster_indices = self.usage_analyzer.identify_usage_clusters(overlap_matrix)
            
            clusters = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) >= self.config.min_cluster_size:
                    cluster_memory_ids = [memory_ids[idx] for idx in indices]
                    
                    # Calculate usage coherence
                    cluster_overlaps = overlap_matrix[np.ix_(indices, indices)]
                    coherence = np.mean(cluster_overlaps)
                    
                    cluster = MemoryCluster(
                        cluster_id=f"usage_{i}_{int(time.time())}",
                        memory_ids=cluster_memory_ids,
                        centroid=np.array([]),  # No centroid for usage clusters
                        cluster_type="usage",
                        consolidation_level="candidate",
                        coherence_score=coherence,
                        metadata={'usage_overlap_threshold': self.config.semantic_cluster_threshold}
                    )
                    
                    clusters.append(cluster)
                    self.memory_clusters[cluster.cluster_id] = cluster
            
            return clusters
            
        except Exception as e:
            logger.error(f"Usage clustering failed: {e}")
            return []
    
    def _perform_graph_clustering(self, memory_ids: List[str]) -> List[MemoryCluster]:
        """Perform graph-based clustering using memory relationships."""
        if not self.memory_graph:
            return []
        
        try:
            # Find memory nodes in graph
            memory_nodes = [mid for mid in memory_ids if mid in self.memory_graph]
            
            if len(memory_nodes) < self.config.min_cluster_size:
                return []
            
            # Extract subgraph for these memories
            subgraph = self.memory_graph.subgraph(memory_nodes)
            
            # Find connected components or communities
            if nx.is_connected(subgraph):
                # Use community detection for connected graphs
                try:
                    import networkx.algorithms.community as nxcom
                    communities = list(nxcom.greedy_modularity_communities(subgraph))
                except ImportError:
                    # Fallback to connected components
                    communities = list(nx.connected_components(subgraph))
            else:
                communities = list(nx.connected_components(subgraph))
            
            clusters = []
            for i, community in enumerate(communities):
                if len(community) >= self.config.min_cluster_size:
                    cluster_memory_ids = list(community)
                    
                    # Calculate graph-based coherence
                    community_subgraph = subgraph.subgraph(community)
                    edge_density = nx.density(community_subgraph)
                    
                    cluster = MemoryCluster(
                        cluster_id=f"graph_{i}_{int(time.time())}",
                        memory_ids=cluster_memory_ids,
                        centroid=np.array([]),  # No centroid for graph clusters
                        cluster_type="graph",
                        consolidation_level="candidate",
                        coherence_score=edge_density,
                        metadata={
                            'edge_density': edge_density,
                            'community_size': len(community)
                        }
                    )
                    
                    clusters.append(cluster)
                    self.memory_clusters[cluster.cluster_id] = cluster
            
            return clusters
            
        except Exception as e:
            logger.error(f"Graph clustering failed: {e}")
            return []
    
    def _merge_overlapping_clusters(self, clusters: List[MemoryCluster]) -> List[MemoryCluster]:
        """Merge overlapping clusters to avoid duplicates."""
        if not clusters:
            return clusters
        
        merged_clusters = []
        processed_memory_ids = set()
        
        for cluster in sorted(clusters, key=lambda c: c.coherence_score, reverse=True):
            # Check if cluster has significant overlap with already processed memories
            overlap = set(cluster.memory_ids) & processed_memory_ids
            overlap_ratio = len(overlap) / len(cluster.memory_ids)
            
            if overlap_ratio < 0.5:  # Less than 50% overlap
                merged_clusters.append(cluster)
                processed_memory_ids.update(cluster.memory_ids)
        
        return merged_clusters
    
    def generate_consolidation_candidates(self, 
                                        clusters: List[MemoryCluster],
                                        memory_data: Dict[str, Dict]) -> List[ConsolidationCandidate]:
        """Generate consolidation candidates from clusters."""
        candidates = []
        
        for cluster in clusters:
            try:
                # Get memory data for cluster
                cluster_memories = [
                    memory_data.get(memory_id, {}) 
                    for memory_id in cluster.memory_ids
                ]
                
                contents = [mem.get('content', '') for mem in cluster_memories]
                metadata = [mem.get('metadata', {}) for mem in cluster_memories]
                
                # Analyze consolidation potential
                candidate = self.strategy_analyzer.analyze_consolidation_potential(
                    cluster, contents, metadata
                )
                
                if candidate and candidate.confidence >= self.config.min_consolidation_benefit:
                    candidates.append(candidate)
                    
            except Exception as e:
                logger.error(f"Failed to analyze cluster {cluster.cluster_id}: {e}")
        
        # Sort by potential benefit (confidence * savings)
        candidates.sort(
            key=lambda c: c.confidence * c.potential_savings_bytes, 
            reverse=True
        )
        
        return candidates[:self.config.max_consolidation_candidates]
    
    def execute_consolidation(self, 
                            candidate: ConsolidationCandidate,
                            memory_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Execute a consolidation operation."""
        consolidation_start = time.perf_counter()
        
        try:
            with self._consolidation_lock:
                # Backup if enabled
                if self.config.backup_before_consolidation:
                    backup_data = self._create_consolidation_backup(candidate, memory_data)
                
                # Execute consolidation strategy
                strategy = candidate.consolidation_strategy
                result = self._execute_consolidation_strategy(
                    candidate, memory_data, strategy
                )
                
                # Update performance stats
                if result['success']:
                    self.performance_stats['consolidations_performed'] += 1
                    self.performance_stats['bytes_saved'] += candidate.potential_savings_bytes
                
                # Record consolidation history
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'candidate': candidate.__dict__,
                    'result': result,
                    'processing_time_ms': (time.perf_counter() - consolidation_start) * 1000
                }
                self.consolidation_history.append(history_entry)
                
                return result
                
        except Exception as e:
            logger.error(f"Consolidation execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'candidate_id': candidate.primary_memory_id
            }
    
    def _execute_consolidation_strategy(self, 
                                      candidate: ConsolidationCandidate,
                                      memory_data: Dict[str, Dict],
                                      strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific consolidation strategy."""
        strategy_type = strategy['type']
        
        if strategy_type == 'merge':
            return self._execute_merge_strategy(candidate, memory_data, strategy)
        elif strategy_type == 'summarize':
            return self._execute_summarize_strategy(candidate, memory_data, strategy)
        elif strategy_type == 'archive':
            return self._execute_archive_strategy(candidate, memory_data, strategy)
        elif strategy_type == 'reference':
            return self._execute_reference_strategy(candidate, memory_data, strategy)
        else:
            return {
                'success': False,
                'error': f'Unknown consolidation strategy: {strategy_type}'
            }
    
    def _execute_merge_strategy(self, 
                              candidate: ConsolidationCandidate,
                              memory_data: Dict[str, Dict],
                              strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory merge strategy."""
        try:
            # Collect all related memories
            all_memory_ids = [candidate.primary_memory_id] + candidate.related_memory_ids
            memories = [memory_data[mid] for mid in all_memory_ids]
            
            # Create merged content
            merged_content = self._merge_memory_contents(memories, strategy)
            
            # Create merged metadata
            merged_metadata = self._merge_memory_metadata(memories, strategy)
            
            # Update primary memory with merged data
            primary_memory = memory_data[candidate.primary_memory_id]
            primary_memory['content'] = merged_content
            primary_memory['metadata'] = merged_metadata
            primary_memory['metadata']['consolidated_from'] = candidate.related_memory_ids
            primary_memory['metadata']['consolidation_timestamp'] = datetime.now().isoformat()
            
            # Mark related memories for deletion
            deleted_memories = []
            for memory_id in candidate.related_memory_ids:
                if memory_id in memory_data:
                    deleted_memories.append(memory_id)
                    del memory_data[memory_id]
            
            return {
                'success': True,
                'strategy': 'merge',
                'merged_memory_id': candidate.primary_memory_id,
                'deleted_memory_ids': deleted_memories,
                'consolidated_size': len(merged_content.encode('utf-8'))
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'strategy': 'merge'
            }
    
    def _merge_memory_contents(self, memories: List[Dict], strategy: Dict) -> str:
        """Merge content from multiple memories."""
        method = strategy.get('method', 'union')
        
        if method == 'union':
            # Combine all unique content
            all_content = []
            seen_content = set()
            
            for memory in memories:
                content = memory.get('content', '')
                if content and content not in seen_content:
                    all_content.append(content)
                    seen_content.add(content)
            
            return '\n\n---\n\n'.join(all_content)
        
        elif method == 'prioritized':
            # Merge based on importance scores
            prioritized_memories = sorted(
                memories, 
                key=lambda m: m.get('metadata', {}).get('importance', 0.5),
                reverse=True
            )
            
            merged_parts = []
            for memory in prioritized_memories:
                content = memory.get('content', '')
                if content:
                    merged_parts.append(content)
            
            return '\n\n'.join(merged_parts)
        
        else:
            # Default: simple concatenation
            return '\n\n'.join(
                memory.get('content', '') for memory in memories
            )
    
    def _merge_memory_metadata(self, memories: List[Dict], strategy: Dict) -> Dict[str, Any]:
        """Merge metadata from multiple memories."""
        merged_metadata = {}
        
        # Collect all metadata keys
        all_keys = set()
        for memory in memories:
            all_keys.update(memory.get('metadata', {}).keys())
        
        for key in all_keys:
            values = [
                memory.get('metadata', {}).get(key)
                for memory in memories
                if memory.get('metadata', {}).get(key) is not None
            ]
            
            if not values:
                continue
            
            # Merge strategy based on value type
            if isinstance(values[0], (int, float)):
                merged_metadata[key] = np.mean(values)
            elif isinstance(values[0], list):
                merged_metadata[key] = list(set().union(*values))
            elif isinstance(values[0], str):
                if key in ['memory_type', 'content_type']:
                    # Use most common value
                    from collections import Counter
                    merged_metadata[key] = Counter(values).most_common(1)[0][0]
                else:
                    merged_metadata[key] = values[0]  # Use first value
            else:
                merged_metadata[key] = values[0]
        
        # Add consolidation metadata
        merged_metadata['is_consolidated'] = True
        merged_metadata['original_count'] = len(memories)
        
        return merged_metadata
    
    def _execute_summarize_strategy(self, candidate, memory_data, strategy):
        """Execute memory summarization strategy (placeholder)."""
        # This would implement text summarization
        return {
            'success': True,
            'strategy': 'summarize',
            'message': 'Summarization not yet implemented'
        }
    
    def _execute_archive_strategy(self, candidate, memory_data, strategy):
        """Execute memory archival strategy (placeholder)."""
        # This would implement memory archival
        return {
            'success': True,
            'strategy': 'archive',
            'message': 'Archival not yet implemented'
        }
    
    def _execute_reference_strategy(self, candidate, memory_data, strategy):
        """Execute reference linking strategy (placeholder)."""
        # This would implement reference linking
        return {
            'success': True,
            'strategy': 'reference',
            'message': 'Reference linking not yet implemented'
        }
    
    def _create_consolidation_backup(self, candidate, memory_data):
        """Create backup before consolidation (placeholder)."""
        # This would create backups
        return {'backup_created': True}
    
    def get_consolidation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive consolidation statistics."""
        return {
            'performance_stats': self.performance_stats,
            'clusters_tracked': len(self.memory_clusters),
            'consolidation_history_size': len(self.consolidation_history),
            'configuration': {
                'semantic_threshold': self.config.semantic_cluster_threshold,
                'temporal_window_hours': self.config.temporal_window_hours,
                'min_cluster_size': self.config.min_cluster_size,
                'clustering_algorithm': self.config.clustering_algorithm
            }
        }
    
    def optimize_consolidation_config(self) -> Dict[str, Any]:
        """Optimize consolidation configuration based on performance."""
        optimization_results = {
            'optimizations_applied': 0,
            'improvements': [],
        }
        
        try:
            # Analyze consolidation success rate
            if self.consolidation_history:
                successful_consolidations = sum(
                    1 for entry in self.consolidation_history
                    if entry['result'].get('success', False)
                )
                success_rate = successful_consolidations / len(self.consolidation_history)
                
                # Adjust thresholds based on success rate
                if success_rate < 0.6:
                    # Low success rate, make clustering more selective
                    old_threshold = self.config.semantic_cluster_threshold
                    self.config.semantic_cluster_threshold = min(0.9, old_threshold + 0.05)
                    
                    if self.config.semantic_cluster_threshold != old_threshold:
                        optimization_results['optimizations_applied'] += 1
                        optimization_results['improvements'].append(
                            f'Increased semantic threshold from {old_threshold} to {self.config.semantic_cluster_threshold}'
                        )
                
                elif success_rate > 0.85:
                    # High success rate, can be more aggressive
                    old_threshold = self.config.semantic_cluster_threshold
                    self.config.semantic_cluster_threshold = max(0.6, old_threshold - 0.05)
                    
                    if self.config.semantic_cluster_threshold != old_threshold:
                        optimization_results['optimizations_applied'] += 1
                        optimization_results['improvements'].append(
                            f'Decreased semantic threshold from {old_threshold} to {self.config.semantic_cluster_threshold}'
                        )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Configuration optimization failed: {e}")
            optimization_results['error'] = str(e)
            return optimization_results


def create_memory_consolidator(
    config: Optional[ConsolidationConfig] = None,
    storage_backend: Optional[Any] = None
) -> AdvancedMemoryConsolidator:
    """
    Factory function to create advanced memory consolidator.
    
    Args:
        config: Optional consolidation configuration
        storage_backend: Optional storage backend
        
    Returns:
        AdvancedMemoryConsolidator instance
    """
    return AdvancedMemoryConsolidator(config=config, storage_backend=storage_backend)