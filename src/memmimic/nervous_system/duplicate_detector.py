"""
SemanticDuplicateDetector - Vector-based Duplicate Detection

Implements semantic similarity detection using sentence transformers and cosine similarity
with intelligent duplicate resolution and relationship mapping.

Performance Target: >99% accuracy with <2ms detection time
Similarity Thresholds: 0.85+ duplicates, 0.7-0.84 related, <0.7 unique
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from statistics import mean
import logging

from .interfaces import DuplicateDetectorInterface, DuplicateAnalysis
from ..memory.storage.amms_storage import Memory
from ..errors import get_error_logger

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

@dataclass 
class SimilarityThresholds:
    """Similarity thresholds for duplicate detection"""
    duplicate_threshold: float = 0.85  # 85%+ similarity = duplicate
    related_threshold: float = 0.70    # 70-84% similarity = related
    unique_threshold: float = 0.70     # <70% similarity = unique

class SemanticDuplicateDetector(DuplicateDetectorInterface):
    """
    Semantic duplicate detection using sentence transformers and vector similarity.
    
    Provides real-time duplicate detection with intelligent resolution strategies
    and bidirectional relationship mapping for memory connections.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 thresholds: Optional[SimilarityThresholds] = None,
                 cache_size: int = 1000):
        self.model_name = model_name
        self.thresholds = thresholds or SimilarityThresholds()
        self.cache_size = cache_size
        self.logger = get_error_logger("duplicate_detector")
        
        # Model and embedding components
        self._model = None
        self._embedding_cache = {}
        self._similarity_cache = {}
        
        # Memory storage for comparison (would integrate with AMMS in production)
        self._memory_embeddings = {}
        self._memory_index = {}
        
        # Performance metrics
        self._detection_count = 0
        self._duplicate_found_count = 0
        self._relationship_mapped_count = 0
        self._processing_times = []
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize semantic similarity model and embedding cache"""
        if self._initialized:
            return
        
        start_time = time.perf_counter()
        
        try:
            if EMBEDDINGS_AVAILABLE:
                # Load sentence transformer model
                self._model = SentenceTransformer(self.model_name)
                self.logger.info(f"Loaded sentence transformer model: {self.model_name}")
            else:
                self.logger.warning("Sentence transformers not available, using fallback similarity")
                self._model = None
            
            # Initialize embedding cache and similarity structures
            await self._initialize_memory_index()
            
            init_time = (time.perf_counter() - start_time) * 1000
            self._initialized = True
            
            self.logger.info(
                f"SemanticDuplicateDetector initialized in {init_time:.2f}ms",
                extra={"initialization_time_ms": init_time, "embeddings_available": EMBEDDINGS_AVAILABLE}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SemanticDuplicateDetector: {e}")
            # Continue with fallback mode
            self._model = None
            self._initialized = True
    
    async def _initialize_memory_index(self) -> None:
        """Initialize memory index for similarity comparisons"""
        # In production, this would load existing memory embeddings from AMMS storage
        # For now, simulate initialization
        await asyncio.sleep(0.001)  # 1ms simulation
        
        # Would populate from AMMS:
        # memories = await self.amms_storage.get_all_memories()
        # for memory in memories:
        #     embedding = await self._get_embedding(memory.content)
        #     self._memory_embeddings[memory.id] = embedding
        #     self._memory_index[memory.id] = memory
    
    async def detect_duplicates(self, content: str, memory_type: str) -> DuplicateAnalysis:
        """
        Detect semantic duplicates with <2ms performance target.
        
        Uses vector similarity comparison against existing memories
        with intelligent resolution strategy determination.
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            # Generate embedding for new content
            content_embedding = await self._get_embedding_cached(content)
            
            # Find most similar existing memories
            similar_memories = await self._find_similar_memories(content_embedding, content)
            
            # Analyze similarity and determine resolution strategy
            analysis = await self._analyze_similarity(content, similar_memories, memory_type)
            
            # Performance tracking
            processing_time = (time.perf_counter() - start_time) * 1000
            self._processing_times.append(processing_time)
            self._detection_count += 1
            
            if analysis.is_duplicate:
                self._duplicate_found_count += 1
            
            self.logger.debug(
                f"Duplicate detection completed: {analysis.similarity_score:.3f} "
                f"({'duplicate' if analysis.is_duplicate else 'unique'})",
                extra={
                    "processing_time_ms": processing_time,
                    "similarity_score": analysis.similarity_score,
                    "is_duplicate": analysis.is_duplicate
                }
            )
            
            return analysis
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(
                f"Duplicate detection failed: {e}",
                extra={"processing_time_ms": processing_time}
            )
            
            # Return safe fallback analysis
            return DuplicateAnalysis(
                is_duplicate=False,
                similarity_score=0.0,
                duplicate_memory_id=None,
                resolution_action='store',
                relationship_strength=0.0,
                preservation_metadata={'error': str(e), 'fallback_mode': True}
            )
    
    async def _get_embedding_cached(self, content: str) -> np.ndarray:
        """Get embedding with LRU caching for performance"""
        content_hash = str(hash(content))
        
        if content_hash in self._embedding_cache:
            return self._embedding_cache[content_hash]
        
        embedding = await self._get_embedding(content)
        
        # Maintain cache size limit
        if len(self._embedding_cache) >= self.cache_size:
            # Remove oldest entry (simplified LRU)
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[content_hash] = embedding
        return embedding
    
    async def _get_embedding(self, content: str) -> np.ndarray:
        """Generate semantic embedding for content"""
        if self._model and EMBEDDINGS_AVAILABLE:
            # Use sentence transformer model
            embedding = self._model.encode([content], convert_to_numpy=True)[0]
            return embedding
        else:
            # Fallback: simple text-based features
            return await self._generate_fallback_embedding(content)
    
    async def _generate_fallback_embedding(self, content: str) -> np.ndarray:
        """Generate fallback embedding using text features when ML models unavailable"""
        # Simple text-based features for similarity
        features = []
        
        # Length features
        features.append(len(content))
        features.append(len(content.split()))
        
        # Character-based features
        features.append(content.count(' '))
        features.append(content.count('.'))
        features.append(content.count(','))
        
        # Word-based features (top 50 common words as dimensions)
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'must', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'my', 'your', 'his', 'her', 'its', 'our', 'their'
        ]
        
        content_lower = content.lower()
        for word in common_words:
            features.append(content_lower.count(word))
        
        # Normalize features to 0-1 range
        features = np.array(features, dtype=float)
        if features.max() > 0:
            features = features / features.max()
        
        return features
    
    async def _find_similar_memories(self, content_embedding: np.ndarray, content: str) -> List[Tuple[str, float, Memory]]:
        """Find memories with highest similarity to content"""
        similarities = []
        
        # Compare against all existing memory embeddings
        for memory_id, memory_embedding in self._memory_embeddings.items():
            if memory_id in self._memory_index:
                similarity = await self._calculate_similarity(content_embedding, memory_embedding)
                memory = self._memory_index[memory_id]
                similarities.append((memory_id, similarity, memory))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 5 most similar
        return similarities[:5]
    
    async def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            if EMBEDDINGS_AVAILABLE:
                # Use sklearn cosine similarity
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                return float(similarity)
            else:
                # Fallback: normalized dot product
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return float(dot_product / (norm1 * norm2))
        except Exception as e:
            self.logger.debug(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def _analyze_similarity(self, content: str, similar_memories: List[Tuple[str, float, Memory]], memory_type: str) -> DuplicateAnalysis:
        """Analyze similarity results and determine resolution strategy"""
        if not similar_memories:
            return DuplicateAnalysis(
                is_duplicate=False,
                similarity_score=0.0,
                duplicate_memory_id=None,
                resolution_action='store',
                relationship_strength=0.0,
                preservation_metadata={}
            )
        
        # Get highest similarity
        best_match_id, best_similarity, best_match_memory = similar_memories[0]
        
        # Determine if duplicate based on threshold
        is_duplicate = best_similarity >= self.thresholds.duplicate_threshold
        
        # Determine resolution action
        if is_duplicate:
            # High similarity - merge or enhance existing
            if best_similarity >= 0.95:
                resolution_action = 'reference'  # Nearly identical
            else:
                resolution_action = 'enhance_existing'  # Similar but with new info
        elif best_similarity >= self.thresholds.related_threshold:
            resolution_action = 'store_as_related'  # Related but unique enough
        else:
            resolution_action = 'store'  # Unique content
        
        # Calculate relationship strength for related memories
        relationship_strength = max(0.0, (best_similarity - 0.5) / 0.5) if best_similarity >= 0.5 else 0.0
        
        # Preservation metadata for intelligent handling
        preservation_metadata = {
            'similar_memories': [
                {
                    'memory_id': mem_id,
                    'similarity_score': sim_score,
                    'content_preview': mem.content[:100] + '...' if len(mem.content) > 100 else mem.content
                }
                for mem_id, sim_score, mem in similar_memories[:3]
            ],
            'analysis_timestamp': time.time(),
            'detection_method': 'semantic_embedding' if self._model else 'fallback_features'
        }
        
        return DuplicateAnalysis(
            is_duplicate=is_duplicate,
            similarity_score=best_similarity,
            duplicate_memory_id=best_match_id if is_duplicate else None,
            resolution_action=resolution_action,
            relationship_strength=relationship_strength,
            preservation_metadata=preservation_metadata
        )
    
    async def resolve_duplicate(self, analysis: DuplicateAnalysis, new_content: str) -> Memory:
        """
        Resolve duplicate with intelligent merge or reference strategy.
        Returns the resolved memory (existing enhanced or new memory).
        """
        if not analysis.is_duplicate:
            # Not a duplicate, create new memory
            return Memory(content=new_content)
        
        resolution_action = analysis.resolution_action
        
        if resolution_action == 'reference':
            # Nearly identical - just reference existing
            existing_memory = self._memory_index.get(analysis.duplicate_memory_id)
            if existing_memory:
                # Add reference metadata
                if not hasattr(existing_memory, 'metadata'):
                    existing_memory.metadata = {}
                
                if 'references' not in existing_memory.metadata:
                    existing_memory.metadata['references'] = []
                
                existing_memory.metadata['references'].append({
                    'content': new_content,
                    'timestamp': time.time(),
                    'similarity_score': analysis.similarity_score
                })
                
                return existing_memory
        
        elif resolution_action == 'enhance_existing':
            # Similar with new information - enhance existing
            existing_memory = self._memory_index.get(analysis.duplicate_memory_id)
            if existing_memory:
                # Intelligent content enhancement
                enhanced_content = await self._enhance_memory_content(
                    existing_memory.content, 
                    new_content,
                    analysis.similarity_score
                )
                
                existing_memory.content = enhanced_content
                
                # Update metadata
                if not hasattr(existing_memory, 'metadata'):
                    existing_memory.metadata = {}
                
                existing_memory.metadata['enhanced'] = True
                existing_memory.metadata['enhancement_history'] = existing_memory.metadata.get('enhancement_history', [])
                existing_memory.metadata['enhancement_history'].append({
                    'original_addition': new_content,
                    'timestamp': time.time(),
                    'similarity_score': analysis.similarity_score
                })
                
                return existing_memory
        
        # Fallback: create new memory with relationship metadata
        new_memory = Memory(content=new_content)
        new_memory.metadata = {
            'related_memories': [analysis.duplicate_memory_id],
            'relationship_strength': analysis.relationship_strength,
            'resolution_action': resolution_action
        }
        
        return new_memory
    
    async def _enhance_memory_content(self, existing_content: str, new_content: str, similarity_score: float) -> str:
        """Intelligently enhance existing memory with new content"""
        # Simple enhancement strategy - in production would be more sophisticated
        if similarity_score >= 0.90:
            # Very similar - add as addendum
            return f"{existing_content}\n\n[Enhanced with additional context:] {new_content}"
        else:
            # Moderately similar - merge relevant parts
            return f"{existing_content}\n\n[Related insight:] {new_content}"
    
    async def map_relationships(self, memory: Memory, similar_memories: List[Memory]) -> Dict[str, float]:
        """Map bidirectional relationships with strength scoring"""
        relationships = {}
        
        if not hasattr(memory, 'content') or not memory.content:
            return relationships
        
        memory_embedding = await self._get_embedding_cached(memory.content)
        
        for similar_memory in similar_memories:
            if hasattr(similar_memory, 'id') and hasattr(similar_memory, 'content'):
                similar_embedding = await self._get_embedding_cached(similar_memory.content)
                
                # Calculate bidirectional relationship strength
                similarity = await self._calculate_similarity(memory_embedding, similar_embedding)
                
                if similarity >= self.thresholds.related_threshold:
                    relationships[similar_memory.id] = similarity
        
        self._relationship_mapped_count += len(relationships)
        return relationships
    
    async def process_async(self, data: Any) -> Any:
        """Process data asynchronously for interface compliance"""
        if isinstance(data, tuple) and len(data) == 2:
            content, memory_type = data
            return await self.detect_duplicates(content, memory_type)
        return await self.detect_duplicates(str(data), "interaction")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for monitoring"""
        avg_processing_time = mean(self._processing_times) if self._processing_times else 0.0
        duplicate_rate = self._duplicate_found_count / max(1, self._detection_count)
        
        return {
            'total_detections': self._detection_count,
            'duplicates_found': self._duplicate_found_count,
            'relationships_mapped': self._relationship_mapped_count,
            'duplicate_detection_rate': duplicate_rate,
            'average_processing_time_ms': avg_processing_time,
            'target_processing_time_ms': 2.0,
            'performance_target_met': avg_processing_time < 2.0,
            'accuracy_target': 0.99,
            'embeddings_available': EMBEDDINGS_AVAILABLE,
            'cache_size': len(self._embedding_cache),
            'memory_index_size': len(self._memory_index),
            'thresholds': {
                'duplicate_threshold': self.thresholds.duplicate_threshold,
                'related_threshold': self.thresholds.related_threshold,
                'unique_threshold': self.thresholds.unique_threshold
            }
        }