#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Semantic Similarity - Advanced duplicate detection using embeddings
"""

import asyncio
import logging
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from .storage.amms_storage import Memory


class SemanticSimilarityDetector:
    """
    Advanced semantic similarity detection using sentence embeddings
    
    Uses SentenceTransformer models to compute semantic similarity
    between memory contents for better duplicate detection
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.logger = logging.getLogger(__name__)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Initialized SentenceTransformer: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize SentenceTransformer: {e}")
            self.model = None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts
        
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if not self.model:
            # Fallback to simple word overlap
            return self._word_overlap_similarity(text1, text2)
        
        try:
            # Get embeddings for both texts
            embeddings = self.model.encode([text1, text2])
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(embeddings[0], embeddings[1])
            
            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            normalized_similarity = (similarity + 1) / 2
            
            return float(normalized_similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing semantic similarity: {e}")
            return self._word_overlap_similarity(text1, text2)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def find_similar_memories(
        self, 
        content: str, 
        memories: List[Memory], 
        threshold: float = 0.75,
        max_results: int = 5
    ) -> List[Tuple[Memory, float]]:
        """
        Find memories similar to the given content
        
        Args:
            content: Content to compare against
            memories: List of memories to search through
            threshold: Minimum similarity threshold
            max_results: Maximum number of similar memories to return
            
        Returns:
            List of (Memory, similarity_score) tuples, sorted by similarity
        """
        similar_memories = []
        
        for memory in memories:
            try:
                similarity = self.compute_similarity(content, memory.content)
                
                if similarity >= threshold:
                    similar_memories.append((memory, similarity))
                    
            except Exception as e:
                self.logger.warning(f"Error comparing memory {memory.id}: {e}")
                continue
        
        # Sort by similarity (highest first) and limit results
        similar_memories.sort(key=lambda x: x[1], reverse=True)
        return similar_memories[:max_results]
    
    def is_likely_duplicate(self, content: str, existing_content: str, threshold: float = 0.85) -> bool:
        """
        Check if content is likely a duplicate of existing content
        
        Args:
            content: New content to check
            existing_content: Existing memory content
            threshold: Similarity threshold for duplicate detection
            
        Returns:
            bool: True if likely duplicate
        """
        similarity = self.compute_similarity(content, existing_content)
        return similarity >= threshold
    
    async def batch_similarity_check(
        self, 
        content: str, 
        memories: List[Memory], 
        duplicate_threshold: float = 0.85,
        similar_threshold: float = 0.75
    ) -> dict:
        """
        Perform batch similarity analysis
        
        Returns:
            dict: Analysis results with duplicates and similar memories
        """
        results = {
            "duplicates": [],
            "similar": [],
            "max_similarity": 0.0,
            "analysis_summary": ""
        }
        
        for memory in memories:
            try:
                similarity = self.compute_similarity(content, memory.content)
                results["max_similarity"] = max(results["max_similarity"], similarity)
                
                if similarity >= duplicate_threshold:
                    results["duplicates"].append((memory, similarity))
                elif similarity >= similar_threshold:
                    results["similar"].append((memory, similarity))
                    
            except Exception as e:
                self.logger.warning(f"Error in batch similarity check for memory {memory.id}: {e}")
                continue
        
        # Sort results by similarity
        results["duplicates"].sort(key=lambda x: x[1], reverse=True)
        results["similar"].sort(key=lambda x: x[1], reverse=True)
        
        # Generate analysis summary
        num_duplicates = len(results["duplicates"])
        num_similar = len(results["similar"])
        
        if num_duplicates > 0:
            results["analysis_summary"] = f"Found {num_duplicates} likely duplicate(s)"
        elif num_similar > 0:
            results["analysis_summary"] = f"Found {num_similar} similar memorie(s)"
        else:
            results["analysis_summary"] = "No similar content found"
        
        return results


# Global instance for reuse
_semantic_detector: Optional[SemanticSimilarityDetector] = None

def get_semantic_detector() -> SemanticSimilarityDetector:
    """Get or create global semantic similarity detector instance"""
    global _semantic_detector
    if _semantic_detector is None:
        _semantic_detector = SemanticSimilarityDetector()
    return _semantic_detector