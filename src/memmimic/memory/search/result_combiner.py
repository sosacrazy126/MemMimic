#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result Combiner Module

Handles combination and scoring of results from different search methods.
Extracted from memmimic_recall_cxd.py for improved maintainability.
"""

import logging
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class ResultCombiner:
    """
    Combines and scores results from multiple search methods.
    
    Provides intelligent result fusion with configurable scoring strategies
    and convergence bonuses for results found by multiple methods.
    """
    
    def __init__(self):
        """Initialize result combiner."""
        self.combination_strategies = {
            "weighted_sum": self._weighted_sum_strategy,
            "max_score": self._max_score_strategy,
            "harmonic_mean": self._harmonic_mean_strategy,
            "geometric_mean": self._geometric_mean_strategy
        }
    
    def combine_and_score(
        self,
        semantic_results: List[Dict[str, Any]],
        wordnet_results: List[Dict[str, Any]],
        semantic_weight: float = 0.7,
        wordnet_weight: float = 0.3,
        convergence_bonus: float = 0.1,
        combination_strategy: str = "weighted_sum"
    ) -> List[Dict[str, Any]]:
        """
        Combine results from semantic and WordNet search methods.
        
        Args:
            semantic_results: Results from semantic search
            wordnet_results: Results from WordNet search
            semantic_weight: Weight for semantic scores (0.0-1.0)
            wordnet_weight: Weight for WordNet scores (0.0-1.0)
            convergence_bonus: Bonus for results found by both methods
            combination_strategy: Strategy for combining scores
            
        Returns:
            List of combined and ranked results
        """
        try:
            # Normalize weights
            total_weight = semantic_weight + wordnet_weight
            if total_weight > 0:
                semantic_weight = semantic_weight / total_weight
                wordnet_weight = wordnet_weight / total_weight
            
            # Group results by content similarity
            content_groups = self._group_results_by_content(
                semantic_results, wordnet_results
            )
            
            # Combine scores for each content group
            combined_results = []
            strategy_func = self.combination_strategies.get(
                combination_strategy, self._weighted_sum_strategy
            )
            
            for content_hash, group_data in content_groups.items():
                combined_result = strategy_func(
                    group_data,
                    semantic_weight,
                    wordnet_weight,
                    convergence_bonus
                )
                
                if combined_result:
                    combined_results.append(combined_result)
            
            # Sort by combined score (highest first)
            combined_results.sort(
                key=lambda x: x.get("combined_score", 0.0), reverse=True
            )
            
            logger.debug(f"Combined {len(combined_results)} unique results")
            return combined_results
            
        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            # Return semantic results as fallback
            return semantic_results
    
    def _group_results_by_content(
        self,
        semantic_results: List[Dict],
        wordnet_results: List[Dict]
    ) -> Dict[str, Dict]:
        """Group results by content similarity to identify overlaps."""
        content_groups = defaultdict(lambda: {
            "semantic": None,
            "wordnet": None,
            "content": "",
            "all_results": []
        })
        
        # Process semantic results
        for result in semantic_results:
            content = result.get("content", "")
            content_hash = self._get_content_hash(content)
            
            content_groups[content_hash]["semantic"] = result
            content_groups[content_hash]["content"] = content
            content_groups[content_hash]["all_results"].append(("semantic", result))
        
        # Process WordNet results
        for result in wordnet_results:
            content = result.get("content", "")
            content_hash = self._get_content_hash(content)
            
            # Check if this content already exists (from semantic search)
            if content_hash in content_groups:
                content_groups[content_hash]["wordnet"] = result
            else:
                # New content from WordNet only
                content_groups[content_hash]["wordnet"] = result
                content_groups[content_hash]["content"] = content
            
            content_groups[content_hash]["all_results"].append(("wordnet", result))
        
        return dict(content_groups)
    
    def _get_content_hash(self, content: str) -> str:
        """Generate a hash for content to identify duplicates."""
        # Normalize content for better duplicate detection
        normalized = content.lower().strip()
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        # Create hash
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _weighted_sum_strategy(
        self,
        group_data: Dict,
        semantic_weight: float,
        wordnet_weight: float,
        convergence_bonus: float
    ) -> Optional[Dict[str, Any]]:
        """Weighted sum combination strategy."""
        try:
            semantic_result = group_data["semantic"]
            wordnet_result = group_data["wordnet"]
            
            # Get scores
            semantic_score = semantic_result.get("semantic_score", 0.0) if semantic_result else 0.0
            wordnet_score = wordnet_result.get("wordnet_score", 0.0) if wordnet_result else 0.0
            
            # Calculate weighted combination
            combined_score = (
                semantic_score * semantic_weight +
                wordnet_score * wordnet_weight
            )
            
            # Apply convergence bonus if found by both methods
            convergence = semantic_result is not None and wordnet_result is not None
            if convergence:
                combined_score += convergence_bonus
            
            # Create combined result
            base_result = semantic_result or wordnet_result
            if not base_result:
                return None
            
            combined_result = base_result.copy()
            combined_result.update({
                "combined_score": float(combined_score),
                "semantic_score": float(semantic_score),
                "wordnet_score": float(wordnet_score),
                "convergence": convergence,
                "search_method": self._determine_search_method(
                    semantic_result, wordnet_result
                ),
                "combination_strategy": "weighted_sum"
            })
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Weighted sum strategy failed: {e}")
            return None
    
    def _max_score_strategy(
        self,
        group_data: Dict,
        semantic_weight: float,
        wordnet_weight: float,
        convergence_bonus: float
    ) -> Optional[Dict[str, Any]]:
        """Maximum score combination strategy."""
        try:
            semantic_result = group_data["semantic"]
            wordnet_result = group_data["wordnet"]
            
            semantic_score = semantic_result.get("semantic_score", 0.0) if semantic_result else 0.0
            wordnet_score = wordnet_result.get("wordnet_score", 0.0) if wordnet_result else 0.0
            
            # Take maximum score
            combined_score = max(semantic_score, wordnet_score)
            
            # Apply convergence bonus
            convergence = semantic_result is not None and wordnet_result is not None
            if convergence:
                combined_score += convergence_bonus
            
            # Use result with higher score as base
            base_result = (
                semantic_result if semantic_score >= wordnet_score
                else wordnet_result
            )
            
            if not base_result:
                return None
            
            combined_result = base_result.copy()
            combined_result.update({
                "combined_score": float(combined_score),
                "semantic_score": float(semantic_score),
                "wordnet_score": float(wordnet_score),
                "convergence": convergence,
                "search_method": self._determine_search_method(
                    semantic_result, wordnet_result
                ),
                "combination_strategy": "max_score"
            })
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Max score strategy failed: {e}")
            return None
    
    def _harmonic_mean_strategy(
        self,
        group_data: Dict,
        semantic_weight: float,
        wordnet_weight: float,
        convergence_bonus: float
    ) -> Optional[Dict[str, Any]]:
        """Harmonic mean combination strategy."""
        try:
            semantic_result = group_data["semantic"]
            wordnet_result = group_data["wordnet"]
            
            semantic_score = semantic_result.get("semantic_score", 0.0) if semantic_result else 0.0
            wordnet_score = wordnet_result.get("wordnet_score", 0.0) if wordnet_result else 0.0
            
            # Calculate harmonic mean (only if both scores > 0)
            if semantic_score > 0 and wordnet_score > 0:
                combined_score = 2 * semantic_score * wordnet_score / (semantic_score + wordnet_score)
            else:
                combined_score = max(semantic_score, wordnet_score)
            
            # Apply convergence bonus
            convergence = semantic_result is not None and wordnet_result is not None
            if convergence:
                combined_score += convergence_bonus
            
            base_result = semantic_result or wordnet_result
            if not base_result:
                return None
            
            combined_result = base_result.copy()
            combined_result.update({
                "combined_score": float(combined_score),
                "semantic_score": float(semantic_score),
                "wordnet_score": float(wordnet_score),
                "convergence": convergence,
                "search_method": self._determine_search_method(
                    semantic_result, wordnet_result
                ),
                "combination_strategy": "harmonic_mean"
            })
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Harmonic mean strategy failed: {e}")
            return None
    
    def _geometric_mean_strategy(
        self,
        group_data: Dict,
        semantic_weight: float,
        wordnet_weight: float,
        convergence_bonus: float
    ) -> Optional[Dict[str, Any]]:
        """Geometric mean combination strategy."""
        try:
            semantic_result = group_data["semantic"]
            wordnet_result = group_data["wordnet"]
            
            semantic_score = semantic_result.get("semantic_score", 0.0) if semantic_result else 0.0
            wordnet_score = wordnet_result.get("wordnet_score", 0.0) if wordnet_result else 0.0
            
            # Calculate geometric mean (only if both scores > 0)
            if semantic_score > 0 and wordnet_score > 0:
                combined_score = (semantic_score * wordnet_score) ** 0.5
            else:
                combined_score = max(semantic_score, wordnet_score)
            
            # Apply convergence bonus
            convergence = semantic_result is not None and wordnet_result is not None
            if convergence:
                combined_score += convergence_bonus
            
            base_result = semantic_result or wordnet_result
            if not base_result:
                return None
            
            combined_result = base_result.copy()
            combined_result.update({
                "combined_score": float(combined_score),
                "semantic_score": float(semantic_score),
                "wordnet_score": float(wordnet_score),
                "convergence": convergence,
                "search_method": self._determine_search_method(
                    semantic_result, wordnet_result
                ),
                "combination_strategy": "geometric_mean"
            })
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Geometric mean strategy failed: {e}")
            return None
    
    def _determine_search_method(
        self,
        semantic_result: Optional[Dict],
        wordnet_result: Optional[Dict]
    ) -> str:
        """Determine the search method label for the combined result."""
        if semantic_result and wordnet_result:
            return "hybrid_convergence"
        elif semantic_result:
            return "semantic_dominant"
        elif wordnet_result:
            return "wordnet_dominant"
        else:
            return "unknown"
    
    def get_combination_statistics(
        self,
        combined_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get statistics about the combination process."""
        try:
            if not combined_results:
                return {"total_results": 0}
            
            # Count by search method
            method_counts = defaultdict(int)
            convergence_count = 0
            
            total_semantic_score = 0.0
            total_wordnet_score = 0.0
            total_combined_score = 0.0
            
            for result in combined_results:
                method = result.get("search_method", "unknown")
                method_counts[method] += 1
                
                if result.get("convergence", False):
                    convergence_count += 1
                
                total_semantic_score += result.get("semantic_score", 0.0)
                total_wordnet_score += result.get("wordnet_score", 0.0)
                total_combined_score += result.get("combined_score", 0.0)
            
            total_results = len(combined_results)
            
            return {
                "total_results": total_results,
                "convergence_results": convergence_count,
                "convergence_rate": convergence_count / total_results,
                "method_distribution": dict(method_counts),
                "average_semantic_score": total_semantic_score / total_results,
                "average_wordnet_score": total_wordnet_score / total_results,
                "average_combined_score": total_combined_score / total_results
            }
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {"total_results": len(combined_results), "error": str(e)}