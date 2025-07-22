#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Hybrid Search Engine - Core hybrid search logic

Extracted from memmimic_recall_cxd.py for improved maintainability.
Provides the main hybrid search functionality combining semantic and WordNet search.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from .wordnet_expander import WordNetExpander, ensure_wordnet
from .semantic_processor import SemanticProcessor
from .result_combiner import ResultCombiner
from ..storage.amms_storage import AMMSStorage
from ...errors import MemMimicError, DatabaseError, handle_errors

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Core hybrid search engine combining semantic and WordNet-based search.
    
    This is the main orchestrator that coordinates between different search
    components to provide comprehensive memory retrieval capabilities.
    """
    
    def __init__(self, db_name: Optional[str] = None):
        """Initialize the hybrid search engine."""
        self.wordnet_expander = WordNetExpander()
        self.semantic_processor = SemanticProcessor()
        self.result_combiner = ResultCombiner()
        self.memory_store = self._get_memory_store(db_name)
        
        # Ensure WordNet is available
        ensure_wordnet()
    
    def _get_memory_store(self, db_name: Optional[str] = None):
        """Get memory store instance."""
        try:
            return AMMSStorage(db_name or "memmimic.db")
        except Exception as e:
            logger.error(f"Failed to initialize memory store: {e}")
            raise DatabaseError("Memory store initialization failed") from e
    
    @handle_errors(catch=[MemMimicError], log_level="ERROR")
    def search_memories_hybrid(
        self,
        query: str,
        limit: int = 5,
        function_filter: str = "ALL",
        db_name: Optional[str] = None,
        semantic_weight: float = 0.7,
        wordnet_weight: float = 0.3,
        convergence_bonus: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and WordNet approaches.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            function_filter: CXD function filter (CONTROL, CONTEXT, DATA, ALL)
            db_name: Optional database name override
            semantic_weight: Weight for semantic search results (0.0-1.0)
            wordnet_weight: Weight for WordNet search results (0.0-1.0)
            convergence_bonus: Bonus for results found by both methods
            
        Returns:
            List of search results with combined scoring
        """
        start_time = time.time()
        
        try:
            # Phase 1: Semantic search
            semantic_results = self.semantic_processor.search(
                query=query,
                memory_store=self.memory_store,
                limit=limit * 2  # Get more results for better fusion
            )
            
            # Phase 2: WordNet-enhanced search
            expanded_queries = self.wordnet_expander.expand_query(query)
            wordnet_results = self.wordnet_expander.search_with_expansion(
                expanded_queries=expanded_queries,
                memory_store=self.memory_store,
                limit=limit * 2
            )
            
            # Phase 3: Combine and score results
            combined_results = self.result_combiner.combine_and_score(
                semantic_results=semantic_results,
                wordnet_results=wordnet_results,
                semantic_weight=semantic_weight,
                wordnet_weight=wordnet_weight,
                convergence_bonus=convergence_bonus
            )
            
            # Phase 4: Apply CXD filtering
            if function_filter != "ALL":
                combined_results = self._apply_cxd_filter(
                    combined_results, function_filter
                )
            
            # Phase 5: Limit and format results
            final_results = combined_results[:limit]
            
            search_time = (time.time() - start_time) * 1000
            logger.info(f"Hybrid search completed in {search_time:.2f}ms")
            
            return self._format_results(final_results, search_time)
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise MemMimicError(f"Search operation failed: {str(e)}") from e
    
    def _apply_cxd_filter(self, results: List[Dict], function_filter: str) -> List[Dict]:
        """Apply CXD function filtering to results."""
        if function_filter == "ALL":
            return results
        
        filter_map = {
            "CONTROL": ["C", "Control"],
            "CONTEXT": ["X", "Context"], 
            "DATA": ["D", "Data"]
        }
        
        valid_functions = filter_map.get(function_filter, [])
        if not valid_functions:
            return results
        
        filtered_results = []
        for result in results:
            cxd_function = result.get("cxd_function", "")
            if cxd_function in valid_functions:
                filtered_results.append(result)
        
        return filtered_results
    
    def _format_results(self, results: List[Dict], search_time: float) -> List[Dict]:
        """Format search results with metadata."""
        formatted_results = []
        
        for i, result in enumerate(results, 1):
            formatted_result = {
                "rank": i,
                "content": result.get("content", ""),
                "combined_score": result.get("combined_score", 0.0),
                "semantic_score": result.get("semantic_score", 0.0),
                "wordnet_score": result.get("wordnet_score", 0.0),
                "convergence": result.get("convergence", False),
                "cxd_function": result.get("cxd_function", ""),
                "memory_type": result.get("memory_type", ""),
                "created_at": result.get("created_at", ""),
                "search_method": result.get("search_method", "hybrid"),
            }
            formatted_results.append(formatted_result)
        
        # Add search metadata
        search_metadata = {
            "search_time_ms": search_time,
            "total_results": len(formatted_results),
            "search_type": "hybrid",
            "timestamp": time.time()
        }
        
        return {
            "results": formatted_results,
            "metadata": search_metadata
        }


# Backward compatibility function
def search_memories_hybrid(
    query: str,
    limit: int = 5,
    function_filter: str = "ALL",
    db_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Backward compatibility wrapper for the original search function.
    
    This maintains API compatibility while using the new modular architecture.
    """
    engine = HybridSearchEngine(db_name)
    return engine.search_memories_hybrid(
        query=query,
        limit=limit,
        function_filter=function_filter,
        db_name=db_name
    )