#!/usr/bin/env python3
"""
Enhanced MCP Wrapper - AMMS Integration and Performance Optimization
Provides enhanced capabilities for all MemMimic MCP tools with AMMS integration
"""

import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .mcp_performance_monitor import get_performance_monitor


class EnhancedMCPWrapper:
    """
    Enhanced wrapper for MemMimic MCP tools with AMMS integration

    Features:
    - AMMS-aware memory operations
    - Performance monitoring and optimization
    - Enhanced error handling and recovery
    - Caching and optimization
    - Real-time metrics collection
    """

    def __init__(self, tool_name: str, db_path: Optional[str] = None):
        self.tool_name = tool_name

        # Resolve database path
        if db_path is None:
            # Default to MemMimic database
            from pathlib import Path

            memmimic_root = Path(__file__).parent.parent.parent
            self.db_path = str(memmimic_root / "memmimic_memories.db")
        else:
            self.db_path = db_path

        self.logger = logging.getLogger(f"mcp.{tool_name}")
        self.performance_monitor = get_performance_monitor()

        # Initialize AMMS components
        self._init_amms_components()

        # Performance cache
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

        self.logger.info(f"Enhanced MCP wrapper initialized for {tool_name}")

    def _init_amms_components(self):
        """Initialize AMMS components for enhanced operations"""
        try:
            from memmimic.memory import UnifiedMemoryStore

            # Initialize UnifiedMemoryStore with AMMS
            self.memory_store = UnifiedMemoryStore(self.db_path)

            # Get active memory pool for enhanced operations
            if hasattr(self.memory_store, "active_pool"):
                self.active_pool = self.memory_store.active_pool
            else:
                self.active_pool = None

            self.logger.info("AMMS components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize AMMS components: {e}")
            # Fallback to legacy memory store
            try:
                from memmimic.memory import MemoryStore

                self.memory_store = MemoryStore(self.db_path)
                self.active_pool = None
                self.logger.warning("Using legacy memory store as fallback")
            except Exception as e2:
                self.logger.error(f"Failed to initialize fallback memory store: {e2}")
                self.memory_store = None
                self.active_pool = None

    @contextmanager
    def operation_context(self, operation_name: str, **kwargs):
        """Context manager for tracking MCP operations with enhanced monitoring"""
        operation_id = self.performance_monitor.start_operation(
            f"{self.tool_name}.{operation_name}"
        )
        start_time = time.perf_counter()

        try:
            yield operation_id

            # Calculate performance metrics
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Enhanced metrics
            extra_metrics = {"duration_ms": duration_ms, **kwargs}

            self.performance_monitor.finish_operation(
                operation_id, success=True, **extra_metrics
            )

        except Exception as e:
            self.performance_monitor.finish_operation(
                operation_id, success=False, error_message=str(e)
            )
            raise

    def enhanced_search(
        self, query: str, function_filter: str = "ALL", limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search with AMMS integration and performance optimization

        Args:
            query: Search query
            function_filter: CXD function filter
            limit: Maximum results

        Returns:
            Enhanced search results with AMMS optimization
        """
        with self.operation_context(
            "enhanced_search", query_length=len(query), limit=limit
        ) as op_id:
            # Check cache first
            cache_key = f"search:{query}:{function_filter}:{limit}"
            if cache_key in self._cache:
                cache_entry = self._cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self._cache_ttl:
                    self.logger.debug(f"Cache hit for search query: {query}")
                    self.performance_monitor.finish_operation(
                        op_id, success=True, cache_hits=1
                    )
                    return cache_entry["results"]

            # Use AMMS-enhanced search if available
            if self.active_pool:
                results = self._amms_enhanced_search(query, function_filter, limit)
            else:
                # Fallback to legacy search
                results = self._legacy_search(query, function_filter, limit)

            # Cache results
            self._cache[cache_key] = {"timestamp": time.time(), "results": results}

            # Enhanced result post-processing
            enhanced_results = self._enhance_search_results(results, query)

            self.performance_monitor.finish_operation(
                op_id,
                success=True,
                results_returned=len(enhanced_results),
                memories_processed=len(results),
            )

            return enhanced_results

    def _amms_enhanced_search(
        self, query: str, function_filter: str, limit: int
    ) -> List[Dict[str, Any]]:
        """AMMS-enhanced search with active memory pool optimization"""
        try:
            # Use active memory pool for faster search
            pool_results = self.active_pool.search_active_memories(
                query, limit=limit * 2
            )

            if not pool_results:
                # Fallback to full memory store search
                return self._legacy_search(query, function_filter, limit)

            # Convert active pool results to standard format
            results = []
            for result in pool_results:
                memory = result.get("memory")
                if memory:
                    results.append(
                        {
                            "memory": memory,
                            "relevance_score": result.get("relevance_score", 0.0),
                            "importance_score": result.get("importance_score", 0.0),
                            "search_method": "amms_active_pool",
                        }
                    )

            # Apply CXD filtering if needed
            if function_filter != "ALL":
                results = self._apply_cxd_filter(results, function_filter)

            return results[:limit]

        except Exception as e:
            self.logger.error(f"AMMS enhanced search failed: {e}")
            return self._legacy_search(query, function_filter, limit)

    def _legacy_search(
        self, query: str, function_filter: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Legacy search implementation"""
        try:
            memories = self.memory_store.search(query, limit=limit)
            results = []

            for memory in memories:
                results.append(
                    {
                        "memory": memory,
                        "relevance_score": 0.5,  # Default relevance
                        "search_method": "legacy",
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"Legacy search failed: {e}")
            return []

    def _apply_cxd_filter(
        self, results: List[Dict[str, Any]], function_filter: str
    ) -> List[Dict[str, Any]]:
        """Apply CXD function filtering to results"""
        if function_filter == "ALL":
            return results

        try:
            from memmimic.cxd.classifiers.optimized_meta import (
                create_optimized_classifier,
            )

            classifier = create_optimized_classifier()

            filtered_results = []
            for result in results:
                memory = result["memory"]
                content = getattr(memory, "content", "")

                # Classify content
                cxd_result = classifier.classify(content)
                dominant_function = (
                    getattr(cxd_result.dominant_function, "value", "UNKNOWN")
                    if cxd_result.dominant_function
                    else "UNKNOWN"
                )

                # Filter by function
                if dominant_function == function_filter or function_filter == "ALL":
                    result["cxd_function"] = dominant_function
                    result["cxd_confidence"] = getattr(
                        cxd_result, "average_confidence", 0.0
                    )
                    filtered_results.append(result)

            return filtered_results

        except Exception as e:
            self.logger.error(f"CXD filtering failed: {e}")
            return results

    def _enhance_search_results(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Enhance search results with additional metadata and optimization"""
        enhanced_results = []

        for result in results:
            memory = result["memory"]

            # Add enhanced metadata
            enhanced_result = {
                **result,
                "memory_id": getattr(memory, "id", None),
                "memory_type": getattr(memory, "memory_type", "unknown"),
                "confidence": getattr(memory, "confidence", 0.0),
                "created_at": getattr(memory, "created_at", ""),
                "access_count": getattr(memory, "access_count", 0),
                "content_length": len(getattr(memory, "content", "")),
                "query_relevance": self._calculate_query_relevance(memory, query),
                "enhanced_by": self.tool_name,
            }

            # Add AMMS-specific metadata if available
            if hasattr(memory, "importance_score"):
                enhanced_result["importance_score"] = memory.importance_score
            if hasattr(memory, "last_access_time"):
                enhanced_result["last_access_time"] = memory.last_access_time
            if hasattr(memory, "archive_status"):
                enhanced_result["archive_status"] = memory.archive_status

            enhanced_results.append(enhanced_result)

        return enhanced_results

    def _calculate_query_relevance(self, memory, query: str) -> float:
        """Calculate relevance score between memory and query"""
        try:
            content = getattr(memory, "content", "").lower()
            query_lower = query.lower()

            # Simple relevance calculation
            query_words = query_lower.split()
            content_words = content.split()

            if not query_words or not content_words:
                return 0.0

            # Calculate word overlap
            query_set = set(query_words)
            content_set = set(content_words)

            overlap = len(query_set.intersection(content_set))
            relevance = overlap / len(query_set)

            return min(relevance, 1.0)

        except Exception as e:
            self.logger.debug(f"Relevance calculation failed: {e}")
            return 0.0

    def enhanced_status(self) -> Dict[str, Any]:
        """Enhanced system status with AMMS integration"""
        with self.operation_context("enhanced_status") as op_id:
            try:
                # Get basic system status
                from memmimic.mcp.memmimic_status import (
                    analyze_memory_statistics,
                    analyze_tales_statistics,
                    check_cxd_status,
                )

                # Basic statistics
                memory_stats = analyze_memory_statistics(self.memory_store)
                tales_stats = analyze_tales_statistics()
                cxd_status = check_cxd_status()

                # AMMS-enhanced status
                amms_status = {}
                if self.active_pool:
                    amms_status = self.active_pool.get_active_pool_status()

                # Performance metrics
                performance_stats = self.performance_monitor.get_system_health()

                # Combined status
                enhanced_status = {
                    "timestamp": datetime.now().isoformat(),
                    "system_health": performance_stats["status"],
                    "memory_statistics": memory_stats,
                    "tales_statistics": tales_stats,
                    "cxd_status": cxd_status,
                    "amms_status": amms_status,
                    "performance_metrics": performance_stats,
                    "enhanced_features": {
                        "amms_enabled": self.active_pool is not None,
                        "performance_monitoring": True,
                        "caching_enabled": True,
                        "enhanced_search": True,
                    },
                }

                return enhanced_status

            except Exception as e:
                self.logger.error(f"Enhanced status failed: {e}")
                return {
                    "timestamp": datetime.now().isoformat(),
                    "system_health": "ERROR",
                    "error": str(e),
                }

    def enhanced_remember(
        self,
        content: str,
        memory_type: str = "interaction",
        confidence: float = 0.8,
        **kwargs,
    ) -> Dict[str, Any]:
        """Enhanced memory creation with AMMS integration"""
        with self.operation_context(
            "enhanced_remember", content_length=len(content), memory_type=memory_type
        ) as op_id:
            try:
                # Use AMMS-enhanced memory creation
                if self.active_pool:
                    # Create memory with AMMS
                    from memmimic.memory.memory import Memory

                    memory = Memory(content, memory_type, confidence)

                    # Add with enhanced CXD classification
                    cxd_function = kwargs.get("cxd_function")
                    cxd_confidence = kwargs.get("cxd_confidence", 0.0)

                    if not cxd_function:
                        # Auto-classify with CXD
                        try:
                            from memmimic.cxd.classifiers.optimized_meta import (
                                create_optimized_classifier,
                            )

                            classifier = create_optimized_classifier()
                            cxd_result = classifier.classify(content)
                            cxd_function = (
                                getattr(cxd_result.dominant_function, "value", "DATA")
                                if cxd_result.dominant_function
                                else "DATA"
                            )
                            cxd_confidence = getattr(
                                cxd_result, "average_confidence", 0.0
                            )
                        except Exception as e:
                            self.logger.debug(f"CXD auto-classification failed: {e}")
                            cxd_function = "DATA"
                            cxd_confidence = 0.0

                    memory_id = self.active_pool.add_memory(
                        memory, cxd_function, cxd_confidence
                    )

                    return {
                        "memory_id": memory_id,
                        "memory_type": memory_type,
                        "confidence": confidence,
                        "cxd_function": cxd_function,
                        "cxd_confidence": cxd_confidence,
                        "amms_enhanced": True,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    # Fallback to legacy memory creation
                    memory_id = self.memory_store.add_memory(
                        content, memory_type, confidence
                    )

                    return {
                        "memory_id": memory_id,
                        "memory_type": memory_type,
                        "confidence": confidence,
                        "amms_enhanced": False,
                        "timestamp": datetime.now().isoformat(),
                    }

            except Exception as e:
                self.logger.error(f"Enhanced remember failed: {e}")
                raise

    def clear_cache(self):
        """Clear performance cache"""
        self._cache.clear()
        self.logger.info("Performance cache cleared")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this tool"""
        return self.performance_monitor.get_operation_stats(self.tool_name)


# Utility functions for MCP tool integration
def enhance_mcp_tool(
    tool_name: str, db_path: Optional[str] = None
) -> EnhancedMCPWrapper:
    """Create enhanced MCP wrapper for a tool"""
    return EnhancedMCPWrapper(tool_name, db_path)


def enhanced_error_handler(func: Callable) -> Callable:
    """Enhanced error handler decorator for MCP tools"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log error with enhanced context
            import traceback

            error_context = {
                "function": func.__name__,
                "args": str(args)[:200],  # Truncate for safety
                "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }

            logger = logging.getLogger("mcp.error_handler")
            logger.error(f"Enhanced MCP error: {error_context}")

            # Return structured error response
            return {
                "error": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_context": error_context,
                "timestamp": datetime.now().isoformat(),
            }

    return wrapper


if __name__ == "__main__":
    # Test the enhanced wrapper
    wrapper = EnhancedMCPWrapper("test_tool")

    # Test enhanced search
    results = wrapper.enhanced_search("test query", "ALL", 5)
    print(f"Search results: {len(results)}")

    # Test enhanced status
    status = wrapper.enhanced_status()
    print(f"System status: {status['system_health']}")

    # Test performance metrics
    metrics = wrapper.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
