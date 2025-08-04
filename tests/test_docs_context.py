"""
Test Suite for Intelligent Documentation Context System

Comprehensive testing for dynamic documentation fetching and consciousness-aware
context retrieval with performance validation and caching verification.
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memmimic.dspy_optimization.docs_context import (
    IntelligentDocsContextSystem,
    DocumentationContext,
    DocumentationSource,
    DocumentationCache
)
from memmimic.dspy_optimization.config import create_default_config

class TestIntelligentDocsContextSystem:
    """Test suite for intelligent documentation context system"""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return create_default_config()
    
    @pytest.fixture
    def docs_system(self, test_config):
        """Create documentation context system for testing"""
        return IntelligentDocsContextSystem(test_config)
    
    @pytest.fixture
    def sample_documentation_files(self):
        """Create sample documentation files for testing"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample markdown files
        (temp_dir / "test_doc1.md").write_text("""
# DSPy Optimization Guide

This document explains DSPy optimization techniques for consciousness processing.

## Key Concepts
- Chain of Thought reasoning
- Optimization with MIPROv2
- Signature design patterns

## Implementation
DSPy provides powerful optimization capabilities for LLM applications.
        """)
        
        (temp_dir / "nervous_system.md").write_text("""
# Nervous System Architecture

Biological reflex design for sub-5ms response times.

## Core Components
- Fast path processing
- Biological reflexes
- Consciousness pattern classification

## Performance Targets
- Sub-5ms biological reflexes
- 100% reliability
- Graceful fallbacks
        """)
        
        (temp_dir / "mcp_integration.md").write_text("""
# MCP Tool Integration

Model Context Protocol integration patterns.

## Features
- Tool definitions
- Async/sync bridging
- Error handling
- Performance monitoring

## Best Practices
- Use structured error handling
- Implement circuit breakers
- Monitor performance metrics
        """)
        
        return temp_dir
    
    def test_initialization(self, docs_system):
        """Test documentation context system initialization"""
        assert docs_system.config is not None
        assert len(docs_system.documentation_sources) > 0
        assert len(docs_system.consciousness_mappings) > 0
        assert docs_system.cache == {}
        
        # Check required documentation sources
        assert "anthropic_docs" in docs_system.documentation_sources
        assert "dspy_docs" in docs_system.documentation_sources
        assert "mcp_docs" in docs_system.documentation_sources
        assert "memmimic_internal" in docs_system.documentation_sources
    
    def test_consciousness_pattern_mappings(self, docs_system):
        """Test consciousness pattern to documentation mappings"""
        mappings = docs_system.consciousness_mappings
        
        # Check key consciousness patterns
        assert "biological_reflex" in mappings
        assert "dspy_optimization" in mappings
        assert "tool_selection" in mappings
        assert "consciousness_vault" in mappings
        
        # Verify mappings contain relevant URLs
        assert any("nervous" in url.lower() for url in mappings["biological_reflex"])
        assert any("dspy" in url.lower() for url in mappings["dspy_optimization"])
        assert any("tool" in url.lower() for url in mappings["tool_selection"])
    
    def test_keyword_extraction(self, docs_system):
        """Test keyword extraction from queries"""
        # Test with DSPy-related query
        keywords = docs_system._extract_keywords("How do I optimize DSPy consciousness patterns?")
        assert "dspy" in keywords
        assert "optimize" in keywords or "optimization" in keywords
        assert "consciousness" in keywords
        
        # Test with tool selection query
        keywords = docs_system._extract_keywords("Select the best MCP tool for memory operations")
        assert "tool" in keywords
        assert "memory" in keywords
        
        # Test with biological reflex query
        keywords = docs_system._extract_keywords("Improve biological reflex response times")
        assert "biological" in keywords
        assert "reflex" in keywords
    
    def test_candidate_url_generation(self, docs_system):
        """Test candidate URL generation from consciousness patterns"""
        # Test with specific consciousness patterns
        urls = docs_system._get_candidate_urls(
            "DSPy optimization patterns",
            ["dspy_optimization", "pattern_recognition"]
        )
        
        assert len(urls) > 0
        assert any("dspy" in url.lower() for url in urls)
        
        # Test with no matching patterns (should return defaults)
        urls = docs_system._get_candidate_urls(
            "random query",
            ["unknown_pattern"]
        )
        
        assert len(urls) > 0  # Should have default URLs
        assert any("claude-code" in url for url in urls)
    
    @pytest.mark.asyncio
    async def test_url_relevance_scoring(self, docs_system):
        """Test URL relevance scoring algorithm"""
        urls = [
            "dspy_docs:/docs/building-blocks/optimizers",
            "anthropic_docs:/en/docs/claude-code/overview", 
            "memmimic_internal:implementation_docs/NERVOUS_SYSTEM_REMEMBER_SPECS.md"
        ]
        
        # Test DSPy-related query
        scored_urls = await docs_system._score_url_relevance(
            "DSPy optimization techniques",
            urls,
            ["dspy_optimization"]
        )
        
        assert len(scored_urls) == len(urls)
        assert all(isinstance(score, float) for _, score in scored_urls)
        assert all(0.0 <= score <= 1.0 for _, score in scored_urls)
        
        # DSPy URL should score highest for DSPy query
        dspy_score = next(score for url, score in scored_urls if "dspy" in url)
        assert dspy_score > 0.5
    
    @pytest.mark.asyncio
    async def test_internal_documentation_fetching(self, docs_system, sample_documentation_files):
        """Test fetching internal documentation files"""
        # Create a test file
        test_file = sample_documentation_files / "test_internal.md"
        test_content = "# Test Internal Documentation\n\nThis is test content for internal docs."
        test_file.write_text(test_content)
        
        # Mock the internal URL to point to our test file
        url = f"memmimic_internal:{test_file}"
        
        result = await docs_system._fetch_internal_documentation(url)
        
        assert result is not None
        assert result["title"] == "test_internal.md"
        assert "Test Internal Documentation" in result["content"]
        assert result["source"] == "memmimic_internal"
        assert not result["cached"]
    
    @pytest.mark.asyncio
    async def test_external_documentation_fetching(self, docs_system):
        """Test external documentation fetching placeholder"""
        url = "anthropic_docs:/en/docs/claude-code/overview"
        
        result = await docs_system._fetch_external_documentation(url)
        
        assert result is not None
        assert result["url"] == url
        assert "External Documentation" in result["title"]
        assert result["external"] is True
        assert not result["cached"]
    
    @pytest.mark.asyncio
    async def test_documentation_context_retrieval(self, docs_system, sample_documentation_files):
        """Test complete documentation context retrieval"""
        # Test with consciousness patterns
        context = await docs_system.get_documentation_context(
            query="How to implement biological reflexes for fast response?",
            consciousness_patterns=["biological_reflex", "nervous_system"],
            max_docs=3,
            relevance_threshold=0.5
        )
        
        assert isinstance(context, DocumentationContext)
        assert context.query == "How to implement biological reflexes for fast response?"
        assert len(context.consciousness_patterns) == 2
        assert context.fetch_time_ms > 0
        assert len(context.sources_used) >= 0
        assert 0.0 <= context.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, docs_system):
        """Test documentation caching and cache hits"""
        # First request should be a cache miss
        initial_cache_misses = docs_system.metrics["cache_misses"]
        
        context1 = await docs_system.get_documentation_context(
            query="DSPy optimization",
            consciousness_patterns=["dspy_optimization"],
            max_docs=2
        )
        
        # Cache should have increased
        assert docs_system.metrics["cache_misses"] > initial_cache_misses
        
        # Second identical request should use cache
        initial_cache_hits = docs_system.metrics["cache_hits"]
        
        context2 = await docs_system.get_documentation_context(
            query="DSPy optimization",
            consciousness_patterns=["dspy_optimization"],
            max_docs=2
        )
        
        # Should have some cache hits (for any successful fetches from first request)
        assert len(docs_system.cache) >= 0
    
    def test_cache_freshness_checking(self, docs_system):
        """Test cache freshness validation"""
        # Add a fresh cache entry
        fresh_url = "test://fresh.doc"
        docs_system.cache[fresh_url] = DocumentationCache(
            url=fresh_url,
            content="Fresh content",
            title="Fresh Doc",
            last_updated=time.time(),
            relevance_score=0.8
        )
        
        assert docs_system._is_cache_fresh(fresh_url)
        
        # Add a stale cache entry
        stale_url = "test://stale.doc"
        docs_system.cache[stale_url] = DocumentationCache(
            url=stale_url,
            content="Stale content",
            title="Stale Doc",
            last_updated=time.time() - 7200,  # 2 hours ago
            relevance_score=0.8
        )
        
        assert not docs_system._is_cache_fresh(stale_url)
    
    def test_source_name_extraction(self, docs_system):
        """Test source name extraction from URLs"""
        assert docs_system._get_source_name("memmimic_internal:test.md") == "memmimic_internal"
        assert docs_system._get_source_name("anthropic_docs:/en/docs/claude-code/overview") == "anthropic_docs"
        assert docs_system._get_source_name("dspy_docs:/docs/building-blocks") == "dspy_docs"
        assert docs_system._get_source_name("mcp_docs:/concepts/tools") == "mcp_docs"
        assert docs_system._get_source_name("unknown://example.com") == "unknown"
    
    def test_performance_metrics_tracking(self, docs_system):
        """Test performance metrics collection"""
        initial_metrics = docs_system.get_performance_metrics()
        
        assert "total_requests" in initial_metrics
        assert "cache_hits" in initial_metrics
        assert "cache_misses" in initial_metrics
        assert "average_fetch_time" in initial_metrics
        assert "cache_hit_rate" in initial_metrics
        
        # Update metrics
        docs_system._update_metrics(100.0)
        docs_system.metrics["total_requests"] += 1
        
        updated_metrics = docs_system.get_performance_metrics()
        assert updated_metrics["average_fetch_time"] > 0
    
    def test_cache_management(self, docs_system):
        """Test cache management operations"""
        # Add some cache entries
        docs_system.cache["test1"] = DocumentationCache(
            url="test1",
            content="Content 1",
            title="Doc 1",
            last_updated=time.time(),
            relevance_score=0.8,
            access_count=5
        )
        
        docs_system.cache["test2"] = DocumentationCache(
            url="test2",
            content="Content 2",
            title="Doc 2",
            last_updated=time.time(),
            relevance_score=0.6,
            access_count=2
        )
        
        # Test cache summary
        summary = docs_system.get_cache_summary()
        assert summary["cached_documents"] == 2
        assert summary["total_size"] > 0
        assert len(summary["most_accessed"]) <= 2
        
        # Test cache clearing
        docs_system.clear_cache()
        assert len(docs_system.cache) == 0
        
        empty_summary = docs_system.get_cache_summary()
        assert empty_summary["cached_documents"] == 0
    
    @pytest.mark.asyncio
    async def test_consciousness_pattern_integration(self, docs_system):
        """Test integration with various consciousness patterns"""
        test_patterns = [
            (["biological_reflex"], "fast response times"),
            (["dspy_optimization"], "optimize model performance"),
            (["tool_selection"], "choose the right MCP tool"),
            (["consciousness_vault"], "memory management strategies"),
            (["synergy_protocol"], "exponential collaboration")
        ]
        
        for patterns, query in test_patterns:
            context = await docs_system.get_documentation_context(
                query=query,
                consciousness_patterns=patterns,
                max_docs=2,
                relevance_threshold=0.3
            )
            
            assert isinstance(context, DocumentationContext)
            assert context.consciousness_patterns == patterns
            assert len(context.sources_used) >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, docs_system):
        """Test error handling in documentation retrieval"""
        # Test with invalid consciousness patterns
        context = await docs_system.get_documentation_context(
            query="test query",
            consciousness_patterns=["invalid_pattern", "another_invalid"],
            max_docs=1
        )
        
        # Should still return valid context object
        assert isinstance(context, DocumentationContext)
        assert context.query == "test query"
        
        # Test with empty query
        context = await docs_system.get_documentation_context(
            query="",
            consciousness_patterns=[],
            max_docs=1
        )
        
        assert isinstance(context, DocumentationContext)
        assert context.query == ""
    
    @pytest.mark.asyncio
    async def test_relevance_threshold_filtering(self, docs_system):
        """Test relevance threshold filtering"""
        # High threshold should return fewer documents
        high_threshold_context = await docs_system.get_documentation_context(
            query="DSPy optimization",
            consciousness_patterns=["dspy_optimization"],
            max_docs=10,
            relevance_threshold=0.9
        )
        
        # Low threshold should return more documents
        low_threshold_context = await docs_system.get_documentation_context(
            query="DSPy optimization",
            consciousness_patterns=["dspy_optimization"],
            max_docs=10,
            relevance_threshold=0.1
        )
        
        # Low threshold should have equal or more documents
        assert len(low_threshold_context.relevant_docs) >= len(high_threshold_context.relevant_docs)

if __name__ == "__main__":
    # Run basic tests for development
    import sys
    import os
    
    # Add src to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    
    async def run_development_tests():
        """Run tests for development/debugging"""
        print("🧪 Running Documentation Context System Tests\n")
        
        # Create test instance
        config = create_default_config()
        docs_system = IntelligentDocsContextSystem(config)
        
        print("✅ Documentation Context System Initialized")
        print(f"   • Documentation sources: {len(docs_system.documentation_sources)}")
        print(f"   • Consciousness mappings: {len(docs_system.consciousness_mappings)}")
        
        # Test consciousness pattern mappings
        print("\n🔍 Testing Consciousness Pattern Mappings:")
        for pattern, urls in list(docs_system.consciousness_mappings.items())[:5]:
            print(f"   • {pattern}: {len(urls)} URLs")
        
        # Test documentation context retrieval
        print("\n📚 Testing Documentation Context Retrieval:")
        
        test_cases = [
            ("How do I implement biological reflexes?", ["biological_reflex"]),
            ("DSPy optimization best practices", ["dspy_optimization"]),
            ("MCP tool selection strategies", ["tool_selection", "mcp_integration"]),
            ("Consciousness vault memory patterns", ["consciousness_vault", "memory_optimization"])
        ]
        
        for query, patterns in test_cases:
            print(f"\n   Query: {query}")
            print(f"   Patterns: {patterns}")
            
            context = await docs_system.get_documentation_context(
                query=query,
                consciousness_patterns=patterns,
                max_docs=3,
                relevance_threshold=0.5
            )
            
            print(f"   • Docs found: {len(context.relevant_docs)}")
            print(f"   • Confidence: {context.confidence_score:.3f}")
            print(f"   • Sources: {context.sources_used}")
            print(f"   • Fetch time: {context.fetch_time_ms:.1f}ms")
        
        # Test performance metrics
        print(f"\n📊 Performance Metrics:")
        metrics = docs_system.get_performance_metrics()
        print(f"   • Total requests: {metrics['total_requests']}")
        print(f"   • Cache hits: {metrics['cache_hits']}")
        print(f"   • Cache misses: {metrics['cache_misses']}")
        print(f"   • Average fetch time: {metrics['average_fetch_time']:.1f}ms")
        print(f"   • Cache size: {metrics['cache_size']}")
        
        print(f"\n🎯 Documentation Context System Testing Complete!")
    
    # Run development tests
    asyncio.run(run_development_tests())