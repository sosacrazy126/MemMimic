"""
Comprehensive unit tests for Enhanced Memory Model
Testing all functionality including edge cases and backward compatibility.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from ..enhanced_memory import EnhancedMemory
from ..storage.amms_storage import Memory


class TestEnhancedMemoryBasics:
    """Test basic EnhancedMemory functionality"""
    
    def test_enhanced_memory_creation_with_full_context(self):
        """Test creating enhanced memory with full context"""
        full_text = "This is a comprehensive test case. It contains multiple sentences. The summary should capture the essence."
        
        memory = EnhancedMemory(
            content="Test content",
            full_context=full_text,
            tags=["test", "example"]
        )
        
        assert memory.content == "Test content"
        assert memory.full_context == full_text
        assert memory.tags == ["test", "example"]
        assert memory.tag_count == 2
        assert memory.context_size == len(full_text)
        assert memory.summary is not None
        assert memory.governance_status == "approved"
    
    def test_enhanced_memory_auto_summary_generation(self):
        """Test automatic summary generation from full context"""
        full_text = ("This is the first sentence of our test. "
                    "This is the second sentence with more details. "
                    "This is the third sentence that might be included. "
                    "This is the fourth sentence that probably won't be included.")
        
        memory = EnhancedMemory(
            full_context=full_text,
            tags=["auto-summary"]
        )
        
        # Summary should be generated automatically
        assert memory.summary is not None
        assert len(memory.summary) > 0
        assert len(memory.summary) < len(full_text)
        
        # Content should be derived from summary for backward compatibility
        assert memory.content is not None
        assert len(memory.content) <= 1000
    
    def test_enhanced_memory_backward_compatibility_content_only(self):
        """Test backward compatibility when only content is provided"""
        content = "Simple test content for backward compatibility."
        
        memory = EnhancedMemory(content=content)
        
        assert memory.content == content
        assert memory.summary is not None  # Generated from content
        assert memory.full_context == content  # Should be set to content
        assert memory.context_size == len(content)
    
    def test_enhanced_memory_summary_only(self):
        """Test creation with summary only"""
        summary = "This is a test summary."
        
        memory = EnhancedMemory(summary=summary)
        
        assert memory.summary == summary
        assert memory.content == summary  # Should be set for backward compatibility
        assert memory.full_context is None
        assert memory.context_size == len(summary)


class TestSummaryGeneration:
    """Test intelligent summary generation algorithms"""
    
    def test_summary_generation_short_content(self):
        """Test summary generation for short content"""
        short_content = "This is short content."
        
        memory = EnhancedMemory(full_context=short_content)
        
        # Short content should be used as-is for summary
        assert memory.summary == short_content
    
    def test_summary_generation_multiple_sentences(self):
        """Test summary generation with multiple sentences"""
        long_content = ("This is the first sentence. "
                       "This is the second sentence with more information. "
                       "This is the third sentence. "
                       "This is the fourth sentence. "
                       "This is the fifth sentence that should be excluded.")
        
        memory = EnhancedMemory(full_context=long_content)
        
        # Summary should be shorter than original
        assert len(memory.summary) < len(long_content)
        # Should start with first sentence
        assert memory.summary.startswith("This is the first sentence")
        # Should not exceed performance bounds
        assert len(memory.summary) <= 1000
    
    def test_summary_generation_very_long_content(self):
        """Test summary generation with very long content"""
        very_long_content = ("This is a very long piece of content. " * 100)
        
        memory = EnhancedMemory(full_context=very_long_content)
        
        # Summary should be significantly shorter
        assert len(memory.summary) < len(very_long_content)
        assert len(memory.summary) <= 1000
        # Should end with ellipsis if truncated
        if len(memory.summary) == 1000:
            assert memory.summary.endswith("...")
    
    def test_summary_generation_no_sentences(self):
        """Test summary generation with content that has no clear sentences"""
        no_sentences = "just some words without proper punctuation and structure"
        
        memory = EnhancedMemory(full_context=no_sentences)
        
        # Should still generate a reasonable summary
        assert memory.summary is not None
        assert len(memory.summary) > 0
        assert len(memory.summary) <= 1000
    
    def test_summary_generation_empty_content(self):
        """Test summary generation with empty content"""
        memory = EnhancedMemory(full_context="")
        
        assert memory.summary == ""
        
        memory2 = EnhancedMemory(full_context=None)
        assert memory2.summary is None or memory2.summary == ""


class TestGovernanceAndMetrics:
    """Test governance-related functionality and metrics"""
    
    def test_governance_metrics_calculation(self):
        """Test calculation of governance metrics"""
        memory = EnhancedMemory(
            summary="Short summary",
            full_context="This is the full context with more detailed information.",
            tags=["tag1", "tag2", "tag3"],
            metadata={"key": "value", "number": 123}
        )
        
        metrics = memory.calculate_governance_metrics()
        
        assert metrics['content_size'] == len(memory.full_context)
        assert metrics['summary_length'] == len(memory.summary)
        assert metrics['tag_count'] == 3
        assert metrics['metadata_size'] > 0
        assert metrics['has_summary'] is True
        assert metrics['has_full_context'] is True
        assert 0 < metrics['summary_compression_ratio'] < 1
    
    def test_context_size_calculation(self):
        """Test context size calculation"""
        # With full_context
        memory1 = EnhancedMemory(
            content="Short content",
            full_context="Much longer full context content here"
        )
        assert memory1.context_size == len(memory1.full_context)
        
        # Without full_context, should use content
        memory2 = EnhancedMemory(content="Just content")
        assert memory2.context_size == len(memory2.content)
    
    def test_tag_count_tracking(self):
        """Test tag count tracking"""
        memory = EnhancedMemory(
            content="Test content",
            tags=["tag1", "tag2", "tag3", "tag4", "tag5"]
        )
        
        assert memory.tag_count == 5
        assert len(memory.tags) == memory.tag_count


class TestHashGeneration:
    """Test hash generation for deduplication and integrity"""
    
    def test_summary_hash_generation(self):
        """Test summary hash generation"""
        memory = EnhancedMemory(summary="Test summary for hashing")
        
        assert memory.summary_hash is not None
        assert len(memory.summary_hash) == 16
        assert isinstance(memory.summary_hash, str)
    
    def test_context_hash_generation(self):
        """Test context hash generation"""
        memory = EnhancedMemory(full_context="Test full context for hashing")
        
        assert memory.context_hash is not None
        assert len(memory.context_hash) == 16
        assert isinstance(memory.context_hash, str)
    
    def test_hash_consistency(self):
        """Test that same content produces same hash"""
        content = "Consistent content for hash testing"
        
        memory1 = EnhancedMemory(summary=content)
        memory2 = EnhancedMemory(summary=content)
        
        assert memory1.summary_hash == memory2.summary_hash
    
    def test_hash_uniqueness(self):
        """Test that different content produces different hashes"""
        memory1 = EnhancedMemory(summary="First content")
        memory2 = EnhancedMemory(summary="Second content")
        
        assert memory1.summary_hash != memory2.summary_hash


class TestBackwardCompatibility:
    """Test backward compatibility with legacy Memory objects"""
    
    def test_to_legacy_memory_conversion(self):
        """Test conversion to legacy Memory object"""
        enhanced = EnhancedMemory(
            content="Test content",
            summary="Test summary",
            full_context="Full context content",
            tags=["tag1", "tag2"],
            importance_score=0.8,
            metadata={"key": "value"}
        )
        
        legacy = enhanced.to_legacy_memory()
        
        assert isinstance(legacy, Memory)
        assert legacy.content == enhanced.content
        assert legacy.importance_score == enhanced.importance_score
        assert legacy.metadata == enhanced.metadata
        assert legacy.created_at == enhanced.created_at
        assert legacy.updated_at == enhanced.updated_at
        # Enhanced fields should not be present
        assert not hasattr(legacy, 'summary')
        assert not hasattr(legacy, 'tags')
    
    def test_from_legacy_memory_creation(self):
        """Test creation from legacy Memory object"""
        legacy = Memory(
            content="Legacy content",
            importance_score=0.7,
            metadata={"legacy": True}
        )
        
        enhanced = EnhancedMemory.from_legacy_memory(
            legacy,
            tags=["converted", "legacy"],
            full_context="Enhanced full context"
        )
        
        assert enhanced.content == legacy.content
        assert enhanced.importance_score == legacy.importance_score
        assert enhanced.metadata == legacy.metadata
        assert enhanced.tags == ["converted", "legacy"]
        assert enhanced.full_context == "Enhanced full context"
        assert enhanced.summary is not None  # Should be auto-generated


class TestDisplayAndUtility:
    """Test display and utility methods"""
    
    def test_get_display_content_summary(self):
        """Test getting display content at summary level"""
        memory = EnhancedMemory(
            content="Content",
            summary="Summary",
            full_context="Full context"
        )
        
        display = memory.get_display_content("summary")
        assert display == "Summary"
    
    def test_get_display_content_full(self):
        """Test getting display content at full level"""
        memory = EnhancedMemory(
            content="Content",
            summary="Summary", 
            full_context="Full context"
        )
        
        display = memory.get_display_content("full")
        assert display == "Full context"
    
    def test_get_display_content_fallback(self):
        """Test display content fallback to content"""
        memory = EnhancedMemory(content="Content only")
        
        # Summary level should return summary (generated from content)
        summary_display = memory.get_display_content("summary")
        assert summary_display is not None
        
        # Full level should return content (since no full_context)
        full_display = memory.get_display_content("full")
        assert full_display == "Content only"
    
    def test_string_representations(self):
        """Test string and repr methods"""
        memory = EnhancedMemory(
            content="Test content",
            summary="Test summary",
            tags=["tag1", "tag2"],
            importance_score=0.8
        )
        
        str_repr = str(memory)
        assert "EnhancedMemory" in str_repr
        assert "summary_len" in str_repr
        assert "tags=2" in str_repr
        assert "governance_status=approved" in str_repr
        
        repr_repr = repr(memory)
        assert "EnhancedMemory" in repr_repr
        assert "importance_score=0.8" in repr_repr


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_enhanced_memory(self):
        """Test creating enhanced memory with minimal data"""
        memory = EnhancedMemory()
        
        assert memory.content is None or memory.content == ""
        assert memory.summary is None or memory.summary == ""
        assert memory.tags == []
        assert memory.tag_count == 0
        assert memory.context_size == 0
        assert memory.governance_status == "approved"
    
    def test_very_large_content(self):
        """Test handling of very large content"""
        large_content = "This is a sentence. " * 10000  # Very large content
        
        memory = EnhancedMemory(full_context=large_content)
        
        # Summary should be much smaller than original
        assert len(memory.summary) < len(large_content)
        assert len(memory.summary) <= 1000
        # Should handle large content gracefully
        assert memory.context_size == len(large_content)
    
    def test_unicode_content(self):
        """Test handling of unicode content"""
        unicode_content = "Test with émojis 🚀 and spëcial characters ñoño"
        
        memory = EnhancedMemory(
            full_context=unicode_content,
            tags=["unicode", "test"]
        )
        
        assert memory.full_context == unicode_content
        assert memory.summary is not None
        assert memory.context_hash is not None
        assert memory.summary_hash is not None
    
    def test_none_values_handling(self):
        """Test handling of None values"""
        memory = EnhancedMemory(
            content=None,
            summary=None,
            full_context=None,
            tags=None
        )
        
        # Should handle None values gracefully
        assert memory.tags == []  # Should default to empty list
        assert memory.tag_count == 0
        assert memory.context_size == 0


@pytest.fixture
def sample_enhanced_memory():
    """Fixture providing a sample enhanced memory for testing"""
    return EnhancedMemory(
        content="Sample test content for testing purposes.",
        summary="Sample test content.",
        full_context="Sample test content for testing purposes. This includes additional context and details.",
        tags=["test", "sample", "fixture"],
        importance_score=0.75,
        metadata={"test": True, "category": "fixture"}
    )


class TestIntegrationScenarios:
    """Test realistic usage scenarios"""
    
    def test_typical_usage_scenario(self, sample_enhanced_memory):
        """Test typical enhanced memory usage"""
        memory = sample_enhanced_memory
        
        # Verify all components work together
        assert memory.content is not None
        assert memory.summary is not None
        assert memory.full_context is not None
        assert len(memory.tags) > 0
        assert memory.importance_score > 0
        
        # Test governance metrics
        metrics = memory.calculate_governance_metrics()
        assert metrics['content_size'] > 0
        assert metrics['tag_count'] > 0
        assert metrics['has_summary'] is True
        
        # Test display functionality
        summary_display = memory.get_display_content("summary")
        full_display = memory.get_display_content("full")
        assert len(full_display) >= len(summary_display)
        
        # Test legacy conversion
        legacy = memory.to_legacy_memory()
        assert isinstance(legacy, Memory)
    
    def test_performance_optimized_creation(self):
        """Test creation optimized for performance scenarios"""
        # Simulate rapid memory creation
        memories = []
        
        for i in range(100):
            memory = EnhancedMemory(
                content=f"Performance test content {i}",
                tags=[f"test_{i}", "performance"],
                importance_score=i / 100.0
            )
            memories.append(memory)
        
        # Verify all memories created successfully
        assert len(memories) == 100
        
        # Verify hashes are unique (no collisions)
        summary_hashes = [m.summary_hash for m in memories if m.summary_hash]
        assert len(set(summary_hashes)) == len(summary_hashes)
    
    def test_governance_compliance_scenario(self):
        """Test scenario focused on governance compliance"""
        memory = EnhancedMemory(
            summary="Governance test summary",
            full_context="Full governance test context with detailed information",
            tags=["governance", "compliance", "test"],
            metadata={"compliance": True, "level": "high"}
        )
        
        metrics = memory.calculate_governance_metrics()
        
        # Verify all governance metrics are calculable
        required_metrics = [
            'content_size', 'summary_length', 'tag_count', 
            'metadata_size', 'has_summary', 'has_full_context',
            'summary_compression_ratio'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert metrics[metric] is not None


if __name__ == "__main__":
    # Run tests directly
    import sys
    pytest.main([__file__] + sys.argv[1:])