"""
Enhanced Memory Model for MemMimic v2.0
Dual-layer storage with intelligent summary generation and governance support.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .storage.amms_storage import Memory


@dataclass
class EnhancedMemory(Memory):
    """
    Enhanced Memory with dual-layer support extending existing Memory foundation.
    
    Provides backward compatibility while adding v2.0 capabilities:
    - Dual-layer storage (summary + full_context)
    - Intelligent summary generation
    - Tag system for enhanced organization
    - Governance metadata tracking
    - Performance optimization fields
    """
    
    # Override parent's content field to make it optional
    content: str = field(default="")
    
    # New dual-layer fields
    summary: Optional[str] = None
    full_context: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Governance tracking
    governance_status: str = "approved"
    context_size: int = 0
    tag_count: int = 0
    
    # Performance optimization fields
    summary_hash: Optional[str] = None
    context_hash: Optional[str] = None
    
    def __post_init__(self):
        """Auto-generate summary and calculate metrics for governance and performance"""
        # No need to call super().__post_init__() as Memory doesn't have one
        
        # Determine content source priority: explicit content > summary > full_context
        if not self.content and self.summary:
            # Use summary as content, truncated for legacy compatibility
            self.content = self.summary[:1000] if len(self.summary) > 1000 else self.summary
        elif not self.content and self.full_context:
            # Use beginning of full_context as content if no summary
            temp_summary = self._generate_intelligent_summary(self.full_context)
            self.content = temp_summary[:1000] if len(temp_summary) > 1000 else temp_summary
        
        # Generate summary if not provided
        if self.full_context and not self.summary:
            self.summary = self._generate_intelligent_summary(self.full_context)
        elif self.content and not self.summary:
            # Generate summary from content if not provided
            self.summary = self._generate_intelligent_summary(self.content)
            # Set full_context if not provided
            if not self.full_context:
                self.full_context = self.content
        
        # Calculate metrics for governance and telemetry
        self.context_size = len(self.full_context or self.content or "")
        self.tag_count = len(self.tags)
        
        # Generate hashes for deduplication and integrity
        if self.summary:
            self.summary_hash = hashlib.sha256(self.summary.encode()).hexdigest()[:16]
        if self.full_context:
            self.context_hash = hashlib.sha256(self.full_context.encode()).hexdigest()[:16]
    
    def _generate_intelligent_summary(self, content: str) -> str:
        """
        Generate concise summary optimized for <5ms retrieval performance.
        
        Strategy:
        1. Extract key sentences (first 2-3 sentences)
        2. Preserve essential information and context
        3. Maintain semantic coherence
        4. Target 200-500 characters for optimal performance
        """
        if not content or not content.strip():
            return ""
        
        content = content.strip()
        
        # If content is already short enough, use as-is
        if len(content) <= 500:
            return content
        
        # Split into sentences
        sentences = self._split_sentences(content)
        
        if not sentences:
            # Fallback: truncate at word boundary
            return self._truncate_at_word_boundary(content, 500)
        
        # Select first 2-3 sentences based on total length
        summary_sentences = []
        current_length = 0
        target_length = 500
        
        for sentence in sentences[:4]:  # Consider first 4 sentences max
            sentence_length = len(sentence)
            if current_length + sentence_length > target_length and summary_sentences:
                break
            summary_sentences.append(sentence)
            current_length += sentence_length
            
            # Minimum viable summary (at least 1 sentence)
            if len(summary_sentences) >= 2 and current_length >= 200:
                break
        
        summary = '. '.join(summary_sentences)
        
        # Ensure summary doesn't exceed performance bounds
        if len(summary) > 1000:
            summary = self._truncate_at_word_boundary(summary, 997) + "..."
        
        return summary.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics"""
        # Simple sentence splitting - handles most common cases
        sentence_endings = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up and filter empty sentences
        sentences = []
        for sentence in sentence_endings:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                # Ensure sentence ends with punctuation
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                sentences.append(sentence)
        
        return sentences
    
    def _truncate_at_word_boundary(self, text: str, max_length: int) -> str:
        """Truncate text at word boundary to avoid breaking words"""
        if len(text) <= max_length:
            return text
        
        # Find last space before max_length
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # Don't truncate too aggressively
            return text[:last_space]
        else:
            return truncated
    
    def get_display_content(self, context_level: str = "summary") -> str:
        """Get content appropriate for display at specified context level"""
        if context_level == "summary":
            return self.summary or self.content or ""
        elif context_level == "full":
            return self.full_context or self.content or ""
        else:
            return self.content or ""
    
    def calculate_governance_metrics(self) -> Dict[str, Any]:
        """Calculate governance-relevant metrics for validation"""
        return {
            'content_size': self.context_size,
            'summary_length': len(self.summary) if self.summary else 0,
            'tag_count': self.tag_count,
            'metadata_size': len(str(self.metadata)) if self.metadata else 0,
            'has_summary': bool(self.summary),
            'has_full_context': bool(self.full_context),
            'summary_compression_ratio': (
                len(self.summary) / len(self.full_context) 
                if self.summary and self.full_context and len(self.full_context) > 0 
                else 0.0
            )
        }
    
    def to_legacy_memory(self) -> Memory:
        """Convert to legacy Memory object for backward compatibility"""
        return Memory(
            id=self.id,
            content=self.content,
            metadata=self.metadata,
            importance_score=self.importance_score,
            created_at=self.created_at,
            updated_at=self.updated_at
        )
    
    @classmethod
    def from_legacy_memory(cls, memory: Memory, **enhanced_fields) -> 'EnhancedMemory':
        """Create EnhancedMemory from legacy Memory object"""
        return cls(
            id=memory.id,
            content=memory.content,
            metadata=memory.metadata,
            importance_score=memory.importance_score,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            **enhanced_fields
        )
    
    def __str__(self) -> str:
        """String representation for logging and debugging"""
        return (
            f"EnhancedMemory(id={self.id}, "
            f"summary_len={len(self.summary) if self.summary else 0}, "
            f"context_size={self.context_size}, "
            f"tags={len(self.tags)}, "
            f"governance_status={self.governance_status})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation for development"""
        return (
            f"EnhancedMemory(id={self.id!r}, content_len={len(self.content) if self.content else 0}, "
            f"summary_len={len(self.summary) if self.summary else 0}, "
            f"context_size={self.context_size}, tags={self.tags!r}, "
            f"governance_status={self.governance_status!r}, "
            f"importance_score={self.importance_score}, "
            f"created_at={self.created_at!r})"
        )