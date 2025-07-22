#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemMimic Memory Quality Gate - Intelligent Memory Approval System
Uses existing ContextualAssistant for quality judgment and duplicate detection
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

from .storage.amms_storage import Memory, AMMSStorage
from ..assistant import ContextualAssistant
from .quality_types import MemoryQualityResult
from .persistent_queue import PersistentMemoryQueue
from .semantic_similarity import get_semantic_detector
from ..config import get_performance_config


class MemoryQualityGate:
    """
    Intelligent memory quality control using existing ContextualAssistant
    
    Provides:
    - Duplicate detection using semantic search
    - Quality assessment using assistant's confidence system
    - Automatic approval for high-quality, unique memories
    - Human review queue for borderline cases
    """
    
    def __init__(self, assistant: ContextualAssistant, queue_db_path: str = "memory_queue.db"):
        self.assistant = assistant
        self.memory_store = assistant.memory_store
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = get_performance_config()
        memory_config = self.config.memory_config
        
        # Quality thresholds from configuration
        self.auto_approve_threshold = memory_config.get('auto_approve_threshold', 0.8)
        self.auto_reject_threshold = memory_config.get('auto_reject_threshold', 0.3)
        self.duplicate_threshold = memory_config.get('duplicate_threshold', 0.85)
        self.min_content_length = memory_config.get('min_content_length', 10)
        
        # Persistent memory queue
        self.persistent_queue = PersistentMemoryQueue(queue_db_path)
    
    async def evaluate_memory(
        self, 
        content: str, 
        memory_type: str = "interaction"
    ) -> MemoryQualityResult:
        """
        Evaluate memory quality using the assistant's intelligence
        
        Returns MemoryQualityResult with approval decision
        """
        try:
            # Basic validation
            if len(content.strip()) < self.min_content_length:
                return MemoryQualityResult(
                    approved=False,
                    reason="Content too short (minimum 10 characters)",
                    confidence=0.0,
                    auto_decision=True
                )
            
            # Check for duplicates using existing search
            duplicates = await self._find_duplicates(content)
            if duplicates:
                similarity_score = duplicates[0][1] if duplicates else 0.0
                if similarity_score > self.duplicate_threshold:
                    return MemoryQualityResult(
                        approved=False,
                        reason=f"High similarity to existing memory ({similarity_score:.2f})",
                        confidence=1.0 - similarity_score,
                        duplicates=[dup[0] for dup in duplicates],
                        auto_decision=True
                    )
            
            # Use assistant's thinking process for quality assessment
            quality_assessment = await self._assess_content_quality(content, memory_type)
            
            # Make approval decision
            if quality_assessment["confidence"] >= self.auto_approve_threshold:
                return MemoryQualityResult(
                    approved=True,
                    reason="High quality content with good confidence",
                    confidence=quality_assessment["confidence"],
                    auto_decision=True
                )
            elif quality_assessment["confidence"] <= self.auto_reject_threshold:
                return MemoryQualityResult(
                    approved=False,
                    reason="Low quality content with poor confidence",
                    confidence=quality_assessment["confidence"],
                    suggested_content=quality_assessment.get("suggestion"),
                    auto_decision=True
                )
            else:
                # Requires human review
                return MemoryQualityResult(
                    approved=False,  # Pending approval
                    reason="Borderline quality - requires human review",
                    confidence=quality_assessment["confidence"],
                    suggested_content=quality_assessment.get("suggestion"),
                    auto_decision=False
                )
                
        except Exception as e:
            self.logger.error(f"Quality evaluation failed: {e}")
            return MemoryQualityResult(
                approved=False,
                reason=f"Evaluation error: {str(e)}",
                confidence=0.0,
                auto_decision=True
            )
    
    async def _find_duplicates(self, content: str) -> List[Tuple[Memory, float]]:
        """Find potential duplicate memories using semantic similarity"""
        try:
            # Use existing search to find similar memories
            similar_memories = await self.memory_store.search_memories(content, limit=10)
            
            # Use semantic similarity detector for enhanced duplicate detection
            semantic_detector = get_semantic_detector()
            duplicates = []
            
            for memory in similar_memories:
                # Calculate semantic similarity using embeddings
                similarity = semantic_detector.compute_similarity(content, memory.content)
                if similarity > 0.7:  # Potential duplicate threshold
                    duplicates.append((memory, similarity))
            
            # Sort by similarity (highest first)
            duplicates.sort(key=lambda x: x[1], reverse=True)
            return duplicates
            
        except Exception as e:
            self.logger.error(f"Semantic duplicate detection failed: {e}")
            # Fallback to simple text similarity
            return await self._find_duplicates_fallback(content)
    
    async def _find_duplicates_fallback(self, content: str) -> List[Tuple[Memory, float]]:
        """Fallback duplicate detection using simple word overlap"""
        try:
            similar_memories = await self.memory_store.search_memories(content, limit=5)
            duplicates = []
            
            for memory in similar_memories:
                similarity = self._calculate_text_similarity(content, memory.content)
                if similarity > 0.7:
                    duplicates.append((memory, similarity))
            
            duplicates.sort(key=lambda x: x[1], reverse=True)
            return duplicates
        except Exception as e:
            self.logger.error(f"Fallback duplicate detection failed: {e}")
            return []
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _assess_content_quality(self, content: str, memory_type: str) -> Dict[str, Any]:
        """Enhanced quality assessment with semantic context analysis"""
        try:
            # Simple quality scoring based on content characteristics
            base_confidence = self._calculate_simple_quality_score(content, memory_type)
            
            # Get memory context for assessment using semantic similarity
            similar_memories = await self._find_duplicates(content)
            context_strength = "high" if len(similar_memories) > 2 else "medium" if similar_memories else "none"
            
            # Enhance confidence based on semantic context
            semantic_boost = 0.0
            if similar_memories:
                # Check if this content adds new information vs existing memories
                semantic_detector = get_semantic_detector()
                max_similarity = max([sim for _, sim in similar_memories[:3]]) if similar_memories else 0.0
                
                # If content is too similar to existing, reduce confidence
                if max_similarity > 0.9:
                    semantic_boost = -0.2  # Likely redundant
                elif max_similarity > 0.8:
                    semantic_boost = -0.1  # Somewhat redundant
                elif 0.5 <= max_similarity <= 0.7:
                    semantic_boost = 0.1   # Good related context
                else:
                    semantic_boost = 0.05  # Novel content
            
            final_confidence = max(0.0, min(1.0, base_confidence + semantic_boost))
            
            assessment = {
                "confidence": final_confidence,
                "base_confidence": base_confidence,
                "semantic_boost": semantic_boost,
                "memories_used": len(similar_memories),
                "context_strength": context_strength,
                "max_similarity": max([sim for _, sim in similar_memories[:1]]) if similar_memories else 0.0
            }
            
            # Add suggestions based on assessment
            if assessment["confidence"] < 0.6:
                assessment["suggestion"] = self._generate_improvement_suggestion(content)
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Content quality assessment failed: {e}")
            return {"confidence": 0.5, "error": str(e)}
    
    def _calculate_simple_quality_score(self, content: str, memory_type: str) -> float:
        """Calculate quality score based on content characteristics"""
        score = 0.5  # Base score
        
        # Length-based scoring
        if len(content) > 100:
            score += 0.15
        elif len(content) > 50:
            score += 0.1
        elif len(content) < 20:
            score -= 0.2
        
        # Structure-based scoring
        if '.' in content or '!' in content or '?' in content:
            score += 0.1  # Has sentence structure
        
        # Information density
        words = content.split()
        if len(words) > 10:
            score += 0.1
        
        # Memory type scoring
        if memory_type == "milestone":
            score += 0.1  # Milestones are generally important
        elif memory_type == "reflection":
            score += 0.05  # Reflections have value
        
        # Keyword-based scoring
        quality_keywords = ["important", "key", "critical", "learned", "discovered", "achieved"]
        if any(keyword in content.lower() for keyword in quality_keywords):
            score += 0.1
        
        # Noise detection
        noise_indicators = ["test", "testing", "tmp", "temporary", "debug"]
        if any(noise in content.lower() for noise in noise_indicators):
            score -= 0.2
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def _generate_improvement_suggestion(self, content: str) -> str:
        """Generate suggestions for improving low-quality content"""
        suggestions = []
        
        if len(content) < 30:
            suggestions.append("Add more detail and context")
        
        if not any(word in content.lower() for word in ["why", "how", "what", "when", "where"]):
            suggestions.append("Include key information (who, what, when, where, why)")
        
        if content.count('.') < 1:
            suggestions.append("Use complete sentences")
        
        if suggestions:
            return "Suggestions: " + "; ".join(suggestions)
        else:
            return "Consider adding more context and detail"
    
    async def queue_for_review(self, content: str, memory_type: str, quality_result: MemoryQualityResult) -> str:
        """Queue memory for human review"""
        return self.persistent_queue.add_to_queue(content, memory_type, quality_result)
    
    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get all memories awaiting review"""
        return self.persistent_queue.get_pending_reviews()
    
    async def approve_pending(self, queue_id: str, reviewer_note: str = "") -> bool:
        """Approve a pending memory"""
        # Get memory details from queue
        memory_details = self.persistent_queue.get_memory_details(queue_id)
        if not memory_details or memory_details["status"] != "pending_review":
            return False
        
        # Store the memory
        memory = Memory(
            content=memory_details["content"],
            metadata={
                "type": memory_details["memory_type"], 
                "reviewer_approved": True, 
                "reviewer_note": reviewer_note,
                "queue_id": queue_id
            }
        )
        
        try:
            memory_id = await self.memory_store.store_memory(memory)
            # Mark as approved in queue
            success = self.persistent_queue.approve_memory(queue_id, reviewer_note)
            if success:
                self.logger.info(f"Memory approved and stored: {queue_id} -> {memory_id}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to approve pending memory: {e}")
            return False
    
    async def reject_pending(self, queue_id: str, reason: str) -> bool:
        """Reject a pending memory"""
        return self.persistent_queue.reject_memory(queue_id, reason)


# Convenience function for creating quality gate
def create_quality_gate(assistant_name: str = "memmimic") -> MemoryQualityGate:
    """Create a MemoryQualityGate with default assistant"""
    assistant = ContextualAssistant(assistant_name)
    return MemoryQualityGate(assistant)