"""
InternalQualityGate - 6-Dimensional Quality Assessment

Replaces external quality approval queues with intelligent internal assessment
providing automatic content validation and enhancement suggestions.

Quality Dimensions:
1. Content clarity and coherence
2. Information density and value  
3. Factual accuracy and reliability
4. Contextual relevance
5. Uniqueness and novelty
6. Long-term importance potential
"""

import asyncio
import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from statistics import mean

from .interfaces import QualityGateInterface, QualityAssessment
from ..errors import get_error_logger

@dataclass
class QualityThresholds:
    """Quality thresholds for different memory types"""
    interaction: float = 0.6
    milestone: float = 0.7
    reflection: float = 0.65
    technical: float = 0.75
    synthetic: float = 0.8
    error: float = 0.5

class InternalQualityGate(QualityGateInterface):
    """
    Internal quality assessment system replacing external approval queues.
    
    Provides 6-dimensional quality analysis with automatic enhancement suggestions
    and confidence-based storage decisions without human intervention.
    """
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
        self.logger = get_error_logger("quality_gate")
        
        # Performance metrics
        self._assessment_count = 0
        self._enhancement_count = 0
        self._auto_approval_count = 0
        self._processing_times = []
        
        # Quality patterns learned from assessments
        self._quality_patterns = {
            'high_quality_indicators': [
                'specific examples', 'clear structure', 'actionable insights',
                'quantified results', 'context provided', 'implications discussed'
            ],
            'low_quality_indicators': [
                'vague statements', 'no context', 'unclear meaning',
                'duplicate information', 'irrelevant content', 'poor grammar'
            ],
            'enhancement_patterns': {
                'clarity': ['add context', 'provide examples', 'clarify terms'],
                'density': ['add details', 'include metrics', 'expand concepts'],
                'accuracy': ['verify facts', 'add sources', 'qualify statements'],
                'relevance': ['connect to context', 'explain importance', 'link to goals'],
                'uniqueness': ['highlight novelty', 'compare to existing', 'emphasize differences'],
                'importance': ['explain impact', 'discuss implications', 'project outcomes']
            }
        }
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize quality assessment components"""
        if self._initialized:
            return
        
        self.logger.info("Initializing InternalQualityGate")
        
        # Initialize quality assessment models/patterns
        # In production, this might load ML models or pattern databases
        await self._load_quality_patterns()
        
        self._initialized = True
        self.logger.info("InternalQualityGate initialization complete")
    
    async def _load_quality_patterns(self) -> None:
        """Load quality assessment patterns and models"""
        # Simulate loading time for realistic performance
        await asyncio.sleep(0.001)  # 1ms simulation
        
        # In production implementation, this would:
        # - Load pre-trained quality assessment models
        # - Initialize semantic similarity engines
        # - Load domain-specific quality patterns
        # - Set up factual verification systems
        pass
    
    async def assess_quality(self, content: str, memory_type: str = "interaction") -> QualityAssessment:
        """
        Assess memory quality across 6 dimensions with <3ms performance target.
        
        Returns comprehensive quality assessment with enhancement suggestions
        and automatic approval decision.
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            # Parallel assessment of all 6 dimensions
            async with asyncio.TaskGroup() as tg:
                clarity_task = tg.create_task(self._assess_clarity(content))
                density_task = tg.create_task(self._assess_information_density(content))
                accuracy_task = tg.create_task(self._assess_factual_accuracy(content))
                relevance_task = tg.create_task(self._assess_contextual_relevance(content, memory_type))
                uniqueness_task = tg.create_task(self._assess_uniqueness(content))
                importance_task = tg.create_task(self._assess_importance_potential(content, memory_type))
            
            # Collect dimension scores
            dimensions = {
                'clarity': clarity_task.result(),
                'information_density': density_task.result(),
                'factual_accuracy': accuracy_task.result(),
                'contextual_relevance': relevance_task.result(),
                'uniqueness': uniqueness_task.result(),
                'importance_potential': importance_task.result()
            }
            
            # Calculate overall quality score with weighted average
            weights = {
                'clarity': 0.25,
                'information_density': 0.20,
                'factual_accuracy': 0.20,
                'contextual_relevance': 0.15,
                'uniqueness': 0.10,
                'importance_potential': 0.10
            }
            
            overall_score = sum(dimensions[dim] * weights[dim] for dim in dimensions)
            
            # Generate enhancement suggestions
            enhancement_suggestions = await self._generate_enhancement_suggestions(dimensions, content)
            
            # Determine auto-approval based on threshold and confidence
            threshold = self.get_quality_threshold(memory_type)
            confidence = self._calculate_confidence(dimensions, content)
            auto_approve = overall_score >= threshold and confidence >= 0.7
            needs_enhancement = overall_score < threshold
            
            if auto_approve:
                self._auto_approval_count += 1
            
            # Performance tracking
            processing_time = (time.perf_counter() - start_time) * 1000
            self._processing_times.append(processing_time)
            self._assessment_count += 1
            
            assessment = QualityAssessment(
                overall_score=overall_score,
                dimensions=dimensions,
                enhancement_suggestions=enhancement_suggestions,
                confidence=confidence,
                needs_enhancement=needs_enhancement,
                auto_approve=auto_approve
            )
            
            self.logger.debug(
                f"Quality assessment completed: {overall_score:.3f} (threshold: {threshold:.3f})",
                extra={
                    "processing_time_ms": processing_time,
                    "auto_approve": auto_approve,
                    "memory_type": memory_type
                }
            )
            
            return assessment
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Quality assessment failed: {e}", extra={"processing_time_ms": processing_time})
            
            # Return conservative fallback assessment
            return QualityAssessment(
                overall_score=0.5,
                dimensions={dim: 0.5 for dim in ['clarity', 'information_density', 'factual_accuracy', 
                                               'contextual_relevance', 'uniqueness', 'importance_potential']},
                enhancement_suggestions=['Quality assessment failed - manual review recommended'],
                confidence=0.3,
                needs_enhancement=True,
                auto_approve=False
            )
    
    async def _assess_clarity(self, content: str) -> float:
        """Assess content clarity and coherence (0.0-1.0)"""
        score = 0.5  # Base score
        
        # Length considerations
        if 20 <= len(content) <= 2000:
            score += 0.1
        
        # Structure indicators
        sentences = content.split('.')
        if len(sentences) >= 2:
            score += 0.1
        
        # Clear language indicators
        clarity_indicators = ['because', 'therefore', 'specifically', 'for example', 'in other words']
        for indicator in clarity_indicators:
            if indicator.lower() in content.lower():
                score += 0.05
        
        # Complexity penalties
        avg_word_length = mean(len(word) for word in content.split()) if content.split() else 0
        if avg_word_length > 8:
            score -= 0.1
        
        # Grammar and readability (simplified)
        if content.count(',') > len(content) / 50:  # Too many commas
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    async def _assess_information_density(self, content: str) -> float:
        """Assess information density and value (0.0-1.0)"""
        score = 0.4  # Base score
        
        # Content substantiveness
        words = content.split()
        if len(words) >= 10:
            score += 0.2
        
        # Information-rich words
        info_rich_words = ['implement', 'analyze', 'result', 'because', 'discovered', 
                          'achieved', 'measured', 'configured', 'optimized', 'validated']
        info_count = sum(1 for word in words if any(rich in word.lower() for rich in info_rich_words))
        score += min(0.3, info_count * 0.05)
        
        # Specific details (numbers, dates, technical terms)
        has_numbers = bool(re.search(r'\d+', content))
        has_technical_terms = any(term in content.lower() for term in 
                                ['api', 'database', 'function', 'class', 'method', 'algorithm'])
        
        if has_numbers:
            score += 0.1
        if has_technical_terms:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _assess_factual_accuracy(self, content: str) -> float:
        """Assess factual accuracy and reliability (0.0-1.0)"""
        score = 0.7  # Assume generally accurate unless indicators suggest otherwise
        
        # Certainty indicators (positive)
        certainty_indicators = ['confirmed', 'verified', 'tested', 'measured', 'documented']
        for indicator in certainty_indicators:
            if indicator.lower() in content.lower():
                score += 0.05
        
        # Uncertainty indicators (negative)
        uncertainty_indicators = ['maybe', 'possibly', 'might be', 'seems like', 'probably']
        for indicator in uncertainty_indicators:
            if indicator.lower() in content.lower():
                score -= 0.1
        
        # Absolute statements without evidence (risky)
        absolute_indicators = ['always', 'never', 'all', 'none', 'impossible', 'definitely']
        absolute_count = sum(1 for indicator in absolute_indicators if indicator.lower() in content.lower())
        if absolute_count > 0:
            score -= min(0.2, absolute_count * 0.1)
        
        return max(0.0, min(1.0, score))
    
    async def _assess_contextual_relevance(self, content: str, memory_type: str) -> float:
        """Assess contextual relevance (0.0-1.0)"""
        score = 0.6  # Base relevance score
        
        # Memory type specific relevance
        type_keywords = {
            'interaction': ['user', 'request', 'response', 'conversation', 'asked', 'discussed'],
            'milestone': ['completed', 'achieved', 'finished', 'delivered', 'success', 'milestone'],
            'reflection': ['learned', 'realized', 'understood', 'insight', 'analysis', 'conclusion'],
            'technical': ['implemented', 'configured', 'optimized', 'architecture', 'system', 'code'],
            'error': ['error', 'failed', 'issue', 'problem', 'debug', 'fix']
        }
        
        relevant_keywords = type_keywords.get(memory_type, [])
        keyword_matches = sum(1 for keyword in relevant_keywords if keyword.lower() in content.lower())
        score += min(0.3, keyword_matches * 0.1)
        
        # Context indicators
        context_indicators = ['context', 'background', 'related to', 'in connection with', 'regarding']
        for indicator in context_indicators:
            if indicator.lower() in content.lower():
                score += 0.05
        
        return max(0.0, min(1.0, score))
    
    async def _assess_uniqueness(self, content: str) -> float:
        """Assess uniqueness and novelty (0.0-1.0)"""
        score = 0.6  # Assume moderate uniqueness as baseline
        
        # Novel terminology or concepts
        novel_indicators = ['new', 'novel', 'innovative', 'breakthrough', 'discovered', 'first time']
        for indicator in novel_indicators:
            if indicator.lower() in content.lower():
                score += 0.1
        
        # Generic/common phrases (reduce uniqueness)
        generic_phrases = ['hello', 'thank you', 'please help', 'how are you', 'good morning']
        for phrase in generic_phrases:
            if phrase.lower() in content.lower():
                score -= 0.2
        
        # Specific details increase uniqueness
        has_specific_details = any(indicator in content for indicator in [':', '(', ')', '"', "'"])
        if has_specific_details:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _assess_importance_potential(self, content: str, memory_type: str) -> float:
        """Assess long-term importance potential (0.0-1.0)"""
        score = 0.5  # Base importance
        
        # Memory type importance weighting
        type_importance = {
            'milestone': 0.9,
            'technical': 0.8,
            'reflection': 0.7,
            'interaction': 0.5,
            'error': 0.4
        }
        
        base_importance = type_importance.get(memory_type, 0.5)
        score = base_importance
        
        # Importance indicators
        importance_indicators = ['critical', 'important', 'key', 'essential', 'significant', 
                               'breakthrough', 'achievement', 'lesson learned']
        for indicator in importance_indicators:
            if indicator.lower() in content.lower():
                score += 0.05
        
        # Future reference potential
        reference_indicators = ['remember', 'reference', 'future', 'template', 'pattern', 'strategy']
        for indicator in reference_indicators:
            if indicator.lower() in content.lower():
                score += 0.05
        
        return max(0.0, min(1.0, score))
    
    async def _generate_enhancement_suggestions(self, dimensions: Dict[str, float], content: str) -> List[str]:
        """Generate specific enhancement suggestions based on dimension scores"""
        suggestions = []
        
        # Identify lowest scoring dimensions for targeted suggestions
        low_threshold = 0.6
        
        if dimensions['clarity'] < low_threshold:
            suggestions.extend([
                'Add more context and background information',
                'Use clearer, more specific language',
                'Break down complex ideas into smaller parts'
            ])
        
        if dimensions['information_density'] < low_threshold:
            suggestions.extend([
                'Include more specific details and examples',
                'Add quantitative information where relevant',
                'Expand on key concepts and implications'
            ])
        
        if dimensions['factual_accuracy'] < low_threshold:
            suggestions.extend([
                'Verify factual claims and add sources',
                'Qualify uncertain statements appropriately',
                'Avoid absolute statements without evidence'
            ])
        
        if dimensions['contextual_relevance'] < low_threshold:
            suggestions.extend([
                'Connect information to broader context',
                'Explain relevance to current goals/projects',
                'Add background information for better understanding'
            ])
        
        if dimensions['uniqueness'] < low_threshold:
            suggestions.extend([
                'Highlight what makes this information unique',
                'Compare to existing knowledge or approaches',
                'Emphasize novel aspects or insights'
            ])
        
        if dimensions['importance_potential'] < low_threshold:
            suggestions.extend([
                'Explain long-term significance and implications',
                'Discuss potential future applications',
                'Connect to strategic goals or outcomes'
            ])
        
        # Limit suggestions to most relevant ones
        return suggestions[:3] if suggestions else ['Content meets quality standards']
    
    def _calculate_confidence(self, dimensions: Dict[str, float], content: str) -> float:
        """Calculate confidence in quality assessment"""
        # Base confidence on content length and dimension consistency
        base_confidence = 0.7
        
        # Content length confidence
        content_length = len(content)
        if 50 <= content_length <= 1000:
            base_confidence += 0.1
        elif content_length < 20:
            base_confidence -= 0.2
        
        # Dimension consistency (low variance = higher confidence)
        dimension_values = list(dimensions.values())
        variance = sum((x - mean(dimension_values))**2 for x in dimension_values) / len(dimension_values)
        if variance < 0.1:
            base_confidence += 0.1
        elif variance > 0.3:
            base_confidence -= 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    async def enhance_content(self, content: str, assessment: QualityAssessment) -> str:
        """
        Enhance content based on quality assessment suggestions.
        Returns enhanced version of the content.
        """
        if not assessment.needs_enhancement:
            return content
        
        enhanced_content = content
        
        # Apply enhancement based on suggestions (simplified implementation)
        for suggestion in assessment.enhancement_suggestions:
            if 'context' in suggestion.lower():
                enhanced_content = f"[Enhanced with context] {enhanced_content}"
            elif 'specific' in suggestion.lower():
                enhanced_content = f"{enhanced_content} [Additional specificity needed]"
            elif 'verify' in suggestion.lower():
                enhanced_content = f"{enhanced_content} [Verification recommended]"
        
        self._enhancement_count += 1
        return enhanced_content
    
    def get_quality_threshold(self, memory_type: str) -> float:
        """Get quality threshold for automatic approval by memory type"""
        return getattr(self.thresholds, memory_type, 0.6)
    
    async def process_async(self, data: any) -> any:
        """Process data asynchronously for interface compliance"""
        if isinstance(data, tuple) and len(data) == 2:
            content, memory_type = data
            return await self.assess_quality(content, memory_type)
        return await self.assess_quality(str(data))
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for monitoring"""
        avg_processing_time = mean(self._processing_times) if self._processing_times else 0.0
        
        return {
            'total_assessments': self._assessment_count,
            'auto_approvals': self._auto_approval_count,
            'enhancements': self._enhancement_count,
            'auto_approval_rate': self._auto_approval_count / max(1, self._assessment_count),
            'average_processing_time_ms': avg_processing_time,
            'target_processing_time_ms': 3.0,
            'performance_target_met': avg_processing_time < 3.0,
            'thresholds': {
                'interaction': self.thresholds.interaction,
                'milestone': self.thresholds.milestone,
                'reflection': self.thresholds.reflection,
                'technical': self.thresholds.technical,
                'synthetic': self.thresholds.synthetic,
                'error': self.thresholds.error
            }
        }