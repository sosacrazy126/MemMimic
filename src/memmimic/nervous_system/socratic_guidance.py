"""
InternalSocraticGuidance - Wisdom-based Decision Making

Replaces external guided tools (update_memory_guided, delete_memory_guided)
with internal Socratic questioning framework and contextual decision wisdom.

Questioning Framework: What? Why? How? What if? So what?
Decision Engine: Context-aware guidance with impact analysis and alternatives
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from statistics import mean

from .interfaces import SocraticGuidanceInterface, SocraticGuidance
from ..memory.storage.amms_storage import Memory
from ..errors import get_error_logger

class QuestioningDepth(Enum):
    """Socratic questioning depth levels"""
    SURFACE = 1      # Basic what/why questions
    ANALYTICAL = 3   # Standard depth for most scenarios  
    DEEP = 5         # Complex analysis requiring thorough exploration

@dataclass
class DecisionContext:
    """Context for Socratic decision making"""
    memory_type: str
    user_patterns: Dict[str, Any]
    system_health: Dict[str, Any]
    related_memories: List[Memory]
    temporal_context: Dict[str, Any]

class InternalSocraticGuidance(SocraticGuidanceInterface):
    """
    Internal Socratic guidance system providing wisdom-based decisions.
    
    Uses progressive questioning framework and contextual awareness
    to guide memory management decisions without external intervention.
    """
    
    def __init__(self, default_depth: QuestioningDepth = QuestioningDepth.ANALYTICAL):
        self.default_depth = default_depth
        self.logger = get_error_logger("socratic_guidance")
        
        # Socratic questioning patterns
        self.questioning_patterns = {
            'what': [
                'What is the core content/essence of this?',
                'What makes this significant or unique?',
                'What are the key elements present?',
                'What context is missing or unclear?'
            ],
            'why': [
                'Why is this memory important to preserve?',
                'Why might this information be valuable later?',
                'Why does this deserve cognitive resources?',
                'Why would someone need to reference this?'
            ],
            'how': [
                'How does this connect to existing knowledge?',
                'How might this information be used in the future?',
                'How does this fit into broader patterns or goals?',
                'How can the quality or usefulness be improved?'
            ],
            'what_if': [
                'What if this information becomes more/less relevant?',
                'What if similar situations arise in the future?',
                'What if this memory is lost or forgotten?',
                'What if the context changes significantly?'
            ],
            'so_what': [
                'So what is the broader significance of this?',
                'So what are the implications for future decisions?',
                'So what action should be taken based on this?',
                'So what would be the consequences of different choices?'
            ]
        }
        
        # Decision wisdom patterns learned from experience
        self.wisdom_patterns = {
            'memory_preservation': {
                'high_value_indicators': [
                    'unique insights', 'problem solutions', 'lessons learned',
                    'breakthrough moments', 'error resolutions', 'strategic decisions'
                ],
                'low_value_indicators': [
                    'routine confirmations', 'redundant information', 'temporary status',
                    'outdated details', 'trivial interactions', 'noise content'
                ]
            },
            'update_guidance': {
                'enhancement_scenarios': [
                    'new insights available', 'context has changed', 'errors discovered',
                    'additional details emerged', 'clarity can be improved'
                ],
                'preservation_scenarios': [
                    'complete and accurate', 'historical significance', 'reference value',
                    'well-structured content', 'optimal detail level'
                ]
            },
            'deletion_guidance': {
                'safe_deletion_indicators': [
                    'completely obsolete', 'proven incorrect', 'replaced by better version',
                    'privacy concerns', 'low quality with no improvement potential'
                ],
                'preserve_indicators': [
                    'historical value', 'learning experience', 'reference potential',
                    'unique perspective', 'part of important sequence'
                ]
            }
        }
        
        # Performance metrics
        self._guidance_count = 0
        self._update_guidances = 0
        self._deletion_guidances = 0
        self._decision_guidances = 0
        self._processing_times = []
        
        # Guidance cache for similar scenarios
        self._guidance_cache = {}
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Socratic guidance system"""
        if self._initialized:
            return
        
        self.logger.info("Initializing InternalSocraticGuidance")
        
        # Load wisdom patterns and decision trees
        await self._load_wisdom_patterns()
        
        self._initialized = True
        self.logger.info("InternalSocraticGuidance initialization complete")
    
    async def _load_wisdom_patterns(self) -> None:
        """Load wisdom patterns and decision trees"""
        # Simulate loading time for realistic performance
        await asyncio.sleep(0.001)  # 1ms simulation
        
        # In production, this would:
        # - Load learned wisdom patterns from storage
        # - Initialize decision trees from experience
        # - Set up contextual awareness models
        # - Load user preference patterns
        pass
    
    async def guide_memory_decision(self, content: str, context: Dict[str, Any]) -> SocraticGuidance:
        """
        Provide Socratic guidance for general memory decisions.
        
        Uses progressive questioning and contextual wisdom to guide
        memory storage, enhancement, and management decisions.
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            # Extract decision context
            memory_type = context.get('memory_type', 'interaction')
            depth = QuestioningDepth(context.get('questioning_depth', self.default_depth.value))
            
            # Generate Socratic questions
            questions = await self._generate_questions(content, memory_type, depth)
            
            # Apply wisdom-based analysis
            wisdom_analysis = await self._apply_wisdom_analysis(content, context, questions)
            
            # Generate guidance recommendation
            recommendation = await self._generate_recommendation(content, context, wisdom_analysis)
            
            # Provide reasoning and alternatives
            reasoning = await self._generate_reasoning(wisdom_analysis, context)
            alternatives = await self._generate_alternatives(recommendation, context)
            
            # Calculate confidence based on analysis quality
            confidence = self._calculate_guidance_confidence(wisdom_analysis, context)
            
            # Impact analysis for decision consequences
            impact_analysis = await self._analyze_decision_impact(recommendation, context)
            
            # Performance tracking
            processing_time = (time.perf_counter() - start_time) * 1000
            self._processing_times.append(processing_time)
            self._guidance_count += 1
            self._decision_guidances += 1
            
            guidance = SocraticGuidance(
                recommendation=recommendation,
                reasoning=reasoning,
                questions=questions,
                confidence=confidence,
                alternatives=alternatives,
                impact_analysis=impact_analysis
            )
            
            self.logger.debug(
                f"Memory decision guidance completed: {recommendation}",
                extra={
                    "processing_time_ms": processing_time,
                    "confidence": confidence,
                    "memory_type": memory_type
                }
            )
            
            return guidance
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Memory decision guidance failed: {e}")
            
            # Return conservative fallback guidance
            return SocraticGuidance(
                recommendation="Store with standard quality assessment",
                reasoning=["Socratic analysis encountered an error", "Defaulting to conservative approach"],
                questions=["What information is most important to preserve?"],
                confidence=0.5,
                alternatives=["Manual review recommended", "Apply standard quality filters"],
                impact_analysis={'error': str(e), 'fallback_mode': True}
            )
    
    async def guide_memory_update(self, memory_id: str, proposed_update: str) -> SocraticGuidance:
        """
        Guide memory update decisions with Socratic analysis.
        
        Replaces update_memory_guided with internal wisdom-based guidance.
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            # Simulate retrieving existing memory (would use AMMS in production)
            existing_memory = await self._get_memory_by_id(memory_id)
            
            if not existing_memory:
                return SocraticGuidance(
                    recommendation="Memory not found - cannot provide update guidance",
                    reasoning=["Memory ID does not exist in system"],
                    questions=["Does the memory ID exist?", "Has the memory been deleted?"],
                    confidence=0.9,
                    alternatives=["Verify memory ID", "Create new memory instead"],
                    impact_analysis={'status': 'memory_not_found'}
                )
            
            # Socratic questioning for update decision
            questions = [
                f"What specific improvements does this update provide?",
                f"Why is the current version insufficient?",
                f"How will this update change the memory's value or meaning?",
                f"What if the original context is lost in the update?",
                f"So what is the best way to preserve both old and new information?"
            ]
            
            # Wisdom-based update analysis
            update_analysis = await self._analyze_update_wisdom(
                existing_memory.content, proposed_update
            )
            
            # Generate update recommendation
            if update_analysis['enhancement_value'] > 0.7:
                recommendation = "Update recommended - significant improvement identified"
            elif update_analysis['preservation_risk'] > 0.6:
                recommendation = "Update with caution - preserve original context"
            else:
                recommendation = "Update not recommended - minimal benefit with preservation risk"
            
            reasoning = [
                f"Enhancement value: {update_analysis['enhancement_value']:.2f}",
                f"Preservation risk: {update_analysis['preservation_risk']:.2f}",
                f"Context alignment: {update_analysis['context_alignment']:.2f}"
            ]
            
            alternatives = [
                "Create new related memory instead of updating",
                "Merge both versions with clear attribution",
                "Add update as memory enhancement note"
            ]
            
            confidence = min(0.9, update_analysis['enhancement_value'] + 0.3)
            
            impact_analysis = {
                'original_memory_preservation': update_analysis['preservation_risk'],
                'information_enhancement': update_analysis['enhancement_value'],
                'context_preservation': update_analysis['context_alignment']
            }
            
            # Performance tracking
            processing_time = (time.perf_counter() - start_time) * 1000
            self._processing_times.append(processing_time)
            self._guidance_count += 1
            self._update_guidances += 1
            
            guidance = SocraticGuidance(
                recommendation=recommendation,
                reasoning=reasoning,
                questions=questions,
                confidence=confidence,
                alternatives=alternatives,
                impact_analysis=impact_analysis
            )
            
            self.logger.debug(
                f"Memory update guidance completed: {recommendation}",
                extra={"processing_time_ms": processing_time, "memory_id": memory_id}
            )
            
            return guidance
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Memory update guidance failed: {e}")
            
            return SocraticGuidance(
                recommendation="Update guidance unavailable - manual review required",
                reasoning=[f"Analysis error: {str(e)}"],
                questions=["Is the update truly necessary?"],
                confidence=0.3,
                alternatives=["Skip update", "Manual analysis required"],
                impact_analysis={'error': str(e)}
            )
    
    async def guide_memory_deletion(self, memory_id: str, reason: Optional[str] = None) -> SocraticGuidance:
        """
        Guide memory deletion decisions with impact analysis.
        
        Replaces delete_memory_guided with internal wisdom-based guidance.
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        try:
            # Simulate retrieving existing memory (would use AMMS in production)
            existing_memory = await self._get_memory_by_id(memory_id)
            
            if not existing_memory:
                return SocraticGuidance(
                    recommendation="Memory not found - deletion not possible",
                    reasoning=["Memory ID does not exist in system"],
                    questions=["Does the memory ID exist?", "Has it already been deleted?"],
                    confidence=0.9,
                    alternatives=["Verify memory ID", "Check deletion history"],
                    impact_analysis={'status': 'memory_not_found'}
                )
            
            # Socratic questioning for deletion decision
            questions = [
                "What value does this memory currently provide?",
                "Why is deletion being considered?",
                "How might this information be needed in the future?",
                "What if this memory contains unique insights or context?",
                "So what are the consequences of permanent removal?"
            ]
            
            # Wisdom-based deletion analysis
            deletion_analysis = await self._analyze_deletion_wisdom(
                existing_memory.content, reason or "No reason provided"
            )
            
            # Generate deletion recommendation
            if deletion_analysis['deletion_safety'] > 0.8:
                recommendation = "Deletion approved - low risk of information loss"
            elif deletion_analysis['preservation_value'] > 0.7:
                recommendation = "Deletion not recommended - high preservation value"
            else:
                recommendation = "Deletion with caution - consider archiving instead"
            
            reasoning = [
                f"Deletion safety score: {deletion_analysis['deletion_safety']:.2f}",
                f"Preservation value: {deletion_analysis['preservation_value']:.2f}",
                f"Future reference potential: {deletion_analysis['reference_potential']:.2f}",
                f"Reason analysis: {deletion_analysis['reason_validity']:.2f}"
            ]
            
            alternatives = [
                "Archive instead of delete",
                "Mark as low priority but preserve",
                "Move to separate storage for potential future reference"
            ]
            
            confidence = deletion_analysis['deletion_safety']
            
            impact_analysis = {
                'information_loss_risk': 1.0 - deletion_analysis['deletion_safety'],
                'preservation_value_lost': deletion_analysis['preservation_value'],
                'future_reference_impact': deletion_analysis['reference_potential'],
                'reason_justification': deletion_analysis['reason_validity']
            }
            
            # Performance tracking  
            processing_time = (time.perf_counter() - start_time) * 1000
            self._processing_times.append(processing_time)
            self._guidance_count += 1
            self._deletion_guidances += 1
            
            guidance = SocraticGuidance(
                recommendation=recommendation,
                reasoning=reasoning,
                questions=questions,
                confidence=confidence,
                alternatives=alternatives,
                impact_analysis=impact_analysis
            )
            
            self.logger.debug(
                f"Memory deletion guidance completed: {recommendation}",
                extra={"processing_time_ms": processing_time, "memory_id": memory_id}
            )
            
            return guidance
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            self.logger.error(f"Memory deletion guidance failed: {e}")
            
            return SocraticGuidance(
                recommendation="Deletion not recommended due to analysis error",
                reasoning=[f"Analysis error: {str(e)}", "Conservative approach: preserve memory"],
                questions=["Is deletion truly necessary?"],
                confidence=0.2,
                alternatives=["Skip deletion", "Manual review required"],
                impact_analysis={'error': str(e), 'conservative_approach': True}
            )
    
    async def _generate_questions(self, content: str, memory_type: str, depth: QuestioningDepth) -> List[str]:
        """Generate Socratic questions based on content and depth"""
        questions = []
        
        # Select question categories based on depth
        if depth == QuestioningDepth.SURFACE:
            categories = ['what', 'why']
        elif depth == QuestioningDepth.ANALYTICAL:
            categories = ['what', 'why', 'how']
        else:  # DEEP
            categories = ['what', 'why', 'how', 'what_if', 'so_what']
        
        # Generate questions from each category
        for category in categories:
            category_questions = self.questioning_patterns[category]
            # Select most relevant question (simplified selection)
            questions.append(category_questions[0])
        
        return questions
    
    async def _apply_wisdom_analysis(self, content: str, context: Dict[str, Any], questions: List[str]) -> Dict[str, Any]:
        """Apply wisdom patterns to analyze content and context"""
        analysis = {
            'value_indicators': 0,
            'preservation_factors': 0,
            'enhancement_potential': 0,
            'context_relevance': 0
        }
        
        # Analyze value indicators
        high_value_count = sum(1 for indicator in self.wisdom_patterns['memory_preservation']['high_value_indicators']
                              if indicator.lower() in content.lower())
        low_value_count = sum(1 for indicator in self.wisdom_patterns['memory_preservation']['low_value_indicators']
                             if indicator.lower() in content.lower())
        
        analysis['value_indicators'] = max(0, (high_value_count - low_value_count) / 6.0)
        
        # Analyze preservation factors
        memory_type = context.get('memory_type', 'interaction')
        type_weights = {'milestone': 0.9, 'technical': 0.8, 'reflection': 0.7, 'interaction': 0.5}
        analysis['preservation_factors'] = type_weights.get(memory_type, 0.5)
        
        # Analyze enhancement potential
        analysis['enhancement_potential'] = 0.7 if len(content) < 100 else 0.5
        
        # Analyze context relevance
        analysis['context_relevance'] = 0.8 if len(content.split()) >= 10 else 0.4
        
        return analysis
    
    async def _generate_recommendation(self, content: str, context: Dict[str, Any], wisdom_analysis: Dict[str, Any]) -> str:
        """Generate recommendation based on Socratic analysis"""
        value_score = wisdom_analysis['value_indicators']
        preservation_score = wisdom_analysis['preservation_factors']
        
        overall_score = (value_score + preservation_score) / 2
        
        if overall_score >= 0.7:
            return "Store memory with high priority - significant value identified"
        elif overall_score >= 0.5:
            return "Store memory with standard processing - moderate value"
        else:
            return "Consider enhancement before storage - limited current value"
    
    async def _generate_reasoning(self, wisdom_analysis: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate reasoning based on wisdom analysis"""
        reasoning = []
        
        if wisdom_analysis['value_indicators'] > 0.6:
            reasoning.append("High-value content indicators detected")
        
        if wisdom_analysis['preservation_factors'] > 0.7:
            reasoning.append("Memory type indicates high preservation value")
        
        if wisdom_analysis['enhancement_potential'] > 0.6:
            reasoning.append("Content has potential for quality enhancement")
        
        if not reasoning:
            reasoning.append("Standard memory processing recommended")
        
        return reasoning[:3]  # Limit to top 3 reasons
    
    async def _generate_alternatives(self, recommendation: str, context: Dict[str, Any]) -> List[str]:
        """Generate alternative approaches based on recommendation"""
        alternatives = []
        
        if "high priority" in recommendation:
            alternatives = [
                "Store with enhanced metadata for easy retrieval",
                "Create cross-references to related memories",
                "Mark for periodic review and validation"
            ]
        elif "enhancement" in recommendation:
            alternatives = [
                "Request additional context before storage", 
                "Store as draft for later completion",
                "Combine with related information for richer content"
            ]
        else:
            alternatives = [
                "Apply standard quality filters",
                "Store with minimal metadata",
                "Consider batch processing with similar content"
            ]
        
        return alternatives
    
    async def _analyze_update_wisdom(self, original_content: str, proposed_update: str) -> Dict[str, float]:
        """Analyze wisdom factors for memory updates"""
        # Simplified analysis - production would be more sophisticated
        return {
            'enhancement_value': 0.7 if len(proposed_update) > len(original_content) else 0.4,
            'preservation_risk': 0.3 if 'enhance' in proposed_update.lower() else 0.6,
            'context_alignment': 0.8 if any(word in proposed_update.lower() for word in original_content.lower().split()[:5]) else 0.4
        }
    
    async def _analyze_deletion_wisdom(self, content: str, reason: str) -> Dict[str, float]:
        """Analyze wisdom factors for memory deletion"""
        # Simplified analysis - production would be more sophisticated
        safety_indicators = ['obsolete', 'incorrect', 'replaced', 'duplicate']
        preserve_indicators = ['unique', 'historical', 'reference', 'learning']
        
        deletion_safety = 0.8 if any(indicator in reason.lower() for indicator in safety_indicators) else 0.3
        preservation_value = 0.8 if any(indicator in content.lower() for indicator in preserve_indicators) else 0.4
        
        return {
            'deletion_safety': deletion_safety,
            'preservation_value': preservation_value,
            'reference_potential': 0.6 if len(content) > 100 else 0.3,
            'reason_validity': 0.8 if len(reason) > 10 else 0.4
        }
    
    async def _get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Simulate memory retrieval (would use AMMS in production)"""
        # Simulate memory retrieval
        if memory_id.startswith('mem_'):
            return Memory(content=f"Sample memory content for {memory_id}")
        return None
    
    def _calculate_guidance_confidence(self, wisdom_analysis: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate confidence in guidance recommendation"""
        base_confidence = 0.7
        
        # Adjust based on analysis quality
        value_factor = wisdom_analysis.get('value_indicators', 0.5) * 0.2
        context_factor = wisdom_analysis.get('context_relevance', 0.5) * 0.1
        
        return min(0.95, base_confidence + value_factor + context_factor)
    
    async def _analyze_decision_impact(self, recommendation: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of decision recommendation"""
        return {
            'memory_value_impact': 0.8 if 'high priority' in recommendation else 0.5,
            'resource_usage_impact': 0.3 if 'enhancement' in recommendation else 0.1,
            'future_reference_impact': 0.7 if 'store' in recommendation else 0.2,
            'system_performance_impact': 0.1  # Minimal impact expected
        }
    
    def get_wisdom_patterns(self) -> Dict[str, Any]:
        """Get learned wisdom patterns for decision making"""
        return {
            'questioning_patterns': self.questioning_patterns,
            'wisdom_patterns': self.wisdom_patterns,
            'usage_statistics': {
                'total_guidances': self._guidance_count,
                'update_guidances': self._update_guidances,
                'deletion_guidances': self._deletion_guidances,
                'decision_guidances': self._decision_guidances
            }
        }
    
    async def process_async(self, data: Any) -> Any:
        """Process data asynchronously for interface compliance"""
        if isinstance(data, tuple) and len(data) == 2:
            content, context = data
            return await self.guide_memory_decision(content, context)
        return await self.guide_memory_decision(str(data), {})
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for monitoring"""
        avg_processing_time = mean(self._processing_times) if self._processing_times else 0.0
        
        return {
            'total_guidances': self._guidance_count,
            'update_guidances': self._update_guidances,
            'deletion_guidances': self._deletion_guidances,
            'decision_guidances': self._decision_guidances,
            'average_processing_time_ms': avg_processing_time,
            'target_processing_time_ms': 5.0,
            'performance_target_met': avg_processing_time < 5.0,
            'cache_size': len(self._guidance_cache),
            'wisdom_patterns_loaded': len(self.wisdom_patterns),
            'questioning_depth_default': self.default_depth.value
        }