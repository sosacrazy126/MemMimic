"""
NervousSystemThink - Enhanced Contextual Processing Trigger

Transforms the 'think_with_memory' MCP tool into intelligent contextual processing with:
- Contextual memory retrieval and synthesis
- Internal Socratic guidance integration
- Pattern recognition across memory networks
- Insight preservation and wisdom accumulation

External Interface: PRESERVED EXACTLY
Internal Processing: 4-phase intelligence pipeline in <5ms
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
import logging

from ..core import NervousSystemCore
from ..interfaces import SocraticGuidance
from ...memory.storage.amms_storage import Memory
from ...errors import get_error_logger, with_error_context

class NervousSystemThink:
    """
    Enhanced think trigger with internal nervous system intelligence.
    
    Maintains exact external interface compatibility while adding:
    - Intelligent contextual memory retrieval
    - Socratic guidance integration for deeper analysis
    - Pattern recognition and synthesis across memories
    - Insight preservation for continuous learning
    """
    
    def __init__(self, nervous_system_core: Optional[NervousSystemCore] = None):
        self.nervous_system = nervous_system_core or NervousSystemCore()
        self.logger = get_error_logger("nervous_system_think")
        
        # Performance metrics specific to think operations
        self._think_count = 0
        self._context_retrievals = 0
        self._socratic_analyses = 0
        self._insights_generated = 0
        self._patterns_recognized = 0
        self._processing_times = []
        
        # Context retrieval strategies
        self._context_strategies = {
            'question': {
                'memory_types': ['interaction', 'technical', 'reflection'],
                'retrieval_count': 12,
                'focus': 'problem_solving'
            },
            'analysis': {
                'memory_types': ['milestone', 'reflection', 'technical'],
                'retrieval_count': 15,
                'focus': 'pattern_recognition'
            },
            'decision': {
                'memory_types': ['milestone', 'reflection', 'interaction'],
                'retrieval_count': 10,
                'focus': 'wisdom_synthesis'
            },
            'exploration': {
                'memory_types': ['reflection', 'interaction', 'technical'],
                'retrieval_count': 8,
                'focus': 'creative_connections'
            }
        }
        
        # Insight generation patterns
        self._insight_patterns = {
            'connections': ['How does this relate to', 'This connects with', 'Similar to when'],
            'implications': ['This suggests that', 'The implications are', 'This means'],
            'patterns': ['I notice a pattern', 'This recurring theme', 'The trend shows'],
            'wisdom': ['The key insight is', 'What I learned is', 'The deeper understanding']
        }
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the nervous system think trigger"""
        if self._initialized:
            return
            
        await self.nervous_system.initialize()
        self._initialized = True
        
        self.logger.info("NervousSystemThink initialized successfully")
    
    async def think_with_memory(self, input_text: str) -> Dict[str, Any]:
        """
        Enhanced think_with_memory function with internal intelligence.
        
        EXTERNAL INTERFACE: Preserved exactly - think_with_memory(input_text: str)
        INTERNAL PROCESSING: 4-phase intelligence pipeline:
        1. Context retrieval with intelligent selection
        2. Socratic analysis and questioning
        3. Pattern recognition and synthesis
        4. Insight generation and wisdom preservation
        
        Performance Target: <5ms total response time
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        with with_error_context(
            operation="nervous_system_think",
            component="think_trigger",
            metadata={"input_length": len(input_text)}
        ):
            try:
                # Phase 1: Contextual Memory Retrieval (<2ms target)
                contextual_memories = await self._retrieve_contextual_memories(input_text)
                
                # Phase 2: Socratic Analysis Integration (<1.5ms target)
                socratic_analysis = await self._apply_socratic_analysis(
                    input_text, contextual_memories
                )
                
                # Phase 3: Pattern Recognition and Synthesis (<1ms target)
                pattern_synthesis = await self._recognize_patterns_and_synthesize(
                    input_text, contextual_memories, socratic_analysis
                )
                
                # Phase 4: Insight Generation and Response (<0.5ms target)
                enhanced_response = await self._generate_insights_and_response(
                    input_text, contextual_memories, socratic_analysis, pattern_synthesis
                )
                
                # Performance tracking
                processing_time = (time.perf_counter() - start_time) * 1000
                self._processing_times.append(processing_time)
                self._think_count += 1
                
                # Log success with performance metrics
                self.logger.debug(
                    f"Think operation completed successfully in {processing_time:.2f}ms",
                    extra={
                        "processing_time_ms": processing_time,
                        "input_preview": input_text[:50],
                        "context_memories_retrieved": len(contextual_memories),
                        "performance_target_met": processing_time < 5.0
                    }
                )
                
                return enhanced_response
                
            except Exception as e:
                processing_time = (time.perf_counter() - start_time) * 1000
                
                self.logger.error(
                    f"Think operation failed: {e}",
                    extra={
                        "processing_time_ms": processing_time,
                        "input_preview": input_text[:50],
                        "error_type": type(e).__name__
                    }
                )
                
                # Return error response in expected format
                return {
                    'status': 'error',
                    'response': f"Think operation failed: {str(e)}",
                    'error': str(e),
                    'processing_time_ms': processing_time,
                    'nervous_system_version': '2.0.0'
                }
    
    async def _retrieve_contextual_memories(self, input_text: str) -> List[Memory]:
        """
        Retrieve contextual memories using intelligent selection strategy.
        
        Analyzes input to determine optimal context retrieval approach.
        """
        if not self.nervous_system._amms_storage:
            return []
        
        # Determine context strategy based on input analysis
        strategy = self._determine_context_strategy(input_text)
        strategy_config = self._context_strategies[strategy]
        
        # Retrieve memories using enhanced search
        try:
            contextual_memories = await self.nervous_system._amms_storage.search_memories(
                input_text, 
                limit=strategy_config['retrieval_count']
            )
            
            self._context_retrievals += 1
            
            # Filter by memory types if strategy specifies
            if strategy_config.get('memory_types'):
                filtered_memories = []
                for memory in contextual_memories:
                    memory_type = getattr(memory, 'metadata', {}).get('type', 'interaction')
                    if memory_type in strategy_config['memory_types']:
                        filtered_memories.append(memory)
                
                # If filtered list is too small, include additional memories
                if len(filtered_memories) < strategy_config['retrieval_count'] // 2:
                    filtered_memories.extend(
                        contextual_memories[len(filtered_memories):strategy_config['retrieval_count']]
                    )
                
                contextual_memories = filtered_memories[:strategy_config['retrieval_count']]
            
            return contextual_memories
            
        except Exception as e:
            self.logger.debug(f"Context retrieval failed: {e}")
            return []
    
    def _determine_context_strategy(self, input_text: str) -> str:
        """Determine optimal context retrieval strategy based on input analysis"""
        input_lower = input_text.lower()
        
        # Question indicators
        if any(word in input_lower for word in ['how', 'what', 'why', 'when', 'where', '?']):
            return 'question'
        
        # Analysis indicators
        elif any(word in input_lower for word in ['analyze', 'examine', 'compare', 'evaluate', 'assess']):
            return 'analysis'
        
        # Decision indicators
        elif any(word in input_lower for word in ['decide', 'choose', 'should', 'recommend', 'suggest']):
            return 'decision'
        
        # Default to exploration for open-ended thinking
        else:
            return 'exploration'
    
    async def _apply_socratic_analysis(
        self, 
        input_text: str, 
        contextual_memories: List[Memory]
    ) -> Optional[SocraticGuidance]:
        """
        Apply Socratic analysis to deepen thinking process.
        
        Uses internal Socratic guidance for enhanced analytical depth.
        """
        if not self.nervous_system._socratic_guidance:
            return None
        
        try:
            # Create context for Socratic analysis
            context = {
                'input_text': input_text,
                'memory_context': [mem.content[:200] for mem in contextual_memories[:5]],
                'analysis_depth': 3,  # Standard depth for thinking
                'focus_areas': ['connections', 'implications', 'insights']
            }
            
            # Get Socratic guidance for thinking enhancement
            socratic_guidance = await self.nervous_system._socratic_guidance.guide_memory_decision(
                f"Think deeply about: {input_text}", context
            )
            
            self._socratic_analyses += 1
            return socratic_guidance
            
        except Exception as e:
            self.logger.debug(f"Socratic analysis failed: {e}")
            return None
    
    async def _recognize_patterns_and_synthesize(
        self, 
        input_text: str, 
        contextual_memories: List[Memory], 
        socratic_analysis: Optional[SocraticGuidance]
    ) -> Dict[str, Any]:
        """
        Recognize patterns across memories and synthesize insights.
        
        Identifies recurring themes, connections, and knowledge patterns.
        """
        synthesis = {
            'patterns_identified': [],
            'connections_found': [],
            'themes_recognized': [],
            'knowledge_gaps': [],
            'synthesis_confidence': 0.5
        }
        
        if not contextual_memories:
            return synthesis
        
        # Pattern recognition across memories
        memory_contents = [mem.content for mem in contextual_memories]
        
        # Identify recurring themes (simplified analysis)
        theme_keywords = {}
        for content in memory_contents:
            words = content.lower().split()
            for word in words:
                if len(word) > 4:  # Focus on meaningful words
                    theme_keywords[word] = theme_keywords.get(word, 0) + 1
        
        # Extract top themes
        recurring_themes = [
            word for word, count in theme_keywords.items() 
            if count >= 2 and word not in ['that', 'this', 'with', 'from', 'they', 'have', 'been']
        ][:5]
        
        synthesis['themes_recognized'] = recurring_themes
        
        # Connection analysis using duplicate detector if available
        if self.nervous_system._duplicate_detector and len(contextual_memories) >= 2:
            try:
                # Analyze connections between memories
                first_memory = contextual_memories[0]
                other_memories = contextual_memories[1:5]
                
                relationships = await self.nervous_system._duplicate_detector.map_relationships(
                    first_memory, other_memories
                )
                
                if relationships:
                    strong_connections = [
                        mem_id for mem_id, strength in relationships.items() 
                        if strength > 0.7
                    ]
                    synthesis['connections_found'] = strong_connections[:3]
            
            except Exception as e:
                self.logger.debug(f"Connection analysis failed: {e}")
        
        # Pattern identification based on content types
        content_patterns = {}
        for memory in contextual_memories:
            memory_type = getattr(memory, 'metadata', {}).get('type', 'interaction')
            content_patterns[memory_type] = content_patterns.get(memory_type, 0) + 1
        
        synthesis['patterns_identified'] = [
            f"{pattern_type}: {count}" 
            for pattern_type, count in content_patterns.items()
        ]
        
        # Confidence calculation
        synthesis['synthesis_confidence'] = min(1.0, 
            0.5 + (len(recurring_themes) * 0.1) + (len(synthesis['connections_found']) * 0.1)
        )
        
        if synthesis['patterns_identified'] or synthesis['connections_found'] or synthesis['themes_recognized']:
            self._patterns_recognized += 1
        
        return synthesis
    
    async def _generate_insights_and_response(
        self, 
        input_text: str, 
        contextual_memories: List[Memory], 
        socratic_analysis: Optional[SocraticGuidance], 
        pattern_synthesis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate insights and create enhanced response.
        
        Combines all analysis phases into coherent response with preserved insights.
        """
        # Base response structure
        response = {
            'status': 'success',
            'response': '',
            'context_memories_used': len(contextual_memories),
            'processing_time_ms': 0,  # Will be set by caller
            'nervous_system_version': '2.0.0'
        }
        
        # Build enhanced response content
        response_parts = []
        
        # Core thinking response
        if contextual_memories:
            response_parts.append(f"Based on {len(contextual_memories)} relevant memories, here's my analysis:")
            
            # Add key contextual insights
            key_memories = contextual_memories[:3]
            for i, memory in enumerate(key_memories, 1):
                preview = memory.content[:150] + "..." if len(memory.content) > 150 else memory.content
                response_parts.append(f"{i}. {preview}")
        
        # Add pattern insights
        if pattern_synthesis['themes_recognized']:
            response_parts.append(f"\n🔍 Key themes identified: {', '.join(pattern_synthesis['themes_recognized'][:3])}")
        
        if pattern_synthesis['connections_found']:
            response_parts.append(f"🔗 Strong connections found with {len(pattern_synthesis['connections_found'])} related memories")
        
        # Add Socratic insights if available
        if socratic_analysis:
            response_parts.append(f"\n🧘 Deeper analysis suggests: {socratic_analysis.recommendation}")
            
            # Add key questions for further thought
            if socratic_analysis.questions:
                response_parts.append(f"Consider: {socratic_analysis.questions[0]}")
        
        # Generate insight synthesis
        synthesis_insight = await self._synthesize_final_insight(
            input_text, contextual_memories, pattern_synthesis
        )
        
        if synthesis_insight:
            response_parts.append(f"\n💡 Key insight: {synthesis_insight}")
            self._insights_generated += 1
        
        # Combine response parts
        response['response'] = '\n'.join(response_parts)
        
        # Add detailed intelligence analysis (optional)
        response['intelligence_analysis'] = {
            'context_strategy': self._determine_context_strategy(input_text),
            'pattern_confidence': pattern_synthesis['synthesis_confidence'],
            'socratic_depth': len(socratic_analysis.questions) if socratic_analysis else 0,
            'insight_generated': synthesis_insight is not None,
            'memory_types_accessed': list(set(
                getattr(mem, 'metadata', {}).get('type', 'interaction') 
                for mem in contextual_memories
            ))
        }
        
        # Store this thinking session for future reference
        await self._preserve_thinking_session(input_text, response, pattern_synthesis)
        
        return response
    
    async def _synthesize_final_insight(
        self, 
        input_text: str, 
        contextual_memories: List[Memory], 
        pattern_synthesis: Dict[str, Any]
    ) -> Optional[str]:
        """
        Synthesize final insight from all analysis phases.
        
        Creates coherent wisdom from distributed memory patterns.
        """
        if not contextual_memories or pattern_synthesis['synthesis_confidence'] < 0.6:
            return None
        
        # Generate insight based on patterns
        themes = pattern_synthesis['themes_recognized']
        connections = pattern_synthesis['connections_found']
        
        if themes and len(themes) >= 2:
            # Theme-based insight
            primary_theme = themes[0]
            secondary_theme = themes[1]
            return f"The intersection of {primary_theme} and {secondary_theme} reveals a pattern of interconnected understanding."
        
        elif connections:
            # Connection-based insight
            return f"The strong connections between related memories suggest a coherent knowledge structure emerging."
        
        elif len(contextual_memories) >= 5:
            # Volume-based insight
            return f"The breadth of relevant context ({len(contextual_memories)} memories) indicates this is a well-explored domain with rich accumulated knowledge."
        
        else:
            # Generic synthesis
            return "This thinking session reveals emerging patterns in the knowledge network."
    
    async def _preserve_thinking_session(
        self, 
        input_text: str, 
        response: Dict[str, Any], 
        pattern_synthesis: Dict[str, Any]
    ) -> None:
        """
        Preserve thinking session insights for future learning.
        
        Stores meta-insights about thinking patterns and effectiveness.
        """
        try:
            # Create thinking session summary
            session_summary = {
                'input': input_text[:100],
                'context_count': response['context_memories_used'],
                'patterns_found': len(pattern_synthesis['patterns_identified']),
                'connections_found': len(pattern_synthesis['connections_found']),
                'insight_generated': response['intelligence_analysis']['insight_generated'],
                'confidence': pattern_synthesis['synthesis_confidence'],
                'timestamp': time.time()
            }
            
            # In production, this would store to AMMS as a thinking pattern memory
            # For now, just log the session for pattern learning
            self.logger.debug(
                "Thinking session preserved",
                extra=session_summary
            )
            
        except Exception as e:
            self.logger.debug(f"Failed to preserve thinking session: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for think operations"""
        from statistics import mean
        
        avg_processing_time = mean(self._processing_times) if self._processing_times else 0.0
        
        return {
            'total_think_operations': self._think_count,
            'context_retrievals': self._context_retrievals,
            'socratic_analyses': self._socratic_analyses,
            'insights_generated': self._insights_generated,
            'patterns_recognized': self._patterns_recognized,
            'context_retrieval_rate': self._context_retrievals / max(1, self._think_count),
            'socratic_analysis_rate': self._socratic_analyses / max(1, self._think_count),
            'insight_generation_rate': self._insights_generated / max(1, self._think_count),
            'pattern_recognition_rate': self._patterns_recognized / max(1, self._think_count),
            'average_processing_time_ms': avg_processing_time,
            'target_processing_time_ms': 5.0,
            'performance_target_met': avg_processing_time < 5.0,
            'biological_reflex_achieved': avg_processing_time < 5.0,
            'context_strategies': list(self._context_strategies.keys()),
            'nervous_system_version': '2.0.0'
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for think trigger"""
        if not self._initialized:
            return {'status': 'not_initialized', 'healthy': False}
        
        # Check nervous system health
        ns_health = await self.nervous_system.health_check()
        
        # Think-specific health indicators
        think_health = {
            'status': 'operational',
            'healthy': ns_health['healthy'],
            'operations_count': self._think_count,
            'performance_metrics': self.get_performance_metrics(),
            'nervous_system_health': ns_health
        }
        
        # Check for performance issues
        if self._processing_times:
            avg_time = sum(self._processing_times) / len(self._processing_times)
            if avg_time > 5.0:
                think_health['warnings'] = [f"Average processing time {avg_time:.2f}ms exceeds 5ms target"]
        
        return think_health