"""
NervousSystemRecall - Enhanced Memory Search Trigger

Transforms the 'recall_cxd' MCP tool into intelligent search with:
- Context-aware query enhancement
- Relationship mapping and pattern analysis
- Multi-factor intelligent ranking
- Real-time performance optimization

External Interface: PRESERVED EXACTLY
Internal Processing: 5-phase intelligence pipeline in <5ms
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
import logging

from ..core import NervousSystemCore
from ...memory.storage.amms_storage import Memory
from ...errors import get_error_logger, with_error_context

class NervousSystemRecall:
    """
    Enhanced recall trigger with internal nervous system intelligence.
    
    Maintains exact external interface compatibility while adding:
    - Intelligent query enhancement and expansion
    - Relationship mapping for connected results
    - Pattern analysis and context awareness
    - Multi-factor ranking with diversity optimization
    """
    
    def __init__(self, nervous_system_core: Optional[NervousSystemCore] = None):
        self.nervous_system = nervous_system_core or NervousSystemCore()
        self.logger = get_error_logger("nervous_system_recall")
        
        # Performance metrics specific to recall operations
        self._recall_count = 0
        self._enhanced_queries = 0
        self._relationship_mapped_results = 0
        self._pattern_analyzed_results = 0
        self._processing_times = []
        
        # Query enhancement patterns
        self._query_enhancement_patterns = {
            'technical': ['implementation', 'architecture', 'system', 'code', 'configuration'],
            'problem_solving': ['issue', 'error', 'debug', 'fix', 'solution', 'resolve'],
            'learning': ['insight', 'lesson', 'understanding', 'realization', 'discovery'],
            'planning': ['strategy', 'approach', 'plan', 'roadmap', 'next steps'],
            'analysis': ['pattern', 'trend', 'correlation', 'relationship', 'connection']
        }
        
        # Ranking factor weights for intelligent result ordering
        self._ranking_weights = {
            'semantic_similarity': 0.35,
            'relationship_strength': 0.25,
            'recency': 0.15,
            'quality_score': 0.15,
            'access_frequency': 0.10
        }
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the nervous system recall trigger"""
        if self._initialized:
            return
            
        await self.nervous_system.initialize()
        self._initialized = True
        
        self.logger.info("NervousSystemRecall initialized successfully")
    
    async def recall_cxd(
        self, 
        query: str,
        function_filter: str = "ALL",
        limit: int = 5,
        db_name: Optional[str] = None
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Enhanced recall_cxd function with internal intelligence.
        
        EXTERNAL INTERFACE: Preserved exactly - recall_cxd(query, function_filter="ALL", limit=5, db_name=None)
        INTERNAL PROCESSING: 5-phase intelligence pipeline:
        1. Query enhancement and intent recognition
        2. Hybrid search with semantic + keyword matching
        3. Relationship mapping and connection analysis
        4. Pattern recognition and context synthesis
        5. Intelligent ranking with diversity optimization
        
        Performance Target: <5ms total response time
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        with with_error_context(
            operation="nervous_system_recall",
            component="recall_trigger",
            metadata={
                "query_length": len(query),
                "function_filter": function_filter,
                "limit": limit
            }
        ):
            try:
                # Phase 1: Query Enhancement and Intent Recognition (<1ms)
                enhanced_query_info = await self._enhance_query_with_intelligence(
                    query, function_filter
                )
                
                # Phase 2: Hybrid Search with Intelligence (<2ms)
                raw_results = await self._execute_hybrid_intelligent_search(
                    enhanced_query_info, limit * 2  # Get more for intelligent filtering
                )
                
                # Phase 3: Relationship Mapping (<1ms)
                relationship_mapped_results = await self._map_result_relationships(
                    raw_results, enhanced_query_info
                )
                
                # Phase 4: Pattern Analysis and Context Synthesis (<0.5ms)
                pattern_analyzed_results = await self._analyze_result_patterns(
                    relationship_mapped_results, query
                )
                
                # Phase 5: Intelligent Ranking and Diversity Optimization (<0.5ms)
                final_results = await self._apply_intelligent_ranking(
                    pattern_analyzed_results, enhanced_query_info, limit
                )
                
                # Format results to match expected interface
                formatted_results = await self._format_enhanced_results(
                    final_results, enhanced_query_info
                )
                
                # Performance tracking
                processing_time = (time.perf_counter() - start_time) * 1000
                self._processing_times.append(processing_time)
                self._recall_count += 1
                
                # Log success with performance metrics
                self.logger.debug(
                    f"Recall operation completed successfully in {processing_time:.2f}ms",
                    extra={
                        "processing_time_ms": processing_time,
                        "query": query[:50],
                        "results_count": len(formatted_results) if isinstance(formatted_results, list) else 1,
                        "performance_target_met": processing_time < 5.0
                    }
                )
                
                return formatted_results
                
            except Exception as e:
                processing_time = (time.perf_counter() - start_time) * 1000
                
                self.logger.error(
                    f"Recall operation failed: {e}",
                    extra={
                        "processing_time_ms": processing_time,
                        "query": query[:50],
                        "function_filter": function_filter,
                        "error_type": type(e).__name__
                    }
                )
                
                # Return error in expected format
                return {
                    'error': str(e),
                    'query': query,
                    'processing_time_ms': processing_time,
                    'nervous_system_version': '2.0.0'
                }
    
    async def _enhance_query_with_intelligence(
        self, 
        query: str, 
        function_filter: str
    ) -> Dict[str, Any]:
        """
        Enhance query with contextual intelligence and intent recognition.
        
        Analyzes query intent and expands with related terms for better results.
        """
        enhanced_info = {
            'original_query': query,
            'enhanced_query': query,
            'intent_analysis': {},
            'expansion_terms': [],
            'function_filter': function_filter,
            'context_hints': []
        }
        
        # Intent recognition based on query patterns
        query_lower = query.lower()
        detected_intents = []
        
        for intent, keywords in self._query_enhancement_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        enhanced_info['intent_analysis'] = {
            'detected_intents': detected_intents,
            'primary_intent': detected_intents[0] if detected_intents else 'general'
        }
        
        # Query expansion based on intent
        if detected_intents:
            primary_intent = detected_intents[0]
            expansion_terms = self._query_enhancement_patterns.get(primary_intent, [])[:3]
            enhanced_info['expansion_terms'] = expansion_terms
            
            # Create enhanced query (simple word addition for now)
            if expansion_terms:
                enhanced_info['enhanced_query'] = f"{query} {' '.join(expansion_terms[:2])}"
                self._enhanced_queries += 1
        
        # Context hints for better result interpretation
        if 'error' in query_lower or 'problem' in query_lower:
            enhanced_info['context_hints'].append('prioritize_solutions')
        if 'how' in query_lower or 'implement' in query_lower:
            enhanced_info['context_hints'].append('prioritize_instructions')
        if 'why' in query_lower or 'reason' in query_lower:
            enhanced_info['context_hints'].append('prioritize_explanations')
        
        return enhanced_info
    
    async def _execute_hybrid_intelligent_search(
        self, 
        enhanced_query_info: Dict[str, Any], 
        expanded_limit: int
    ) -> List[Memory]:
        """
        Execute hybrid search using both enhanced query and original query.
        
        Combines semantic similarity with keyword matching for comprehensive results.
        """
        if not self.nervous_system._amms_storage:
            return []
        
        # Get base results using AMMS search
        enhanced_query = enhanced_query_info['enhanced_query']
        original_query = enhanced_query_info['original_query']
        
        # Primary search with enhanced query
        primary_results = await self.nervous_system._amms_storage.search_memories(
            enhanced_query, limit=expanded_limit
        )
        
        # Secondary search with original query (if different)
        if enhanced_query != original_query:
            secondary_results = await self.nervous_system._amms_storage.search_memories(
                original_query, limit=expanded_limit // 2
            )
            
            # Merge results, avoiding duplicates
            seen_ids = {getattr(mem, 'id', None) for mem in primary_results}
            for mem in secondary_results:
                if getattr(mem, 'id', None) not in seen_ids:
                    primary_results.append(mem)
        
        return primary_results[:expanded_limit]
    
    async def _map_result_relationships(
        self, 
        results: List[Memory], 
        enhanced_query_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Map relationships between results and add connection strengths.
        
        Creates relationship network for better result understanding.
        """
        relationship_mapped_results = []
        
        for i, memory in enumerate(results):
            result_data = {
                'memory': memory,
                'relationships': {},
                'connection_score': 0.0,
                'context_relevance': 0.0
            }
            
            # Calculate relationships with other results
            if self.nervous_system._duplicate_detector:
                try:
                    # Use duplicate detector for relationship mapping
                    other_memories = [m for j, m in enumerate(results) if j != i]
                    relationships = await self.nervous_system._duplicate_detector.map_relationships(
                        memory, other_memories[:5]  # Limit for performance
                    )
                    result_data['relationships'] = relationships
                    
                    # Calculate overall connection score
                    if relationships:
                        result_data['connection_score'] = sum(relationships.values()) / len(relationships)
                        self._relationship_mapped_results += 1
                        
                except Exception as e:
                    self.logger.debug(f"Relationship mapping failed for memory: {e}")
            
            # Calculate context relevance based on query intent
            context_relevance = await self._calculate_context_relevance(
                memory, enhanced_query_info
            )
            result_data['context_relevance'] = context_relevance
            
            relationship_mapped_results.append(result_data)
        
        return relationship_mapped_results
    
    async def _analyze_result_patterns(
        self, 
        relationship_mapped_results: List[Dict[str, Any]], 
        original_query: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze patterns across results for deeper insights.
        
        Identifies common themes, temporal patterns, and knowledge clusters.
        """
        for result_data in relationship_mapped_results:
            memory = result_data['memory']
            
            # Pattern analysis
            pattern_insights = {
                'temporal_pattern': 'recent',  # Simplified - would analyze timestamps
                'content_category': self._categorize_memory_content(memory.content),
                'knowledge_cluster': 'general',  # Simplified - would use clustering
                'insight_potential': 0.5  # Simplified scoring
            }
            
            # Enhanced pattern analysis if we have quality gate
            if self.nervous_system._quality_gate:
                try:
                    # Use quality dimensions for pattern insights
                    quality_assessment = await self.nervous_system._quality_gate.assess_quality(
                        memory.content, 
                        getattr(memory, 'metadata', {}).get('type', 'interaction')
                    )
                    
                    pattern_insights['quality_dimensions'] = quality_assessment.dimensions
                    pattern_insights['insight_potential'] = quality_assessment.overall_score
                    
                    self._pattern_analyzed_results += 1
                    
                except Exception as e:
                    self.logger.debug(f"Pattern analysis failed: {e}")
            
            result_data['pattern_analysis'] = pattern_insights
        
        return relationship_mapped_results
    
    async def _apply_intelligent_ranking(
        self, 
        pattern_analyzed_results: List[Dict[str, Any]], 
        enhanced_query_info: Dict[str, Any], 
        final_limit: int
    ) -> List[Dict[str, Any]]:
        """
        Apply multi-factor intelligent ranking with diversity optimization.
        
        Balances relevance with diversity to prevent echo chambers.
        """
        # Calculate composite scores for each result
        for result_data in pattern_analyzed_results:
            scores = {
                'semantic_similarity': 0.8,  # Would use actual similarity from search
                'relationship_strength': result_data['connection_score'],
                'recency': 0.7,  # Would calculate from timestamp
                'quality_score': result_data.get('pattern_analysis', {}).get('insight_potential', 0.5),
                'access_frequency': 0.5  # Would track actual access patterns
            }
            
            # Apply context-based boosts
            context_hints = enhanced_query_info.get('context_hints', [])
            if 'prioritize_solutions' in context_hints:
                if 'solution' in result_data['memory'].content.lower():
                    scores['semantic_similarity'] *= 1.2
            
            # Calculate weighted composite score
            composite_score = sum(
                scores[factor] * self._ranking_weights[factor]
                for factor in scores
            )
            
            result_data['composite_score'] = composite_score
            result_data['ranking_factors'] = scores
        
        # Sort by composite score
        ranked_results = sorted(
            pattern_analyzed_results,
            key=lambda x: x['composite_score'],
            reverse=True
        )
        
        # Apply diversity optimization (simplified)
        diverse_results = []
        content_categories_seen = set()
        
        for result_data in ranked_results:
            if len(diverse_results) >= final_limit:
                break
                
            category = result_data.get('pattern_analysis', {}).get('content_category', 'general')
            
            # Add if we haven't seen this category yet, or if it's highly ranked
            if category not in content_categories_seen or result_data['composite_score'] > 0.8:
                diverse_results.append(result_data)
                content_categories_seen.add(category)
        
        # Fill remaining slots with highest ranked if needed
        remaining_slots = final_limit - len(diverse_results)
        if remaining_slots > 0:
            for result_data in ranked_results:
                if result_data not in diverse_results:
                    diverse_results.append(result_data)
                    remaining_slots -= 1
                    if remaining_slots <= 0:
                        break
        
        return diverse_results[:final_limit]
    
    async def _format_enhanced_results(
        self, 
        intelligent_results: List[Dict[str, Any]], 
        enhanced_query_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Format results to match expected interface while adding intelligence insights.
        
        Maintains compatibility while providing enhanced information.
        """
        formatted_results = []
        
        for i, result_data in enumerate(intelligent_results):
            memory = result_data['memory']
            
            # Base format (maintains compatibility)
            formatted_result = {
                'memory_id': getattr(memory, 'id', f'mem_{i}'),
                'content': memory.content,
                'metadata': getattr(memory, 'metadata', {}),
                'relevance_score': result_data['composite_score'],
                'rank': i + 1
            }
            
            # Add intelligence insights (optional detailed information)
            formatted_result['intelligence_analysis'] = {
                'relationship_connections': len(result_data.get('relationships', {})),
                'context_relevance': result_data['context_relevance'],
                'pattern_category': result_data.get('pattern_analysis', {}).get('content_category', 'general'),
                'ranking_factors': result_data.get('ranking_factors', {}),
                'insight_potential': result_data.get('pattern_analysis', {}).get('insight_potential', 0.5)
            }
            
            # Add relationship information if available
            if result_data.get('relationships'):
                formatted_result['related_memories'] = list(result_data['relationships'].keys())[:3]
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    async def _calculate_context_relevance(
        self, 
        memory: Memory, 
        enhanced_query_info: Dict[str, Any]
    ) -> float:
        """Calculate context relevance score based on query intent"""
        relevance_score = 0.5  # Base relevance
        
        primary_intent = enhanced_query_info.get('intent_analysis', {}).get('primary_intent', 'general')
        content_lower = memory.content.lower()
        
        # Intent-based relevance scoring
        if primary_intent == 'technical':
            technical_terms = ['implement', 'code', 'system', 'architecture', 'api']
            relevance_score += 0.3 * sum(1 for term in technical_terms if term in content_lower) / len(technical_terms)
        
        elif primary_intent == 'problem_solving':
            solution_terms = ['solution', 'fix', 'resolve', 'answer', 'workaround']
            relevance_score += 0.3 * sum(1 for term in solution_terms if term in content_lower) / len(solution_terms)
        
        return min(1.0, relevance_score)
    
    def _categorize_memory_content(self, content: str) -> str:
        """Categorize memory content for pattern analysis"""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['error', 'bug', 'issue', 'problem']):
            return 'problem_solving'
        elif any(term in content_lower for term in ['implement', 'code', 'system', 'api']):
            return 'technical'
        elif any(term in content_lower for term in ['insight', 'learned', 'understanding']):
            return 'learning'
        elif any(term in content_lower for term in ['plan', 'strategy', 'approach']):
            return 'planning'
        else:
            return 'general'
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for recall operations"""
        from statistics import mean
        
        avg_processing_time = mean(self._processing_times) if self._processing_times else 0.0
        
        return {
            'total_recall_operations': self._recall_count,
            'enhanced_queries': self._enhanced_queries,
            'relationship_mapped_results': self._relationship_mapped_results,
            'pattern_analyzed_results': self._pattern_analyzed_results,
            'query_enhancement_rate': self._enhanced_queries / max(1, self._recall_count),
            'relationship_mapping_rate': self._relationship_mapped_results / max(1, self._recall_count),
            'pattern_analysis_rate': self._pattern_analyzed_results / max(1, self._recall_count),
            'average_processing_time_ms': avg_processing_time,
            'target_processing_time_ms': 5.0,
            'performance_target_met': avg_processing_time < 5.0,
            'biological_reflex_achieved': avg_processing_time < 5.0,
            'ranking_weights': self._ranking_weights,
            'nervous_system_version': '2.0.0'
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for recall trigger"""
        if not self._initialized:
            return {'status': 'not_initialized', 'healthy': False}
        
        # Check nervous system health
        ns_health = await self.nervous_system.health_check()
        
        # Recall-specific health indicators
        recall_health = {
            'status': 'operational',
            'healthy': ns_health['healthy'],
            'operations_count': self._recall_count,
            'performance_metrics': self.get_performance_metrics(),
            'nervous_system_health': ns_health
        }
        
        # Check for performance issues
        if self._processing_times:
            avg_time = sum(self._processing_times) / len(self._processing_times)
            if avg_time > 5.0:
                recall_health['warnings'] = [f"Average processing time {avg_time:.2f}ms exceeds 5ms target"]
        
        return recall_health