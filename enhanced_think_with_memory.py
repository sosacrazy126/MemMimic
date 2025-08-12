#!/usr/bin/env python3
"""
Enhanced Think With Memory System
Combines sequential thinking with iterative memory retrieval
"""

from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import math

class ThoughtPhase(Enum):
    EXPLORATION = "exploration"  # Broad search
    REFINEMENT = "refinement"    # Focused search
    SYNTHESIS = "synthesis"      # Combining insights
    VALIDATION = "validation"    # Checking consistency

@dataclass
class Memory:
    """Memory object compatible with MemMimic storage"""
    id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class EnhancedThinkWithMemory:
    """
    Enhanced thinking system that combines sequential thinking 
    with iterative memory retrieval
    """
    
    def __init__(self, storage_adapter):
        self.storage = storage_adapter
        self.thought_chain = []
        self.memory_pool = {}
        self.search_history = set()
        self.start_time = datetime.now()
        
    def think(self, query: str, max_thoughts: int = 10, confidence_threshold: float = 0.8) -> Dict:
        """
        Main thinking entry point
        
        Args:
            query: The question or topic to think about
            max_thoughts: Maximum number of thoughts to generate
            confidence_threshold: Stop when confidence reaches this level
            
        Returns:
            Dictionary containing the thinking process and results
        """
        # Initialize context
        context = {
            'query': query,
            'understanding': {},
            'confidence': 0.0,
            'phase': ThoughtPhase.EXPLORATION
        }
        
        # Thinking loop
        for thought_num in range(1, max_thoughts + 1):
            thought = self._generate_thought(thought_num, context)
            self.thought_chain.append(thought)
            
            # Update context
            context = self._update_context(context, thought)
            
            # Check if we have sufficient understanding
            if context['confidence'] >= confidence_threshold:
                break
                
        return self._compile_response(context)
    
    def _generate_thought(self, thought_num: int, context: Dict) -> Dict:
        """Generate a single thought with memory retrieval"""
        
        thought = {
            'number': thought_num,
            'timestamp': datetime.now().isoformat(),
            'phase': context['phase'].value,
            'search_queries': [],
            'memories_retrieved': [],
            'insights': [],
            'confidence_delta': 0.0,
            'content': ''
        }
        
        # Phase 1: Determine what we need to search for
        if context['phase'] == ThoughtPhase.EXPLORATION:
            queries = self._generate_exploration_queries(context['query'])
        elif context['phase'] == ThoughtPhase.REFINEMENT:
            queries = self._generate_refinement_queries(context)
        else:  # synthesis or validation
            queries = self._generate_synthesis_queries(context)
        
        thought['search_queries'] = queries
        
        # Phase 2: Retrieve memories
        new_memories = []
        for query in queries:
            if query not in self.search_history:
                memories = self.storage.search(query, limit=3)
                new_memories.extend(memories)
                self.search_history.add(query)
        
        # Add to memory pool
        for memory in new_memories:
            if memory.id not in self.memory_pool:
                self.memory_pool[memory.id] = memory
                thought['memories_retrieved'].append(memory.id)
        
        # Phase 3: Generate insights
        thought['insights'] = self._extract_insights(new_memories, context)
        
        # Phase 4: Update confidence
        thought['confidence_delta'] = self._calculate_confidence_delta(thought)
        
        # Phase 5: Generate thought content
        thought['content'] = self._synthesize_thought_content(thought, context)
        
        return thought
    
    def _generate_exploration_queries(self, original_query: str) -> List[str]:
        """Generate broad exploration queries"""
        queries = [original_query]
        
        # Extract key terms
        words = original_query.lower().split()
        
        # Add individual word searches for important terms
        for word in words:
            if len(word) > 3 and word not in ['what', 'when', 'where', 'which', 'how', 'why']:
                queries.append(word)
        
        # Add question type specific searches
        if 'how' in words:
            queries.extend(['process', 'method', 'implementation'])
        if 'why' in words:
            queries.extend(['reason', 'because', 'purpose'])
        if 'when' in words:
            queries.extend(['timeline', 'date', 'schedule'])
        if 'what' in words:
            queries.extend(['definition', 'description', 'explanation'])
        
        return queries[:5]  # Limit to prevent too many searches
    
    def _generate_refinement_queries(self, context: Dict) -> List[str]:
        """Generate focused queries based on what we've learned"""
        queries = []
        
        # Look for gaps in understanding
        if 'current_status' not in context['understanding']:
            queries.extend(['current', 'status', 'now', 'latest'])
        
        if 'background' not in context['understanding']:
            queries.extend(['background', 'context', 'history'])
        
        # Follow up on insights from previous thoughts
        for thought in self.thought_chain[-3:]:  # Last 3 thoughts
            for insight in thought.get('insights', []):
                # Extract entities from insights
                if 'migration' in insight.lower() and 'migration details' not in self.search_history:
                    queries.append('migration details')
                if 'markdown' in insight.lower() and 'markdown storage' not in self.search_history:
                    queries.append('markdown storage')
                if 'success' in insight.lower() and 'success metrics' not in self.search_history:
                    queries.append('success metrics')
                    
        return queries[:3]
    
    def _generate_synthesis_queries(self, context: Dict) -> List[str]:
        """Generate queries to connect and validate understanding"""
        queries = []
        
        # Look for relationships between found memories
        memory_topics = set()
        for memory_id, memory in self.memory_pool.items():
            # Extract topics from memory content
            content_lower = memory.content.lower()
            if 'migration' in content_lower:
                memory_topics.add('migration')
            if 'success' in content_lower:
                memory_topics.add('success')
            if 'complete' in content_lower:
                memory_topics.add('complete')
            if 'test' in content_lower:
                memory_topics.add('test')
                
        # Search for connections between topics
        if len(memory_topics) >= 2:
            topics_list = list(memory_topics)
            queries.append(' '.join(topics_list[:2]))
        
        # Search for validation
        if context['confidence'] > 0.5:
            queries.append('verify')
            queries.append('confirm')
        
        return queries[:2]
    
    def _extract_insights(self, memories: List[Memory], context: Dict) -> List[str]:
        """Extract insights from retrieved memories"""
        insights = []
        
        if not memories:
            insights.append("No new memories found for this query")
            return insights
        
        # Analyze memory characteristics
        cxd_types = [m.metadata.get('cxd', 'unknown') for m in memories]
        dates = [m.created_at for m in memories]
        importance_scores = [m.importance for m in memories]
        
        # Insight 1: CXD distribution
        cxd_counts = {}
        for cxd in cxd_types:
            cxd_counts[cxd] = cxd_counts.get(cxd, 0) + 1
        
        if cxd_counts:
            dominant_cxd = max(cxd_counts, key=cxd_counts.get)
            insights.append(f"Memories primarily contain {dominant_cxd} information")
        
        # Insight 2: Temporal patterns
        if dates:
            newest = max(dates)
            oldest = min(dates)
            span_days = (newest - oldest).days
            
            if span_days > 30:
                insights.append(f"Memories span {span_days} days of history")
            elif span_days == 0:
                insights.append("All memories are from the same day")
            else:
                insights.append(f"Recent memories from last {span_days} days")
        
        # Insight 3: Importance patterns
        if importance_scores:
            avg_importance = sum(importance_scores) / len(importance_scores)
            if avg_importance > 0.7:
                insights.append("High importance memories found")
            elif avg_importance < 0.3:
                insights.append("Low importance memories, may need more significant sources")
        
        # Insight 4: Content patterns
        common_terms = self._find_common_terms(memories)
        if common_terms:
            insights.append(f"Common themes: {', '.join(common_terms[:3])}")
        
        return insights
    
    def _find_common_terms(self, memories: List[Memory]) -> List[str]:
        """Find common terms across memories"""
        term_counts = {}
        
        # Skip common words
        stop_words = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'have', 'been', 'will'}
        
        for memory in memories:
            words = memory.content.lower().split()
            for word in words:
                if len(word) > 4 and word not in stop_words:
                    term_counts[word] = term_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return terms that appear in multiple memories
        common = [term for term, count in sorted_terms if count >= 2]
        
        return common
    
    def _update_context(self, context: Dict, thought: Dict) -> Dict:
        """Update context based on new thought"""
        
        # Update confidence
        context['confidence'] += thought['confidence_delta']
        context['confidence'] = min(1.0, context['confidence'])  # Cap at 1.0
        
        # Update understanding
        for insight in thought['insights']:
            # Parse insight to extract understanding
            if 'primarily contain' in insight:
                context['understanding']['dominant_info_type'] = insight
            elif 'span' in insight or 'Recent' in insight:
                context['understanding']['temporal_coverage'] = insight
            elif 'importance' in insight:
                context['understanding']['importance_level'] = insight
            elif 'Common themes' in insight:
                context['understanding']['themes'] = insight
        
        # Update phase based on progress
        thought_num = thought['number']
        if thought_num <= 3:
            context['phase'] = ThoughtPhase.EXPLORATION
        elif thought_num <= 6:
            context['phase'] = ThoughtPhase.REFINEMENT
        elif thought_num <= 8:
            context['phase'] = ThoughtPhase.SYNTHESIS
        else:
            context['phase'] = ThoughtPhase.VALIDATION
        
        # Check for saturation (not finding new memories)
        if len(thought['memories_retrieved']) == 0 and thought_num > 3:
            context['confidence'] += 0.1  # Boost confidence if saturated
        
        return context
    
    def _calculate_confidence_delta(self, thought: Dict) -> float:
        """Calculate how much this thought increases confidence"""
        
        delta = 0.0
        
        # More memories = more confidence
        delta += len(thought['memories_retrieved']) * 0.05
        
        # More insights = more confidence  
        delta += len(thought['insights']) * 0.03
        
        # Diverse search queries = more confidence
        delta += len(thought['search_queries']) * 0.02
        
        # Phase bonus
        if thought['phase'] == 'synthesis':
            delta += 0.05
        elif thought['phase'] == 'validation':
            delta += 0.1
        
        # Cap the delta
        delta = min(0.3, delta)
        
        return delta
    
    def _synthesize_thought_content(self, thought: Dict, context: Dict) -> str:
        """Generate human-readable thought content"""
        
        parts = []
        
        # Describe what we're doing
        if thought['phase'] == 'exploration':
            parts.append(f"Exploring the topic by searching for: {', '.join(thought['search_queries'])}")
        elif thought['phase'] == 'refinement':
            parts.append(f"Refining understanding with focused searches: {', '.join(thought['search_queries'])}")
        elif thought['phase'] == 'synthesis':
            parts.append(f"Synthesizing findings from {len(self.memory_pool)} memories")
        else:
            parts.append(f"Validating understanding with final checks")
        
        # Describe what we found
        if thought['memories_retrieved']:
            parts.append(f"Found {len(thought['memories_retrieved'])} new relevant memories")
        else:
            parts.append("No new memories found with these queries")
        
        # Share insights
        if thought['insights']:
            parts.append("Key insights:")
            for insight in thought['insights']:
                parts.append(f"  - {insight}")
        
        # Describe confidence change
        if thought['confidence_delta'] > 0:
            parts.append(f"Confidence increased by {thought['confidence_delta']:.0%}")
        
        return '\n'.join(parts)
    
    def _compile_response(self, context: Dict) -> Dict:
        """Compile final response from thinking process"""
        
        response = {
            'status': 'success',
            'query': context['query'],
            'confidence': round(context['confidence'], 2),
            'thoughts_generated': len(self.thought_chain),
            'memories_examined': len(self.memory_pool),
            'thinking_time': round((datetime.now() - self.start_time).total_seconds(), 2),
            'understanding': context['understanding'],
            'thought_process': [],
            'final_analysis': '',
            'key_memories': []
        }
        
        # Add thought summaries
        for thought in self.thought_chain:
            response['thought_process'].append({
                'number': thought['number'],
                'phase': thought['phase'],
                'content': thought['content'],
                'memories_found': len(thought['memories_retrieved']),
                'insights': thought['insights']
            })
        
        # Generate final analysis
        response['final_analysis'] = self._generate_final_analysis(context)
        
        # Include most relevant memories
        response['key_memories'] = self._select_key_memories()
        
        return response
    
    def _generate_final_analysis(self, context: Dict) -> str:
        """Generate a coherent final analysis"""
        
        parts = []
        
        # Start with query interpretation
        parts.append(f"Analysis of '{context['query']}':")
        parts.append("")
        
        # Add understanding summary
        if context['understanding']:
            parts.append("Understanding gained:")
            for key, value in context['understanding'].items():
                parts.append(f"- {value}")
            parts.append("")
        
        # Add confidence statement
        if context['confidence'] >= 0.8:
            parts.append(f"High confidence analysis (confidence: {context['confidence']:.0%})")
        elif context['confidence'] >= 0.5:
            parts.append(f"Moderate confidence analysis (confidence: {context['confidence']:.0%})")
        else:
            parts.append(f"Low confidence analysis - limited relevant memories found")
        
        # Add memory coverage
        parts.append(f"\nExamined {len(self.memory_pool)} relevant memories across {len(self.thought_chain)} thoughts.")
        
        # Add search coverage
        unique_searches = len(self.search_history)
        parts.append(f"Performed {unique_searches} unique searches to build understanding.")
        
        return '\n'.join(parts)
    
    def _select_key_memories(self, limit: int = 3) -> List[Dict]:
        """Select the most important memories to include"""
        
        if not self.memory_pool:
            return []
        
        # Score all memories
        scored_memories = []
        for memory_id, memory in self.memory_pool.items():
            score = memory.importance
            
            # Boost score if memory was retrieved multiple times
            retrieval_count = sum(
                1 for thought in self.thought_chain 
                if memory_id in thought.get('memories_retrieved', [])
            )
            score += retrieval_count * 0.1
            
            scored_memories.append((score, memory))
        
        # Sort by score
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Return top memories
        key_memories = []
        for score, memory in scored_memories[:limit]:
            key_memories.append({
                'id': memory.id,
                'content_preview': memory.content[:200] + '...' if len(memory.content) > 200 else memory.content,
                'importance': memory.importance,
                'cxd': memory.metadata.get('cxd', 'unknown'),
                'type': memory.metadata.get('type', 'unknown'),
                'created': memory.created_at.isoformat() if hasattr(memory.created_at, 'isoformat') else str(memory.created_at)
            })
        
        return key_memories