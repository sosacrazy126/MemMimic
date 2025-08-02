"""
Memory Lifecycle Management System

Manages the complete lifecycle of memories from creation to archival/pruning.
Tracks lifecycle stages, transitions, and provides intelligent lifecycle management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .memory_evolution_tracker import MemoryEvolutionTracker, MemoryEventType
from ..memory.storage.amms_storage import Memory
from ..errors.exceptions import MemMimicError


class LifecycleStage(Enum):
    """Memory lifecycle stages"""
    CREATED = "created"          # Just created, no activity
    WARMING = "warming"          # Initial usage period
    ACTIVE = "active"            # Regular usage
    PEAK = "peak"               # High usage/importance
    MATURE = "mature"           # Established, stable usage
    DECLINING = "declining"     # Decreasing usage
    DORMANT = "dormant"         # No recent activity
    ARCHIVED = "archived"       # Moved to long-term storage
    DEPRECATED = "deprecated"   # Marked for removal
    PRUNED = "pruned"          # Removed from system


@dataclass
class LifecycleTransition:
    """Represents a lifecycle stage transition"""
    memory_id: str
    from_stage: LifecycleStage
    to_stage: LifecycleStage
    timestamp: datetime
    trigger: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LifecyclePolicy:
    """Defines policies for lifecycle management"""
    stage: LifecycleStage
    max_duration_days: Optional[int] = None
    min_activity_threshold: Optional[float] = None
    auto_transition_enabled: bool = True
    retention_priority: int = 5  # 1-10, higher = keep longer
    transition_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryLifecycleStatus:
    """Current lifecycle status of a memory"""
    memory_id: str
    current_stage: LifecycleStage
    stage_entered: datetime
    stage_duration: timedelta
    next_review: datetime
    lifecycle_score: float  # 0.0-1.0, health of lifecycle progression
    predicted_next_stage: Optional[LifecycleStage] = None
    transition_probability: float = 0.0
    health_flags: List[str] = field(default_factory=list)


class MemoryLifecycleManager:
    """
    Manages the complete lifecycle of memories with intelligent stage transitions.
    
    Core capabilities:
    - Stage tracking: Monitors current lifecycle stage of each memory
    - Transition management: Handles automatic and manual stage transitions
    - Policy enforcement: Applies configurable lifecycle policies
    - Predictive analysis: Predicts future lifecycle transitions
    - Health monitoring: Tracks lifecycle health and anomalies
    - Retention management: Manages memory retention based on lifecycle stage
    """
    
    def __init__(self, 
                 evolution_tracker: MemoryEvolutionTracker,
                 policies_config: Optional[Dict[str, Any]] = None):
        self.evolution_tracker = evolution_tracker
        self._memory_lifecycle_status: Dict[str, MemoryLifecycleStatus] = {}
        self._lifecycle_policies = self._init_default_policies(policies_config)
        self._transition_history: List[LifecycleTransition] = []
        self._last_review = datetime.now()
    
    def _init_default_policies(self, config: Optional[Dict[str, Any]] = None) -> Dict[LifecycleStage, LifecyclePolicy]:
        """Initialize default lifecycle policies"""
        policies = {
            LifecycleStage.CREATED: LifecyclePolicy(
                stage=LifecycleStage.CREATED,
                max_duration_days=7,  # Move to warming after 1 week if no activity
                min_activity_threshold=0.0,
                retention_priority=8,  # High priority for new memories
                transition_conditions={
                    'min_accesses_for_warming': 1,
                    'auto_archive_if_no_activity': True
                }
            ),
            LifecycleStage.WARMING: LifecyclePolicy(
                stage=LifecycleStage.WARMING,
                max_duration_days=14,  # 2 weeks to establish pattern
                min_activity_threshold=0.1,  # At least some activity
                retention_priority=7,
                transition_conditions={
                    'min_accesses_for_active': 3,
                    'access_frequency_threshold': 0.2  # 1 access per 5 days
                }
            ),
            LifecycleStage.ACTIVE: LifecyclePolicy(
                stage=LifecycleStage.ACTIVE,
                max_duration_days=30,
                min_activity_threshold=0.2,
                retention_priority=9,  # High priority for active memories
                transition_conditions={
                    'peak_access_threshold': 1.0,  # 1+ accesses per day
                    'importance_threshold': 0.7
                }
            ),
            LifecycleStage.PEAK: LifecyclePolicy(
                stage=LifecycleStage.PEAK,
                max_duration_days=60,
                min_activity_threshold=0.5,
                retention_priority=10,  # Highest priority
                transition_conditions={
                    'mature_stability_days': 14,  # Stable for 2 weeks
                    'mature_access_threshold': 0.3
                }
            ),
            LifecycleStage.MATURE: LifecyclePolicy(
                stage=LifecycleStage.MATURE,
                max_duration_days=90,
                min_activity_threshold=0.1,
                retention_priority=6,
                transition_conditions={
                    'decline_threshold': 0.05,  # Very low activity
                    'decline_duration_days': 21
                }
            ),
            LifecycleStage.DECLINING: LifecyclePolicy(
                stage=LifecycleStage.DECLINING,
                max_duration_days=30,
                min_activity_threshold=0.01,
                retention_priority=4,
                transition_conditions={
                    'dormant_threshold': 0.0,
                    'dormant_duration_days': 7
                }
            ),
            LifecycleStage.DORMANT: LifecyclePolicy(
                stage=LifecycleStage.DORMANT,
                max_duration_days=60,
                min_activity_threshold=0.0,
                retention_priority=2,
                transition_conditions={
                    'archive_after_days': 45,
                    'reactivation_threshold': 0.1
                }
            ),
            LifecycleStage.ARCHIVED: LifecyclePolicy(
                stage=LifecycleStage.ARCHIVED,
                max_duration_days=365,  # 1 year in archive
                min_activity_threshold=0.0,
                retention_priority=1,
                transition_conditions={
                    'prune_after_days': 365
                }
            )
        }
        
        # Apply custom configuration
        if config:
            for stage_name, policy_config in config.items():
                try:
                    stage = LifecycleStage(stage_name)
                    if stage in policies:
                        # Update existing policy with config values
                        policy = policies[stage]
                        for key, value in policy_config.items():
                            if hasattr(policy, key):
                                setattr(policy, key, value)
                            elif key == 'transition_conditions':
                                policy.transition_conditions.update(value)
                except ValueError:
                    continue  # Skip invalid stage names
        
        return policies
    
    async def track_memory_creation(self, memory: Memory):
        """Track a new memory entering the lifecycle"""
        status = MemoryLifecycleStatus(
            memory_id=memory.id,
            current_stage=LifecycleStage.CREATED,
            stage_entered=datetime.now(),
            stage_duration=timedelta(0),
            next_review=datetime.now() + timedelta(days=1),
            lifecycle_score=1.0  # Start with perfect health
        )
        
        self._memory_lifecycle_status[memory.id] = status
        
        # Track creation event
        await self.evolution_tracker.track_event(
            memory_id=memory.id,
            event_type=MemoryEventType.CREATED,
            context={'lifecycle_stage': LifecycleStage.CREATED.value},
            trigger='memory_creation'
        )
        
        # Record transition
        transition = LifecycleTransition(
            memory_id=memory.id,
            from_stage=LifecycleStage.CREATED,  # Conceptual starting point
            to_stage=LifecycleStage.CREATED,
            timestamp=datetime.now(),
            trigger='memory_creation',
            confidence=1.0,
            metadata={
                'memory_type': memory.metadata.get('type', 'unknown'), 
                'importance': memory.importance_score
            }
        )
        self._transition_history.append(transition)
    
    async def update_memory_activity(self, 
                                   memory_id: str, 
                                   activity_type: str,
                                   context: Dict[str, Any] = None):
        """Update memory activity and check for lifecycle transitions"""
        if memory_id not in self._memory_lifecycle_status:
            # Initialize if not tracked
            status = MemoryLifecycleStatus(
                memory_id=memory_id,
                current_stage=LifecycleStage.CREATED,
                stage_entered=datetime.now() - timedelta(days=1),  # Assume created recently
                stage_duration=timedelta(days=1),
                next_review=datetime.now(),
                lifecycle_score=0.8
            )
            self._memory_lifecycle_status[memory_id] = status
        
        # Update stage duration
        status = self._memory_lifecycle_status[memory_id]
        status.stage_duration = datetime.now() - status.stage_entered
        
        # Check for potential transitions
        await self._evaluate_transitions(memory_id, activity_trigger=activity_type)
    
    async def _evaluate_transitions(self, memory_id: str, activity_trigger: str = None):
        """Evaluate if a memory should transition to a new lifecycle stage"""
        status = self._memory_lifecycle_status[memory_id]
        current_stage = status.current_stage
        
        # Get memory evolution data
        evolution_summary = await self.evolution_tracker.get_memory_evolution_summary(memory_id)
        usage_stats = evolution_summary['usage_stats']
        
        # Determine potential transitions
        potential_stage = await self._determine_optimal_stage(memory_id, usage_stats, current_stage)
        
        if potential_stage != current_stage:
            confidence = await self._calculate_transition_confidence(
                memory_id, current_stage, potential_stage, usage_stats
            )
            
            if confidence > 0.7:  # High confidence threshold
                await self._execute_transition(
                    memory_id, current_stage, potential_stage, 
                    trigger=activity_trigger or 'automatic_evaluation',
                    confidence=confidence
                )
    
    async def _determine_optimal_stage(self, 
                                     memory_id: str, 
                                     usage_stats: Dict[str, Any],
                                     current_stage: LifecycleStage) -> LifecycleStage:
        """Determine the optimal lifecycle stage based on usage patterns"""
        access_frequency = usage_stats.get('access_frequency', 0.0)
        total_accesses = usage_stats.get('total_accesses', 0)
        recall_success_rate = usage_stats.get('recall_success_rate', 0.0)
        last_accessed = usage_stats.get('last_accessed')
        
        # Calculate days since last access
        days_since_access = 999  # Default to very high
        if last_accessed:
            try:
                last_access_time = datetime.fromisoformat(last_accessed)
                days_since_access = (datetime.now() - last_access_time).days
            except ValueError:
                pass
        
        # Stage determination logic
        if total_accesses == 0:
            return LifecycleStage.CREATED
        
        elif days_since_access > 60:
            if current_stage in [LifecycleStage.ARCHIVED, LifecycleStage.DEPRECATED]:
                return current_stage  # Stay in archive/deprecated
            return LifecycleStage.DORMANT
        
        elif days_since_access > 30:
            return LifecycleStage.DECLINING
        
        elif access_frequency >= 1.0:  # Daily or more
            return LifecycleStage.PEAK
        
        elif access_frequency >= 0.3:  # Several times per week
            return LifecycleStage.ACTIVE
        
        elif access_frequency >= 0.1:  # Regular but not frequent
            if current_stage == LifecycleStage.PEAK:
                return LifecycleStage.MATURE  # Transition from peak
            return LifecycleStage.ACTIVE
        
        elif access_frequency > 0.01:  # Occasional access
            return LifecycleStage.WARMING if total_accesses < 5 else LifecycleStage.MATURE
        
        else:
            return LifecycleStage.DECLINING
    
    async def _calculate_transition_confidence(self, 
                                             memory_id: str,
                                             from_stage: LifecycleStage,
                                             to_stage: LifecycleStage,
                                             usage_stats: Dict[str, Any]) -> float:
        """Calculate confidence in a proposed lifecycle transition"""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Usage consistency
        access_frequency = usage_stats.get('access_frequency', 0.0)
        total_accesses = usage_stats.get('total_accesses', 0)
        
        if total_accesses >= 5:  # Enough data points
            confidence += 0.2
        
        # Factor 2: Recall success rate
        recall_success = usage_stats.get('recall_success_rate', 0.0)
        if recall_success > 0.8:
            confidence += 0.1
        elif recall_success < 0.5 and total_accesses > 3:
            confidence -= 0.1
        
        # Factor 3: Stage progression logic
        stage_order = {
            LifecycleStage.CREATED: 0,
            LifecycleStage.WARMING: 1,
            LifecycleStage.ACTIVE: 2,
            LifecycleStage.PEAK: 3,
            LifecycleStage.MATURE: 4,
            LifecycleStage.DECLINING: 5,
            LifecycleStage.DORMANT: 6,
            LifecycleStage.ARCHIVED: 7
        }
        
        from_order = stage_order.get(from_stage, 0)
        to_order = stage_order.get(to_stage, 0)
        
        # Natural progression increases confidence
        if to_order == from_order + 1:  # Next stage
            confidence += 0.2
        elif to_order == from_order - 1 and to_stage == LifecycleStage.ACTIVE:  # Reactivation
            confidence += 0.1
        elif abs(to_order - from_order) > 2:  # Jumping stages
            confidence -= 0.2
        
        # Factor 4: Time in current stage
        status = self._memory_lifecycle_status.get(memory_id)
        if status:
            days_in_stage = status.stage_duration.days
            policy = self._lifecycle_policies.get(from_stage)
            if policy and policy.max_duration_days:
                if days_in_stage >= policy.max_duration_days:
                    confidence += 0.2  # Time for transition
                elif days_in_stage < 1:
                    confidence -= 0.3  # Too soon
        
        return max(0.0, min(1.0, confidence))
    
    async def _execute_transition(self, 
                                memory_id: str,
                                from_stage: LifecycleStage,
                                to_stage: LifecycleStage,
                                trigger: str,
                                confidence: float):
        """Execute a lifecycle stage transition"""
        # Update status
        status = self._memory_lifecycle_status[memory_id]
        status.current_stage = to_stage
        status.stage_entered = datetime.now()
        status.stage_duration = timedelta(0)
        
        # Calculate next review time
        policy = self._lifecycle_policies.get(to_stage)
        if policy and policy.max_duration_days:
            status.next_review = datetime.now() + timedelta(days=min(policy.max_duration_days // 2, 14))
        else:
            status.next_review = datetime.now() + timedelta(days=7)
        
        # Update lifecycle score based on transition appropriateness
        if confidence > 0.9:
            status.lifecycle_score = min(1.0, status.lifecycle_score + 0.1)
        elif confidence < 0.5:
            status.lifecycle_score = max(0.0, status.lifecycle_score - 0.1)
        
        # Record transition
        transition = LifecycleTransition(
            memory_id=memory_id,
            from_stage=from_stage,
            to_stage=to_stage,
            timestamp=datetime.now(),
            trigger=trigger,
            confidence=confidence,
            metadata={'policy_applied': policy.stage.value if policy else None}
        )
        self._transition_history.append(transition)
        
        # Track evolution event
        await self.evolution_tracker.track_event(
            memory_id=memory_id,
            event_type=MemoryEventType.ARCHIVED if to_stage == LifecycleStage.ARCHIVED else MemoryEventType.ACCESSED,
            context={
                'lifecycle_transition': True,
                'from_stage': from_stage.value,
                'to_stage': to_stage.value,
                'confidence': confidence
            },
            trigger=trigger
        )
    
    async def review_all_memories(self) -> Dict[str, Any]:
        """Review all memories for potential lifecycle transitions"""
        reviewed_count = 0
        transitioned_count = 0
        current_time = datetime.now()
        
        for memory_id, status in self._memory_lifecycle_status.items():
            if current_time >= status.next_review:
                await self._evaluate_transitions(memory_id, activity_trigger='scheduled_review')
                reviewed_count += 1
                
                # Check if transition occurred
                new_status = self._memory_lifecycle_status[memory_id]
                if new_status.stage_entered > status.stage_entered:
                    transitioned_count += 1
        
        self._last_review = current_time
        
        return {
            'review_completed': current_time.isoformat(),
            'memories_reviewed': reviewed_count,
            'transitions_executed': transitioned_count,
            'next_review_recommended': (current_time + timedelta(hours=6)).isoformat()
        }
    
    async def get_lifecycle_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive lifecycle analytics"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Stage distribution
        stage_distribution = {}
        for status in self._memory_lifecycle_status.values():
            stage = status.current_stage.value
            stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
        
        # Recent transitions
        recent_transitions = [
            t for t in self._transition_history 
            if t.timestamp >= cutoff_time
        ]
        
        # Transition patterns
        transition_patterns = {}
        for transition in recent_transitions:
            pattern = f"{transition.from_stage.value}->{transition.to_stage.value}"
            transition_patterns[pattern] = transition_patterns.get(pattern, 0) + 1
        
        # Health metrics
        total_memories = len(self._memory_lifecycle_status)
        healthy_memories = sum(
            1 for status in self._memory_lifecycle_status.values() 
            if status.lifecycle_score > 0.7
        )
        
        # Stage durations
        avg_stage_durations = {}
        for stage in LifecycleStage:
            durations = [
                status.stage_duration.days 
                for status in self._memory_lifecycle_status.values()
                if status.current_stage == stage
            ]
            if durations:
                avg_stage_durations[stage.value] = sum(durations) / len(durations)
        
        return {
            'analysis_period_days': days,
            'total_memories_tracked': total_memories,
            'stage_distribution': stage_distribution,
            'recent_transitions': len(recent_transitions),
            'transition_patterns': transition_patterns,
            'health_metrics': {
                'healthy_memories': healthy_memories,
                'health_percentage': (healthy_memories / total_memories * 100) if total_memories > 0 else 0,
                'avg_lifecycle_score': sum(s.lifecycle_score for s in self._memory_lifecycle_status.values()) / total_memories if total_memories > 0 else 0
            },
            'average_stage_durations': avg_stage_durations,
            'policy_effectiveness': await self._calculate_policy_effectiveness()
        }
    
    async def _calculate_policy_effectiveness(self) -> Dict[str, Any]:
        """Calculate effectiveness of lifecycle policies"""
        effectiveness = {}
        
        for stage, policy in self._lifecycle_policies.items():
            # Count memories in this stage
            memories_in_stage = [
                s for s in self._memory_lifecycle_status.values() 
                if s.current_stage == stage
            ]
            
            if not memories_in_stage:
                continue
            
            # Calculate metrics
            avg_duration = sum(s.stage_duration.days for s in memories_in_stage) / len(memories_in_stage)
            avg_score = sum(s.lifecycle_score for s in memories_in_stage) / len(memories_in_stage)
            
            # Policy adherence
            adherence_count = 0
            if policy.max_duration_days:
                adherence_count = sum(
                    1 for s in memories_in_stage 
                    if s.stage_duration.days <= policy.max_duration_days
                )
            
            effectiveness[stage.value] = {
                'memories_count': len(memories_in_stage),
                'average_duration_days': avg_duration,
                'average_lifecycle_score': avg_score,
                'policy_adherence': (adherence_count / len(memories_in_stage)) if policy.max_duration_days else 1.0
            }
        
        return effectiveness
    
    async def predict_future_transitions(self, memory_id: str, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """Predict future lifecycle transitions for a memory"""
        if memory_id not in self._memory_lifecycle_status:
            return []
        
        status = self._memory_lifecycle_status[memory_id]
        current_stage = status.current_stage
        
        # Get evolution data for prediction
        evolution_summary = await self.evolution_tracker.get_memory_evolution_summary(memory_id)
        usage_stats = evolution_summary['usage_stats']
        
        predictions = []
        prediction_date = datetime.now()
        
        # Simple prediction based on current trends
        access_frequency = usage_stats.get('access_frequency', 0.0)
        
        # Predict next 30 days
        for day in range(1, days_ahead + 1):
            prediction_date = datetime.now() + timedelta(days=day)
            
            # Simulate aging effect on access frequency
            aged_frequency = access_frequency * (0.95 ** day)  # 5% decay per day
            
            # Predict stage based on aged frequency
            if aged_frequency >= 1.0:
                predicted_stage = LifecycleStage.PEAK
            elif aged_frequency >= 0.3:
                predicted_stage = LifecycleStage.ACTIVE
            elif aged_frequency >= 0.1:
                predicted_stage = LifecycleStage.MATURE
            elif aged_frequency >= 0.01:
                predicted_stage = LifecycleStage.DECLINING
            else:
                predicted_stage = LifecycleStage.DORMANT
            
            if predicted_stage != current_stage:
                confidence = max(0.1, min(0.9, aged_frequency + 0.3))
                predictions.append({
                    'predicted_date': prediction_date.isoformat(),
                    'from_stage': current_stage.value,
                    'to_stage': predicted_stage.value,
                    'confidence': confidence,
                    'predicted_access_frequency': aged_frequency
                })
                current_stage = predicted_stage  # Update for next prediction
        
        return predictions
    
    async def get_retention_recommendations(self) -> List[Dict[str, Any]]:
        """Generate memory retention recommendations based on lifecycle analysis"""
        recommendations = []
        
        for memory_id, status in self._memory_lifecycle_status.items():
            current_stage = status.current_stage
            policy = self._lifecycle_policies.get(current_stage)
            
            if not policy:
                continue
            
            recommendation = {
                'memory_id': memory_id,
                'current_stage': current_stage.value,
                'retention_priority': policy.retention_priority,
                'action': 'keep',
                'reasoning': []
            }
            
            # Low priority stages
            if current_stage in [LifecycleStage.DORMANT, LifecycleStage.DECLINING]:
                if status.stage_duration.days > 30:
                    recommendation['action'] = 'archive'
                    recommendation['reasoning'].append('Long duration in low-activity stage')
            
            # Archive stage
            elif current_stage == LifecycleStage.ARCHIVED:
                if status.stage_duration.days > 180:
                    recommendation['action'] = 'consider_pruning'
                    recommendation['reasoning'].append('Extended time in archive')
            
            # High value stages
            elif current_stage in [LifecycleStage.PEAK, LifecycleStage.ACTIVE]:
                recommendation['action'] = 'prioritize'
                recommendation['reasoning'].append('High activity/importance')
            
            # Health-based recommendations
            if status.lifecycle_score < 0.3:
                recommendation['action'] = 'review'
                recommendation['reasoning'].append('Low lifecycle health score')
            
            recommendations.append(recommendation)
        
        # Sort by retention priority (lowest first = candidates for pruning)
        recommendations.sort(key=lambda x: x['retention_priority'])
        
        return recommendations
    
    def get_memory_lifecycle_status(self, memory_id: str) -> Optional[MemoryLifecycleStatus]:
        """Get current lifecycle status for a specific memory"""
        return self._memory_lifecycle_status.get(memory_id)
    
    async def force_transition(self, 
                             memory_id: str, 
                             to_stage: LifecycleStage,
                             reason: str = "manual_override") -> bool:
        """Force a memory to transition to a specific stage"""
        if memory_id not in self._memory_lifecycle_status:
            return False
        
        current_status = self._memory_lifecycle_status[memory_id]
        from_stage = current_status.current_stage
        
        await self._execute_transition(
            memory_id=memory_id,
            from_stage=from_stage,
            to_stage=to_stage,
            trigger=reason,
            confidence=1.0  # Manual override has full confidence
        )
        
        return True