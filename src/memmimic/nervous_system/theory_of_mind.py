"""
TheoryOfMindCapabilities - Agent Mental State Understanding

Implements theory of mind capabilities for understanding and predicting other
agents' mental states, intentions, and behaviors. This enables sophisticated
multi-agent coordination and collaborative intelligence.

Key capabilities:
- Mental state modeling and prediction
- Intention recognition and forecasting
- Belief state tracking and updating
- Collaborative goal alignment
- Empathetic response generation
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import numpy as np

from ..errors import MemMimicError, with_error_context, get_error_logger


class MentalStateType(Enum):
    """Types of mental states that can be modeled"""
    BELIEF = "belief"           # What the agent believes to be true
    DESIRE = "desire"           # What the agent wants to achieve
    INTENTION = "intention"     # What the agent plans to do
    EMOTION = "emotion"         # Emotional state of the agent
    KNOWLEDGE = "knowledge"     # What the agent knows
    CAPABILITY = "capability"   # What the agent can do
    ATTENTION = "attention"     # What the agent is focusing on


class ConfidenceLevel(Enum):
    """Confidence levels for mental state predictions"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


@dataclass
class MentalState:
    """Represents a mental state of an agent"""
    agent_id: str
    state_type: MentalStateType
    content: Dict[str, Any]
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)
    source: str = "observation"  # observation, inference, communication
    expires_at: Optional[float] = None


@dataclass
class IntentionPrediction:
    """Prediction of an agent's future intentions"""
    agent_id: str
    predicted_action: str
    predicted_goal: str
    confidence: float
    time_horizon: float  # seconds into the future
    prerequisites: List[str] = field(default_factory=list)
    alternative_actions: List[Tuple[str, float]] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class CollaborationOpportunity:
    """Identified opportunity for agent collaboration"""
    opportunity_id: str
    involved_agents: List[str]
    shared_goal: str
    collaboration_type: str  # 'complementary', 'parallel', 'sequential'
    potential_benefit: float
    coordination_requirements: List[str] = field(default_factory=list)
    estimated_duration: Optional[float] = None


class TheoryOfMindCapabilities:
    """
    Theory of Mind System for Multi-Agent Understanding
    
    Provides sophisticated mental state modeling and prediction capabilities
    to enable empathetic and coordinated multi-agent interactions.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = get_error_logger("theory_of_mind")
        
        # Mental state tracking
        self.mental_states: Dict[str, Dict[MentalStateType, MentalState]] = {}
        self.state_history: Dict[str, List[MentalState]] = {}
        
        # Prediction systems
        self.intention_predictions: Dict[str, List[IntentionPrediction]] = {}
        self.collaboration_opportunities: Dict[str, CollaborationOpportunity] = {}
        
        # Learning and adaptation
        self.interaction_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.prediction_accuracy: Dict[str, List[float]] = {}
        
        # Configuration
        self.max_history_length = 1000
        self.prediction_horizon = 300.0  # 5 minutes
        self.confidence_threshold = 0.6
        
        # Empathy and emotional modeling
        self.emotional_models: Dict[str, Dict[str, float]] = {}
        self.empathy_responses: Dict[str, List[str]] = {}
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize theory of mind capabilities"""
        if self._initialized:
            return
            
        try:
            # Initialize mental state models for known agents
            await self._initialize_mental_models()
            
            # Load historical interaction patterns
            await self._load_interaction_history()
            
            # Initialize empathy models
            await self._initialize_empathy_models()
            
            self._initialized = True
            self.logger.info("Theory of mind capabilities initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize theory of mind: {e}")
            raise MemMimicError(f"Theory of mind initialization failed: {e}")
    
    async def observe_agent_action(self, agent_id: str, action: str, context: Dict[str, Any]) -> None:
        """Observe and learn from an agent's action"""
        if not self._initialized:
            await self.initialize()
        
        with with_error_context(
            operation="observe_agent_action",
            component="theory_of_mind",
            metadata={"agent_id": agent_id, "action": action}
        ):
            # Record interaction pattern
            interaction = {
                'action': action,
                'context': context,
                'timestamp': time.time(),
                'observer': self.agent_id
            }
            
            if agent_id not in self.interaction_patterns:
                self.interaction_patterns[agent_id] = []
            
            self.interaction_patterns[agent_id].append(interaction)
            
            # Limit history size
            if len(self.interaction_patterns[agent_id]) > self.max_history_length:
                self.interaction_patterns[agent_id].pop(0)
            
            # Update mental state models
            await self._update_mental_states_from_observation(agent_id, action, context)
            
            # Generate new predictions
            await self._update_intention_predictions(agent_id)
            
            # Identify collaboration opportunities
            await self._identify_collaboration_opportunities(agent_id, action, context)
    
    async def _update_mental_states_from_observation(self, agent_id: str, action: str, 
                                                   context: Dict[str, Any]) -> None:
        """Update mental state models based on observed action"""
        if agent_id not in self.mental_states:
            self.mental_states[agent_id] = {}
        
        # Infer beliefs from action
        belief_content = await self._infer_beliefs_from_action(action, context)
        if belief_content:
            belief_state = MentalState(
                agent_id=agent_id,
                state_type=MentalStateType.BELIEF,
                content=belief_content,
                confidence=0.7,
                evidence=[f"observed_action: {action}"],
                source="inference"
            )
            self.mental_states[agent_id][MentalStateType.BELIEF] = belief_state
        
        # Infer intentions from action
        intention_content = await self._infer_intentions_from_action(action, context)
        if intention_content:
            intention_state = MentalState(
                agent_id=agent_id,
                state_type=MentalStateType.INTENTION,
                content=intention_content,
                confidence=0.6,
                evidence=[f"observed_action: {action}"],
                source="inference"
            )
            self.mental_states[agent_id][MentalStateType.INTENTION] = intention_state
        
        # Update attention state
        attention_content = await self._infer_attention_from_action(action, context)
        if attention_content:
            attention_state = MentalState(
                agent_id=agent_id,
                state_type=MentalStateType.ATTENTION,
                content=attention_content,
                confidence=0.8,
                evidence=[f"observed_action: {action}"],
                source="observation"
            )
            self.mental_states[agent_id][MentalStateType.ATTENTION] = attention_state
        
        # Store in history
        if agent_id not in self.state_history:
            self.state_history[agent_id] = []
        
        for state in self.mental_states[agent_id].values():
            self.state_history[agent_id].append(state)
            
        # Limit history size
        if len(self.state_history[agent_id]) > self.max_history_length:
            self.state_history[agent_id] = self.state_history[agent_id][-self.max_history_length:]
    
    async def _infer_beliefs_from_action(self, action: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Infer what the agent believes based on their action"""
        # Simple belief inference patterns
        belief_patterns = {
            'search': {'believes_information_exists': True, 'believes_search_will_help': True},
            'remember': {'believes_information_important': True, 'believes_memory_reliable': True},
            'think': {'believes_analysis_needed': True, 'believes_thinking_helps': True},
            'analyze': {'believes_patterns_exist': True, 'believes_analysis_valuable': True}
        }
        
        for pattern, beliefs in belief_patterns.items():
            if pattern in action.lower():
                return {
                    'inferred_beliefs': beliefs,
                    'action_trigger': action,
                    'context_factors': list(context.keys())
                }
        
        return None
    
    async def _infer_intentions_from_action(self, action: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Infer what the agent intends to do based on their action"""
        intention_patterns = {
            'search': 'find_information',
            'remember': 'store_knowledge',
            'think': 'analyze_situation',
            'analyze': 'understand_patterns',
            'create': 'generate_content',
            'update': 'modify_information'
        }
        
        for pattern, intention in intention_patterns.items():
            if pattern in action.lower():
                return {
                    'primary_intention': intention,
                    'action_sequence': [action],
                    'goal_indicators': context.get('goal_indicators', [])
                }
        
        return None
    
    async def _infer_attention_from_action(self, action: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Infer what the agent is paying attention to"""
        attention_indicators = []
        
        # Extract attention from context
        if 'query' in context:
            attention_indicators.append(f"query: {context['query']}")
        if 'topic' in context:
            attention_indicators.append(f"topic: {context['topic']}")
        if 'memory_type' in context:
            attention_indicators.append(f"memory_type: {context['memory_type']}")
        
        if attention_indicators:
            return {
                'focus_areas': attention_indicators,
                'attention_level': 'high' if len(attention_indicators) > 2 else 'medium',
                'duration_estimate': 60.0  # seconds
            }
        
        return None
    
    async def _update_intention_predictions(self, agent_id: str) -> None:
        """Update predictions about agent's future intentions"""
        if agent_id not in self.mental_states:
            return
        
        current_states = self.mental_states[agent_id]
        recent_patterns = self.interaction_patterns.get(agent_id, [])[-10:]  # Last 10 actions
        
        # Generate predictions based on current state and patterns
        predictions = []
        
        # Predict based on current intention
        if MentalStateType.INTENTION in current_states:
            intention_state = current_states[MentalStateType.INTENTION]
            primary_intention = intention_state.content.get('primary_intention')
            
            if primary_intention:
                prediction = IntentionPrediction(
                    agent_id=agent_id,
                    predicted_action=self._predict_next_action(primary_intention),
                    predicted_goal=primary_intention,
                    confidence=intention_state.confidence * 0.8,
                    time_horizon=60.0,
                    reasoning=f"Based on current intention: {primary_intention}"
                )
                predictions.append(prediction)
        
        # Predict based on patterns
        if len(recent_patterns) >= 3:
            pattern_prediction = await self._predict_from_patterns(agent_id, recent_patterns)
            if pattern_prediction:
                predictions.append(pattern_prediction)
        
        self.intention_predictions[agent_id] = predictions
    
    def _predict_next_action(self, intention: str) -> str:
        """Predict the next likely action based on intention"""
        action_mappings = {
            'find_information': 'search_or_recall',
            'store_knowledge': 'remember_or_save',
            'analyze_situation': 'think_or_analyze',
            'understand_patterns': 'analyze_or_compare',
            'generate_content': 'create_or_compose',
            'modify_information': 'update_or_edit'
        }
        
        return action_mappings.get(intention, 'unknown_action')
    
    async def _predict_from_patterns(self, agent_id: str, recent_patterns: List[Dict[str, Any]]) -> Optional[IntentionPrediction]:
        """Predict future intentions based on recent interaction patterns"""
        if len(recent_patterns) < 3:
            return None
        
        # Simple pattern analysis
        actions = [p['action'] for p in recent_patterns]
        action_frequency = {}
        
        for action in actions:
            action_frequency[action] = action_frequency.get(action, 0) + 1
        
        # Find most common action
        most_common_action = max(action_frequency, key=action_frequency.get)
        confidence = action_frequency[most_common_action] / len(actions)
        
        if confidence >= 0.5:
            return IntentionPrediction(
                agent_id=agent_id,
                predicted_action=most_common_action,
                predicted_goal=f"continue_pattern_{most_common_action}",
                confidence=confidence,
                time_horizon=120.0,
                reasoning=f"Pattern analysis: {most_common_action} appears {action_frequency[most_common_action]}/{len(actions)} times"
            )
        
        return None
    
    async def _identify_collaboration_opportunities(self, agent_id: str, action: str, 
                                                  context: Dict[str, Any]) -> None:
        """Identify opportunities for collaboration with other agents"""
        # Look for complementary goals or capabilities
        for other_agent_id, other_states in self.mental_states.items():
            if other_agent_id == agent_id:
                continue
            
            # Check for complementary intentions
            if MentalStateType.INTENTION in other_states:
                other_intention = other_states[MentalStateType.INTENTION]
                collaboration = await self._assess_collaboration_potential(
                    agent_id, other_agent_id, action, other_intention
                )
                
                if collaboration:
                    opportunity_id = f"{agent_id}_{other_agent_id}_{int(time.time())}"
                    self.collaboration_opportunities[opportunity_id] = collaboration
    
    async def _assess_collaboration_potential(self, agent1_id: str, agent2_id: str, 
                                           action: str, other_intention: MentalState) -> Optional[CollaborationOpportunity]:
        """Assess potential for collaboration between two agents"""
        # Simple collaboration assessment
        complementary_pairs = {
            ('search', 'analyze'): 'complementary',
            ('remember', 'think'): 'complementary',
            ('create', 'analyze'): 'sequential',
            ('search', 'search'): 'parallel'
        }
        
        other_action = other_intention.content.get('primary_intention', '')
        
        for (action1, action2), collab_type in complementary_pairs.items():
            if (action1 in action.lower() and action2 in other_action.lower()) or \
               (action2 in action.lower() and action1 in other_action.lower()):
                
                return CollaborationOpportunity(
                    opportunity_id="",  # Will be set by caller
                    involved_agents=[agent1_id, agent2_id],
                    shared_goal=f"collaborative_{action1}_{action2}",
                    collaboration_type=collab_type,
                    potential_benefit=0.7,
                    coordination_requirements=[f"sync_{action1}_with_{action2}"]
                )
        
        return None
    
    async def get_agent_mental_state(self, agent_id: str, state_type: MentalStateType = None) -> Dict[str, Any]:
        """Get the current mental state model for an agent"""
        if agent_id not in self.mental_states:
            return {}
        
        if state_type:
            state = self.mental_states[agent_id].get(state_type)
            return {
                'agent_id': agent_id,
                'state_type': state_type.value,
                'state': state.__dict__ if state else None
            }
        
        return {
            'agent_id': agent_id,
            'mental_states': {
                state_type.value: state.__dict__ 
                for state_type, state in self.mental_states[agent_id].items()
            }
        }
    
    async def predict_agent_behavior(self, agent_id: str, time_horizon: float = None) -> List[IntentionPrediction]:
        """Predict an agent's future behavior"""
        time_horizon = time_horizon or self.prediction_horizon
        
        predictions = self.intention_predictions.get(agent_id, [])
        
        # Filter predictions within time horizon
        current_time = time.time()
        relevant_predictions = [
            p for p in predictions 
            if p.time_horizon <= time_horizon
        ]
        
        return relevant_predictions
    
    async def get_collaboration_opportunities(self, agent_id: str = None) -> List[CollaborationOpportunity]:
        """Get available collaboration opportunities"""
        opportunities = list(self.collaboration_opportunities.values())
        
        if agent_id:
            opportunities = [
                opp for opp in opportunities 
                if agent_id in opp.involved_agents
            ]
        
        return opportunities
    
    async def generate_empathetic_response(self, agent_id: str, situation: str) -> Optional[str]:
        """Generate an empathetic response based on agent's mental state"""
        if agent_id not in self.mental_states:
            return None
        
        # Simple empathy response generation
        mental_states = self.mental_states[agent_id]
        
        if MentalStateType.EMOTION in mental_states:
            emotion_state = mental_states[MentalStateType.EMOTION]
            emotion = emotion_state.content.get('primary_emotion', 'neutral')
            
            empathy_templates = {
                'frustrated': "I understand this might be frustrating. Let me help you find a solution.",
                'confused': "I can see this is confusing. Let me break it down step by step.",
                'excited': "I can sense your enthusiasm! That's great energy to work with.",
                'worried': "I understand your concerns. Let's work through this together.",
                'neutral': "I'm here to help you with whatever you need."
            }
            
            return empathy_templates.get(emotion, empathy_templates['neutral'])
        
        return "I'm here to help you achieve your goals."
    
    async def _initialize_mental_models(self) -> None:
        """Initialize mental state models"""
        # Initialize empty models - will be populated through observation
        pass
    
    async def _load_interaction_history(self) -> None:
        """Load historical interaction patterns"""
        # Implementation would load from persistent storage
        pass
    
    async def _initialize_empathy_models(self) -> None:
        """Initialize empathy and emotional modeling"""
        # Initialize basic emotional response patterns
        self.empathy_responses = {
            'support': ["I'm here to help", "Let's work through this together"],
            'encouragement': ["You're doing great", "Keep up the good work"],
            'understanding': ["I understand", "That makes sense"],
            'clarification': ["Let me clarify", "To make sure I understand"]
        }
    
    def get_theory_of_mind_metrics(self) -> Dict[str, Any]:
        """Get metrics about theory of mind performance"""
        total_agents = len(self.mental_states)
        total_predictions = sum(len(preds) for preds in self.intention_predictions.values())
        total_collaborations = len(self.collaboration_opportunities)
        
        avg_prediction_accuracy = 0.0
        if self.prediction_accuracy:
            all_accuracies = [acc for accs in self.prediction_accuracy.values() for acc in accs]
            if all_accuracies:
                avg_prediction_accuracy = sum(all_accuracies) / len(all_accuracies)
        
        return {
            'tracked_agents': total_agents,
            'active_predictions': total_predictions,
            'collaboration_opportunities': total_collaborations,
            'average_prediction_accuracy': avg_prediction_accuracy,
            'interaction_patterns_learned': sum(len(patterns) for patterns in self.interaction_patterns.values()),
            'mental_state_types_modeled': len(MentalStateType),
            'empathy_responses_available': len(self.empathy_responses)
        }
