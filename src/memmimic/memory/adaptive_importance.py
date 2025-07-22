"""
Adaptive Learning System for Memory Importance Scoring with Reinforcement Learning.

Implements intelligent importance scoring that learns from user interactions, access patterns,
and system performance to continuously improve memory management decisions.
"""

import logging
import numpy as np
import pickle
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import math

logger = logging.getLogger(__name__)


@dataclass
class MemoryInteraction:
    """Represents a user interaction with a memory item."""
    memory_id: str
    interaction_type: str  # access, search, modify, delete, archive
    timestamp: datetime
    user_context: Dict[str, Any] = field(default_factory=dict)
    explicit_feedback: Optional[float] = None  # User rating -1 to 1
    implicit_feedback: float = 0.0  # Derived from behavior
    session_id: Optional[str] = None
    relevance_score: float = 0.5  # How relevant was this to user's intent


@dataclass 
class MemoryState:
    """Represents the current state of a memory item for RL."""
    memory_id: str
    content_features: np.ndarray  # Semantic features
    temporal_features: np.ndarray  # Time-based features
    usage_features: np.ndarray  # Access pattern features
    context_features: np.ndarray  # Context and metadata features
    current_importance: float = 0.5
    age_days: float = 0.0
    access_count: int = 0
    last_access_hours: float = 24.0
    memory_type: str = "interaction"


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive importance learning."""
    # Reinforcement Learning parameters
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    
    # Feature dimensions
    content_feature_dim: int = 64
    temporal_feature_dim: int = 12
    usage_feature_dim: int = 8
    context_feature_dim: int = 16
    
    # Learning parameters
    batch_size: int = 32
    memory_replay_size: int = 1000
    update_frequency: int = 10
    target_network_update: int = 100
    
    # Importance scoring
    importance_decay_rate: float = 0.02
    min_importance_score: float = 0.0
    max_importance_score: float = 1.0
    
    # System parameters
    max_history_size: int = 10000
    model_save_interval_minutes: int = 60
    performance_evaluation_interval: int = 100


class ImportanceQLearning:
    """Q-Learning based importance scorer for memory items."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.state_dim = (
            config.content_feature_dim + 
            config.temporal_feature_dim + 
            config.usage_feature_dim + 
            config.context_feature_dim
        )
        
        # Q-network (simple neural network approximation)
        self.q_network = self._initialize_q_network()
        self.target_network = self._initialize_q_network()
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=config.memory_replay_size)
        
        # Learning state
        self.epsilon = config.exploration_rate
        self.step_count = 0
        self.update_count = 0
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        
    def _initialize_q_network(self) -> Dict[str, np.ndarray]:
        """Initialize Q-network with random weights."""
        # Simple 2-layer network: input -> hidden -> output
        hidden_size = max(64, self.state_dim // 2)
        
        return {
            'W1': np.random.normal(0, 0.1, (self.state_dim, hidden_size)),
            'b1': np.zeros(hidden_size),
            'W2': np.random.normal(0, 0.1, (hidden_size, 1)),  # Single output for importance
            'b2': np.zeros(1)
        }
    
    def _forward_pass(self, state: np.ndarray, network: Dict[str, np.ndarray]) -> float:
        """Forward pass through the network."""
        # Hidden layer with ReLU activation
        hidden = np.maximum(0, np.dot(state, network['W1']) + network['b1'])
        
        # Output layer with sigmoid activation for importance score
        output = np.dot(hidden, network['W2']) + network['b2']
        importance = 1.0 / (1.0 + np.exp(-output[0]))  # Sigmoid
        
        return importance
    
    def predict_importance(self, state: MemoryState) -> float:
        """Predict importance score for a memory state."""
        state_vector = self._encode_state(state)
        importance = self._forward_pass(state_vector, self.q_network)
        
        # Add exploration noise during training
        if np.random.random() < self.epsilon:
            noise = np.random.normal(0, 0.1)
            importance = np.clip(importance + noise, 0.0, 1.0)
        
        return importance
    
    def update_importance(self, 
                         state: MemoryState, 
                         reward: float, 
                         next_state: Optional[MemoryState] = None):
        """Update Q-network based on observed reward."""
        state_vector = self._encode_state(state)
        
        # Store experience for replay learning
        experience = {
            'state': state_vector,
            'reward': reward,
            'next_state': self._encode_state(next_state) if next_state else None,
            'timestamp': datetime.now()
        }
        self.experience_buffer.append(experience)
        
        self.step_count += 1
        
        # Update network periodically using experience replay
        if (len(self.experience_buffer) >= self.config.batch_size and 
            self.step_count % self.config.update_frequency == 0):
            self._update_network()
    
    def _encode_state(self, state: MemoryState) -> np.ndarray:
        """Encode memory state into feature vector."""
        return np.concatenate([
            state.content_features,
            state.temporal_features, 
            state.usage_features,
            state.context_features
        ])
    
    def _update_network(self):
        """Update Q-network using experience replay."""
        if len(self.experience_buffer) < self.config.batch_size:
            return
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(
            len(self.experience_buffer), 
            self.config.batch_size, 
            replace=False
        )
        batch_experiences = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare batch data
        states = np.array([exp['state'] for exp in batch_experiences])
        rewards = np.array([exp['reward'] for exp in batch_experiences])
        next_states = np.array([
            exp['next_state'] if exp['next_state'] is not None else np.zeros(self.state_dim)
            for exp in batch_experiences
        ])
        
        # Compute target values using target network
        targets = []
        for i, exp in enumerate(batch_experiences):
            if exp['next_state'] is not None:
                next_q_value = self._forward_pass(next_states[i], self.target_network)
                target = rewards[i] + self.config.discount_factor * next_q_value
            else:
                target = rewards[i]
            targets.append(target)
        
        targets = np.array(targets)
        
        # Update main network (simplified gradient descent)
        for i in range(len(batch_experiences)):
            current_q = self._forward_pass(states[i], self.q_network)
            error = targets[i] - current_q
            
            # Simple gradient update (this would be more sophisticated in practice)
            self._apply_gradient_update(states[i], error)
        
        self.update_count += 1
        
        # Update target network periodically
        if self.update_count % self.config.target_network_update == 0:
            self.target_network = {k: v.copy() for k, v in self.q_network.items()}
        
        # Decay exploration rate
        self.epsilon = max(
            self.config.min_exploration_rate,
            self.epsilon * self.config.exploration_decay
        )
    
    def _apply_gradient_update(self, state: np.ndarray, error: float):
        """Apply gradient update to network weights (simplified)."""
        # This is a simplified version - in practice you'd use proper backpropagation
        learning_rate = self.config.learning_rate
        
        # Update output layer
        hidden = np.maximum(0, np.dot(state, self.q_network['W1']) + self.q_network['b1'])
        
        # Gradient for output layer
        self.q_network['W2'] += learning_rate * error * hidden.reshape(-1, 1)
        self.q_network['b2'] += learning_rate * error
        
        # Gradient for hidden layer (simplified)
        hidden_error = error * self.q_network['W2'].flatten()
        hidden_error[hidden <= 0] = 0  # ReLU derivative
        
        self.q_network['W1'] += learning_rate * np.outer(state, hidden_error)
        self.q_network['b1'] += learning_rate * hidden_error
    
    def save_model(self, filepath: Path):
        """Save Q-network model to file."""
        model_data = {
            'q_network': self.q_network,
            'target_network': self.target_network,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'update_count': self.update_count,
            'performance_history': list(self.performance_history)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: Path):
        """Load Q-network model from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_network = model_data['q_network']
            self.target_network = model_data['target_network']
            self.epsilon = model_data['epsilon']
            self.step_count = model_data['step_count']
            self.update_count = model_data['update_count']
            self.performance_history = deque(model_data['performance_history'], maxlen=1000)
            
            logger.info(f"Loaded Q-learning model from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load Q-learning model: {e}")
            # Initialize with random weights
            self.__init__(self.config)


class FeatureExtractor:
    """Extracts features from memory items for ML processing."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        
        # Feature extraction models (simplified - in practice would use embeddings)
        self.content_vocabulary = {}
        self.temporal_patterns = {}
        
    def extract_content_features(self, memory_content: str, memory_metadata: Dict) -> np.ndarray:
        """Extract semantic content features."""
        # Simplified content feature extraction
        # In practice, this would use sentence embeddings or similar
        
        features = np.zeros(self.config.content_feature_dim)
        
        # Basic text statistics
        if memory_content:
            features[0] = len(memory_content) / 1000.0  # Length normalized
            features[1] = len(memory_content.split()) / 100.0  # Word count normalized
            features[2] = memory_content.count('?') / max(len(memory_content), 1)  # Question ratio
            features[3] = memory_content.count('!') / max(len(memory_content), 1)  # Exclamation ratio
        
        # Memory type features
        memory_type = memory_metadata.get('memory_type', 'unknown')
        type_mapping = {
            'milestone': 0.9,
            'reflection': 0.7, 
            'interaction': 0.5,
            'system': 0.3,
            'temporary': 0.1
        }
        features[4] = type_mapping.get(memory_type, 0.5)
        
        # CXD classification features
        cxd_functions = memory_metadata.get('cxd_functions', [])
        if cxd_functions:
            features[5] = len([f for f in cxd_functions if f == 'CONTROL']) / len(cxd_functions)
            features[6] = len([f for f in cxd_functions if f == 'CONTEXT']) / len(cxd_functions)
            features[7] = len([f for f in cxd_functions if f == 'DATA']) / len(cxd_functions)
        
        # Confidence and quality features
        features[8] = memory_metadata.get('confidence', 0.5)
        features[9] = memory_metadata.get('importance', 0.5)
        
        # Fill remaining features with noise to prevent overfitting
        remaining = self.config.content_feature_dim - 10
        features[10:10+remaining] = np.random.normal(0, 0.1, remaining)
        
        return features[:self.config.content_feature_dim]
    
    def extract_temporal_features(self, memory_timestamp: datetime, current_time: datetime) -> np.ndarray:
        """Extract time-based features."""
        features = np.zeros(self.config.temporal_feature_dim)
        
        time_diff = current_time - memory_timestamp
        hours_diff = time_diff.total_seconds() / 3600
        
        # Age features
        features[0] = min(1.0, hours_diff / (24 * 30))  # Age in months, capped at 1
        features[1] = min(1.0, hours_diff / (24 * 7))   # Age in weeks, capped at 1
        features[2] = min(1.0, hours_diff / 24)         # Age in days, capped at 1
        
        # Temporal context
        features[3] = memory_timestamp.hour / 23.0       # Hour of day
        features[4] = memory_timestamp.weekday() / 6.0   # Day of week
        features[5] = (memory_timestamp.month - 1) / 11.0 # Month of year
        
        # Recency decay
        features[6] = np.exp(-hours_diff / (24 * 7))     # Weekly decay
        features[7] = np.exp(-hours_diff / (24 * 30))    # Monthly decay
        
        # Current time context
        features[8] = current_time.hour / 23.0
        features[9] = current_time.weekday() / 6.0
        
        # Time-based importance modifiers
        features[10] = 1.0 if hours_diff < 24 else 0.0  # Recently created
        features[11] = 1.0 if hours_diff > 24*30 else 0.0  # Old memory
        
        return features
    
    def extract_usage_features(self, access_history: List[datetime], current_time: datetime) -> np.ndarray:
        """Extract usage pattern features."""
        features = np.zeros(self.config.usage_feature_dim)
        
        if not access_history:
            return features
        
        # Basic access statistics
        features[0] = min(1.0, len(access_history) / 100.0)  # Total accesses normalized
        
        # Recent access patterns
        recent_accesses = [
            t for t in access_history 
            if (current_time - t).total_seconds() / 3600 <= 24
        ]
        features[1] = min(1.0, len(recent_accesses) / 10.0)  # Recent accesses
        
        # Access frequency
        if len(access_history) > 1:
            time_span = (access_history[-1] - access_history[0]).total_seconds() / 3600
            frequency = len(access_history) / max(time_span, 1.0)
            features[2] = min(1.0, frequency / 1.0)  # Accesses per hour
        
        # Access pattern regularity
        if len(access_history) > 2:
            intervals = []
            for i in range(1, len(access_history)):
                interval = (access_history[i] - access_history[i-1]).total_seconds() / 3600
                intervals.append(interval)
            
            if intervals:
                features[3] = min(1.0, np.std(intervals) / np.mean(intervals))  # Coefficient of variation
        
        # Last access recency
        last_access = max(access_history)
        hours_since_last = (current_time - last_access).total_seconds() / 3600
        features[4] = np.exp(-hours_since_last / 24)  # Recency score
        
        # Access trend (increasing/decreasing)
        if len(access_history) > 4:
            recent_count = len([t for t in access_history[-5:] if (current_time - t).hours <= 24])
            older_count = len([t for t in access_history[-10:-5] if (current_time - t).hours <= 48])
            if older_count > 0:
                features[5] = min(2.0, recent_count / older_count) / 2.0  # Trend indicator
        
        # Peak access times
        if access_history:
            hours = [t.hour for t in access_history]
            most_common_hour = max(set(hours), key=hours.count)
            features[6] = most_common_hour / 23.0
            
            # Weekend vs weekday usage
            weekday_accesses = [t for t in access_history if t.weekday() < 5]
            weekend_accesses = [t for t in access_history if t.weekday() >= 5]
            if len(access_history) > 0:
                features[7] = len(weekday_accesses) / len(access_history)
        
        return features
    
    def extract_context_features(self, memory_metadata: Dict, user_context: Dict) -> np.ndarray:
        """Extract contextual and metadata features."""
        features = np.zeros(self.config.context_feature_dim)
        
        # Memory metadata features
        features[0] = memory_metadata.get('user_rating', 0.5)
        features[1] = memory_metadata.get('system_confidence', 0.5)
        features[2] = memory_metadata.get('relevance_score', 0.5)
        
        # Content type indicators
        content_type = memory_metadata.get('content_type', 'text')
        type_features = {
            'text': [1, 0, 0],
            'code': [0, 1, 0],
            'data': [0, 0, 1]
        }
        type_vec = type_features.get(content_type, [0, 0, 0])
        features[3:6] = type_vec
        
        # Source and origin features
        features[6] = 1.0 if memory_metadata.get('user_generated', False) else 0.0
        features[7] = 1.0 if memory_metadata.get('system_generated', False) else 0.0
        features[8] = memory_metadata.get('quality_score', 0.5)
        
        # User context features
        features[9] = user_context.get('session_activity_level', 0.5)
        features[10] = user_context.get('task_relevance', 0.5)
        features[11] = 1.0 if user_context.get('explicit_search', False) else 0.0
        
        # Interaction type features
        interaction_type = user_context.get('last_interaction', 'view')
        interaction_weights = {
            'view': 0.2,
            'search': 0.4, 
            'modify': 0.8,
            'share': 0.9,
            'bookmark': 1.0
        }
        features[12] = interaction_weights.get(interaction_type, 0.2)
        
        # Temporal context
        features[13] = 1.0 if user_context.get('peak_hours', False) else 0.0
        features[14] = user_context.get('attention_score', 0.5)
        features[15] = user_context.get('task_complexity', 0.5)
        
        return features


class AdaptiveImportanceScorer:
    """
    Adaptive learning system for memory importance scoring using reinforcement learning.
    
    Features:
    - Q-learning based importance prediction
    - Multi-dimensional feature extraction
    - Continuous learning from user interactions
    - Performance-based model adaptation
    - Explainable importance decisions
    """
    
    def __init__(self, config: Optional[AdaptiveConfig] = None, model_path: Optional[Path] = None):
        """
        Initialize adaptive importance scorer.
        
        Args:
            config: Configuration for adaptive learning
            model_path: Path to save/load models
        """
        self.config = config or AdaptiveConfig()
        self.model_path = model_path or Path("models/adaptive_importance")
        
        # Core components
        self.q_learner = ImportanceQLearning(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        
        # State tracking
        self.interaction_history: Dict[str, List[MemoryInteraction]] = defaultdict(list)
        self.memory_states: Dict[str, MemoryState] = {}
        self.performance_metrics = {
            'prediction_accuracy': 0.0,
            'user_satisfaction': 0.0,
            'system_performance': 0.0,
            'total_interactions': 0,
            'successful_predictions': 0
        }
        
        # Learning coordination
        self._learning_lock = threading.RLock()
        self._last_model_save = datetime.now()
        self._learning_active = True
        
        # Load existing model if available
        self._load_model_if_exists()
        
        logger.info("AdaptiveImportanceScorer initialized")
    
    def score_importance(self, 
                        memory_id: str,
                        memory_content: str,
                        memory_metadata: Dict[str, Any],
                        user_context: Optional[Dict[str, Any]] = None,
                        current_time: Optional[datetime] = None) -> float:
        """
        Score memory importance using adaptive learning.
        
        Args:
            memory_id: Unique memory identifier
            memory_content: Memory content text
            memory_metadata: Memory metadata and context
            user_context: Current user context
            current_time: Current timestamp
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        current_time = current_time or datetime.now()
        user_context = user_context or {}
        
        with self._learning_lock:
            # Extract features
            memory_timestamp = memory_metadata.get('created_at', current_time)
            access_history = [
                interaction.timestamp 
                for interaction in self.interaction_history[memory_id]
            ]
            
            content_features = self.feature_extractor.extract_content_features(
                memory_content, memory_metadata
            )
            temporal_features = self.feature_extractor.extract_temporal_features(
                memory_timestamp, current_time
            )
            usage_features = self.feature_extractor.extract_usage_features(
                access_history, current_time
            )
            context_features = self.feature_extractor.extract_context_features(
                memory_metadata, user_context
            )
            
            # Create memory state
            memory_state = MemoryState(
                memory_id=memory_id,
                content_features=content_features,
                temporal_features=temporal_features,
                usage_features=usage_features,
                context_features=context_features,
                current_importance=memory_metadata.get('importance', 0.5),
                age_days=(current_time - memory_timestamp).days,
                access_count=len(access_history),
                last_access_hours=(
                    (current_time - max(access_history)).total_seconds() / 3600 
                    if access_history else 24.0
                ),
                memory_type=memory_metadata.get('memory_type', 'interaction')
            )
            
            # Update state tracking
            self.memory_states[memory_id] = memory_state
            
            # Predict importance using Q-learning
            predicted_importance = self.q_learner.predict_importance(memory_state)
            
            # Apply decay and bounds
            decayed_importance = self._apply_temporal_decay(
                predicted_importance, memory_state.age_days
            )
            
            final_importance = np.clip(
                decayed_importance,
                self.config.min_importance_score,
                self.config.max_importance_score
            )
            
            return final_importance
    
    def record_interaction(self,
                          memory_id: str,
                          interaction_type: str,
                          user_context: Optional[Dict[str, Any]] = None,
                          explicit_feedback: Optional[float] = None,
                          relevance_score: float = 0.5):
        """
        Record user interaction for learning.
        
        Args:
            memory_id: Memory identifier
            interaction_type: Type of interaction (access, search, modify, etc.)
            user_context: User context information
            explicit_feedback: Optional explicit user feedback (-1 to 1)
            relevance_score: How relevant this was to user's intent
        """
        interaction = MemoryInteraction(
            memory_id=memory_id,
            interaction_type=interaction_type,
            timestamp=datetime.now(),
            user_context=user_context or {},
            explicit_feedback=explicit_feedback,
            relevance_score=relevance_score,
            session_id=user_context.get('session_id') if user_context else None
        )
        
        # Calculate implicit feedback based on interaction type
        interaction.implicit_feedback = self._calculate_implicit_feedback(interaction)
        
        with self._learning_lock:
            self.interaction_history[memory_id].append(interaction)
            
            # Limit history size per memory
            max_history = self.config.max_history_size // max(len(self.memory_states), 1)
            if len(self.interaction_history[memory_id]) > max_history:
                self.interaction_history[memory_id] = self.interaction_history[memory_id][-max_history:]
            
            # Update learning model
            self._update_learning_model(interaction)
            
            # Update performance metrics
            self.performance_metrics['total_interactions'] += 1
        
        logger.debug(f"Recorded interaction: {interaction_type} for {memory_id}")
    
    def _calculate_implicit_feedback(self, interaction: MemoryInteraction) -> float:
        """Calculate implicit feedback score from interaction."""
        base_scores = {
            'access': 0.3,
            'search': 0.2,
            'view': 0.1,
            'modify': 0.8,
            'delete': -0.9,
            'archive': -0.5,
            'share': 0.9,
            'bookmark': 0.8,
            'copy': 0.6,
            'reference': 0.7
        }
        
        base_score = base_scores.get(interaction.interaction_type, 0.1)
        
        # Adjust based on context
        context = interaction.user_context
        
        # Time spent (if available)
        time_spent = context.get('time_spent_seconds', 0)
        if time_spent > 0:
            time_factor = min(1.0, time_spent / 60.0)  # Normalize to 1 minute
            base_score += time_factor * 0.2
        
        # Search result ranking (if from search)
        if interaction.interaction_type == 'search':
            rank = context.get('search_rank', 10)
            rank_factor = max(0, 1.0 - rank / 10.0)  # Higher rank = better
            base_score += rank_factor * 0.3
        
        # Task relevance
        task_relevance = context.get('task_relevance', 0.5)
        base_score += (task_relevance - 0.5) * 0.4
        
        return np.clip(base_score, -1.0, 1.0)
    
    def _update_learning_model(self, interaction: MemoryInteraction):
        """Update Q-learning model based on interaction."""
        memory_id = interaction.memory_id
        
        if memory_id not in self.memory_states:
            return  # No state to update
        
        current_state = self.memory_states[memory_id]
        
        # Calculate reward based on feedback
        reward = self._calculate_reward(interaction, current_state)
        
        # Update Q-learning model
        self.q_learner.update_importance(
            state=current_state,
            reward=reward,
            next_state=None  # Terminal state for this interaction
        )
        
        # Update performance tracking
        if abs(reward) > 0.5:  # Significant feedback
            predicted_importance = self.q_learner.predict_importance(current_state)
            actual_importance = self._infer_actual_importance(interaction)
            
            prediction_error = abs(predicted_importance - actual_importance)
            
            # Update accuracy tracking
            if prediction_error < 0.2:  # Within 20% = successful prediction
                self.performance_metrics['successful_predictions'] += 1
            
            total_predictions = max(1, self.performance_metrics['total_interactions'] // 10)
            self.performance_metrics['prediction_accuracy'] = (
                self.performance_metrics['successful_predictions'] / total_predictions
            )
    
    def _calculate_reward(self, interaction: MemoryInteraction, state: MemoryState) -> float:
        """Calculate reward signal for learning."""
        # Base reward from explicit and implicit feedback
        reward = 0.0
        
        if interaction.explicit_feedback is not None:
            reward += interaction.explicit_feedback * 0.7
        
        reward += interaction.implicit_feedback * 0.5
        
        # Relevance reward
        reward += (interaction.relevance_score - 0.5) * 0.3
        
        # Time-based reward adjustments
        if interaction.interaction_type in ['access', 'search', 'view']:
            # Recent access to high-importance items should be rewarded more
            time_factor = np.exp(-state.last_access_hours / 24.0)
            reward += time_factor * 0.2
        
        # Memory type importance
        type_importance = {
            'milestone': 0.2,
            'reflection': 0.1,
            'interaction': 0.0,
            'system': -0.1,
            'temporary': -0.2
        }
        reward += type_importance.get(state.memory_type, 0.0)
        
        return np.clip(reward, -1.0, 1.0)
    
    def _infer_actual_importance(self, interaction: MemoryInteraction) -> float:
        """Infer actual importance from user behavior."""
        # Simple heuristic - in practice this would be more sophisticated
        if interaction.explicit_feedback is not None:
            return (interaction.explicit_feedback + 1.0) / 2.0  # Convert -1,1 to 0,1
        
        # Infer from interaction type and context
        type_importance = {
            'modify': 0.9,
            'share': 0.8,
            'bookmark': 0.8,
            'reference': 0.7,
            'copy': 0.6,
            'access': 0.5,
            'view': 0.4,
            'search': 0.3,
            'archive': 0.2,
            'delete': 0.1
        }
        
        base_importance = type_importance.get(interaction.interaction_type, 0.5)
        
        # Adjust based on relevance
        adjusted_importance = (
            base_importance * 0.7 + 
            interaction.relevance_score * 0.3
        )
        
        return np.clip(adjusted_importance, 0.0, 1.0)
    
    def _apply_temporal_decay(self, importance: float, age_days: float) -> float:
        """Apply temporal decay to importance score."""
        decay_factor = np.exp(-age_days * self.config.importance_decay_rate)
        return importance * decay_factor
    
    def get_explanation(self, memory_id: str) -> Dict[str, Any]:
        """
        Get explanation for importance score.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Dictionary explaining the importance score
        """
        if memory_id not in self.memory_states:
            return {'error': 'Memory state not found'}
        
        state = self.memory_states[memory_id]
        interactions = self.interaction_history[memory_id]
        
        # Feature contributions (simplified)
        content_contribution = np.mean(state.content_features) * 0.3
        temporal_contribution = np.mean(state.temporal_features) * 0.2
        usage_contribution = np.mean(state.usage_features) * 0.3
        context_contribution = np.mean(state.context_features) * 0.2
        
        return {
            'memory_id': memory_id,
            'current_importance': state.current_importance,
            'feature_contributions': {
                'content': content_contribution,
                'temporal': temporal_contribution,
                'usage': usage_contribution,
                'context': context_contribution
            },
            'key_factors': {
                'age_days': state.age_days,
                'access_count': state.access_count,
                'last_access_hours': state.last_access_hours,
                'memory_type': state.memory_type,
                'recent_interactions': len([
                    i for i in interactions 
                    if (datetime.now() - i.timestamp).days < 7
                ])
            },
            'learning_confidence': min(1.0, len(interactions) / 10.0),
            'model_version': f"QL_{self.q_learner.step_count}"
        }
    
    def optimize_model(self) -> Dict[str, Any]:
        """
        Optimize model based on performance metrics.
        
        Returns:
            Dictionary with optimization results
        """
        optimization_results = {
            'optimizations_applied': 0,
            'improvements': [],
        }
        
        try:
            # Adjust learning rate based on performance
            current_accuracy = self.performance_metrics['prediction_accuracy']
            
            if current_accuracy < 0.6:
                # Low accuracy, increase learning rate
                old_lr = self.config.learning_rate
                self.config.learning_rate = min(0.1, old_lr * 1.2)
                
                if self.config.learning_rate != old_lr:
                    optimization_results['optimizations_applied'] += 1
                    optimization_results['improvements'].append(
                        f'Increased learning rate from {old_lr:.4f} to {self.config.learning_rate:.4f}'
                    )
            
            elif current_accuracy > 0.85:
                # High accuracy, can reduce learning rate for stability
                old_lr = self.config.learning_rate
                self.config.learning_rate = max(0.001, old_lr * 0.9)
                
                if self.config.learning_rate != old_lr:
                    optimization_results['optimizations_applied'] += 1
                    optimization_results['improvements'].append(
                        f'Reduced learning rate from {old_lr:.4f} to {self.config.learning_rate:.4f}'
                    )
            
            # Adjust exploration rate
            total_interactions = self.performance_metrics['total_interactions']
            if total_interactions > 1000:
                # After sufficient exploration, reduce exploration rate
                old_epsilon = self.q_learner.epsilon
                target_epsilon = max(0.01, 0.1 * np.exp(-total_interactions / 10000))
                self.q_learner.epsilon = target_epsilon
                
                if abs(old_epsilon - target_epsilon) > 0.01:
                    optimization_results['optimizations_applied'] += 1
                    optimization_results['improvements'].append(
                        f'Adjusted exploration rate from {old_epsilon:.3f} to {target_epsilon:.3f}'
                    )
            
            # Clean up old interaction history
            cleanup_count = 0
            cutoff_date = datetime.now() - timedelta(days=30)
            
            with self._learning_lock:
                for memory_id in list(self.interaction_history.keys()):
                    old_count = len(self.interaction_history[memory_id])
                    self.interaction_history[memory_id] = [
                        i for i in self.interaction_history[memory_id]
                        if i.timestamp > cutoff_date
                    ]
                    cleanup_count += old_count - len(self.interaction_history[memory_id])
            
            if cleanup_count > 0:
                optimization_results['optimizations_applied'] += 1
                optimization_results['improvements'].append(
                    f'Cleaned up {cleanup_count} old interaction records'
                )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            optimization_results['error'] = str(e)
            return optimization_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._learning_lock:
            total_memories = len(self.memory_states)
            total_interactions = sum(len(hist) for hist in self.interaction_history.values())
            
            # Calculate average importance scores by type
            importance_by_type = defaultdict(list)
            for state in self.memory_states.values():
                importance_by_type[state.memory_type].append(state.current_importance)
            
            avg_importance_by_type = {
                mem_type: np.mean(scores) 
                for mem_type, scores in importance_by_type.items()
            }
            
            # Q-learning specific stats
            q_stats = {
                'epsilon': self.q_learner.epsilon,
                'step_count': self.q_learner.step_count,
                'update_count': self.q_learner.update_count,
                'experience_buffer_size': len(self.q_learner.experience_buffer)
            }
            
        return {
            'learning_performance': self.performance_metrics,
            'data_statistics': {
                'total_memories_tracked': total_memories,
                'total_interactions': total_interactions,
                'avg_interactions_per_memory': total_interactions / max(total_memories, 1),
                'avg_importance_by_type': avg_importance_by_type
            },
            'model_status': {
                'q_learning': q_stats,
                'learning_active': self._learning_active,
                'last_model_save': self._last_model_save.isoformat(),
                'config': {
                    'learning_rate': self.config.learning_rate,
                    'discount_factor': self.config.discount_factor,
                    'exploration_rate': self.q_learner.epsilon
                }
            }
        }
    
    def _save_model_periodically(self):
        """Save model if enough time has passed."""
        if (datetime.now() - self._last_model_save).total_seconds() > self.config.model_save_interval_minutes * 60:
            self.save_model()
    
    def save_model(self):
        """Save the adaptive learning model."""
        try:
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Save Q-learning model
            q_model_path = self.model_path / "q_learning_model.pkl"
            self.q_learner.save_model(q_model_path)
            
            # Save interaction history and other state
            state_data = {
                'interaction_history': dict(self.interaction_history),
                'performance_metrics': self.performance_metrics,
                'config': self.config.__dict__,
                'model_timestamp': datetime.now().isoformat()
            }
            
            state_path = self.model_path / "learning_state.json"
            with open(state_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            self._last_model_save = datetime.now()
            logger.info(f"Saved adaptive importance model to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _load_model_if_exists(self):
        """Load existing model if available."""
        try:
            q_model_path = self.model_path / "q_learning_model.pkl"
            if q_model_path.exists():
                self.q_learner.load_model(q_model_path)
            
            state_path = self.model_path / "learning_state.json"
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state_data = json.load(f)
                
                # Load interaction history
                for memory_id, interactions in state_data.get('interaction_history', {}).items():
                    self.interaction_history[memory_id] = [
                        MemoryInteraction(**interaction) for interaction in interactions
                    ]
                
                # Load performance metrics
                self.performance_metrics.update(state_data.get('performance_metrics', {}))
                
                logger.info("Loaded existing adaptive importance model")
                
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
    
    def shutdown(self):
        """Shutdown adaptive importance scorer."""
        logger.info("Shutting down adaptive importance scorer...")
        
        self._learning_active = False
        
        # Save final model state
        self.save_model()
        
        logger.info("Adaptive importance scorer shutdown complete")


# Factory function for easy integration
def create_adaptive_importance_scorer(
    config: Optional[AdaptiveConfig] = None,
    model_path: Optional[Path] = None
) -> AdaptiveImportanceScorer:
    """
    Factory function to create adaptive importance scorer.
    
    Args:
        config: Optional adaptive learning configuration
        model_path: Optional path for saving/loading models
        
    Returns:
        AdaptiveImportanceScorer instance
    """
    return AdaptiveImportanceScorer(config=config, model_path=model_path)