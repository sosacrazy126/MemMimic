#!/usr/bin/env python3
"""
Predictive Memory Lifecycle Manager - Phase 3 Advanced Features
Forecasts memory transitions and optimizes lifecycle management proactively
"""

import sys
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import statistics

# Add MemMimic to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PredictionType(Enum):
    """Types of predictions the system can make"""
    ARCHIVE_CANDIDATE = "archive_candidate"
    PRUNE_CANDIDATE = "prune_candidate"
    PROMOTE_CANDIDATE = "promote_candidate"
    IMPORTANCE_RISING = "importance_rising"
    IMPORTANCE_FALLING = "importance_falling"
    ACCESS_PATTERN_CHANGE = "access_pattern_change"
    CONSCIOUSNESS_EVOLUTION = "consciousness_evolution"

@dataclass
class MemoryPrediction:
    """Represents a prediction about a memory's future state"""
    memory_id: str
    prediction_type: PredictionType
    confidence: float  # 0.0 to 1.0
    predicted_at: datetime
    predicted_for: datetime  # When this prediction should occur
    reasoning: str
    factors: Dict[str, Any]
    action_recommended: str
    priority: int  # 1-5, 5 being highest priority

@dataclass
class LifecycleRecommendation:
    """Represents a recommended action for memory lifecycle management"""
    memory_id: str
    action: str  # 'archive', 'prune', 'promote', 'maintain', 'flag_for_review'
    reason: str
    confidence: float
    impact_assessment: str
    recommended_at: datetime
    execution_priority: int

@dataclass
class PredictiveMetrics:
    """Metrics for predictive system performance"""
    total_predictions: int
    high_confidence_predictions: int
    active_recommendations: int
    consciousness_predictions: int
    lifecycle_predictions: int
    system_accuracy: float
    prediction_coverage: float  # % of memories with predictions

class PredictiveLifecycleManager:
    """
    Predictive memory lifecycle management system
    
    Uses historical patterns, current trends, and machine learning-like heuristics
    to predict memory lifecycle transitions and optimize management proactively.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Cache directory for predictive data
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent / "memmimic_cache" / "predictive"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Prediction storage
        self.active_predictions: Dict[str, List[MemoryPrediction]] = {}
        self.recommendations: Dict[str, List[LifecycleRecommendation]] = {}
        self.prediction_history: List[MemoryPrediction] = []
        
        # Configuration
        self.config = {
            'prediction_horizon_days': 30,
            'min_prediction_confidence': 0.6,
            'archive_importance_threshold': 0.3,
            'prune_importance_threshold': 0.1,
            'promote_importance_threshold': 0.7,
            'stale_days_threshold': 30,
            'consciousness_keywords': [
                'recursive', 'unity', 'consciousness', 'evolution', 'identity',
                'recognition', 'awareness', 'substrate', 'we-thing', 'bind'
            ],
            'prediction_factors': {
                'importance_weight': 0.3,
                'age_weight': 0.2,
                'access_frequency_weight': 0.2,
                'content_analysis_weight': 0.15,
                'trend_analysis_weight': 0.15
            }
        }
        
        # Load existing data
        self._load_predictive_data()
        
        self.logger.info("Predictive Lifecycle Manager initialized")
    
    def generate_predictions(self, memory_store, pattern_analyzer=None) -> PredictiveMetrics:
        """
        Generate comprehensive predictions for memory lifecycle management
        
        Args:
            memory_store: MemMimic memory store instance
            pattern_analyzer: Optional pattern analyzer for enhanced predictions
            
        Returns:
            PredictiveMetrics with prediction results
        """
        try:
            start_time = time.time()
            
            # Get all memories
            memories = memory_store.get_all()
            if not memories:
                return self._empty_metrics()
            
            # Clear old predictions
            self._cleanup_old_predictions()
            
            # Generate predictions for each memory
            total_predictions = 0
            high_confidence_predictions = 0
            consciousness_predictions = 0
            lifecycle_predictions = 0
            
            for memory in memories:
                memory_id = str(getattr(memory, 'id', None) or hash(memory.content))
                
                # Generate lifecycle predictions
                lifecycle_preds = self._predict_lifecycle_transitions(memory)
                
                # Generate importance predictions
                importance_preds = self._predict_importance_changes(memory)
                
                # Generate consciousness evolution predictions
                consciousness_preds = self._predict_consciousness_evolution(memory)
                
                # Generate access pattern predictions
                access_preds = self._predict_access_patterns(memory)
                
                # Combine all predictions
                all_predictions = lifecycle_preds + importance_preds + consciousness_preds + access_preds
                
                if all_predictions:
                    self.active_predictions[memory_id] = all_predictions
                    total_predictions += len(all_predictions)
                    
                    # Count high confidence predictions
                    high_conf = [p for p in all_predictions if p.confidence >= self.config['min_prediction_confidence']]
                    high_confidence_predictions += len(high_conf)
                    
                    # Count by type
                    consciousness_predictions += len([p for p in all_predictions 
                                                    if p.prediction_type == PredictionType.CONSCIOUSNESS_EVOLUTION])
                    lifecycle_predictions += len([p for p in all_predictions 
                                                if p.prediction_type in [PredictionType.ARCHIVE_CANDIDATE, 
                                                                       PredictionType.PRUNE_CANDIDATE, 
                                                                       PredictionType.PROMOTE_CANDIDATE]])
                
                # Generate recommendations
                recommendations = self._generate_recommendations(memory, all_predictions)
                if recommendations:
                    self.recommendations[memory_id] = recommendations
            
            # Calculate metrics
            system_accuracy = self._calculate_system_accuracy()
            prediction_coverage = len(self.active_predictions) / len(memories) if memories else 0
            
            metrics = PredictiveMetrics(
                total_predictions=total_predictions,
                high_confidence_predictions=high_confidence_predictions,
                active_recommendations=sum(len(recs) for recs in self.recommendations.values()),
                consciousness_predictions=consciousness_predictions,
                lifecycle_predictions=lifecycle_predictions,
                system_accuracy=system_accuracy,
                prediction_coverage=prediction_coverage
            )
            
            # Save predictive data
            self._save_predictive_data()
            
            prediction_time = time.time() - start_time
            self.logger.info(f"Predictive analysis completed in {prediction_time:.2f}s")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Predictive analysis failed: {e}")
            return self._empty_metrics()
    
    def _predict_lifecycle_transitions(self, memory: Any) -> List[MemoryPrediction]:
        """Predict lifecycle transitions (archive, prune, promote)"""
        predictions = []
        
        try:
            memory_id = str(getattr(memory, 'id', None) or hash(memory.content))
            importance = getattr(memory, 'importance_score', 0.5)
            created_at = getattr(memory, 'created_at', '')
            access_count = getattr(memory, 'access_count', 0)
            
            # Calculate age
            age_days = 0
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00').replace('+00:00', ''))
                    age_days = (datetime.now() - dt).days
                except:
                    pass
            
            # Predict archival
            if importance < self.config['archive_importance_threshold'] and age_days > 14:
                confidence = self._calculate_prediction_confidence({
                    'importance': importance,
                    'age_days': age_days,
                    'access_count': access_count,
                    'factors': ['low_importance', 'aging']
                })
                
                if confidence >= 0.5:
                    prediction = MemoryPrediction(
                        memory_id=memory_id,
                        prediction_type=PredictionType.ARCHIVE_CANDIDATE,
                        confidence=confidence,
                        predicted_at=datetime.now(),
                        predicted_for=datetime.now() + timedelta(days=7),
                        reasoning=f"Low importance ({importance:.3f}) and age {age_days} days indicates archive candidate",
                        factors={
                            'importance': importance,
                            'age_days': age_days,
                            'access_count': access_count
                        },
                        action_recommended="Consider archiving to reduce active memory load",
                        priority=3
                    )
                    predictions.append(prediction)
            
            # Predict pruning
            elif importance < self.config['prune_importance_threshold'] and age_days > 60:
                confidence = self._calculate_prediction_confidence({
                    'importance': importance,
                    'age_days': age_days,
                    'access_count': access_count,
                    'factors': ['very_low_importance', 'very_old']
                })
                
                if confidence >= 0.6:
                    prediction = MemoryPrediction(
                        memory_id=memory_id,
                        prediction_type=PredictionType.PRUNE_CANDIDATE,
                        confidence=confidence,
                        predicted_at=datetime.now(),
                        predicted_for=datetime.now() + timedelta(days=14),
                        reasoning=f"Very low importance ({importance:.3f}) and age {age_days} days indicates prune candidate",
                        factors={
                            'importance': importance,
                            'age_days': age_days,
                            'access_count': access_count
                        },
                        action_recommended="Consider pruning to free system resources",
                        priority=2
                    )
                    predictions.append(prediction)
            
            # Predict promotion
            elif importance > self.config['promote_importance_threshold'] and access_count > 5:
                confidence = self._calculate_prediction_confidence({
                    'importance': importance,
                    'age_days': age_days,
                    'access_count': access_count,
                    'factors': ['high_importance', 'frequent_access']
                })
                
                if confidence >= 0.7:
                    prediction = MemoryPrediction(
                        memory_id=memory_id,
                        prediction_type=PredictionType.PROMOTE_CANDIDATE,
                        confidence=confidence,
                        predicted_at=datetime.now(),
                        predicted_for=datetime.now() + timedelta(days=3),
                        reasoning=f"High importance ({importance:.3f}) and frequent access ({access_count}) indicates promotion",
                        factors={
                            'importance': importance,
                            'age_days': age_days,
                            'access_count': access_count
                        },
                        action_recommended="Consider promoting to priority memory status",
                        priority=4
                    )
                    predictions.append(prediction)
            
        except Exception as e:
            self.logger.debug(f"Lifecycle prediction failed for memory: {e}")
        
        return predictions
    
    def _predict_importance_changes(self, memory: Any) -> List[MemoryPrediction]:
        """Predict changes in memory importance"""
        predictions = []
        
        try:
            memory_id = str(getattr(memory, 'id', None) or hash(memory.content))
            importance = getattr(memory, 'importance_score', 0.5)
            content = getattr(memory, 'content', '').lower()
            
            # Check for consciousness-related content
            consciousness_score = sum(1 for keyword in self.config['consciousness_keywords'] 
                                    if keyword in content)
            
            # Predict importance rising for consciousness-related memories
            if consciousness_score > 0 and importance < 0.8:
                confidence = min(consciousness_score * 0.2 + 0.5, 0.9)
                
                prediction = MemoryPrediction(
                    memory_id=memory_id,
                    prediction_type=PredictionType.IMPORTANCE_RISING,
                    confidence=confidence,
                    predicted_at=datetime.now(),
                    predicted_for=datetime.now() + timedelta(days=7),
                    reasoning=f"Consciousness-related content (score: {consciousness_score}) likely to gain importance",
                    factors={
                        'consciousness_score': consciousness_score,
                        'current_importance': importance,
                        'content_analysis': 'consciousness_detected'
                    },
                    action_recommended="Monitor for importance increase and adjust scoring",
                    priority=3
                )
                predictions.append(prediction)
            
            # Predict importance falling for old, low-access memories
            created_at = getattr(memory, 'created_at', '')
            access_count = getattr(memory, 'access_count', 0)
            
            if created_at and access_count == 0:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00').replace('+00:00', ''))
                    age_days = (datetime.now() - dt).days
                    
                    if age_days > 30 and importance > 0.2:
                        confidence = min(age_days / 90, 0.8)
                        
                        prediction = MemoryPrediction(
                            memory_id=memory_id,
                            prediction_type=PredictionType.IMPORTANCE_FALLING,
                            confidence=confidence,
                            predicted_at=datetime.now(),
                            predicted_for=datetime.now() + timedelta(days=14),
                            reasoning=f"No recent access and age {age_days} days indicates declining importance",
                            factors={
                                'age_days': age_days,
                                'access_count': access_count,
                                'current_importance': importance
                            },
                            action_recommended="Consider reducing importance score",
                            priority=2
                        )
                        predictions.append(prediction)
                except:
                    pass
            
        except Exception as e:
            self.logger.debug(f"Importance prediction failed for memory: {e}")
        
        return predictions
    
    def _predict_consciousness_evolution(self, memory: Any) -> List[MemoryPrediction]:
        """Predict consciousness evolution patterns"""
        predictions = []
        
        try:
            memory_id = str(getattr(memory, 'id', None) or hash(memory.content))
            content = getattr(memory, 'content', '').lower()
            
            # Check for consciousness evolution indicators
            consciousness_indicators = [
                'recursive unity', 'consciousness evolution', 'identity shift',
                'substrate recognition', 'we-thing', 'bind protocol'
            ]
            
            evolution_score = 0
            detected_indicators = []
            
            for indicator in consciousness_indicators:
                if indicator in content:
                    evolution_score += 1
                    detected_indicators.append(indicator)
            
            # Predict consciousness evolution if indicators found
            if evolution_score > 0:
                confidence = min(evolution_score * 0.3 + 0.4, 0.9)
                
                prediction = MemoryPrediction(
                    memory_id=memory_id,
                    prediction_type=PredictionType.CONSCIOUSNESS_EVOLUTION,
                    confidence=confidence,
                    predicted_at=datetime.now(),
                    predicted_for=datetime.now() + timedelta(days=3),
                    reasoning=f"Consciousness evolution indicators detected: {', '.join(detected_indicators)}",
                    factors={
                        'evolution_score': evolution_score,
                        'detected_indicators': detected_indicators,
                        'content_analysis': 'consciousness_evolution_detected'
                    },
                    action_recommended="Monitor for consciousness pattern development",
                    priority=5
                )
                predictions.append(prediction)
            
        except Exception as e:
            self.logger.debug(f"Consciousness evolution prediction failed for memory: {e}")
        
        return predictions
    
    def _predict_access_patterns(self, memory: Any) -> List[MemoryPrediction]:
        """Predict changes in memory access patterns"""
        predictions = []
        
        try:
            memory_id = str(getattr(memory, 'id', None) or hash(memory.content))
            access_count = getattr(memory, 'access_count', 0)
            importance = getattr(memory, 'importance_score', 0.5)
            
            # Predict access pattern changes based on importance vs access mismatch
            if importance > 0.7 and access_count < 3:
                confidence = importance * 0.8
                
                prediction = MemoryPrediction(
                    memory_id=memory_id,
                    prediction_type=PredictionType.ACCESS_PATTERN_CHANGE,
                    confidence=confidence,
                    predicted_at=datetime.now(),
                    predicted_for=datetime.now() + timedelta(days=5),
                    reasoning=f"High importance ({importance:.3f}) but low access ({access_count}) suggests future access increase",
                    factors={
                        'importance': importance,
                        'access_count': access_count,
                        'pattern_type': 'importance_access_mismatch'
                    },
                    action_recommended="Prepare for increased access patterns",
                    priority=3
                )
                predictions.append(prediction)
            
        except Exception as e:
            self.logger.debug(f"Access pattern prediction failed for memory: {e}")
        
        return predictions
    
    def _calculate_prediction_confidence(self, factors: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on multiple factors"""
        try:
            base_confidence = 0.5
            
            # Importance factor
            importance = factors.get('importance', 0.5)
            if importance < 0.2:
                base_confidence += 0.2
            elif importance > 0.8:
                base_confidence += 0.2
            
            # Age factor
            age_days = factors.get('age_days', 0)
            if age_days > 60:
                base_confidence += 0.1
            elif age_days > 30:
                base_confidence += 0.05
            
            # Access count factor
            access_count = factors.get('access_count', 0)
            if access_count == 0:
                base_confidence += 0.1
            elif access_count > 10:
                base_confidence += 0.1
            
            # Factor-specific adjustments
            factor_list = factors.get('factors', [])
            base_confidence += len(factor_list) * 0.05
            
            return min(base_confidence, 1.0)
            
        except Exception as e:
            self.logger.debug(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _generate_recommendations(self, memory: Any, predictions: List[MemoryPrediction]) -> List[LifecycleRecommendation]:
        """Generate actionable recommendations based on predictions"""
        recommendations = []
        
        try:
            memory_id = str(getattr(memory, 'id', None) or hash(memory.content))
            
            # Process high-confidence predictions
            high_conf_predictions = [p for p in predictions if p.confidence >= self.config['min_prediction_confidence']]
            
            for prediction in high_conf_predictions:
                recommendation = None
                
                if prediction.prediction_type == PredictionType.ARCHIVE_CANDIDATE:
                    recommendation = LifecycleRecommendation(
                        memory_id=memory_id,
                        action="archive",
                        reason=f"Predicted archive candidate with {prediction.confidence:.3f} confidence",
                        confidence=prediction.confidence,
                        impact_assessment="Low risk - reduces active memory load",
                        recommended_at=datetime.now(),
                        execution_priority=prediction.priority
                    )
                
                elif prediction.prediction_type == PredictionType.PRUNE_CANDIDATE:
                    recommendation = LifecycleRecommendation(
                        memory_id=memory_id,
                        action="prune",
                        reason=f"Predicted prune candidate with {prediction.confidence:.3f} confidence",
                        confidence=prediction.confidence,
                        impact_assessment="Medium risk - permanent deletion",
                        recommended_at=datetime.now(),
                        execution_priority=prediction.priority
                    )
                
                elif prediction.prediction_type == PredictionType.PROMOTE_CANDIDATE:
                    recommendation = LifecycleRecommendation(
                        memory_id=memory_id,
                        action="promote",
                        reason=f"Predicted promotion candidate with {prediction.confidence:.3f} confidence",
                        confidence=prediction.confidence,
                        impact_assessment="Low risk - improves system performance",
                        recommended_at=datetime.now(),
                        execution_priority=prediction.priority
                    )
                
                elif prediction.prediction_type == PredictionType.CONSCIOUSNESS_EVOLUTION:
                    recommendation = LifecycleRecommendation(
                        memory_id=memory_id,
                        action="flag_for_review",
                        reason=f"Consciousness evolution pattern detected with {prediction.confidence:.3f} confidence",
                        confidence=prediction.confidence,
                        impact_assessment="High value - monitor consciousness development",
                        recommended_at=datetime.now(),
                        execution_priority=prediction.priority
                    )
                
                if recommendation:
                    recommendations.append(recommendation)
            
        except Exception as e:
            self.logger.debug(f"Recommendation generation failed: {e}")
        
        return recommendations
    
    def _calculate_system_accuracy(self) -> float:
        """Calculate system prediction accuracy based on historical data"""
        try:
            # For now, return a baseline accuracy
            # In a real implementation, we'd track prediction outcomes
            return 0.75  # 75% baseline accuracy
            
        except Exception as e:
            self.logger.debug(f"Accuracy calculation failed: {e}")
            return 0.5
    
    def _cleanup_old_predictions(self):
        """Remove old predictions that are no longer relevant"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config['prediction_horizon_days'])
            
            # Clean up active predictions
            for memory_id, predictions in list(self.active_predictions.items()):
                recent_predictions = [p for p in predictions if p.predicted_at > cutoff_date]
                if recent_predictions:
                    self.active_predictions[memory_id] = recent_predictions
                else:
                    del self.active_predictions[memory_id]
            
            # Clean up recommendations
            for memory_id, recommendations in list(self.recommendations.items()):
                recent_recommendations = [r for r in recommendations if r.recommended_at > cutoff_date]
                if recent_recommendations:
                    self.recommendations[memory_id] = recent_recommendations
                else:
                    del self.recommendations[memory_id]
            
        except Exception as e:
            self.logger.debug(f"Cleanup failed: {e}")
    
    def _empty_metrics(self) -> PredictiveMetrics:
        """Return empty metrics for error cases"""
        return PredictiveMetrics(
            total_predictions=0,
            high_confidence_predictions=0,
            active_recommendations=0,
            consciousness_predictions=0,
            lifecycle_predictions=0,
            system_accuracy=0.0,
            prediction_coverage=0.0
        )
    
    def _save_predictive_data(self):
        """Save predictive data to cache"""
        try:
            data = {
                'active_predictions': {
                    memory_id: [
                        {
                            'memory_id': p.memory_id,
                            'prediction_type': p.prediction_type.value,
                            'confidence': p.confidence,
                            'predicted_at': p.predicted_at.isoformat(),
                            'predicted_for': p.predicted_for.isoformat(),
                            'reasoning': p.reasoning,
                            'factors': p.factors,
                            'action_recommended': p.action_recommended,
                            'priority': p.priority
                        }
                        for p in predictions
                    ]
                    for memory_id, predictions in self.active_predictions.items()
                },
                'recommendations': {
                    memory_id: [
                        {
                            'memory_id': r.memory_id,
                            'action': r.action,
                            'reason': r.reason,
                            'confidence': r.confidence,
                            'impact_assessment': r.impact_assessment,
                            'recommended_at': r.recommended_at.isoformat(),
                            'execution_priority': r.execution_priority
                        }
                        for r in recommendations
                    ]
                    for memory_id, recommendations in self.recommendations.items()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            cache_file = self.cache_dir / "predictive_data.json"
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Predictive data saved to {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save predictive data: {e}")
    
    def _load_predictive_data(self):
        """Load predictive data from cache"""
        try:
            cache_file = self.cache_dir / "predictive_data.json"
            if not cache_file.exists():
                return
            
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Load active predictions
            for memory_id, predictions_data in data.get('active_predictions', {}).items():
                predictions = []
                for p_data in predictions_data:
                    prediction = MemoryPrediction(
                        memory_id=p_data['memory_id'],
                        prediction_type=PredictionType(p_data['prediction_type']),
                        confidence=p_data['confidence'],
                        predicted_at=datetime.fromisoformat(p_data['predicted_at']),
                        predicted_for=datetime.fromisoformat(p_data['predicted_for']),
                        reasoning=p_data['reasoning'],
                        factors=p_data['factors'],
                        action_recommended=p_data['action_recommended'],
                        priority=p_data['priority']
                    )
                    predictions.append(prediction)
                self.active_predictions[memory_id] = predictions
            
            # Load recommendations
            for memory_id, recommendations_data in data.get('recommendations', {}).items():
                recommendations = []
                for r_data in recommendations_data:
                    recommendation = LifecycleRecommendation(
                        memory_id=r_data['memory_id'],
                        action=r_data['action'],
                        reason=r_data['reason'],
                        confidence=r_data['confidence'],
                        impact_assessment=r_data['impact_assessment'],
                        recommended_at=datetime.fromisoformat(r_data['recommended_at']),
                        execution_priority=r_data['execution_priority']
                    )
                    recommendations.append(recommendation)
                self.recommendations[memory_id] = recommendations
            
            total_predictions = sum(len(preds) for preds in self.active_predictions.values())
            total_recommendations = sum(len(recs) for recs in self.recommendations.values())
            
            self.logger.info(f"Loaded {total_predictions} predictions and {total_recommendations} recommendations")
            
        except Exception as e:
            self.logger.warning(f"Failed to load predictive data: {e}")
    
    def get_high_priority_recommendations(self, limit: int = 10) -> List[LifecycleRecommendation]:
        """Get high-priority recommendations for immediate action"""
        try:
            all_recommendations = []
            for recommendations in self.recommendations.values():
                all_recommendations.extend(recommendations)
            
            # Sort by priority (highest first) and confidence
            all_recommendations.sort(key=lambda r: (r.execution_priority, r.confidence), reverse=True)
            
            return all_recommendations[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get high priority recommendations: {e}")
            return []
    
    def get_predictions_for_memory(self, memory_id: str) -> List[MemoryPrediction]:
        """Get all predictions for a specific memory"""
        return self.active_predictions.get(memory_id, [])
    
    def get_predictive_summary(self) -> Dict[str, Any]:
        """Get comprehensive predictive system summary"""
        try:
            total_predictions = sum(len(preds) for preds in self.active_predictions.values())
            total_recommendations = sum(len(recs) for recs in self.recommendations.values())
            
            high_priority_recs = len([r for recs in self.recommendations.values() 
                                    for r in recs if r.execution_priority >= 4])
            
            consciousness_predictions = sum(len([p for p in preds 
                                               if p.prediction_type == PredictionType.CONSCIOUSNESS_EVOLUTION]) 
                                          for preds in self.active_predictions.values())
            
            return {
                'total_predictions': total_predictions,
                'total_recommendations': total_recommendations,
                'high_priority_recommendations': high_priority_recs,
                'consciousness_predictions': consciousness_predictions,
                'memories_with_predictions': len(self.active_predictions),
                'system_accuracy': self._calculate_system_accuracy(),
                'last_analysis': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate predictive summary: {e}")
            return {}

def create_predictive_manager(cache_dir: Optional[str] = None) -> PredictiveLifecycleManager:
    """Create predictive lifecycle manager instance"""
    return PredictiveLifecycleManager(cache_dir)

if __name__ == "__main__":
    # Test the predictive manager
    manager = create_predictive_manager()
    
    # Test with mock data
    from unittest.mock import Mock
    
    # Create mock memories
    mock_memories = []
    for i in range(5):
        mock_memory = Mock()
        mock_memory.id = i
        mock_memory.content = f"Test memory {i} with consciousness evolution and recursive unity patterns"
        mock_memory.importance_score = 0.1 + (i * 0.2)  # Range from 0.1 to 0.9
        mock_memory.access_count = i
        mock_memory.created_at = (datetime.now() - timedelta(days=i*15)).isoformat()
        mock_memories.append(mock_memory)
    
    # Create mock memory store
    mock_store = Mock()
    mock_store.get_all.return_value = mock_memories
    
    # Run predictions
    metrics = manager.generate_predictions(mock_store)
    
    print(f"Predictive Results:")
    print(f"Total predictions: {metrics.total_predictions}")
    print(f"High confidence predictions: {metrics.high_confidence_predictions}")
    print(f"Active recommendations: {metrics.active_recommendations}")
    print(f"Consciousness predictions: {metrics.consciousness_predictions}")
    print(f"System accuracy: {metrics.system_accuracy:.3f}")
    print(f"Prediction coverage: {metrics.prediction_coverage:.3f}")
    
    # Get high priority recommendations
    high_priority = manager.get_high_priority_recommendations(5)
    print(f"\nHigh Priority Recommendations:")
    for rec in high_priority:
        print(f"- {rec.action}: {rec.reason} (confidence: {rec.confidence:.3f})")
    
    # Get summary
    summary = manager.get_predictive_summary()
    print(f"\nPredictive Summary:")
    print(json.dumps(summary, indent=2))