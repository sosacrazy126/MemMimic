"""
Neural Memory Importance Scoring System

Advanced neural network-based importance scoring for memories using multi-factor analysis.
Implements continuous learning and adaptation to user patterns and content semantics.
"""

import logging
import math
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics.pairwise import cosine_similarity

from ..memory.storage.amms_storage import Memory
from ..errors import MLModelError, handle_errors, with_error_context, get_error_logger

logger = get_error_logger("neural_importance_scorer")


@dataclass
class MemoryFeatures:
    """Feature vector for memory importance scoring"""
    
    # Content features
    content_length: float
    content_complexity: float  # Based on vocabulary richness
    semantic_density: float    # TF-IDF based semantic richness
    
    # Temporal features
    recency_score: float      # How recent the memory is
    age_factor: float         # Age-based decay
    temporal_cluster: int     # Time-based clustering
    
    # Usage features  
    access_frequency: float   # How often accessed
    access_recency: float     # When last accessed
    interaction_score: float  # User interaction patterns
    
    # Context features
    cxd_classification: str   # CXD pattern classification
    metadata_richness: float  # Richness of metadata
    cross_references: int     # References to other memories
    
    # Derived features
    importance_momentum: float = 0.0  # Trending importance
    cluster_centrality: float = 0.0   # Position within content clusters
    uniqueness_score: float = 0.0     # How unique the content is
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numerical vector for ML models"""
        # Encode categorical CXD classification as numerical
        cxd_encoding = hash(self.cxd_classification) % 100 / 100.0
        
        return np.array([
            self.content_length,
            self.content_complexity,
            self.semantic_density,
            self.recency_score,
            self.age_factor,
            self.temporal_cluster,
            self.access_frequency,
            self.access_recency,
            self.interaction_score,
            cxd_encoding,
            self.metadata_richness,
            self.cross_references,
            self.importance_momentum,
            self.cluster_centrality,
            self.uniqueness_score
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for interpretability"""
        return [
            'content_length', 'content_complexity', 'semantic_density',
            'recency_score', 'age_factor', 'temporal_cluster',
            'access_frequency', 'access_recency', 'interaction_score',
            'cxd_classification', 'metadata_richness', 'cross_references',
            'importance_momentum', 'cluster_centrality', 'uniqueness_score'
        ]


class FeatureExtractor:
    """Extracts ML features from memory objects"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.content_clusters = None
        self.is_fitted = False
        
    def fit(self, memories: List[Memory]) -> None:
        """Fit feature extractors on memory corpus"""
        logger.info(f"Fitting feature extractors on {len(memories)} memories")
        
        if not memories:
            logger.warning("No memories provided for fitting")
            return
            
        # Fit TF-IDF vectorizer
        contents = [mem.content for mem in memories]
        self.tfidf_vectorizer.fit(contents)
        
        # Create content clusters for centrality calculation
        tfidf_matrix = self.tfidf_vectorizer.transform(contents)
        n_clusters = min(10, len(memories) // 5 + 1)  # Adaptive cluster count
        self.content_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.content_clusters.fit(tfidf_matrix.toarray())
        
        # Fit scaler for numerical features
        feature_vectors = []
        for memory in memories:
            features = self._extract_base_features(memory)
            feature_vectors.append(features.to_vector())
        
        if feature_vectors:
            self.scaler.fit(feature_vectors)
            self.is_fitted = True
        
        logger.info("Feature extractors fitted successfully")
    
    def extract_features(self, memory: Memory, all_memories: Optional[List[Memory]] = None) -> MemoryFeatures:
        """Extract comprehensive features from a memory"""
        base_features = self._extract_base_features(memory)
        
        if self.is_fitted and all_memories:
            # Add advanced features requiring corpus context
            base_features = self._add_corpus_features(memory, base_features, all_memories)
        
        return base_features
    
    def _extract_base_features(self, memory: Memory) -> MemoryFeatures:
        """Extract basic features from memory"""
        content = memory.content
        metadata = memory.metadata or {}
        
        # Content features
        content_length = len(content) / 1000.0  # Normalize to ~[0,1]
        content_complexity = self._calculate_complexity(content)
        
        # Temporal features
        now = datetime.now()
        age = (now - memory.created_at).total_seconds() / (24 * 3600)  # Days
        recency_score = math.exp(-age / 30.0)  # Exponential decay over ~30 days
        age_factor = 1.0 / (1.0 + age / 365.0)  # Yearly normalization
        temporal_cluster = int(age / 7) % 10  # Weekly clusters, mod 10
        
        # Usage features (mock - would be real in production)
        access_frequency = metadata.get('access_count', 0) / 100.0
        last_access = metadata.get('last_accessed')
        if last_access:
            access_age = (now - datetime.fromisoformat(last_access)).total_seconds() / (24 * 3600)
            access_recency = math.exp(-access_age / 7.0)  # Week-based decay
        else:
            access_recency = 0.0
        
        interaction_score = metadata.get('interaction_score', 0.5)
        
        # Context features
        cxd_classification = metadata.get('cxd', 'unknown')
        metadata_richness = len(str(metadata)) / 1000.0  # Normalize
        cross_references = metadata.get('cross_references', 0)
        
        return MemoryFeatures(
            content_length=content_length,
            content_complexity=content_complexity,
            semantic_density=0.0,  # Will be calculated if corpus available
            recency_score=recency_score,
            age_factor=age_factor,
            temporal_cluster=temporal_cluster,
            access_frequency=access_frequency,
            access_recency=access_recency,
            interaction_score=interaction_score,
            cxd_classification=cxd_classification,
            metadata_richness=metadata_richness,
            cross_references=cross_references
        )
    
    def _add_corpus_features(self, memory: Memory, base_features: MemoryFeatures, 
                           all_memories: List[Memory]) -> MemoryFeatures:
        """Add features that require corpus context"""
        try:
            # Semantic density using TF-IDF
            tfidf_vector = self.tfidf_vectorizer.transform([memory.content])
            semantic_density = float(np.mean(tfidf_vector.toarray()))
            
            # Cluster centrality
            cluster_label = self.content_clusters.predict(tfidf_vector.toarray())[0]
            cluster_center = self.content_clusters.cluster_centers_[cluster_label]
            centrality = float(cosine_similarity(tfidf_vector.toarray(), [cluster_center])[0, 0])
            
            # Uniqueness score (distance from all other memories)
            all_contents = [mem.content for mem in all_memories if mem.id != memory.id]
            if all_contents:
                all_vectors = self.tfidf_vectorizer.transform(all_contents)
                similarities = cosine_similarity(tfidf_vector, all_vectors)[0]
                uniqueness = 1.0 - float(np.mean(similarities))
            else:
                uniqueness = 1.0
            
            # Update features
            base_features.semantic_density = semantic_density
            base_features.cluster_centrality = centrality
            base_features.uniqueness_score = uniqueness
            
        except Exception as e:
            logger.warning(f"Failed to calculate corpus features: {e}")
            
        return base_features
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate content complexity based on vocabulary richness"""
        if not text:
            return 0.0
            
        words = text.lower().split()
        if not words:
            return 0.0
            
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / len(words)
        
        # Average word length (complexity indicator)
        avg_word_length = sum(len(word) for word in unique_words) / len(unique_words)
        normalized_word_length = min(avg_word_length / 10.0, 1.0)  # Cap at 1.0
        
        return (vocabulary_richness + normalized_word_length) / 2.0


class NeuralImportanceScorer:
    """
    Neural network-based memory importance scoring system.
    
    Uses multi-layer perceptron to predict memory importance based on
    extracted features, with online learning and adaptation capabilities.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("memmimic_neural_scorer.pkl")
        self.feature_extractor = FeatureExtractor()
        
        # Neural network model
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),  # Three hidden layers
            activation='relu',
            solver='adam',
            alpha=0.001,           # L2 regularization
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        # Tracking metrics
        self.training_history = []
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Load existing model if available
        self.is_trained = False
        self._load_model()
        
        logger.info(f"NeuralImportanceScorer initialized (trained: {self.is_trained})")
    
    @handle_errors(catch=[Exception], reraise=True)
    def train(self, memories: List[Memory], target_scores: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Train the neural importance scorer.
        
        Args:
            memories: List of memories to train on
            target_scores: Optional target importance scores (if None, generates from heuristics)
            
        Returns:
            Training metrics and statistics
        """
        with with_error_context(
            operation="neural_scorer_training",
            component="ml.neural_importance_scorer",
            metadata={"memory_count": len(memories)}
        ):
            start_time = time.perf_counter()
            
            if len(memories) < 10:
                raise MLModelError("Insufficient training data", 
                                 error_code="INSUFFICIENT_DATA",
                                 context={"required": 10, "provided": len(memories)})
            
            logger.info(f"Training neural importance scorer on {len(memories)} memories")
            
            # Fit feature extractors first
            self.feature_extractor.fit(memories)
            
            # Extract features
            feature_vectors = []
            for memory in memories:
                features = self.feature_extractor.extract_features(memory, memories)
                feature_vectors.append(features.to_vector())
            
            X = np.array(feature_vectors)
            
            # Generate or use provided target scores
            if target_scores is None:
                y = self._generate_training_targets(memories, feature_vectors)
            else:
                y = np.array(target_scores)
                
            # Normalize features
            X_scaled = self.feature_extractor.scaler.transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Calculate training metrics
            train_score = self.model.score(X_scaled, y)
            training_time = time.perf_counter() - start_time
            
            # Store training history
            training_record = {
                'timestamp': datetime.now(),
                'memory_count': len(memories),
                'train_score': train_score,
                'training_time_seconds': training_time,
                'model_iterations': self.model.n_iter_
            }
            self.training_history.append(training_record)
            
            # Save model
            self._save_model()
            
            logger.info(f"Training completed - Score: {train_score:.4f}, Time: {training_time:.2f}s")
            
            return {
                'train_score': train_score,
                'training_time_seconds': training_time,
                'memory_count': len(memories),
                'feature_count': X.shape[1],
                'model_iterations': self.model.n_iter_,
                'feature_names': MemoryFeatures.get_feature_names()
            }
    
    def predict_importance(self, memory: Memory, all_memories: Optional[List[Memory]] = None) -> float:
        """
        Predict importance score for a memory.
        
        Args:
            memory: Memory to score
            all_memories: Optional full memory corpus for context features
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        if not self.is_trained:
            logger.warning("Model not trained, using fallback scoring")
            return self._fallback_importance(memory)
        
        # Check cache
        cache_key = f"{memory.id}_{hash(memory.content[:100])}"
        if cache_key in self.prediction_cache:
            self.cache_hits += 1
            return self.prediction_cache[cache_key]
        
        self.cache_misses += 1
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(memory, all_memories)
            feature_vector = features.to_vector().reshape(1, -1)
            
            # Normalize and predict
            feature_scaled = self.feature_extractor.scaler.transform(feature_vector)
            importance = float(self.model.predict(feature_scaled)[0])
            
            # Ensure valid range [0, 1]
            importance = max(0.0, min(1.0, importance))
            
            # Cache result
            self.prediction_cache[cache_key] = importance
            
            return importance
            
        except Exception as e:
            logger.error(f"Prediction failed for memory {memory.id}: {e}")
            return self._fallback_importance(memory)
    
    def batch_predict(self, memories: List[Memory]) -> List[float]:
        """Efficiently predict importance for multiple memories"""
        if not self.is_trained:
            return [self._fallback_importance(mem) for mem in memories]
        
        try:
            feature_vectors = []
            for memory in memories:
                features = self.feature_extractor.extract_features(memory, memories)
                feature_vectors.append(features.to_vector())
            
            X = np.array(feature_vectors)
            X_scaled = self.feature_extractor.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            # Ensure valid range
            return [max(0.0, min(1.0, pred)) for pred in predictions]
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return [self._fallback_importance(mem) for mem in memories]
    
    def update_model(self, memories: List[Memory], target_scores: List[float]) -> None:
        """Update model with new training data (online learning)"""
        if not self.is_trained:
            logger.warning("Cannot update untrained model")
            return
        
        try:
            # Extract features for new memories
            feature_vectors = []
            for memory in memories:
                features = self.feature_extractor.extract_features(memory, memories)
                feature_vectors.append(features.to_vector())
            
            X = np.array(feature_vectors)
            X_scaled = self.feature_extractor.scaler.transform(X)
            y = np.array(target_scores)
            
            # Partial fit (online learning)
            self.model.partial_fit(X_scaled, y)
            
            # Clear cache to ensure fresh predictions
            self.prediction_cache.clear()
            
            # Save updated model
            self._save_model()
            
            logger.info(f"Model updated with {len(memories)} new memories")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings (approximate via model weights)"""
        if not self.is_trained:
            return {}
        
        try:
            # Use first layer weights as proxy for feature importance
            weights = np.abs(self.model.coefs_[0]).mean(axis=1)
            feature_names = MemoryFeatures.get_feature_names()
            
            importance_dict = dict(zip(feature_names, weights))
            
            # Normalize to sum to 1
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scorer statistics and performance metrics"""
        total_predictions = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_predictions if total_predictions > 0 else 0
        
        return {
            'is_trained': self.is_trained,
            'training_history_count': len(self.training_history),
            'cache_hit_rate': cache_hit_rate,
            'total_predictions': total_predictions,
            'model_path': str(self.model_path),
            'feature_extractor_fitted': self.feature_extractor.is_fitted,
            'last_training': self.training_history[-1] if self.training_history else None
        }
    
    def _generate_training_targets(self, memories: List[Memory], feature_vectors: List[np.ndarray]) -> np.ndarray:
        """Generate training targets using heuristic importance scoring"""
        targets = []
        
        for i, (memory, features) in enumerate(zip(memories, feature_vectors)):
            # Heuristic importance based on multiple factors
            features_obj = self.feature_extractor.extract_features(memory, memories)
            
            # Base importance from content and recency
            base_score = (
                features_obj.content_complexity * 0.3 +
                features_obj.recency_score * 0.2 +
                features_obj.semantic_density * 0.2 +
                features_obj.access_frequency * 0.15 +
                features_obj.uniqueness_score * 0.15
            )
            
            # Apply modifiers
            if features_obj.cxd_classification in ['CONTROL', 'CONTEXT']:
                base_score *= 1.2  # Boost important classifications
            
            if features_obj.cross_references > 0:
                base_score *= (1 + 0.1 * features_obj.cross_references)
            
            # Normalize to [0, 1]
            importance = max(0.0, min(1.0, base_score))
            targets.append(importance)
        
        return np.array(targets)
    
    def _fallback_importance(self, memory: Memory) -> float:
        """Fallback importance scoring when model is unavailable"""
        content_score = min(len(memory.content) / 1000.0, 1.0)
        
        # Boost based on CXD classification
        cxd_pattern = memory.metadata.get('cxd', 'unknown')
        if cxd_pattern in ['CONTROL', 'CONTEXT']:
            content_score *= 1.3
        elif cxd_pattern == 'DATA':
            content_score *= 1.1
        
        # Age factor
        age_days = (datetime.now() - memory.created_at).total_seconds() / (24 * 3600)
        age_factor = math.exp(-age_days / 30.0)  # 30-day half-life
        
        return max(0.1, min(1.0, content_score * 0.7 + age_factor * 0.3))
    
    def _save_model(self) -> None:
        """Save trained model and feature extractor to disk"""
        try:
            model_data = {
                'model': self.model,
                'feature_extractor': self.feature_extractor,
                'training_history': self.training_history,
                'is_trained': self.is_trained
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.debug(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _load_model(self) -> None:
        """Load trained model and feature extractor from disk"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.feature_extractor = model_data['feature_extractor']
                self.training_history = model_data.get('training_history', [])
                self.is_trained = model_data.get('is_trained', False)
                
                logger.info(f"Model loaded from {self.model_path}")
                
        except Exception as e:
            logger.debug(f"Could not load existing model: {e}")