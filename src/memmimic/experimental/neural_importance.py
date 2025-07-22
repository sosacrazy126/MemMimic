#!/usr/bin/env python3
"""
Neural Memory Importance Scoring

Advanced neural network-based memory importance scoring using deep learning
to understand context, relationships, and temporal relevance patterns.
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Extended memory context for neural analysis"""
    content: str
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    user_feedback: Optional[float] = None  # -1 to 1 relevance score
    semantic_embedding: Optional[np.ndarray] = None
    contextual_features: Dict[str, float] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)  # Related memory IDs


class MemoryImportanceNetwork(nn.Module):
    """Neural network for memory importance scoring"""
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Feature extraction layers
        layers = []
        prev_dim = embedding_dim + 10  # +10 for contextual features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final importance score layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Importance score 0-1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, embeddings: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute importance scores"""
        # Combine embeddings and contextual features
        combined = torch.cat([embeddings, features], dim=1)
        importance_scores = self.network(combined)
        return importance_scores.squeeze()


class NeuralMemoryScorer:
    """Neural network-based memory importance scoring system"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained transformer for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Initialize importance network
        self.importance_network = MemoryImportanceNetwork().to(self.device)
        self.optimizer = optim.Adam(self.importance_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Training data storage
        self.training_data: List[Tuple[MemoryContext, float]] = []
        self.feature_scaler = StandardScaler()
        
        # Memory analysis components
        self.memory_clusters = None
        self.cluster_importance = {}
    
    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract semantic embeddings for texts"""
        self.embedding_model.eval()
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and encode
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                # Use mean pooling
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def extract_contextual_features(self, memory: MemoryContext) -> np.ndarray:
        """Extract contextual features for neural network"""
        now = datetime.utcnow()
        
        # Temporal features
        days_old = (now - memory.timestamp).days
        recency_score = 1.0 / (1.0 + days_old * 0.1)  # Exponential decay
        
        # Access pattern features
        access_frequency = memory.access_count / max(1, days_old)
        days_since_last_access = 0
        if memory.last_accessed:
            days_since_last_access = (now - memory.last_accessed).days
        
        # Content features
        content_length = len(memory.content)
        word_count = len(memory.content.split())
        
        # Relationship features
        relationship_count = len(memory.relationships)
        
        # User feedback (if available)
        user_score = memory.user_feedback or 0.0
        
        features = np.array([
            recency_score,
            access_frequency,
            days_since_last_access,
            content_length / 1000.0,  # Normalize
            word_count / 100.0,  # Normalize
            relationship_count / 10.0,  # Normalize
            user_score,
            memory.access_count / 100.0,  # Normalize
            1.0 if memory.user_feedback else 0.0,  # Has user feedback
            len(memory.contextual_features) / 10.0  # Additional feature count
        ])
        
        return features
    
    def compute_importance_score(self, memory: MemoryContext) -> float:
        """Compute neural importance score for memory"""
        # Extract features
        if memory.semantic_embedding is None:
            memory.semantic_embedding = self.extract_embeddings([memory.content])[0]
        
        contextual_features = self.extract_contextual_features(memory)
        
        # Prepare tensors
        embedding_tensor = torch.FloatTensor(memory.semantic_embedding).unsqueeze(0).to(self.device)
        feature_tensor = torch.FloatTensor(contextual_features).unsqueeze(0).to(self.device)
        
        # Get neural network prediction
        self.importance_network.eval()
        with torch.no_grad():
            importance_score = self.importance_network(embedding_tensor, feature_tensor)
            return importance_score.item()
    
    def train_importance_model(self, 
                              training_memories: List[MemoryContext],
                              target_scores: List[float],
                              epochs: int = 100,
                              batch_size: int = 32):
        """Train the importance scoring model"""
        logger.info(f"Training neural importance model with {len(training_memories)} samples")
        
        # Extract embeddings for all training data
        texts = [memory.content for memory in training_memories]
        embeddings = self.extract_embeddings(texts)
        
        # Extract contextual features
        features = np.array([
            self.extract_contextual_features(memory) 
            for memory in training_memories
        ])
        
        # Scale features
        features = self.feature_scaler.fit_transform(features)
        
        # Convert to tensors
        embedding_tensors = torch.FloatTensor(embeddings).to(self.device)
        feature_tensors = torch.FloatTensor(features).to(self.device)
        target_tensors = torch.FloatTensor(target_scores).to(self.device)
        
        # Training loop
        self.importance_network.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = (len(training_memories) + batch_size - 1) // batch_size
            
            for i in range(0, len(training_memories), batch_size):
                end_idx = min(i + batch_size, len(training_memories))
                
                batch_embeddings = embedding_tensors[i:end_idx]
                batch_features = feature_tensors[i:end_idx]
                batch_targets = target_tensors[i:end_idx]
                
                # Forward pass
                predictions = self.importance_network(batch_embeddings, batch_features)
                loss = self.criterion(predictions, batch_targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    def analyze_memory_clusters(self, memories: List[MemoryContext], n_clusters: int = 10):
        """Analyze memory clusters for importance patterns"""
        if not memories:
            return
        
        # Extract embeddings
        texts = [memory.content for memory in memories]
        embeddings = self.extract_embeddings(texts)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        self.memory_clusters = {
            'model': kmeans,
            'labels': cluster_labels,
            'centers': kmeans.cluster_centers_
        }
        
        # Analyze cluster importance patterns
        for cluster_id in range(n_clusters):
            cluster_memories = [
                memories[i] for i in range(len(memories))
                if cluster_labels[i] == cluster_id
            ]
            
            if cluster_memories:
                # Compute average importance for cluster
                importance_scores = [
                    self.compute_importance_score(memory)
                    for memory in cluster_memories
                ]
                
                avg_importance = np.mean(importance_scores)
                self.cluster_importance[cluster_id] = {
                    'average_importance': avg_importance,
                    'memory_count': len(cluster_memories),
                    'sample_content': cluster_memories[0].content[:100],
                    'importance_std': np.std(importance_scores)
                }
        
        logger.info(f"Analyzed {n_clusters} memory clusters")
    
    def get_cluster_insights(self) -> Dict[str, Any]:
        """Get insights from memory clustering analysis"""
        if not self.cluster_importance:
            return {}
        
        insights = {
            'total_clusters': len(self.cluster_importance),
            'high_importance_clusters': [],
            'low_importance_clusters': [],
            'cluster_summary': {}
        }
        
        for cluster_id, data in self.cluster_importance.items():
            if data['average_importance'] > 0.7:
                insights['high_importance_clusters'].append({
                    'cluster_id': cluster_id,
                    'importance': data['average_importance'],
                    'sample': data['sample_content']
                })
            elif data['average_importance'] < 0.3:
                insights['low_importance_clusters'].append({
                    'cluster_id': cluster_id,
                    'importance': data['average_importance'],
                    'sample': data['sample_content']
                })
            
            insights['cluster_summary'][cluster_id] = data
        
        return insights
    
    def adaptive_learning(self, 
                         memory: MemoryContext,
                         actual_importance: float,
                         learning_rate: float = 0.01):
        """Adapt model based on user feedback"""
        # Extract features
        if memory.semantic_embedding is None:
            memory.semantic_embedding = self.extract_embeddings([memory.content])[0]
        
        contextual_features = self.extract_contextual_features(memory)
        
        # Prepare tensors
        embedding_tensor = torch.FloatTensor(memory.semantic_embedding).unsqueeze(0).to(self.device)
        feature_tensor = torch.FloatTensor(contextual_features).unsqueeze(0).to(self.device)
        target_tensor = torch.FloatTensor([actual_importance]).to(self.device)
        
        # Single gradient step
        self.importance_network.train()
        self.optimizer.zero_grad()
        
        prediction = self.importance_network(embedding_tensor, feature_tensor)
        loss = self.criterion(prediction, target_tensor)
        
        loss.backward()
        
        # Scale gradients by learning rate
        for param in self.importance_network.parameters():
            if param.grad is not None:
                param.grad *= learning_rate
        
        self.optimizer.step()
        
        logger.info(f"Adapted model based on feedback: predicted {prediction.item():.3f}, actual {actual_importance:.3f}")
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.importance_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_scaler': self.feature_scaler,
            'cluster_importance': self.cluster_importance
        }, path)
        logger.info(f"Saved neural importance model to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.importance_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.feature_scaler = checkpoint['feature_scaler']
        self.cluster_importance = checkpoint['cluster_importance']
        
        logger.info(f"Loaded neural importance model from {path}")


# Example usage and testing
async def main():
    """Example usage of neural memory importance scoring"""
    # Initialize neural scorer
    scorer = NeuralMemoryScorer()
    
    # Create sample memories
    sample_memories = [
        MemoryContext(
            content="Important meeting notes about project deadline",
            timestamp=datetime.utcnow() - timedelta(days=1),
            access_count=5,
            last_accessed=datetime.utcnow() - timedelta(hours=2),
            user_feedback=0.9
        ),
        MemoryContext(
            content="Random thought about lunch",
            timestamp=datetime.utcnow() - timedelta(days=7),
            access_count=1,
            user_feedback=0.2
        ),
        MemoryContext(
            content="Critical system architecture decisions for scalability",
            timestamp=datetime.utcnow() - timedelta(days=3),
            access_count=10,
            last_accessed=datetime.utcnow() - timedelta(minutes=30),
            user_feedback=0.95
        )
    ]
    
    # Compute importance scores
    print("Computing neural importance scores:")
    for i, memory in enumerate(sample_memories):
        score = scorer.compute_importance_score(memory)
        print(f"Memory {i + 1}: {score:.3f} - {memory.content[:50]}...")
    
    # Generate training data (in practice, this would come from user feedback)
    target_scores = [0.8, 0.2, 0.95]  # Based on user feedback
    
    # Train the model
    scorer.train_importance_model(sample_memories, target_scores, epochs=50)
    
    # Re-compute scores after training
    print("\nScores after training:")
    for i, memory in enumerate(sample_memories):
        score = scorer.compute_importance_score(memory)
        print(f"Memory {i + 1}: {score:.3f} (target: {target_scores[i]:.3f})")
    
    # Analyze clusters
    scorer.analyze_memory_clusters(sample_memories, n_clusters=2)
    insights = scorer.get_cluster_insights()
    
    print(f"\nCluster insights: {json.dumps(insights, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())