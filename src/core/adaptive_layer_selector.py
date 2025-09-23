"""
Adaptive Layer Selector for Layerwise Adapter

This module implements intelligent layer selection algorithms that automatically
determine the most important layers for different tasks and data characteristics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for different task types"""
    task_type: str  # 'recommendation', 'classification', 'regression'
    domain: str  # 'e-commerce', 'social', 'entertainment'
    data_size: str  # 'small', 'medium', 'large'
    feature_types: List[str]  # ['categorical', 'numerical', 'text', 'image']
    sparsity_level: float  # 0.0 to 1.0
    interaction_density: float  # 0.0 to 1.0


@dataclass
class LayerImportanceProfile:
    """Profile containing layer importance scores and metadata"""
    layer_scores: Dict[str, float]
    importance_distribution: str  # 'bottom-heavy', 'top-heavy', 'uniform', 'multi-modal'
    critical_layers: List[str]
    redundant_layers: List[str]
    task_config: TaskConfig
    confidence_score: float


class ImportancePredictor(nn.Module):
    """Neural network to predict layer importance based on task characteristics"""
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DynamicThreshold:
    """Dynamic threshold computation based on data characteristics"""
    
    def __init__(self):
        self.threshold_history = []
        self.performance_history = []
    
    def compute_threshold(self, importance_scores: np.ndarray, 
                         task_config: TaskConfig) -> float:
        """Compute dynamic threshold for layer selection"""
        
        # Base threshold computation using statistical methods
        mean_importance = np.mean(importance_scores)
        std_importance = np.std(importance_scores)
        
        # Adjust based on task characteristics
        if task_config.data_size == 'large':
            # More aggressive selection for large datasets
            threshold = mean_importance + 0.5 * std_importance
        elif task_config.sparsity_level > 0.8:
            # More conservative for sparse data
            threshold = mean_importance + 0.2 * std_importance
        else:
            # Standard threshold
            threshold = mean_importance + 0.3 * std_importance
        
        return max(threshold, 0.1)  # Minimum threshold
    
    def update_threshold(self, threshold: float, performance: float):
        """Update threshold based on performance feedback"""
        self.threshold_history.append(threshold)
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.threshold_history) > 100:
            self.threshold_history = self.threshold_history[-100:]
            self.performance_history = self.performance_history[-100:]


class AdaptiveLayerSelector:
    """Main class for intelligent layer selection"""
    
    def __init__(self, model_name: str = "adaptive_selector"):
        self.model_name = model_name
        self.importance_predictor = ImportancePredictor()
        self.dynamic_threshold = DynamicThreshold()
        self.task_profiles = {}
        self.layer_patterns = {}
        
        # Task type encodings
        self.task_encodings = {
            'recommendation': [1, 0, 0],
            'classification': [0, 1, 0],
            'regression': [0, 0, 1]
        }
        
        self.domain_encodings = {
            'e-commerce': [1, 0, 0],
            'social': [0, 1, 0],
            'entertainment': [0, 0, 1]
        }
        
        logger.info(f"Initialized AdaptiveLayerSelector: {model_name}")
    
    def encode_task_config(self, task_config: TaskConfig) -> np.ndarray:
        """Encode task configuration into feature vector"""
        features = []
        
        # Task type encoding
        features.extend(self.task_encodings.get(task_config.task_type, [0, 0, 0]))
        
        # Domain encoding
        features.extend(self.domain_encodings.get(task_config.domain, [0, 0, 0]))
        
        # Data size encoding
        size_encoding = {'small': 0.25, 'medium': 0.5, 'large': 1.0}
        features.append(size_encoding.get(task_config.data_size, 0.5))
        
        # Feature types encoding (binary)
        feature_types = ['categorical', 'numerical', 'text', 'image']
        for ftype in feature_types:
            features.append(1.0 if ftype in task_config.feature_types else 0.0)
        
        # Sparsity and interaction density
        features.extend([task_config.sparsity_level, task_config.interaction_density])
        
        return np.array(features, dtype=np.float32)
    
    def analyze_layer_patterns(self, fisher_info: Dict[str, torch.Tensor], 
                              task_config: TaskConfig) -> LayerImportanceProfile:
        """Analyze layer importance patterns for a given task"""
        
        # Compute layer-wise importance scores
        layer_scores = {}
        for layer_name, fisher_tensor in fisher_info.items():
            if fisher_tensor.numel() > 0:
                importance = torch.sum(fisher_tensor).item()
                layer_scores[layer_name] = importance
        
        # Normalize scores
        if layer_scores:
            max_score = max(layer_scores.values())
            if max_score > 0:
                layer_scores = {k: v/max_score for k, v in layer_scores.items()}
        
        # Determine importance distribution
        scores_array = np.array(list(layer_scores.values()))
        distribution_type = self._classify_distribution(scores_array)
        
        # Identify critical and redundant layers
        threshold = self.dynamic_threshold.compute_threshold(scores_array, task_config)
        critical_layers = [name for name, score in layer_scores.items() 
                          if score >= threshold]
        redundant_layers = [name for name, score in layer_scores.items() 
                           if score < threshold * 0.3]
        
        # Compute confidence score
        confidence = self._compute_confidence(scores_array, task_config)
        
        profile = LayerImportanceProfile(
            layer_scores=layer_scores,
            importance_distribution=distribution_type,
            critical_layers=critical_layers,
            redundant_layers=redundant_layers,
            task_config=task_config,
            confidence_score=confidence
        )
        
        return profile
    
    def _classify_distribution(self, scores: np.ndarray) -> str:
        """Classify the type of importance distribution"""
        if len(scores) < 3:
            return 'uniform'
        
        # Sort scores to analyze distribution
        sorted_scores = np.sort(scores)[::-1]  # Descending order
        
        # Calculate distribution metrics
        top_10_percent = int(max(1, len(scores) * 0.1))
        bottom_10_percent = int(max(1, len(scores) * 0.1))
        
        top_sum = np.sum(sorted_scores[:top_10_percent])
        bottom_sum = np.sum(sorted_scores[-bottom_10_percent:])
        total_sum = np.sum(scores)
        
        top_ratio = top_sum / total_sum if total_sum > 0 else 0
        bottom_ratio = bottom_sum / total_sum if total_sum > 0 else 0
        
        # Classify distribution
        if top_ratio > 0.7:
            return 'top-heavy'
        elif bottom_ratio > 0.3:
            return 'bottom-heavy'
        elif self._is_multimodal(scores):
            return 'multi-modal'
        else:
            return 'uniform'
    
    def _is_multimodal(self, scores: np.ndarray) -> bool:
        """Check if the distribution is multi-modal using KMeans clustering"""
        if len(scores) < 6:
            return False
        
        try:
            # Try clustering into 3 groups
            scores_reshaped = scores.reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scores_reshaped)
            
            # Check if clusters are well-separated
            cluster_centers = kmeans.cluster_centers_.flatten()
            min_separation = np.min(np.diff(np.sort(cluster_centers)))
            avg_intra_distance = np.mean([
                np.std(scores[clusters == i]) for i in range(3)
                if np.sum(clusters == i) > 0
            ])
            
            return min_separation > 2 * avg_intra_distance
        except:
            return False
    
    def _compute_confidence(self, scores: np.ndarray, task_config: TaskConfig) -> float:
        """Compute confidence score for the layer selection"""
        if len(scores) == 0:
            return 0.0
        
        # Base confidence from score variance
        score_variance = np.var(scores)
        base_confidence = min(1.0, score_variance * 2)
        
        # Adjust based on data characteristics
        data_factor = 1.0
        if task_config.data_size == 'large':
            data_factor *= 1.2
        if task_config.sparsity_level < 0.5:
            data_factor *= 1.1
        
        return min(1.0, base_confidence * data_factor)
    
    def auto_select_layers(self, model: nn.Module, task_config: TaskConfig,
                          dataloader, device: torch.device = None) -> LayerImportanceProfile:
        """Automatically select important layers for a given task"""
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Auto-selecting layers for task: {task_config.task_type}")
        
        # Compute Fisher information
        fisher_info = self._compute_fisher_information(model, dataloader, device)
        
        # Analyze patterns
        profile = self.analyze_layer_patterns(fisher_info, task_config)
        
        # Store profile for future reference
        task_key = f"{task_config.task_type}_{task_config.domain}_{task_config.data_size}"
        self.task_profiles[task_key] = profile
        
        logger.info(f"Selected {len(profile.critical_layers)} critical layers")
        logger.info(f"Distribution type: {profile.importance_distribution}")
        logger.info(f"Confidence score: {profile.confidence_score:.3f}")
        
        return profile
    
    def _compute_fisher_information(self, model: nn.Module, dataloader, 
                                  device: torch.device) -> Dict[str, torch.Tensor]:
        """Compute Fisher information matrix for the model"""
        
        model.eval()
        fisher_info = {}
        
        # Initialize Fisher information storage
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
        
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 100:  # Limit samples for efficiency
                break
                
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [item.to(device) if torch.is_tensor(item) else item 
                        for item in batch]
            elif torch.is_tensor(batch):
                batch = batch.to(device)
            
            model.zero_grad()
            
            try:
                # Forward pass
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    output = model(batch[0])
                    if len(batch) > 1:
                        target = batch[1]
                    else:
                        target = None
                else:
                    output = model(batch)
                    target = None
                
                # Compute loss (use simple MSE or CrossEntropy based on output)
                if target is not None:
                    if output.shape[-1] == 1:
                        loss = nn.MSELoss()(output.squeeze(), target.float().to(device))
                    else:
                        loss = nn.CrossEntropyLoss()(output, target.long().to(device))
                else:
                    # Use output variance as loss for unsupervised case
                    loss = torch.var(output)
                
                # Backward pass
                loss.backward()
                
                # Accumulate Fisher information
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher_info[name] += param.grad.data ** 2
                
                total_samples += batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
                
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Normalize by number of samples
        if total_samples > 0:
            for name in fisher_info:
                fisher_info[name] /= total_samples
        
        return fisher_info
    
    def save_profile(self, profile: LayerImportanceProfile, filepath: Path):
        """Save layer importance profile to file"""
        profile_data = {
            'layer_scores': profile.layer_scores,
            'importance_distribution': profile.importance_distribution,
            'critical_layers': profile.critical_layers,
            'redundant_layers': profile.redundant_layers,
            'task_config': {
                'task_type': profile.task_config.task_type,
                'domain': profile.task_config.domain,
                'data_size': profile.task_config.data_size,
                'feature_types': profile.task_config.feature_types,
                'sparsity_level': profile.task_config.sparsity_level,
                'interaction_density': profile.task_config.interaction_density
            },
            'confidence_score': profile.confidence_score
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"Saved layer importance profile to {filepath}")
    
    def load_profile(self, filepath: Path) -> LayerImportanceProfile:
        """Load layer importance profile from file"""
        with open(filepath, 'r') as f:
            profile_data = json.load(f)
        
        task_config = TaskConfig(**profile_data['task_config'])
        
        profile = LayerImportanceProfile(
            layer_scores=profile_data['layer_scores'],
            importance_distribution=profile_data['importance_distribution'],
            critical_layers=profile_data['critical_layers'],
            redundant_layers=profile_data['redundant_layers'],
            task_config=task_config,
            confidence_score=profile_data['confidence_score']
        )
        
        logger.info(f"Loaded layer importance profile from {filepath}")
        return profile
    
    def get_layer_recommendations(self, task_config: TaskConfig) -> Dict[str, Any]:
        """Get layer selection recommendations based on task configuration"""
        
        # Find similar task profiles
        similar_profiles = self._find_similar_profiles(task_config)
        
        if not similar_profiles:
            return {
                'recommendation': 'No similar profiles found. Run auto_select_layers first.',
                'confidence': 0.0
            }
        
        # Aggregate recommendations from similar profiles
        aggregated_scores = {}
        total_confidence = 0
        
        for profile, similarity in similar_profiles:
            weight = similarity * profile.confidence_score
            total_confidence += weight
            
            for layer, score in profile.layer_scores.items():
                if layer not in aggregated_scores:
                    aggregated_scores[layer] = 0
                aggregated_scores[layer] += score * weight
        
        # Normalize aggregated scores
        if total_confidence > 0:
            aggregated_scores = {k: v/total_confidence 
                               for k, v in aggregated_scores.items()}
        
        # Generate recommendations
        sorted_layers = sorted(aggregated_scores.items(), 
                             key=lambda x: x[1], reverse=True)
        
        top_layers = [layer for layer, score in sorted_layers[:5]]
        
        return {
            'recommended_layers': top_layers,
            'layer_scores': aggregated_scores,
            'confidence': total_confidence / len(similar_profiles) if similar_profiles else 0.0,
            'num_similar_profiles': len(similar_profiles)
        }
    
    def _find_similar_profiles(self, task_config: TaskConfig, 
                              threshold: float = 0.7) -> List[Tuple[LayerImportanceProfile, float]]:
        """Find profiles similar to the given task configuration"""
        
        target_encoding = self.encode_task_config(task_config)
        similar_profiles = []
        
        for profile in self.task_profiles.values():
            profile_encoding = self.encode_task_config(profile.task_config)
            
            # Compute cosine similarity
            similarity = np.dot(target_encoding, profile_encoding) / (
                np.linalg.norm(target_encoding) * np.linalg.norm(profile_encoding) + 1e-8
            )
            
            if similarity >= threshold:
                similar_profiles.append((profile, similarity))
        
        # Sort by similarity
        similar_profiles.sort(key=lambda x: x[1], reverse=True)
        
        return similar_profiles
