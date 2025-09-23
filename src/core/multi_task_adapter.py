"""
Multi-Task Layerwise Adapter Framework

This module implements a multi-task learning approach for the Layerwise Adapter,
enabling cross-task knowledge transfer and shared layer importance learning
across different recommendation scenarios and domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import copy

# Import our modules
from .adaptive_layer_selector import AdaptiveLayerSelector, TaskConfig, LayerImportanceProfile
from .fisher_information import LayerwiseFisherAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskMetadata:
    """Metadata for a specific task"""
    task_id: str
    task_name: str
    task_type: str  # 'recommendation', 'classification', 'regression'
    domain: str
    dataset_size: int
    num_users: int
    num_items: int
    sparsity: float
    data_characteristics: Dict[str, Any]


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task learning"""
    shared_embedding_dim: int = 64
    task_specific_dim: int = 32
    meta_learning_rate: float = 0.001
    task_learning_rate: float = 0.0001
    temperature: float = 3.0
    alpha_shared: float = 0.6  # Weight for shared layers
    alpha_specific: float = 0.4  # Weight for task-specific layers
    num_meta_epochs: int = 100
    adaptation_steps: int = 5
    importance_threshold: float = 0.3


class SharedImportanceMatrix:
    """Maintains shared layer importance across tasks"""
    
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.importance_matrix = torch.zeros((num_layers, num_layers))  # Layer x Task matrix
        self.task_count = 0
        self.layer_names = []
        self.confidence_scores = torch.zeros(num_layers)
        
        # Track which tasks contributed to each layer's importance
        self.task_contributions = defaultdict(list)
        
        logger.info(f"Initialized SharedImportanceMatrix for {num_layers} layers")
    
    def update_importance(self, task_id: str, layer_importance: Dict[str, float],
                         confidence: float = 1.0):
        """Update shared importance matrix with new task information"""
        
        # Convert layer importance to tensor
        importance_values = []
        if not self.layer_names:
            self.layer_names = list(layer_importance.keys())
        
        for layer_name in self.layer_names:
            importance = layer_importance.get(layer_name, 0.0)
            importance_values.append(importance)
        
        importance_tensor = torch.tensor(importance_values, dtype=torch.float32)
        
        # Update matrix using exponential moving average
        if self.task_count == 0:
            self.importance_matrix[:, 0] = importance_tensor[:self.num_layers]
        else:
            # Add new column for new task or update existing
            if self.task_count < self.importance_matrix.shape[1]:
                self.importance_matrix[:, self.task_count] = importance_tensor[:self.num_layers]
            else:
                # Expand matrix if needed
                new_col = importance_tensor[:self.num_layers].unsqueeze(1)
                self.importance_matrix = torch.cat([self.importance_matrix, new_col], dim=1)
        
        # Update confidence scores
        self.confidence_scores = self.confidence_scores * 0.9 + importance_tensor[:self.num_layers] * 0.1 * confidence
        
        # Track task contributions
        for i, layer_name in enumerate(self.layer_names[:self.num_layers]):
            self.task_contributions[layer_name].append((task_id, importance_tensor[i].item()))
        
        self.task_count += 1
        
        logger.info(f"Updated shared importance matrix with task {task_id}")
    
    def get_shared_importance(self, top_k: Optional[int] = None) -> Dict[str, float]:
        """Get shared layer importance across all tasks"""
        
        if self.task_count == 0:
            return {}
        
        # Compute average importance across tasks
        avg_importance = torch.mean(self.importance_matrix[:, :self.task_count], dim=1)
        
        # Create dictionary
        shared_importance = {}
        for i, layer_name in enumerate(self.layer_names[:len(avg_importance)]):
            shared_importance[layer_name] = avg_importance[i].item()
        
        # Sort and potentially limit to top-k
        sorted_importance = dict(sorted(shared_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        if top_k:
            sorted_importance = dict(list(sorted_importance.items())[:top_k])
        
        return sorted_importance
    
    def get_task_similarity(self, task_id1: str, task_id2: str) -> float:
        """Compute similarity between two tasks based on layer importance"""
        
        # Find task indices
        task1_importance = None
        task2_importance = None
        
        for layer_name in self.layer_names:
            for tid, importance in self.task_contributions[layer_name]:
                if tid == task_id1 and task1_importance is None:
                    task1_importance = []
                if tid == task_id2 and task2_importance is None:
                    task2_importance = []
        
        # Simplified similarity computation
        if task1_importance is None or task2_importance is None:
            return 0.0
        
        # Use cosine similarity
        vec1 = torch.tensor([imp for _, imp in self.task_contributions[layer_name] 
                           if _ == task_id1][:self.num_layers])
        vec2 = torch.tensor([imp for _, imp in self.task_contributions[layer_name] 
                           if _ == task_id2][:self.num_layers])
        
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        # Pad to same length
        max_len = max(len(vec1), len(vec2))
        if len(vec1) < max_len:
            vec1 = torch.cat([vec1, torch.zeros(max_len - len(vec1))])
        if len(vec2) < max_len:
            vec2 = torch.cat([vec2, torch.zeros(max_len - len(vec2))])
        
        similarity = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
        return similarity.item()
    
    def get_critical_layers(self, threshold: float = 0.5) -> List[str]:
        """Get layers that are critical across multiple tasks"""
        
        shared_importance = self.get_shared_importance()
        critical_layers = [layer for layer, importance in shared_importance.items()
                          if importance >= threshold]
        
        return critical_layers


class TaskSpecificHead(nn.Module):
    """Task-specific head for multi-task learning"""
    
    def __init__(self, shared_dim: int, task_dim: int, output_dim: int, 
                 task_type: str = 'recommendation'):
        super().__init__()
        
        self.task_type = task_type
        self.shared_dim = shared_dim
        self.task_dim = task_dim
        
        # Task-specific layers
        self.task_projection = nn.Linear(shared_dim, task_dim)
        self.task_norm = nn.LayerNorm(task_dim)
        
        # Output layers based on task type
        if task_type == 'recommendation':
            self.output_layers = nn.Sequential(
                nn.Linear(task_dim, task_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(task_dim // 2, output_dim)
            )
        elif task_type == 'classification':
            self.output_layers = nn.Sequential(
                nn.Linear(task_dim, task_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(task_dim // 2, output_dim),
                nn.Softmax(dim=-1)
            )
        else:  # regression
            self.output_layers = nn.Sequential(
                nn.Linear(task_dim, task_dim // 2),
                nn.ReLU(),
                nn.Linear(task_dim // 2, output_dim)
            )
    
    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        # Project to task-specific space
        task_features = self.task_projection(shared_features)
        task_features = self.task_norm(task_features)
        task_features = F.relu(task_features)
        
        # Generate output
        output = self.output_layers(task_features)
        return output


class MultiTaskLayerwiseAdapter(nn.Module):
    """Multi-task learning framework for Layerwise Adapter"""
    
    def __init__(self, config: MultiTaskConfig, base_model: nn.Module):
        super().__init__()
        
        self.config = config
        self.base_model = base_model
        
        # Shared components
        self.shared_importance_matrix = None
        self.task_heads = nn.ModuleDict()
        self.task_metadata = {}
        
        # Meta-learning components
        self.meta_optimizer = None
        self.task_optimizers = {}
        
        # Shared feature extractor (based on critical layers)
        self.shared_extractor = None
        
        # Knowledge transfer components
        self.transfer_weights = nn.ParameterDict()
        
        logger.info("Initialized MultiTaskLayerwiseAdapter")
    
    def add_task(self, task_id: str, task_metadata: TaskMetadata, 
                 output_dim: int = 1):
        """Add a new task to the multi-task framework"""
        
        self.task_metadata[task_id] = task_metadata
        
        # Create task-specific head
        task_head = TaskSpecificHead(
            shared_dim=self.config.shared_embedding_dim,
            task_dim=self.config.task_specific_dim,
            output_dim=output_dim,
            task_type=task_metadata.task_type
        )
        
        self.task_heads[task_id] = task_head
        
        # Initialize task optimizer
        self.task_optimizers[task_id] = torch.optim.Adam(
            task_head.parameters(), 
            lr=self.config.task_learning_rate
        )
        
        # Initialize transfer weights
        self.transfer_weights[task_id] = nn.Parameter(torch.ones(1))
        
        logger.info(f"Added task {task_id} to multi-task framework")
    
    def analyze_task_importance(self, task_id: str, dataloader, 
                              device: torch.device = None) -> LayerImportanceProfile:
        """Analyze layer importance for a specific task"""
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use adaptive layer selector to analyze importance
        task_metadata = self.task_metadata[task_id]
        
        task_config = TaskConfig(
            task_type=task_metadata.task_type,
            domain=task_metadata.domain,
            data_size='large' if task_metadata.dataset_size > 100000 else 'small',
            feature_types=['categorical', 'numerical'],
            sparsity_level=task_metadata.sparsity,
            interaction_density=1 - task_metadata.sparsity
        )
        
        # Create adaptive selector
        selector = AdaptiveLayerSelector()
        
        # Analyze importance
        profile = selector.auto_select_layers(self.base_model, task_config, dataloader, device)
        
        # Update shared importance matrix
        if self.shared_importance_matrix is None:
            num_layers = len(profile.layer_scores)
            self.shared_importance_matrix = SharedImportanceMatrix(num_layers)
        
        self.shared_importance_matrix.update_importance(
            task_id, profile.layer_scores, profile.confidence_score
        )
        
        return profile
    
    def build_shared_extractor(self):
        """Build shared feature extractor based on critical layers"""
        
        if self.shared_importance_matrix is None:
            logger.warning("No shared importance matrix available. Run analyze_task_importance first.")
            return
        
        # Get critical layers
        critical_layers = self.shared_importance_matrix.get_critical_layers(
            threshold=self.config.importance_threshold
        )
        
        if not critical_layers:
            logger.warning("No critical layers found. Using all layers.")
            critical_layers = self.shared_importance_matrix.layer_names
        
        # Build extractor from critical layers
        extractor_layers = []
        
        for layer_name in critical_layers:
            # Find corresponding layer in base model
            for name, module in self.base_model.named_modules():
                if layer_name in name:
                    extractor_layers.append(copy.deepcopy(module))
                    break
        
        if extractor_layers:
            self.shared_extractor = nn.Sequential(*extractor_layers)
            logger.info(f"Built shared extractor with {len(extractor_layers)} critical layers")
        else:
            # Fallback: use a simple linear layer
            self.shared_extractor = nn.Linear(128, self.config.shared_embedding_dim)
            logger.warning("Using fallback linear layer for shared extractor")
    
    def forward(self, task_id: str, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a specific task"""
        
        if task_id not in self.task_heads:
            raise ValueError(f"Task {task_id} not found. Add task first.")
        
        # Extract shared features
        if self.shared_extractor is not None:
            try:
                shared_features = self.shared_extractor(x)
            except:
                # Fallback: use average pooling
                shared_features = F.adaptive_avg_pool1d(x.view(x.size(0), -1).unsqueeze(1), 
                                                       self.config.shared_embedding_dim).squeeze(1)
        else:
            # Simple projection
            shared_features = F.adaptive_avg_pool1d(x.view(x.size(0), -1).unsqueeze(1), 
                                                   self.config.shared_embedding_dim).squeeze(1)
        
        # Task-specific processing
        task_output = self.task_heads[task_id](shared_features)
        
        return task_output
    
    def compute_transfer_loss(self, source_task: str, target_task: str,
                            source_features: torch.Tensor, 
                            target_features: torch.Tensor) -> torch.Tensor:
        """Compute knowledge transfer loss between tasks"""
        
        # Similarity-based transfer
        task_similarity = self.shared_importance_matrix.get_task_similarity(
            source_task, target_task
        )
        
        # Transfer weight
        transfer_weight = self.transfer_weights[target_task] * task_similarity
        
        # Feature alignment loss
        transfer_loss = F.mse_loss(source_features, target_features) * transfer_weight
        
        return transfer_loss
    
    def meta_train_step(self, task_batches: Dict[str, Any], device: torch.device):
        """Perform one meta-training step across all tasks"""
        
        total_loss = 0.0
        task_losses = {}
        
        # Initialize meta optimizer if needed
        if self.meta_optimizer is None:
            shared_params = []
            if self.shared_extractor is not None:
                shared_params.extend(self.shared_extractor.parameters())
            shared_params.extend(self.transfer_weights.parameters())
            
            self.meta_optimizer = torch.optim.Adam(
                shared_params, lr=self.config.meta_learning_rate
            )
        
        # Process each task
        for task_id, batch in task_batches.items():
            if task_id not in self.task_heads:
                continue
            
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(device), batch[1].to(device)
            else:
                inputs, targets = batch.to(device), None
            
            # Forward pass
            outputs = self.forward(task_id, inputs)
            
            # Compute task-specific loss
            task_metadata = self.task_metadata[task_id]
            
            if task_metadata.task_type == 'recommendation':
                if targets is not None:
                    task_loss = F.mse_loss(outputs.squeeze(), targets.float())
                else:
                    task_loss = torch.var(outputs)
            elif task_metadata.task_type == 'classification':
                task_loss = F.cross_entropy(outputs, targets.long())
            else:  # regression
                task_loss = F.mse_loss(outputs.squeeze(), targets.float())
            
            task_losses[task_id] = task_loss.item()
            total_loss += task_loss
        
        # Add transfer losses between similar tasks
        task_ids = list(task_batches.keys())
        for i, source_task in enumerate(task_ids):
            for target_task in task_ids[i+1:]:
                if source_task in self.task_heads and target_task in self.task_heads:
                    # Get features for both tasks
                    source_inputs = task_batches[source_task][0].to(device)
                    target_inputs = task_batches[target_task][0].to(device)
                    
                    # Extract shared features
                    with torch.no_grad():
                        source_features = self.shared_extractor(source_inputs) if self.shared_extractor else source_inputs.mean(dim=1)
                        target_features = self.shared_extractor(target_inputs) if self.shared_extractor else target_inputs.mean(dim=1)
                    
                    # Compute transfer loss
                    transfer_loss = self.compute_transfer_loss(
                        source_task, target_task, source_features, target_features
                    )
                    
                    total_loss += transfer_loss * 0.1  # Weight transfer loss
        
        # Backward pass
        self.meta_optimizer.zero_grad()
        for optimizer in self.task_optimizers.values():
            optimizer.zero_grad()
        
        total_loss.backward()
        
        # Update parameters
        self.meta_optimizer.step()
        for optimizer in self.task_optimizers.values():
            optimizer.step()
        
        return total_loss.item(), task_losses
    
    def adapt_to_new_task(self, new_task_id: str, new_task_metadata: TaskMetadata,
                         support_dataloader, adaptation_steps: int = None) -> float:
        """Adapt the model to a new task using few-shot learning"""
        
        if adaptation_steps is None:
            adaptation_steps = self.config.adaptation_steps
        
        logger.info(f"Adapting to new task: {new_task_id}")
        
        # Add new task
        self.add_task(new_task_id, new_task_metadata)
        
        # Find most similar existing task
        best_similarity = 0.0
        best_similar_task = None
        
        for existing_task_id in self.task_metadata:
            if existing_task_id == new_task_id:
                continue
            
            # Simple similarity based on domain and task type
            existing_metadata = self.task_metadata[existing_task_id]
            similarity = 0.0
            
            if existing_metadata.domain == new_task_metadata.domain:
                similarity += 0.5
            if existing_metadata.task_type == new_task_metadata.task_type:
                similarity += 0.5
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_similar_task = existing_task_id
        
        # Initialize new task head with similar task parameters
        if best_similar_task and best_similarity > 0.5:
            logger.info(f"Initializing from similar task: {best_similar_task}")
            
            # Copy parameters from similar task
            similar_head = self.task_heads[best_similar_task]
            new_head = self.task_heads[new_task_id]
            
            # Copy compatible parameters
            for (name1, param1), (name2, param2) in zip(
                similar_head.named_parameters(), new_head.named_parameters()
            ):
                if param1.shape == param2.shape:
                    param2.data.copy_(param1.data)
        
        # Fine-tune on support set
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        optimizer = torch.optim.Adam(
            self.task_heads[new_task_id].parameters(),
            lr=self.config.task_learning_rate * 2  # Higher LR for adaptation
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for step in range(adaptation_steps):
            for batch in support_dataloader:
                # Prepare batch
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    inputs, targets = batch.to(device), None
                
                # Forward pass
                outputs = self.forward(new_task_id, inputs)
                
                # Compute loss
                if new_task_metadata.task_type == 'recommendation':
                    if targets is not None:
                        loss = F.mse_loss(outputs.squeeze(), targets.float())
                    else:
                        loss = torch.var(outputs)
                elif new_task_metadata.task_type == 'classification':
                    loss = F.cross_entropy(outputs, targets.long())
                else:  # regression
                    loss = F.mse_loss(outputs.squeeze(), targets.float())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Adaptation completed for {new_task_id}. Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate_task_transfer(self, source_tasks: List[str], 
                              target_task: str, test_dataloader) -> Dict[str, float]:
        """Evaluate knowledge transfer effectiveness"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        
        results = {}
        
        with torch.no_grad():
            total_loss = 0.0
            num_batches = 0
            
            for batch in test_dataloader:
                # Prepare batch
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                else:
                    inputs, targets = batch.to(device), None
                
                # Forward pass
                outputs = self.forward(target_task, inputs)
                
                # Compute loss
                target_metadata = self.task_metadata[target_task]
                
                if target_metadata.task_type == 'recommendation':
                    if targets is not None:
                        loss = F.mse_loss(outputs.squeeze(), targets.float())
                    else:
                        loss = torch.var(outputs)
                elif target_metadata.task_type == 'classification':
                    loss = F.cross_entropy(outputs, targets.long())
                else:  # regression
                    loss = F.mse_loss(outputs.squeeze(), targets.float())
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        results[target_task] = {
            'test_loss': avg_loss,
            'source_tasks': source_tasks,
            'transfer_effectiveness': self._compute_transfer_effectiveness(source_tasks, target_task)
        }
        
        return results
    
    def _compute_transfer_effectiveness(self, source_tasks: List[str], 
                                      target_task: str) -> float:
        """Compute transfer effectiveness score"""
        
        if not self.shared_importance_matrix:
            return 0.0
        
        total_similarity = 0.0
        for source_task in source_tasks:
            similarity = self.shared_importance_matrix.get_task_similarity(
                source_task, target_task
            )
            total_similarity += similarity
        
        avg_similarity = total_similarity / len(source_tasks) if source_tasks else 0.0
        return avg_similarity
    
    def generate_multi_task_report(self) -> str:
        """Generate comprehensive multi-task learning report"""
        
        report_lines = [
            "# Multi-Task Layerwise Adapter Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Framework Configuration",
            f"- **Shared Embedding Dimension**: {self.config.shared_embedding_dim}",
            f"- **Task-Specific Dimension**: {self.config.task_specific_dim}",
            f"- **Number of Tasks**: {len(self.task_metadata)}",
            "",
            "## Task Overview"
        ]
        
        for task_id, metadata in self.task_metadata.items():
            report_lines.extend([
                f"### {task_id}",
                f"- **Type**: {metadata.task_type}",
                f"- **Domain**: {metadata.domain}",
                f"- **Dataset Size**: {metadata.dataset_size:,}",
                f"- **Sparsity**: {metadata.sparsity:.3f}",
                ""
            ])
        
        # Shared importance analysis
        if self.shared_importance_matrix:
            shared_importance = self.shared_importance_matrix.get_shared_importance(top_k=10)
            critical_layers = self.shared_importance_matrix.get_critical_layers()
            
            report_lines.extend([
                "## Shared Layer Importance Analysis",
                f"- **Critical Layers**: {len(critical_layers)}",
                "",
                "### Top 10 Most Important Layers"
            ])
            
            for i, (layer, importance) in enumerate(shared_importance.items(), 1):
                report_lines.append(f"{i}. **{layer}**: {importance:.4f}")
            
            report_lines.extend([
                "",
                "### Critical Layers",
                ", ".join(critical_layers) if critical_layers else "None identified"
            ])
        
        # Task similarity analysis
        if len(self.task_metadata) > 1 and self.shared_importance_matrix:
            report_lines.extend([
                "",
                "## Task Similarity Analysis"
            ])
            
            task_ids = list(self.task_metadata.keys())
            for i, task1 in enumerate(task_ids):
                for task2 in task_ids[i+1:]:
                    similarity = self.shared_importance_matrix.get_task_similarity(task1, task2)
                    report_lines.append(f"- **{task1} ↔ {task2}**: {similarity:.3f}")
        
        return "\n".join(report_lines)
    
    def save_multi_task_model(self, filepath: Path):
        """Save multi-task model"""
        
        torch.save({
            'config': self.config,
            'task_metadata': self.task_metadata,
            'model_state_dict': self.state_dict(),
            'shared_importance_matrix': self.shared_importance_matrix,
            'task_optimizers_state': {
                task_id: optimizer.state_dict() 
                for task_id, optimizer in self.task_optimizers.items()
            }
        }, filepath)
        
        logger.info(f"Multi-task model saved to {filepath}")
    
    def load_multi_task_model(self, filepath: Path):
        """Load multi-task model"""
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.config = checkpoint['config']
        self.task_metadata = checkpoint['task_metadata']
        self.load_state_dict(checkpoint['model_state_dict'])
        self.shared_importance_matrix = checkpoint.get('shared_importance_matrix')
        
        # Restore optimizers
        if 'task_optimizers_state' in checkpoint:
            for task_id, optimizer_state in checkpoint['task_optimizers_state'].items():
                if task_id in self.task_optimizers:
                    self.task_optimizers[task_id].load_state_dict(optimizer_state)
        
        logger.info(f"Multi-task model loaded from {filepath}")


def create_multi_task_experiment(base_model: nn.Module, 
                                task_configs: List[Tuple[str, TaskMetadata]],
                                dataloaders: Dict[str, Any]) -> MultiTaskLayerwiseAdapter:
    """Create and initialize a multi-task experiment"""
    
    config = MultiTaskConfig()
    multi_task_adapter = MultiTaskLayerwiseAdapter(config, base_model)
    
    # Add all tasks
    for task_id, task_metadata in task_configs:
        multi_task_adapter.add_task(task_id, task_metadata)
    
    # Analyze task importance for each task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for task_id, _ in task_configs:
        if task_id in dataloaders:
            logger.info(f"Analyzing importance for task: {task_id}")
            multi_task_adapter.analyze_task_importance(
                task_id, dataloaders[task_id], device
            )
    
    # Build shared extractor
    multi_task_adapter.build_shared_extractor()
    
    logger.info(f"Multi-task experiment created with {len(task_configs)} tasks")
    
    return multi_task_adapter
