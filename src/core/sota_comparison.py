"""
SOTA Recommendation Algorithm Comparison Framework

This module implements comprehensive comparison experiments between the
Layerwise Adapter approach and state-of-the-art recommendation algorithms
including DeepFM, Wide&Deep, AutoInt, and other modern approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for different recommendation models"""
    model_name: str
    embedding_dim: int = 64
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 512
    num_epochs: int = 50
    use_batch_norm: bool = True
    activation: str = 'relu'


@dataclass
class ComparisonMetrics:
    """Metrics for model comparison"""
    model_name: str
    rmse: float
    mae: float
    auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    num_parameters: int = 0


class DeepFMModel(nn.Module):
    """DeepFM implementation for recommendation"""
    
    def __init__(self, num_users: int, num_items: int, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        
        # FM component
        self.fm_linear = nn.Linear(num_users + num_items, 1)
        
        # Deep component
        deep_dims = config.hidden_dims or [256, 128, 64]
        deep_layers = []
        
        input_dim = self.embedding_dim * 2
        for hidden_dim in deep_dims:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU() if config.activation == 'relu' else nn.Tanh(),
                nn.Dropout(config.dropout_rate)
            ])
            if config.use_batch_norm:
                deep_layers.insert(-1, nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        deep_layers.append(nn.Linear(input_dim, 1))
        self.deep_network = nn.Sequential(*deep_layers)
        
        # Final layer
        self.final_layer = nn.Linear(2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # FM component (simplified)
        fm_input = torch.cat([user_emb, item_emb], dim=1)
        fm_output = torch.sum(fm_input, dim=1, keepdim=True)
        
        # Deep component
        deep_input = torch.cat([user_emb, item_emb], dim=1)
        deep_output = self.deep_network(deep_input)
        
        # Combine FM and Deep
        combined = torch.cat([fm_output, deep_output], dim=1)
        output = self.final_layer(combined)
        
        return output.squeeze()


class WideAndDeepModel(nn.Module):
    """Wide & Deep model implementation"""
    
    def __init__(self, num_users: int, num_items: int, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # Wide component (linear)
        self.wide_layer = nn.Linear(num_users + num_items, 1)
        
        # Deep component
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        
        deep_dims = config.hidden_dims or [256, 128, 64]
        deep_layers = []
        
        input_dim = self.embedding_dim * 2
        for hidden_dim in deep_dims:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU() if config.activation == 'relu' else nn.Tanh(),
                nn.Dropout(config.dropout_rate)
            ])
            if config.use_batch_norm:
                deep_layers.insert(-1, nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        deep_layers.append(nn.Linear(input_dim, 1))
        self.deep_network = nn.Sequential(*deep_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        batch_size = user_ids.size(0)
        
        # Wide component (one-hot encoding simulation)
        wide_user = F.one_hot(user_ids, num_classes=self.user_embedding.num_embeddings).float()
        wide_item = F.one_hot(item_ids, num_classes=self.item_embedding.num_embeddings).float()
        wide_input = torch.cat([wide_user, wide_item], dim=1)
        wide_output = self.wide_layer(wide_input)
        
        # Deep component
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        deep_input = torch.cat([user_emb, item_emb], dim=1)
        deep_output = self.deep_network(deep_input)
        
        # Combine wide and deep
        output = wide_output + deep_output
        
        return output.squeeze()


class AutoIntModel(nn.Module):
    """AutoInt model with automatic feature interaction learning"""
    
    def __init__(self, num_users: int, num_items: int, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.num_heads = 4
        self.num_layers = 3
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(self.embedding_dim, self.num_heads, batch_first=True)
            for _ in range(self.num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim)
            for _ in range(self.num_layers)
        ])
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Stack embeddings for attention
        # Shape: (batch_size, 2, embedding_dim)
        stacked_emb = torch.stack([user_emb, item_emb], dim=1)
        
        # Apply attention layers
        x = stacked_emb
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention
            attn_output, _ = attention(x, x, x)
            # Residual connection and layer norm
            x = layer_norm(x + attn_output)
        
        # Flatten for output layer
        x = x.view(x.size(0), -1)
        output = self.output_layer(x)
        
        return output.squeeze()


class NCFModel(nn.Module):
    """Neural Collaborative Filtering model"""
    
    def __init__(self, num_users: int, num_items: int, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # GMF component
        self.gmf_user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, self.embedding_dim)
        
        # MLP component
        self.mlp_user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, self.embedding_dim)
        
        # MLP layers
        mlp_dims = config.hidden_dims or [256, 128, 64]
        mlp_layers = []
        
        input_dim = self.embedding_dim * 2
        for hidden_dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.final_layer = nn.Linear(self.embedding_dim + mlp_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF component
        gmf_user_emb = self.gmf_user_embedding(user_ids)
        gmf_item_emb = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user_emb * gmf_item_emb  # Element-wise product
        
        # MLP component
        mlp_user_emb = self.mlp_user_embedding(user_ids)
        mlp_item_emb = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.final_layer(combined)
        
        return output.squeeze()


class LayerwiseAdapterModel(nn.Module):
    """Our Layerwise Adapter model for comparison"""
    
    def __init__(self, num_users: int, num_items: int, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # Base embeddings
        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        
        # Layerwise adapter components
        self.adapter_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            )
            for hidden_dim in (config.hidden_dims or [128, 64, 32])
        ])
        
        # Fisher information simulation (learnable importance weights)
        self.layer_importance = nn.Parameter(torch.ones(len(self.adapter_layers)))
        
        # Output layer
        final_dim = (config.hidden_dims or [128, 64, 32])[-1]
        self.output_layer = nn.Linear(final_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Base embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Combine embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        # Apply adapter layers with importance weighting
        for i, layer in enumerate(self.adapter_layers):
            layer_output = layer(x)
            # Apply importance weighting
            importance = torch.sigmoid(self.layer_importance[i])
            x = layer_output * importance
        
        # Final prediction
        output = self.output_layer(x)
        
        return output.squeeze()


class SOTAComparisonFramework:
    """Framework for comparing SOTA recommendation algorithms"""
    
    def __init__(self, results_dir: Path = Path("results/sota_comparison")):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_configs = {
            'DeepFM': ModelConfig(
                model_name='DeepFM',
                embedding_dim=64,
                hidden_dims=[256, 128, 64],
                dropout_rate=0.2,
                learning_rate=0.001
            ),
            'WideAndDeep': ModelConfig(
                model_name='WideAndDeep',
                embedding_dim=64,
                hidden_dims=[256, 128, 64],
                dropout_rate=0.2,
                learning_rate=0.001
            ),
            'AutoInt': ModelConfig(
                model_name='AutoInt',
                embedding_dim=64,
                dropout_rate=0.2,
                learning_rate=0.001
            ),
            'NCF': ModelConfig(
                model_name='NCF',
                embedding_dim=64,
                hidden_dims=[256, 128, 64],
                dropout_rate=0.2,
                learning_rate=0.001
            ),
            'LayerwiseAdapter': ModelConfig(
                model_name='LayerwiseAdapter',
                embedding_dim=64,
                hidden_dims=[128, 64, 32],
                dropout_rate=0.2,
                learning_rate=0.001
            )
        }
        
        self.model_classes = {
            'DeepFM': DeepFMModel,
            'WideAndDeep': WideAndDeepModel,
            'AutoInt': AutoIntModel,
            'NCF': NCFModel,
            'LayerwiseAdapter': LayerwiseAdapterModel
        }
        
        self.comparison_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized SOTAComparisonFramework with {len(self.model_configs)} models")
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[torch.utils.data.DataLoader, 
                                                   torch.utils.data.DataLoader, 
                                                   Dict[str, int]]:
        """Prepare data for training and testing"""
        
        # Encode users and items
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        df = df.copy()
        df['user_encoded'] = user_encoder.fit_transform(df['user_id'])
        df['item_encoded'] = item_encoder.fit_transform(df['item_id'])
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Create datasets
        train_dataset = self._create_dataset(train_df)
        test_dataset = self._create_dataset(test_df)
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=512, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=512, shuffle=False
        )
        
        # Data info
        data_info = {
            'num_users': df['user_encoded'].nunique(),
            'num_items': df['item_encoded'].nunique(),
            'num_interactions': len(df)
        }
        
        return train_loader, test_loader, data_info
    
    def _create_dataset(self, df: pd.DataFrame):
        """Create PyTorch dataset"""
        
        class RatingDataset(torch.utils.data.Dataset):
            def __init__(self, df):
                self.users = torch.LongTensor(df['user_encoded'].values)
                self.items = torch.LongTensor(df['item_encoded'].values)
                self.ratings = torch.FloatTensor(df['rating'].values)
                
            def __len__(self):
                return len(self.users)
                
            def __getitem__(self, idx):
                return self.users[idx], self.items[idx], self.ratings[idx]
        
        return RatingDataset(df)
    
    def train_model(self, model: nn.Module, train_loader, config: ModelConfig,
                   device: torch.device) -> Dict[str, float]:
        """Train a single model"""
        
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()
        
        training_metrics = {
            'training_time': 0.0,
            'final_loss': 0.0
        }
        
        start_time = time.time()
        
        for epoch in range(min(config.num_epochs, 20)):  # Limit epochs for comparison
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (users, items, ratings) in enumerate(train_loader):
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                
                optimizer.zero_grad()
                predictions = model(users, items)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Limit batches for quick comparison
                if batch_idx >= 100:
                    break
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            if epoch % 5 == 0:
                logger.info(f"  Epoch {epoch}/{config.num_epochs}, Loss: {avg_loss:.4f}")
        
        training_metrics['training_time'] = time.time() - start_time
        training_metrics['final_loss'] = avg_loss
        
        return training_metrics
    
    def evaluate_model(self, model: nn.Module, test_loader, 
                      device: torch.device) -> Dict[str, float]:
        """Evaluate a single model"""
        
        model.eval()
        predictions = []
        actuals = []
        
        inference_start = time.time()
        
        with torch.no_grad():
            for batch_idx, (users, items, ratings) in enumerate(test_loader):
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                
                pred = model(users, items)
                
                predictions.extend(pred.cpu().numpy())
                actuals.extend(ratings.cpu().numpy())
                
                # Limit batches for quick evaluation
                if batch_idx >= 50:
                    break
        
        inference_time = time.time() - inference_start
        
        # Compute metrics
        if len(predictions) == 0:
            return {'rmse': float('inf'), 'mae': float('inf'), 'inference_time': inference_time}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        # Additional metrics for classification-like evaluation
        try:
            # Convert to binary classification (rating >= 4 is positive)
            binary_actuals = (actuals >= 4).astype(int)
            binary_predictions = (predictions >= 4).astype(int)
            
            precision = np.sum((binary_predictions == 1) & (binary_actuals == 1)) / max(np.sum(binary_predictions == 1), 1)
            recall = np.sum((binary_predictions == 1) & (binary_actuals == 1)) / max(np.sum(binary_actuals == 1), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
            # AUC if possible
            try:
                auc_score = roc_auc_score(binary_actuals, predictions)
            except:
                auc_score = None
                
        except:
            precision = recall = f1 = auc_score = None
        
        return {
            'rmse': rmse,
            'mae': mae,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc_score,
            'inference_time': inference_time
        }
    
    def get_model_complexity(self, model: nn.Module) -> Dict[str, Union[int, float]]:
        """Get model complexity metrics"""
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough approximation)
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        total_memory = param_memory + buffer_memory
        
        return {
            'num_parameters': num_params,
            'memory_usage_mb': total_memory / (1024 * 1024)
        }
    
    def run_comparison(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """Run comprehensive comparison across all models"""
        
        logger.info(f"Starting SOTA comparison on {dataset_name}")
        
        # Prepare data
        train_loader, test_loader, data_info = self.prepare_data(df)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        comparison_results = {
            'dataset_name': dataset_name,
            'dataset_info': data_info,
            'model_results': {},
            'timestamp': self.timestamp
        }
        
        # Test each model
        for model_name in self.model_configs:
            logger.info(f"Testing model: {model_name}")
            
            try:
                # Create model
                config = self.model_configs[model_name]
                model_class = self.model_classes[model_name]
                
                model = model_class(
                    num_users=data_info['num_users'],
                    num_items=data_info['num_items'],
                    config=config
                )
                
                # Get model complexity
                complexity_metrics = self.get_model_complexity(model)
                
                # Train model
                logger.info(f"  Training {model_name}...")
                training_metrics = self.train_model(model, train_loader, config, device)
                
                # Evaluate model
                logger.info(f"  Evaluating {model_name}...")
                eval_metrics = self.evaluate_model(model, test_loader, device)
                
                # Combine metrics
                model_metrics = ComparisonMetrics(
                    model_name=model_name,
                    rmse=eval_metrics['rmse'],
                    mae=eval_metrics['mae'],
                    auc=eval_metrics.get('auc'),
                    precision=eval_metrics.get('precision'),
                    recall=eval_metrics.get('recall'),
                    f1_score=eval_metrics.get('f1_score'),
                    training_time=training_metrics['training_time'],
                    inference_time=eval_metrics['inference_time'],
                    memory_usage=complexity_metrics['memory_usage_mb'],
                    num_parameters=complexity_metrics['num_parameters']
                )
                
                comparison_results['model_results'][model_name] = {
                    'rmse': model_metrics.rmse,
                    'mae': model_metrics.mae,
                    'auc': model_metrics.auc,
                    'precision': model_metrics.precision,
                    'recall': model_metrics.recall,
                    'f1_score': model_metrics.f1_score,
                    'training_time': model_metrics.training_time,
                    'inference_time': model_metrics.inference_time,
                    'memory_usage_mb': model_metrics.memory_usage,
                    'num_parameters': model_metrics.num_parameters
                }
                
                logger.info(f"  {model_name} - RMSE: {model_metrics.rmse:.4f}, "
                           f"MAE: {model_metrics.mae:.4f}, "
                           f"Params: {model_metrics.num_parameters:,}")
                
            except Exception as e:
                logger.error(f"Error testing {model_name}: {e}")
                logger.error(traceback.format_exc())
                
                comparison_results['model_results'][model_name] = {
                    'error': str(e),
                    'rmse': float('inf'),
                    'mae': float('inf')
                }
        
        # Save results
        result_file = self.results_dir / f"sota_comparison_{dataset_name}_{self.timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        logger.info(f"SOTA comparison completed. Results saved to {result_file}")
        
        return comparison_results
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive comparison report"""
        
        report_lines = [
            "# SOTA Recommendation Algorithms Comparison Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Dataset**: {results['dataset_name']}",
            "",
            "## Dataset Information",
            f"- **Users**: {results['dataset_info']['num_users']:,}",
            f"- **Items**: {results['dataset_info']['num_items']:,}",
            f"- **Interactions**: {results['dataset_info']['num_interactions']:,}",
            "",
            "## Model Performance Comparison",
            ""
        ]
        
        # Sort models by RMSE
        model_results = results['model_results']
        sorted_models = sorted(model_results.items(), 
                             key=lambda x: x[1].get('rmse', float('inf')))
        
        # Performance table
        report_lines.extend([
            "| Model | RMSE | MAE | Params | Memory (MB) | Train Time (s) | Inference Time (s) |",
            "|-------|------|-----|--------|-------------|----------------|-------------------|"
        ])
        
        for model_name, metrics in sorted_models:
            if 'error' not in metrics:
                report_lines.append(
                    f"| {model_name} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | "
                    f"{metrics.get('num_parameters', 0):,} | "
                    f"{metrics.get('memory_usage_mb', 0):.1f} | "
                    f"{metrics.get('training_time', 0):.1f} | "
                    f"{metrics.get('inference_time', 0):.3f} |"
                )
        
        report_lines.extend([
            "",
            "## Key Findings",
            ""
        ])
        
        # Find best model
        best_model = min(model_results.items(), 
                        key=lambda x: x[1].get('rmse', float('inf')))
        
        if 'error' not in best_model[1]:
            report_lines.extend([
                f"### Best Performing Model: {best_model[0]}",
                f"- **RMSE**: {best_model[1]['rmse']:.4f}",
                f"- **MAE**: {best_model[1]['mae']:.4f}",
                f"- **Parameters**: {best_model[1].get('num_parameters', 0):,}",
                f"- **Training Time**: {best_model[1].get('training_time', 0):.1f}s",
                ""
            ])
        
        # Efficiency analysis
        if 'LayerwiseAdapter' in model_results and 'error' not in model_results['LayerwiseAdapter']:
            layerwise_metrics = model_results['LayerwiseAdapter']
            report_lines.extend([
                "### Layerwise Adapter Analysis",
                f"- **RMSE**: {layerwise_metrics['rmse']:.4f}",
                f"- **MAE**: {layerwise_metrics['mae']:.4f}",
                f"- **Parameters**: {layerwise_metrics.get('num_parameters', 0):,}",
                f"- **Memory Usage**: {layerwise_metrics.get('memory_usage_mb', 0):.1f} MB",
                ""
            ])
            
            # Compare with best baseline
            baselines = [name for name in model_results.keys() if name != 'LayerwiseAdapter']
            if baselines:
                best_baseline = min([(name, model_results[name]) for name in baselines], 
                                  key=lambda x: x[1].get('rmse', float('inf')))
                
                if 'error' not in best_baseline[1]:
                    rmse_improvement = (best_baseline[1]['rmse'] - layerwise_metrics['rmse']) / best_baseline[1]['rmse'] * 100
                    param_reduction = (best_baseline[1].get('num_parameters', 0) - layerwise_metrics.get('num_parameters', 0)) / best_baseline[1].get('num_parameters', 1) * 100
                    
                    report_lines.extend([
                        f"#### Comparison with Best Baseline ({best_baseline[0]})",
                        f"- **RMSE Improvement**: {rmse_improvement:+.1f}%",
                        f"- **Parameter Reduction**: {param_reduction:+.1f}%",
                        ""
                    ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.results_dir / f"comparison_report_{results['dataset_name']}_{self.timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Comparison report saved to {report_file}")
        
        return report_content
    
    def run_multi_dataset_comparison(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run comparison across multiple datasets"""
        
        logger.info(f"Starting multi-dataset SOTA comparison on {len(datasets)} datasets")
        
        all_results = {
            'datasets': {},
            'summary': {},
            'timestamp': self.timestamp
        }
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            
            try:
                dataset_results = self.run_comparison(df, dataset_name)
                all_results['datasets'][dataset_name] = dataset_results
                
                # Generate individual report
                self.generate_comparison_report(dataset_results)
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                all_results['datasets'][dataset_name] = {
                    'error': str(e)
                }
        
        # Generate summary analysis
        all_results['summary'] = self._generate_summary_analysis(all_results['datasets'])
        
        # Save comprehensive results
        comprehensive_file = self.results_dir / f"multi_dataset_sota_comparison_{self.timestamp}.json"
        with open(comprehensive_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Multi-dataset SOTA comparison completed. Results saved to {comprehensive_file}")
        
        return all_results
    
    def _generate_summary_analysis(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary analysis across datasets"""
        
        model_performance = defaultdict(list)
        
        for dataset_name, results in dataset_results.items():
            if 'error' in results or 'model_results' not in results:
                continue
                
            for model_name, metrics in results['model_results'].items():
                if 'error' not in metrics:
                    model_performance[model_name].append({
                        'dataset': dataset_name,
                        'rmse': metrics['rmse'],
                        'mae': metrics['mae']
                    })
        
        # Compute average performance
        summary = {}
        for model_name, performances in model_performance.items():
            if performances:
                avg_rmse = np.mean([p['rmse'] for p in performances])
                avg_mae = np.mean([p['mae'] for p in performances])
                
                summary[model_name] = {
                    'avg_rmse': avg_rmse,
                    'avg_mae': avg_mae,
                    'num_datasets': len(performances),
                    'datasets': [p['dataset'] for p in performances]
                }
        
        return summary
