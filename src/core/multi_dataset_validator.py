"""
Multi-Dataset Validation Framework for Layerwise Adapter

This module provides a comprehensive framework for validating the Layerwise Adapter
approach across multiple recommendation datasets including Netflix, Yelp, MovieLens,
and Last.fm to demonstrate generalization capabilities.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from datetime import datetime
import requests
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from .adaptive_layer_selector import AdaptiveLayerSelector, TaskConfig, LayerImportanceProfile
from .fisher_information import LayerwiseFisherAnalyzer
from ..recommender.base_recommender import BaseRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for different datasets"""
    name: str
    url: Optional[str] = None
    local_path: Optional[Path] = None
    user_col: str = 'user_id'
    item_col: str = 'item_id'
    rating_col: str = 'rating'
    timestamp_col: Optional[str] = None
    min_interactions: int = 5
    test_size: float = 0.2
    has_features: bool = False
    feature_cols: List[str] = None


class DatasetDownloader:
    """Download and prepare various recommendation datasets"""
    
    def __init__(self, data_dir: Path = Path("dataset")):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'movielens_100k': DatasetConfig(
                name='movielens_100k',
                url='https://files.grouplens.org/datasets/movielens/ml-100k.zip',
                local_path=self.data_dir / 'movielens' / '100k',
                user_col='user_id',
                item_col='movie_id',
                rating_col='rating',
                timestamp_col='timestamp'
            ),
            'movielens_1m': DatasetConfig(
                name='movielens_1m',
                url='https://files.grouplens.org/datasets/movielens/ml-1m.zip',
                local_path=self.data_dir / 'movielens' / '1m',
                user_col='user_id',
                item_col='movie_id',
                rating_col='rating',
                timestamp_col='timestamp'
            ),
            'amazon_books': DatasetConfig(
                name='amazon_books',
                local_path=self.data_dir / 'amazon',
                user_col='user_id',
                item_col='asin',
                rating_col='overall',
                timestamp_col='unixReviewTime'
            ),
            'yelp': DatasetConfig(
                name='yelp',
                local_path=self.data_dir / 'yelp',
                user_col='user_id',
                item_col='business_id',
                rating_col='stars',
                timestamp_col='date'
            )
        }
        
        logger.info(f"Initialized DatasetDownloader with data_dir: {data_dir}")
    
    def download_dataset(self, dataset_name: str) -> Path:
        """Download and extract a dataset"""
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.datasets[dataset_name]
        
        if config.url is None:
            logger.info(f"Dataset {dataset_name} should be manually placed in {config.local_path}")
            return config.local_path
        
        # Create dataset directory
        dataset_dir = config.local_path
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download if not exists
        zip_path = dataset_dir / f"{dataset_name}.zip"
        
        if not zip_path.exists():
            logger.info(f"Downloading {dataset_name} from {config.url}")
            response = requests.get(config.url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {dataset_name} to {zip_path}")
        
        # Extract if needed
        if not any(dataset_dir.glob("*.dat")) and not any(dataset_dir.glob("*.csv")):
            logger.info(f"Extracting {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
        
        return dataset_dir
    
    def load_movielens_100k(self) -> pd.DataFrame:
        """Load MovieLens 100K dataset"""
        dataset_dir = self.download_dataset('movielens_100k')
        
        # Look for the extracted directory
        ml_dirs = list(dataset_dir.glob("ml-*"))
        if ml_dirs:
            data_file = ml_dirs[0] / "u.data"
        else:
            data_file = dataset_dir / "u.data"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Could not find ratings file at {data_file}")
        
        # Load ratings data
        df = pd.read_csv(data_file, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        
        logger.info(f"Loaded MovieLens 100K: {len(df)} interactions, "
                   f"{df['user_id'].nunique()} users, {df['movie_id'].nunique()} items")
        
        return df
    
    def load_movielens_1m(self) -> pd.DataFrame:
        """Load MovieLens 1M dataset"""
        dataset_dir = self.download_dataset('movielens_1m')
        
        # Look for the extracted directory
        ml_dirs = list(dataset_dir.glob("ml-*"))
        if ml_dirs:
            data_file = ml_dirs[0] / "ratings.dat"
        else:
            data_file = dataset_dir / "ratings.dat"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Could not find ratings file at {data_file}")
        
        # Load ratings data
        df = pd.read_csv(data_file, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                        engine='python')
        
        logger.info(f"Loaded MovieLens 1M: {len(df)} interactions, "
                   f"{df['user_id'].nunique()} users, {df['movie_id'].nunique()} items")
        
        return df
    
    def load_amazon_books(self) -> pd.DataFrame:
        """Load Amazon Books dataset"""
        config = self.datasets['amazon_books']
        
        # Try different possible file locations
        possible_files = [
            config.local_path / "Books_reviews.parquet",
            config.local_path / "books_reviews.parquet",
            config.local_path / "Books.parquet"
        ]
        
        data_file = None
        for file_path in possible_files:
            if file_path.exists():
                data_file = file_path
                break
        
        if data_file is None:
            logger.warning(f"Amazon Books data not found. Please place books review data in {config.local_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(data_file)
            
            # Standardize column names
            column_mapping = {
                'reviewerID': 'user_id',
                'asin': 'item_id', 
                'overall': 'rating',
                'unixReviewTime': 'timestamp'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Filter to required columns
            required_cols = ['user_id', 'item_id', 'rating']
            available_cols = [col for col in required_cols if col in df.columns]
            df = df[available_cols]
            
            logger.info(f"Loaded Amazon Books: {len(df)} interactions, "
                       f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Amazon Books data: {e}")
            return pd.DataFrame()
    
    def load_yelp_dataset(self) -> pd.DataFrame:
        """Load Yelp dataset (placeholder - requires manual download)"""
        config = self.datasets['yelp']
        
        logger.info("Yelp dataset requires manual download from https://www.yelp.com/dataset")
        logger.info(f"Please place yelp_academic_dataset_review.json in {config.local_path}")
        
        # Try to load if available
        review_file = config.local_path / "yelp_academic_dataset_review.json"
        
        if not review_file.exists():
            logger.warning(f"Yelp dataset not found at {review_file}")
            return pd.DataFrame()
        
        try:
            # Load JSON lines file
            reviews = []
            with open(review_file, 'r') as f:
                for line in f:
                    if len(reviews) >= 100000:  # Limit for demo
                        break
                    reviews.append(json.loads(line))
            
            df = pd.DataFrame(reviews)
            
            # Standardize columns
            df = df.rename(columns={
                'user_id': 'user_id',
                'business_id': 'item_id',
                'stars': 'rating',
                'date': 'timestamp'
            })
            
            # Filter to required columns
            required_cols = ['user_id', 'item_id', 'rating']
            df = df[required_cols]
            
            logger.info(f"Loaded Yelp dataset: {len(df)} interactions, "
                       f"{df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Yelp data: {e}")
            return pd.DataFrame()


class MultiDatasetValidator:
    """Main class for multi-dataset validation of Layerwise Adapter"""
    
    def __init__(self, results_dir: Path = Path("results/multi_dataset_validation")):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.downloader = DatasetDownloader()
        self.adaptive_selector = AdaptiveLayerSelector()
        self.fisher_analyzer = LayerwiseFisherAnalyzer()
        
        self.validation_results = {}
        self.dataset_profiles = {}
        
        # Timestamp for this validation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized MultiDatasetValidator with results_dir: {results_dir}")
    
    def prepare_dataset(self, df: pd.DataFrame, dataset_name: str, 
                       min_interactions: int = 5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Prepare dataset for training and validation"""
        
        if df.empty:
            return df, {}
        
        logger.info(f"Preparing dataset: {dataset_name}")
        
        # Filter users and items with minimum interactions
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        df_filtered = df[
            (df['user_id'].isin(valid_users)) & 
            (df['item_id'].isin(valid_items))
        ].copy()
        
        # Encode users and items
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        df_filtered['user_encoded'] = user_encoder.fit_transform(df_filtered['user_id'])
        df_filtered['item_encoded'] = item_encoder.fit_transform(df_filtered['item_id'])
        
        # Dataset statistics
        stats = {
            'original_interactions': len(df),
            'filtered_interactions': len(df_filtered),
            'num_users': df_filtered['user_encoded'].nunique(),
            'num_items': df_filtered['item_encoded'].nunique(),
            'sparsity': 1 - (len(df_filtered) / (df_filtered['user_encoded'].nunique() * df_filtered['item_encoded'].nunique())),
            'rating_stats': {
                'mean': df_filtered['rating'].mean(),
                'std': df_filtered['rating'].std(),
                'min': df_filtered['rating'].min(),
                'max': df_filtered['rating'].max()
            }
        }
        
        logger.info(f"Dataset {dataset_name} prepared: {stats['filtered_interactions']} interactions, "
                   f"{stats['num_users']} users, {stats['num_items']} items, "
                   f"sparsity: {stats['sparsity']:.4f}")
        
        return df_filtered, stats
    
    def create_task_config(self, dataset_name: str, dataset_stats: Dict[str, Any]) -> TaskConfig:
        """Create task configuration based on dataset characteristics"""
        
        # Determine data size
        num_interactions = dataset_stats['filtered_interactions']
        if num_interactions < 100000:
            data_size = 'small'
        elif num_interactions < 1000000:
            data_size = 'medium'
        else:
            data_size = 'large'
        
        # Determine domain
        if 'movie' in dataset_name.lower():
            domain = 'entertainment'
        elif 'amazon' in dataset_name.lower() or 'book' in dataset_name.lower():
            domain = 'e-commerce'
        elif 'yelp' in dataset_name.lower():
            domain = 'social'
        else:
            domain = 'entertainment'
        
        return TaskConfig(
            task_type='recommendation',
            domain=domain,
            data_size=data_size,
            feature_types=['categorical', 'numerical'],
            sparsity_level=dataset_stats['sparsity'],
            interaction_density=1 - dataset_stats['sparsity']
        )
    
    def validate_on_dataset(self, dataset_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate Layerwise Adapter on a single dataset"""
        
        if df.empty:
            logger.warning(f"Skipping empty dataset: {dataset_name}")
            return {}
        
        logger.info(f"Starting validation on dataset: {dataset_name}")
        
        # Prepare dataset
        df_prepared, dataset_stats = self.prepare_dataset(df, dataset_name)
        
        if df_prepared.empty:
            logger.warning(f"Dataset {dataset_name} is empty after preparation")
            return {}
        
        # Create task configuration
        task_config = self.create_task_config(dataset_name, dataset_stats)
        
        # Split data
        train_df, test_df = train_test_split(
            df_prepared, test_size=0.2, random_state=42, stratify=None
        )
        
        # Create simple recommendation model for testing
        model = self._create_simple_model(dataset_stats['num_users'], dataset_stats['num_items'])
        
        # Create dataloaders
        train_loader = self._create_dataloader(train_df)
        test_loader = self._create_dataloader(test_df)
        
        # Train model briefly
        self._train_model(model, train_loader, epochs=5)
        
        # Apply adaptive layer selection
        layer_profile = self.adaptive_selector.auto_select_layers(
            model, task_config, train_loader
        )
        
        # Evaluate model
        metrics = self._evaluate_model(model, test_loader)
        
        # Compile results
        results = {
            'dataset_name': dataset_name,
            'dataset_stats': dataset_stats,
            'task_config': {
                'task_type': task_config.task_type,
                'domain': task_config.domain,
                'data_size': task_config.data_size,
                'sparsity_level': task_config.sparsity_level
            },
            'layer_profile': {
                'layer_scores': layer_profile.layer_scores,
                'importance_distribution': layer_profile.importance_distribution,
                'critical_layers': layer_profile.critical_layers,
                'confidence_score': layer_profile.confidence_score
            },
            'performance_metrics': metrics,
            'timestamp': self.timestamp
        }
        
        # Store profile for cross-dataset analysis
        self.dataset_profiles[dataset_name] = layer_profile
        
        logger.info(f"Completed validation on {dataset_name}: RMSE={metrics.get('rmse', 'N/A'):.4f}")
        
        return results
    
    def _create_simple_model(self, num_users: int, num_items: int, 
                           embedding_dim: int = 64) -> nn.Module:
        """Create a simple neural collaborative filtering model for testing"""
        
        class SimpleNCF(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(embedding_dim * 2, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1)
                )
                
            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)
                concat_emb = torch.cat([user_emb, item_emb], dim=1)
                output = self.mlp(concat_emb)
                return output.squeeze()
        
        return SimpleNCF(num_users, num_items, embedding_dim)
    
    def _create_dataloader(self, df: pd.DataFrame, batch_size: int = 512):
        """Create PyTorch DataLoader from DataFrame"""
        
        class RatingDataset(torch.utils.data.Dataset):
            def __init__(self, df):
                self.users = torch.LongTensor(df['user_encoded'].values)
                self.items = torch.LongTensor(df['item_encoded'].values)
                self.ratings = torch.FloatTensor(df['rating'].values)
                
            def __len__(self):
                return len(self.users)
                
            def __getitem__(self, idx):
                return self.users[idx], self.items[idx], self.ratings[idx]
        
        dataset = RatingDataset(df)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _train_model(self, model: nn.Module, dataloader, epochs: int = 5):
        """Train the model for a few epochs"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (users, items, ratings) in enumerate(dataloader):
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                
                optimizer.zero_grad()
                predictions = model(users, items)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx >= 100:  # Limit batches for quick training
                    break
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def _evaluate_model(self, model: nn.Module, dataloader) -> Dict[str, float]:
        """Evaluate model performance"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_idx, (users, items, ratings) in enumerate(dataloader):
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                
                pred = model(users, items)
                
                predictions.extend(pred.cpu().numpy())
                actuals.extend(ratings.cpu().numpy())
                
                if batch_idx >= 50:  # Limit for quick evaluation
                    break
        
        if len(predictions) == 0:
            return {'rmse': float('inf'), 'mae': float('inf')}
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'num_predictions': len(predictions)
        }
    
    def run_multi_dataset_validation(self) -> Dict[str, Any]:
        """Run validation across multiple datasets"""
        
        logger.info("Starting multi-dataset validation")
        
        # Dataset loading functions
        dataset_loaders = {
            'movielens_100k': self.downloader.load_movielens_100k,
            'movielens_1m': self.downloader.load_movielens_1m,
            'amazon_books': self.downloader.load_amazon_books,
            'yelp': self.downloader.load_yelp_dataset
        }
        
        all_results = {}
        
        for dataset_name, loader_func in dataset_loaders.items():
            try:
                logger.info(f"Loading dataset: {dataset_name}")
                df = loader_func()
                
                if not df.empty:
                    results = self.validate_on_dataset(dataset_name, df)
                    if results:
                        all_results[dataset_name] = results
                        
                        # Save individual results
                        result_file = self.results_dir / f"{dataset_name}_validation_{self.timestamp}.json"
                        with open(result_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                        logger.info(f"Saved results for {dataset_name} to {result_file}")
                else:
                    logger.warning(f"Dataset {dataset_name} is empty or could not be loaded")
                    
            except Exception as e:
                logger.error(f"Error validating dataset {dataset_name}: {e}")
                continue
        
        # Generate comparative analysis
        if all_results:
            comparative_analysis = self._generate_comparative_analysis(all_results)
            all_results['comparative_analysis'] = comparative_analysis
        
        # Save comprehensive results
        comprehensive_file = self.results_dir / f"multi_dataset_validation_{self.timestamp}.json"
        with open(comprehensive_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Multi-dataset validation completed. Results saved to {comprehensive_file}")
        
        return all_results
    
    def _generate_comparative_analysis(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis across datasets"""
        
        logger.info("Generating comparative analysis")
        
        # Extract key metrics
        dataset_metrics = {}
        layer_patterns = {}
        
        for dataset_name, results in all_results.items():
            if 'performance_metrics' in results and 'layer_profile' in results:
                dataset_metrics[dataset_name] = {
                    'rmse': results['performance_metrics'].get('rmse', float('inf')),
                    'mae': results['performance_metrics'].get('mae', float('inf')),
                    'sparsity': results['dataset_stats']['sparsity'],
                    'num_interactions': results['dataset_stats']['filtered_interactions'],
                    'domain': results['task_config']['domain']
                }
                
                layer_patterns[dataset_name] = {
                    'distribution': results['layer_profile']['importance_distribution'],
                    'num_critical_layers': len(results['layer_profile']['critical_layers']),
                    'confidence': results['layer_profile']['confidence_score']
                }
        
        # Analyze patterns
        analysis = {
            'dataset_count': len(dataset_metrics),
            'performance_summary': {
                'avg_rmse': np.mean([m['rmse'] for m in dataset_metrics.values() if m['rmse'] != float('inf')]),
                'avg_mae': np.mean([m['mae'] for m in dataset_metrics.values() if m['mae'] != float('inf')]),
                'best_dataset': min(dataset_metrics.items(), key=lambda x: x[1]['rmse'])[0] if dataset_metrics else None
            },
            'layer_pattern_analysis': {
                'distribution_types': {dist: sum(1 for p in layer_patterns.values() if p['distribution'] == dist) 
                                     for dist in set(p['distribution'] for p in layer_patterns.values())},
                'avg_critical_layers': np.mean([p['num_critical_layers'] for p in layer_patterns.values()]),
                'avg_confidence': np.mean([p['confidence'] for p in layer_patterns.values()])
            },
            'cross_domain_insights': self._analyze_cross_domain_patterns(dataset_metrics, layer_patterns)
        }
        
        return analysis
    
    def _analyze_cross_domain_patterns(self, dataset_metrics: Dict, layer_patterns: Dict) -> Dict[str, Any]:
        """Analyze patterns across different domains"""
        
        # Group by domain
        domain_groups = {}
        for dataset_name, metrics in dataset_metrics.items():
            domain = metrics['domain']
            if domain not in domain_groups:
                domain_groups[domain] = {'datasets': [], 'metrics': [], 'patterns': []}
            
            domain_groups[domain]['datasets'].append(dataset_name)
            domain_groups[domain]['metrics'].append(metrics)
            if dataset_name in layer_patterns:
                domain_groups[domain]['patterns'].append(layer_patterns[dataset_name])
        
        # Analyze domain-specific patterns
        domain_analysis = {}
        for domain, data in domain_groups.items():
            if data['metrics']:
                domain_analysis[domain] = {
                    'num_datasets': len(data['datasets']),
                    'avg_performance': {
                        'rmse': np.mean([m['rmse'] for m in data['metrics'] if m['rmse'] != float('inf')]),
                        'mae': np.mean([m['mae'] for m in data['metrics'] if m['mae'] != float('inf')])
                    },
                    'common_patterns': {
                        'distribution_types': [p['distribution'] for p in data['patterns']],
                        'avg_critical_layers': np.mean([p['num_critical_layers'] for p in data['patterns']]) if data['patterns'] else 0
                    }
                }
        
        return domain_analysis
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report"""
        
        report_lines = [
            "# Multi-Dataset Validation Report for Layerwise Adapter",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Validation ID**: {self.timestamp}",
            "",
            "## Executive Summary",
            ""
        ]
        
        if 'comparative_analysis' in results:
            analysis = results['comparative_analysis']
            report_lines.extend([
                f"- **Datasets Validated**: {analysis['dataset_count']}",
                f"- **Average RMSE**: {analysis['performance_summary']['avg_rmse']:.4f}",
                f"- **Average MAE**: {analysis['performance_summary']['avg_mae']:.4f}",
                f"- **Best Performing Dataset**: {analysis['performance_summary']['best_dataset']}",
                ""
            ])
        
        # Dataset-specific results
        report_lines.extend([
            "## Dataset-Specific Results",
            ""
        ])
        
        for dataset_name, result in results.items():
            if dataset_name == 'comparative_analysis':
                continue
                
            if 'performance_metrics' in result:
                report_lines.extend([
                    f"### {dataset_name}",
                    f"- **Interactions**: {result['dataset_stats']['filtered_interactions']:,}",
                    f"- **Users**: {result['dataset_stats']['num_users']:,}",
                    f"- **Items**: {result['dataset_stats']['num_items']:,}",
                    f"- **Sparsity**: {result['dataset_stats']['sparsity']:.4f}",
                    f"- **RMSE**: {result['performance_metrics']['rmse']:.4f}",
                    f"- **MAE**: {result['performance_metrics']['mae']:.4f}",
                    f"- **Layer Distribution**: {result['layer_profile']['importance_distribution']}",
                    f"- **Critical Layers**: {len(result['layer_profile']['critical_layers'])}",
                    f"- **Confidence**: {result['layer_profile']['confidence_score']:.3f}",
                    ""
                ])
        
        # Cross-domain analysis
        if 'comparative_analysis' in results and 'cross_domain_insights' in results['comparative_analysis']:
            report_lines.extend([
                "## Cross-Domain Analysis",
                ""
            ])
            
            domain_insights = results['comparative_analysis']['cross_domain_insights']
            for domain, analysis in domain_insights.items():
                report_lines.extend([
                    f"### {domain.title()} Domain",
                    f"- **Datasets**: {analysis['num_datasets']}",
                    f"- **Average RMSE**: {analysis['avg_performance']['rmse']:.4f}",
                    f"- **Average MAE**: {analysis['avg_performance']['mae']:.4f}",
                    f"- **Average Critical Layers**: {analysis['common_patterns']['avg_critical_layers']:.1f}",
                    ""
                ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.results_dir / f"validation_report_{self.timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Validation report saved to {report_file}")
        
        return report_content
