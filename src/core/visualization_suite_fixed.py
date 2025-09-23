"""
Core Visualization Module for Layerwise Adapter (Fixed Version)

This module provides comprehensive visualization capabilities for analyzing
Fisher information matrices, layer importance distributions, performance
comparisons, and other key metrics of the Layerwise Adapter approach.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from pathlib import Path
import logging
from datetime import datetime
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# Configure matplotlib and seaborn
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional Plotly imports (will fallback to matplotlib if not available)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Interactive plots will be disabled.")


class LayerwiseVisualizationSuite:
    """Comprehensive visualization suite for Layerwise Adapter analysis"""
    
    def __init__(self, output_dir: Path = Path("results/visualizations")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of visualizations
        self.dirs = {
            'fisher': self.output_dir / 'fisher_analysis',
            'layer_importance': self.output_dir / 'layer_importance', 
            'performance': self.output_dir / 'performance_comparison',
            'architecture': self.output_dir / 'architecture_diagrams',
            'interactive': self.output_dir / 'interactive_plots'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Color schemes
        self.color_schemes = {
            'importance': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
            'performance': ['#3A86FF', '#06FFA5', '#FFBE0B', '#FB5607', '#8338EC'],
            'distribution': ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5', '#3A86FF']
        }
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("Initialized LayerwiseVisualizationSuite with output_dir: %s", output_dir)
    
    def create_fisher_heatmap(self, fisher_info: Dict[str, torch.Tensor], 
                             title: str = "Fisher Information Matrix Heatmap",
                             save_name: Optional[str] = None) -> str:
        """Create heatmap visualization of Fisher information matrix"""
        
        logger.info("Creating Fisher information heatmap")
        
        # Process Fisher information into matrix format
        layer_names = []
        importance_values = []
        
        for layer_name, fisher_tensor in fisher_info.items():
            if fisher_tensor.numel() > 0:
                layer_names.append(layer_name.replace('.', '_'))
                # Compute layer-wise importance as sum of Fisher values
                importance = torch.sum(fisher_tensor).item()
                importance_values.append(importance)
        
        if not layer_names:
            logger.warning("No Fisher information available for visualization")
            return ""
        
        # Normalize importance values
        max_importance = max(importance_values) if importance_values else 1
        normalized_values = [v / max_importance for v in importance_values]
        
        # Create figure
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left plot: Bar chart of layer importance
        bars = ax1.barh(layer_names, normalized_values, 
                       color=cm.viridis(np.array(normalized_values)))
        ax1.set_xlim(0, 1.1)
        ax1.set_xlabel('Normalized Fisher Information')
        ax1.set_ylabel('Layer Names')
        ax1.set_title('Layer-wise Fisher Information Importance')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, value in enumerate(normalized_values):
            ax1.text(value + 0.02, i, f'{value:.3f}', 
                    va='center', fontsize=9)
        
        # Right plot: Heatmap matrix
        # Create a matrix representation for visualization
        matrix_size = int(np.ceil(np.sqrt(len(layer_names))))
        fisher_matrix = np.zeros((matrix_size, matrix_size))
        
        for i, importance in enumerate(normalized_values):
            row = i // matrix_size
            col = i % matrix_size
            if row < matrix_size and col < matrix_size:
                fisher_matrix[row, col] = importance
        
        im = ax2.imshow(fisher_matrix, cmap='viridis', aspect='auto')
        ax2.set_title('Fisher Information Matrix Visualization')
        ax2.set_xlabel('Layer Index (Column)')
        ax2.set_ylabel('Layer Index (Row)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Normalized Fisher Information')
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        if save_name is None:
            save_name = f"fisher_heatmap_{self.timestamp}"
        
        save_path = self.dirs['fisher'] / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Fisher heatmap saved to %s", save_path)
        return str(save_path)
    
    def create_layer_importance_distribution(self, layer_profiles: Dict[str, Any],
                                           title: str = "Layer Importance Distribution Analysis",
                                           save_name: Optional[str] = None) -> str:
        """Create comprehensive layer importance distribution visualization"""
        
        logger.info("Creating layer importance distribution visualization")
        
        # Prepare data
        all_scores = []
        dataset_labels = []
        
        for dataset_name, profile in layer_profiles.items():
            if isinstance(profile, dict) and 'layer_scores' in profile:
                scores = list(profile['layer_scores'].values())
                all_scores.extend(scores)
                dataset_labels.extend([dataset_name] * len(scores))
        
        if not all_scores:
            logger.warning("No layer importance data available for visualization")
            return ""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall distribution histogram
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(all_scores, bins=30, alpha=0.7, color=self.color_schemes['importance'][0],
                edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Layer Importance Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Layer Importance Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_score = np.mean(all_scores)
        median_score = np.median(all_scores)
        ax1.axvline(mean_score, color='red', linestyle='--', 
                   label=f'Mean: {mean_score:.3f}')
        ax1.axvline(median_score, color='orange', linestyle='--',
                   label=f'Median: {median_score:.3f}')
        ax1.legend()
        
        # 2. Distribution by dataset
        ax2 = fig.add_subplot(gs[0, 2])
        if len(set(dataset_labels)) > 1:
            df_scores = pd.DataFrame({
                'scores': all_scores,
                'dataset': dataset_labels
            })
            
            # Box plot by dataset
            sns.boxplot(data=df_scores, y='dataset', x='scores', ax=ax2)
            ax2.set_title('Distribution by Dataset')
            ax2.set_xlabel('Importance Score')
        else:
            ax2.text(0.5, 0.5, 'Single Dataset\nAnalysis', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14)
            ax2.set_title('Dataset Analysis')
        
        # 3. Distribution patterns heatmap
        ax3 = fig.add_subplot(gs[1, :])
        
        # Create pattern analysis
        pattern_data = {}
        for dataset_name, profile in layer_profiles.items():
            if isinstance(profile, dict) and 'layer_scores' in profile:
                scores = np.array(list(profile['layer_scores'].values()))
                
                # Analyze patterns
                if len(scores) > 0:
                    pattern_data[dataset_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'max': np.max(scores),
                        'min': np.min(scores),
                        'top_10_percent': np.percentile(scores, 90),
                        'bottom_10_percent': np.percentile(scores, 10)
                    }
        
        if pattern_data:
            pattern_df = pd.DataFrame(pattern_data).T
            
            # Normalize for heatmap
            pattern_normalized = (pattern_df - pattern_df.min()) / (pattern_df.max() - pattern_df.min())
            
            sns.heatmap(pattern_normalized, annot=True, fmt='.2f', 
                       cmap='viridis', ax=ax3)
            ax3.set_title('Layer Importance Pattern Analysis by Dataset')
            ax3.set_xlabel('Statistical Measures')
            ax3.set_ylabel('Datasets')
        
        # 4. Multi-modal analysis
        ax4 = fig.add_subplot(gs[2, 0])
        
        # Distribution type counts
        dist_types = [profile.get('importance_distribution', 'unknown') 
                     for profile in layer_profiles.values()
                     if isinstance(profile, dict)]
        
        if dist_types:
            dist_counts = pd.Series(dist_types).value_counts()
            colors = self.color_schemes['distribution'][:len(dist_counts)]
            
            ax4.pie(dist_counts.values, labels=dist_counts.index,
                   autopct='%1.1f%%', colors=colors)
            ax4.set_title('Distribution Type Classification')
        
        # 5. Layer ranking analysis
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Find most important layers across datasets
        layer_importance_agg = {}
        for dataset_name, profile in layer_profiles.items():
            if isinstance(profile, dict) and 'layer_scores' in profile:
                for layer_name, score in profile['layer_scores'].items():
                    if layer_name not in layer_importance_agg:
                        layer_importance_agg[layer_name] = []
                    layer_importance_agg[layer_name].append(score)
        
        # Average importance across datasets
        avg_importance = {layer: np.mean(scores) 
                         for layer, scores in layer_importance_agg.items()}
        
        if avg_importance:
            top_layers = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            layer_names = [name.replace('.', '_')[:15] for name, _ in top_layers]  # Truncate long names
            layer_scores = [score for _, score in top_layers]
            
            ax5.barh(layer_names, layer_scores, 
                    color=self.color_schemes['importance'][1])
            ax5.set_xlabel('Average Importance Score')
            ax5.set_title('Top 10 Most Important Layers')
            ax5.grid(True, alpha=0.3)
            
            # Add value labels
            for i, score in enumerate(layer_scores):
                ax5.text(score + 0.01, i, f'{score:.3f}', 
                        va='center', fontsize=8)
        
        # 6. Confidence analysis
        ax6 = fig.add_subplot(gs[2, 2])
        
        confidence_scores = []
        dataset_names = []
        
        for dataset_name, profile in layer_profiles.items():
            if isinstance(profile, dict) and 'confidence_score' in profile:
                confidence_scores.append(profile['confidence_score'])
                dataset_names.append(dataset_name)
        
        if confidence_scores:
            bars = ax6.bar(range(len(dataset_names)), confidence_scores,
                          color=self.color_schemes['performance'][:len(dataset_names)])
            ax6.set_xticks(range(len(dataset_names)))
            ax6.set_xticklabels([name[:10] for name in dataset_names], rotation=45)
            ax6.set_ylabel('Confidence Score')
            ax6.set_title('Layer Selection Confidence')
            ax6.set_ylim(0, 1)
            ax6.grid(True, alpha=0.3)
            
            # Add value labels
            for i, score in enumerate(confidence_scores):
                ax6.text(i, score + 0.02, f'{score:.2f}', 
                        ha='center', fontsize=9)
        
        plt.suptitle(title, fontsize=18, y=0.98)
        
        # Save plot
        if save_name is None:
            save_name = f"layer_importance_distribution_{self.timestamp}"
        
        save_path = self.dirs['layer_importance'] / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Layer importance distribution saved to %s", save_path)
        return str(save_path)
    
    def create_performance_comparison(self, results: Dict[str, Any],
                                    title: str = "Performance Comparison Across Datasets",
                                    save_name: Optional[str] = None) -> str:
        """Create comprehensive performance comparison visualization"""
        
        logger.info("Creating performance comparison visualization")
        
        # Extract performance metrics
        dataset_names = []
        rmse_scores = []
        mae_scores = []
        sparsity_levels = []
        interaction_counts = []
        
        for dataset_name, result in results.items():
            if isinstance(result, dict) and 'performance_metrics' in result:
                dataset_names.append(dataset_name)
                
                metrics = result['performance_metrics']
                rmse_scores.append(metrics.get('rmse', np.nan))
                mae_scores.append(metrics.get('mae', np.nan))
                
                stats = result.get('dataset_stats', {})
                sparsity_levels.append(stats.get('sparsity', np.nan))
                interaction_counts.append(stats.get('filtered_interactions', 0))
        
        if not dataset_names:
            logger.warning("No performance data available for visualization")
            return ""
        
        # Create figure
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. RMSE and MAE comparison
        x_pos = np.arange(len(dataset_names))
        width = 0.35
        
        # Filter out NaN values for plotting
        valid_rmse = [rmse if not np.isnan(rmse) else 0 for rmse in rmse_scores]
        valid_mae = [mae if not np.isnan(mae) else 0 for mae in mae_scores]
        
        bars1 = ax1.bar(x_pos - width/2, valid_rmse, width, 
                       label='RMSE', color=self.color_schemes['performance'][0],
                       alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, valid_mae, width,
                       label='MAE', color=self.color_schemes['performance'][1],
                       alpha=0.8)
        
        ax1.set_xlabel('Datasets')
        ax1.set_ylabel('Error Score')
        ax1.set_title('RMSE vs MAE Performance Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name[:12] for name in dataset_names], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, valid_rmse)):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        for i, (bar, value) in enumerate(zip(bars2, valid_mae)):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Performance vs Sparsity
        valid_indices = [i for i, (rmse, sparsity) in enumerate(zip(rmse_scores, sparsity_levels))
                        if not (np.isnan(rmse) or np.isnan(sparsity))]
        
        if valid_indices:
            valid_rmse_scatter = [rmse_scores[i] for i in valid_indices]
            valid_sparsity_scatter = [sparsity_levels[i] for i in valid_indices]
            valid_names_scatter = [dataset_names[i] for i in valid_indices]
            
            ax2.scatter(valid_sparsity_scatter, valid_rmse_scatter,
                       c=range(len(valid_indices)), 
                       cmap='viridis', s=100, alpha=0.7)
            
            ax2.set_xlabel('Dataset Sparsity')
            ax2.set_ylabel('RMSE')
            ax2.set_title('Performance vs Dataset Sparsity')
            ax2.grid(True, alpha=0.3)
            
            # Add dataset labels
            for i, name in enumerate(valid_names_scatter):
                ax2.annotate(name[:8], 
                           (valid_sparsity_scatter[i], valid_rmse_scatter[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        # 3. Dataset characteristics
        valid_counts = [count for count in interaction_counts if count > 0]
        valid_names_counts = [dataset_names[i] for i, count in enumerate(interaction_counts) if count > 0]
        
        if valid_counts:
            ax3.bar(valid_names_counts, valid_counts,
                   color=self.color_schemes['performance'][2],
                   alpha=0.7)
            ax3.set_xlabel('Datasets')
            ax3.set_ylabel('Number of Interactions (log scale)')
            ax3.set_title('Dataset Size Comparison')
            ax3.set_yscale('log')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(ax3.patches, valid_counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                        f'{value:,}', ha='center', va='bottom', fontsize=8,
                        rotation=90)
        
        # 4. Performance ranking
        performance_ranking = []
        for name, rmse, mae in zip(dataset_names, rmse_scores, mae_scores):
            if not (np.isnan(rmse) or np.isnan(mae)):
                # Combined score (lower is better)
                combined_score = (rmse + mae) / 2
                performance_ranking.append((name, combined_score))
        
        if performance_ranking:
            performance_ranking.sort(key=lambda x: x[1])
            
            rank_names = [name for name, _ in performance_ranking]
            rank_scores = [score for _, score in performance_ranking]
            
            # Use a simple color gradient
            colors = cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rank_scores)))
            
            bars4 = ax4.barh(rank_names, rank_scores, color=colors)
            ax4.set_xlabel('Combined Error Score (RMSE + MAE)/2')
            ax4.set_ylabel('Datasets (Best to Worst)')
            ax4.set_title('Overall Performance Ranking')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, score in zip(bars4, rank_scores):
                ax4.text(score + max(rank_scores) * 0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', va='center', fontsize=9)
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        if save_name is None:
            save_name = f"performance_comparison_{self.timestamp}"
        
        save_path = self.dirs['performance'] / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Performance comparison saved to %s", save_path)
        return str(save_path)
    
    def create_architecture_diagram(self, model_info: Dict[str, Any],
                                  title: str = "Layerwise Adapter Architecture",
                                  save_name: Optional[str] = None) -> str:
        """Create architecture diagram showing layer importance and connections"""
        
        logger.info("Creating architecture diagram")
        
        # Create figure
        _, ax = plt.subplots(figsize=(16, 12))
        
        # Extract layer information
        layers = model_info.get('layers', [])
        layer_scores = model_info.get('layer_scores', {})
        critical_layers = model_info.get('critical_layers', [])
        
        if not layers:
            # Create a generic architecture if no specific info provided
            layers = ['Input', 'Embedding', 'Layer1', 'Layer2', 'Layer3', 'Output']
            layer_scores = {layer: np.random.random() for layer in layers}
        
        # Calculate positions
        layer_positions = {}
        layer_heights = {}
        
        for i, layer in enumerate(layers):
            x = i * 2
            y = 0
            layer_positions[layer] = (x, y)
            
            # Height based on importance
            importance = layer_scores.get(layer, 0.5)
            layer_heights[layer] = 0.5 + importance * 1.5
        
        # Draw layers
        for layer in layers:
            x, y = layer_positions[layer]
            height = layer_heights[layer]
            
            # Color based on importance
            importance = layer_scores.get(layer, 0.5)
            color = cm.viridis(importance)
            
            # Special highlighting for critical layers
            edge_color = 'red' if layer in critical_layers else 'black'
            edge_width = 3 if layer in critical_layers else 1
            
            # Draw rectangle for layer
            rect = Rectangle((x-0.4, y-height/2), 0.8, height,
                           facecolor=color, edgecolor=edge_color,
                           linewidth=edge_width, alpha=0.8)
            ax.add_patch(rect)
            
            # Add layer label
            ax.text(x, y-height/2-0.3, layer.replace('.', '\n'), 
                   ha='center', va='top', fontsize=10, fontweight='bold')
            
            # Add importance score
            ax.text(x, y+height/2+0.1, f'{importance:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        # Draw connections between layers
        for i in range(len(layers)-1):
            layer1 = layers[i]
            layer2 = layers[i+1]
            
            x1, y1 = layer_positions[layer1]
            x2, y2 = layer_positions[layer2]
            
            # Connection strength based on average importance
            avg_importance = (layer_scores.get(layer1, 0.5) + layer_scores.get(layer2, 0.5)) / 2
            line_width = 1 + avg_importance * 3
            
            ax.arrow(x1+0.4, y1, x2-x1-0.8, y2-y1,
                    head_width=0.1, head_length=0.1,
                    fc='gray', ec='gray', alpha=0.6,
                    linewidth=line_width)
        
        # Set axis properties
        ax.set_xlim(-1, (len(layers)-1)*2 + 1)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=cm.viridis(0.8), 
                         label='High Importance'),
            plt.Rectangle((0, 0), 1, 1, facecolor=cm.viridis(0.4),
                         label='Medium Importance'),
            plt.Rectangle((0, 0), 1, 1, facecolor=cm.viridis(0.1),
                         label='Low Importance'),
            plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='red',
                         linewidth=3, label='Critical Layer')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1, 1))
        
        plt.title(title, fontsize=18, pad=20)
        plt.tight_layout()
        
        # Save plot
        if save_name is None:
            save_name = f"architecture_diagram_{self.timestamp}"
        
        save_path = self.dirs['architecture'] / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Architecture diagram saved to %s", save_path)
        return str(save_path)
    
    def _create_matplotlib_dashboard(self, all_data: Dict[str, Any],
                                   title: str = "Layerwise Adapter Dashboard",
                                   save_name: Optional[str] = None) -> str:
        """Create matplotlib-based dashboard as fallback"""
        
        logger.info("Creating matplotlib dashboard")
        
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Extract data
        datasets = []
        rmse_values = []
        mae_values = []
        
        for dataset_name, data in all_data.items():
            if isinstance(data, dict) and 'performance_metrics' in data:
                datasets.append(dataset_name)
                
                metrics = data['performance_metrics']
                rmse_values.append(metrics.get('rmse', 0))
                mae_values.append(metrics.get('mae', 0))
        
        # 1. Performance metrics
        if datasets:
            x_pos = np.arange(len(datasets))
            width = 0.35
            
            ax1.bar(x_pos - width/2, rmse_values, width, label='RMSE', alpha=0.8)
            ax1.bar(x_pos + width/2, mae_values, width, label='MAE', alpha=0.8)
            ax1.set_xlabel('Datasets')
            ax1.set_ylabel('Error Score')
            ax1.set_title('Performance Metrics')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(datasets, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Layer importance (if available)
        layer_data = []
        for dataset_name, data in all_data.items():
            if isinstance(data, dict) and 'layer_profile' in data:
                profile = data['layer_profile']
                if isinstance(profile, dict) and 'layer_scores' in profile:
                    scores = list(profile['layer_scores'].values())
                    layer_data.extend(scores)
        
        if layer_data:
            ax2.hist(layer_data, bins=20, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Layer Importance Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Layer Importance Distribution')
            ax2.grid(True, alpha=0.3)
        
        # 3. Dataset characteristics
        interaction_counts = []
        dataset_names_counts = []
        
        for dataset_name, data in all_data.items():
            if isinstance(data, dict) and 'dataset_stats' in data:
                stats = data['dataset_stats']
                count = stats.get('filtered_interactions', 0)
                if count > 0:
                    interaction_counts.append(count)
                    dataset_names_counts.append(dataset_name)
        
        if interaction_counts:
            ax3.bar(dataset_names_counts, interaction_counts, alpha=0.7)
            ax3.set_xlabel('Datasets')
            ax3.set_ylabel('Number of Interactions')
            ax3.set_title('Dataset Sizes')
            ax3.set_yscale('log')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 4. Distribution types
        dist_types = []
        for dataset_name, data in all_data.items():
            if isinstance(data, dict) and 'layer_profile' in data:
                profile = data['layer_profile']
                if isinstance(profile, dict):
                    dist_type = profile.get('importance_distribution', 'unknown')
                    dist_types.append(dist_type)
        
        if dist_types:
            dist_counts = pd.Series(dist_types).value_counts()
            ax4.pie(dist_counts.values, labels=dist_counts.index, autopct='%1.1f%%')
            ax4.set_title('Distribution Types')
        
        plt.suptitle(title, fontsize=18, y=0.98)
        plt.tight_layout()
        
        # Save plot
        if save_name is None:
            save_name = f"matplotlib_dashboard_{self.timestamp}"
        
        save_path = self.dirs['interactive'] / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Matplotlib dashboard saved to %s", save_path)
        return str(save_path)
    
    def create_interactive_dashboard(self, all_data: Dict[str, Any],
                                   title: str = "Layerwise Adapter Interactive Dashboard",
                                   save_name: Optional[str] = None) -> str:
        """Create interactive dashboard (Plotly if available, matplotlib fallback)"""
        
        if not PLOTLY_AVAILABLE:
            return self._create_matplotlib_dashboard(all_data, title, save_name)
        
        # If Plotly is available, create interactive version
        logger.info("Creating interactive Plotly dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'Layer Importance',
                          'Dataset Characteristics', 'Distribution Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Extract and plot data (simplified version)
        datasets = []
        rmse_values = []
        
        for dataset_name, data in all_data.items():
            if isinstance(data, dict) and 'performance_metrics' in data:
                datasets.append(dataset_name)
                rmse_values.append(data['performance_metrics'].get('rmse', 0))
        
        if datasets:
            fig.add_trace(
                go.Bar(x=datasets, y=rmse_values, name='RMSE'),
                row=1, col=1
            )
        
        # Update layout and save
        fig.update_layout(title_text=title, showlegend=True, height=800)
        
        if save_name is None:
            save_name = f"interactive_dashboard_{self.timestamp}"
        
        save_path = self.dirs['interactive'] / f"{save_name}.html"
        pyo.plot(fig, filename=str(save_path), auto_open=False)
        
        logger.info("Interactive dashboard saved to %s", save_path)
        return str(save_path)
    
    def generate_visualization_report(self, generated_plots: List[str]) -> str:
        """Generate a report summarizing all created visualizations"""
        
        report_lines = [
            f"# Layerwise Adapter Visualization Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Report ID**: {self.timestamp}",
            "",
            "## Generated Visualizations",
            ""
        ]
        
        for plot_path in generated_plots:
            plot_name = Path(plot_path).stem
            plot_type = Path(plot_path).parent.name
            
            report_lines.extend([
                f"### {plot_name.replace('_', ' ').title()}",
                f"- **Type**: {plot_type.replace('_', ' ').title()}",
                f"- **File**: `{plot_path}`",
                f"- **Description**: {self._get_plot_description(plot_type)}",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_dir / f"visualization_report_{self.timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info("Visualization report saved to %s", report_file)
        return str(report_file)
    
    def _get_plot_description(self, plot_type: str) -> str:
        """Get description for different plot types"""
        descriptions = {
            'fisher_analysis': 'Fisher information matrix visualization showing layer-wise importance',
            'layer_importance': 'Comprehensive analysis of layer importance distributions',
            'performance_comparison': 'Performance metrics comparison across datasets',
            'architecture_diagrams': 'Visual representation of model architecture with importance',
            'interactive_plots': 'Interactive dashboard for exploring results'
        }
        
        return descriptions.get(plot_type, 'Visualization analysis')
    
    def create_comprehensive_report(self, all_data: Dict[str, Any]) -> List[str]:
        """Create all visualizations and return list of generated files"""
        
        logger.info("Creating comprehensive visualization report")
        
        generated_plots = []
        
        try:
            # 1. Fisher information heatmap
            if 'fisher_info' in all_data:
                fisher_plot = self.create_fisher_heatmap(all_data['fisher_info'])
                if fisher_plot:
                    generated_plots.append(fisher_plot)
            
            # 2. Layer importance distribution
            if 'layer_profiles' in all_data:
                importance_plot = self.create_layer_importance_distribution(all_data['layer_profiles'])
                if importance_plot:
                    generated_plots.append(importance_plot)
            
            # 3. Performance comparison
            if 'results' in all_data:
                performance_plot = self.create_performance_comparison(all_data['results'])
                if performance_plot:
                    generated_plots.append(performance_plot)
            
            # 4. Architecture diagram
            if 'model_info' in all_data:
                arch_plot = self.create_architecture_diagram(all_data['model_info'])
                if arch_plot:
                    generated_plots.append(arch_plot)
            
            # 5. Interactive dashboard
            dashboard_plot = self.create_interactive_dashboard(all_data)
            if dashboard_plot:
                generated_plots.append(dashboard_plot)
            
            # 6. Generate summary report
            report_file = self.generate_visualization_report(generated_plots)
            generated_plots.append(report_file)
            
        except Exception as e:
            logger.error("Error creating comprehensive report: %s", e)
        
        logger.info("Generated %d visualization files", len(generated_plots))
        return generated_plots
