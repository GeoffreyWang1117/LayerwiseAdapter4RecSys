"""
数据工具类

提供通用的数据处理和特征工程功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

class DataUtils:
    """数据处理工具类"""
    
    @staticmethod
    def calculate_sparsity_metrics(interaction_matrix: csr_matrix) -> Dict[str, float]:
        """
        计算稀疏性相关指标
        
        Args:
            interaction_matrix: 用户-物品交互矩阵
            
        Returns:
            稀疏性指标字典
        """
        n_users, n_items = interaction_matrix.shape
        n_interactions = interaction_matrix.nnz
        
        # 基本指标
        density = n_interactions / (n_users * n_items) * 100
        sparsity = 100 - density
        
        # 用户和物品的交互统计
        user_interactions = np.array(interaction_matrix.sum(axis=1)).flatten()
        item_interactions = np.array(interaction_matrix.sum(axis=0)).flatten()
        
        metrics = {
            'density': density,
            'sparsity': sparsity,
            'total_interactions': n_interactions,
            'avg_user_interactions': user_interactions.mean(),
            'avg_item_interactions': item_interactions.mean(),
            'user_interactions_std': user_interactions.std(),
            'item_interactions_std': item_interactions.std(),
            'gini_user': DataUtils._gini_coefficient(user_interactions),
            'gini_item': DataUtils._gini_coefficient(item_interactions),
        }
        
        return metrics
    
    @staticmethod
    def _gini_coefficient(x: np.ndarray) -> float:
        """计算基尼系数"""
        if len(x) == 0:
            return 0
        
        # 排序
        sorted_x = np.sort(x)
        n = len(x)
        
        # 计算基尼系数
        cumsum = np.cumsum(sorted_x)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return gini
    
    @staticmethod
    def analyze_rating_distribution(ratings: np.ndarray) -> Dict[str, Any]:
        """
        分析评分分布
        
        Args:
            ratings: 评分数组
            
        Returns:
            评分分布统计
        """
        rating_counts = pd.Series(ratings).value_counts().sort_index()
        
        analysis = {
            'mean': ratings.mean(),
            'std': ratings.std(),
            'median': np.median(ratings),
            'mode': rating_counts.index[0],
            'distribution': rating_counts.to_dict(),
            'skewness': pd.Series(ratings).skew(),
            'kurtosis': pd.Series(ratings).kurtosis(),
        }
        
        return analysis
    
    @staticmethod
    def create_cold_start_splits(df: pd.DataFrame, 
                                cold_user_ratio: float = 0.1,
                                cold_item_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        创建冷启动场景的数据划分
        
        Args:
            df: 完整数据集
            cold_user_ratio: 冷启动用户比例
            cold_item_ratio: 冷启动物品比例
            
        Returns:
            不同场景的数据集
        """
        # 随机选择冷启动用户和物品
        all_users = df['user_idx'].unique()
        all_items = df['item_idx'].unique()
        
        n_cold_users = int(len(all_users) * cold_user_ratio)
        n_cold_items = int(len(all_items) * cold_item_ratio)
        
        cold_users = np.random.choice(all_users, n_cold_users, replace=False)
        cold_items = np.random.choice(all_items, n_cold_items, replace=False)
        
        # 创建不同场景的数据集
        splits = {
            'warm_start': df[
                (~df['user_idx'].isin(cold_users)) & 
                (~df['item_idx'].isin(cold_items))
            ],
            'cold_user': df[df['user_idx'].isin(cold_users)],
            'cold_item': df[df['item_idx'].isin(cold_items)],
        }
        
        return splits
    
    @staticmethod
    def create_temporal_splits(df: pd.DataFrame, 
                              split_ratios: List[float] = [0.6, 0.2, 0.2]) -> Dict[str, pd.DataFrame]:
        """
        基于时间的数据划分
        
        Args:
            df: 包含时间戳的数据集
            split_ratios: [训练, 验证, 测试] 比例
            
        Returns:
            时间划分的数据集
        """
        if 'timestamp' not in df.columns:
            raise ValueError("数据集缺少timestamp列")
        
        # 按时间排序
        df_sorted = df.sort_values('timestamp')
        n = len(df_sorted)
        
        # 计算分割点
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])
        
        splits = {
            'train': df_sorted.iloc[:train_end],
            'val': df_sorted.iloc[train_end:val_end],
            'test': df_sorted.iloc[val_end:],
        }
        
        return splits
    
    @staticmethod
    def visualize_data_statistics(stats: Dict[str, Any], save_path: Optional[str] = None):
        """
        可视化数据统计信息
        
        Args:
            stats: 统计信息字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Amazon Dataset Statistics', fontsize=16)
        
        # 1. 类别数据量对比
        if 'categories' in stats:
            categories = list(stats['categories'].keys())
            interactions = [stats['categories'][cat]['total_interactions'] for cat in categories]
            
            axes[0, 0].bar(categories, interactions)
            axes[0, 0].set_title('Interactions by Category')
            axes[0, 0].set_ylabel('Number of Interactions')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 密度分布
        if 'categories' in stats:
            densities = [stats['categories'][cat]['density'] for cat in categories]
            
            axes[0, 1].bar(categories, densities)
            axes[0, 1].set_title('Data Density by Category')
            axes[0, 1].set_ylabel('Density (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 评分分布
        if 'rating_distribution' in stats:
            ratings = list(stats['rating_distribution'].keys())
            counts = list(stats['rating_distribution'].values())
            
            axes[1, 0].bar(ratings, counts)
            axes[1, 0].set_title('Rating Distribution')
            axes[1, 0].set_xlabel('Rating')
            axes[1, 0].set_ylabel('Count')
        
        # 4. 用户-物品比例
        if 'categories' in stats:
            user_counts = [stats['categories'][cat]['n_users'] for cat in categories]
            item_counts = [stats['categories'][cat]['n_items'] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, user_counts, width, label='Users', alpha=0.7)
            axes[1, 1].bar(x + width/2, item_counts, width, label='Items', alpha=0.7)
            axes[1, 1].set_title('Users vs Items by Category')
            axes[1, 1].set_xlabel('Category')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(categories, rotation=45)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
    
    @staticmethod
    def save_processed_data(data: Dict[str, Any], save_dir: str, category: str):
        """
        保存处理后的数据
        
        Args:
            data: 处理后的数据字典
            save_dir: 保存目录
            category: 数据类别
        """
        save_path = Path(save_dir) / category
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存统计信息
        with open(save_path / 'stats.json', 'w') as f:
            json.dump(data['stats'], f, indent=2)
        
        # 保存数据集
        for split_name, split_data in data['data'].items():
            split_data.to_parquet(save_path / f'{split_name}.parquet')
        
        # 保存交互矩阵
        np.savez_compressed(
            save_path / 'interaction_matrix.npz',
            data=data['matrix'].data,
            indices=data['matrix'].indices,
            indptr=data['matrix'].indptr,
            shape=data['matrix'].shape
        )
        
        print(f"数据已保存至: {save_path}")
    
    @staticmethod
    def load_processed_data(load_dir: str, category: str) -> Dict[str, Any]:
        """
        加载已处理的数据
        
        Args:
            load_dir: 数据目录
            category: 数据类别
            
        Returns:
            加载的数据字典
        """
        load_path = Path(load_dir) / category
        
        # 加载统计信息
        with open(load_path / 'stats.json', 'r') as f:
            stats = json.load(f)
        
        # 加载数据集
        data = {}
        for split_file in load_path.glob('*.parquet'):
            split_name = split_file.stem
            if split_name != 'interaction_matrix':
                data[split_name] = pd.read_parquet(split_file)
        
        # 加载交互矩阵
        matrix_data = np.load(load_path / 'interaction_matrix.npz')
        interaction_matrix = csr_matrix(
            (matrix_data['data'], matrix_data['indices'], matrix_data['indptr']),
            shape=matrix_data['shape']
        )
        
        return {
            'stats': stats,
            'data': data,
            'matrix': interaction_matrix,
        }
    
    @staticmethod
    def generate_negative_samples(positive_interactions: pd.DataFrame,
                                 n_users: int, n_items: int,
                                 negative_ratio: int = 4) -> pd.DataFrame:
        """
        生成负样本
        
        Args:
            positive_interactions: 正样本交互数据
            n_users: 用户总数
            n_items: 物品总数
            negative_ratio: 负样本比例
            
        Returns:
            负样本数据框
        """
        # 创建正样本集合
        positive_set = set(
            zip(positive_interactions['user_idx'], positive_interactions['item_idx'])
        )
        
        negative_samples = []
        n_negatives = len(positive_interactions) * negative_ratio
        
        while len(negative_samples) < n_negatives:
            # 随机生成用户-物品对
            user_idx = np.random.randint(0, n_users)
            item_idx = np.random.randint(0, n_items)
            
            # 检查是否为负样本
            if (user_idx, item_idx) not in positive_set:
                negative_samples.append({
                    'user_idx': user_idx,
                    'item_idx': item_idx,
                    'rating': 0,  # 负样本标记为0
                    'is_positive': False
                })
        
        return pd.DataFrame(negative_samples)
