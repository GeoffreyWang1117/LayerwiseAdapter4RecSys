"""
Amazon数据处理器

用于加载和预处理Amazon Reviews数据集，支持多种产品类别
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')

@dataclass
class DataConfig:
    """数据处理配置"""
    min_user_interactions: int = 5  # 最少用户交互次数
    min_item_interactions: int = 5  # 最少物品交互次数
    test_ratio: float = 0.2  # 测试集比例
    val_ratio: float = 0.1   # 验证集比例
    random_seed: int = 42    # 随机种子
    max_users: Optional[int] = None  # 最大用户数限制
    max_items: Optional[int] = None  # 最大物品数限制

class AmazonDataProcessor:
    """Amazon数据处理器"""
    
    def __init__(self, data_path: str = "dataset/amazon", config: Optional[DataConfig] = None):
        """
        初始化数据处理器
        
        Args:
            data_path: 数据目录路径
            config: 数据处理配置
        """
        self.data_path = Path(data_path)
        self.config = config or DataConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 类别映射
        self.available_categories = [
            'All_Beauty', 'Electronics', 'Books', 'Movies_and_TV', 
            'Home_and_Kitchen', 'Arts_Crafts_and_Sewing', 'Automotive',
            'Office_Products', 'Sports_and_Outdoors', 'Toys_and_Games'
        ]
        
        # 编码器
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
    def get_available_categories(self) -> List[str]:
        """获取可用的产品类别"""
        available = []
        for category in self.available_categories:
            reviews_file = self.data_path / f"{category}_reviews.parquet"
            if reviews_file.exists():
                available.append(category)
        return available
    
    def load_category_data(self, category: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载指定类别的数据
        
        Args:
            category: 产品类别名称
            
        Returns:
            reviews_df: 评论数据
            meta_df: 元数据
        """
        if category not in self.available_categories:
            raise ValueError(f"类别 {category} 不支持。可用类别: {self.available_categories}")
        
        reviews_file = self.data_path / f"{category}_reviews.parquet"
        meta_file = self.data_path / f"{category}_meta.parquet"
        
        if not reviews_file.exists():
            raise FileNotFoundError(f"评论文件不存在: {reviews_file}")
        
        self.logger.info(f"加载类别 {category} 的数据...")
        
        # 加载评论数据
        reviews_df = pd.read_parquet(reviews_file)
        self.logger.info(f"加载了 {len(reviews_df):,} 条评论记录")
        
        # 加载元数据
        meta_df = None
        if meta_file.exists():
            meta_df = pd.read_parquet(meta_file)
            self.logger.info(f"加载了 {len(meta_df):,} 条元数据记录")
        
        return reviews_df, meta_df
    
    def normalize_reviews_data(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化评论数据格式
        
        Args:
            reviews_df: 原始评论数据
            
        Returns:
            标准化后的评论数据
        """
        # 统一列名
        column_mapping = {
            'parent_asin': 'item_id',
            'asin': 'item_id',
            'user_id': 'user_id',
            'rating': 'rating',
            'timestamp': 'timestamp',
            'text': 'review_text',
            'title': 'review_title',
            'helpful_vote': 'helpful_vote',
            'verified_purchase': 'verified_purchase'
        }
        
        # 找到实际存在的列
        available_columns = {}
        for old_col, new_col in column_mapping.items():
            if old_col in reviews_df.columns:
                available_columns[old_col] = new_col
        
        # 重命名列
        normalized_df = reviews_df.rename(columns=available_columns)
        
        # 确保必要列存在
        required_columns = ['user_id', 'item_id', 'rating']
        missing_columns = [col for col in required_columns if col not in normalized_df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")
        
        # 数据清理
        normalized_df = normalized_df.dropna(subset=['user_id', 'item_id', 'rating'])
        
        # 过滤异常评分
        normalized_df = normalized_df[
            (normalized_df['rating'] >= 1) & (normalized_df['rating'] <= 5)
        ]
        
        self.logger.info(f"标准化后保留 {len(normalized_df):,} 条记录")
        return normalized_df
    
    def filter_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤低频用户和物品
        
        Args:
            df: 标准化的评论数据
            
        Returns:
            过滤后的数据
        """
        original_size = len(df)
        
        # 过滤低频用户
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.config.min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        # 过滤低频物品
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= self.config.min_item_interactions].index
        df = df[df['item_id'].isin(valid_items)]
        
        # 限制用户和物品数量
        if self.config.max_users is not None:
            top_users = df['user_id'].value_counts().head(self.config.max_users).index
            df = df[df['user_id'].isin(top_users)]
        
        if self.config.max_items is not None:
            top_items = df['item_id'].value_counts().head(self.config.max_items).index
            df = df[df['item_id'].isin(top_items)]
        
        filtered_size = len(df)
        self.logger.info(
            f"过滤后保留 {filtered_size:,} 条记录 "
            f"({filtered_size/original_size*100:.1f}%)"
        )
        
        return df
    
    def encode_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        编码用户和物品ID
        
        Args:
            df: 过滤后的数据
            
        Returns:
            编码后的数据
        """
        df = df.copy()
        
        # 编码用户ID
        df['user_idx'] = self.user_encoder.fit_transform(df['user_id'])
        
        # 编码物品ID
        df['item_idx'] = self.item_encoder.fit_transform(df['item_id'])
        
        self.logger.info(
            f"编码完成: {len(self.user_encoder.classes_):,} 个用户, "
            f"{len(self.item_encoder.classes_):,} 个物品"
        )
        
        return df
    
    def create_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """
        创建用户-物品交互矩阵
        
        Args:
            df: 编码后的数据
            
        Returns:
            稀疏交互矩阵
        """
        n_users = df['user_idx'].max() + 1
        n_items = df['item_idx'].max() + 1
        
        # 创建稀疏矩阵
        interaction_matrix = csr_matrix(
            (df['rating'].values, (df['user_idx'].values, df['item_idx'].values)),
            shape=(n_users, n_items)
        )
        
        density = interaction_matrix.nnz / (n_users * n_items) * 100
        self.logger.info(
            f"交互矩阵: {n_users:,} x {n_items:,}, "
            f"密度: {density:.4f}%, 非零元素: {interaction_matrix.nnz:,}"
        )
        
        return interaction_matrix
    
    def train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        划分训练、验证、测试集
        
        Args:
            df: 完整数据集
            
        Returns:
            train_df, val_df, test_df
        """
        np.random.seed(self.config.random_seed)
        
        # 为每个用户随机分配数据
        def split_user_data(user_data):
            n = len(user_data)
            n_test = max(1, int(n * self.config.test_ratio))
            n_val = max(1, int(n * self.config.val_ratio))
            n_train = n - n_test - n_val
            
            if n_train <= 0:
                n_train = 1
                n_val = min(n_val, n - n_train - 1)
                n_test = n - n_train - n_val
            
            indices = np.random.permutation(n)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]
            
            return user_data.iloc[train_idx], user_data.iloc[val_idx], user_data.iloc[test_idx]
        
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for user_id in df['user_idx'].unique():
            user_data = df[df['user_idx'] == user_id].sort_values('timestamp') if 'timestamp' in df.columns else df[df['user_idx'] == user_id]
            train_data, val_data, test_data = split_user_data(user_data)
            
            train_dfs.append(train_data)
            val_dfs.append(val_data)
            test_dfs.append(test_data)
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        self.logger.info(
            f"数据划分完成: 训练集 {len(train_df):,}, "
            f"验证集 {len(val_df):,}, 测试集 {len(test_df):,}"
        )
        
        return train_df, val_df, test_df
    
    def process_category(self, category: str) -> Dict:
        """
        处理指定类别的完整数据流程
        
        Args:
            category: 产品类别
            
        Returns:
            处理结果字典
        """
        self.logger.info(f"开始处理类别: {category}")
        
        # 1. 加载数据
        reviews_df, meta_df = self.load_category_data(category)
        
        # 2. 标准化数据
        normalized_df = self.normalize_reviews_data(reviews_df)
        
        # 3. 过滤交互
        filtered_df = self.filter_interactions(normalized_df)
        
        # 4. 编码ID
        encoded_df = self.encode_ids(filtered_df)
        
        # 5. 创建交互矩阵
        interaction_matrix = self.create_interaction_matrix(encoded_df)
        
        # 6. 划分数据集
        train_df, val_df, test_df = self.train_test_split(encoded_df)
        
        # 7. 计算统计信息
        stats = {
            'category': category,
            'total_interactions': len(encoded_df),
            'n_users': len(self.user_encoder.classes_),
            'n_items': len(self.item_encoder.classes_),
            'density': interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]) * 100,
            'avg_rating': encoded_df['rating'].mean(),
            'rating_std': encoded_df['rating'].std(),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
        }
        
        result = {
            'stats': stats,
            'data': {
                'full': encoded_df,
                'train': train_df,
                'val': val_df,
                'test': test_df,
            },
            'matrix': interaction_matrix,
            'encoders': {
                'user': self.user_encoder,
                'item': self.item_encoder,
            },
            'meta': meta_df,
        }
        
        self.logger.info(f"类别 {category} 处理完成")
        return result
    
    def get_data_summary(self) -> pd.DataFrame:
        """获取所有可用类别的数据摘要"""
        categories = self.get_available_categories()
        summary_data = []
        
        for category in categories:
            try:
                reviews_df, _ = self.load_category_data(category)
                normalized_df = self.normalize_reviews_data(reviews_df)
                
                summary_data.append({
                    'category': category,
                    'total_reviews': len(reviews_df),
                    'after_cleaning': len(normalized_df),
                    'n_users': normalized_df['user_id'].nunique(),
                    'n_items': normalized_df['item_id'].nunique(),
                    'avg_rating': normalized_df['rating'].mean(),
                    'density': len(normalized_df) / (normalized_df['user_id'].nunique() * normalized_df['item_id'].nunique()) * 100
                })
            except Exception as e:
                self.logger.warning(f"处理类别 {category} 时出错: {e}")
        
        return pd.DataFrame(summary_data)
