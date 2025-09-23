"""
Amazon推荐系统基线实现

基于真实Amazon数据的协同过滤推荐系统
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix, coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AmazonRecommenderBaseline:
    """Amazon推荐系统基线"""
    
    def __init__(self, category: str = 'All_Beauty', max_users: int = 5000, max_items: int = 2000):
        """
        初始化推荐系统
        
        Args:
            category: Amazon产品类别
            max_users: 最大用户数
            max_items: 最大物品数
        """
        self.category = category
        self.max_users = max_users
        self.max_items = max_items
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # 推荐模型
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        
        # 数据
        self.train_data = None
        self.test_data = None
        self.user_means = None
        
    def load_and_preprocess_data(self, data_path: str = "dataset/amazon") -> pd.DataFrame:
        """加载和预处理数据"""
        self.logger.info(f"加载类别 {self.category} 的数据...")
        
        # 加载数据
        reviews_file = Path(data_path) / f"{self.category}_reviews.parquet"
        if not reviews_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {reviews_file}")
        
        df = pd.read_parquet(reviews_file)
        self.logger.info(f"原始数据: {len(df):,} 条记录")
        
        # 统一列名
        df = df.rename(columns={'parent_asin': 'item_id'})
        
        # 确保必要列存在
        required_cols = ['user_id', 'item_id', 'rating']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"缺少必要列: {required_cols}")
        
        # 数据清理
        df = df.dropna(subset=required_cols)
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
        
        # 过滤低频用户和物品
        min_interactions = 5
        
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        df = df[df['item_id'].isin(valid_items)]
        
        # 限制数据规模
        if self.max_users and len(df['user_id'].unique()) > self.max_users:
            top_users = df['user_id'].value_counts().head(self.max_users).index
            df = df[df['user_id'].isin(top_users)]
        
        if self.max_items and len(df['item_id'].unique()) > self.max_items:
            top_items = df['item_id'].value_counts().head(self.max_items).index
            df = df[df['item_id'].isin(top_items)]
        
        # 编码用户和物品ID
        df['user_idx'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_idx'] = self.item_encoder.fit_transform(df['item_id'])
        
        self.logger.info(
            f"预处理后: {len(df):,} 条记录, "
            f"{df['user_idx'].nunique():,} 用户, "
            f"{df['item_idx'].nunique():,} 物品"
        )
        
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分训练和测试集"""
        self.logger.info("划分训练和测试集...")
        
        # 为每个用户随机划分数据
        train_list, test_list = [], []
        
        for user_idx in df['user_idx'].unique():
            user_data = df[df['user_idx'] == user_idx]
            
            if len(user_data) < 2:
                train_list.append(user_data)
                continue
            
            # 随机划分
            user_train, user_test = train_test_split(
                user_data, test_size=test_size, random_state=42
            )
            
            train_list.append(user_train)
            test_list.append(user_test)
        
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        self.logger.info(f"训练集: {len(train_df):,}, 测试集: {len(test_df):,}")
        
        return train_df, test_df
    
    def build_user_item_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """构建用户-物品交互矩阵"""
        self.logger.info("构建用户-物品交互矩阵...")
        
        n_users = df['user_idx'].max() + 1
        n_items = df['item_idx'].max() + 1
        
        # 创建稀疏矩阵
        matrix = csr_matrix(
            (df['rating'].values, (df['user_idx'].values, df['item_idx'].values)),
            shape=(n_users, n_items)
        )
        
        density = matrix.nnz / (n_users * n_items) * 100
        self.logger.info(f"交互矩阵: {n_users} x {n_items}, 密度: {density:.4f}%")
        
        return matrix
    
    def compute_item_similarity(self, matrix: csr_matrix) -> np.ndarray:
        """计算物品相似度矩阵"""
        self.logger.info("计算物品相似度矩阵...")
        
        # 使用余弦相似度
        item_similarity = cosine_similarity(matrix.T)
        
        # 将对角线设为0（物品与自身的相似度）
        np.fill_diagonal(item_similarity, 0)
        
        return item_similarity
    
    def compute_user_similarity(self, matrix: csr_matrix) -> np.ndarray:
        """计算用户相似度矩阵"""
        self.logger.info("计算用户相似度矩阵...")
        
        # 使用余弦相似度
        user_similarity = cosine_similarity(matrix)
        
        # 将对角线设为0
        np.fill_diagonal(user_similarity, 0)
        
        return user_similarity
    
    def predict_item_based(self, user_idx: int, item_idx: int, k: int = 50) -> float:
        """基于物品的协同过滤预测"""
        if self.item_similarity_matrix is None:
            return self.user_means[user_idx] if user_idx < len(self.user_means) else 3.0
        
        # 找到用户评价过的物品
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return 3.0  # 默认评分
        
        # 计算目标物品与已评价物品的相似度
        if item_idx >= self.item_similarity_matrix.shape[0]:
            return self.user_means[user_idx] if user_idx < len(self.user_means) else 3.0
        
        similarities = self.item_similarity_matrix[item_idx][rated_items]
        
        # 选择top-k相似物品
        top_k_indices = np.argsort(similarities)[-k:]
        top_k_items = rated_items[top_k_indices]
        top_k_similarities = similarities[top_k_indices]
        
        # 计算加权平均评分
        if np.sum(np.abs(top_k_similarities)) == 0:
            return self.user_means[user_idx] if user_idx < len(self.user_means) else 3.0
        
        weighted_ratings = user_ratings[top_k_items] * top_k_similarities
        prediction = np.sum(weighted_ratings) / np.sum(np.abs(top_k_similarities))
        
        return max(1.0, min(5.0, prediction))
    
    def predict_user_based(self, user_idx: int, item_idx: int, k: int = 50) -> float:
        """基于用户的协同过滤预测"""
        if self.user_similarity_matrix is None:
            return 3.0
        
        # 找到评价过该物品的用户
        item_ratings = self.user_item_matrix[:, item_idx].toarray().flatten()
        rated_users = np.where(item_ratings > 0)[0]
        
        if len(rated_users) == 0:
            return 3.0
        
        # 计算用户相似度
        if user_idx >= self.user_similarity_matrix.shape[0]:
            return 3.0
        
        similarities = self.user_similarity_matrix[user_idx][rated_users]
        
        # 选择top-k相似用户
        top_k_indices = np.argsort(similarities)[-k:]
        top_k_users = rated_users[top_k_indices]
        top_k_similarities = similarities[top_k_indices]
        
        # 计算加权平均评分
        if np.sum(np.abs(top_k_similarities)) == 0:
            return 3.0
        
        weighted_ratings = item_ratings[top_k_users] * top_k_similarities
        prediction = np.sum(weighted_ratings) / np.sum(np.abs(top_k_similarities))
        
        return max(1.0, min(5.0, prediction))
    
    def fit(self, df: pd.DataFrame):
        """训练推荐模型"""
        self.logger.info("开始训练推荐模型...")
        
        # 划分数据
        self.train_data, self.test_data = self.split_data(df)
        
        # 构建交互矩阵
        self.user_item_matrix = self.build_user_item_matrix(self.train_data)
        
        # 计算用户平均评分
        self.user_means = np.array([
            self.user_item_matrix[i].mean() if self.user_item_matrix[i].nnz > 0 else 3.0
            for i in range(self.user_item_matrix.shape[0])
        ])
        
        # 计算相似度矩阵
        self.item_similarity_matrix = self.compute_item_similarity(self.user_item_matrix)
        self.user_similarity_matrix = self.compute_user_similarity(self.user_item_matrix)
        
        self.logger.info("模型训练完成")
    
    def evaluate(self, method: str = 'item_based') -> Dict[str, float]:
        """评估推荐模型"""
        self.logger.info(f"评估 {method} 推荐模型...")
        
        predictions = []
        ground_truth = []
        
        for _, row in self.test_data.iterrows():
            user_idx = row['user_idx']
            item_idx = row['item_idx']
            true_rating = row['rating']
            
            if method == 'item_based':
                pred_rating = self.predict_item_based(user_idx, item_idx)
            else:
                pred_rating = self.predict_user_based(user_idx, item_idx)
            
            predictions.append(pred_rating)
            ground_truth.append(true_rating)
        
        # 计算评估指标
        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        
        results = {
            'method': method,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'test_size': len(ground_truth)
        }
        
        self.logger.info(f"{method} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return results
    
    def get_recommendations(self, user_idx: int, n_recommendations: int = 10, method: str = 'item_based') -> List[Tuple[int, float]]:
        """为用户生成推荐"""
        if user_idx >= self.user_item_matrix.shape[0]:
            return []
        
        # 获取用户已评价的物品
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        rated_items = set(np.where(user_ratings > 0)[0])
        
        # 预测未评价物品的评分
        item_scores = []
        for item_idx in range(self.user_item_matrix.shape[1]):
            if item_idx not in rated_items:
                if method == 'item_based':
                    score = self.predict_item_based(user_idx, item_idx)
                else:
                    score = self.predict_user_based(user_idx, item_idx)
                item_scores.append((item_idx, score))
        
        # 按评分排序并返回top-N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_recommendations]

def run_amazon_baseline_experiment():
    """运行Amazon基线实验"""
    print("🚀 开始Amazon推荐系统基线实验...")
    
    # 初始化推荐系统
    recommender = AmazonRecommenderBaseline(
        category='All_Beauty',
        max_users=3000,
        max_items=1500
    )
    
    # 加载和预处理数据
    df = recommender.load_and_preprocess_data()
    
    # 训练模型
    start_time = time.time()
    recommender.fit(df)
    training_time = time.time() - start_time
    
    print(f"⏱️ 训练时间: {training_time:.2f} 秒")
    
    # 评估模型
    results = {}
    
    # 基于物品的协同过滤
    results['item_based'] = recommender.evaluate('item_based')
    
    # 基于用户的协同过滤
    results['user_based'] = recommender.evaluate('user_based')
    
    # 打印结果
    print("\n📊 基线模型评估结果:")
    print("-" * 50)
    for method, metrics in results.items():
        print(f"{method.upper()}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  测试样本数: {metrics['test_size']:,}")
        print()
    
    # 生成推荐示例
    print("📋 推荐示例:")
    user_idx = 0  # 第一个用户
    recommendations_item = recommender.get_recommendations(user_idx, n_recommendations=5, method='item_based')
    recommendations_user = recommender.get_recommendations(user_idx, n_recommendations=5, method='user_based')
    
    print(f"用户 {user_idx} 的推荐 (基于物品):")
    for item_idx, score in recommendations_item:
        print(f"  物品 {item_idx}: 预测评分 {score:.2f}")
    
    print(f"用户 {user_idx} 的推荐 (基于用户):")
    for item_idx, score in recommendations_user:
        print(f"  物品 {item_idx}: 预测评分 {score:.2f}")
    
    return results, recommender

if __name__ == "__main__":
    results, model = run_amazon_baseline_experiment()
    print("\n✅ Amazon基线实验完成!")
