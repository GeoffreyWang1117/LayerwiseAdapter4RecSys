"""
独立的Amazon数据处理测试脚本
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataConfig:
    """数据处理配置"""
    min_user_interactions: int = 5
    min_item_interactions: int = 5  
    test_ratio: float = 0.2
    val_ratio: float = 0.1   
    random_seed: int = 42    
    max_users: Optional[int] = None  
    max_items: Optional[int] = None  

class SimpleAmazonProcessor:
    """简化的Amazon数据处理器"""
    
    def __init__(self, data_path: str = "dataset/amazon", config: Optional[DataConfig] = None):
        self.data_path = Path(data_path)
        self.config = config or DataConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.available_categories = [
            'All_Beauty', 'Electronics', 'Books', 'Movies_and_TV', 
            'Home_and_Kitchen', 'Arts_Crafts_and_Sewing', 'Automotive',
            'Office_Products', 'Sports_and_Outdoors', 'Toys_and_Games'
        ]
        
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
    
    def load_category_data(self, category: str) -> pd.DataFrame:
        """加载指定类别的数据"""
        reviews_file = self.data_path / f"{category}_reviews.parquet"
        
        if not reviews_file.exists():
            raise FileNotFoundError(f"评论文件不存在: {reviews_file}")
        
        self.logger.info(f"加载类别 {category} 的数据...")
        reviews_df = pd.read_parquet(reviews_file)
        self.logger.info(f"加载了 {len(reviews_df):,} 条评论记录")
        
        return reviews_df
    
    def normalize_reviews_data(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """标准化评论数据格式"""
        # 统一列名
        column_mapping = {
            'parent_asin': 'item_id',
            'asin': 'item_id',
            'user_id': 'user_id',
            'rating': 'rating',
            'timestamp': 'timestamp',
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
        normalized_df = normalized_df[
            (normalized_df['rating'] >= 1) & (normalized_df['rating'] <= 5)
        ]
        
        self.logger.info(f"标准化后保留 {len(normalized_df):,} 条记录")
        return normalized_df
    
    def filter_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤低频用户和物品"""
        original_size = len(df)
        
        # 过滤低频用户
        if 'user_id' in df.columns:
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.config.min_user_interactions].index
            df = df[df['user_id'].isin(valid_users)]
        
        # 过滤低频物品
        if 'item_id' in df.columns:
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
        """编码用户和物品ID"""
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
        """创建用户-物品交互矩阵"""
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
    
    def get_data_summary(self) -> pd.DataFrame:
        """获取所有可用类别的数据摘要"""
        categories = self.get_available_categories()
        summary_data = []
        
        for category in categories:
            try:
                reviews_df = self.load_category_data(category)
                normalized_df = self.normalize_reviews_data(reviews_df)
                
                n_users = normalized_df['user_id'].nunique()
                n_items = normalized_df['item_id'].nunique()
                
                summary_data.append({
                    'category': category,
                    'total_reviews': len(reviews_df),
                    'after_cleaning': len(normalized_df),
                    'n_users': n_users,
                    'n_items': n_items,
                    'avg_rating': normalized_df['rating'].mean(),
                    'density': len(normalized_df) / (n_users * n_items) * 100
                })
            except Exception as e:
                self.logger.warning(f"处理类别 {category} 时出错: {e}")
        
        return pd.DataFrame(summary_data)
    
    def process_category(self, category: str) -> Dict:
        """处理指定类别的完整数据流程"""
        self.logger.info(f"开始处理类别: {category}")
        
        # 1. 加载数据
        reviews_df = self.load_category_data(category)
        
        # 2. 标准化数据
        normalized_df = self.normalize_reviews_data(reviews_df)
        
        # 3. 过滤交互
        filtered_df = self.filter_interactions(normalized_df)
        
        # 4. 编码ID
        encoded_df = self.encode_ids(filtered_df)
        
        # 5. 创建交互矩阵
        interaction_matrix = self.create_interaction_matrix(encoded_df)
        
        # 6. 计算统计信息
        stats = {
            'category': category,
            'total_interactions': len(encoded_df),
            'n_users': len(self.user_encoder.classes_),
            'n_items': len(self.item_encoder.classes_),
            'density': interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]) * 100,
            'avg_rating': encoded_df['rating'].mean(),
            'rating_std': encoded_df['rating'].std(),
        }
        
        result = {
            'stats': stats,
            'data': encoded_df,
            'matrix': interaction_matrix,
        }
        
        self.logger.info(f"类别 {category} 处理完成")
        return result

def test_data_processor():
    """测试数据处理器"""
    print("🚀 开始测试Amazon数据处理器...")
    
    # 创建配置
    config = DataConfig(
        min_user_interactions=5,
        min_item_interactions=5,
        max_users=5000,   # 限制用户数用于测试
        max_items=2000    # 限制物品数用于测试
    )
    
    # 初始化处理器
    processor = SimpleAmazonProcessor(config=config)
    
    # 获取可用类别
    categories = processor.get_available_categories()
    print(f"📋 可用类别: {categories}")
    
    # 获取数据摘要
    print("\n📊 生成数据摘要...")
    summary = processor.get_data_summary()
    print(summary.to_string(index=False))
    
    # 选择一个较小的类别进行详细测试
    test_category = 'All_Beauty'  # 相对较小的数据集
    
    if test_category in categories:
        print(f"\n🔍 详细处理类别: {test_category}")
        result = processor.process_category(test_category)
        
        # 打印处理结果
        print("\n📈 处理结果统计:")
        stats = result['stats']
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
        
        # 分析评分分布
        print("\n📊 评分分布:")
        ratings = result['data']['rating'].values
        rating_counts = pd.Series(ratings).value_counts().sort_index()
        for rating, count in rating_counts.items():
            print(f"  评分 {rating}: {count:,} 次 ({count/len(ratings)*100:.1f}%)")
        
        print(f"\n✅ 类别 {test_category} 处理完成!")
        return result
    else:
        print(f"❌ 测试类别 {test_category} 不可用")
        return None

if __name__ == "__main__":
    result = test_data_processor()
    print("\n🎉 数据处理器测试完成!")
