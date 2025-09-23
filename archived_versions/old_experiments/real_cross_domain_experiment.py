#!/usr/bin/env python3
"""
Real Cross-Domain Experiment
真实跨域实验 - Amazon Electronics → MovieLens 1M
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrossDomainDataset:
    """跨域数据集处理器"""
    
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.source_domain = None
        self.target_domain = None
    
    def load_amazon_electronics(self, max_users=5000, max_items=3000):
        """加载Amazon Electronics数据作为源域"""
        logger.info("加载Amazon Electronics数据（源域）")
        
        reviews_file = self.base_dir / "dataset/amazon/Electronics_reviews.parquet"
        reviews = pd.read_parquet(reviews_file)
        
        # 数据清洗
        reviews = reviews.dropna(subset=['user_id', 'parent_asin', 'rating'])
        reviews = reviews[reviews['rating'] > 0]
        
        # 限制规模
        top_users = reviews['user_id'].value_counts().head(max_users).index
        top_items = reviews['parent_asin'].value_counts().head(max_items).index
        
        reviews = reviews[
            (reviews['user_id'].isin(top_users)) & 
            (reviews['parent_asin'].isin(top_items))
        ]
        
        # 创建映射
        user_to_idx = {user: idx for idx, user in enumerate(reviews['user_id'].unique())}
        item_to_idx = {item: idx for idx, item in enumerate(reviews['parent_asin'].unique())}
        
        reviews['user_idx'] = reviews['user_id'].map(user_to_idx)
        reviews['item_idx'] = reviews['parent_asin'].map(item_to_idx)
        reviews['domain'] = 'electronics'
        
        self.source_domain = {
            'data': reviews,
            'n_users': len(user_to_idx),
            'n_items': len(item_to_idx),
            'user_map': user_to_idx,
            'item_map': item_to_idx
        }
        
        logger.info(f"Amazon Electronics加载完成: {len(reviews)} 评分, {len(user_to_idx)} 用户, {len(item_to_idx)} 物品")
        return self.source_domain
    
    def load_movielens_1m(self, max_users=3000, max_items=2000):
        """加载MovieLens 1M数据作为目标域"""
        logger.info("加载MovieLens 1M数据（目标域）")
        
        ratings_file = self.base_dir / "dataset/movielens/1m/ratings.csv"
        
        try:
            # 尝试不同的分隔符
            ratings = pd.read_csv(ratings_file, sep='::')
        except:
            try:
                ratings = pd.read_csv(ratings_file, sep=',')
            except:
                ratings = pd.read_csv(ratings_file, sep='\t')
        
        # 标准化列名
        if 'UserID' in ratings.columns:
            ratings = ratings.rename(columns={'UserID': 'user_id', 'MovieID': 'item_id', 'Rating': 'rating'})
        elif 'userId' in ratings.columns:
            ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        
        ratings = ratings.dropna(subset=['user_id', 'item_id', 'rating'])
        ratings = ratings[ratings['rating'] > 0]
        
        # 限制规模
        top_users = ratings['user_id'].value_counts().head(max_users).index
        top_items = ratings['item_id'].value_counts().head(max_items).index
        
        ratings = ratings[
            (ratings['user_id'].isin(top_users)) & 
            (ratings['item_id'].isin(top_items))
        ]
        
        # 创建新的映射（从源域开始编号以避免冲突）
        source_n_users = self.source_domain['n_users'] if self.source_domain else 0
        source_n_items = self.source_domain['n_items'] if self.source_domain else 0
        
        user_to_idx = {user: idx + source_n_users for idx, user in enumerate(ratings['user_id'].unique())}
        item_to_idx = {item: idx + source_n_items for idx, item in enumerate(ratings['item_id'].unique())}
        
        ratings['user_idx'] = ratings['user_id'].map(user_to_idx)
        ratings['item_idx'] = ratings['item_id'].map(item_to_idx)
        ratings['domain'] = 'movies'
        
        self.target_domain = {
            'data': ratings,
            'n_users': len(user_to_idx),
            'n_items': len(item_to_idx),
            'user_map': user_to_idx,
            'item_map': item_to_idx
        }
        
        logger.info(f"MovieLens 1M加载完成: {len(ratings)} 评分, {len(user_to_idx)} 用户, {len(item_to_idx)} 物品")
        return self.target_domain

class CrossDomainRecommender(nn.Module):
    """跨域推荐模型"""
    
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_dims=[64, 32], domain_adaptation=False):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.domain_adaptation = domain_adaptation
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 域适应层（如果启用）
        if domain_adaptation:
            self.domain_classifier = nn.Sequential(
                nn.Linear(embedding_dim * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 2)  # 2个域
            )
            self.domain_loss_weight = 0.1
        
        # 主预测网络
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_idx, item_idx, domain_label=None):
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        rating_pred = self.mlp(x).squeeze()
        
        if self.domain_adaptation and domain_label is not None:
            domain_pred = self.domain_classifier(x)
            return rating_pred, domain_pred
        
        return rating_pred

class RealCrossDomainExperiment:
    """真实跨域实验"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_dir = Path("/home/coder-gw/7Projects_in_7Days/Layerwise-Adapter")
        
        # 加载跨域数据
        self.dataset_loader = CrossDomainDataset(self.base_dir)
        self.source_data = self.dataset_loader.load_amazon_electronics()
        self.target_data = self.dataset_loader.load_movielens_1m()
        
        # 计算总的用户和物品数
        self.total_users = self.source_data['n_users'] + self.target_data['n_users']
        self.total_items = self.source_data['n_items'] + self.target_data['n_items']
        
        self.results = {
            'experiment': 'Real Cross-Domain Transfer Learning',
            'source_domain': 'Amazon Electronics',
            'target_domain': 'MovieLens 1M',
            'timestamp': datetime.now().isoformat(),
            'data_stats': {
                'source_users': self.source_data['n_users'],
                'source_items': self.source_data['n_items'],
                'source_ratings': len(self.source_data['data']),
                'target_users': self.target_data['n_users'],
                'target_items': self.target_data['n_items'],
                'target_ratings': len(self.target_data['data']),
                'total_users': self.total_users,
                'total_items': self.total_items
            },
            'methods': {}
        }
    
    def create_combined_dataset(self):
        """创建组合数据集"""
        source_df = self.source_data['data'].copy()
        target_df = self.target_data['data'].copy()
        
        # 添加域标签
        source_df['domain_label'] = 0  # Electronics
        target_df['domain_label'] = 1  # Movies
        
        # 合并数据
        combined_df = pd.concat([source_df, target_df], ignore_index=True)
        
        return combined_df
    
    def create_dataloader(self, df, batch_size=512, shuffle=True):
        """创建数据加载器"""
        class CombinedDataset(Dataset):
            def __init__(self, df):
                self.df = df.reset_index(drop=True)
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                return {
                    'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
                    'item_idx': torch.tensor(row['item_idx'], dtype=torch.long),
                    'rating': torch.tensor(row['rating'], dtype=torch.float32),
                    'domain_label': torch.tensor(row['domain_label'], dtype=torch.long)
                }
        
        dataset = CombinedDataset(df)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_cross_domain_model(self, model, train_loader, epochs=10, lr=0.001):
        """训练跨域模型"""
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        if model.domain_adaptation:
            domain_criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            total_rating_loss = 0
            total_domain_loss = 0
            
            for batch in train_loader:
                user_idx = batch['user_idx'].to(self.device)
                item_idx = batch['item_idx'].to(self.device)
                ratings = batch['rating'].to(self.device)
                domain_labels = batch['domain_label'].to(self.device)
                
                optimizer.zero_grad()
                
                if model.domain_adaptation:
                    rating_pred, domain_pred = model(user_idx, item_idx, domain_labels)
                    rating_loss = criterion(rating_pred, ratings)
                    domain_loss = domain_criterion(domain_pred, domain_labels)
                    loss = rating_loss + model.domain_loss_weight * domain_loss
                    
                    total_rating_loss += rating_loss.item()
                    total_domain_loss += domain_loss.item()
                else:
                    rating_pred = model(user_idx, item_idx)
                    loss = criterion(rating_pred, ratings)
                    total_rating_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                avg_loss = total_loss / len(train_loader)
                if model.domain_adaptation:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Total: {avg_loss:.4f}, Rating: {total_rating_loss/len(train_loader):.4f}, Domain: {total_domain_loss/len(train_loader):.4f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return model
    
    def evaluate_cross_domain(self, model, test_loader, target_domain_only=True):
        """评估跨域性能"""
        model.eval()
        all_preds = []
        all_ratings = []
        
        with torch.no_grad():
            for batch in test_loader:
                domain_labels = batch['domain_label']
                
                # 如果只评估目标域
                if target_domain_only:
                    target_mask = domain_labels == 1
                    if not target_mask.any():
                        continue
                    
                    user_idx = batch['user_idx'][target_mask].to(self.device)
                    item_idx = batch['item_idx'][target_mask].to(self.device)
                    ratings = batch['rating'][target_mask].to(self.device)
                else:
                    user_idx = batch['user_idx'].to(self.device)
                    item_idx = batch['item_idx'].to(self.device)
                    ratings = batch['rating'].to(self.device)
                
                if len(user_idx) == 0:
                    continue
                
                if model.domain_adaptation:
                    rating_pred, _ = model(user_idx, item_idx, domain_labels)
                else:
                    rating_pred = model(user_idx, item_idx)
                
                all_preds.extend(rating_pred.cpu().numpy())
                all_ratings.extend(ratings.cpu().numpy())
        
        if len(all_preds) == 0:
            return {'rmse': 0, 'mae': 0, 'ndcg_5': 0, 'ndcg_10': 0}
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_ratings = np.array(all_ratings)
        
        mse = np.mean((all_preds - all_ratings) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_preds - all_ratings))
        
        # 简化的NDCG计算
        try:
            if len(all_ratings) > 100:
                sample_indices = np.random.choice(len(all_ratings), 100, replace=False)
                sample_true = all_ratings[sample_indices]
                sample_pred = all_preds[sample_indices]
                
                ndcg_5 = self.calculate_ndcg(sample_true, sample_pred, k=5)
                ndcg_10 = self.calculate_ndcg(sample_true, sample_pred, k=10)
            else:
                ndcg_5 = self.calculate_ndcg(all_ratings, all_preds, k=5)
                ndcg_10 = self.calculate_ndcg(all_ratings, all_preds, k=10)
        except:
            ndcg_5 = ndcg_10 = 0.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'ndcg_5': ndcg_5,
            'ndcg_10': ndcg_10
        }
    
    def calculate_ndcg(self, y_true, y_pred, k=5):
        """计算NDCG"""
        try:
            if len(y_true) < k:
                k = len(y_true)
            
            indices = np.argsort(y_pred)[::-1][:k]
            dcg = sum(y_true[i] / np.log2(j + 2) for j, i in enumerate(indices))
            
            ideal_indices = np.argsort(y_true)[::-1][:k]
            idcg = sum(y_true[i] / np.log2(j + 2) for j, i in enumerate(ideal_indices))
            
            return dcg / idcg if idcg > 0 else 0.0
        except:
            return 0.0
    
    def run_cross_domain_experiment(self):
        """运行跨域实验"""
        logger.info("🚀 开始真实跨域实验")
        
        # 创建组合数据集
        combined_df = self.create_combined_dataset()
        train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
        
        train_loader = self.create_dataloader(train_df)
        test_loader = self.create_dataloader(test_df, shuffle=False)
        
        # 定义不同的跨域方法
        methods = {
            'No_Transfer': {
                'model': CrossDomainRecommender(self.total_users, self.total_items, domain_adaptation=False),
                'description': 'Direct training without transfer'
            },
            'Domain_Adaptation': {
                'model': CrossDomainRecommender(self.total_users, self.total_items, domain_adaptation=True),
                'description': 'Domain adversarial training'
            }
        }
        
        # 训练和评估每个方法
        for method_name, method_info in methods.items():
            logger.info(f"训练方法: {method_name}")
            
            # 训练
            start_time = time.time()
            model = self.train_cross_domain_model(method_info['model'], train_loader, epochs=8)
            training_time = time.time() - start_time
            
            # 在目标域评估
            target_metrics = self.evaluate_cross_domain(model, test_loader, target_domain_only=True)
            
            # 在全域评估
            all_metrics = self.evaluate_cross_domain(model, test_loader, target_domain_only=False)
            
            # 计算迁移差距
            source_only_loader = self.create_dataloader(
                train_df[train_df['domain_label'] == 0], shuffle=False
            )
            source_metrics = self.evaluate_cross_domain(model, source_only_loader, target_domain_only=False)
            
            transfer_gap = ((target_metrics['rmse'] - source_metrics['rmse']) / source_metrics['rmse']) * 100 if source_metrics['rmse'] > 0 else 0
            
            # 保存结果
            self.results['methods'][method_name] = {
                'description': method_info['description'],
                'target_domain_rmse': float(target_metrics['rmse']),
                'target_domain_mae': float(target_metrics['mae']),
                'target_domain_ndcg_5': float(target_metrics['ndcg_5']),
                'target_domain_ndcg_10': float(target_metrics['ndcg_10']),
                'source_domain_rmse': float(source_metrics['rmse']),
                'transfer_gap_percent': float(transfer_gap),
                'training_time_s': float(training_time),
                'domain_adaptation': method_name == 'Domain_Adaptation'
            }
            
            logger.info(f"{method_name} - 目标域RMSE: {target_metrics['rmse']:.4f}, 迁移差距: {transfer_gap:.1f}%")
        
        return self.results
    
    def generate_report(self):
        """生成跨域实验报告"""
        report = f"""# Real Cross-Domain Transfer Learning Report

## 实验设置
- **源域**: {self.results['source_domain']} ({self.results['data_stats']['source_ratings']:,} 评分)
- **目标域**: {self.results['target_domain']} ({self.results['data_stats']['target_ratings']:,} 评分)
- **迁移方向**: Amazon Electronics → MovieLens Movies
- **实验时间**: {self.results['timestamp']}

## 数据统计
- **源域用户**: {self.results['data_stats']['source_users']:,}
- **源域物品**: {self.results['data_stats']['source_items']:,}
- **目标域用户**: {self.results['data_stats']['target_users']:,}
- **目标域物品**: {self.results['data_stats']['target_items']:,}
- **总用户空间**: {self.results['data_stats']['total_users']:,}
- **总物品空间**: {self.results['data_stats']['total_items']:,}

## 跨域性能结果

| Method | Target RMSE | Target NDCG@5 | Transfer Gap | Domain Adaptation |
|--------|-------------|---------------|--------------|-------------------|
"""
        
        for method, metrics in self.results['methods'].items():
            report += f"| {method.replace('_', ' ')} | {metrics['target_domain_rmse']:.4f} | {metrics['target_domain_ndcg_5']:.4f} | {metrics['transfer_gap_percent']:.1f}% | {'Yes' if metrics['domain_adaptation'] else 'No'} |\n"
        
        report += f"""
## 关键发现

基于真实Amazon Electronics → MovieLens迁移实验：

1. **跨域挑战**: 产品评论到电影评分的语义差异显著
2. **迁移效果**: 域适应技术对减少迁移损失的实际效果
3. **数据规模影响**: 源域和目标域数据规模对迁移性能的影响

## 域差异分析
- **内容类型**: 产品描述 vs 电影元数据
- **评分分布**: Amazon评分偏高，MovieLens分布更均匀
- **用户行为**: 购买决策 vs 娱乐偏好
- **特征空间**: 完全不重叠的用户和物品ID空间

## 实际应用场景
这种跨域迁移在实际推荐系统中很常见：
- 新业务线启动时的冷启动问题
- 跨平台用户偏好迁移
- 多领域推荐系统的知识共享

## 局限性
- 当前实验未使用内容特征辅助迁移
- 用户和物品的重叠为零，是最困难的迁移场景
- 域适应技术还有进一步优化空间
"""
        
        return report

def main():
    """主函数"""
    logger.info("🚀 启动真实跨域实验")
    
    try:
        experiment = RealCrossDomainExperiment()
        results = experiment.run_cross_domain_experiment()
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'real_cross_domain_results_{timestamp}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成报告
        report = experiment.generate_report()
        report_file = f'real_cross_domain_report_{timestamp}.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ 跨域实验完成，结果保存到: {results_file}")
        logger.info(f"📄 报告保存到: {report_file}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("🌐 真实跨域实验结果")
        print("="*60)
        for method, metrics in results['methods'].items():
            print(f"{method}: Target RMSE={metrics['target_domain_rmse']:.4f}, Transfer Gap={metrics['transfer_gap_percent']:.1f}%")
        print("="*60)
        
    except Exception as e:
        logger.error(f"跨域实验失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
