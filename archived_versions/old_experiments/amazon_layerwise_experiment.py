"""
Layerwise Adapter Amazon实验

将Fisher信息分析和layerwise重要性分析应用到真实Amazon推荐场景中
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class LayerwiseConfig:
    """Layerwise实验配置"""
    # 模型参数
    embedding_dim: int = 64
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2
    
    # 训练参数
    learning_rate: float = 0.001
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    
    # Fisher信息参数
    fisher_sample_size: int = 1000
    importance_threshold: float = 0.1
    
    # 数据参数
    max_users: int = 3000
    max_items: int = 2000
    test_ratio: float = 0.2
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]

class NeuralCollaborativeFiltering(nn.Module):
    """神经协同过滤模型"""
    
    def __init__(self, n_users: int, n_items: int, config: LayerwiseConfig):
        super().__init__()
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(n_items, config.embedding_dim)
        
        # MLP层
        input_dim = config.embedding_dim * 2
        self.layers = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # 拼接特征
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # MLP层
        for layer in self.layers:
            x = layer(x)
        
        # 输出
        output = self.output_layer(x)
        return torch.sigmoid(output) * 4 + 1  # 映射到[1, 5]范围

class LayerwiseFisherAnalyzer:
    """Layerwise Fisher信息分析器"""
    
    def __init__(self, model: nn.Module, config: LayerwiseConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Fisher信息存储
        self.layer_fisher_info = {}
        self.layer_importance_scores = {}
        
    def compute_fisher_information(self, dataloader) -> Dict[str, torch.Tensor]:
        """计算每层的Fisher信息矩阵"""
        self.logger.info("计算Fisher信息矩阵...")
        
        self.model.eval()
        fisher_info = {}
        
        # 初始化Fisher信息矩阵
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)
        
        n_samples = 0
        max_samples = self.config.fisher_sample_size
        
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(dataloader):
            if n_samples >= max_samples:
                break
            
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            ratings = ratings.to(self.device)
            
            # 前向传播
            self.model.zero_grad()
            outputs = self.model(user_ids, item_ids)
            loss = F.mse_loss(outputs.squeeze(), ratings)
            
            # 反向传播计算梯度
            loss.backward()
            
            # 累积梯度平方（Fisher信息的对角近似）
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
            
            n_samples += user_ids.size(0)
        
        # 标准化Fisher信息
        for name in fisher_info:
            fisher_info[name] /= n_samples
        
        self.layer_fisher_info = fisher_info
        self.logger.info(f"Fisher信息计算完成，样本数: {n_samples}")
        
        return fisher_info
    
    def analyze_layer_importance(self) -> Dict[str, float]:
        """分析每层的重要性"""
        self.logger.info("分析层级重要性...")
        
        layer_importance = {}
        
        # 按层级分组参数
        layer_groups = self._group_parameters_by_layer()
        
        for layer_name, param_names in layer_groups.items():
            total_importance = 0.0
            param_count = 0
            
            for param_name in param_names:
                if param_name in self.layer_fisher_info:
                    fisher_tensor = self.layer_fisher_info[param_name]
                    # 计算Fisher信息的总和作为重要性指标
                    if len(fisher_tensor.shape) == 1:
                        fisher_sum = fisher_tensor.sum()
                    else:
                        fisher_sum = fisher_tensor.sum()
                    total_importance += fisher_sum.item()
                    param_count += 1
            
            if param_count > 0:
                layer_importance[layer_name] = total_importance / param_count
            else:
                layer_importance[layer_name] = 0.0
        
        # 标准化重要性分数
        max_importance = max(layer_importance.values()) if layer_importance else 1.0
        if max_importance > 0:
            for layer_name in layer_importance:
                layer_importance[layer_name] /= max_importance
        
        self.layer_importance_scores = layer_importance
        
        self.logger.info("层级重要性分析完成")
        return layer_importance
    
    def _group_parameters_by_layer(self) -> Dict[str, List[str]]:
        """按层级分组参数"""
        layer_groups = {
            'embedding': [],
            'mlp_layers': [],
            'output': []
        }
        
        for name, _ in self.model.named_parameters():
            if 'embedding' in name:
                layer_groups['embedding'].append(name)
            elif 'output_layer' in name:
                layer_groups['output'].append(name)
            else:
                layer_groups['mlp_layers'].append(name)
        
        return layer_groups
    
    def identify_critical_layers(self, threshold: float = None) -> List[str]:
        """识别关键层"""
        if threshold is None:
            threshold = self.config.importance_threshold
        
        critical_layers = []
        for layer_name, importance in self.layer_importance_scores.items():
            if importance >= threshold:
                critical_layers.append(layer_name)
        
        self.logger.info(f"识别出 {len(critical_layers)} 个关键层")
        return critical_layers

class AmazonLayerwiseExperiment:
    """Amazon Layerwise实验主类"""
    
    def __init__(self, category: str = 'All_Beauty', config: Optional[LayerwiseConfig] = None):
        self.category = category
        self.config = config or LayerwiseConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 数据编码器
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # 模型和分析器
        self.model = None
        self.fisher_analyzer = None
        
        # 实验结果
        self.results = {}
        
    def load_and_preprocess_data(self, data_path: str = "dataset/amazon") -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """加载和预处理数据"""
        self.logger.info(f"加载Amazon {self.category} 数据...")
        
        # 加载数据
        reviews_file = Path(data_path) / f"{self.category}_reviews.parquet"
        df = pd.read_parquet(reviews_file)
        
        # 预处理
        df = df.rename(columns={'parent_asin': 'item_id'})
        df = df.dropna(subset=['user_id', 'item_id', 'rating'])
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
        
        # 过滤低频交互
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= 5].index
        valid_items = item_counts[item_counts >= 5].index
        
        df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        
        # 限制数据规模
        if len(df['user_id'].unique()) > self.config.max_users:
            top_users = df['user_id'].value_counts().head(self.config.max_users).index
            df = df[df['user_id'].isin(top_users)]
        
        if len(df['item_id'].unique()) > self.config.max_items:
            top_items = df['item_id'].value_counts().head(self.config.max_items).index
            df = df[df['item_id'].isin(top_items)]
        
        # 编码用户和物品ID
        df['user_idx'] = self.user_encoder.fit_transform(df['user_id'])
        df['item_idx'] = self.item_encoder.fit_transform(df['item_id'])
        
        self.logger.info(f"预处理完成: {len(df):,} 交互, {df['user_idx'].nunique():,} 用户, {df['item_idx'].nunique():,} 物品")
        
        # 划分训练测试集
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=self.config.test_ratio, random_state=42)
        
        # 创建数据加载器
        train_loader = self._create_dataloader(train_df, shuffle=True)
        test_loader = self._create_dataloader(test_df, shuffle=False)
        
        return train_loader, test_loader
    
    def _create_dataloader(self, df: pd.DataFrame, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """创建数据加载器"""
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(df['user_idx'].values),
            torch.LongTensor(df['item_idx'].values),
            torch.FloatTensor(df['rating'].values)
        )
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2
        )
    
    def train_model(self, train_loader, test_loader):
        """训练神经协同过滤模型"""
        self.logger.info("开始训练神经协同过滤模型...")
        
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        # 初始化模型
        self.model = NeuralCollaborativeFiltering(n_users, n_items, self.config).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # 训练循环
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for user_ids, item_ids, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(user_ids, item_ids)
                loss = criterion(outputs.squeeze(), ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for user_ids, item_ids, ratings in test_loader:
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    ratings = ratings.to(self.device)
                    
                    outputs = self.model(user_ids, item_ids)
                    loss = criterion(outputs.squeeze(), ratings)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")
            
            # 早停机制
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    self.logger.info(f"早停于epoch {epoch}")
                    break
        
        self.logger.info("模型训练完成")
    
    def run_layerwise_analysis(self, train_loader):
        """运行layerwise分析"""
        self.logger.info("开始Layerwise Fisher信息分析...")
        
        # 初始化Fisher分析器
        self.fisher_analyzer = LayerwiseFisherAnalyzer(self.model, self.config)
        
        # 计算Fisher信息
        fisher_info = self.fisher_analyzer.compute_fisher_information(train_loader)
        
        # 分析层级重要性
        layer_importance = self.fisher_analyzer.analyze_layer_importance()
        
        # 识别关键层
        critical_layers = self.fisher_analyzer.identify_critical_layers()
        
        # 保存结果
        self.results['fisher_info'] = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v for k, v in fisher_info.items()}
        self.results['layer_importance'] = layer_importance
        self.results['critical_layers'] = critical_layers
        
        return layer_importance, critical_layers
    
    def evaluate_model(self, test_loader) -> Dict[str, float]:
        """评估模型性能"""
        self.logger.info("评估模型性能...")
        
        self.model.eval()
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                
                outputs = self.model(user_ids, item_ids)
                predictions.extend(outputs.squeeze().cpu().numpy())
                ground_truth.extend(ratings.numpy())
        
        # 计算评估指标
        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'test_samples': len(ground_truth)
        }
        
        self.results['performance'] = metrics
        self.logger.info(f"模型性能 - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return metrics
    
    def save_results(self, save_dir: str = "results/amazon_layerwise"):
        """保存实验结果"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON结果
        results_file = save_path / f"amazon_layerwise_{self.category}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 生成可视化
        self._create_visualizations(save_path, timestamp)
        
        # 生成报告
        self._generate_report(save_path, timestamp)
        
        self.logger.info(f"结果已保存至: {save_path}")
    
    def _create_visualizations(self, save_path: Path, timestamp: str):
        """创建可视化图表"""
        # 层级重要性可视化
        if 'layer_importance' in self.results:
            plt.figure(figsize=(10, 6))
            layers = list(self.results['layer_importance'].keys())
            importance = list(self.results['layer_importance'].values())
            
            plt.bar(layers, importance, alpha=0.7)
            plt.title(f'Layer Importance Analysis - {self.category}')
            plt.xlabel('Layer')
            plt.ylabel('Importance Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(save_path / f"layer_importance_{self.category}_{timestamp}.png", dpi=300)
            plt.close()
    
    def _generate_report(self, save_path: Path, timestamp: str):
        """生成实验报告"""
        report_file = save_path / f"report_{self.category}_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Amazon Layerwise Analysis Report - {self.category}\n\n")
            f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 模型性能
            if 'performance' in self.results:
                perf = self.results['performance']
                f.write("## 模型性能\n\n")
                f.write(f"- RMSE: {perf['rmse']:.4f}\n")
                f.write(f"- MAE: {perf['mae']:.4f}\n")
                f.write(f"- 测试样本数: {perf['test_samples']:,}\n\n")
            
            # 层级重要性
            if 'layer_importance' in self.results:
                f.write("## 层级重要性分析\n\n")
                for layer, importance in self.results['layer_importance'].items():
                    f.write(f"- {layer}: {importance:.4f}\n")
                f.write("\n")
            
            # 关键层
            if 'critical_layers' in self.results:
                f.write("## 关键层识别\n\n")
                f.write(f"识别出 {len(self.results['critical_layers'])} 个关键层:\n")
                for layer in self.results['critical_layers']:
                    f.write(f"- {layer}\n")

def run_amazon_layerwise_experiment():
    """运行Amazon Layerwise实验"""
    print("🚀 开始Amazon Layerwise实验...")
    
    # 实验配置
    config = LayerwiseConfig(
        embedding_dim=32,
        hidden_dims=[64, 32, 16],
        max_epochs=50,
        batch_size=256,
        max_users=2000,
        max_items=1500
    )
    
    # 初始化实验
    experiment = AmazonLayerwiseExperiment('All_Beauty', config)
    
    # 加载数据
    train_loader, test_loader = experiment.load_and_preprocess_data()
    
    # 训练模型
    experiment.train_model(train_loader, test_loader)
    
    # Layerwise分析
    layer_importance, critical_layers = experiment.run_layerwise_analysis(train_loader)
    
    # 评估模型
    metrics = experiment.evaluate_model(test_loader)
    
    # 保存结果
    experiment.save_results()
    
    # 打印结果摘要
    print("\n📊 实验结果摘要:")
    print(f"模型性能 - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    print(f"层级重要性: {layer_importance}")
    print(f"关键层: {critical_layers}")
    
    print("\n✅ Amazon Layerwise实验完成!")
    return experiment

if __name__ == "__main__":
    experiment = run_amazon_layerwise_experiment()
