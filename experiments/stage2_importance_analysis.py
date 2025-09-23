#!/usr/bin/env python3
"""
阶段2：核心层重要性分析
基于阶段1的稳定模型，实现Fisher信息、梯度分析、层消融等核心方法
使用真实Amazon数据进行层重要性评估
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

class HonestDataLoader:
    """诚实的数据加载器 - 与阶段1兼容"""
    def __init__(self, data_path='dataset/amazon/Electronics_reviews.parquet'):
        self.data_path = data_path
        
    def load_real_data(self):
        """加载真实Amazon数据"""
        logger.info("📊 加载真实Amazon Electronics数据...")
        df = pd.read_parquet(self.data_path)
        logger.info(f"原始数据: {len(df):,} 条记录")
        
        # 验证数据真实性
        self._validate_data(df)
        
        # 质量过滤
        df = self._filter_quality_data(df)
        
        return df
    
    def _validate_data(self, df):
        """验证数据真实性"""
        logger.info("🔍 验证Amazon Electronics数据真实性...")
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"列名: {df.columns.tolist()}")
        
        # 文本多样性检查
        if 'text' in df.columns:
            unique_texts = df['text'].nunique()
            total_texts = len(df)
            diversity = unique_texts / total_texts
            logger.info(f"文本唯一性: {unique_texts:,}/{total_texts:,} = {diversity:.3f}")
            
            if diversity < 0.7:
                logger.warning(f"⚠️ 文本多样性较低: {diversity:.3f}")
            else:
                logger.info("✅ 文本多样性验证通过")
        
        # 统计分析
        if 'rating' in df.columns:
            rating_dist = df['rating'].value_counts().sort_index()
            logger.info("评分分布:")
            for rating, count in rating_dist.items():
                pct = count / len(df) * 100
                logger.info(f"  {rating}星: {count:,} ({pct:.1f}%)")
        
        logger.info("✅ 数据真实性验证完成")
    
    def _filter_quality_data(self, df):
        """过滤高质量数据"""
        initial_count = len(df)
        
        # 基本过滤
        df = df.dropna(subset=['text', 'rating'])
        df = df[df['text'].str.len() > 10]  # 至少10个字符
        df = df[df['rating'].isin([1.0, 2.0, 3.0, 4.0, 5.0])]  # 有效评分
        
        final_count = len(df)
        retention_rate = final_count / initial_count
        logger.info(f"质量过滤: {initial_count:,} -> {final_count:,} ({retention_rate:.1%}保留)")
        
        return df

class StableTransformer(nn.Module):
    """与阶段1兼容的稳定Transformer模型"""
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=12, 
                 hidden_dim=2048, max_seq_len=256, num_classes=2, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, return_layer_outputs=False):
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:seq_len]
        x = self.dropout(x)
        
        # 注意力掩码
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        
        # 层输出存储
        layer_outputs = []
        
        # Transformer层
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)
            if return_layer_outputs:
                layer_outputs.append(x.clone())
        
        # 最终处理
        x = self.layer_norm(x)
        
        # 池化 (使用[CLS]位置或平均池化)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # 分类
        logits = self.classifier(self.dropout(x))
        
        if return_layer_outputs:
            return logits, layer_outputs
        return logits

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, attention_mask=None):
        # Self-attention
        if attention_mask is not None:
            # 转换掩码格式
            attn_mask = (attention_mask == 0)
        else:
            attn_mask = None
            
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x

class CoreImportanceAnalyzer:
    """核心重要性分析器 - Fisher信息、梯度分析、层消融"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.importance_scores = {}
        self.analysis_results = {}
        
    def analyze_layer_importance(self, dataloader, methods=['fisher', 'gradient', 'ablation']):
        """分析层重要性"""
        logger.info("🔍 开始核心层重要性分析...")
        
        results = {}
        
        if 'fisher' in methods:
            logger.info("🎯 计算Fisher信息矩阵...")
            results['fisher'] = self._compute_fisher_information(dataloader)
            
        if 'gradient' in methods:
            logger.info("📈 计算梯度重要性...")
            results['gradient'] = self._compute_gradient_importance(dataloader)
            
        if 'ablation' in methods:
            logger.info("✂️ 执行层消融分析...")
            results['ablation'] = self._compute_ablation_importance(dataloader)
            
        self.analysis_results = results
        return results
    
    def _compute_fisher_information(self, dataloader):
        """计算Fisher信息矩阵"""
        self.model.train()  # 需要训练模式来计算梯度
        fisher_info = {}
        
        # 初始化Fisher信息存储
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)
        
        total_samples = 0
        
        # 不使用no_grad，因为需要计算梯度
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="Fisher信息计算")):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 对每个样本计算Fisher信息
            for i in range(min(data.size(0), 8)):  # 限制批次内样本数量
                self.model.zero_grad()
                
                # 单样本前向传播
                single_data = data[i:i+1]
                single_target = targets[i:i+1]
                
                outputs = self.model(single_data)
                log_probs = F.log_softmax(outputs, dim=1)
                
                # 单样本损失
                sample_log_prob = log_probs[0, single_target[0]]
                sample_log_prob.backward()
                
                # 累积Fisher信息 (梯度的平方)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_info[name] += param.grad.data ** 2
                
                total_samples += 1
            
            if batch_idx >= 15:  # 限制计算量
                break
        
        # 归一化Fisher信息
        for name in fisher_info:
            fisher_info[name] /= total_samples
        
        # 按层聚合Fisher信息
        layer_fisher = {}
        for layer_idx in range(self.model.num_layers):
            layer_fisher[f'layer_{layer_idx}'] = 0
            param_count = 0
            
            for name, fisher_val in fisher_info.items():
                if f'layers.{layer_idx}' in name:
                    layer_fisher[f'layer_{layer_idx}'] += fisher_val.mean().item()
                    param_count += 1
            
            if param_count > 0:
                layer_fisher[f'layer_{layer_idx}'] /= param_count
        
        logger.info("✅ Fisher信息计算完成")
        return layer_fisher
    
    def _compute_gradient_importance(self, dataloader):
        """计算梯度重要性"""
        self.model.eval()
        gradient_norms = {f'layer_{i}': [] for i in range(self.model.num_layers)}
        
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="梯度重要性计算")):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            # 计算每层的梯度范数
            for layer_idx in range(self.model.num_layers):
                layer_grad_norm = 0
                param_count = 0
                
                for name, param in self.model.named_parameters():
                    if f'layers.{layer_idx}' in name and param.grad is not None:
                        layer_grad_norm += param.grad.norm().item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    layer_grad_norm = np.sqrt(layer_grad_norm / param_count)
                    gradient_norms[f'layer_{layer_idx}'].append(layer_grad_norm)
            
            if batch_idx >= 50:  # 限制计算量
                break
        
        # 计算平均梯度范数
        layer_gradient_importance = {}
        for layer_name, norms in gradient_norms.items():
            if norms:
                layer_gradient_importance[layer_name] = np.mean(norms)
            else:
                layer_gradient_importance[layer_name] = 0.0
        
        logger.info("✅ 梯度重要性计算完成")
        return layer_gradient_importance
    
    def _compute_ablation_importance(self, dataloader):
        """计算层消融重要性"""
        self.model.eval()
        
        # 获取基准性能
        baseline_acc = self._evaluate_model(dataloader)
        logger.info(f"基准准确率: {baseline_acc:.4f}")
        
        ablation_importance = {}
        
        for layer_idx in range(self.model.num_layers):
            logger.info(f"消融第{layer_idx}层...")
            
            # 保存原始权重
            original_weights = {}
            for name, param in self.model.named_parameters():
                if f'layers.{layer_idx}' in name:
                    original_weights[name] = param.data.clone()
                    param.data.zero_()  # 将权重设为0
            
            # 评估消融后的性能
            ablated_acc = self._evaluate_model(dataloader)
            importance = baseline_acc - ablated_acc
            ablation_importance[f'layer_{layer_idx}'] = importance
            
            logger.info(f"消融后准确率: {ablated_acc:.4f}, 重要性: {importance:.4f}")
            
            # 恢复原始权重
            for name, original_weight in original_weights.items():
                for param_name, param in self.model.named_parameters():
                    if param_name == name:
                        param.data.copy_(original_weight)
                        break
        
        logger.info("✅ 层消融分析完成")
        return ablation_importance
    
    def _evaluate_model(self, dataloader):
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total
    
    def select_important_layers(self, top_k=6, method='fisher'):
        """选择重要层"""
        if method not in self.analysis_results:
            raise ValueError(f"方法 {method} 的分析结果不存在")
        
        importance_scores = self.analysis_results[method]
        
        # 排序并选择top-k
        sorted_layers = sorted(importance_scores.items(), 
                             key=lambda x: x[1], reverse=True)
        
        selected_layers = [layer for layer, score in sorted_layers[:top_k]]
        selected_scores = {layer: score for layer, score in sorted_layers[:top_k]}
        
        logger.info(f"✅ 使用{method}方法选择top-{top_k}重要层:")
        for layer, score in sorted_layers[:top_k]:
            logger.info(f"  {layer}: {score:.6f}")
        
        return selected_layers, selected_scores
    
    def create_compressed_model(self, selected_layers, original_vocab_size):
        """创建压缩模型"""
        logger.info(f"🏗️ 创建压缩模型，保留{len(selected_layers)}层...")
        
        # 创建新的压缩模型
        compressed_model = StableTransformer(
            vocab_size=original_vocab_size,
            embed_dim=self.model.embed_dim,
            num_heads=8,  # 保持原有配置
            num_layers=len(selected_layers),
            hidden_dim=2048,
            max_seq_len=self.model.max_seq_len,
            num_classes=2,
            dropout=0.1
        )
        
        # 复制选中层的权重
        layer_indices = [int(layer.split('_')[1]) for layer in selected_layers]
        layer_indices.sort()
        
        for new_idx, old_idx in enumerate(layer_indices):
            # 复制transformer层权重
            old_layer = self.model.layers[old_idx]
            new_layer = compressed_model.layers[new_idx]
            
            new_layer.load_state_dict(old_layer.state_dict())
        
        # 复制其他组件
        compressed_model.embedding.load_state_dict(self.model.embedding.state_dict())
        compressed_model.layer_norm.load_state_dict(self.model.layer_norm.state_dict())
        compressed_model.classifier.load_state_dict(self.model.classifier.state_dict())
        
        # 位置嵌入需要特殊处理
        compressed_model.pos_embedding.data = self.model.pos_embedding.data.clone()
        
        logger.info(f"✅ 压缩模型创建完成: {self.model.num_layers} -> {len(selected_layers)}层")
        
        return compressed_model
    
    def evaluate_compression(self, original_model, compressed_model, test_dataloader):
        """评估压缩效果"""
        logger.info("📊 评估压缩效果...")
        
        # 评估原始模型
        original_acc = self._evaluate_model_with_loader(original_model, test_dataloader)
        
        # 评估压缩模型
        compressed_acc = self._evaluate_model_with_loader(compressed_model, test_dataloader)
        
        # 计算压缩比
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        compression_ratio = original_params / compressed_params
        
        results = {
            'original_accuracy': original_acc,
            'compressed_accuracy': compressed_acc,
            'accuracy_retention': compressed_acc / original_acc,
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'compression_ratio': compression_ratio,
            'parameter_reduction': 1 - (compressed_params / original_params)
        }
        
        logger.info("🎯 压缩效果评估:")
        logger.info(f"  原始准确率: {original_acc:.4f}")
        logger.info(f"  压缩准确率: {compressed_acc:.4f}")
        logger.info(f"  准确率保持: {results['accuracy_retention']:.2%}")
        logger.info(f"  压缩比: {compression_ratio:.1f}x")
        logger.info(f"  参数减少: {results['parameter_reduction']:.1%}")
        
        return results
    
    def _evaluate_model_with_loader(self, model, dataloader):
        """使用数据加载器评估模型"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total

def load_stage1_results():
    """加载阶段1的结果"""
    results_path = 'results/stage1_complete_results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def prepare_data_from_stage1():
    """基于阶段1准备数据"""
    # 重新加载和处理数据以保持一致性
    loader = HonestDataLoader()
    df = loader.load_real_data()
    
    # 创建正负例
    positive_samples = df[df['rating'] >= 4].sample(n=min(15000, len(df[df['rating'] >= 4])))
    negative_samples = df[df['rating'] <= 2].sample(n=min(15000, len(df[df['rating'] <= 2])))
    
    # 合并数据
    final_df = pd.concat([positive_samples, negative_samples]).sample(frac=1).reset_index(drop=True)
    final_df['label'] = (final_df['rating'] >= 4).astype(int)
    
    # 简单tokenization (与阶段1保持一致)
    vocab = set()
    for text in final_df['text']:
        vocab.update(text.lower().split())
    
    vocab = ['<PAD>', '<UNK>'] + sorted(list(vocab))[:10000]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    def tokenize_text(text, max_len=256):
        tokens = text.lower().split()[:max_len]
        indices = [word_to_idx.get(token, 1) for token in tokens]  # 1 for <UNK>
        indices.extend([0] * (max_len - len(indices)))  # 0 for <PAD>
        return indices[:max_len]
    
    # Tokenization
    tokenized_texts = [tokenize_text(text) for text in final_df['text']]
    
    # 转换为tensor
    X = torch.tensor(tokenized_texts, dtype=torch.long)
    y = torch.tensor(final_df['label'].values, dtype=torch.long)
    
    # 数据分割
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return train_loader, val_loader, test_loader, len(vocab)

def main():
    """主函数"""
    logger.info("🚀 开始阶段2：核心层重要性分析")
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 使用设备: {device}")
    
    # 加载阶段1结果
    stage1_results = load_stage1_results()
    if stage1_results:
        logger.info("✅ 阶段1结果已加载")
    
    # 准备数据
    logger.info("📊 准备数据...")
    train_loader, val_loader, test_loader, vocab_size = prepare_data_from_stage1()
    logger.info(f"✅ 数据准备完成，词汇表大小: {vocab_size}")
    
    # 创建模型 (与阶段1相同的架构)
    logger.info("🏗️ 创建Transformer模型...")
    model = StableTransformer(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=12,
        hidden_dim=2048,
        max_seq_len=256,
        num_classes=2,
        dropout=0.1
    )
    model.to(device)
    
    # 检查是否有预训练模型
    if os.path.exists('best_model_epoch_6.pt'):
        logger.info("📂 加载预训练模型...")
        checkpoint = torch.load('best_model_epoch_6.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ 预训练模型加载完成")
    else:
        logger.info("⚠️ 未找到预训练模型，使用随机初始化")
    
    # 创建重要性分析器
    logger.info("🔍 创建核心重要性分析器...")
    analyzer = CoreImportanceAnalyzer(model, device)
    
    # 执行核心重要性分析
    logger.info("🎯 执行核心重要性分析...")
    importance_results = analyzer.analyze_layer_importance(
        val_loader, 
        methods=['fisher', 'gradient', 'ablation']
    )
    
    # 选择重要层
    logger.info("🎯 选择重要层...")
    selected_layers_fisher, fisher_scores = analyzer.select_important_layers(
        top_k=6, method='fisher'
    )
    selected_layers_gradient, gradient_scores = analyzer.select_important_layers(
        top_k=6, method='gradient'
    )
    selected_layers_ablation, ablation_scores = analyzer.select_important_layers(
        top_k=6, method='ablation'
    )
    
    # 创建压缩模型 (使用Fisher方法)
    logger.info("🏗️ 创建压缩模型...")
    compressed_model = analyzer.create_compressed_model(selected_layers_fisher, vocab_size)
    compressed_model.to(device)
    
    # 评估压缩效果
    logger.info("📊 评估压缩效果...")
    compression_results = analyzer.evaluate_compression(
        model, compressed_model, test_loader
    )
    
    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model_config': {
            'vocab_size': vocab_size,
            'embed_dim': 512,
            'num_heads': 8,
            'original_layers': 12,
            'compressed_layers': len(selected_layers_fisher)
        },
        'importance_analysis': importance_results,
        'layer_selection': {
            'fisher': {
                'selected_layers': selected_layers_fisher,
                'scores': fisher_scores
            },
            'gradient': {
                'selected_layers': selected_layers_gradient,  
                'scores': gradient_scores
            },
            'ablation': {
                'selected_layers': selected_layers_ablation,
                'scores': ablation_scores
            }
        },
        'compression_results': compression_results
    }
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    results_path = 'results/stage2_importance_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 阶段2结果已保存: {results_path}")
    
    # 可视化结果
    logger.info("📊 生成可视化图表...")
    create_visualization(importance_results, selected_layers_fisher)
    
    logger.info("🎉 阶段2完成！")
    logger.info("🔜 准备运行阶段3: 高级重要性分析方法")
    
    return results

def create_visualization(importance_results, selected_layers):
    """创建可视化图表"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Fisher信息重要性
    if 'fisher' in importance_results:
        fisher_scores = importance_results['fisher']
        layers = list(fisher_scores.keys())
        scores = list(fisher_scores.values())
        
        ax = axes[0, 0]
        bars = ax.bar(range(len(layers)), scores, alpha=0.7)
        ax.set_title('Fisher Information Layer Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Transformer Layers')
        ax.set_ylabel('Fisher Information Score')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f'L{i}' for i in range(len(layers))], rotation=45)
        
        # 高亮选中的层
        for i, layer in enumerate(layers):
            if layer in selected_layers:
                bars[i].set_color('red')
                bars[i].set_alpha(0.9)
    
    # 2. 梯度重要性
    if 'gradient' in importance_results:
        gradient_scores = importance_results['gradient']
        layers = list(gradient_scores.keys())
        scores = list(gradient_scores.values())
        
        ax = axes[0, 1]
        ax.bar(range(len(layers)), scores, alpha=0.7, color='green')
        ax.set_title('Gradient Norm Layer Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Transformer Layers')
        ax.set_ylabel('Gradient Norm Score')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f'L{i}' for i in range(len(layers))], rotation=45)
    
    # 3. 消融重要性
    if 'ablation' in importance_results:
        ablation_scores = importance_results['ablation']
        layers = list(ablation_scores.keys())
        scores = list(ablation_scores.values())
        
        ax = axes[1, 0]
        ax.bar(range(len(layers)), scores, alpha=0.7, color='orange')
        ax.set_title('Ablation Study Layer Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Transformer Layers')
        ax.set_ylabel('Performance Drop (Importance)')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f'L{i}' for i in range(len(layers))], rotation=45)
    
    # 4. 方法对比
    ax = axes[1, 1]
    methods = list(importance_results.keys())
    layer_indices = range(12)  # 假设12层
    
    x = np.arange(len(layer_indices))
    width = 0.25
    
    for i, method in enumerate(methods):
        scores = [importance_results[method].get(f'layer_{j}', 0) for j in layer_indices]
        ax.bar(x + i * width, scores, width, label=method.title(), alpha=0.7)
    
    ax.set_title('Layer Importance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Transformer Layers')
    ax.set_ylabel('Normalized Importance Score')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'L{i}' for i in layer_indices])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('results/stage2_importance_visualization.png', dpi=300, bbox_inches='tight')
    logger.info("📊 可视化图表已保存: results/stage2_importance_visualization.png")
    plt.close()

if __name__ == "__main__":
    main()
