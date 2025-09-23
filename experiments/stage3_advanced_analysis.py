#!/usr/bin/env python3
"""
阶段3：高级层重要性分析
实现SHAP、互信息、Layer Conductance、PII、Dropout不确定性、激活修补等高级方法
基于阶段1-2的稳定基础，使用真实Amazon数据
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
from sklearn.metrics import accuracy_score, classification_report, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 尝试导入SHAP (如果可用)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP未安装，将跳过SHAP分析")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

class HonestDataLoader:
    """诚实的数据加载器 - 与阶段1-2兼容"""
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
    """与阶段1-2兼容的稳定Transformer模型"""
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

class AdvancedImportanceAnalyzer:
    """高级重要性分析器 - SHAP、互信息、Layer Conductance等高级方法"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.importance_scores = {}
        self.analysis_results = {}
        
    def analyze_advanced_importance(self, dataloader, methods=['mutual_info', 'layer_conductance', 'pii', 'dropout_uncertainty', 'activation_patching']):
        """分析高级层重要性"""
        logger.info("🔬 开始高级层重要性分析...")
        
        results = {}
        
        if 'shap' in methods and SHAP_AVAILABLE:
            logger.info("🎯 计算SHAP值...")
            results['shap'] = self._compute_shap_importance(dataloader)
            
        if 'mutual_info' in methods:
            logger.info("📊 计算互信息重要性...")
            results['mutual_info'] = self._compute_mutual_info_importance(dataloader)
            
        if 'layer_conductance' in methods:
            logger.info("⚡ 计算Layer Conductance...")
            results['layer_conductance'] = self._compute_layer_conductance(dataloader)
            
        if 'pii' in methods:
            logger.info("🎯 计算参数影响指数(PII)...")
            results['pii'] = self._compute_parameter_influence_index(dataloader)
            
        if 'dropout_uncertainty' in methods:
            logger.info("🎲 计算Dropout不确定性...")
            results['dropout_uncertainty'] = self._compute_dropout_uncertainty(dataloader)
            
        if 'activation_patching' in methods:
            logger.info("🔧 执行激活修补分析...")
            results['activation_patching'] = self._compute_activation_patching(dataloader)
            
        self.analysis_results = results
        return results
    
    def _compute_shap_importance(self, dataloader):
        """计算SHAP重要性（如果可用）"""
        if not SHAP_AVAILABLE:
            logger.warning("⚠️ SHAP不可用，跳过SHAP分析")
            return {}
        
        # 简化版SHAP分析 - 使用层级别的SHAP
        logger.info("💡 使用简化SHAP方法进行层重要性分析...")
        
        # 获取一小批数据进行SHAP分析
        sample_data, sample_targets = next(iter(dataloader))
        sample_data = sample_data[:16].to(self.device)  # 只取16个样本
        sample_targets = sample_targets[:16].to(self.device)
        
        # 定义层级预测函数
        def layer_prediction_function(layer_masks):
            """基于层掩码的预测函数"""
            batch_predictions = []
            
            for mask in layer_masks:
                # 修改模型的层掩码
                original_layers = []
                for i, layer in enumerate(self.model.layers):
                    original_layers.append(layer.training)
                    if mask[i] == 0:
                        layer.eval()  # 禁用层
                    else:
                        layer.train()  # 启用层
                
                # 预测
                with torch.no_grad():
                    outputs = self.model(sample_data)
                    probs = F.softmax(outputs, dim=1)
                    batch_predictions.append(probs.cpu().numpy())
                
                # 恢复层状态
                for i, layer in enumerate(self.model.layers):
                    layer.train(original_layers[i])
            
            return np.array(batch_predictions)
        
        # 创建层掩码基线
        baseline_mask = np.ones((1, self.model.num_layers))
        
        # 简化的SHAP值计算
        shap_values = {}
        for layer_idx in range(self.model.num_layers):
            # 创建移除该层的掩码
            test_mask = baseline_mask.copy()
            test_mask[0, layer_idx] = 0
            
            # 计算基线和测试预测
            baseline_pred = layer_prediction_function(baseline_mask)[0]
            test_pred = layer_prediction_function(test_mask)[0]
            
            # SHAP值是预测差异
            shap_value = np.mean(np.abs(baseline_pred - test_pred))
            shap_values[f'layer_{layer_idx}'] = shap_value
        
        logger.info("✅ 简化SHAP分析完成")
        return shap_values
    
    def _compute_mutual_info_importance(self, dataloader):
        """计算互信息重要性"""
        self.model.eval()
        layer_activations = {f'layer_{i}': [] for i in range(self.model.num_layers)}
        targets_list = []
        
        # 收集层激活
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="收集层激活")):
                if batch_idx >= 20:  # 限制数据量
                    break
                    
                data, targets = data.to(self.device), targets.to(self.device)
                
                # 获取层输出
                _, layer_outputs = self.model(data, return_layer_outputs=True)
                
                # 存储激活统计
                for layer_idx, layer_output in enumerate(layer_outputs):
                    # 计算层激活的统计特征
                    activation_stats = torch.cat([
                        layer_output.mean(dim=(1, 2)),  # 平均激活
                        layer_output.std(dim=(1, 2)),   # 激活标准差
                        layer_output.max(dim=2)[0].max(dim=1)[0],  # 最大激活
                    ], dim=0)
                    
                    layer_activations[f'layer_{layer_idx}'].append(activation_stats.cpu().numpy())
                
                targets_list.extend(targets.cpu().numpy())
        
        # 计算互信息
        mutual_info_scores = {}
        for layer_name, activations in layer_activations.items():
            if activations:
                # 将激活展平
                activation_matrix = np.vstack(activations)
                
                # 计算每个特征维度与目标的互信息
                try:
                    mi_scores = mutual_info_classif(activation_matrix, targets_list[:len(activation_matrix)])
                    mutual_info_scores[layer_name] = np.mean(mi_scores)
                except Exception as e:
                    logger.warning(f"互信息计算失败 {layer_name}: {e}")
                    mutual_info_scores[layer_name] = 0.0
        
        logger.info("✅ 互信息分析完成")
        return mutual_info_scores
    
    def _compute_layer_conductance(self, dataloader):
        """计算Layer Conductance"""
        self.model.train()
        conductance_scores = {f'layer_{i}': [] for i in range(self.model.num_layers)}
        
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="Layer Conductance计算")):
            if batch_idx >= 30:  # 限制计算量
                break
                
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 注册钩子收集梯度
            layer_gradients = {}
            hooks = []
            
            def create_hook(layer_name):
                def hook_fn(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        layer_gradients[layer_name] = grad_output[0].clone()
                return hook_fn
            
            # 为每层注册钩子
            for i, layer in enumerate(self.model.layers):
                hook = layer.register_backward_hook(create_hook(f'layer_{i}'))
                hooks.append(hook)
            
            # 前向传播和反向传播
            self.model.zero_grad()
            outputs = self.model(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            # 计算conductance (激活值 × 梯度)
            with torch.no_grad():
                _, layer_outputs = self.model(data, return_layer_outputs=True)
                
                for i, layer_output in enumerate(layer_outputs):
                    layer_name = f'layer_{i}'
                    if layer_name in layer_gradients:
                        # Conductance = |activation × gradient|
                        conductance = torch.abs(layer_output * layer_gradients[layer_name])
                        conductance_score = conductance.mean().item()
                        conductance_scores[layer_name].append(conductance_score)
            
            # 清理钩子
            for hook in hooks:
                hook.remove()
        
        # 计算平均conductance
        layer_conductance = {}
        for layer_name, scores in conductance_scores.items():
            if scores:
                layer_conductance[layer_name] = np.mean(scores)
            else:
                layer_conductance[layer_name] = 0.0
        
        logger.info("✅ Layer Conductance计算完成")
        return layer_conductance
    
    def _compute_parameter_influence_index(self, dataloader):
        """计算参数影响指数(PII)"""
        self.model.train()
        pii_scores = {f'layer_{i}': [] for i in range(self.model.num_layers)}
        
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="PII计算")):
            if batch_idx >= 25:  # 限制计算量
                break
                
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 计算损失和梯度
            self.model.zero_grad()
            outputs = self.model(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            # 计算每层的PII
            for layer_idx in range(self.model.num_layers):
                layer_pii = 0
                param_count = 0
                
                for name, param in self.model.named_parameters():
                    if f'layers.{layer_idx}' in name and param.grad is not None:
                        # PII = |param × gradient|
                        param_influence = torch.abs(param * param.grad)
                        layer_pii += param_influence.sum().item()
                        param_count += param.numel()
                
                if param_count > 0:
                    pii_scores[f'layer_{layer_idx}'].append(layer_pii / param_count)
        
        # 计算平均PII
        layer_pii = {}
        for layer_name, scores in pii_scores.items():
            if scores:
                layer_pii[layer_name] = np.mean(scores)
            else:
                layer_pii[layer_name] = 0.0
        
        logger.info("✅ PII计算完成")
        return layer_pii
    
    def _compute_dropout_uncertainty(self, dataloader):
        """计算Dropout不确定性"""
        self.model.train()  # 保持dropout开启
        uncertainty_scores = {f'layer_{i}': [] for i in range(self.model.num_layers)}
        
        # 定义不同dropout率
        dropout_rates = [0.1, 0.3, 0.5]
        
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="Dropout不确定性计算")):
            if batch_idx >= 15:  # 限制计算量
                break
                
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 对每层测试不同dropout率的影响
            for layer_idx in range(self.model.num_layers):
                layer_uncertainties = []
                
                # 保存原始dropout率
                original_dropout = self.model.dropout.p
                
                for dropout_rate in dropout_rates:
                    # 设置新的dropout率
                    self.model.dropout.p = dropout_rate
                    
                    # 多次前向传播
                    predictions = []
                    for _ in range(5):  # 5次采样
                        with torch.no_grad():
                            outputs = self.model(data)
                            probs = F.softmax(outputs, dim=1)
                            predictions.append(probs.cpu().numpy())
                    
                    # 计算预测不确定性 (标准差)
                    predictions = np.stack(predictions)
                    uncertainty = np.mean(np.std(predictions, axis=0))
                    layer_uncertainties.append(uncertainty)
                
                # 恢复原始dropout率
                self.model.dropout.p = original_dropout
                
                # 层不确定性是不同dropout率的平均不确定性
                uncertainty_scores[f'layer_{layer_idx}'].append(np.mean(layer_uncertainties))
        
        # 计算平均不确定性
        layer_uncertainty = {}
        for layer_name, scores in uncertainty_scores.items():
            if scores:
                layer_uncertainty[layer_name] = np.mean(scores)
            else:
                layer_uncertainty[layer_name] = 0.0
        
        logger.info("✅ Dropout不确定性计算完成")
        return layer_uncertainty
    
    def _compute_activation_patching(self, dataloader):
        """计算激活修补重要性"""
        self.model.eval()
        patching_scores = {f'layer_{i}': [] for i in range(self.model.num_layers)}
        
        # 获取基准预测
        sample_data, sample_targets = next(iter(dataloader))
        sample_data = sample_data[:32].to(self.device)  # 限制样本数
        sample_targets = sample_targets[:32].to(self.device)
        
        with torch.no_grad():
            baseline_outputs = self.model(sample_data)
            baseline_probs = F.softmax(baseline_outputs, dim=1)
        
        # 对每层进行激活修补
        for layer_idx in range(self.model.num_layers):
            logger.info(f"激活修补第{layer_idx}层...")
            
            # 注册钩子进行激活修补
            def create_patching_hook(noise_scale=0.1):
                def hook_fn(module, input, output):
                    # 添加噪声到激活
                    noise = torch.randn_like(output) * noise_scale
                    return output + noise
                return hook_fn
            
            # 注册钩子
            hook = self.model.layers[layer_idx].register_forward_hook(create_patching_hook())
            
            # 修补后的预测
            with torch.no_grad():
                patched_outputs = self.model(sample_data)
                patched_probs = F.softmax(patched_outputs, dim=1)
            
            # 计算预测差异
            prob_diff = torch.abs(baseline_probs - patched_probs).mean().item()
            patching_scores[f'layer_{layer_idx}'].append(prob_diff)
            
            # 移除钩子
            hook.remove()
        
        # 整理结果
        layer_patching = {}
        for layer_name, scores in patching_scores.items():
            if scores:
                layer_patching[layer_name] = np.mean(scores)
            else:
                layer_patching[layer_name] = 0.0
        
        logger.info("✅ 激活修补分析完成")
        return layer_patching
    
    def combine_importance_scores(self, core_results, advanced_results):
        """综合核心和高级重要性分数"""
        logger.info("🔗 综合重要性分析结果...")
        
        # 收集所有方法的分数
        all_methods = {}
        all_methods.update(core_results)
        all_methods.update(advanced_results)
        
        # 标准化分数
        normalized_scores = {}
        for method, scores in all_methods.items():
            if scores:
                score_values = list(scores.values())
                if len(score_values) > 0 and max(score_values) > 0:
                    normalized_scores[method] = {
                        layer: score / max(score_values) 
                        for layer, score in scores.items()
                    }
                else:
                    normalized_scores[method] = scores
        
        # 计算综合分数
        layer_names = [f'layer_{i}' for i in range(self.model.num_layers)]
        comprehensive_scores = {}
        
        for layer_name in layer_names:
            layer_score = 0
            method_count = 0
            
            for method, scores in normalized_scores.items():
                if layer_name in scores:
                    layer_score += scores[layer_name]
                    method_count += 1
            
            if method_count > 0:
                comprehensive_scores[layer_name] = layer_score / method_count
            else:
                comprehensive_scores[layer_name] = 0.0
        
        logger.info("✅ 综合重要性分析完成")
        return comprehensive_scores, normalized_scores
    
    def select_optimal_layers(self, comprehensive_scores, target_compression=2.0):
        """选择最优层组合"""
        logger.info(f"🎯 选择最优层组合，目标压缩比: {target_compression:.1f}x")
        
        # 按重要性排序
        sorted_layers = sorted(comprehensive_scores.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # 计算目标层数
        original_layers = self.model.num_layers
        target_layers = max(1, int(original_layers / target_compression))
        
        # 选择top层
        selected_layers = [layer for layer, score in sorted_layers[:target_layers]]
        selected_scores = {layer: score for layer, score in sorted_layers[:target_layers]}
        
        logger.info(f"✅ 选择{len(selected_layers)}层进行{target_compression:.1f}x压缩:")
        for layer, score in sorted_layers[:target_layers]:
            logger.info(f"  {layer}: {score:.6f}")
        
        return selected_layers, selected_scores

def load_stage2_results():
    """加载阶段2的结果"""
    results_path = 'results/stage2_importance_analysis.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def prepare_data_from_previous_stages():
    """基于前面阶段准备数据"""
    # 重新加载和处理数据以保持一致性
    loader = HonestDataLoader()
    df = loader.load_real_data()
    
    # 创建正负例
    positive_samples = df[df['rating'] >= 4].sample(n=min(15000, len(df[df['rating'] >= 4])))
    negative_samples = df[df['rating'] <= 2].sample(n=min(15000, len(df[df['rating'] <= 2])))
    
    # 合并数据
    final_df = pd.concat([positive_samples, negative_samples]).sample(frac=1).reset_index(drop=True)
    final_df['label'] = (final_df['rating'] >= 4).astype(int)
    
    # 简单tokenization (与前面阶段保持一致)
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
    logger.info("🚀 开始阶段3：高级层重要性分析")
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 使用设备: {device}")
    
    # 加载阶段2结果
    stage2_results = load_stage2_results()
    if stage2_results:
        logger.info("✅ 阶段2结果已加载")
    
    # 准备数据
    logger.info("📊 准备数据...")
    train_loader, val_loader, test_loader, vocab_size = prepare_data_from_previous_stages()
    logger.info(f"✅ 数据准备完成，词汇表大小: {vocab_size}")
    
    # 创建模型 (与前面阶段相同的架构)
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
    logger.info("✅ 模型创建完成")
    
    # 创建高级重要性分析器
    logger.info("🔬 创建高级重要性分析器...")
    analyzer = AdvancedImportanceAnalyzer(model, device)
    
    # 执行高级重要性分析
    logger.info("🎯 执行高级重要性分析...")
    advanced_results = analyzer.analyze_advanced_importance(
        val_loader, 
        methods=['mutual_info', 'layer_conductance', 'pii', 'dropout_uncertainty', 'activation_patching']
    )
    
    # 综合分析结果
    if stage2_results and 'importance_analysis' in stage2_results:
        core_results = stage2_results['importance_analysis']
        logger.info("🔗 综合核心和高级分析结果...")
        comprehensive_scores, normalized_scores = analyzer.combine_importance_scores(
            core_results, advanced_results
        )
    else:
        logger.info("📊 仅使用高级分析结果...")
        comprehensive_scores = advanced_results.get('mutual_info', {})
        normalized_scores = advanced_results
    
    # 选择最优层
    logger.info("🎯 选择最优层组合...")
    selected_layers_2x, scores_2x = analyzer.select_optimal_layers(
        comprehensive_scores, target_compression=2.0
    )
    selected_layers_3x, scores_3x = analyzer.select_optimal_layers(
        comprehensive_scores, target_compression=3.0
    )
    selected_layers_4x, scores_4x = analyzer.select_optimal_layers(
        comprehensive_scores, target_compression=4.0
    )
    
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        """递归转换numpy类型为JSON可序列化类型"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # 保存结果
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model_config': {
            'vocab_size': vocab_size,
            'embed_dim': 512,
            'num_heads': 8,
            'original_layers': 12
        },
        'advanced_analysis': convert_to_serializable(advanced_results),
        'comprehensive_scores': convert_to_serializable(comprehensive_scores),
        'normalized_scores': convert_to_serializable(normalized_scores),
        'optimal_selections': {
            '2x_compression': {
                'selected_layers': selected_layers_2x,
                'scores': convert_to_serializable(scores_2x),
                'target_layers': len(selected_layers_2x)
            },
            '3x_compression': {
                'selected_layers': selected_layers_3x,
                'scores': convert_to_serializable(scores_3x),
                'target_layers': len(selected_layers_3x)
            },
            '4x_compression': {
                'selected_layers': selected_layers_4x,
                'scores': convert_to_serializable(scores_4x),
                'target_layers': len(selected_layers_4x)
            }
        }
    }
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    results_path = 'results/stage3_advanced_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 阶段3结果已保存: {results_path}")
    
    # 可视化结果
    logger.info("📊 生成可视化图表...")
    create_advanced_visualization(advanced_results, comprehensive_scores, normalized_scores)
    
    logger.info("🎉 阶段3完成！")
    logger.info("🔜 准备运行阶段4: 模型集成和最终评估")
    
    return results

def create_advanced_visualization(advanced_results, comprehensive_scores, normalized_scores):
    """创建高级分析可视化图表"""
    plt.style.use('seaborn-v0_8')
    
    # 创建大图表
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 高级方法对比热图
    ax1 = plt.subplot(3, 3, 1)
    methods_data = []
    method_names = []
    
    for method, scores in advanced_results.items():
        if scores:
            method_names.append(method.replace('_', ' ').title())
            layer_scores = [scores.get(f'layer_{i}', 0) for i in range(12)]
            methods_data.append(layer_scores)
    
    if methods_data:
        sns.heatmap(methods_data, xticklabels=[f'L{i}' for i in range(12)], 
                   yticklabels=method_names, annot=True, fmt='.3f', ax=ax1)
        ax1.set_title('Advanced Methods Heatmap', fontweight='bold')
    
    # 2. 互信息重要性
    if 'mutual_info' in advanced_results:
        ax2 = plt.subplot(3, 3, 2)
        mi_scores = advanced_results['mutual_info']
        layers = list(mi_scores.keys())
        scores = list(mi_scores.values())
        
        bars = ax2.bar(range(len(layers)), scores, alpha=0.7, color='purple')
        ax2.set_title('Mutual Information', fontweight='bold')
        ax2.set_xlabel('Layers')
        ax2.set_ylabel('MI Score')
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels([f'L{i}' for i in range(len(layers))])
    
    # 3. Layer Conductance
    if 'layer_conductance' in advanced_results:
        ax3 = plt.subplot(3, 3, 3)
        lc_scores = advanced_results['layer_conductance']
        layers = list(lc_scores.keys())
        scores = list(lc_scores.values())
        
        ax3.bar(range(len(layers)), scores, alpha=0.7, color='orange')
        ax3.set_title('Layer Conductance', fontweight='bold')
        ax3.set_xlabel('Layers')
        ax3.set_ylabel('Conductance')
        ax3.set_xticks(range(len(layers)))
        ax3.set_xticklabels([f'L{i}' for i in range(len(layers))])
    
    # 4. PII分数
    if 'pii' in advanced_results:
        ax4 = plt.subplot(3, 3, 4)
        pii_scores = advanced_results['pii']
        layers = list(pii_scores.keys())
        scores = list(pii_scores.values())
        
        ax4.bar(range(len(layers)), scores, alpha=0.7, color='green')
        ax4.set_title('Parameter Influence Index (PII)', fontweight='bold')
        ax4.set_xlabel('Layers')
        ax4.set_ylabel('PII Score')
        ax4.set_xticks(range(len(layers)))
        ax4.set_xticklabels([f'L{i}' for i in range(len(layers))])
    
    # 5. Dropout不确定性
    if 'dropout_uncertainty' in advanced_results:
        ax5 = plt.subplot(3, 3, 5)
        du_scores = advanced_results['dropout_uncertainty']
        layers = list(du_scores.keys())
        scores = list(du_scores.values())
        
        ax5.bar(range(len(layers)), scores, alpha=0.7, color='red')
        ax5.set_title('Dropout Uncertainty', fontweight='bold')
        ax5.set_xlabel('Layers')
        ax5.set_ylabel('Uncertainty')
        ax5.set_xticks(range(len(layers)))
        ax5.set_xticklabels([f'L{i}' for i in range(len(layers))])
    
    # 6. 激活修补
    if 'activation_patching' in advanced_results:
        ax6 = plt.subplot(3, 3, 6)
        ap_scores = advanced_results['activation_patching']
        layers = list(ap_scores.keys())
        scores = list(ap_scores.values())
        
        ax6.bar(range(len(layers)), scores, alpha=0.7, color='cyan')
        ax6.set_title('Activation Patching', fontweight='bold')
        ax6.set_xlabel('Layers')
        ax6.set_ylabel('Patching Effect')
        ax6.set_xticks(range(len(layers)))
        ax6.set_xticklabels([f'L{i}' for i in range(len(layers))])
    
    # 7. 综合重要性分数
    ax7 = plt.subplot(3, 3, 7)
    if comprehensive_scores:
        layers = list(comprehensive_scores.keys())
        scores = list(comprehensive_scores.values())
        
        bars = ax7.bar(range(len(layers)), scores, alpha=0.8, color='gold')
        ax7.set_title('Comprehensive Importance Scores', fontweight='bold', fontsize=12)
        ax7.set_xlabel('Layers')
        ax7.set_ylabel('Combined Score')
        ax7.set_xticks(range(len(layers)))
        ax7.set_xticklabels([f'L{i}' for i in range(len(layers))])
        
        # 高亮top-6层
        sorted_indices = np.argsort(scores)[::-1][:6]
        for idx in sorted_indices:
            bars[idx].set_color('red')
            bars[idx].set_alpha(0.9)
    
    # 8. 标准化分数对比
    ax8 = plt.subplot(3, 3, 8)
    if normalized_scores:
        methods = list(normalized_scores.keys())
        layer_indices = range(12)
        
        x = np.arange(len(layer_indices))
        width = 0.12
        
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'yellow', 'pink']
        
        for i, method in enumerate(methods[:6]):  # 限制方法数量
            if method in normalized_scores:
                scores = [normalized_scores[method].get(f'layer_{j}', 0) for j in layer_indices]
                ax8.bar(x + i * width, scores, width, label=method.replace('_', ' ').title(), 
                       alpha=0.7, color=colors[i % len(colors)])
        
        ax8.set_title('Normalized Scores Comparison', fontweight='bold')
        ax8.set_xlabel('Layers')
        ax8.set_ylabel('Normalized Score')
        ax8.set_xticks(x + width * 2)
        ax8.set_xticklabels([f'L{i}' for i in layer_indices])
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 9. 层选择结果
    ax9 = plt.subplot(3, 3, 9)
    if comprehensive_scores:
        # 显示不同压缩比的层选择
        compressions = [2.0, 3.0, 4.0]
        layer_counts = []
        
        for compression in compressions:
            target_layers = max(1, int(12 / compression))
            layer_counts.append(target_layers)
        
        bars = ax9.bar(range(len(compressions)), layer_counts, alpha=0.7, color='brown')
        ax9.set_title('Layer Selection for Different Compressions', fontweight='bold')
        ax9.set_xlabel('Compression Ratio')
        ax9.set_ylabel('Selected Layers')
        ax9.set_xticks(range(len(compressions)))
        ax9.set_xticklabels([f'{c:.1f}x' for c in compressions])
        
        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, layer_counts)):
            ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('results/stage3_advanced_visualization.png', dpi=300, bbox_inches='tight')
    logger.info("📊 高级分析可视化图表已保存: results/stage3_advanced_visualization.png")
    plt.close()

if __name__ == "__main__":
    main()
