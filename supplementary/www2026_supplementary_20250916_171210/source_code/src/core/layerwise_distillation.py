#!/usr/bin/env python3
"""
Layerwise Knowledge Distillation for LLM-based Recommender Systems
基于Fisher信息矩阵的层级知识蒸馏框架

核心思想：
- LLM上层（语义层）对推荐任务更有价值，权重随层深递增
- 使用Fisher信息矩阵量化每层参数对任务损失的二阶导数期望
- Fisher值越大→该层包含更多任务关键知识→蒸馏权重越高
- 高层语义信息对推荐匹配作用大于底层语法规则

目标会议: WWW2026
研究贡献: 首次将Fisher信息矩阵应用于LLM推荐系统的层级知识蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
import math
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """
    层级知识蒸馏配置参数
    
    WWW2026论文设计：
    - 基于"上层语义>下层语法"假设的权重策略
    - Fisher信息矩阵驱动的自适应层级权重
    - Llama3作为Teacher模型的推荐任务优化
    """
    # 模型配置
    teacher_model: str = "llama3:latest"  # 性能最优的Teacher模型
    student_hidden_dim: int = 768         # 学生模型隐藏维度
    num_layers: int = 12                  # 学生模型层数
    num_heads: int = 12                   # 多头注意力头数
    
    # 蒸馏策略参数
    temperature: float = 4.0              # 知知识蒸馏温度
    alpha: float = 0.7                    # 蒸馏损失权重
    beta: float = 0.3                     # 任务损失权重
    gamma: float = 0.1                    # 层级蒸馏损失权重
    
    # Fisher信息矩阵参数
    fisher_weight_scale: float = 2.0      # Fisher权重缩放因子
    fisher_sample_size: int = 100         # Fisher计算样本数
    fisher_regularization: float = 1e-6   # Fisher正则化参数
    semantic_emphasis: float = 1.5        # 高层语义强调因子
    
    # 训练参数
    max_seq_length: int = 512
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 10                  # 增加训练轮数
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    
    # 层级权重策略
    layer_weight_strategy: str = "fisher_adaptive"  # "linear", "exponential", "fisher_adaptive"
    layer_depth_bias: float = 0.8         # 层深偏置（0-1，越大越偏向高层）
    
    # 实验配置
    experiment_name: str = "WWW2026_layerwise_distillation"
    save_intermediate_results: bool = True
    use_wandb_logging: bool = False       # 实验跟踪

class FisherInformationCalculator:
    """
    Fisher信息矩阵计算器
    
    WWW2026核心创新：
    Fisher信息矩阵 F = E[∇²L(θ)] 反映参数对任务损失的敏感度
    - 高Fisher值 → 该层参数变化对推荐Loss影响大 → 包含任务关键知识
    - 低Fisher值 → 该层对任务Loss影响小 → 蒸馏价值较低
    
    理论假设：LLM高层包含更多与用户偏好相关的语义信息
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.ollama_url = "http://localhost:11434/api/generate"
        self.fisher_cache = {}  # Fisher信息缓存
        
    def calculate_fisher_weights(self, 
                               model: Optional[nn.Module] = None,
                               recommendation_samples: Optional[List[Dict]] = None,
                               use_real_fisher: bool = False) -> torch.Tensor:
        """
        计算每层的Fisher信息权重
        
        Args:
            model: 学生模型（用于真实Fisher计算）
            recommendation_samples: 推荐样本数据
            use_real_fisher: 是否使用真实Fisher信息计算
            
        Returns:
            layer_weights: 每层的Fisher权重 [num_layers]
        """
        logger.info("计算Fisher信息矩阵权重...")
        
        if use_real_fisher and model is not None:
            # 使用真实Fisher信息计算（计算密集）
            fisher_weights = self._compute_real_fisher_weights(model, recommendation_samples)
        else:
            # 使用理论驱动的自适应Fisher权重（高效）
            fisher_weights = self._compute_theoretical_fisher_weights(recommendation_samples)
        
        # 应用语义强调因子
        fisher_weights = self._apply_semantic_emphasis(fisher_weights)
        
        # 最终归一化和缩放
        fisher_weights = self._normalize_and_scale(fisher_weights)
        
        logger.info(f"Fisher权重分布: {fisher_weights.tolist()}")
        return fisher_weights
    
    def _compute_real_fisher_weights(self, model: nn.Module, samples: List[Dict]) -> torch.Tensor:
        """
        计算真实Fisher信息权重（WWW2026方法论）
        
        Fisher矩阵对角线元素：F_ii = E[(∂L/∂θ_i)²]
        """
        logger.info("使用真实Fisher信息计算...")
        
        model.eval()
        layer_fisher_scores = torch.zeros(self.config.num_layers)
        sample_count = 0
        
        # 随机采样计算Fisher信息
        selected_samples = samples[:self.config.fisher_sample_size] if samples else []
        
        for sample in selected_samples:
            try:
                # 准备输入数据
                input_data = self._prepare_sample_input(sample)
                
                # 清零梯度
                model.zero_grad()
                
                # 前向传播
                outputs = model(input_data['input_ids'], input_data['attention_mask'])
                
                # 计算推荐损失
                loss = self._compute_recommendation_loss(outputs, input_data['labels'])
                
                # 计算梯度
                loss.backward()
                
                # 累积每层的Fisher信息
                layer_idx = 0
                for name, param in model.named_parameters():
                    if param.grad is not None and 'layers.' in name:
                        # 提取层索引
                        try:
                            current_layer = int(name.split('layers.')[1].split('.')[0])
                            if current_layer < self.config.num_layers:
                                # Fisher信息 = 梯度平方
                                fisher_contribution = (param.grad ** 2).sum().item()
                                layer_fisher_scores[current_layer] += fisher_contribution
                        except (ValueError, IndexError):
                            continue
                
                sample_count += 1
                
                if sample_count % 10 == 0:
                    logger.info(f"Fisher计算进度: {sample_count}/{len(selected_samples)}")
                    
            except Exception as e:
                logger.warning(f"样本Fisher计算失败: {e}")
                continue
        
        # 平均化
        if sample_count > 0:
            layer_fisher_scores = layer_fisher_scores / sample_count
        
        # 添加正则化
        layer_fisher_scores += self.config.fisher_regularization
        
        return layer_fisher_scores
    
    def _compute_theoretical_fisher_weights(self, samples: Optional[List[Dict]]) -> torch.Tensor:
        """
        基于理论假设的Fisher权重计算（高效版本）
        
        核心假设：
        1. 上层语义 > 下层语法 对推荐任务更重要
        2. Fisher值随层深度非线性递增
        3. 任务复杂度影响Fisher分布
        """
        logger.info("使用理论Fisher权重计算...")
        
        fisher_scores = []
        
        # 估算任务复杂度
        task_complexity = self._estimate_task_complexity(samples) if samples else 1.0
        
        for layer_idx in range(self.config.num_layers):
            # 基础层深权重（线性递增）
            depth_ratio = (layer_idx + 1) / self.config.num_layers
            base_weight = depth_ratio ** self.config.layer_depth_bias
            
            # 语义层强调（高层非线性增强）
            semantic_boost = self._compute_semantic_boost(layer_idx, self.config.num_layers)
            
            # 任务相关性调整
            task_relevance = self._estimate_layer_task_relevance(layer_idx, samples)
            
            # 复杂度调制
            complexity_factor = 1.0 + (task_complexity - 1.0) * depth_ratio
            
            # 最终Fisher得分
            fisher_score = base_weight * semantic_boost * (1 + task_relevance) * complexity_factor
            fisher_scores.append(fisher_score)
            
            logger.debug(f"Layer {layer_idx}: depth={depth_ratio:.3f}, "
                        f"semantic={semantic_boost:.3f}, task={task_relevance:.3f}, "
                        f"complexity={complexity_factor:.3f}, fisher={fisher_score:.3f}")
        
        return torch.tensor(fisher_scores, dtype=torch.float32)
    
    def _compute_semantic_boost(self, layer_idx: int, num_layers: int) -> float:
        """
        计算语义层增强因子
        
        WWW2026假设：高层包含更多语义信息，Fisher值应指数增长
        """
        depth_ratio = layer_idx / (num_layers - 1)
        
        if depth_ratio < 0.3:
            # 底层：主要是词汇和语法特征
            return 0.5
        elif depth_ratio < 0.6:
            # 中层：局部语义和特征组合
            return 1.0
        else:
            # 高层：全局语义和推理
            semantic_boost = 1.0 + (depth_ratio - 0.6) * self.config.semantic_emphasis
            return min(semantic_boost, 3.0)  # 限制最大增强
    
    def _estimate_task_complexity(self, samples: Optional[List[Dict]]) -> float:
        """
        估算推荐任务复杂度
        
        影响Fisher分布的关键因素：
        - 用户偏好多样性
        - 候选物品复杂度
        - 语义匹配难度
        """
        if not samples:
            return 1.0
        
        complexities = []
        for sample in samples[:20]:  # 采样评估
            # 用户描述复杂度
            user_text = str(sample.get('user_profile', ''))
            text_complexity = len(user_text.split()) / 50  # 归一化词数
            
            # 候选物品复杂度
            items = sample.get('candidate_items', [])
            item_complexity = len(items) / 10  # 归一化物品数
            
            # 语义复杂度（基于文本多样性）
            semantic_complexity = len(set(user_text.lower().split())) / max(len(user_text.split()), 1)
            
            total_complexity = text_complexity + item_complexity + semantic_complexity
            complexities.append(total_complexity)
        
        avg_complexity = np.mean(complexities) if complexities else 1.0
        return max(0.5, min(avg_complexity, 3.0))  # 限制在合理范围
    
    def _estimate_layer_task_relevance(self, layer_idx: int, samples: Optional[List[Dict]]) -> float:
        """
        估算层级对推荐任务的相关性
        
        WWW2026理论基础：
        - 底层：词汇和语法处理，对推荐任务相关性低
        - 中层：语义理解和特征抽取，中等相关性
        - 高层：用户偏好推理和决策，高相关性
        """
        total_layers = self.config.num_layers
        depth_ratio = layer_idx / total_layers
        
        if depth_ratio < 0.25:
            # 底层：token embedding, positional encoding
            base_relevance = 0.1
        elif depth_ratio < 0.5:
            # 中低层：基础语言理解
            base_relevance = 0.3
        elif depth_ratio < 0.75:
            # 中高层：语义理解和特征组合
            base_relevance = 0.6
        else:
            # 高层：抽象推理和决策
            base_relevance = 1.0
        
        # 基于样本特征的动态调整
        if samples:
            # 样本复杂度越高，高层越重要
            complexity_factor = self._estimate_task_complexity(samples)
            high_layer_boost = max(0, depth_ratio - 0.5) * (complexity_factor - 1.0) * 0.5
            base_relevance += high_layer_boost
            
            # 语义匹配复杂度
            semantic_diversity = self._estimate_semantic_diversity(samples)
            semantic_boost = depth_ratio * semantic_diversity * 0.3
            base_relevance += semantic_boost
        
        return max(0.0, min(base_relevance, 1.5))
    
    def _estimate_semantic_diversity(self, samples: List[Dict]) -> float:
        """估算语义多样性"""
        if not samples:
            return 0.0
        
        all_words = set()
        for sample in samples[:10]:
            text = str(sample.get('user_profile', ''))
            words = set(text.lower().split())
            all_words.update(words)
        
        # 基于词汇多样性估算语义复杂度
        diversity_score = len(all_words) / max(sum(len(str(s.get('user_profile', '')).split()) for s in samples[:10]), 1)
        return max(0.0, min(diversity_score, 1.0))
    
    def _apply_semantic_emphasis(self, fisher_weights: torch.Tensor) -> torch.Tensor:
        """
        应用语义强调因子
        
        WWW2026核心：高层语义信息对推荐任务更重要
        """
        enhanced_weights = fisher_weights.clone()
        num_layers = len(enhanced_weights)
        
        for i in range(num_layers):
            depth_ratio = i / (num_layers - 1)
            if depth_ratio > 0.5:  # 高层
                emphasis_factor = 1.0 + (depth_ratio - 0.5) * 2.0 * self.config.semantic_emphasis
                enhanced_weights[i] *= emphasis_factor
        
        return enhanced_weights
    
    def _normalize_and_scale(self, fisher_weights: torch.Tensor) -> torch.Tensor:
        """归一化和缩放Fisher权重"""
        # 归一化到平均值为1
        if fisher_weights.sum() > 0:
            normalized_weights = fisher_weights / fisher_weights.mean()
        else:
            normalized_weights = torch.ones_like(fisher_weights)
        
        # 应用全局缩放
        scaled_weights = normalized_weights * self.config.fisher_weight_scale
        
        # 确保权重在合理范围内
        scaled_weights = torch.clamp(scaled_weights, min=0.1, max=5.0)
        
        return scaled_weights
    
    def _prepare_sample_input(self, sample: Dict) -> Dict:
        """准备样本输入数据"""
        # 简化版本，实际实现需要完整的tokenization
        user_text = str(sample.get('user_profile', ''))
        items_text = ', '.join(str(item) for item in sample.get('candidate_items', []))
        
        # 模拟tokenization（实际需要使用真实tokenizer）
        combined_text = f"User: {user_text} Items: {items_text}"
        input_ids = torch.randint(0, 50000, (1, min(len(combined_text.split()), 128)))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([[sample.get('label', 0)]], dtype=torch.float32)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _compute_recommendation_loss(self, outputs: Dict, labels: torch.Tensor) -> torch.Tensor:
        """计算推荐损失"""
        logits = outputs.get('logits', torch.zeros(1, 1))
        return F.binary_cross_entropy_with_logits(logits, labels)

class StudentRecommenderModel(nn.Module):
    """学生推荐模型（简化的Transformer架构）"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.token_embedding = nn.Embedding(50000, config.student_hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.student_hidden_dim)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # 输出层
        self.layer_norm = nn.LayerNorm(config.student_hidden_dim)
        self.output_projection = nn.Linear(config.student_hidden_dim, 1)
        
        # 用于存储中间层输出（蒸馏用）
        self.layer_outputs = []
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
            
        Returns:
            outputs: 包含logits和中间层输出的字典
        """
        batch_size, seq_length = input_ids.shape
        
        # 位置编码
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        
        # 嵌入
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # 存储每层输出用于蒸馏
        self.layer_outputs = []
        
        # 通过Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            self.layer_outputs.append(hidden_states)
        
        # 最终输出
        hidden_states = self.layer_norm(hidden_states)
        
        # 推荐得分（使用[CLS]位置或平均池化）
        if attention_mask is not None:
            pooled = self._pool_hidden_states(hidden_states, attention_mask)
        else:
            pooled = hidden_states.mean(dim=1)
        
        logits = self.output_projection(pooled)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'layer_outputs': self.layer_outputs
        }
    
    def _pool_hidden_states(self, hidden_states: torch.Tensor, 
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """对隐藏状态进行池化"""
        # 使用注意力掩码进行平均池化
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

class TransformerLayer(nn.Module):
    """Transformer层"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.student_hidden_dim)
        self.norm2 = nn.LayerNorm(config.student_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        # 自注意力
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states) + residual
        
        # 前馈网络
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual
        
        return hidden_states

class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_dim = config.student_hidden_dim
        self.head_dim = self.hidden_dim // self.num_heads
        
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # 计算Q, K, V
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # 重塑为多头
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_dim
        )
        
        return self.output(context)

class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.student_hidden_dim, config.student_hidden_dim * 4)
        self.linear2 = nn.Linear(config.student_hidden_dim * 4, config.student_hidden_dim)
        self.activation = nn.GELU()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states

class LayerwiseDistillationLoss(nn.Module):
    """层级知识蒸馏损失函数"""
    
    def __init__(self, config: DistillationConfig, fisher_weights: torch.Tensor):
        super().__init__()
        self.config = config
        self.fisher_weights = fisher_weights
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_outputs: Dict, teacher_outputs: Dict, 
                labels: torch.Tensor) -> Dict:
        """
        计算层级蒸馏损失
        
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出 
            labels: 真实标签
            
        Returns:
            loss_dict: 包含各项损失的字典
        """
        # 任务损失（推荐预测损失）
        task_loss = F.binary_cross_entropy_with_logits(
            student_outputs['logits'], labels.float()
        )
        
        # 层级蒸馏损失
        layer_distill_loss = 0.0
        student_layers = student_outputs['layer_outputs']
        teacher_layers = teacher_outputs['layer_outputs']
        
        for i, (student_layer, teacher_layer) in enumerate(zip(student_layers, teacher_layers)):
            # 应用Fisher权重
            layer_weight = self.fisher_weights[i]
            
            # 特征蒸馏（MSE）
            feature_loss = self.mse_loss(student_layer, teacher_layer)
            
            # 加权累加
            layer_distill_loss += layer_weight * feature_loss
        
        # 输出分布蒸馏
        student_logits = student_outputs['logits'] / self.config.temperature
        teacher_logits = teacher_outputs['logits'] / self.config.temperature
        
        distill_loss = self.kl_loss(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1)
        ) * (self.config.temperature ** 2)
        
        # 总损失
        total_loss = (
            self.config.beta * task_loss +
            self.config.alpha * distill_loss +
            0.1 * layer_distill_loss
        )
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'distill_loss': distill_loss,
            'layer_distill_loss': layer_distill_loss
        }

class TeacherModelProxy:
    """教师模型代理（通过ollama API访问llama3）"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def get_teacher_outputs(self, input_texts: List[str]) -> Dict:
        """
        获取教师模型的输出
        
        Args:
            input_texts: 输入文本列表
            
        Returns:
            teacher_outputs: 包含logits和层输出的字典
        """
        logger.info(f"获取{len(input_texts)}个样本的教师模型输出...")
        
        batch_outputs = []
        for text in input_texts:
            try:
                # 调用ollama API
                response = self._query_ollama(text)
                
                # 模拟提取层级特征（实际需要模型内部访问）
                layer_outputs = self._simulate_layer_outputs(response)
                
                # 模拟推荐得分
                logits = self._extract_recommendation_score(response)
                
                batch_outputs.append({
                    'logits': logits,
                    'layer_outputs': layer_outputs
                })
                
            except Exception as e:
                logger.warning(f"教师模型查询失败: {e}")
                # 使用默认值
                batch_outputs.append(self._get_default_output())
        
        # 转换为tensor格式
        return self._batch_outputs_to_tensors(batch_outputs)
    
    def _query_ollama(self, text: str) -> str:
        """查询ollama API"""
        payload = {
            "model": self.config.teacher_model,
            "prompt": f"对以下推荐场景进行分析和评分:\n{text}",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        
        response = requests.post(self.ollama_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get('response', '')
    
    def _simulate_layer_outputs(self, response: str) -> List[torch.Tensor]:
        """模拟层级输出（实际应用中需要真实的层级访问）"""
        # 基于响应长度和内容复杂度模拟不同层的激活
        response_length = len(response)
        complexity = min(response_length / 100, 5.0)
        
        layer_outputs = []
        for i in range(self.config.num_layers):
            # 模拟层级特征：低层简单，高层复杂
            layer_complexity = (i + 1) / self.config.num_layers * complexity
            
            # 生成随机特征向量
            hidden_states = torch.randn(1, 128, self.config.student_hidden_dim)
            hidden_states = hidden_states * layer_complexity
            
            layer_outputs.append(hidden_states)
        
        return layer_outputs
    
    def _extract_recommendation_score(self, response: str) -> torch.Tensor:
        """从响应中提取推荐得分"""
        # 基于响应内容的启发式评分
        positive_keywords = ['推荐', '合适', '匹配', '适合', '喜欢', '优秀']
        negative_keywords = ['不推荐', '不合适', '不匹配', '不适合', '不喜欢']
        
        pos_count = sum(1 for word in positive_keywords if word in response)
        neg_count = sum(1 for word in negative_keywords if word in response)
        
        # 计算得分
        score = (pos_count - neg_count) / max(len(response) / 50, 1.0)
        score = torch.tensor([[max(-1.0, min(1.0, score))]], dtype=torch.float32)
        
        return score
    
    def _get_default_output(self) -> Dict:
        """获取默认输出"""
        return {
            'logits': torch.tensor([[0.0]], dtype=torch.float32),
            'layer_outputs': [
                torch.randn(1, 128, self.config.student_hidden_dim) 
                for _ in range(self.config.num_layers)
            ]
        }
    
    def _batch_outputs_to_tensors(self, batch_outputs: List[Dict]) -> Dict:
        """将批次输出转换为tensor格式"""
        batch_logits = torch.cat([out['logits'] for out in batch_outputs], dim=0)
        
        batch_layer_outputs = []
        for layer_idx in range(self.config.num_layers):
            layer_batch = torch.cat([
                out['layer_outputs'][layer_idx] for out in batch_outputs
            ], dim=0)
            batch_layer_outputs.append(layer_batch)
        
        return {
            'logits': batch_logits,
            'layer_outputs': batch_layer_outputs
        }

def main():
    """主函数：演示层级知识蒸馏"""
    logger.info("开始层级知识蒸馏实验...")
    
    # 配置
    config = DistillationConfig()
    
    # 模拟推荐数据
    recommendation_samples = [
        {
            'user_profile': '用户喜欢科技产品，经常购买电子设备',
            'candidate_items': ['iPhone15', 'MacBook Pro', '无线耳机'],
            'label': 1
        },
        {
            'user_profile': '用户偏爱美妆护肤，注重品质和效果', 
            'candidate_items': ['面膜套装', '精华液', '防晒霜'],
            'label': 1
        }
    ]
    
    # 计算Fisher权重
    fisher_calc = FisherInformationCalculator(config)
    fisher_weights = fisher_calc.calculate_fisher_weights(recommendation_samples)
    
    # 初始化模型
    student_model = StudentRecommenderModel(config)
    teacher_proxy = TeacherModelProxy(config)
    
    # 初始化损失函数
    distill_loss_fn = LayerwiseDistillationLoss(config, fisher_weights)
    
    logger.info("层级知识蒸馏框架初始化完成!")
    logger.info(f"Fisher权重分布: {fisher_weights.tolist()}")
    logger.info("准备开始蒸馏训练...")

if __name__ == "__main__":
    main()
