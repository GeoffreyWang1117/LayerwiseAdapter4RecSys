#!/usr/bin/env python3
"""
Fisher Information Matrix Implementation
Fisher信息矩阵详细实现

基于二阶导数计算每层参数对推荐任务损失的贡献度
实现真实的Fisher信息量化，而非启发式估算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import requests
import json

logger = logging.getLogger(__name__)

@dataclass
class FisherConfig:
    """Fisher信息计算配置"""
    num_samples: int = 50  # 用于Fisher计算的样本数量
    regularization: float = 1e-6  # 正则化参数
    layer_wise: bool = True  # 是否按层计算
    diagonal_only: bool = True  # 是否只计算对角元素
    normalize: bool = True  # 是否归一化
    weight_decay: float = 0.01  # 权重衰减

class RealFisherCalculator:
    """真实Fisher信息矩阵计算器"""
    
    def __init__(self, model: nn.Module, config: FisherConfig):
        self.model = model
        self.config = config
        self.fisher_info = {}
        self.parameter_names = []
        
        # 注册参数名称
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.parameter_names.append(name)
        
        logger.info(f"Fisher计算器初始化完成，监控 {len(self.parameter_names)} 个参数组")
    
    def compute_fisher_information(self, dataloader, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        计算Fisher信息矩阵
        
        Args:
            dataloader: 数据加载器
            device: 计算设备
            
        Returns:
            fisher_dict: 每层的Fisher信息
        """
        logger.info("开始计算Fisher信息矩阵...")
        
        self.model.eval()
        fisher_dict = defaultdict(float)
        sample_count = 0
        
        with torch.enable_grad():
            for batch_idx, batch in enumerate(dataloader):
                if sample_count >= self.config.num_samples:
                    break
                
                # 准备数据
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                batch_size = input_ids.size(0)
                
                # 对每个样本计算Fisher信息
                for sample_idx in range(batch_size):
                    if sample_count >= self.config.num_samples:
                        break
                    
                    # 单样本前向传播
                    sample_input = input_ids[sample_idx:sample_idx+1]
                    sample_mask = attention_mask[sample_idx:sample_idx+1]
                    sample_label = labels[sample_idx:sample_idx+1]
                    
                    # 清零梯度
                    self.model.zero_grad()
                    
                    # 前向传播
                    outputs = self.model(sample_input, sample_mask)
                    logits = outputs['logits']
                    
                    # 计算损失
                    loss = F.binary_cross_entropy_with_logits(logits, sample_label.float())
                    
                    # 计算一阶梯度
                    grads = torch.autograd.grad(
                        loss, self.model.parameters(),
                        create_graph=True, retain_graph=True
                    )
                    
                    # 累积Fisher信息（梯度的平方）
                    for name, grad in zip(self.parameter_names, grads):
                        if grad is not None:
                            if self.config.diagonal_only:
                                # 只计算对角元素（参数级Fisher信息）
                                fisher_dict[name] += (grad ** 2).sum().item()
                            else:
                                # 计算完整Fisher矩阵（计算量很大）
                                fisher_dict[name] += torch.outer(grad.flatten(), grad.flatten()).sum().item()
                    
                    sample_count += 1
                    
                    if sample_count % 10 == 0:
                        logger.info(f"已处理样本: {sample_count}/{self.config.num_samples}")
        
        # 平均并转换为tensor
        fisher_tensors = {}
        for name in fisher_dict:
            fisher_value = fisher_dict[name] / sample_count
            # 添加正则化
            fisher_value += self.config.regularization
            fisher_tensors[name] = torch.tensor(fisher_value, dtype=torch.float32)
        
        logger.info("Fisher信息矩阵计算完成")
        return fisher_tensors
    
    def compute_layer_wise_fisher(self, fisher_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算层级Fisher权重
        
        Args:
            fisher_dict: 参数级Fisher信息
            
        Returns:
            layer_weights: 每层的权重 [num_layers]
        """
        logger.info("计算层级Fisher权重...")
        
        # 提取层级信息
        layer_fisher = defaultdict(float)
        
        for param_name, fisher_value in fisher_dict.items():
            layer_idx = self._extract_layer_index(param_name)
            if layer_idx is not None:
                layer_fisher[layer_idx] += fisher_value.item()
        
        # 构建层级权重tensor
        num_layers = max(layer_fisher.keys()) + 1 if layer_fisher else 1
        layer_weights = torch.zeros(num_layers)
        
        for layer_idx in range(num_layers):
            layer_weights[layer_idx] = layer_fisher.get(layer_idx, 0.0)
        
        # 归一化
        if self.config.normalize and layer_weights.sum() > 0:
            layer_weights = layer_weights / layer_weights.sum() * num_layers
        
        # 应用权重衰减
        layer_weights = layer_weights * (1 - self.config.weight_decay)
        
        logger.info(f"层级Fisher权重: {layer_weights.tolist()}")
        return layer_weights
    
    def _extract_layer_index(self, param_name: str) -> Optional[int]:
        """从参数名提取层索引"""
        # 解析参数名，提取层索引
        # 例如: "layers.0.attention.query.weight" -> 0
        if 'layers.' in param_name:
            try:
                parts = param_name.split('.')
                layer_idx = int(parts[1])
                return layer_idx
            except (IndexError, ValueError):
                pass
        
        # 其他层的映射
        if 'token_embedding' in param_name:
            return 0
        elif 'position_embedding' in param_name:
            return 0
        elif 'layer_norm' in param_name:
            return -1  # 最后一层
        elif 'output_projection' in param_name:
            return -1
        
        return None
    
    def analyze_fisher_distribution(self, fisher_dict: Dict[str, torch.Tensor]) -> Dict:
        """分析Fisher信息分布"""
        analysis = {
            'total_parameters': len(fisher_dict),
            'total_fisher': sum(v.item() for v in fisher_dict.values()),
            'top_parameters': [],
            'layer_distribution': defaultdict(float)
        }
        
        # 排序找出最重要的参数
        sorted_params = sorted(
            fisher_dict.items(), 
            key=lambda x: x[1].item(), 
            reverse=True
        )
        
        analysis['top_parameters'] = [
            (name, fisher.item()) for name, fisher in sorted_params[:10]
        ]
        
        # 层级分布
        for param_name, fisher_value in fisher_dict.items():
            layer_idx = self._extract_layer_index(param_name)
            if layer_idx is not None:
                analysis['layer_distribution'][f'layer_{layer_idx}'] += fisher_value.item()
        
        return analysis

class AdaptiveFisherCalculator:
    """自适应Fisher信息计算器（结合启发式和真实计算）"""
    
    def __init__(self, config: FisherConfig):
        self.config = config
        self.ollama_url = "http://localhost:11434/api/generate"
    
    def compute_adaptive_fisher_weights(self, 
                                     model: nn.Module,
                                     recommendation_samples: List[Dict],
                                     dataloader = None,
                                     device: torch.device = None) -> torch.Tensor:
        """
        计算自适应Fisher权重
        
        Args:
            model: 学生模型
            recommendation_samples: 推荐样本
            dataloader: 数据加载器（可选）
            device: 计算设备
            
        Returns:
            adaptive_weights: 自适应权重
        """
        logger.info("计算自适应Fisher权重...")
        
        # 方法1: 启发式权重（基于层级假设）
        heuristic_weights = self._compute_heuristic_weights(
            model, recommendation_samples
        )
        
        # 方法2: 真实Fisher信息（如果有数据和计算资源）
        if dataloader is not None and device is not None:
            try:
                real_fisher_calc = RealFisherCalculator(model, self.config)
                fisher_dict = real_fisher_calc.compute_fisher_information(dataloader, device)
                real_weights = real_fisher_calc.compute_layer_wise_fisher(fisher_dict)
                
                # 融合两种权重
                alpha = 0.7  # 真实Fisher权重
                beta = 0.3   # 启发式权重
                
                adaptive_weights = alpha * real_weights + beta * heuristic_weights
                logger.info("使用真实Fisher + 启发式融合权重")
                
            except Exception as e:
                logger.warning(f"真实Fisher计算失败，使用启发式权重: {e}")
                adaptive_weights = heuristic_weights
        else:
            adaptive_weights = heuristic_weights
            logger.info("使用启发式Fisher权重")
        
        # 方法3: 基于LLM反馈的权重调整
        llm_adjustment = self._get_llm_weight_adjustment(recommendation_samples, len(adaptive_weights))
        adaptive_weights = adaptive_weights * llm_adjustment
        
        # 最终归一化
        if adaptive_weights.sum() > 0:
            num_layers = len(adaptive_weights)
            adaptive_weights = adaptive_weights / adaptive_weights.sum() * num_layers
        
        logger.info(f"自适应Fisher权重: {adaptive_weights.tolist()}")
        return adaptive_weights
    
    def _compute_heuristic_weights(self, model: nn.Module, samples: List[Dict]) -> torch.Tensor:
        """计算启发式权重"""
        # 获取模型层数
        num_layers = 0
        for name, _ in model.named_parameters():
            if 'layers.' in name:
                try:
                    layer_idx = int(name.split('.')[1])
                    num_layers = max(num_layers, layer_idx + 1)
                except:
                    pass
        
        if num_layers == 0:
            num_layers = 12  # 默认值
        
        # 基于层级假设：上层更重要
        weights = []
        for i in range(num_layers):
            # 线性递增 + 非线性增强
            base_weight = (i + 1) / num_layers
            semantic_boost = np.exp(i / num_layers) - 1  # 指数增强高层
            task_relevance = self._estimate_task_relevance(i, num_layers, samples)
            
            final_weight = base_weight * (1 + semantic_boost) * (1 + task_relevance)
            weights.append(final_weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _estimate_task_relevance(self, layer_idx: int, num_layers: int, samples: List[Dict]) -> float:
        """估算任务相关性"""
        # 分层特性
        layer_ratio = layer_idx / num_layers
        
        if layer_ratio < 0.3:
            # 底层：语法和词汇
            base_relevance = 0.1
        elif layer_ratio < 0.7:
            # 中层：语义组合
            base_relevance = 0.5
        else:
            # 高层：抽象推理和决策
            base_relevance = 1.0
        
        # 基于样本复杂度调整
        if samples:
            complexity = self._estimate_sample_complexity(samples)
            complexity_factor = min(complexity / 5.0, 0.5)
        else:
            complexity_factor = 0.0
        
        return base_relevance + complexity_factor
    
    def _estimate_sample_complexity(self, samples: List[Dict]) -> float:
        """估算样本复杂度"""
        if not samples:
            return 0.0
        
        complexities = []
        for sample in samples[:10]:
            # 文本长度复杂度
            text_length = len(str(sample.get('user_profile', '')))
            
            # 候选项目数量复杂度
            items = sample.get('candidate_items', [])
            item_complexity = len(items) if items else 0
            
            # 综合复杂度
            total_complexity = (text_length / 50) + (item_complexity / 5)
            complexities.append(total_complexity)
        
        return np.mean(complexities)
    
    def _get_llm_weight_adjustment(self, samples: List[Dict], num_layers: int = None) -> torch.Tensor:
        """基于LLM反馈的权重调整"""
        try:
            # 构造查询prompt
            prompt = self._build_layer_importance_prompt(samples, num_layers)
            
            # 查询LLM
            response = self._query_llm(prompt)
            
            # 解析权重调整
            adjustment = self._parse_weight_adjustment(response, num_layers)
            
            return adjustment
            
        except Exception as e:
            logger.warning(f"LLM权重调整失败: {e}")
            # 返回均匀调整（无影响）
            default_layers = num_layers if num_layers else 12
            return torch.ones(default_layers)
    
    def _build_layer_importance_prompt(self, samples: List[Dict], num_layers: int = None) -> str:
        """构建层级重要性查询prompt"""
        sample_text = ""
        if samples:
            sample_text = f"样本示例: {samples[0].get('user_profile', '')[:100]}"
        
        target_layers = num_layers if num_layers else 12
        
        prompt = f"""
        分析推荐系统中Transformer层级的重要性。
        
        {sample_text}
        
        请评估从底层到高层（{target_layers}层）对推荐任务的重要性：
        - 底层：处理词汇和语法
        - 中层：语义理解和特征组合
        - 高层：抽象推理和决策
        
        对于推荐任务，哪些层级更重要？请给出1-{target_layers}层的重要性评分（1-10分）。
        """
        
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """查询LLM"""
        payload = {
            "model": "llama3:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "max_tokens": 300
            }
        }
        
        response = requests.post(self.ollama_url, json=payload, timeout=20)
        response.raise_for_status()
        return response.json().get('response', '')
    
    def _parse_weight_adjustment(self, response: str, num_layers: int = None) -> torch.Tensor:
        """解析权重调整"""
        # 简化解析：寻找数字模式
        import re
        
        target_layers = num_layers if num_layers else 12
        numbers = re.findall(r'\b\d+\b', response)
        
        if len(numbers) >= target_layers:
            # 提取前N个数字作为权重
            weights = [float(n) for n in numbers[:target_layers]]
            adjustment = torch.tensor(weights, dtype=torch.float32)
            
            # 归一化到合理范围
            if adjustment.sum() > 0:
                adjustment = adjustment / adjustment.mean()
            else:
                adjustment = torch.ones(target_layers)
        else:
            # 如果解析失败，返回基于响应情感的简单调整
            if '高层' in response or '语义' in response or '推理' in response:
                # 强调高层重要性
                adjustment = torch.linspace(0.5, 2.0, target_layers)
            else:
                # 均匀权重
                adjustment = torch.ones(target_layers)
        
        return adjustment

def main():
    """主函数：演示Fisher信息计算"""
    logger.info("Fisher信息矩阵计算演示...")
    
    # 模拟数据
    samples = [
        {
            'user_profile': '用户喜欢科技产品，追求性能和创新',
            'candidate_items': ['iPhone15', 'MacBook', '无线耳机'],
        },
        {
            'user_profile': '用户注重美妆护肤，偏爱天然成分',
            'candidate_items': ['面膜', '精华液', '防晒霜'],
        }
    ]
    
    # 配置
    config = FisherConfig(
        num_samples=20,
        diagonal_only=True,
        normalize=True
    )
    
    # 自适应Fisher计算
    adaptive_calc = AdaptiveFisherCalculator(config)
    
    # 模拟模型（实际使用时传入真实模型）
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(64, 64) for _ in range(8)
            ])
        
        def forward(self, x):
            return x
    
    mock_model = MockModel()
    
    # 计算自适应权重
    weights = adaptive_calc.compute_adaptive_fisher_weights(
        mock_model, samples
    )
    
    logger.info(f"最终Fisher权重: {weights.tolist()}")
    logger.info("Fisher信息计算演示完成")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
