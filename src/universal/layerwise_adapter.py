#!/usr/bin/env python3
"""
通用Layerwise-Adapter框架核心实现
Universal Framework for Layer Importance Analysis and Model Compression

Author: Layerwise-Adapter Research Team
Date: 2024-12-20
Version: 1.0.0-alpha
"""

import os
import abc
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """支持的任务类型"""
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    RECOMMENDATION = "recommendation"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QA = "question_answering"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

class ModalityType(Enum):
    """支持的模态类型"""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    GRAPH = "graph"
    TABULAR = "tabular"

@dataclass
class AnalysisConfig:
    """分析配置"""
    model_name: str
    task_type: TaskType
    modality_type: ModalityType
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    max_samples: int = 10000
    compression_targets: List[float] = None
    analysis_methods: List[str] = None
    
    def __post_init__(self):
        if self.compression_targets is None:
            self.compression_targets = [1.5, 2.0, 2.5, 3.0]
        if self.analysis_methods is None:
            self.analysis_methods = ['fisher_information', 'gradient_based', 'layer_ablation']

class Layer:
    """通用层表示"""
    def __init__(self, module: nn.Module, layer_idx: int, layer_name: str):
        self.module = module
        self.layer_idx = layer_idx  
        self.layer_name = layer_name
        self.parameters = sum(p.numel() for p in module.parameters())
        
    def __repr__(self):
        return f"Layer({self.layer_idx}, {self.layer_name}, {self.parameters} params)"

class UniversalModel(abc.ABC):
    """通用模型抽象基类"""
    
    def __init__(self, model: nn.Module, config: AnalysisConfig):
        self.model = model
        self.config = config
        self.layers: List[Layer] = []
        self.hooks: List = []
        self.activations: Dict[int, torch.Tensor] = {}
        self._initialize_layers()
        
    @abc.abstractmethod
    def _initialize_layers(self):
        """初始化层列表 - 子类必须实现"""
        pass
        
    @abc.abstractmethod
    def get_layer_output(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """获取指定层的输出 - 子类必须实现"""
        pass
        
    def forward_with_hooks(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """带钩子的前向传播，获取所有层的激活"""
        self.activations.clear()
        
        def make_hook(idx):
            def hook(module, input, output):
                self.activations[idx] = output.detach()
            return hook
            
        # 注册钩子
        self.hooks = []
        for layer in self.layers:
            hook = layer.module.register_forward_hook(make_hook(layer.layer_idx))
            self.hooks.append(hook)
            
        # 前向传播
        _ = self.model(x)
        
        # 移除钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
            
        return self.activations.copy()
        
    def get_layer_parameters(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """获取指定层的参数"""
        if layer_idx >= len(self.layers):
            raise IndexError(f"Layer index {layer_idx} out of range")
        
        layer = self.layers[layer_idx]
        return {name: param for name, param in layer.module.named_parameters()}

class ImportanceAnalyzer(abc.ABC):
    """重要性分析器抽象基类"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abc.abstractmethod
    def analyze(self, model: UniversalModel, data_loader) -> Dict[int, float]:
        """分析各层重要性
        
        Args:
            model: 通用模型实例
            data_loader: 数据加载器
            
        Returns:
            Dict[int, float]: 层索引到重要性得分的映射
        """
        pass
        
    def normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """标准化重要性得分到[0,1]区间"""
        if not scores:
            return {}
            
        values = list(scores.values())
        min_val, max_val = min(values), max(values)
        
        if max_val == min_val:
            return {k: 1.0 for k in scores.keys()}
            
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

class FisherInformationAnalyzer(ImportanceAnalyzer):
    """Fisher信息分析器"""
    
    def __init__(self):
        super().__init__("fisher_information")
        
    def analyze(self, model: UniversalModel, data_loader) -> Dict[int, float]:
        """基于Fisher信息矩阵计算层重要性"""
        logger.info("开始Fisher信息分析...")
        
        model.model.eval()
        layer_fisher_scores = defaultdict(float)
        
        for batch_idx, (data, targets) in enumerate(data_loader):
            if batch_idx >= 10:  # 限制计算量
                break
                
            data = data.to(model.config.device)
            targets = targets.to(model.config.device)
            
            # 前向传播
            outputs = model.model(data)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # 反向传播
            model.model.zero_grad()
            loss.backward()
            
            # 计算每层的Fisher信息
            for layer in model.layers:
                fisher_score = 0.0
                for param in layer.module.parameters():
                    if param.grad is not None:
                        fisher_score += (param.grad ** 2).sum().item()
                layer_fisher_scores[layer.layer_idx] += fisher_score
                
        # 平均化
        for layer_idx in layer_fisher_scores:
            layer_fisher_scores[layer_idx] /= min(len(data_loader), 10)
            
        logger.info(f"Fisher信息分析完成，共分析{len(layer_fisher_scores)}层")
        return dict(layer_fisher_scores)

class GradientBasedAnalyzer(ImportanceAnalyzer):
    """基于梯度的分析器"""
    
    def __init__(self):
        super().__init__("gradient_based")
        
    def analyze(self, model: UniversalModel, data_loader) -> Dict[int, float]:
        """基于梯度幅度计算层重要性"""
        logger.info("开始梯度分析...")
        
        model.model.eval()
        layer_gradient_scores = defaultdict(float)
        
        for batch_idx, (data, targets) in enumerate(data_loader):
            if batch_idx >= 10:  # 限制计算量
                break
                
            data = data.to(model.config.device)
            targets = targets.to(model.config.device)
            
            # 前向传播
            outputs = model.model(data)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # 反向传播
            model.model.zero_grad()
            loss.backward()
            
            # 计算每层的梯度幅度
            for layer in model.layers:
                gradient_norm = 0.0
                for param in layer.module.parameters():
                    if param.grad is not None:
                        gradient_norm += param.grad.norm().item()
                layer_gradient_scores[layer.layer_idx] += gradient_norm
                
        # 平均化
        for layer_idx in layer_gradient_scores:
            layer_gradient_scores[layer_idx] /= min(len(data_loader), 10)
            
        logger.info(f"梯度分析完成，共分析{len(layer_gradient_scores)}层")
        return dict(layer_gradient_scores)

class LayerAblationAnalyzer(ImportanceAnalyzer):
    """层消融分析器"""
    
    def __init__(self):
        super().__init__("layer_ablation")
        
    def analyze(self, model: UniversalModel, data_loader) -> Dict[int, float]:
        """通过层消融计算层重要性"""
        logger.info("开始层消融分析...")
        
        # 获取基线性能
        baseline_acc = self._evaluate_model(model, data_loader)
        logger.info(f"基线准确率: {baseline_acc:.4f}")
        
        layer_importance_scores = {}
        
        for layer in model.layers:
            # 临时禁用该层
            original_forward = layer.module.forward
            layer.module.forward = lambda x: x  # 跳过该层
            
            # 评估性能下降
            ablated_acc = self._evaluate_model(model, data_loader)
            importance = baseline_acc - ablated_acc
            layer_importance_scores[layer.layer_idx] = max(0, importance)
            
            # 恢复原始forward函数
            layer.module.forward = original_forward
            
            logger.info(f"Layer {layer.layer_idx}: 消融后准确率={ablated_acc:.4f}, 重要性={importance:.4f}")
            
        logger.info(f"层消融分析完成，共分析{len(layer_importance_scores)}层")
        return layer_importance_scores
        
    def _evaluate_model(self, model: UniversalModel, data_loader) -> float:
        """评估模型性能"""
        model.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 5:  # 限制计算量
                    break
                    
                data = data.to(model.config.device)
                targets = targets.to(model.config.device)
                
                outputs = model.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        return correct / total if total > 0 else 0.0

class AnalysisMethodRegistry:
    """分析方法注册表"""
    
    def __init__(self):
        self.methods = {
            'fisher_information': FisherInformationAnalyzer,
            'gradient_based': GradientBasedAnalyzer,
            'layer_ablation': LayerAblationAnalyzer,
        }
        
    def get_analyzer(self, method_name: str) -> ImportanceAnalyzer:
        """获取分析器实例"""
        if method_name not in self.methods:
            raise ValueError(f"Unknown analysis method: {method_name}")
        return self.methods[method_name]()
        
    def add_method(self, name: str, analyzer_class):
        """添加新的分析方法"""
        self.methods[name] = analyzer_class

class UniversalLayerwiseAdapter:
    """通用层级适配器主类"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.registry = AnalysisMethodRegistry()
        self.model: Optional[UniversalModel] = None
        self.analysis_results: Dict[str, Dict[int, float]] = {}
        
    def load_model(self, model: nn.Module) -> UniversalModel:
        """加载并适配模型"""
        # 这里需要根据模型类型选择合适的适配器
        # 暂时返回一个通用实现
        logger.info(f"加载模型: {self.config.model_name}")
        
        # TODO: 实现自动模型类型检测和适配器选择
        if self.config.modality_type == ModalityType.TEXT:
            self.model = TextModelAdapter(model, self.config)
        elif self.config.modality_type == ModalityType.VISION:
            self.model = VisionModelAdapter(model, self.config)
        else:
            raise NotImplementedError(f"Modality {self.config.modality_type} not yet supported")
            
        logger.info(f"模型适配完成，共检测到{len(self.model.layers)}层")
        return self.model
        
    def analyze_importance(self, data_loader) -> Dict[str, Dict[int, float]]:
        """执行层重要性分析"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        logger.info(f"开始重要性分析，使用方法: {self.config.analysis_methods}")
        
        for method_name in self.config.analysis_methods:
            analyzer = self.registry.get_analyzer(method_name)
            scores = analyzer.analyze(self.model, data_loader)
            normalized_scores = analyzer.normalize_scores(scores)
            self.analysis_results[method_name] = normalized_scores
            
        logger.info("所有分析方法执行完成")
        return self.analysis_results
        
    def compute_consensus_ranking(self) -> Dict[int, float]:
        """计算多方法一致性排名"""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Call analyze_importance() first.")
            
        # 计算所有方法的平均重要性得分
        consensus_scores = defaultdict(float)
        method_count = len(self.analysis_results)
        
        for method_results in self.analysis_results.values():
            for layer_idx, score in method_results.items():
                consensus_scores[layer_idx] += score / method_count
                
        logger.info(f"一致性排名计算完成，基于{method_count}种方法")
        return dict(consensus_scores)
        
    def generate_compression_plan(self, target_ratio: float) -> Dict[str, Any]:
        """生成压缩方案"""
        consensus_scores = self.compute_consensus_ranking()
        
        # 按重要性排序
        sorted_layers = sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 计算需要保留的层数
        total_layers = len(sorted_layers)
        keep_layers = int(total_layers / target_ratio)
        
        compression_plan = {
            'target_ratio': target_ratio,
            'original_layers': total_layers,
            'compressed_layers': keep_layers,
            'removed_layers': total_layers - keep_layers,
            'keep_layer_indices': [idx for idx, _ in sorted_layers[:keep_layers]],
            'remove_layer_indices': [idx for idx, _ in sorted_layers[keep_layers:]],
            'layer_importance_ranking': sorted_layers
        }
        
        logger.info(f"压缩方案生成完成: {total_layers}层 → {keep_layers}层 (压缩比{target_ratio}x)")
        return compression_plan

# 模态特定适配器实现
class TextModelAdapter(UniversalModel):
    """文本模型适配器"""
    
    def _initialize_layers(self):
        """初始化文本模型的层"""
        # 这里需要根据具体的文本模型架构来实现
        # 暂时提供一个通用实现
        layer_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU, nn.TransformerEncoderLayer)):
                self.layers.append(Layer(module, layer_idx, name))
                layer_idx += 1
                
    def get_layer_output(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """获取指定层的输出"""
        # TODO: 实现获取特定层输出的逻辑
        raise NotImplementedError("TextModelAdapter.get_layer_output not implemented")

class VisionModelAdapter(UniversalModel):
    """视觉模型适配器"""
    
    def _initialize_layers(self):
        """初始化视觉模型的层"""
        layer_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                self.layers.append(Layer(module, layer_idx, name))
                layer_idx += 1
                
    def get_layer_output(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """获取指定层的输出"""
        # TODO: 实现获取特定层输出的逻辑
        raise NotImplementedError("VisionModelAdapter.get_layer_output not implemented")

# 便利函数
def create_analyzer(model_name: str, task_type: str, modality_type: str, **kwargs) -> UniversalLayerwiseAdapter:
    """创建分析器的便利函数"""
    config = AnalysisConfig(
        model_name=model_name,
        task_type=TaskType(task_type),
        modality_type=ModalityType(modality_type),
        **kwargs
    )
    return UniversalLayerwiseAdapter(config)

if __name__ == "__main__":
    # 示例用法
    from collections import defaultdict
    
    # 创建配置
    config = AnalysisConfig(
        model_name="bert-base-uncased",
        task_type=TaskType.CLASSIFICATION,
        modality_type=ModalityType.TEXT,
        analysis_methods=['fisher_information', 'gradient_based']
    )
    
    # 创建分析器
    adapter = UniversalLayerwiseAdapter(config)
    
    print("🚀 通用Layerwise-Adapter框架初始化完成!")
    print(f"📊 支持的任务类型: {[t.value for t in TaskType]}")
    print(f"🔧 支持的模态类型: {[m.value for m in ModalityType]}")
    print(f"⚡ 支持的分析方法: {list(adapter.registry.methods.keys())}")
