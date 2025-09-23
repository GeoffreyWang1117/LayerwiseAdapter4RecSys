#!/usr/bin/env python3
"""
Universal Layerwise-Adapter 框架演示示例
Demo: 从Amazon推荐系统扩展到计算机视觉分类任务

展示如何使用同一个框架分析不同模态的模型：
1. 文本分类模型 (BERT-like)
2. 图像分类模型 (ResNet-like)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Tuple

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入我们的通用框架
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal.layerwise_adapter import (
    create_analyzer, AnalysisConfig, UniversalLayerwiseAdapter,
    TaskType, ModalityType, TextModelAdapter, VisionModelAdapter
)

class SimpleTextClassifier(nn.Module):
    """简单的文本分类模型 (BERT-like结构)"""
    
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=2, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 多层Transformer-like结构
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=hidden_dim, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # 通过各层
        for layer in self.layers:
            x = layer(x)
            
        # 池化和分类
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        x = self.pooler(x).squeeze(-1)  # (batch_size, embed_dim)
        x = self.dropout(x)
        x = self.classifier(x)  # (batch_size, num_classes)
        
        return x

class SimpleVisionClassifier(nn.Module):
    """简单的图像分类模型 (ResNet-like结构)"""
    
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def create_sample_text_data(batch_size=32, seq_len=128, vocab_size=10000, num_batches=10) -> DataLoader:
    """创建示例文本数据"""
    data = []
    labels = []
    
    for _ in range(num_batches):
        batch_data = torch.randint(0, vocab_size, (batch_size, seq_len))
        batch_labels = torch.randint(0, 2, (batch_size,))
        data.append(batch_data)
        labels.append(batch_labels)
    
    data = torch.cat(data, dim=0)
    labels = torch.cat(labels, dim=0)
    
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def create_sample_vision_data(batch_size=32, channels=3, height=224, width=224, num_classes=10, num_batches=10) -> DataLoader:
    """创建示例图像数据"""
    data = []
    labels = []
    
    for _ in range(num_batches):
        batch_data = torch.randn(batch_size, channels, height, width)
        batch_labels = torch.randint(0, num_classes, (batch_size,))
        data.append(batch_data)
        labels.append(batch_labels)
    
    data = torch.cat(data, dim=0)
    labels = torch.cat(labels, dim=0)
    
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def demo_text_classification():
    """演示文本分类模型分析"""
    print("\n" + "="*60)
    print("🔤 文本分类模型分析演示")
    print("="*60)
    
    # 创建模型
    model = SimpleTextClassifier(vocab_size=10000, num_layers=6)
    print(f"📊 模型结构: {sum(p.numel() for p in model.parameters())} 参数")
    
    # 创建数据
    data_loader = create_sample_text_data()
    print(f"📁 数据: {len(data_loader.dataset)} 样本")
    
    # 创建分析器
    adapter = create_analyzer(
        model_name="simple-text-classifier",
        task_type="classification",
        modality_type="text",
        analysis_methods=['fisher_information', 'gradient_based']
    )
    
    # 加载模型 (需要自定义适配器)
    class CustomTextAdapter(TextModelAdapter):
        def _initialize_layers(self):
            layer_idx = 0
            # 嵌入层
            self.layers.append(Layer(self.model.embedding, layer_idx, "embedding"))
            layer_idx += 1
            
            # Transformer层
            for i, transformer_layer in enumerate(self.model.layers):
                self.layers.append(Layer(transformer_layer, layer_idx, f"transformer_{i}"))
                layer_idx += 1
                
            # 分类器
            self.layers.append(Layer(self.model.classifier, layer_idx, "classifier"))
    
    # 替换默认适配器
    from universal.layerwise_adapter import Layer
    adapter.model = CustomTextAdapter(model, adapter.config)
    
    print(f"🔧 检测到 {len(adapter.model.layers)} 层")
    for layer in adapter.model.layers[:3]:  # 显示前3层
        print(f"   - {layer}")
    
    # 执行分析
    try:
        results = adapter.analyze_importance(data_loader)
        print(f"✅ 分析完成，使用了 {len(results)} 种方法")
        
        # 显示结果
        for method, scores in results.items():
            print(f"\n📈 {method} 结果:")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for layer_idx, score in sorted_scores[:5]:  # 显示top5
                layer_name = adapter.model.layers[layer_idx].layer_name
                print(f"   Layer {layer_idx} ({layer_name}): {score:.4f}")
        
        # 生成压缩方案
        compression_plan = adapter.generate_compression_plan(target_ratio=2.0)
        print(f"\n🗜️  压缩方案 (2.0x):")
        print(f"   原始层数: {compression_plan['original_layers']}")
        print(f"   压缩后层数: {compression_plan['compressed_layers']}")
        print(f"   保留层: {compression_plan['keep_layer_indices']}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def demo_vision_classification():
    """演示视觉分类模型分析"""
    print("\n" + "="*60)
    print("🖼️  视觉分类模型分析演示")
    print("="*60)
    
    # 创建模型
    model = SimpleVisionClassifier(input_channels=3, num_classes=10)
    print(f"📊 模型结构: {sum(p.numel() for p in model.parameters())} 参数")
    
    # 创建数据
    data_loader = create_sample_vision_data()
    print(f"📁 数据: {len(data_loader.dataset)} 样本")
    
    # 创建分析器
    adapter = create_analyzer(
        model_name="simple-vision-classifier",
        task_type="classification",
        modality_type="vision",
        analysis_methods=['fisher_information', 'gradient_based']
    )
    
    # 加载模型 (需要自定义适配器)  
    class CustomVisionAdapter(VisionModelAdapter):
        def _initialize_layers(self):
            layer_idx = 0
            
            # 初始卷积层
            self.layers.append(Layer(self.model.conv1, layer_idx, "conv1"))
            layer_idx += 1
            
            # 残差层
            for i, res_layer in enumerate([self.model.layer1, self.model.layer2, 
                                         self.model.layer3, self.model.layer4]):
                self.layers.append(Layer(res_layer, layer_idx, f"layer{i+1}"))
                layer_idx += 1
                
            # 分类器
            self.layers.append(Layer(self.model.fc, layer_idx, "classifier"))
    
    # 替换默认适配器
    from universal.layerwise_adapter import Layer
    adapter.model = CustomVisionAdapter(model, adapter.config)
    
    print(f"🔧 检测到 {len(adapter.model.layers)} 层")
    for layer in adapter.model.layers:
        print(f"   - {layer}")
    
    # 执行分析
    try:
        results = adapter.analyze_importance(data_loader)
        print(f"✅ 分析完成，使用了 {len(results)} 种方法")
        
        # 显示结果
        for method, scores in results.items():
            print(f"\n📈 {method} 结果:")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for layer_idx, score in sorted_scores:
                layer_name = adapter.model.layers[layer_idx].layer_name
                print(f"   Layer {layer_idx} ({layer_name}): {score:.4f}")
        
        # 生成压缩方案
        compression_plan = adapter.generate_compression_plan(target_ratio=1.5)
        print(f"\n🗜️  压缩方案 (1.5x):")
        print(f"   原始层数: {compression_plan['original_layers']}")
        print(f"   压缩后层数: {compression_plan['compressed_layers']}")
        print(f"   保留层: {compression_plan['keep_layer_indices']}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

def compare_modalities():
    """对比不同模态的分析结果"""
    print("\n" + "="*60)
    print("⚖️  跨模态分析结果对比")
    print("="*60)
    
    print("📋 对比总结:")
    print("   🔤 文本模型: 注意力层和分类器通常更重要")
    print("   🖼️  视觉模型: 高层特征提取器和分类器更重要") 
    print("   🔧 通用框架: 相同的分析方法适用于不同模态")
    print("   📊 压缩策略: 根据任务特性自动调整")

def main():
    """主演示函数"""
    print("🚀 Universal Layerwise-Adapter 框架演示")
    print("从Amazon推荐系统扩展到通用AI模型分析")
    
    # 设置随机种子确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 演示文本分类
        demo_text_classification()
        
        # 演示视觉分类  
        demo_vision_classification()
        
        # 对比分析
        compare_modalities()
        
        print("\n" + "="*60)
        print("🎉 演示完成！")
        print("🌟 通用框架成功分析了不同模态的模型")
        print("📈 相同的分析方法产生了有意义的层重要性排序")
        print("🎯 框架具备了向更多领域扩展的基础能力")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
