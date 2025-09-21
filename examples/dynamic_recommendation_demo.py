#!/usr/bin/env python3
"""
动态层选择推荐系统实战示例
演示如何在实际推荐场景中使用动态层选择机制
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

# 导入我们的动态层选择模块
import sys
sys.path.append('.')
sys.path.append('..')

from experiments.dynamic_layer_selection import (
    DynamicLayerSelector, 
    InputComplexityAnalyzer,
    ResourceMonitor,
    DynamicLayerConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserContext:
    """用户上下文信息"""
    user_id: str
    query: str
    category: str
    device_type: str  # 'mobile', 'edge', 'cloud'
    time_budget_ms: float = 100.0  # 时间预算
    memory_budget_mb: float = 500.0  # 内存预算

@dataclass
class RecommendationResult:
    """推荐结果"""
    items: List[str]
    scores: List[float]
    selected_layers: int
    inference_time_ms: float
    memory_usage_mb: float
    complexity_score: float
    quality_estimate: float

class DynamicRecommendationSystem:
    """动态层选择推荐系统"""
    
    def __init__(self, config: DynamicLayerConfig = None):
        self.config = config or DynamicLayerConfig()
        self.layer_selector = DynamicLayerSelector(self.config)
        self.complexity_analyzer = InputComplexityAnalyzer()
        self.resource_monitor = ResourceMonitor()
        
        # 模拟的物品库
        self.item_catalog = self._create_mock_item_catalog()
        
        # 预计算的用户-物品特征（实际中从数据库获取）
        self.user_features = self._create_mock_user_features()
        self.item_features = self._create_mock_item_features()
        
        logger.info("🚀 动态推荐系统初始化完成")
        
    def _create_mock_item_catalog(self) -> Dict[str, Dict]:
        """创建模拟物品目录"""
        categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports']
        items = {}
        
        for i in range(1000):
            item_id = f"item_{i:04d}"
            items[item_id] = {
                'title': f"Product {i}",
                'category': np.random.choice(categories),
                'price': np.random.uniform(10, 500),
                'rating': np.random.uniform(3.0, 5.0),
                'popularity': np.random.exponential(100)
            }
            
        return items
        
    def _create_mock_user_features(self) -> Dict[str, torch.Tensor]:
        """创建模拟用户特征"""
        users = {}
        for i in range(100):
            user_id = f"user_{i:03d}"
            # 64维用户特征向量
            users[user_id] = torch.randn(64)
        return users
        
    def _create_mock_item_features(self) -> Dict[str, torch.Tensor]:
        """创建模拟物品特征"""
        items = {}
        for item_id in self.item_catalog.keys():
            # 64维物品特征向量
            items[item_id] = torch.randn(64)
        return items
        
    def analyze_query_complexity(self, query: str, category: str) -> float:
        """分析查询复杂度"""
        # 简化的查询复杂度分析
        complexity_factors = {
            'query_length': min(1.0, len(query.split()) / 10),  # 查询词数
            'category_specificity': {
                'Electronics': 0.8,  # 电子产品需求复杂
                'Books': 0.6,        # 书籍需求中等
                'Clothing': 0.7,     # 服装需求较复杂
                'Home': 0.5,         # 家居需求简单
                'Sports': 0.6        # 运动用品中等
            }.get(category, 0.5),
            'semantic_complexity': min(1.0, len(set(query.lower().split())) / len(query.split()) if query.split() else 0)
        }
        
        # 综合复杂度评分
        overall_complexity = (
            complexity_factors['query_length'] * 0.4 +
            complexity_factors['category_specificity'] * 0.4 +
            complexity_factors['semantic_complexity'] * 0.2
        )
        
        return min(1.0, max(0.1, overall_complexity))
        
    def select_candidate_items(self, user_context: UserContext, num_candidates: int = 100) -> List[str]:
        """选择候选物品"""
        # 基于类别过滤
        if user_context.category != 'all':
            candidates = [
                item_id for item_id, item_info in self.item_catalog.items()
                if item_info['category'].lower() == user_context.category.lower()
            ]
        else:
            candidates = list(self.item_catalog.keys())
            
        # 随机选择候选项（实际中会使用更智能的策略）
        if len(candidates) > num_candidates:
            candidates = np.random.choice(candidates, num_candidates, replace=False).tolist()
            
        return candidates
        
    def simulate_model_inference(self, user_id: str, item_ids: List[str], 
                                num_layers: int) -> Tuple[List[float], float, float]:
        """模拟模型推理过程"""
        start_time = time.time()
        
        # 获取用户和物品特征
        user_feat = self.user_features.get(user_id, torch.randn(64))
        item_feats = torch.stack([
            self.item_features.get(item_id, torch.randn(64)) 
            for item_id in item_ids
        ])
        
        # 模拟不同层数的计算复杂度
        base_computation = len(item_ids) * 64  # 基础计算量
        layer_computation = base_computation * num_layers * 0.1  # 层级计算量
        
        # 模拟计算延迟
        simulated_delay = layer_computation / 1000000  # 转换为秒
        time.sleep(min(0.01, simulated_delay))  # 最多延迟10ms
        
        # 计算相似度分数（简化）
        scores = torch.cosine_similarity(
            user_feat.unsqueeze(0), item_feats, dim=1
        ).tolist()
        
        # 添加基于层数的性能影响
        layer_performance_factor = {
            4: 0.85, 8: 0.92, 12: 1.0, 16: 1.0,
            20: 0.95, 24: 0.90, 32: 0.85
        }.get(num_layers, 0.9)
        
        scores = [s * layer_performance_factor for s in scores]
        
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        memory_usage = num_layers * 12.5  # 简化的内存使用估算
        
        return scores, inference_time, memory_usage
        
    def recommend(self, user_context: UserContext, top_k: int = 10) -> RecommendationResult:
        """执行推荐"""
        logger.info(f"🎯 为用户 {user_context.user_id} 生成推荐...")
        
        # 1. 分析查询复杂度
        complexity_score = self.analyze_query_complexity(
            user_context.query, user_context.category
        )
        logger.info(f"  📊 查询复杂度: {complexity_score:.3f}")
        
        # 2. 选择最优层数
        resource_budget = {
            'mobile': 'mobile',
            'edge': 'edge',
            'cloud': 'cloud'
        }.get(user_context.device_type, 'cloud')
        
        selected_layers = self.layer_selector.select_optimal_layers(
            complexity_score, resource_budget, 'balanced'
        )
        logger.info(f"  🏗️ 选择层数: {selected_layers}")
        
        # 3. 选择候选物品
        candidate_items = self.select_candidate_items(user_context)
        logger.info(f"  📦 候选物品数: {len(candidate_items)}")
        
        # 4. 执行模型推理
        scores, inference_time, memory_usage = self.simulate_model_inference(
            user_context.user_id, candidate_items, selected_layers
        )
        
        # 5. 排序并选择top-k
        item_scores = list(zip(candidate_items, scores))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_items = [item for item, score in item_scores[:top_k]]
        top_scores = [score for item, score in item_scores[:top_k]]
        
        # 6. 估算推荐质量
        quality_estimate = np.mean(top_scores) * self._get_layer_quality_factor(selected_layers)
        
        logger.info(f"  ⚡ 推理时间: {inference_time:.1f}ms")
        logger.info(f"  💾 内存使用: {memory_usage:.1f}MB")
        logger.info(f"  🎯 质量估算: {quality_estimate:.3f}")
        
        return RecommendationResult(
            items=top_items,
            scores=top_scores,
            selected_layers=selected_layers,
            inference_time_ms=inference_time,
            memory_usage_mb=memory_usage,
            complexity_score=complexity_score,
            quality_estimate=quality_estimate
        )
        
    def _get_layer_quality_factor(self, num_layers: int) -> float:
        """获取层数对应的质量因子"""
        return {
            4: 0.85, 8: 0.92, 12: 1.0, 16: 1.0,
            20: 0.95, 24: 0.90, 32: 0.85
        }.get(num_layers, 0.9)
        
    def batch_recommend(self, user_contexts: List[UserContext], 
                       top_k: int = 10) -> List[RecommendationResult]:
        """批量推荐"""
        results = []
        total_time = 0
        total_memory = 0
        
        logger.info(f"🔄 开始批量推荐，用户数: {len(user_contexts)}")
        
        for i, context in enumerate(user_contexts):
            result = self.recommend(context, top_k)
            results.append(result)
            
            total_time += result.inference_time_ms
            total_memory += result.memory_usage_mb
            
            if (i + 1) % 10 == 0:
                avg_time = total_time / (i + 1)
                avg_memory = total_memory / (i + 1)
                logger.info(f"  📈 进度: {i+1}/{len(user_contexts)}, "
                          f"平均时间: {avg_time:.1f}ms, 平均内存: {avg_memory:.1f}MB")
        
        return results
        
    def analyze_performance(self, results: List[RecommendationResult]) -> Dict[str, Any]:
        """分析推荐性能"""
        layer_usage = {}
        device_performance = {'mobile': [], 'edge': [], 'cloud': []}
        
        for result in results:
            # 统计层数使用情况
            layer_count = layer_usage.get(result.selected_layers, 0)
            layer_usage[result.selected_layers] = layer_count + 1
            
        # 计算总体指标
        total_time = sum(r.inference_time_ms for r in results)
        total_memory = sum(r.memory_usage_mb for r in results)
        avg_quality = np.mean([r.quality_estimate for r in results])
        
        analysis = {
            'total_requests': len(results),
            'avg_inference_time_ms': total_time / len(results),
            'avg_memory_usage_mb': total_memory / len(results),
            'avg_quality_estimate': avg_quality,
            'layer_usage_distribution': layer_usage,
            'efficiency_metrics': {
                'total_time_saved_vs_max_layers': self._calculate_time_savings(results),
                'total_memory_saved_vs_max_layers': self._calculate_memory_savings(results),
                'quality_retention_rate': avg_quality / 1.0  # 假设1.0为最高质量
            }
        }
        
        return analysis
        
    def _calculate_time_savings(self, results: List[RecommendationResult]) -> float:
        """计算相对于最大层数的时间节省"""
        actual_time = sum(r.inference_time_ms for r in results)
        # 假设最大层数(16层)的推理时间是当前的1.6倍
        max_layer_time = sum(r.inference_time_ms * (16 / r.selected_layers) for r in results)
        return (max_layer_time - actual_time) / max_layer_time if max_layer_time > 0 else 0
        
    def _calculate_memory_savings(self, results: List[RecommendationResult]) -> float:
        """计算相对于最大层数的内存节省"""
        actual_memory = sum(r.memory_usage_mb for r in results)
        max_layer_memory = sum(r.memory_usage_mb * (16 / r.selected_layers) for r in results)
        return (max_layer_memory - actual_memory) / max_layer_memory if max_layer_memory > 0 else 0

def create_test_scenarios() -> List[UserContext]:
    """创建测试场景"""
    scenarios = [
        # 移动端场景
        UserContext("user_001", "轻薄笔记本电脑", "Electronics", "mobile", 50.0, 100.0),
        UserContext("user_002", "运动鞋", "Sports", "mobile", 50.0, 100.0),
        UserContext("user_003", "小说", "Books", "mobile", 50.0, 100.0),
        
        # 边缘计算场景
        UserContext("user_004", "智能家居设备推荐", "Electronics", "edge", 100.0, 500.0),
        UserContext("user_005", "办公用品批量采购", "Home", "edge", 100.0, 500.0),
        UserContext("user_006", "专业摄影设备", "Electronics", "edge", 100.0, 500.0),
        
        # 云端场景
        UserContext("user_007", "高端游戏设备定制化推荐方案", "Electronics", "cloud", 200.0, 2000.0),
        UserContext("user_008", "企业级服装采购解决方案", "Clothing", "cloud", 200.0, 2000.0),
        UserContext("user_009", "全方位健身器材配套推荐", "Sports", "cloud", 200.0, 2000.0),
        UserContext("user_010", "专业厨房设备整体方案", "Home", "cloud", 200.0, 2000.0),
    ]
    
    return scenarios

def main():
    """主函数 - 动态层选择推荐系统演示"""
    logger.info("🎬 开始动态层选择推荐系统演示...")
    
    # 初始化推荐系统
    recommender = DynamicRecommendationSystem()
    
    # 创建测试场景
    test_scenarios = create_test_scenarios()
    
    # 执行批量推荐
    logger.info(f"\n{'='*60}")
    logger.info("🔥 执行批量推荐测试")
    logger.info(f"{'='*60}")
    
    results = recommender.batch_recommend(test_scenarios, top_k=5)
    
    # 分析性能
    performance_analysis = recommender.analyze_performance(results)
    
    # 输出详细结果
    logger.info(f"\n{'='*60}")
    logger.info("📊 推荐结果详情")
    logger.info(f"{'='*60}")
    
    for i, (scenario, result) in enumerate(zip(test_scenarios, results)):
        logger.info(f"\n【场景 {i+1}】{scenario.device_type.upper()} - {scenario.query}")
        logger.info(f"  🎯 选择层数: {result.selected_layers}")
        logger.info(f"  📊 复杂度: {result.complexity_score:.3f}")
        logger.info(f"  ⚡ 推理时间: {result.inference_time_ms:.1f}ms")
        logger.info(f"  💾 内存使用: {result.memory_usage_mb:.1f}MB")
        logger.info(f"  🏆 质量估算: {result.quality_estimate:.3f}")
        logger.info(f"  📦 推荐物品: {result.items[:3]}")  # 显示前3个推荐
    
    # 输出性能分析
    logger.info(f"\n{'='*60}")
    logger.info("📈 整体性能分析")
    logger.info(f"{'='*60}")
    
    logger.info(f"📊 总请求数: {performance_analysis['total_requests']}")
    logger.info(f"⚡ 平均推理时间: {performance_analysis['avg_inference_time_ms']:.1f}ms")
    logger.info(f"💾 平均内存使用: {performance_analysis['avg_memory_usage_mb']:.1f}MB")
    logger.info(f"🎯 平均质量评分: {performance_analysis['avg_quality_estimate']:.3f}")
    
    logger.info(f"\n🏗️ 层数使用分布:")
    for layers, count in sorted(performance_analysis['layer_usage_distribution'].items()):
        percentage = count / performance_analysis['total_requests'] * 100
        logger.info(f"  {layers} 层: {count} 次 ({percentage:.1f}%)")
    
    efficiency = performance_analysis['efficiency_metrics']
    logger.info(f"\n⚡ 效率指标:")
    logger.info(f"  时间节省率: {efficiency['total_time_saved_vs_max_layers']:.1%}")
    logger.info(f"  内存节省率: {efficiency['total_memory_saved_vs_max_layers']:.1%}")
    logger.info(f"  质量保持率: {efficiency['quality_retention_rate']:.1%}")
    
    # 保存结果
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('results/dynamic_layer_selection')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果
    detailed_results = {
        'timestamp': timestamp,
        'scenarios': [
            {
                'user_id': s.user_id,
                'query': s.query,
                'category': s.category,
                'device_type': s.device_type,
                'result': {
                    'selected_layers': r.selected_layers,
                    'complexity_score': r.complexity_score,
                    'inference_time_ms': r.inference_time_ms,
                    'memory_usage_mb': r.memory_usage_mb,
                    'quality_estimate': r.quality_estimate,
                    'top_items': r.items
                }
            }
            for s, r in zip(test_scenarios, results)
        ],
        'performance_analysis': performance_analysis
    }
    
    results_file = results_dir / f'dynamic_recommendation_demo_{timestamp}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n✅ 演示完成！结果已保存至: {results_file}")
    logger.info(f"🎉 动态层选择推荐系统运行成功！")

if __name__ == "__main__":
    main()
