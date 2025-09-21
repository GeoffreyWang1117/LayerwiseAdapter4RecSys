#!/usr/bin/env python3
"""
高级动态层选择实现 - 基于机器学习的智能层选择器
包含学习型复杂度分析和预测性层选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdvancedLayerConfig:
    """高级动态层选择配置"""
    max_layers: int = 32
    min_layers: int = 4
    layer_options: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 32])
    
    # 学习型参数
    learning_enabled: bool = True
    min_samples_for_learning: int = 100
    retrain_interval: int = 1000  # 每1000次推理重新训练
    
    # 性能目标
    latency_budget_ms: float = 100.0
    memory_budget_mb: float = 500.0
    quality_threshold: float = 0.85
    
    # 特征权重
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'query_complexity': 0.3,
        'user_history': 0.2,
        'category_difficulty': 0.2,
        'resource_constraint': 0.2,
        'quality_requirement': 0.1
    })

class SmartComplexityAnalyzer:
    """智能复杂度分析器"""
    
    def __init__(self):
        self.category_complexity_map = {
            'Electronics': 0.8,
            'Books': 0.6,
            'Clothing': 0.7,
            'Home': 0.5,
            'Sports': 0.6,
            'all': 0.7
        }
        
        self.user_interaction_history = {}  # 用户交互历史
        self.category_performance_history = {}  # 类别性能历史
        
    def analyze_query_complexity(self, query: str, user_id: str, category: str) -> Dict[str, float]:
        """分析查询复杂度 - 返回多维复杂度特征"""
        
        # 1. 文本复杂度
        words = query.lower().split()
        text_complexity = {
            'length': min(1.0, len(words) / 20),  # 文本长度
            'vocabulary_diversity': len(set(words)) / len(words) if words else 0,  # 词汇多样性
            'semantic_density': self._calculate_semantic_density(words),  # 语义密度
            'query_specificity': self._calculate_query_specificity(words)  # 查询具体性
        }
        
        # 2. 用户历史复杂度
        user_complexity = self._analyze_user_history_complexity(user_id)
        
        # 3. 类别复杂度
        category_complexity = {
            'base_difficulty': self.category_complexity_map.get(category, 0.5),
            'historical_performance': self._get_category_historical_complexity(category)
        }
        
        # 4. 综合复杂度评分
        overall_complexity = (
            np.mean(list(text_complexity.values())) * 0.4 +
            user_complexity * 0.3 +
            np.mean(list(category_complexity.values())) * 0.3
        )
        
        return {
            'text_complexity': text_complexity,
            'user_complexity': user_complexity,
            'category_complexity': category_complexity,
            'overall_complexity': min(1.0, max(0.1, overall_complexity))
        }
        
    def _calculate_semantic_density(self, words: List[str]) -> float:
        """计算语义密度"""
        if not words:
            return 0.0
            
        # 简化的语义密度计算 - 基于词汇重要性
        important_words = ['高端', '专业', '定制', '智能', '全方位', '批量', '企业级']
        semantic_score = sum(1 for word in words if any(imp in word for imp in important_words))
        
        return min(1.0, semantic_score / len(words))
        
    def _calculate_query_specificity(self, words: List[str]) -> float:
        """计算查询具体性"""
        if not words:
            return 0.0
            
        # 具体性指标 - 基于修饰词和限定词
        specific_indicators = ['型号', '品牌', '规格', '配置', '颜色', '尺寸', '材质']
        specificity_score = sum(1 for word in words if any(spec in word for spec in specific_indicators))
        
        return min(1.0, specificity_score / max(1, len(words) - 2))  # 排除常见功能词
        
    def _analyze_user_history_complexity(self, user_id: str) -> float:
        """分析用户历史复杂度"""
        if user_id not in self.user_interaction_history:
            return 0.5  # 新用户默认中等复杂度
            
        history = self.user_interaction_history[user_id]
        return np.mean(history.get('complexity_scores', [0.5]))
        
    def _get_category_historical_complexity(self, category: str) -> float:
        """获取类别历史复杂度"""
        if category not in self.category_performance_history:
            return 0.5
            
        history = self.category_performance_history[category]
        return np.mean(history.get('avg_complexity', [0.5]))
        
    def update_user_history(self, user_id: str, complexity_score: float, performance: float):
        """更新用户历史记录"""
        if user_id not in self.user_interaction_history:
            self.user_interaction_history[user_id] = {
                'complexity_scores': [],
                'performance_scores': [],
                'interaction_count': 0
            }
            
        history = self.user_interaction_history[user_id]
        history['complexity_scores'].append(complexity_score)
        history['performance_scores'].append(performance)
        history['interaction_count'] += 1
        
        # 保持历史记录大小
        if len(history['complexity_scores']) > 50:
            history['complexity_scores'] = history['complexity_scores'][-50:]
            history['performance_scores'] = history['performance_scores'][-50:]
            
    def update_category_history(self, category: str, complexity_score: float, performance: float):
        """更新类别历史记录"""
        if category not in self.category_performance_history:
            self.category_performance_history[category] = {
                'avg_complexity': [],
                'avg_performance': [],
                'sample_count': 0
            }
            
        history = self.category_performance_history[category]
        history['avg_complexity'].append(complexity_score)
        history['avg_performance'].append(performance)
        history['sample_count'] += 1
        
        # 保持历史记录大小
        if len(history['avg_complexity']) > 100:
            history['avg_complexity'] = history['avg_complexity'][-100:]
            history['avg_performance'] = history['avg_performance'][-100:]

class PredictiveLayerSelector:
    """预测性层选择器"""
    
    def __init__(self, config: AdvancedLayerConfig):
        self.config = config
        self.complexity_analyzer = SmartComplexityAnalyzer()
        
        # 机器学习模型
        self.layer_predictor = None  # 层数预测模型
        self.performance_predictor = None  # 性能预测模型
        self.scaler = StandardScaler()
        
        # 训练数据收集
        self.training_data = []
        self.is_trained = False
        self.inference_count = 0
        
        # 性能历史
        self.performance_history = {
            layer: {'latency': [], 'memory': [], 'quality': []}
            for layer in self.config.layer_options
        }
        
        logger.info("🧠 初始化预测性层选择器")
        
    def select_optimal_layers(self, query: str, user_id: str, category: str,
                            device_type: str = 'cloud', performance_mode: str = 'balanced') -> Dict[str, Any]:
        """选择最优层数 - 使用机器学习预测"""
        
        # 1. 分析复杂度特征
        complexity_analysis = self.complexity_analyzer.analyze_query_complexity(query, user_id, category)
        
        # 2. 提取特征向量
        features = self._extract_features(complexity_analysis, device_type, performance_mode)
        
        # 3. 预测最优层数
        if self.is_trained and self.layer_predictor is not None:
            predicted_layers = self._predict_optimal_layers(features)
        else:
            predicted_layers = self._fallback_layer_selection(complexity_analysis, device_type)
            
        # 4. 验证和调整
        final_layers = self._validate_and_adjust_layers(predicted_layers, device_type, performance_mode)
        
        # 5. 预测性能指标
        predicted_performance = self._predict_performance_metrics(final_layers, features)
        
        selection_result = {
            'selected_layers': final_layers,
            'complexity_analysis': complexity_analysis,
            'features': features,
            'predicted_performance': predicted_performance,
            'selection_confidence': self._calculate_selection_confidence(features, final_layers)
        }
        
        self.inference_count += 1
        return selection_result
        
    def _extract_features(self, complexity_analysis: Dict, device_type: str, performance_mode: str) -> np.ndarray:
        """提取特征向量"""
        features = []
        
        # 复杂度特征
        text_comp = complexity_analysis['text_complexity']
        features.extend([
            text_comp['length'],
            text_comp['vocabulary_diversity'],
            text_comp['semantic_density'],
            text_comp['query_specificity']
        ])
        
        # 用户和类别特征
        features.append(complexity_analysis['user_complexity'])
        features.extend(list(complexity_analysis['category_complexity'].values()))
        
        # 设备和性能模式特征 (one-hot编码)
        device_features = [0, 0, 0]  # mobile, edge, cloud
        device_idx = {'mobile': 0, 'edge': 1, 'cloud': 2}.get(device_type, 2)
        device_features[device_idx] = 1
        features.extend(device_features)
        
        mode_features = [0, 0, 0]  # fast, balanced, accurate
        mode_idx = {'fast': 0, 'balanced': 1, 'accurate': 2}.get(performance_mode, 1)
        mode_features[mode_idx] = 1
        features.extend(mode_features)
        
        return np.array(features)
        
    def _predict_optimal_layers(self, features: np.ndarray) -> int:
        """使用机器学习模型预测最优层数"""
        try:
            # 标准化特征
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # 预测层数
            predicted_layers_continuous = self.layer_predictor.predict(features_scaled)[0]
            
            # 映射到可用的层数选项
            predicted_layers = min(self.config.layer_options, 
                                 key=lambda x: abs(x - predicted_layers_continuous))
            
            return predicted_layers
            
        except Exception as e:
            logger.warning(f"预测失败，使用回退策略: {e}")
            return self._fallback_layer_selection({'overall_complexity': 0.5}, 'cloud')
            
    def _fallback_layer_selection(self, complexity_analysis: Dict, device_type: str) -> int:
        """回退层选择策略"""
        complexity = complexity_analysis.get('overall_complexity', 0.5)
        
        if device_type == 'mobile':
            return 4 if complexity < 0.6 else 8
        elif device_type == 'edge':
            return 8 if complexity < 0.7 else 12
        else:  # cloud
            if complexity < 0.4:
                return 8
            elif complexity < 0.7:
                return 12
            else:
                return 16
                
    def _validate_and_adjust_layers(self, predicted_layers: int, device_type: str, performance_mode: str) -> int:
        """验证和调整层数选择"""
        # 设备约束
        device_max_layers = {'mobile': 8, 'edge': 16, 'cloud': 32}
        max_allowed = device_max_layers.get(device_type, 32)
        
        # 性能模式调整
        if performance_mode == 'fast':
            max_allowed = min(max_allowed, 12)
        elif performance_mode == 'accurate':
            # 精确模式允许更多层数，但仍需符合设备约束
            pass
            
        # 确保在可用选项内
        validated_layers = min(predicted_layers, max_allowed)
        validated_layers = max(validated_layers, self.config.min_layers)
        
        # 映射到最近的可用层数
        final_layers = min(self.config.layer_options, 
                          key=lambda x: abs(x - validated_layers))
        
        return final_layers
        
    def _predict_performance_metrics(self, layers: int, features: np.ndarray) -> Dict[str, float]:
        """预测性能指标"""
        if layers in self.performance_history:
            history = self.performance_history[layers]
            
            return {
                'expected_latency_ms': np.mean(history['latency']) if history['latency'] else layers * 8.0,
                'expected_memory_mb': np.mean(history['memory']) if history['memory'] else layers * 12.5,
                'expected_quality': np.mean(history['quality']) if history['quality'] else self._estimate_quality_by_layers(layers)
            }
        else:
            return {
                'expected_latency_ms': layers * 8.0,
                'expected_memory_mb': layers * 12.5,
                'expected_quality': self._estimate_quality_by_layers(layers)
            }
            
    def _estimate_quality_by_layers(self, layers: int) -> float:
        """基于层数估算质量"""
        quality_map = {4: 0.85, 8: 0.92, 12: 1.0, 16: 1.0, 20: 0.95, 24: 0.90, 32: 0.85}
        return quality_map.get(layers, 0.9)
        
    def _calculate_selection_confidence(self, features: np.ndarray, selected_layers: int) -> float:
        """计算选择置信度"""
        if not self.is_trained:
            return 0.5
            
        # 基于历史数据的置信度计算
        base_confidence = 0.7
        
        # 如果有足够的历史数据，提高置信度
        if len(self.training_data) > self.config.min_samples_for_learning:
            base_confidence = 0.85
            
        return base_confidence
        
    def record_performance(self, layers: int, latency_ms: float, memory_mb: float, quality: float,
                          complexity_analysis: Dict, user_id: str, category: str):
        """记录性能数据用于学习"""
        
        # 更新性能历史
        if layers in self.performance_history:
            history = self.performance_history[layers]
            history['latency'].append(latency_ms)
            history['memory'].append(memory_mb)
            history['quality'].append(quality)
            
            # 保持历史记录大小
            for metric in ['latency', 'memory', 'quality']:
                if len(history[metric]) > 200:
                    history[metric] = history[metric][-200:]
        
        # 收集训练数据
        features = self._extract_features(complexity_analysis, 'cloud', 'balanced')  # 简化
        self.training_data.append({
            'features': features,
            'optimal_layers': layers,
            'latency': latency_ms,
            'memory': memory_mb,
            'quality': quality
        })
        
        # 更新复杂度分析器的历史记录
        overall_complexity = complexity_analysis.get('overall_complexity', 0.5)
        self.complexity_analyzer.update_user_history(user_id, overall_complexity, quality)
        self.complexity_analyzer.update_category_history(category, overall_complexity, quality)
        
        # 检查是否需要重新训练
        if (len(self.training_data) >= self.config.min_samples_for_learning and 
            self.inference_count % self.config.retrain_interval == 0):
            self._retrain_models()
            
    def _retrain_models(self):
        """重新训练模型"""
        if len(self.training_data) < self.config.min_samples_for_learning:
            return
            
        logger.info(f"🔄 重新训练模型，训练样本数: {len(self.training_data)}")
        
        try:
            # 准备训练数据
            X = np.array([data['features'] for data in self.training_data])
            y_layers = np.array([data['optimal_layers'] for data in self.training_data])
            
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 训练层数预测模型
            self.layer_predictor = RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10
            )
            self.layer_predictor.fit(X_scaled, y_layers)
            
            # 训练性能预测模型
            y_quality = np.array([data['quality'] for data in self.training_data])
            self.performance_predictor = LinearRegression()
            self.performance_predictor.fit(X_scaled, y_quality)
            
            self.is_trained = True
            logger.info("✅ 模型重新训练完成")
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'layer_predictor': self.layer_predictor,
            'performance_predictor': self.performance_predictor,
            'scaler': self.scaler,
            'training_data': self.training_data,
            'performance_history': self.performance_history,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"💾 模型已保存至: {filepath}")
        
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.layer_predictor = model_data['layer_predictor']
            self.performance_predictor = model_data['performance_predictor']
            self.scaler = model_data['scaler']
            self.training_data = model_data['training_data']
            self.performance_history = model_data['performance_history']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"📁 模型已从 {filepath} 加载")
            
        except Exception as e:
            logger.warning(f"模型加载失败: {e}")

class AdvancedDynamicRecommendationSystem:
    """高级动态推荐系统"""
    
    def __init__(self, config: AdvancedLayerConfig = None):
        self.config = config or AdvancedLayerConfig()
        self.layer_selector = PredictiveLayerSelector(self.config)
        
        # 模拟数据
        self.item_catalog = self._create_mock_item_catalog()
        self.user_features = self._create_mock_user_features()
        self.item_features = self._create_mock_item_features()
        
        logger.info("🚀 高级动态推荐系统初始化完成")
        
    def _create_mock_item_catalog(self) -> Dict[str, Dict]:
        """创建模拟物品目录"""
        categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports']
        items = {}
        
        for i in range(2000):
            item_id = f"item_{i:04d}"
            items[item_id] = {
                'title': f"Product {i}",
                'category': np.random.choice(categories),
                'price': np.random.uniform(10, 1000),
                'rating': np.random.uniform(3.0, 5.0),
                'popularity': np.random.exponential(100),
                'complexity_score': np.random.uniform(0.3, 0.9)
            }
            
        return items
        
    def _create_mock_user_features(self) -> Dict[str, torch.Tensor]:
        """创建模拟用户特征"""
        users = {}
        for i in range(200):
            user_id = f"user_{i:03d}"
            users[user_id] = torch.randn(128)  # 更高维特征
        return users
        
    def _create_mock_item_features(self) -> Dict[str, torch.Tensor]:
        """创建模拟物品特征"""
        items = {}
        for item_id in self.item_catalog.keys():
            items[item_id] = torch.randn(128)  # 更高维特征
        return items
        
    def recommend(self, query: str, user_id: str, category: str = 'all',
                 device_type: str = 'cloud', performance_mode: str = 'balanced',
                 top_k: int = 10) -> Dict[str, Any]:
        """执行高级动态推荐"""
        
        # 1. 选择最优层数
        selection_result = self.layer_selector.select_optimal_layers(
            query, user_id, category, device_type, performance_mode
        )
        
        selected_layers = selection_result['selected_layers']
        predicted_performance = selection_result['predicted_performance']
        
        logger.info(f"🎯 用户 {user_id} 查询: '{query[:30]}...'")
        logger.info(f"  🏗️ 选择层数: {selected_layers}")
        logger.info(f"  📊 整体复杂度: {selection_result['complexity_analysis']['overall_complexity']:.3f}")
        logger.info(f"  🔮 预期性能: 延迟 {predicted_performance['expected_latency_ms']:.1f}ms, "
                   f"质量 {predicted_performance['expected_quality']:.3f}")
        
        # 2. 选择候选物品
        candidate_items = self._select_advanced_candidates(query, category, user_id, top_k * 10)
        
        # 3. 执行推理
        start_time = pd.Timestamp.now()
        scores, actual_latency, actual_memory = self._simulate_advanced_inference(
            user_id, candidate_items, selected_layers
        )
        inference_time = (pd.Timestamp.now() - start_time).total_seconds() * 1000
        
        # 4. 排序和选择
        item_scores = list(zip(candidate_items, scores))
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_items = [item for item, score in item_scores[:top_k]]
        top_scores = [score for item, score in item_scores[:top_k]]
        
        # 5. 计算实际质量
        actual_quality = np.mean(top_scores) * self._get_layer_quality_factor(selected_layers)
        
        # 6. 记录性能数据用于学习
        self.layer_selector.record_performance(
            selected_layers, inference_time, actual_memory, actual_quality,
            selection_result['complexity_analysis'], user_id, category
        )
        
        return {
            'items': top_items,
            'scores': top_scores,
            'selected_layers': selected_layers,
            'inference_time_ms': inference_time,
            'memory_usage_mb': actual_memory,
            'actual_quality': actual_quality,
            'selection_analysis': selection_result,
            'predicted_vs_actual': {
                'predicted_latency': predicted_performance['expected_latency_ms'],
                'actual_latency': inference_time,
                'predicted_quality': predicted_performance['expected_quality'],
                'actual_quality': actual_quality
            }
        }
        
    def _select_advanced_candidates(self, query: str, category: str, user_id: str, num_candidates: int) -> List[str]:
        """高级候选物品选择"""
        # 基于查询和类别筛选
        if category != 'all':
            candidates = [
                item_id for item_id, item_info in self.item_catalog.items()
                if item_info['category'].lower() == category.lower()
            ]
        else:
            candidates = list(self.item_catalog.keys())
        
        # 基于查询复杂度优先选择
        query_words = set(query.lower().split())
        scored_candidates = []
        
        for item_id in candidates:
            item = self.item_catalog[item_id]
            # 简化的相关性评分
            title_words = set(item['title'].lower().split())
            relevance = len(query_words.intersection(title_words)) / max(len(query_words), 1)
            
            # 综合评分：相关性 + 流行度 + 评分
            score = relevance * 0.4 + (item['popularity'] / 500) * 0.3 + (item['rating'] / 5) * 0.3
            scored_candidates.append((item_id, score))
        
        # 排序并选择
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [item_id for item_id, score in scored_candidates[:num_candidates]]
        
        return selected
        
    def _simulate_advanced_inference(self, user_id: str, item_ids: List[str], 
                                   num_layers: int) -> Tuple[List[float], float, float]:
        """高级推理模拟"""
        # 获取特征
        user_feat = self.user_features.get(user_id, torch.randn(128))
        item_feats = torch.stack([
            self.item_features.get(item_id, torch.randn(128)) 
            for item_id in item_ids
        ])
        
        # 更复杂的相似度计算
        similarities = torch.cosine_similarity(user_feat.unsqueeze(0), item_feats, dim=1)
        
        # 添加基于物品复杂度的调整
        complexity_adjustments = [
            self.item_catalog.get(item_id, {}).get('complexity_score', 0.5)
            for item_id in item_ids
        ]
        
        adjusted_scores = similarities + torch.tensor(complexity_adjustments) * 0.1
        scores = adjusted_scores.tolist()
        
        # 层数对性能的影响
        layer_performance_factor = self._get_layer_quality_factor(num_layers)
        scores = [s * layer_performance_factor for s in scores]
        
        # 模拟延迟和内存使用
        base_latency = len(item_ids) * 0.1  # 基础延迟
        layer_latency = num_layers * 1.5    # 层级延迟
        total_latency = base_latency + layer_latency + np.random.normal(0, 2)  # 添加噪声
        
        memory_usage = num_layers * 15.0 + np.random.normal(0, 5)  # 内存使用
        
        return scores, max(1, total_latency), max(10, memory_usage)
        
    def _get_layer_quality_factor(self, num_layers: int) -> float:
        """获取层数质量因子"""
        return {4: 0.85, 8: 0.92, 12: 1.0, 16: 1.0, 20: 0.95, 24: 0.90, 32: 0.85}.get(num_layers, 0.9)

def run_advanced_demo():
    """运行高级动态层选择演示"""
    logger.info("🎬 开始高级动态层选择演示...")
    
    # 初始化系统
    config = AdvancedLayerConfig(learning_enabled=True)
    recommender = AdvancedDynamicRecommendationSystem(config)
    
    # 创建更复杂的测试场景
    test_scenarios = [
        ("高端游戏笔记本电脑RTX4090配置推荐", "user_001", "Electronics", "cloud", "accurate"),
        ("轻薄办公笔记本", "user_002", "Electronics", "mobile", "fast"),
        ("专业摄影器材全套配置", "user_003", "Electronics", "cloud", "accurate"),
        ("儿童图书", "user_004", "Books", "mobile", "fast"),
        ("企业级服装采购批量定制", "user_005", "Clothing", "edge", "balanced"),
        ("智能家居全屋解决方案", "user_006", "Home", "cloud", "balanced"),
        ("专业健身器材配套", "user_007", "Sports", "edge", "balanced"),
        ("简单日用品", "user_008", "Home", "mobile", "fast"),
    ]
    
    results = []
    logger.info(f"\n{'='*80}")
    logger.info("🔥 执行高级动态推荐测试")
    logger.info(f"{'='*80}")
    
    for i, (query, user_id, category, device, mode) in enumerate(test_scenarios):
        logger.info(f"\n【测试 {i+1}】{device.upper()} - {mode.upper()}")
        logger.info(f"查询: {query}")
        
        result = recommender.recommend(query, user_id, category, device, mode, top_k=5)
        results.append((query, result))
        
        # 显示预测vs实际对比
        pred_vs_actual = result['predicted_vs_actual']
        logger.info(f"  📊 预测vs实际:")
        logger.info(f"    延迟: {pred_vs_actual['predicted_latency']:.1f}ms → {pred_vs_actual['actual_latency']:.1f}ms")
        logger.info(f"    质量: {pred_vs_actual['predicted_quality']:.3f} → {pred_vs_actual['actual_quality']:.3f}")
        
    # 分析学习效果
    logger.info(f"\n{'='*80}")
    logger.info("📈 学习效果分析")
    logger.info(f"{'='*80}")
    
    logger.info(f"📚 收集训练样本数: {len(recommender.layer_selector.training_data)}")
    logger.info(f"🧠 模型训练状态: {'已训练' if recommender.layer_selector.is_trained else '未训练'}")
    logger.info(f"🔄 推理计数: {recommender.layer_selector.inference_count}")
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('results/advanced_dynamic_selection')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = results_dir / f'advanced_layer_selector_model_{timestamp}.pkl'
    recommender.layer_selector.save_model(str(model_file))
    
    # 保存结果
    demo_results = {
        'timestamp': timestamp,
        'config': {
            'learning_enabled': config.learning_enabled,
            'layer_options': config.layer_options,
            'min_samples_for_learning': config.min_samples_for_learning
        },
        'scenarios': [
            {
                'query': query,
                'result': {k: v for k, v in result.items() if k != 'selection_analysis'}  # 简化保存
            }
            for query, result in results
        ],
        'model_stats': {
            'training_samples': len(recommender.layer_selector.training_data),
            'is_trained': recommender.layer_selector.is_trained,
            'inference_count': recommender.layer_selector.inference_count
        }
    }
    
    results_file = results_dir / f'advanced_demo_results_{timestamp}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(demo_results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n✅ 高级演示完成！")
    logger.info(f"📁 模型已保存至: {model_file}")
    logger.info(f"📊 结果已保存至: {results_file}")

if __name__ == "__main__":
    run_advanced_demo()
