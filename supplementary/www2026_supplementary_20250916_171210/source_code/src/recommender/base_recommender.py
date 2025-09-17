#!/usr/bin/env python3
"""
Base Recommender for Layerwise Knowledge Distillation
基于Fisher信息矩阵的层级知识蒸馏推荐系统基类

WWW2026研究目标：
- 使用Llama3作为Teacher模型（经验证性能最优）
- 集成Fisher信息计算用于层级权重分配
- 支持多种推荐任务的知识蒸馏
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import random
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RecommendationConfig:
    """推荐系统配置"""
    # 模型配置
    teacher_model: str = "llama3:latest"  # WWW2026指定的最优Teacher模型
    ollama_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.1              # 低温度确保一致性
    max_tokens: int = 512
    timeout: int = 30
    
    # 推荐参数
    top_k: int = 5                        # 推荐数量
    candidate_size: int = 50              # 候选集大小
    user_profile_max_length: int = 500    # 用户画像最大长度
    
    # Fisher集成参数
    use_fisher_weighting: bool = True     # 是否使用Fisher权重
    fisher_temperature: float = 2.0       # Fisher权重温度
    semantic_boost: float = 1.5           # 语义层增强
    
    # 数据配置
    data_dir: str = "dataset/amazon"
    sample_size: int = 1000
    cache_dir: str = "cache/recommendations"

class BaseRecommender(ABC):
    """
    基础推荐器抽象类
    
    WWW2026框架设计：
    - 标准化Teacher模型接口（Llama3）
    - 集成Fisher信息权重计算
    - 支持层级知识蒸馏流程
    """
    
    def __init__(self, config: Optional[RecommendationConfig] = None):
        """
        初始化基础推荐器
        
        Args:
            config: 推荐配置
        """
        self.config = config or RecommendationConfig()
        
        # 数据缓存
        self.products_cache = {}
        self.reviews_cache = {}
        self.user_profiles_cache = {}
        
        # 可用的Amazon数据集类别
        self.categories = [
            "All_Beauty", "Arts_Crafts_and_Sewing", "Automotive", 
            "Books", "Electronics", "Home_and_Kitchen",
            "Movies_and_TV", "Office_Products", "Sports_and_Outdoors", 
            "Toys_and_Games"
        ]
        
        # Fisher权重缓存（用于蒸馏）
        self.fisher_weights_cache = {}
        
        # 初始化缓存目录
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"基础推荐器初始化完成 - Teacher模型: {self.config.teacher_model}")
    
    @abstractmethod
    def generate_recommendations(self, 
                               user_profile: str, 
                               category: str, 
                               top_k: int = None) -> List[Dict]:
        """
        生成推荐结果（抽象方法）
        
        Args:
            user_profile: 用户画像
            category: 产品类别  
            top_k: 推荐数量
            
        Returns:
            推荐结果列表
        """
        pass
    
    @abstractmethod
    def compute_recommendation_features(self, 
                                     user_profile: str, 
                                     items: List[Dict]) -> torch.Tensor:
        """
        计算推荐特征（用于Fisher信息计算）
        
        Args:
            user_profile: 用户画像
            items: 候选物品列表
            
        Returns:
            特征张量
        """
        pass
        
    def load_sample_data(self, category: str, sample_size: int = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        加载指定类别的样本数据
        
        Args:
            category: 数据类别
            sample_size: 采样大小
            
        Returns:
            (products_df, reviews_df): 产品和评论数据
        """
        sample_size = sample_size or self.config.sample_size
        logger.info(f"加载 {category} 类别样本数据 (样本大小: {sample_size})")
        
        # 检查缓存
        if category in self.products_cache and category in self.reviews_cache:
            logger.info(f"使用缓存数据: {category}")
            return self.products_cache[category], self.reviews_cache[category]
        
        # 文件路径
        meta_file = Path(self.config.data_dir) / f"{category}_meta.parquet"
        reviews_file = Path(self.config.data_dir) / f"{category}_reviews.parquet"
        
        if not meta_file.exists() or not reviews_file.exists():
            logger.error(f"数据文件不存在: {category}")
            return None, None
            
        try:
            # 加载和采样产品数据
            meta_df = pd.read_parquet(meta_file)
            if len(meta_df) > sample_size:
                meta_df = meta_df.sample(n=sample_size, random_state=42)
            
            # 加载相关评论数据
            product_ids = set(meta_df['parent_asin'].tolist())
            reviews_df = pd.read_parquet(reviews_file)
            reviews_df = reviews_df[reviews_df['parent_asin'].isin(product_ids)]
            
            # 限制评论数据大小
            max_reviews = sample_size * 3
            if len(reviews_df) > max_reviews:
                reviews_df = reviews_df.sample(n=max_reviews, random_state=42)
            
            # 缓存数据
            self.products_cache[category] = meta_df
            self.reviews_cache[category] = reviews_df
            
            logger.info(f"成功加载 {len(meta_df)} 个产品和 {len(reviews_df)} 条评论")
            return meta_df, reviews_df
            
        except Exception as e:
            logger.error(f"加载数据失败 {category}: {e}")
            return None, None
    
    def query_teacher_model(self, 
                          prompt: str, 
                          max_tokens: int = None,
                          temperature: float = None) -> str:
        """
        查询Teacher模型 (Llama3)
        
        WWW2026指定：使用Llama3作为性能最优的Teacher模型
        
        Args:
            prompt: 输入提示
            max_tokens: 最大token数  
            temperature: 温度参数
            
        Returns:
            模型回复
        """
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        try:
            payload = {
                "model": self.config.teacher_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            response = requests.post(
                self.config.ollama_url, 
                json=payload, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Teacher模型查询失败: {e}")
            return ""
    
    def compute_fisher_informed_score(self, 
                                    user_profile: str, 
                                    item: Dict, 
                                    fisher_weights: Optional[torch.Tensor] = None) -> float:
        """
        计算Fisher信息增强的推荐得分
        
        WWW2026创新点：
        将Fisher权重应用于推荐得分计算，突出高层语义信息的贡献
        
        Args:
            user_profile: 用户画像
            item: 候选物品
            fisher_weights: Fisher权重（可选）
            
        Returns:
            Fisher增强的推荐得分
        """
        if not self.config.use_fisher_weighting:
            return self._compute_base_score(user_profile, item)
        
        # 获取或计算Fisher权重
        if fisher_weights is None:
            fisher_weights = self._get_cached_fisher_weights()
        
        # 基础推荐得分
        base_score = self._compute_base_score(user_profile, item)
        
        # 提取语义层特征（模拟高层语义匹配）
        semantic_features = self._extract_semantic_features(user_profile, item)
        
        # 应用Fisher权重（重点增强高层语义）
        if len(fisher_weights) > 0:
            # 使用高层权重（后50%层）增强语义得分
            high_layer_weights = fisher_weights[len(fisher_weights)//2:]
            semantic_boost = high_layer_weights.mean().item() * self.config.semantic_boost
            
            # Fisher增强得分
            fisher_score = base_score * (1.0 + semantic_boost * semantic_features)
        else:
            fisher_score = base_score
        
        return max(0.0, min(fisher_score, 1.0))  # 归一化到[0,1]
    
    def _compute_base_score(self, user_profile: str, item: Dict) -> float:
        """计算基础推荐得分"""
        # 简化版本：基于关键词匹配
        item_text = f"{item.get('title', '')} {item.get('description', '')}"
        
        # 文本相似度（简化）
        user_words = set(user_profile.lower().split())
        item_words = set(item_text.lower().split())
        
        if len(user_words) == 0 or len(item_words) == 0:
            return 0.0
        
        overlap = len(user_words.intersection(item_words))
        similarity = overlap / (len(user_words) + len(item_words) - overlap + 1e-6)
        
        # 考虑物品评分
        rating_boost = float(item.get('average_rating', 3.0)) / 5.0
        
        return similarity * 0.7 + rating_boost * 0.3
    
    def _extract_semantic_features(self, user_profile: str, item: Dict) -> float:
        """
        提取语义特征匹配度
        
        模拟高层语义理解：用户偏好与物品特性的深层匹配
        """
        # 简化版本：基于文本复杂度和情感匹配
        user_complexity = len(set(user_profile.lower().split())) / max(len(user_profile.split()), 1)
        
        item_text = f"{item.get('title', '')} {item.get('description', '')}"
        item_complexity = len(set(item_text.lower().split())) / max(len(item_text.split()), 1)
        
        # 语义复杂度匹配
        semantic_match = 1.0 - abs(user_complexity - item_complexity)
        
        return max(0.0, min(semantic_match, 1.0))
    
    def _get_cached_fisher_weights(self) -> torch.Tensor:
        """获取缓存的Fisher权重"""
        cache_key = f"fisher_weights_{self.config.teacher_model}"
        
        if cache_key not in self.fisher_weights_cache:
            # 生成默认Fisher权重（基于理论假设）
            num_layers = 32  # Llama3层数
            layer_weights = []
            
            for i in range(num_layers):
                # 上层权重更大
                depth_ratio = i / (num_layers - 1)
                weight = 0.5 + depth_ratio * 1.5  # 0.5到2.0的权重
                layer_weights.append(weight)
            
            self.fisher_weights_cache[cache_key] = torch.tensor(layer_weights)
        
        return self.fisher_weights_cache[cache_key]
    
    def generate_user_profile(self, user_reviews: pd.DataFrame) -> str:
        """
        基于用户评论生成用户画像
        
        Args:
            user_reviews: 用户的评论数据
            
        Returns:
            用户画像描述
        """
        if user_reviews.empty:
            return "No previous reviews available"
        
        # 提取关键信息
        avg_rating = user_reviews['rating'].mean()
        review_count = len(user_reviews)
        recent_reviews = user_reviews.head(3)['text'].tolist()
        
        # 构建prompt
        prompt = f"""Based on the following user review history, create a brief user profile:

User Statistics:
- Average Rating: {avg_rating:.1f}/5.0
- Total Reviews: {review_count}

Recent Reviews:
{chr(10).join([f"- {review[:100]}..." for review in recent_reviews])}

Please provide a concise user profile (2-3 sentences) describing their preferences and shopping patterns:"""

        return self.query_ollama(prompt, max_tokens=150)
    
    def recommend_products(self, category: str, user_profile: str, top_k: int = 5) -> List[Dict]:
        """
        为用户推荐产品
        
        Args:
            category: 产品类别
            user_profile: 用户画像
            top_k: 推荐产品数量
            
        Returns:
            推荐产品列表
        """
        if category not in self.products_cache:
            print(f"❌ No data loaded for category: {category}")
            return []
        
        products_df = self.products_cache[category]
        
        # 随机选择一些产品作为候选
        candidate_products = products_df.sample(n=min(20, len(products_df))).to_dict('records')
        
        # 构建推荐prompt
        products_info = []
        for i, product in enumerate(candidate_products):
            title = str(product.get('title', 'Unknown'))[:100]
            price = product.get('price', 'N/A')
            rating = product.get('average_rating', 'N/A')
            products_info.append(f"{i+1}. {title} (Price: {price}, Rating: {rating})")
        
        prompt = f"""You are an AI shopping assistant. Based on the user profile below, recommend the top {top_k} products from the following list:

User Profile: {user_profile}

Available Products:
{chr(10).join(products_info)}

Please recommend the top {top_k} products that best match this user's preferences. 
Respond with just the product numbers (e.g., "1, 3, 7, 12, 15") and a brief explanation for each choice:"""

        response = self.query_ollama(prompt, max_tokens=300)
        
        # 解析推荐结果
        recommendations = []
        try:
            # 简单解析数字
            import re
            numbers = re.findall(r'\b(\d+)\b', response)
            recommended_indices = [int(n) - 1 for n in numbers[:top_k] if 0 <= int(n) - 1 < len(candidate_products)]
            
            for idx in recommended_indices:
                product = candidate_products[idx]
                recommendations.append({
                    'asin': product.get('parent_asin', ''),
                    'title': product.get('title', 'Unknown'),
                    'price': product.get('price', 'N/A'),
                    'rating': product.get('average_rating', 'N/A'),
                    'explanation': f"Selected based on user profile compatibility"
                })
                
        except Exception as e:
            print(f"⚠️ Error parsing recommendations: {e}")
            # 回退到随机推荐
            for i in range(min(top_k, len(candidate_products))):
                product = candidate_products[i]
                recommendations.append({
                    'asin': product.get('parent_asin', ''),
                    'title': product.get('title', 'Unknown'),
                    'price': product.get('price', 'N/A'),
                    'rating': product.get('average_rating', 'N/A'),
                    'explanation': "Random selection due to parsing error"
                })
        
        return recommendations
    
    def run_recommendation_demo(self, category: str = "Electronics", num_users: int = 3):
        """
        运行推荐演示
        
        Args:
            category: 要演示的产品类别
            num_users: 模拟用户数量
        """
        print(f"🚀 Starting Amazon Ollama Recommendation Demo")
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📦 Category: {category}")
        print("=" * 60)
        
        # 加载数据
        products_df, reviews_df = self.load_sample_data(category, sample_size=500)
        if products_df is None or reviews_df is None:
            print("❌ Failed to load data")
            return
        
        # 模拟用户推荐
        unique_users = reviews_df['user_id'].unique()
        selected_users = np.random.choice(unique_users, min(num_users, len(unique_users)), replace=False)
        
        results = []
        
        for i, user_id in enumerate(selected_users):
            print(f"\n👤 User {i+1}/{num_users} (ID: {user_id})")
            print("-" * 40)
            
            # 获取用户历史评论
            user_reviews = reviews_df[reviews_df['user_id'] == user_id]
            print(f"📝 Historical reviews: {len(user_reviews)}")
            
            # 生成用户画像
            print("🧠 Generating user profile...")
            user_profile = self.generate_user_profile(user_reviews)
            print(f"👤 User Profile: {user_profile}")
            
            # 生成推荐
            print("🎯 Generating recommendations...")
            recommendations = self.recommend_products(category, user_profile, top_k=3)
            
            print(f"\n📋 Top 3 Recommendations:")
            for j, rec in enumerate(recommendations):
                print(f"  {j+1}. {rec['title'][:80]}...")
                print(f"     Price: {rec['price']}, Rating: {rec['rating']}")
                print(f"     Reason: {rec['explanation']}")
                print()
            
            results.append({
                'user_id': user_id,
                'user_profile': user_profile,
                'recommendations': recommendations,
                'historical_reviews': len(user_reviews)
            })
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"amazon_recommendations_{category}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Demo completed! Results saved to {output_file}")
        print(f"📊 Generated recommendations for {len(results)} users")
        
        return results

class Llama3Recommender(BaseRecommender):
    """
    Llama3推荐器实现
    
    WWW2026核心：基于Llama3的Fisher信息驱动推荐系统
    """
    
    def __init__(self, config: Optional[RecommendationConfig] = None):
        super().__init__(config)
        logger.info("Llama3推荐器初始化完成")
    
    def generate_recommendations(self, 
                               user_profile: str, 
                               category: str, 
                               top_k: int = None) -> List[Dict]:
        """
        生成Llama3驱动的推荐结果
        
        Args:
            user_profile: 用户画像
            category: 产品类别  
            top_k: 推荐数量
            
        Returns:
            推荐结果列表
        """
        top_k = top_k or self.config.top_k
        
        # 加载候选数据
        products_df, _ = self.load_sample_data(category)
        if products_df is None:
            logger.error(f"无法加载类别数据: {category}")
            return []
        
        # 选择候选物品
        candidates = products_df.sample(
            n=min(self.config.candidate_size, len(products_df))
        ).to_dict('records')
        
        # 使用Fisher增强得分排序
        scored_items = []
        for item in candidates:
            score = self.compute_fisher_informed_score(user_profile, item)
            scored_items.append((score, item))
        
        # 排序并选择Top-K
        scored_items.sort(key=lambda x: x[0], reverse=True)
        top_items = scored_items[:top_k]
        
        # 使用Llama3生成最终推荐解释
        recommendations = []
        for score, item in top_items:
            explanation = self._generate_explanation(user_profile, item, score)
            
            recommendations.append({
                'asin': item.get('parent_asin', ''),
                'title': item.get('title', 'Unknown'),
                'price': item.get('price', 'N/A'),
                'rating': item.get('average_rating', 'N/A'),
                'fisher_score': score,
                'explanation': explanation
            })
        
        return recommendations
    
    def compute_recommendation_features(self, 
                                     user_profile: str, 
                                     items: List[Dict]) -> torch.Tensor:
        """
        计算推荐特征向量（用于Fisher信息计算）
        
        Args:
            user_profile: 用户画像
            items: 候选物品列表
            
        Returns:
            特征张量 [num_items, feature_dim]
        """
        features = []
        
        for item in items:
            # 基础特征
            price_feature = self._normalize_price(item.get('price', 0))
            rating_feature = float(item.get('average_rating', 3.0)) / 5.0
            
            # 文本特征（简化）
            text_similarity = self._compute_base_score(user_profile, item)
            
            # 语义特征
            semantic_feature = self._extract_semantic_features(user_profile, item)
            
            # 组合特征向量
            feature_vector = [
                price_feature, 
                rating_feature, 
                text_similarity, 
                semantic_feature
            ]
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _generate_explanation(self, user_profile: str, item: Dict, score: float) -> str:
        """
        使用Llama3生成推荐解释
        """
        prompt = f"""基于用户画像为推荐结果生成简洁解释。

用户画像：{user_profile[:200]}

推荐商品：{item.get('title', 'Unknown')}
商品评分：{item.get('average_rating', 'N/A')}
推荐得分：{score:.3f}

请用1-2句话解释为什么推荐这个商品给该用户："""

        explanation = self.query_teacher_model(prompt, max_tokens=100, temperature=0.1)
        
        if not explanation:
            return f"基于用户偏好匹配，推荐得分: {score:.3f}"
        
        return explanation[:200]  # 限制长度
    
    def _normalize_price(self, price) -> float:
        """归一化价格特征"""
        try:
            if isinstance(price, str):
                # 提取数字
                import re
                numbers = re.findall(r'\d+\.?\d*', price)
                if numbers:
                    price = float(numbers[0])
                else:
                    return 0.5  # 默认中等价格
            
            price = float(price)
            # 简单归一化到[0,1]，假设价格范围0-1000
            return min(price / 1000.0, 1.0)
        except:
            return 0.5

def main():
    """主函数：演示Llama3推荐器"""
    logger.info("🎯 Llama3 Fisher信息驱动推荐系统")
    
    # 检查ollama服务
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama服务运行正常")
        else:
            logger.error("❌ Ollama服务错误")
            return
    except:
        logger.error("❌ 无法连接Ollama服务，请先启动ollama")
        return
    
    # 初始化Llama3推荐器
    config = RecommendationConfig(
        teacher_model="llama3:latest",
        use_fisher_weighting=True,
        top_k=3
    )
    recommender = Llama3Recommender(config)
    
    # 演示推荐
    demo_user_profile = "用户喜欢高科技产品，注重性能和创新，经常购买电子设备，偏爱知名品牌。"
    demo_category = "Electronics"
    
    logger.info(f"演示用户画像: {demo_user_profile}")
    logger.info(f"演示类别: {demo_category}")
    
    recommendations = recommender.generate_recommendations(
        user_profile=demo_user_profile,
        category=demo_category,
        top_k=3
    )
    
    if recommendations:
        logger.info("🎉 推荐结果生成成功!")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. {rec['title'][:50]}...")
            logger.info(f"   Fisher得分: {rec['fisher_score']:.3f}")
            logger.info(f"   解释: {rec['explanation']}")
            logger.info("")
    else:
        logger.error("推荐生成失败")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
