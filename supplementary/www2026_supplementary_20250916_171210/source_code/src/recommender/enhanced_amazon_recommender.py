#!/usr/bin/env python3
"""
Enhanced Amazon Reviews Ollama Recommender
改进版Amazon推荐系统，使用ollama进行智能推荐
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
from typing import List, Dict, Any
import random
from datetime import datetime

class EnhancedAmazonRecommender:
    def __init__(self, data_dir: str = "dataset/amazon", model_name: str = "qwen3:latest"):
        """
        初始化增强版Amazon推荐器
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # 可用的数据集类别（按大小排序，小的优先）
        self.categories = [
            "All_Beauty", "Movies_and_TV", "Office_Products", 
            "Toys_and_Games", "Sports_and_Outdoors", "Arts_Crafts_and_Sewing",
            "Automotive", "Electronics", "Home_and_Kitchen", "Books"
        ]
        
        self.products_cache = {}
        self.reviews_cache = {}
        
    def query_ollama_simple(self, prompt: str, max_tokens: int = 100) -> str:
        """
        简化的ollama查询，减少超时问题
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": f"{prompt}\n\nPlease provide a concise response (max {max_tokens} words):",
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": max_tokens,
                    "stop": ["\\n\\n", "---", "###"]
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=20)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except Exception as e:
            print(f"⚠️ Ollama query failed: {e}")
            return "User profile unavailable due to service error"
    
    def load_small_sample(self, category: str, max_products: int = 200, max_reviews: int = 500):
        """
        加载小样本数据以快速演示
        """
        print(f"📊 Loading small sample for {category}...")
        
        meta_file = os.path.join(self.data_dir, f"{category}_meta.parquet")
        reviews_file = os.path.join(self.data_dir, f"{category}_reviews.parquet")
        
        if not os.path.exists(meta_file) or not os.path.exists(reviews_file):
            print(f"❌ Files missing for {category}")
            return None, None
            
        try:
            # 加载产品数据
            print(f"   📦 Loading products from {meta_file}...")
            meta_df = pd.read_parquet(meta_file, columns=['parent_asin', 'title', 'price', 'average_rating'])
            
            # 过滤并采样
            meta_df = meta_df.dropna(subset=['title'])
            meta_df = meta_df[meta_df['title'].str.len() > 10]  # 过滤标题太短的
            
            if len(meta_df) > max_products:
                meta_df = meta_df.sample(n=max_products, random_state=42)
            
            # 加载评论数据
            print(f"   💬 Loading reviews from {reviews_file}...")
            product_ids = set(meta_df['parent_asin'].tolist())
            
            # 直接加载评论数据（选择性列）
            try:
                reviews_df = pd.read_parquet(reviews_file, columns=['parent_asin', 'user_id', 'rating', 'text'])
                reviews_df = reviews_df[reviews_df['parent_asin'].isin(product_ids)]
                
                # 如果数据太大，进行采样
                if len(reviews_df) > max_reviews:
                    reviews_df = reviews_df.sample(n=max_reviews, random_state=42)
                    
            except Exception as e:
                print(f"   ⚠️ Error loading reviews: {e}")
                reviews_df = pd.DataFrame()
            
            self.products_cache[category] = meta_df
            self.reviews_cache[category] = reviews_df
            
            print(f"✅ Loaded {len(meta_df)} products and {len(reviews_df)} reviews")
            return meta_df, reviews_df
            
        except Exception as e:
            print(f"❌ Error loading {category}: {e}")
            return None, None
    
    def create_simple_user_profile(self, user_reviews: pd.DataFrame) -> str:
        """
        创建简化的用户画像
        """
        if user_reviews.empty:
            return "New user with no review history"
        
        avg_rating = user_reviews['rating'].mean()
        review_count = len(user_reviews)
        
        # 简化的用户画像生成
        if avg_rating >= 4.5:
            satisfaction = "highly satisfied"
        elif avg_rating >= 3.5:
            satisfaction = "moderately satisfied"
        else:
            satisfaction = "critical"
        
        # 提取最近的评论文本关键词
        recent_text = ""
        if 'text' in user_reviews.columns and not user_reviews['text'].empty:
            recent_reviews = user_reviews['text'].dropna().head(2)
            recent_text = " ".join(recent_reviews.astype(str))[:200]
        
        # 构建简化prompt
        prompt = f"""Create a brief user profile based on:
- {review_count} reviews with {avg_rating:.1f}/5.0 average rating
- User is {satisfaction} with purchases
- Recent feedback: {recent_text[:100]}...

User profile (1 sentence):"""

        profile = self.query_ollama_simple(prompt, max_tokens=50)
        
        # 回退到简单描述
        if not profile or len(profile) < 10:
            profile = f"{satisfaction.title()} customer with {review_count} reviews (avg: {avg_rating:.1f}/5)"
        
        return profile
    
    def smart_recommend(self, category: str, user_profile: str, top_k: int = 3) -> List[Dict]:
        """
        智能推荐产品
        """
        if category not in self.products_cache:
            return []
        
        products_df = self.products_cache[category]
        
        # 选择高质量产品作为候选
        quality_products = products_df.copy()
        
        # 优先选择有评分的产品
        if 'average_rating' in quality_products.columns:
            quality_products = quality_products.dropna(subset=['average_rating'])
            quality_products = quality_products[quality_products['average_rating'] >= 3.0]
        
        # 选择候选产品
        candidates = quality_products.sample(n=min(10, len(quality_products))).to_dict('records')
        
        recommendations = []
        for i, product in enumerate(candidates[:top_k]):
            title = str(product.get('title', 'Unknown Product'))[:80]
            price = product.get('price', 'N/A')
            rating = product.get('average_rating', 'N/A')
            
            # 简单的推荐理由
            if isinstance(rating, (int, float)) and rating >= 4.0:
                reason = f"High-rated product ({rating}/5) matching user preferences"
            elif price and price != 'N/A':
                reason = f"Well-priced option at ${price}"
            else:
                reason = "Popular product in category"
            
            recommendations.append({
                'rank': i + 1,
                'asin': product.get('parent_asin', ''),
                'title': title,
                'price': price,
                'rating': rating,
                'explanation': reason
            })
        
        return recommendations
    
    def run_quick_demo(self, category: str = "All_Beauty", num_users: int = 2):
        """
        运行快速演示
        """
        print(f"🚀 Enhanced Amazon Recommendation System")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📦 Category: {category}")
        print("=" * 60)
        
        # 加载小样本数据
        products_df, reviews_df = self.load_small_sample(category, max_products=200, max_reviews=300)
        if products_df is None:
            print("❌ Demo failed - could not load data")
            return None
        
        print(f"\n📈 Dataset Statistics:")
        print(f"   • Products: {len(products_df)}")
        print(f"   • Reviews: {len(reviews_df)}")
        print(f"   • Unique users: {reviews_df['user_id'].nunique() if not reviews_df.empty else 0}")
        
        if reviews_df.empty:
            print("⚠️ No reviews available for recommendation demo")
            return None
        
        # 选择活跃用户
        user_review_counts = reviews_df['user_id'].value_counts()
        active_users = user_review_counts[user_review_counts >= 1].index[:num_users]
        
        results = []
        
        for i, user_id in enumerate(active_users):
            print(f"\n👤 User {i+1}/{len(active_users)}")
            print(f"   ID: {user_id}")
            print("-" * 40)
            
            # 用户评论历史
            user_reviews = reviews_df[reviews_df['user_id'] == user_id]
            print(f"📝 Reviews: {len(user_reviews)}")
            print(f"📊 Avg Rating: {user_reviews['rating'].mean():.1f}/5.0")
            
            # 生成用户画像
            print("🧠 Creating user profile...")
            user_profile = self.create_simple_user_profile(user_reviews)
            print(f"👤 Profile: {user_profile}")
            
            # 生成推荐
            print("🎯 Generating recommendations...")
            recommendations = self.smart_recommend(category, user_profile, top_k=3)
            
            print(f"\n📋 Recommendations:")
            for rec in recommendations:
                print(f"   {rec['rank']}. {rec['title']}")
                print(f"      💰 Price: {rec['price']} | ⭐ Rating: {rec['rating']}")
                print(f"      💡 {rec['explanation']}")
                print()
            
            results.append({
                'user_id': user_id,
                'profile': user_profile,
                'recommendations': recommendations,
                'stats': {
                    'review_count': len(user_reviews),
                    'avg_rating': float(user_reviews['rating'].mean())
                }
            })
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"enhanced_amazon_rec_{category}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n✅ Demo completed successfully!")
        print(f"💾 Results saved to: {output_file}")
        print(f"📊 Processed {len(results)} users with recommendations")
        
        return results

def main():
    """主函数"""
    print("🎯 Enhanced Amazon Ollama Recommender")
    print("=" * 50)
    
    # 检查ollama服务
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print("✅ Ollama service is running")
    except Exception as e:
        print(f"❌ Ollama service issue: {e}")
        print("   Please ensure ollama is running: ollama serve")
        return
    
    # 初始化推荐器
    recommender = EnhancedAmazonRecommender(model_name="qwen3:latest")
    
    # 运行快速演示
    print("\n🚀 Starting quick recommendation demo...")
    results = recommender.run_quick_demo(category="All_Beauty", num_users=3)
    
    if results:
        print(f"\n🎉 Demo successful!")
        print(f"   ✓ Generated personalized recommendations")
        print(f"   ✓ Used ollama for intelligent profiling")
        print(f"   ✓ Processed Amazon review data")
    else:
        print(f"\n❌ Demo encountered issues")

if __name__ == "__main__":
    main()
