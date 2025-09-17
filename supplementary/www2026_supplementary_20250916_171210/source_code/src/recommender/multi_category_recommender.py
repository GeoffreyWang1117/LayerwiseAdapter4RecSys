#!/usr/bin/env python3
"""
Multi-Category Amazon Recommender
支持多个类别的Amazon推荐系统
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime
from typing import List, Dict

class MultiCategoryRecommender:
    def __init__(self, data_dir: str = "dataset/amazon"):
        self.data_dir = data_dir
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # 按文件大小排序的类别
        self.categories = [
            "All_Beauty", "Movies_and_TV", "Office_Products", 
            "Toys_and_Games", "Arts_Crafts_and_Sewing", "Sports_and_Outdoors"
        ]
    
    def check_category_availability(self):
        """检查可用的数据类别"""
        available = []
        for category in self.categories:
            meta_file = os.path.join(self.data_dir, f"{category}_meta.parquet")
            reviews_file = os.path.join(self.data_dir, f"{category}_reviews.parquet")
            
            if os.path.exists(meta_file) and os.path.exists(reviews_file):
                size_mb = (os.path.getsize(meta_file) + os.path.getsize(reviews_file)) / (1024*1024)
                available.append({
                    'category': category,
                    'size_mb': round(size_mb, 1)
                })
        
        return available
    
    def load_category_sample(self, category: str, max_products: int = 100, max_reviews: int = 200):
        """加载类别样本数据"""
        print(f"\n📊 Loading {category} sample...")
        
        meta_file = os.path.join(self.data_dir, f"{category}_meta.parquet")
        reviews_file = os.path.join(self.data_dir, f"{category}_reviews.parquet")
        
        try:
            # 加载产品
            meta_df = pd.read_parquet(meta_file)
            meta_df = meta_df.dropna(subset=['title']).head(max_products)
            
            # 加载评论
            reviews_df = pd.read_parquet(reviews_file)
            product_ids = set(meta_df['parent_asin'].tolist())
            reviews_df = reviews_df[reviews_df['parent_asin'].isin(product_ids)].head(max_reviews)
            
            print(f"   ✓ {len(meta_df)} products, {len(reviews_df)} reviews")
            return meta_df, reviews_df
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None, None
    
    def simple_user_analysis(self, user_reviews: pd.DataFrame) -> Dict:
        """简单用户分析"""
        if user_reviews.empty:
            return {
                'type': 'new_user',
                'description': 'New user with no history',
                'preferences': 'unknown'
            }
        
        avg_rating = user_reviews['rating'].mean()
        review_count = len(user_reviews)
        
        if avg_rating >= 4.5:
            user_type = 'satisfied'
            description = f'Highly satisfied customer ({review_count} reviews, {avg_rating:.1f}/5)'
        elif avg_rating >= 3.5:
            user_type = 'moderate'
            description = f'Moderate customer ({review_count} reviews, {avg_rating:.1f}/5)'
        else:
            user_type = 'critical'
            description = f'Critical customer ({review_count} reviews, {avg_rating:.1f}/5)'
        
        return {
            'type': user_type,
            'description': description,
            'review_count': review_count,
            'avg_rating': round(avg_rating, 2)
        }
    
    def recommend_products(self, products_df: pd.DataFrame, user_analysis: Dict, top_k: int = 3) -> List[Dict]:
        """推荐产品"""
        # 过滤高质量产品
        quality_products = products_df.copy()
        
        if 'average_rating' in quality_products.columns:
            quality_products = quality_products.dropna(subset=['average_rating'])
            quality_products = quality_products[quality_products['average_rating'] >= 3.0]
        
        # 根据用户类型调整推荐策略
        if user_analysis['type'] == 'critical':
            # 为挑剔用户推荐高评分产品
            quality_products = quality_products.sort_values('average_rating', ascending=False)
        elif user_analysis['type'] == 'satisfied':
            # 为满意用户推荐多样化产品
            quality_products = quality_products.sample(frac=1, random_state=42)
        
        recommendations = []
        candidates = quality_products.head(top_k).to_dict('records')
        
        for i, product in enumerate(candidates):
            title = str(product.get('title', 'Unknown'))[:60]
            price = product.get('price', 'N/A')
            rating = product.get('average_rating', 'N/A')
            
            # 生成推荐理由
            if user_analysis['type'] == 'critical':
                reason = f"High-quality product for discerning customers"
            elif isinstance(rating, (int, float)) and rating >= 4.0:
                reason = f"Popular choice with {rating}/5 rating"
            else:
                reason = f"Good value product in category"
            
            recommendations.append({
                'rank': i + 1,
                'title': title,
                'price': price,
                'rating': rating,
                'reason': reason,
                'asin': product.get('parent_asin', '')
            })
        
        return recommendations
    
    def run_category_demo(self, category: str, num_users: int = 2):
        """运行类别演示"""
        print(f"🎯 Category Demo: {category}")
        print("=" * 50)
        
        # 加载数据
        products_df, reviews_df = self.load_category_sample(category)
        if products_df is None:
            return None
        
        if reviews_df.empty:
            print("   ⚠️ No reviews available")
            return None
        
        # 选择用户
        user_counts = reviews_df['user_id'].value_counts()
        selected_users = user_counts.head(num_users).index
        
        results = []
        
        for i, user_id in enumerate(selected_users):
            print(f"\n👤 User {i+1}: {user_id[:20]}...")
            
            user_reviews = reviews_df[reviews_df['user_id'] == user_id]
            user_analysis = self.simple_user_analysis(user_reviews)
            
            print(f"   📝 {user_analysis['description']}")
            
            recommendations = self.recommend_products(products_df, user_analysis, top_k=3)
            
            print(f"   🎯 Recommendations:")
            for rec in recommendations:
                print(f"      {rec['rank']}. {rec['title']}")
                print(f"         💰 ${rec['price']} | ⭐ {rec['rating']} | {rec['reason']}")
            
            results.append({
                'user_id': user_id,
                'category': category,
                'user_analysis': user_analysis,
                'recommendations': recommendations
            })
        
        return results
    
    def run_multi_category_demo(self):
        """运行多类别演示"""
        print("🚀 Multi-Category Amazon Recommendation Demo")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 检查可用类别
        available_categories = self.check_category_availability()
        print(f"\n📦 Available Categories:")
        for cat_info in available_categories:
            print(f"   • {cat_info['category']}: {cat_info['size_mb']} MB")
        
        if not available_categories:
            print("❌ No categories available")
            return
        
        # 运行多个类别的演示
        all_results = []
        
        # 选择前3个最小的类别
        demo_categories = [cat['category'] for cat in available_categories[:3]]
        
        for category in demo_categories:
            results = self.run_category_demo(category, num_users=2)
            if results:
                all_results.extend(results)
        
        # 保存结果
        if all_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"multi_category_recommendations_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\n✅ Multi-category demo completed!")
            print(f"💾 Results saved to: {output_file}")
            print(f"📊 Total recommendations: {len(all_results)} users across {len(demo_categories)} categories")
            
            # 统计摘要
            category_counts = {}
            for result in all_results:
                cat = result['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            print(f"\n📈 Summary by category:")
            for cat, count in category_counts.items():
                print(f"   • {cat}: {count} users")
        
        return all_results

def main():
    """主函数"""
    recommender = MultiCategoryRecommender()
    results = recommender.run_multi_category_demo()
    
    if results:
        print(f"\n🎉 Demo successful! Generated recommendations across multiple categories.")
    else:
        print(f"\n❌ Demo failed.")

if __name__ == "__main__":
    main()
