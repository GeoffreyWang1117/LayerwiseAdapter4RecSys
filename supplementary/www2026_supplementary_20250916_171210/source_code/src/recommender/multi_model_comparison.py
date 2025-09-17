#!/usr/bin/env python3
"""
Multi-Model Amazon Recommender Comparison
使用多个ollama模型进行Amazon推荐对比实验
支持: qwen3, llama3, gpt-oss
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple
import concurrent.futures

class MultiModelRecommender:
    def __init__(self, data_dir: str = "../dataset/amazon"):
        self.data_dir = data_dir
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # 可用模型配置
        self.models = {
            "qwen3": {
                "name": "qwen3:latest",
                "description": "Qwen3 - Chinese-optimized model",
                "temperature": 0.7,
                "max_tokens": 150
            },
            "llama3": {
                "name": "llama3:latest", 
                "description": "Llama3 - Meta's advanced model",
                "temperature": 0.6,
                "max_tokens": 150
            },
            "gpt-oss": {
                "name": "gpt-oss:latest",
                "description": "GPT-OSS - Open source GPT variant",
                "temperature": 0.8,
                "max_tokens": 150
            }
        }
        
        # 测试类别（按大小排序）
        self.test_categories = ["All_Beauty", "Movies_and_TV", "Office_Products"]
        
    def check_model_availability(self) -> Dict:
        """检查模型可用性"""
        print("🔍 Checking model availability...")
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                return {}
                
            available_models = response.json().get("models", [])
            model_names = [model["name"] for model in available_models]
            
            available = {}
            for key, config in self.models.items():
                if config["name"] in model_names:
                    available[key] = config
                    print(f"   ✅ {key}: {config['description']}")
                else:
                    print(f"   ❌ {key}: Not available")
                    
            return available
            
        except Exception as e:
            print(f"   ❌ Error checking models: {e}")
            return {}
    
    def query_model(self, model_key: str, prompt: str) -> Tuple[str, float]:
        """查询指定模型"""
        if model_key not in self.models:
            return "Model not available", 0.0
            
        model_config = self.models[model_key]
        start_time = time.time()
        
        try:
            payload = {
                "model": model_config["name"],
                "prompt": f"{prompt}\n\nProvide a concise response (max 100 words):",
                "stream": False,
                "options": {
                    "temperature": model_config["temperature"],
                    "num_predict": model_config["max_tokens"],
                    "stop": ["\\n\\n", "---"]
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=25)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '').strip()
            
            # 清理响应（移除思考标记等）
            if response_text.startswith('<think>'):
                # 尝试提取实际回复
                end_think = response_text.find('</think>')
                if end_think != -1:
                    response_text = response_text[end_think + 8:].strip()
                else:
                    # 如果没有结束标记，取第一个换行后的内容
                    lines = response_text.split('\n')
                    for line in lines:
                        if line.strip() and not line.strip().startswith('<'):
                            response_text = line.strip()
                            break
            
            response_time = time.time() - start_time
            return response_text[:200], response_time  # 限制长度
            
        except Exception as e:
            response_time = time.time() - start_time
            return f"Error: {str(e)[:100]}", response_time
    
    def load_category_data(self, category: str, max_items: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载类别数据"""
        meta_file = os.path.join(self.data_dir, f"{category}_meta.parquet")
        reviews_file = os.path.join(self.data_dir, f"{category}_reviews.parquet")
        
        try:
            # 加载产品数据
            meta_df = pd.read_parquet(meta_file)
            meta_df = meta_df.dropna(subset=['title']).head(max_items)
            
            # 加载评论数据
            reviews_df = pd.read_parquet(reviews_file)
            product_ids = set(meta_df['parent_asin'].tolist())
            reviews_df = reviews_df[reviews_df['parent_asin'].isin(product_ids)].head(max_items * 2)
            
            return meta_df, reviews_df
            
        except Exception as e:
            print(f"Error loading {category}: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def create_user_profile_prompts(self, user_reviews: pd.DataFrame) -> str:
        """创建用户画像提示"""
        if user_reviews.empty:
            return "New user with no purchase history. Recommend popular, well-rated products."
        
        avg_rating = user_reviews['rating'].mean()
        review_count = len(user_reviews)
        
        # 构建提示
        prompt = f"""Analyze this customer profile and suggest their shopping preferences:

Customer Statistics:
- Total reviews: {review_count}
- Average rating given: {avg_rating:.1f}/5.0

Based on this data, what type of shopper is this customer? Describe their likely preferences in 1-2 sentences:"""

        return prompt
    
    def create_recommendation_prompt(self, user_profile: str, products: List[Dict], top_k: int = 3) -> str:
        """创建推荐提示"""
        products_text = []
        for i, product in enumerate(products[:10]):  # 最多10个候选
            title = str(product.get('title', 'Unknown'))[:60]
            price = product.get('price', 'N/A')
            rating = product.get('average_rating', 'N/A')
            products_text.append(f"{i+1}. {title} (Price: ${price}, Rating: {rating}/5)")
        
        prompt = f"""You are a shopping assistant. Based on the customer profile, recommend the top {top_k} products:

Customer Profile: {user_profile}

Available Products:
{chr(10).join(products_text)}

Please select the {top_k} most suitable products and briefly explain why. Format: "Product X: reason (1-2 sentences)":"""

        return prompt
    
    def run_model_comparison(self, category: str = "All_Beauty", num_users: int = 2) -> Dict:
        """运行模型对比实验"""
        print(f"\n🎯 Model Comparison for {category}")
        print("=" * 50)
        
        # 检查可用模型
        available_models = self.check_model_availability()
        if not available_models:
            print("❌ No models available")
            return {}
        
        # 加载数据
        products_df, reviews_df = self.load_category_data(category)
        if products_df.empty or reviews_df.empty:
            print("❌ No data available")
            return {}
        
        print(f"📊 Loaded {len(products_df)} products, {len(reviews_df)} reviews")
        
        # 选择测试用户
        user_counts = reviews_df['user_id'].value_counts()
        test_users = user_counts.head(num_users).index.tolist()
        
        results = {
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "models_tested": list(available_models.keys()),
            "users": {}
        }
        
        for user_idx, user_id in enumerate(test_users):
            print(f"\n👤 User {user_idx + 1}: {user_id[:20]}...")
            
            user_reviews = reviews_df[reviews_df['user_id'] == user_id]
            print(f"   📝 {len(user_reviews)} reviews, avg rating: {user_reviews['rating'].mean():.1f}")
            
            # 生成用户画像提示
            profile_prompt = self.create_user_profile_prompts(user_reviews)
            
            # 选择推荐候选产品
            candidate_products = products_df.sample(n=min(10, len(products_df))).to_dict('records')
            
            user_results = {
                "user_id": user_id,
                "review_count": len(user_reviews),
                "avg_rating": float(user_reviews['rating'].mean()),
                "model_profiles": {},
                "model_recommendations": {},
                "performance_metrics": {}
            }
            
            # 测试每个可用模型
            for model_key in available_models:
                print(f"   🤖 Testing {model_key}...")
                
                # 生成用户画像
                profile_response, profile_time = self.query_model(model_key, profile_prompt)
                user_results["model_profiles"][model_key] = {
                    "response": profile_response,
                    "response_time": profile_time
                }
                
                # 基于画像生成推荐
                rec_prompt = self.create_recommendation_prompt(profile_response, candidate_products)
                rec_response, rec_time = self.query_model(model_key, rec_prompt)
                
                user_results["model_recommendations"][model_key] = {
                    "response": rec_response,
                    "response_time": rec_time,
                    "candidates_count": len(candidate_products)
                }
                
                user_results["performance_metrics"][model_key] = {
                    "total_time": profile_time + rec_time,
                    "profile_time": profile_time,
                    "recommendation_time": rec_time,
                    "profile_length": len(profile_response),
                    "recommendation_length": len(rec_response)
                }
                
                print(f"      ⏱️ Total time: {profile_time + rec_time:.2f}s")
            
            results["users"][f"user_{user_idx + 1}"] = user_results
        
        return results
    
    def run_comprehensive_comparison(self) -> Dict:
        """运行全面的模型对比"""
        print("🚀 Comprehensive Multi-Model Comparison")
        print("📅", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("=" * 60)
        
        all_results = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "categories_tested": self.test_categories,
                "models_available": self.check_model_availability()
            },
            "category_results": {},
            "summary_statistics": {}
        }
        
        if not all_results["experiment_info"]["models_available"]:
            print("❌ No models available for testing")
            return all_results
        
        # 测试每个类别
        for category in self.test_categories:
            print(f"\n📦 Testing category: {category}")
            category_results = self.run_model_comparison(category, num_users=2)
            if category_results:
                all_results["category_results"][category] = category_results
        
        # 生成汇总统计
        self.generate_summary_statistics(all_results)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"multi_model_comparison_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n✅ Comprehensive comparison completed!")
        print(f"💾 Results saved to: {output_file}")
        
        return all_results
    
    def generate_summary_statistics(self, results: Dict):
        """生成汇总统计"""
        models = results["experiment_info"]["models_available"].keys()
        categories = results["category_results"].keys()
        
        summary = {
            "model_performance": {},
            "category_performance": {},
            "overall_metrics": {}
        }
        
        # 按模型统计
        for model in models:
            model_stats = {
                "total_requests": 0,
                "avg_response_time": 0,
                "avg_profile_length": 0,
                "avg_recommendation_length": 0,
                "success_rate": 0
            }
            
            times = []
            profile_lengths = []
            rec_lengths = []
            successes = 0
            total = 0
            
            for category_data in results["category_results"].values():
                for user_data in category_data.get("users", {}).values():
                    if model in user_data.get("performance_metrics", {}):
                        metrics = user_data["performance_metrics"][model]
                        times.append(metrics["total_time"])
                        profile_lengths.append(metrics["profile_length"])
                        rec_lengths.append(metrics["recommendation_length"])
                        
                        # 检查是否成功（响应不包含Error）
                        profile_resp = user_data["model_profiles"][model]["response"]
                        rec_resp = user_data["model_recommendations"][model]["response"]
                        if not profile_resp.startswith("Error") and not rec_resp.startswith("Error"):
                            successes += 1
                        total += 1
            
            if times:
                model_stats.update({
                    "total_requests": total,
                    "avg_response_time": round(np.mean(times), 2),
                    "avg_profile_length": round(np.mean(profile_lengths), 1),
                    "avg_recommendation_length": round(np.mean(rec_lengths), 1),
                    "success_rate": round(successes / total * 100, 1)
                })
            
            summary["model_performance"][model] = model_stats
        
        results["summary_statistics"] = summary
    
    def print_comparison_summary(self, results: Dict):
        """打印对比摘要"""
        print("\n📊 COMPARISON SUMMARY")
        print("=" * 60)
        
        models_available = results["experiment_info"]["models_available"]
        print(f"🤖 Models tested: {len(models_available)}")
        for model, config in models_available.items():
            print(f"   • {model}: {config['description']}")
        
        print(f"\n📦 Categories tested: {len(results['category_results'])}")
        for category in results["category_results"]:
            print(f"   • {category}")
        
        if "summary_statistics" in results:
            print(f"\n⚡ Performance Summary:")
            summary = results["summary_statistics"]["model_performance"]
            
            print(f"{'Model':<12} {'Success%':<10} {'Avg Time(s)':<12} {'Profile Len':<12} {'Rec Len':<10}")
            print("-" * 65)
            
            for model, stats in summary.items():
                print(f"{model:<12} {stats['success_rate']:<10}% "
                      f"{stats['avg_response_time']:<12} "
                      f"{stats['avg_profile_length']:<12} "
                      f"{stats['avg_recommendation_length']:<10}")

def main():
    """主函数"""
    recommender = MultiModelRecommender()
    results = recommender.run_comprehensive_comparison()
    
    if results and results["category_results"]:
        recommender.print_comparison_summary(results)
        print(f"\n🎉 Multi-model comparison successful!")
    else:
        print(f"\n❌ Comparison failed or incomplete")

if __name__ == "__main__":
    main()
