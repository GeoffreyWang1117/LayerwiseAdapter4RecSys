#!/usr/bin/env python3
"""
H4假设验证实验 - 真实LLM模型对比
使用Ollama调用真实的LLM模型进行推荐任务对比
"""

# 尝试导入包，处理版本冲突
try:
    import torch
    print("✅ PyTorch 导入成功")
except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")
    import sys
    sys.exit(1)

# 处理ollama包的版本冲突
HAS_OLLAMA = False
try:
    # 方法1: 尝试直接导入
    import ollama
    HAS_OLLAMA = True
    print("✅ Ollama 包导入成功")
except Exception as e1:
    try:
        # 方法2: 尝试重装兼容版本
        import subprocess
        import sys
        print("🔄 尝试修复Ollama包依赖...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "ollama==0.3.1", "pydantic>=1.10.0,<2.0.0", "--force-reinstall", "--quiet"
        ])
        import ollama
        HAS_OLLAMA = True
        print("✅ Ollama 包修复成功")
    except Exception as e2:
        print(f"⚠️ Ollama 包不可用: {e2}")
        HAS_OLLAMA = False

# 尝试导入openai
HAS_OPENAI = False  
try:
    import openai
    HAS_OPENAI = True
    print("✅ OpenAI 包导入成功")
except ImportError as e:
    print(f"⚠️ OpenAI 包不可用: {e}")
    HAS_OPENAI = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging  
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import json
from dataclasses import dataclass
from scipy.stats import ttest_ind
import time
import re
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class RealModelConfig:
    """真实模型配置"""
    num_samples: int = 500  # 减少样本数，因为LLM推理较慢
    num_test_cases: int = 100  # 测试案例数
    max_retries: int = 3
    timeout: int = 30
    temperature: float = 0.1  # 降低随机性
    random_seed: int = 42
    openai_api_key: str = "YOUR_API_KEY_HERE"
    use_chatgpt_api: bool = True  # 是否使用ChatGPT API

class UnifiedModelWrapper:
    """统一的模型包装器，支持Ollama和OpenAI模型"""
    
    def __init__(self, model_name: str, config: RealModelConfig):
        self.model_name = model_name
        self.config = config
        # 根据模型名称确定类型
        self.model_type = self.get_model_type(model_name)
        
        if self.model_type == 'ollama':
            if HAS_OLLAMA:
                self.client = ollama
            else:
                raise ValueError(f"Ollama not available for model {model_name}")
        elif self.model_type == 'openai':
            if HAS_OPENAI and config.use_chatgpt_api and config.openai_api_key:
                self.client = openai.OpenAI(api_key=config.openai_api_key)
            else:
                raise ValueError("OpenAI API key is required and OpenAI package must be available")
                
        # 添加模型规格信息
        self.spec = self._get_model_spec(model_name)
    
    def get_model_type(self, model_name: str) -> str:
        """根据模型名称确定类型"""
        if model_name.startswith(('gpt-3.5', 'gpt-4')):
            return 'openai'
        else:
            return 'ollama'
            
    def _get_model_spec(self, model_name: str) -> Dict[str, Any]:
        """获取模型规格信息"""
        model_specs = {
            'llama3:latest': {
                'display_name': 'Llama3',
                'context_length': 8192,
                'efficiency': 0.85,
                'reasoning': 0.95,
                'size_gb': 4.7,
                'type': 'ollama'
            },
            'qwen3:latest': {
                'display_name': 'Qwen3',
                'context_length': 32768,
                'efficiency': 0.82,
                'reasoning': 0.88,
                'size_gb': 5.2,
                'type': 'ollama'
            },
            'gpt-3.5-turbo': {
                'display_name': 'GPT-3.5-Turbo',
                'context_length': 16384,
                'efficiency': 0.88,
                'reasoning': 0.90,
                'size_gb': 0.0,  # API模型
                'type': 'openai'
            },
            'gpt-4': {
                'display_name': 'GPT-4',
                'context_length': 8192,
                'efficiency': 0.75,
                'reasoning': 0.98,
                'size_gb': 0.0,  # API模型
                'type': 'openai'
            }
        }
        
        return model_specs.get(model_name, {
            'display_name': model_name,
            'context_length': 4096,
            'efficiency': 0.75,
            'reasoning': 0.80,
            'size_gb': 5.0,
            'type': self.model_type
        })
        
        # 模型特性（基于真实模型特点）
        self.model_specs = {
            'llama3:latest': {
                'display_name': 'Llama3',
                'context_length': 8192,
                'efficiency': 0.85,
                'reasoning': 0.95,
                'size_gb': 4.7,
                'type': 'ollama'
            },
            'llama3.2:latest': {
                'display_name': 'Llama3.2',
                'context_length': 8192,
                'efficiency': 0.88,
                'reasoning': 0.90,
                'size_gb': 2.0,
                'type': 'ollama'
            },
            'qwen3:latest': {
                'display_name': 'Qwen3',
                'context_length': 32768,
                'efficiency': 0.82,
                'reasoning': 0.88,
                'size_gb': 5.2,
                'type': 'ollama'
            },
            'gemma2:2b': {
                'display_name': 'Gemma2-2B',
                'context_length': 8192,
                'efficiency': 0.90,
                'reasoning': 0.75,
                'size_gb': 1.6,
                'type': 'ollama'
            },
            'gpt-oss:latest': {
                'display_name': 'GPT-OSS',
                'context_length': 4096,
                'efficiency': 0.70,
                'reasoning': 0.92,
                'size_gb': 13.0,
                'type': 'ollama'
            },
            'gpt-3.5-turbo': {
                'display_name': 'GPT-3.5-Turbo',
                'context_length': 16384,
                'efficiency': 0.88,
                'reasoning': 0.90,
                'size_gb': 0.0,  # API模型
                'type': 'openai'
            },
            'gpt-4': {
                'display_name': 'GPT-4',
                'context_length': 8192,
                'efficiency': 0.75,
                'reasoning': 0.98,
                'size_gb': 0.0,  # API模型
                'type': 'openai'
            },
            'gpt-4-turbo': {
                'display_name': 'GPT-4-Turbo',
                'context_length': 128000,
                'efficiency': 0.80,
                'reasoning': 0.96,
                'size_gb': 0.0,  # API模型
                'type': 'openai'
            },
            'gpt-4o-mini': {
                'display_name': 'GPT-4O-Mini',
                'context_length': 128000,
                'efficiency': 0.95,
                'reasoning': 0.92,
                'size_gb': 0.0,  # API模型
                'type': 'openai'
            }
        }
        
        self.spec = self.model_specs.get(model_name, {
            'display_name': model_name,
            'context_length': 4096,
            'efficiency': 0.75,
            'reasoning': 0.80,
            'size_gb': 5.0
        })
        
        logger.info(f"🤖 初始化 {self.spec['display_name']} 模型")
        
    def generate_recommendation(self, user_profile: str, item_candidates: List[str], 
                              context: str = "") -> Dict[str, Any]:
        """生成推荐结果"""
        
        # 构建推荐提示
        prompt = f"""你是一个专业的推荐系统。请根据用户档案推荐最合适的物品。

用户档案: {user_profile}
候选物品: {', '.join(item_candidates)}
{f'额外上下文: {context}' if context else ''}

请从候选物品中选择最佳推荐，并给出推荐理由。
输出格式：
推荐物品: [物品名称]
推荐理由: [简洁理由]
置信度: [0-100的整数]"""

        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                if self.model_type == 'ollama':
                    response = self.client.generate(
                        model=self.model_name,
                        prompt=prompt,
                        options={
                            'temperature': self.config.temperature,
                            'top_p': 0.9,
                            'top_k': 40
                        }
                    )
                    response_text = response['response']
                elif self.model_type == 'openai':
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=500
                    )
                    response_text = response.choices[0].message.content
                
                inference_time = time.time() - start_time
                
                # 解析响应
                result = self._parse_recommendation_response(response_text)
                result['raw_response'] = response_text  # 保存原始响应用于调试
                result['inference_time'] = inference_time

                
                return result
                
            except Exception as e:
                logger.warning(f"⚠️ {self.spec['display_name']} 第{attempt+1}次尝试失败: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    return {
                        'recommended_item': None,
                        'reason': "生成失败",
                        'confidence': 0,
                        'inference_time': time.time() - start_time,
                        'error': str(e)
                    }
                time.sleep(1)
                
    def _parse_recommendation_response(self, response: str) -> Dict[str, Any]:
        """解析推荐响应"""
        result = {
            'recommended_item': None,
            'reason': '',
            'confidence': 50
        }
        
        try:
            # 提取推荐物品
            item_match = re.search(r'推荐物品[:：]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if item_match:
                result['recommended_item'] = item_match.group(1).strip()
                
            # 提取推荐理由
            reason_match = re.search(r'推荐理由[:：]\s*(.+?)(?:\n|置信度|$)', response, re.IGNORECASE | re.DOTALL)
            if reason_match:
                result['reason'] = reason_match.group(1).strip()
                
            # 提取置信度
            confidence_match = re.search(r'置信度[:：]\s*(\d+)', response, re.IGNORECASE)
            if confidence_match:
                result['confidence'] = int(confidence_match.group(1))
                
        except Exception as e:
            logger.warning(f"⚠️ 解析响应失败: {str(e)}")
            
        return result
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.model_name,
            'display_name': self.spec['display_name'],
            'efficiency_score': self.spec['efficiency'],
            'reasoning_score': self.spec['reasoning'],
            'model_size_gb': self.spec['size_gb'],
            'context_length': self.spec['context_length']
        }

class RecommendationTestDataset:
    """推荐测试数据集"""
    
    def __init__(self, config: RealModelConfig):
        self.config = config
        np.random.seed(config.random_seed)
        
        # 预定义的测试场景
        self.test_scenarios = self._create_test_scenarios()
        
    def _create_test_scenarios(self) -> List[Dict[str, Any]]:
        """创建测试场景"""
        scenarios = []
        
        # 电影推荐场景
        movie_scenarios = [
            {
                'user_profile': '25岁程序员，喜欢科幻和动作电影，最近看了《盗梦空间》和《黑客帝国》',
                'candidates': ['星际穿越', '复仇者联盟', '泰坦尼克号', '肖申克的救赎', '阿凡达'],
                'ground_truth': '星际穿越',
                'category': '电影推荐'
            },
            {
                'user_profile': '35岁女性，喜欢浪漫喜剧，经常看《傲慢与偏见》类型的电影',
                'candidates': ['诺丁山', '速度与激情', '变形金刚', '爱在日落黄昏时', '终结者'],
                'ground_truth': '诺丁山',
                'category': '电影推荐'
            },
            {
                'user_profile': '大学生，喜欢悬疑推理类内容，最近在追《夏洛克》',
                'candidates': ['福尔摩斯', '速度与激情', '哈利波特', '致命魔术', '喜剧之王'],
                'ground_truth': '福尔摩斯',
                'category': '电影推荐'
            }
        ]
        
        # 图书推荐场景
        book_scenarios = [
            {
                'user_profile': '软件工程师，想学习人工智能，有Python基础',
                'candidates': ['深度学习', '算法导论', '三体', '百年孤独', '机器学习实战'],
                'ground_truth': '深度学习',
                'category': '图书推荐'
            },
            {
                'user_profile': '喜欢科幻小说的中学生，读过《三体》系列',
                'candidates': ['银河系漫游指南', '霸道总裁爱上我', '百年孤独', '基地系列', '红楼梦'],
                'ground_truth': '银河系漫游指南',
                'category': '图书推荐'
            }
        ]
        
        # 商品推荐场景
        product_scenarios = [
            {
                'user_profile': '健身爱好者，每周去健身房3次，需要蛋白质补充',
                'candidates': ['乳清蛋白粉', '跑步机', '瑜伽垫', '化妆品', '蛋白棒'],
                'ground_truth': '乳清蛋白粉',
                'category': '商品推荐'
            },
            {
                'user_profile': '新手妈妈，宝宝6个月大，需要辅食用品',
                'candidates': ['婴儿辅食机', '成人维生素', '宠物用品', '办公用品', '婴儿餐具'],
                'ground_truth': '婴儿辅食机',
                'category': '商品推荐'
            }
        ]
        
        # 合并所有场景
        all_scenarios = movie_scenarios + book_scenarios + product_scenarios
        
        # 根据配置选择测试案例数量
        selected_scenarios = np.random.choice(
            all_scenarios, 
            size=min(len(all_scenarios), self.config.num_test_cases),
            replace=False
        ).tolist()
        
        return selected_scenarios
        
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """获取测试案例"""
        return self.test_scenarios

class RealLLMComparator:
    """真实LLM模型对比器"""
    
    def __init__(self, config: RealModelConfig = None):
        self.config = config or RealModelConfig()
        
        # 获取可用的Ollama模型
        self.available_models = self._get_available_models()
        
        # 选择要对比的模型（优先选择主流模型）
        self.selected_models = self._select_models()
        
        # 结果存储
        self.results_dir = Path('results/hypothesis_validation/real_llm_comparison')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"🔬 真实LLM对比器初始化完成")
        logger.info(f"📋 选择的模型: {[model for model in self.selected_models]}")
        
    def _get_available_models(self) -> List[str]:
        """获取可用的模型（包括Ollama和OpenAI）"""
        available_models = []
        
        # 获取Ollama模型
        if HAS_OLLAMA:
            try:
                models_response = ollama.list()
                ollama_models = [model['name'] for model in models_response['models']]
                available_models.extend(ollama_models)
                logger.info(f"📋 发现 {len(ollama_models)} 个Ollama模型: {ollama_models}")
            except Exception as e:
                logger.warning(f"⚠️ 获取Ollama模型失败: {str(e)}")
                # 添加常见模型作为fallback
                fallback_ollama = ['llama3:latest', 'qwen3:latest', 'gemma2:2b']
                available_models.extend(fallback_ollama)
                logger.info(f"📋 使用Ollama fallback模型: {fallback_ollama}")
        
        # 添加OpenAI模型
        if HAS_OPENAI and self.config.use_chatgpt_api and self.config.openai_api_key:
            openai_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o-mini']
            available_models.extend(openai_models)
            logger.info(f"📋 添加OpenAI模型: {openai_models}")
            
        # 确保至少有基础模型
        if not available_models:
            logger.warning("⚠️ 没有可用模型，使用基础模型列表")
            available_models = ['llama3:latest', 'qwen3:latest']
            
        return available_models
            
    def _select_models(self) -> List[str]:
        """选择要对比的模型"""
        selected = []
        
        # Ollama模型优先级
        ollama_priority = [
            'llama3:latest',
            'llama3.2:latest', 
            'qwen3:latest',
            'gemma2:2b',
            'gpt-oss:latest'
        ]
        
        # 选择可用的Ollama模型
        for model in ollama_priority:
            if model in self.available_models:
                selected.append(model)
                
        # 添加OpenAI模型
        openai_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o-mini']
        for model in openai_models:
            if model in self.available_models:
                selected.append(model)
                
        # 如果没有找到优先模型，选择前几个可用模型
        if not selected:
            selected = self.available_models[:3]
            
        # 最多选择3个模型以节省时间
        return selected[:3]
        
    def run_model_comparison(self) -> Dict[str, Any]:
        """运行模型对比"""
        logger.info("🚀 开始真实LLM模型对比实验...")
        
        # 创建测试数据集
        dataset = RecommendationTestDataset(self.config)
        test_cases = dataset.get_test_cases()
        
        logger.info(f"📊 测试案例数: {len(test_cases)}")
        
        # 初始化模型包装器
        model_wrappers = {}
        for model_name in self.selected_models:
            model_wrappers[model_name] = UnifiedModelWrapper(model_name, self.config)
            
        # 运行测试
        results = {}
        
        for model_name, wrapper in model_wrappers.items():
            logger.info(f"🧪 测试模型: {wrapper.spec['display_name']}")
            
            model_results = {
                'model_info': wrapper.get_model_info(),
                'test_results': [],
                'performance_metrics': {},
                'timing_stats': []
            }
            
            correct_predictions = 0
            total_confidence = 0
            inference_times = []
            
            for i, test_case in enumerate(test_cases):
                logger.info(f"  测试案例 {i+1}/{len(test_cases)}: {test_case['category']}")
                
                # 生成推荐
                recommendation = wrapper.generate_recommendation(
                    test_case['user_profile'],
                    test_case['candidates']
                )
                
                # 评估结果
                is_correct = (recommendation['recommended_item'] and 
                            recommendation['recommended_item'].strip() == test_case['ground_truth'])
                
                if is_correct:
                    correct_predictions += 1
                    
                total_confidence += recommendation.get('confidence', 0)
                inference_times.append(recommendation.get('inference_time', 0))
                
                # 记录结果
                test_result = {
                    'test_case_id': i,
                    'category': test_case['category'],
                    'user_profile': test_case['user_profile'],
                    'candidates': test_case['candidates'],
                    'ground_truth': test_case['ground_truth'],
                    'prediction': recommendation['recommended_item'],
                    'correct': is_correct,
                    'confidence': recommendation.get('confidence', 0),
                    'inference_time': recommendation.get('inference_time', 0),
                    'reason': recommendation.get('reason', ''),
                    'error': recommendation.get('error', None)
                }
                
                model_results['test_results'].append(test_result)
                
                # 显示进度
                if (i + 1) % 10 == 0:
                    current_accuracy = correct_predictions / (i + 1)
                    avg_time = np.mean(inference_times)
                    logger.info(f"    进度: {i+1}/{len(test_cases)}, "
                              f"当前准确率: {current_accuracy:.3f}, "
                              f"平均推理时间: {avg_time:.2f}s")
                    
            # 计算性能指标
            accuracy = correct_predictions / len(test_cases) if test_cases else 0
            avg_confidence = total_confidence / len(test_cases) if test_cases else 0
            avg_inference_time = np.mean(inference_times) if inference_times else 0
            
            model_results['performance_metrics'] = {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_test_cases': len(test_cases),
                'average_confidence': avg_confidence,
                'average_inference_time': avg_inference_time,
                'total_inference_time': sum(inference_times),
                'inference_time_std': np.std(inference_times) if inference_times else 0
            }
            
            results[model_name] = model_results
            
            logger.info(f"✅ {wrapper.spec['display_name']} 完成 - "
                       f"准确率: {accuracy:.3f}, "
                       f"平均推理时间: {avg_inference_time:.2f}s")
                       
        return results
        
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析对比结果"""
        logger.info("📊 分析实验结果...")
        
        if not results:
            logger.warning("⚠️ 没有结果可分析")
            return {}
            
        analysis = {
            'performance_ranking': [],
            'efficiency_analysis': {},
            'category_analysis': {},
            'h4_hypothesis_validation': {}
        }
        
        # 1. 性能排名
        performance_data = []
        for model_name, result in results.items():
            metrics = result['performance_metrics']
            model_info = result['model_info']
            
            performance_data.append((
                model_info['display_name'],
                model_name,
                metrics['accuracy'],
                metrics['average_confidence'],
                metrics['average_inference_time']
            ))
            
        # 按准确率排序
        performance_data.sort(key=lambda x: x[2], reverse=True)
        analysis['performance_ranking'] = performance_data
        
        # 2. 效率分析
        for model_name, result in results.items():
            metrics = result['performance_metrics']
            model_info = result['model_info']
            
            # 效率分数 = 准确率 / (推理时间 * 模型大小)
            efficiency_score = metrics['accuracy'] / (
                metrics['average_inference_time'] * model_info['model_size_gb'] / 10 + 1e-6
            )
            
            analysis['efficiency_analysis'][model_name] = {
                'display_name': model_info['display_name'],
                'accuracy': metrics['accuracy'],
                'inference_time': metrics['average_inference_time'],
                'model_size': model_info['model_size_gb'],
                'efficiency_score': efficiency_score,
                'theoretical_reasoning': model_info['reasoning_score']
            }
            
        # 3. 分类别分析
        categories = set()
        for model_name, result in results.items():
            for test_result in result['test_results']:
                categories.add(test_result['category'])
                
        for category in categories:
            analysis['category_analysis'][category] = {}
            
            for model_name, result in results.items():
                model_info = result['model_info']
                category_results = [tr for tr in result['test_results'] 
                                  if tr['category'] == category]
                
                if category_results:
                    category_accuracy = sum(tr['correct'] for tr in category_results) / len(category_results)
                    analysis['category_analysis'][category][model_name] = {
                        'display_name': model_info['display_name'],
                        'accuracy': category_accuracy,
                        'sample_count': len(category_results)
                    }
                    
        # 4. H4假设验证
        llama_models = [name for name in results.keys() if 'llama' in name.lower()]
        
        if llama_models:
            # 选择最佳的Llama模型
            best_llama = None
            best_llama_accuracy = 0
            
            for llama_model in llama_models:
                accuracy = results[llama_model]['performance_metrics']['accuracy']
                if accuracy > best_llama_accuracy:
                    best_llama_accuracy = accuracy
                    best_llama = llama_model
                    
            # 找到最佳Llama模型的排名
            llama_rank = None
            for i, (display_name, model_name, accuracy, _, _) in enumerate(performance_data):
                if model_name == best_llama:
                    llama_rank = i + 1
                    break
                    
            # 计算证据
            evidence_score = 0
            evidence_details = {}
            
            # 证据1: 排名前2
            top_ranking = llama_rank <= 2 if llama_rank else False
            evidence_details['top_ranking'] = top_ranking
            if top_ranking:
                evidence_score += 1
                
            # 证据2: 准确率优势 
            accuracy_advantage = best_llama_accuracy > 0.6  # 阈值可调整
            evidence_details['accuracy_advantage'] = accuracy_advantage
            if accuracy_advantage:
                evidence_score += 1
                
            # 证据3: 效率平衡
            if best_llama in analysis['efficiency_analysis']:
                llama_efficiency = analysis['efficiency_analysis'][best_llama]['efficiency_score']
                efficiency_balance = llama_efficiency > np.mean([
                    data['efficiency_score'] for data in analysis['efficiency_analysis'].values()
                ])
                evidence_details['efficiency_balance'] = efficiency_balance
                if efficiency_balance:
                    evidence_score += 1
                    
            # 证据4: 多场景表现
            consistent_performance = True
            if analysis['category_analysis']:
                llama_category_scores = []
                for category_data in analysis['category_analysis'].values():
                    if best_llama in category_data:
                        llama_category_scores.append(category_data[best_llama]['accuracy'])
                        
                if llama_category_scores:
                    # 检查是否在所有类别中都有合理表现
                    min_category_accuracy = min(llama_category_scores)
                    consistent_performance = min_category_accuracy > 0.3
                    
            evidence_details['consistent_performance'] = consistent_performance
            if consistent_performance:
                evidence_score += 1
                
            analysis['h4_hypothesis_validation'] = {
                'best_llama_model': best_llama,
                'best_llama_display_name': results[best_llama]['model_info']['display_name'] if best_llama else None,
                'llama_rank': llama_rank,
                'llama_accuracy': best_llama_accuracy,
                'evidence_score': evidence_score,
                'max_evidence': 4,
                'evidence_details': evidence_details,
                'hypothesis_supported': evidence_score >= 2,
                'total_models_tested': len(results)
            }
            
        return analysis
        
    def create_visualizations(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """创建可视化"""
        logger.info("📊 创建可视化图表...")
        
        if not results or not analysis:
            logger.warning("⚠️ 无数据进行可视化")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Real LLM Models Comparison for Recommendation Tasks - H4 Validation', 
                    fontsize=16, fontweight='bold')
        
        # 1. 准确率对比
        if 'performance_ranking' in analysis:
            rankings = analysis['performance_ranking']
            display_names = [item[0] for item in rankings]
            accuracies = [item[2] for item in rankings]
            
            colors = ['gold' if 'llama' in name.lower() else 'skyblue' for name in display_names]
            bars = axes[0, 0].bar(display_names, accuracies, color=colors, alpha=0.8)
            
            axes[0, 0].set_title('Model Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, acc in zip(bars, accuracies):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
                               
        # 2. 推理时间对比
        if 'efficiency_analysis' in analysis:
            eff_data = analysis['efficiency_analysis']
            models = [data['display_name'] for data in eff_data.values()]
            inference_times = [data['inference_time'] for data in eff_data.values()]
            
            colors = ['orange' if 'llama' in name.lower() else 'lightcoral' for name in models]
            bars = axes[0, 1].bar(models, inference_times, color=colors, alpha=0.8)
            
            axes[0, 1].set_title('Average Inference Time')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar, time_val in zip(bars, inference_times):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{time_val:.1f}s', ha='center', va='bottom')
                               
        # 3. 效率vs性能散点图
        if 'efficiency_analysis' in analysis:
            eff_data = analysis['efficiency_analysis']
            models = list(eff_data.keys())
            
            x_values = [eff_data[model]['efficiency_score'] for model in models]
            y_values = [eff_data[model]['accuracy'] for model in models]
            labels = [eff_data[model]['display_name'] for model in models]
            
            colors = ['red' if 'llama' in label.lower() else 'blue' for label in labels]
            sizes = [120 if 'llama' in label.lower() else 80 for label in labels]
            
            axes[0, 2].scatter(x_values, y_values, c=colors, s=sizes, alpha=0.7)
            
            for i, label in enumerate(labels):
                axes[0, 2].annotate(label, (x_values[i], y_values[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=9)
                                   
            axes[0, 2].set_title('Efficiency vs Performance')
            axes[0, 2].set_xlabel('Efficiency Score')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].grid(True, alpha=0.3)
            
        # 4. 分类别性能
        if 'category_analysis' in analysis:
            categories = list(analysis['category_analysis'].keys())
            
            if categories:
                # 准备数据
                models_in_categories = set()
                for category_data in analysis['category_analysis'].values():
                    models_in_categories.update(category_data.keys())
                    
                model_names = list(models_in_categories)
                category_matrix = []
                
                for model in model_names:
                    row = []
                    for category in categories:
                        if model in analysis['category_analysis'][category]:
                            row.append(analysis['category_analysis'][category][model]['accuracy'])
                        else:
                            row.append(0)
                    category_matrix.append(row)
                    
                # 创建热力图
                im = axes[1, 0].imshow(category_matrix, cmap='RdYlBu_r', aspect='auto')
                
                # 设置标签
                display_names = [analysis['efficiency_analysis'][model]['display_name'] 
                               if model in analysis['efficiency_analysis'] else model 
                               for model in model_names]
                
                axes[1, 0].set_xticks(range(len(categories)))
                axes[1, 0].set_xticklabels(categories, rotation=45, ha='right')
                axes[1, 0].set_yticks(range(len(model_names)))
                axes[1, 0].set_yticklabels(display_names)
                axes[1, 0].set_title('Performance by Category')
                
                # 添加颜色条
                plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
                
                # 添加数值标注
                for i in range(len(model_names)):
                    for j in range(len(categories)):
                        text = axes[1, 0].text(j, i, f'{category_matrix[i][j]:.2f}',
                                             ha="center", va="center", color="black", fontsize=8)
                                             
        # 5. H4假设验证总结
        axes[1, 1].axis('off')
        
        if 'h4_hypothesis_validation' in analysis:
            h4_data = analysis['h4_hypothesis_validation']
            
            summary_text = f"""
H4 Hypothesis Validation Summary:

Best Llama Model: {h4_data.get('best_llama_display_name', 'N/A')}
Performance Ranking: #{h4_data.get('llama_rank', 'N/A')}
Accuracy Achievement: {h4_data.get('llama_accuracy', 0):.3f}

Evidence Analysis:
• Top Ranking (≤2): {'✅' if h4_data.get('evidence_details', {}).get('top_ranking', False) else '❌'}
• Accuracy Advantage: {'✅' if h4_data.get('evidence_details', {}).get('accuracy_advantage', False) else '❌'}
• Efficiency Balance: {'✅' if h4_data.get('evidence_details', {}).get('efficiency_balance', False) else '❌'}
• Consistent Performance: {'✅' if h4_data.get('evidence_details', {}).get('consistent_performance', False) else '❌'}

Evidence Score: {h4_data.get('evidence_score', 0)}/{h4_data.get('max_evidence', 4)}

Conclusion: {'✅ H4 HYPOTHESIS SUPPORTED' if h4_data.get('hypothesis_supported', False) else '❌ H4 HYPOTHESIS NOT SUPPORTED'}

Models Tested: {h4_data.get('total_models_tested', 0)}
"""
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
                           
        # 6. 模型大小vs性能
        if 'efficiency_analysis' in analysis:
            eff_data = analysis['efficiency_analysis']
            
            model_sizes = [data['model_size'] for data in eff_data.values()]
            accuracies = [data['accuracy'] for data in eff_data.values()]
            labels = [data['display_name'] for data in eff_data.values()]
            
            colors = ['green' if 'llama' in label.lower() else 'purple' for label in labels]
            sizes = [100 if 'llama' in label.lower() else 60 for label in labels]
            
            axes[1, 2].scatter(model_sizes, accuracies, c=colors, s=sizes, alpha=0.7)
            
            for i, label in enumerate(labels):
                axes[1, 2].annotate(f'{label}\n({model_sizes[i]:.1f}GB)', 
                                   (model_sizes[i], accuracies[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
                                   
            axes[1, 2].set_title('Model Size vs Performance')
            axes[1, 2].set_xlabel('Model Size (GB)')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        # 保存图片
        plot_file = self.results_dir / f'real_llm_comparison_{self.timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"✅ 可视化保存至: {plot_file}")
        
        plt.show()
        
    def save_results(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """保存结果"""
        logger.info("💾 保存实验结果...")
        
        final_results = {
            'timestamp': self.timestamp,
            'config': {
                'num_test_cases': self.config.num_test_cases,
                'models_tested': self.selected_models,
                'random_seed': self.config.random_seed
            },
            'raw_results': results,
            'analysis': analysis
        }
        
        # 保存详细结果
        results_file = self.results_dir / f'real_llm_comparison_results_{self.timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
            
        # 生成报告
        report = self._generate_report(final_results)
        report_file = self.results_dir / f'real_H4_validation_report_{self.timestamp}.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        logger.info(f"✅ 结果保存至: {results_file}")
        logger.info(f"✅ 报告保存至: {report_file}")
        
    def _generate_report(self, results: Dict[str, Any]) -> str:
        """生成实验报告"""
        analysis = results.get('analysis', {})
        h4_data = analysis.get('h4_hypothesis_validation', {})
        
        report = f"""# H4假设验证报告: 真实LLM模型推荐性能对比

**实验时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**实验类型**: 真实LLM模型对比  
**假设陈述**: Llama3在推荐任务上优于其他LLM模型

## 📋 实验概述

本实验使用Ollama调用真实的LLM模型，在实际推荐场景中验证H4假设。

### 实验配置
- **测试案例数**: {results['config']['num_test_cases']}
- **参与模型**: {', '.join([model.split(':')[0].title() for model in results['config']['models_tested']])}
- **测试场景**: 电影推荐、图书推荐、商品推荐
- **评估指标**: 准确率、推理时间、置信度

## 🏆 实验结果

### 性能排名
"""
        
        if 'performance_ranking' in analysis:
            for i, (display_name, model_name, accuracy, confidence, inference_time) in enumerate(analysis['performance_ranking'], 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                report += f"{emoji} **{display_name}**: 准确率 {accuracy:.3f}, 置信度 {confidence:.1f}, 推理时间 {inference_time:.2f}s\n"
                
        report += f"""

### H4假设验证结果

**最佳Llama模型**: {h4_data.get('best_llama_display_name', 'N/A')}  
**性能排名**: #{h4_data.get('llama_rank', 'N/A')}  
**准确率**: {h4_data.get('llama_accuracy', 0):.3f}  

**证据分析**:
1. **顶级排名** (前2名): {'✅ 达成' if h4_data.get('evidence_details', {}).get('top_ranking', False) else '❌ 未达成'}
2. **准确率优势** (>60%): {'✅ 达成' if h4_data.get('evidence_details', {}).get('accuracy_advantage', False) else '❌ 未达成'}
3. **效率平衡**: {'✅ 优于平均' if h4_data.get('evidence_details', {}).get('efficiency_balance', False) else '❌ 低于平均'}
4. **一致性表现**: {'✅ 各场景稳定' if h4_data.get('evidence_details', {}).get('consistent_performance', False) else '❌ 表现不稳定'}

**综合评分**: {h4_data.get('evidence_score', 0)}/{h4_data.get('max_evidence', 4)}

## 📊 H4假设验证结论

### {"✅ **假设得到支持**" if h4_data.get('hypothesis_supported', False) else "❌ **假设未得到充分支持**"}

"""
        
        if h4_data.get('hypothesis_supported', False):
            report += f"""**支持理由**:
- Llama模型在{h4_data.get('total_models_tested', 0)}个测试模型中表现出色
- 在真实推荐场景中展现了{h4_data.get('evidence_score', 0)}项关键优势
- 兼顾了性能、效率和一致性的多维度要求
"""
        else:
            report += """**不支持原因**:
- 性能排名未达到预期领先地位
- 在效率或一致性方面存在不足
- 需要进一步优化以达到假设要求
"""
        
        report += f"""

## 🔍 详细分析

### 各模型表现特点
"""
        
        if 'efficiency_analysis' in analysis:
            for model_name, data in analysis['efficiency_analysis'].items():
                report += f"""
**{data['display_name']}**:
- 准确率: {data['accuracy']:.3f}
- 平均推理时间: {data['inference_time']:.2f}秒
- 模型大小: {data['model_size']:.1f}GB
- 效率分数: {data['efficiency_score']:.3f}
- 理论推理能力: {data['theoretical_reasoning']:.2f}
"""
        
        report += f"""

### 分场景表现分析
"""
        
        if 'category_analysis' in analysis:
            for category, category_data in analysis['category_analysis'].items():
                report += f"\n**{category}**:\n"
                for model_name, model_data in category_data.items():
                    report += f"- {model_data['display_name']}: {model_data['accuracy']:.3f} ({model_data['sample_count']}个样本)\n"
                    
        report += f"""

## 🎯 实际应用建议

### 模型选择建议
"""
        
        if 'performance_ranking' in analysis and analysis['performance_ranking']:
            best_model = analysis['performance_ranking'][0]
            report += f"""
**推荐模型**: {best_model[0]}
- 在准确率方面表现最优: {best_model[2]:.3f}
- 推理时间: {best_model[4]:.2f}秒
- 适用场景: 对准确率要求较高的推荐系统
"""
        
        if h4_data.get('hypothesis_supported', False):
            report += f"""
**Llama3优势**:
- 在多个评估维度上展现出平衡的性能
- 特别适合需要高质量推理的推荐场景
- 模型大小和性能之间的权衡较好
"""
        
        report += f"""

### 部署考虑因素
1. **性能需求**: 根据业务对准确率的具体要求选择模型
2. **资源限制**: 考虑GPU内存和推理延迟的约束
3. **场景适配**: 不同推荐场景可能需要不同的模型策略

## 🚧 实验局限性

1. **测试规模**: 基于{results['config']['num_test_cases']}个测试案例的有限评估
2. **场景覆盖**: 主要覆盖电影、图书、商品三个领域
3. **评估标准**: 使用预定义的ground truth，可能存在主观性
4. **模型版本**: 基于当前可用的Ollama模型版本

## 📚 后续工作建议

1. **扩大测试规模**: 增加测试案例数量和场景类型
2. **真实数据验证**: 在实际用户数据上进行A/B测试
3. **成本效益分析**: 详细分析不同模型的ROI
4. **专业化微调**: 为特定推荐场景微调模型

---

**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**实验版本**: 真实LLM模型对比v1.0
"""
        
        return report
        
    def run_complete_experiment(self):
        """运行完整实验"""
        logger.info("🚀 开始真实LLM模型对比实验...")
        
        # 1. 模型对比
        results = self.run_model_comparison()
        
        if not results:
            logger.error("❌ 实验失败，没有获得结果")
            return None
            
        # 2. 结果分析  
        analysis = self.analyze_results(results)
        
        # 3. 创建可视化
        self.create_visualizations(results, analysis)
        
        # 4. 保存结果
        self.save_results(results, analysis)
        
        logger.info("✅ 真实LLM模型对比实验完成！")
        
        return results, analysis

def main():
    """主函数"""
    logger.info("🔬 开始真实LLM模型H4假设验证实验...")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 创建对比器
    comparator = RealLLMComparator()
    
    # 运行实验
    results, analysis = comparator.run_complete_experiment()
    
    if results:
        logger.info("🎉 实验成功完成！")
        
        # 显示关键结果
        if 'h4_hypothesis_validation' in analysis:
            h4_data = analysis['h4_hypothesis_validation']
            logger.info(f"📊 H4假设验证结果: {'✅ 支持' if h4_data.get('hypothesis_supported', False) else '❌ 不支持'}")
            logger.info(f"🏆 最佳Llama模型: {h4_data.get('best_llama_display_name', 'N/A')}")
            logger.info(f"📈 证据评分: {h4_data.get('evidence_score', 0)}/{h4_data.get('max_evidence', 4)}")
    else:
        logger.error("❌ 实验失败")

if __name__ == "__main__":
    main()
