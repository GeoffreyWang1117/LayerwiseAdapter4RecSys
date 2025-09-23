"""
测试Amazon数据处理器
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')

# 直接包含必要的类定义
@dataclass
class DataConfig:
    """数据处理配置"""
    min_user_interactions: int = 5
    min_item_interactions: int = 5  
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    random_seed: int = 42
    max_users: Optional[int] = None
    max_items: Optional[int] = None
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_data_processor():
    """测试数据处理器"""
    print("🚀 开始测试Amazon数据处理器...")
    
    # 创建配置
    config = DataConfig(
        min_user_interactions=5,
        min_item_interactions=5,
        test_ratio=0.2,
        val_ratio=0.1,
        max_users=10000,  # 限制用户数用于测试
        max_items=5000    # 限制物品数用于测试
    )
    
    # 初始化处理器
    processor = AmazonDataProcessor(config=config)
    
    # 获取可用类别
    categories = processor.get_available_categories()
    print(f"📋 可用类别: {categories}")
    
    # 获取数据摘要
    print("\n📊 生成数据摘要...")
    summary = processor.get_data_summary()
    print(summary.to_string(index=False))
    
    # 选择一个较小的类别进行详细测试
    test_category = 'All_Beauty'  # 相对较小的数据集
    
    if test_category in categories:
        print(f"\n🔍 详细处理类别: {test_category}")
        result = processor.process_category(test_category)
        
        # 打印处理结果
        print("\n📈 处理结果统计:")
        stats = result['stats']
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
        
        # 分析稀疏性
        print("\n🔍 稀疏性分析:")
        sparsity_metrics = DataUtils.calculate_sparsity_metrics(result['matrix'])
        for key, value in sparsity_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # 分析评分分布
        print("\n📊 评分分布分析:")
        ratings = result['data']['full']['rating'].values
        rating_analysis = DataUtils.analyze_rating_distribution(ratings)
        for key, value in rating_analysis.items():
            if key != 'distribution':
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print(f"  评分分布: {rating_analysis['distribution']}")
        
        # 检查数据划分
        print("\n🗂️ 数据划分检查:")
        for split_name, split_data in result['data'].items():
            print(f"  {split_name}: {len(split_data):,} 条记录")
        
        print(f"\n✅ 类别 {test_category} 处理完成!")
        return result
    else:
        print(f"❌ 测试类别 {test_category} 不可用")
        return None

if __name__ == "__main__":
    result = test_data_processor()
    print("\n🎉 数据处理器测试完成!")
