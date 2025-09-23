# Layerwise-Adapter Project Structure

## 项目概览
本项目实现了基于知识蒸馏的层级适配器系统，包含LLM推荐系统和知识蒸馏两大核心模块。

## 目录结构规划

```
Layerwise-Adapter/
├── README.md                           # 项目主文档
├── requirements.txt                    # 项目依赖
├── setup.py                           # 安装配置
│
├── docs/                              # 文档目录
│   ├── PROJECT_FINAL_SUMMARY.md       # 项目总结
│   ├── EXPERIMENT_REPORT.md           # 实验报告
│   ├── DISTILLATION_GUIDE.md          # 蒸馏指南
│   └── API_REFERENCE.md               # API文档
│
├── src/                               # 源代码目录
│   ├── __init__.py
│   ├── core/                          # 核心模块
│   │   ├── __init__.py
│   │   ├── distillation_trainer.py    # 蒸馏训练器
│   │   ├── fisher_information.py      # Fisher信息矩阵
│   │   └── layerwise_distillation.py  # 层级蒸馏
│   │
│   ├── recommender/                   # 推荐系统模块
│   │   ├── __init__.py
│   │   ├── base_recommender.py        # 基础推荐器
│   │   ├── multi_model_recommender.py # 多模型推荐器
│   │   └── ollama_client.py           # Ollama客户端
│   │
│   └── utils/                         # 工具模块
│       ├── __init__.py
│       ├── data_loader.py             # 数据加载
│       └── experiment_utils.py        # 实验工具
│
├── experiments/                       # 实验脚本
│   ├── distillation_experiment.py     # 蒸馏实验
│   ├── recommendation_benchmark.py    # 推荐基准测试
│   └── model_comparison.py           # 模型对比
│
├── results/                          # 结果文件
│   ├── distillation/                 # 蒸馏结果
│   ├── recommendations/              # 推荐结果
│   └── comparisons/                  # 对比结果
│
├── data/                             # 数据目录
│   ├── amazon/                       # Amazon数据集
│   └── processed/                    # 处理后数据
│
├── models/                           # 模型文件
│   ├── teacher/                      # 教师模型
│   ├── student/                      # 学生模型
│   └── adapters/                     # 适配器模型
│
├── configs/                          # 配置文件
│   ├── distillation_config.yaml     # 蒸馏配置
│   ├── model_config.yaml            # 模型配置
│   └── experiment_config.yaml       # 实验配置
│
└── legacy/                           # 历史版本
    ├── LLM-Inference-Recommender/    # 原推荐系统
    └── old_scripts/                  # 旧脚本
```

## 模块说明

### 1. 核心模块 (src/core/)
- **distillation_trainer.py**: 主要的知识蒸馏训练器
- **fisher_information.py**: Fisher信息矩阵计算
- **layerwise_distillation.py**: 层级蒸馏实现

### 2. 推荐系统模块 (src/recommender/)
- **base_recommender.py**: 推荐系统基类
- **multi_model_recommender.py**: 多模型推荐实现
- **ollama_client.py**: Ollama API客户端

### 3. 实验模块 (experiments/)
- **distillation_experiment.py**: 知识蒸馏实验脚本
- **recommendation_benchmark.py**: 推荐系统基准测试
- **model_comparison.py**: 模型性能对比

### 4. 配置管理 (configs/)
- 使用YAML格式管理各种配置参数
- 便于实验参数调整和复现

## 整理计划

1. **代码重构**: 将分散的代码按功能模块重新组织
2. **文档整理**: 将所有MD文件移到docs目录
3. **结果归档**: 按类型整理实验结果
4. **依赖管理**: 更新requirements.txt和setup.py
5. **配置标准化**: 创建标准化的配置文件

## 版本管理

- **当前版本**: v2.0 (知识蒸馏版本)
- **历史版本**: v1.0 (基础推荐系统) -> legacy/目录
- **开发分支**: phase3-multi-teacher-fusion
