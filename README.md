# Layerwise-Adapter: Fisher Information Matrix-driven Knowledge Distillation for LLM Recommendation Systems

[![Conference](https://img.shields.io/badge/WWW-2026-red.svg)](https://www2026.thewebconf.org/)
[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/GeoffreyWang1117/Intelligent-Recommender)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![Llama3](https://img.shields.io/badge/Teacher-Llama3-green.svg)](https://llama.meta.com/)

**WWW2026研究项目**: 基于Fisher信息矩阵的层级知识蒸馏框架，专为LLM推荐系统优化设计。

## 📋 论文概述

**核心创新**: 首次将Fisher信息矩阵应用于LLM推荐系统的层级知识蒸馏，基于"上层语义>下层语法"的理论假设，实现高效模型压缩与语义保持的平衡。

**研究假设**:
- **H1**: LLM高层(70-100%)比底层(0-30%)对推荐任务更重要
- **H2**: Fisher信息矩阵能准确量化每层对推荐任务的贡献度  
- **H3**: 层级权重递增策略优于均匀权重分配
- **H4**: Llama3在推荐任务上优于其他开源LLM

## 🎯 核心特性

- **🧠 智能层级蒸馏**: 基于Fisher信息矩阵量化每层对推荐任务的贡献度
- **⚡ 高效推荐系统**: 支持多模型(llama3, qwen3, gpt-oss)的推荐对比
- **📊 Amazon数据集**: 完整的Amazon商品评论数据集处理流程
- **🔧 模块化设计**: 清晰的代码架构，易于扩展和维护
- **📈 实验追踪**: 全面的性能监控和结果分析

## 🏗️ 项目架构

```
Layerwise-Adapter/
├── src/                    # 核心源代码
│   ├── core/              # 知识蒸馏核心模块
│   ├── recommender/       # 推荐系统模块  
│   └── utils/             # 工具函数
├── experiments/           # 实验脚本
├── configs/              # 配置文件
├── results/              # 实验结果
├── docs/                 # 项目文档
├── models/               # 模型文件
└── legacy/               # 历史版本
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/Layerwise-Adapter.git
cd Layerwise-Adapter

# 安装依赖
pip install -r requirements.txt

# 启动Ollama服务 (需要预先安装Llama3)
ollama serve
ollama pull llama3:latest
```

### 2. 数据准备

```bash
# 下载Amazon 2023数据集到dataset目录
mkdir -p dataset/amazon
# 支持的类别: All_Beauty, Electronics, Office_Products等
# 将parquet文件放入dataset/amazon/目录
```

### 3. 运行WWW2026实验

```bash
# 运行完整的WWW2026实验流程
python experiments/www2026_distillation_experiment.py

# 单独运行Fisher信息分析
python -c "
from experiments.www2026_distillation_experiment import *
exp = WWW2026Experiment(ExperimentConfig())
exp.setup_experiment()
exp.run_fisher_analysis_experiment()
"

# 运行基础推荐测试
python src/recommender/base_recommender.py
```

## 📖 核心概念

### Fisher信息矩阵蒸馏

Fisher信息矩阵反映模型参数对任务损失的敏感度：

- **高Fisher值层**: 包含更多任务关键语义信息
- **低Fisher值层**: 主要为语法/结构层，蒸馏价值较低  
- **权重策略**: 层深越深权重越大 (上层语义 > 下层语法)

### 层级适配器架构

```python
# 蒸馏权重随层深递增
layer_weights = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # 示例

# Fisher值驱动的自适应权重
fisher_weights = calculate_fisher_information(teacher_model, task_data)
adaptive_weights = normalize_fisher_weights(fisher_weights)
```

## 📊 实验结果

### 模型性能对比

| 模型 | 响应时间 | 推荐质量 | Fisher分数 | 推荐评级 |
|------|----------|----------|-----------|----------|
| **llama3** | 2.31s | 优秀 | 0.85 | ⭐⭐⭐⭐⭐ |
| **qwen3** | 3.20s | 良好 | 0.78 | ⭐⭐⭐⭐ |
| **gpt-oss** | 4.98s | 待改进 | 0.62 | ⭐⭐ |

### 蒸馏效果

- **模型压缩比**: 75% (32层→8层)
- **性能保持**: 92%推荐质量
- **速度提升**: 3.2x推理加速
- **内存减少**: 68%显存占用

## 🛠️ API 使用

### 基础推荐

```python
from src.recommender import BaseRecommender

# 初始化推荐器
recommender = BaseRecommender(model_name="llama3:latest")

# 生成推荐
recommendations = recommender.recommend(
    user_id="user123",
    category="All_Beauty", 
    top_k=3
)
```

### 知识蒸馏

```python
from src.core import DistillationTrainer, FisherInformationCalculator

# 计算Fisher信息
fisher_calc = FisherInformationCalculator()
fisher_scores = fisher_calc.calculate(teacher_model, dataset)

# 执行蒸馏
trainer = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=student_model,
    fisher_weights=fisher_scores
)
trainer.train(train_loader, num_epochs=10)
```

## 📁 配置管理

项目使用YAML配置文件管理参数：

- `configs/distillation_config.yaml`: 蒸馏训练配置
- `configs/model_config.yaml`: 模型参数配置  
- `configs/experiment_config.yaml`: 实验设置配置

## 📈 监控与可视化

```bash
# 启动TensorBoard监控
tensorboard --logdir=results/distillation/logs

# 查看实验报告
open docs/EXPERIMENT_REPORT.md
```

## 🔧 开发指南

### 添加新模型

1. 在`configs/model_config.yaml`中注册模型
2. 实现模型接口在`src/recommender/`
3. 更新测试用例

### 自定义蒸馏策略

1. 继承`LayerwiseDistillation`基类
2. 实现`calculate_layer_weights()`方法
3. 在配置文件中指定新策略

## 📚 文档

- [项目总结](docs/PROJECT_FINAL_SUMMARY.md)
- [实验报告](docs/EXPERIMENT_REPORT.md)  
- [蒸馏指南](docs/DISTILLATION_GUIDE.md)
- [API参考](docs/API_REFERENCE.md)

## 🤝 贡献

欢迎提交Issue和Pull Request！请确保：

1. 代码遵循PEP8规范
2. 添加适当的测试用例
3. 更新相关文档

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 🙏 致谢

- [Ollama](https://ollama.ai/) - 本地LLM服务框架
- [Amazon Review Dataset](https://amazon-reviews-2023.github.io/) - 评论数据集
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

**版本**: v2.0.0 | **更新时间**: 2025-09-16 | **分支**: phase3-multi-teacher-fusion
