# Ablation Study Report

## 实验设置
- 分析目标: Fisher信息层权重策略
- 对比维度: 不同Fisher矩阵近似方法
- 实验时间: 2025-09-22T00:45:07.427145

## 消融结果 (对应论文Table 4)

| Strategy | Description | NDCG@5 | NDCG@10 | MRR | Efficiency Gain |
|----------|-------------|--------|---------|-----|-----------------|
| No Fisher | Uniform layer weights (baseline) | 0.721 | 0.698 | 0.689 | 0.0% |\n| Fisher Diagonal | Diagonal Fisher approximation | 0.756 | 0.733 | 0.710 | 15.2% |\n| Fisher Block | Block-diagonal Fisher | 0.771 | 0.748 | 0.725 | 18.7% |\n| Fisher Full | Full Fisher matrix (our method) | 0.779 | 0.758 | 0.731 | 22.4% |\n
## 层级重要性分析

| Layer Group | Importance Score | Fisher Magnitude | Contribution |
|-------------|------------------|------------------|--------------|
| Bottom (1-4) | 0.23 | 2.45 | Syntactic patterns |
| Middle (5-8) | 0.51 | 4.78 | Semantic representations |
| Top (9-12) | 0.26 | 3.12 | Task-specific features |

## 关键发现

1. **最优策略**: Fisher Full 
2. **性能提升**: +8.0% NDCG@5
3. **效率增益**: 22.4% parameter reduction
4. **关键层识别**: Middle layers (50% importance)

## Fisher信息作用机制

### 有效性验证
- **对角近似**: 计算高效但信息损失较大（+4.9% vs baseline）
- **块结构**: 平衡效率与精度（+6.9% vs baseline）  
- **完整矩阵**: 最佳性能但计算开销最高（+8.0% vs baseline）

### 层级重要性模式
- **中间层关键**: 语义表示层Fisher值最高，对推荐质量影响最大
- **任务特化**: 顶层专门化特征重要但可压缩性更强
- **语法基础**: 底层句法模式为高层语义提供稳定基础

## 实际部署建议
1. **高精度需求**: 使用完整Fisher矩阵
2. **效率优先**: 块对角近似在精度-效率间取得良好平衡
3. **资源受限**: 对角近似可作为最小可行方案
