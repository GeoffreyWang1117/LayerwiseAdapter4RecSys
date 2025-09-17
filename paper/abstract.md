# Layerwise Knowledge Distillation for LLM-based Recommender Systems: A Fisher Information Matrix Approach

## Abstract

**Authors**: Zhaohui Wang  
**Conference**: WWW 2026  
**Keywords**: Knowledge Distillation, Large Language Models, Recommender Systems, Fisher Information Matrix, Layerwise Adaptation

---

### Background and Motivation

Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding user preferences and generating personalized recommendations. However, their deployment in real-world recommender systems faces significant challenges due to computational overhead and inference latency. While traditional knowledge distillation methods treat all layers equally, we argue that different layers in LLMs contribute differently to recommendation tasks: **upper layers (semantic layers) contain more task-relevant knowledge than lower layers (syntactic/structural layers)**.

### Key Contribution

This paper introduces a novel **Fisher Information Matrix-driven Layerwise Knowledge Distillation** framework for LLM-based recommender systems. Our core insight is that the Fisher Information Matrix, which captures the second-order derivatives of model parameters with respect to task loss, can effectively quantify each layer's contribution to recommendation performance. Layers with higher Fisher values contain more task-critical knowledge and should receive higher distillation weights.

### Methodology

**1. Fisher Information Quantification**
We define the Fisher Information Matrix for recommendation tasks as:
```
F_ij = E[∂²L(θ)/∂θ_i∂θ_j]
```
where L(θ) represents the recommendation loss. The diagonal elements F_ii reflect individual parameter sensitivity to task performance.

**2. Layerwise Weight Assignment**
Based on our theoretical analysis, we propose an adaptive weighting strategy:
- **Lower layers (0-30%)**: Focus on tokenization and basic linguistic features → Low Fisher weights (0.1-0.3)
- **Middle layers (30-70%)**: Handle semantic composition and feature interaction → Medium Fisher weights (0.3-0.7)  
- **Upper layers (70-100%)**: Capture user preference reasoning and decision-making → High Fisher weights (0.7-2.0)

**3. Teacher-Student Architecture**
We use **Llama3** as the teacher model (validated as optimal for recommendation tasks) and design a lightweight student model with adaptive layer attention guided by Fisher weights.

### Experimental Results

**Dataset**: Amazon Product Reviews (10 categories, 2.3M samples)
**Teacher Model**: Llama3-8B 
**Student Model**: 768-dim, 12-layer Transformer

**Key Findings**:
- **Model Compression**: 75% parameter reduction (32→8 effective layers)
- **Performance Preservation**: 92% recommendation quality maintained
- **Inference Speedup**: 3.2× faster response time
- **Fisher Validation**: Upper layers show 2.4× higher Fisher values than lower layers

**Comparison with Baselines**:
| Method | Model Size | NDCG@5 | MRR | Latency (ms) |
|--------|------------|--------|-----|--------------|
| Llama3 (Full) | 8B | 0.847 | 0.792 | 1,230 |
| Uniform Distillation | 768M | 0.721 | 0.689 | 385 |
| **Fisher Layerwise** | 768M | **0.779** | **0.731** | **387** |

### Technical Innovation

**1. Theoretical Foundation**: First work to establish the connection between Fisher Information and layer importance in LLM recommendation systems.

**2. Adaptive Weight Strategy**: Dynamic Fisher weight calculation based on task complexity and user-item interaction patterns.

**3. Semantic Emphasis**: Novel layer depth bias (β=0.8) that exponentially increases weights for upper semantic layers.

### Practical Impact

Our approach addresses critical deployment challenges in industrial recommender systems:
- **Scalability**: Enables LLM-powered recommendations with limited computational resources
- **Efficiency**: Maintains semantic understanding while reducing inference costs
- **Adaptability**: Framework generalizes to different recommendation domains and LLM architectures

### Conclusion

This work demonstrates that **layerwise knowledge distillation guided by Fisher Information Matrix** can effectively compress LLM-based recommender systems while preserving semantic reasoning capabilities. The key insight that "semantic layers matter more than syntactic layers" for recommendation tasks opens new directions for efficient LLM deployment in personalized systems.

Our Fisher-driven approach represents a significant advancement in knowledge distillation methodology, providing both theoretical rigor and practical applicability for next-generation recommender systems.

---

**Reproducibility**: All code, data, and experimental configurations are available at: [GitHub Repository Link]

**Contact**: zwang000@usc.edu