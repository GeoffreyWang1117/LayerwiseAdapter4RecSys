# WWW2026 Supplementary Materials

This package contains supplementary materials for the paper:
**"Adaptive Layer Truncation for Efficient Knowledge Distillation in LLM-based Recommendation Systems"**

## Package Structure

```
supplementary_materials/
├── README.md                          # This file
├── supplementary_materials.md         # Detailed supplementary document
├── experimental_data/                 # All experimental results
│   ├── results/                      # Raw result files
│   └── analysis/                     # Analysis scripts
├── source_code/                      # Complete source code
│   ├── src/                         # Core modules
│   ├── experiments/                 # Experimental scripts
│   ├── scripts/                     # Utility scripts
│   └── requirements.txt             # Dependencies
└── visualizations/                   # All generated plots
    ├── plots/                       # Main experiment plots
    └── extended_plots/              # Extended analysis plots
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r source_code/requirements.txt
   ```

2. **Run Main Experiment**:
   ```bash
   cd source_code
   python experiments/www2026_adaptive_distillation.py
   ```

3. **Generate Extended Results**:
   ```bash
   python experiments/www2026_extended_experiment.py
   ```

4. **Cross-Domain Validation**:
   ```bash
   python experiments/simple_movielens_validation.py
   ```

## Key Results

- **75% parameter reduction** (8B → 34.8M parameters)
- **43.8% recommendation accuracy** maintained
- **Cross-domain validation** successful (Amazon → MovieLens)
- **4 layer importance methods** validated

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥24GB memory (for teacher model)
- **RAM**: ≥32GB system memory
- **Storage**: ≥50GB free space
- **CUDA**: Version 11.8 or later

## Reproducibility

All experiments use fixed random seeds (42) for reproducibility. Expected runtime:
- Full pipeline: ~2.5 hours on A100 GPU
- Core experiment: ~30 minutes
- Extended validation: ~1 hour

## Support

For questions or issues with the supplementary materials, please contact the authors through the WWW2026 review system.

---
Generated: 2025-09-16 17:12:10
