# Layerwise Adapter Development Guide

## Project Architecture

This is a research framework for layerwise importance analysis in Transformer-based recommendation systems. The codebase follows a modular structure:

- `src/core/`: Core algorithms for Fisher information analysis, layerwise distillation, and QLoRA integration
- `src/recommender/`: Recommendation system implementations with layerwise optimization
- `experiments/`: Research experiments and validation scripts
- `results/`: Generated analysis results and visualizations

## Key Patterns

### Analysis Framework Pattern
Most analysis scripts follow this pattern:
```python
class SomeAnalyzer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path('results/analysis_type')
        
    def analyze_something(self) -> Dict[str, Any]:
        # Core analysis logic
        
    def create_visualizations(self, results):
        # Generate matplotlib visualizations
        
    def save_results(self, results):
        # Save JSON results and markdown reports
```

### Configuration Management
- Use `@dataclass` for configuration objects
- Store configs in `configs/` directory as YAML files
- Default values should support both research and production use

### Data Flow
1. **Input**: Amazon/MovieLens datasets in `dataset/` directory
2. **Processing**: Core algorithms in `src/core/` compute layer importance
3. **Analysis**: Experiments in `experiments/` generate comprehensive reports
4. **Output**: Results saved in `results/` with timestamp-based naming

## Development Workflows

### Running Experiments
```bash
# Core theoretical validation
python experiments/advanced_theoretical_validation.py

# Architecture exploration
python experiments/multi_layer_architecture_exploration.py

# Production validation
python experiments/qlora_integration_validation.py
```

### Adding New Analysis
1. Create new analyzer class in appropriate `src/` subdirectory
2. Add experiment script in `experiments/`
3. Use existing visualization and reporting patterns
4. Follow the dataclass config pattern

## Critical Implementation Details

### Fisher Information Computation
- Uses diagonal approximation for computational efficiency
- Implements both epistemic and aleatoric uncertainty decomposition
- Critical layers identified through concentration measures

### QLoRA Integration
- 4-bit quantization with LoRA adapters (rank=16 optimal)
- Quantized weights stored as int8 with scale/zero-point
- Training freezes base weights, updates only LoRA parameters

### Layer Importance Analysis
- Multi-modal distribution: bottom (10%), middle (50%), top (90%) layers critical
- SHAP values computed for feature-level importance
- Information-theoretic measures track representation efficiency

## Performance Considerations

- Use `torch.cuda.empty_cache()` after large model operations
- Implement batch processing for large-scale analysis
- Results are cached with timestamp-based filenames
- GPU memory management critical for multi-layer analysis

## Testing and Validation

- All experiments include multi-seed validation (default: 5 seeds)
- Statistical significance testing using ANOVA
- Confidence intervals reported for key metrics
- Results reproducible with fixed random seeds
