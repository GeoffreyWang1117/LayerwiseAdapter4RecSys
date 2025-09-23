"""
Core modules for Layerwise-Adapter
"""

# Only import modules that exist and don't have import issues
try:
    from .fisher_information import FisherInformationCalculator  
    __all__ = ["FisherInformationCalculator"]
except ImportError:
    __all__ = []

# Advanced modules (import individually when needed)
# from .adaptive_layer_selector import AdaptiveLayerSelector, TaskConfig
# from .multi_dataset_validator import MultiDatasetValidator
# from .visualization_suite_fixed import LayerwiseVisualizationSuite
# from .rl_layer_optimizer import RLLayerOptimizer
# from .multi_task_adapter import MultiTaskLayerwiseAdapter
# from .sota_comparison import SOTAComparisonFramework
