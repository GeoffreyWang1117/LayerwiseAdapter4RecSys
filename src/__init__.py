"""
Layerwise-Adapter: Knowledge Distillation for LLM Recommendation Systems

This package implements layerwise knowledge distillation techniques
for large language model based recommendation systems.
"""

__version__ = "2.0.0"
__author__ = "Layerwise-Adapter Team"

from .core import (
    LayerwiseDistillation,
    DistillationTrainer,
    FisherInformationCalculator
)

from .recommender import (
    BaseRecommender,
    MultiModelRecommender
)

__all__ = [
    "LayerwiseDistillation",
    "DistillationTrainer", 
    "FisherInformationCalculator",
    "BaseRecommender",
    "MultiModelRecommender"
]
