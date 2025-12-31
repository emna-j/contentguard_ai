

__version__ = "2.1.0"
__author__ = "ContentGuard Team"

from .moderation_engine import ModerationEngine, engine
from .model_loader import DeepfakeModel, get_model_loader
from .metrics import DetailedMetrics
from .explainability import (
    DeepfakeExplainer,
    GradCAM,
    SaliencyMapGenerator,
    AttentionVisualizer,
    ExplainabilityEngine
)

__all__ = [
    'ModerationEngine',
    'engine',
    'DeepfakeModel',
    'get_model_loader',
    'DetailedMetrics',
    'DeepfakeExplainer',
    'GradCAM',
    'SaliencyMapGenerator',
    'AttentionVisualizer',
    'ExplainabilityEngine'
]