"""
Component Classes for Inputs of the WEAC model.
"""

from .config import Config
from .criteria_config import CriteriaConfig
from .layer import Layer, WeakLayer
from .model_input import ModelInput
from .presets import (
    LESS_WEAK_LAYER,
    VERY_WEAK_LAYER,
    WEAK_LAYER,
    WEAK_LAYER_PRESETS,
    weak_layer_from_preset,
)
from .scenario_config import ScenarioConfig, SystemType, TouchdownMode
from .segment import Segment

__all__ = [
    "Config",
    "CriteriaConfig",
    "Layer",
    "LESS_WEAK_LAYER",
    "ModelInput",
    "ScenarioConfig",
    "Segment",
    "SystemType",
    "TouchdownMode",
    "VERY_WEAK_LAYER",
    "WEAK_LAYER",
    "WEAK_LAYER_PRESETS",
    "WeakLayer",
    "weak_layer_from_preset",
]
