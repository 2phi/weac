"""
Component Classes for Inputs of the WEAC model.
"""

from .config import Config
from .criteria_config import CriteriaConfig
from .layer import Layer, WeakLayer
from .model_input import ModelInput
from .segment import Segment
from .scenario_config import ScenarioConfig, SystemType, TouchdownMode

__all__ = [
    "Config",
    "WeakLayer",
    "Layer",
    "Segment",
    "CriteriaConfig",
    "ScenarioConfig",
    "ModelInput",
    "SystemType",
    "TouchdownMode",
]
