"""
This module defines the input data model for the WEAC simulation.

We utilize the pydantic library instead of dataclasses to define the input data model. The advantages of pydantic are:
1. validate the input data for the WEAC simulation, compared to __post_init__ methods.
2. generate JSON schemas for the input data, which is good for API endpoints.
3. generate the documentation for the input data.

Pydantic syntax is for a field:
field_name: type = Field(..., gt=0, description="Description")
- typing, default value, conditions, description
"""
import logging
import json
from typing import List, Literal
from pydantic import BaseModel, Field

from weac_2.components.scenario_config import ScenarioConfig
from weac_2.components.layer import WeakLayer, Layer
from weac_2.components.segment import Segment
from weac_2.components.criteria_config import CriteriaConfig

logger = logging.getLogger(__name__)

class ModelInput(BaseModel):
    """
    Comprehensive input data model for a WEAC simulation.

    Args:
    -----
        scenario_config : ScenarioConfig
            Scenario configuration.
        weak_layer : WeakLayer
            Weak layer properties.
        layers : List[Layer]
            List of snow slab layers.
        segments : List[Segment]
            List of segments defining the slab geometry and loading.
        criteria_config : CriteriaConfig, optional
            Criteria overrides.
    """
    scenario_config: ScenarioConfig = Field(..., description="Scenario configuration")
    weak_layer: WeakLayer = Field(..., description="Weak layer")
    layers: List[Layer] = Field(..., description="List of layers")
    segments: List[Segment] = Field(..., description="Segments")
    
    criteria_config: CriteriaConfig = Field(default=CriteriaConfig(), description="Criteria overrides")

if __name__ == "__main__":
    # Example usage requiring all mandatory fields for proper instantiation
    example_scenario_config = ScenarioConfig(phi=30, touchdown=False, system='skiers')
    example_weak_layer = WeakLayer(rho=200, h=10) # grain_size, temp, E, G_I have defaults
    
    example_layers = [
        Layer(rho=250, h=100), # grain_size, temp have defaults
        Layer(rho=280, h=150)
    ]
    example_segments = [
        Segment(l=5000, k=True, m=80),
        Segment(l=3000, k=False, m=0)
    ]
    example_criteria_overrides = CriteriaConfig() # All fields have defaults

    model_input = ModelInput(
        scenario_config=example_scenario_config,
        weak_layer=example_weak_layer,
        layers=example_layers,
        segments=example_segments,
        criteria_config=example_criteria_overrides
    )
    print(model_input.model_dump_json(indent=2))
    print("\n\n")
    schema_json = json.dumps(ModelInput.model_json_schema(), indent=2)
    print(schema_json)