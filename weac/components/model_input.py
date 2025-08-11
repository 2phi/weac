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

import json
import logging
from typing import List

from pydantic import BaseModel, Field

from weac.components.layer import Layer, WeakLayer
from weac.components.scenario_config import ScenarioConfig
from weac.components.segment import Segment

logger = logging.getLogger(__name__)


class ModelInput(BaseModel):
    """
    Comprehensive input data model for a WEAC simulation.

    Parameters:
    ----------
        scenario_config : ScenarioConfig
            Scenario configuration.
        weak_layer : WeakLayer
            Weak layer properties.
        layers : List[Layer]
            List of snow slab layers.
        segments : List[Segment]
            List of segments defining the slab geometry and loading.
    """

    weak_layer: WeakLayer = Field(
        default_factory=lambda: WeakLayer(rho=125, h=20, E=1.0),
        description="Weak layer",
    )
    layers: List[Layer] = Field(
        default_factory=lambda: [Layer(rho=250, h=100)], description="List of layers"
    )
    scenario_config: ScenarioConfig = Field(
        default_factory=ScenarioConfig, description="Scenario configuration"
    )
    segments: List[Segment] = Field(
        default_factory=lambda: [
            Segment(length=5000, has_foundation=True, m=100),
            Segment(length=5000, has_foundation=True, m=0),
        ],
        description="Segments",
    )

    def model_post_init(self, _ctx):
        # Check that the last segment does not have a mass
        if len(self.segments) == 0:
            raise ValueError("At least one segment is required")
        if len(self.layers) == 0:
            raise ValueError("At least one layer is required")
        if self.segments[-1].m != 0:
            raise ValueError("The last segment must have a mass of 0")


if __name__ == "__main__":
    # Example usage requiring all mandatory fields for proper instantiation
    example_scenario_config = ScenarioConfig(phi=30, system_type="skiers")
    # example_weak_layer = WeakLayer(
    #     rho=200, h=10
    # )  # grain_size, temp, E, G_I have defaults

    example_layers = [
        Layer(rho=250, h=100),  # grain_size, temp have defaults
        Layer(rho=280, h=150),
    ]
    example_segments = [
        Segment(length=5000, has_foundation=True, m=80),
        Segment(length=3000, has_foundation=False, m=0),
    ]

    model_input = ModelInput(
        scenario_config=example_scenario_config,
        layers=example_layers,
        segments=example_segments,
    )
    print(model_input.model_dump_json(indent=2))
    print("\n\n")
    schema_json = json.dumps(ModelInput.model_json_schema(), indent=2)
    print(schema_json)
