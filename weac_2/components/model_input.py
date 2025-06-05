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

from weac_2.components.layers import WeakLayer, Layer

logger = logging.getLogger(__name__)


class Scenario(BaseModel):
    """
    Configuration for the overall scenario, such as slope angle.

    Args:
        phi (float): Slope angle in degrees.
        left_boundary (str): Boundary one of 'inf' or 'free'.
        right_boundary (str): Boundary one of 'inf' or 'free'.
    """
    phi: float = Field(0, description="Slope angle in degrees, counterclockwise positive")
    touchdown: bool = Field(False, description="Whether to calculate the touchdown")
    # TODO: add more descriptive/human-readable system names
    system: Literal['skier', 'skiers', 'pst-', 'pst+'] = Field('skiers', description="Type of system, '-pst', '+pst', ....")
    # left_boundary: str = Field('inf', description="Boundary one of 'inf' or 'free'")
    # right_boundary: str = Field('inf', description="Boundary one of 'inf' or 'free'")


class Segment(BaseModel):
    """
    Defines a segment of the snow slab, its length, foundation support, and applied loads.

    Args:
        length (float): Segment length in mm.
        fractured (bool): Boolean indicating whether the segment is fractured or not.
        skier_weight (float): Skier weight at segments right edge in kg. Defaults to 0.
        surface_load (float): Surface load in kPa. Defaults to 0.
    """
    length: float = Field(..., gt=0, description="Segment length in mm")
    fractured: bool = Field(..., description="Boolean indicating whether the segment is fractured or not")
    skier_weight: float = Field(0, ge=0, description="Skier weight at segment right edge in kg")
    surface_load: float = Field(0, ge=0, description="Surface load in kPa")

class CriteriaOverrides(BaseModel):
    """
    Parameters defining the interaction between different failure modes.

    Args:
        fn (float): Failure mode interaction exponent for normal stress. Defaults to 1.
        fm (float): Failure mode interaction exponent for normal strain. Defaults to 1.
        gn (float): Failure mode interaction exponent for closing energy release rate. Defaults to 1.
        gm (float): Failure mode interaction exponent for shearing energy release rate. Defaults to 1.
    """
    fn: float = Field(1, gt=0, description="Failure mode interaction exponent for normal stress")
    fm: float = Field(1, gt=0, description="Failure mode interaction exponent for normal strain")
    gn: float = Field(1, gt=0, description="Failure mode interaction exponent for closing energy release rate")
    gm: float = Field(1, gt=0, description="Failure mode interaction exponent for shearing energy release rate")

class ModelInput(BaseModel):
    """
    Comprehensive input data model for a WEAC simulation.

    Args:
        scenario_config (ScenarioConfig): Scenario configuration.
        weak_layer (WeakLayer): Weak layer properties.
        layers (List[Layer]): List of snow slab layers.
        segments (List[Segment]): List of segments defining the slab geometry and loading.
        criteria_overrides (CriteriaOverrides): Criteria overrides.
    """
    scenario: Scenario = Field(..., description="Scenario configuration")
    weak_layer: WeakLayer = Field(..., description="Weak layer")
    layers: List[Layer] = Field(..., description="List of layers")
    segments: List[Segment] = Field(..., description="Segments")
    criteria_overrides: CriteriaOverrides = Field(CriteriaOverrides(), description="Criteria overrides")

if __name__ == "__main__":
    # Example usage requiring all mandatory fields for proper instantiation
    example_scenario = Scenario(phi=30, touchdown=False, system='skiers')
    example_weak_layer = WeakLayer(density=200, thickness=10) # grain_size, temp, E, G_I have defaults
    example_layers = [
        Layer(rho=250, t=100), # grain_size, temp have defaults
        Layer(rho=280, t=150)
    ]
    example_segments = [
        Segment(length=5000, fractured=True, skier_weight=80, surface_load=0), # pi has default
        Segment(length=3000, fractured=False, skier_weight=0, surface_load=0)
    ]
    example_criteria_overrides = CriteriaOverrides() # All fields have defaults

    model_input = ModelInput(
        scenario=example_scenario,
        weak_layer=example_weak_layer,
        layers=example_layers,
        segments=example_segments,
        criteria_overrides=example_criteria_overrides
    )
    print(model_input.model_dump_json(indent=2))
    print("\n\n")
    schema_json = json.dumps(ModelInput.model_json_schema(), indent=2)
    print(schema_json)