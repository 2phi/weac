"""
Utilizes the snowpylot library to convert a CAAML file to a WEAC ModelInput.

The snowpylot library is used to parse the CAAML file and extract the snowpit.
The snowpit is then converted to a List of WEAC ModelInput.

Based on the different stability tests performed, several scenarios are created.
Each scenario is a WEAC ModelInput.

The scenarios are created based on the following logic:
- For each PropSawTest, a scenario is created with `the cut length` and `a standard segment.`
- For each ExtColumnTest, a scenario is created with `a standard segment.`
- For each ComprTest, a scenario is created with `a standard segment.`
- For each RBlockTest, a scenario is created with `a standard segment.`

The `a standard segment` is a segment with a length of 1000 mm and a foundation of True.

The `the cut length` is the cut length of the PropSawTest.
The `the column length` is the column length of the PropSawTest.
"""

import logging
from typing import List, Tuple
import numpy as np

from snowpylot import caaml_parser
from snowpylot.snow_pit import SnowPit
from snowpylot.stability_tests import PropSawTest, ExtColumnTest, ComprTest, RBlockTest
from snowpylot.layer import Layer as SnowpylotLayer

# Import WEAC components
from weac_2.components import (
    Layer,
    WeakLayer,
    ScenarioConfig,
    Segment,
    ModelInput,
)
from weac_2.utils.geldsetzer import compute_density

logger = logging.getLogger(__name__)

convert_to_mm = {"cm": 10, "mm": 1, "m": 1000, "dm": 100}
convert_to_deg = {"deg": 1, "rad": 180 / np.pi}


def extract_layers(snowpit: SnowPit) -> List[Layer]:
    """Extract layers from snowpit."""
    sp_layers: List[SnowpylotLayer] = [
        layer for layer in snowpit.snow_profile.layers if layer.depth_top is not None
    ]
    sp_layers = sorted(sp_layers, key=lambda x: x.depth_top[0])  # type: ignore

    layers: List[Layer] = []
    for layer in sp_layers:
        # Extract hardness from [hardness, hardness_top, hardness_bottom]
        if layer.hardness is not None:
            hardness = layer.hardness
        elif layer.hardness_top is not None and layer.hardness_bottom is not None:
            hardness = (layer.hardness_top, layer.hardness_bottom)
        else:
            raise ValueError(
                "Hardness not found for layer: "
                + str(layer.depth_top)
                + " "
                + str(layer.thickness)
            )
        if (
            layer.grain_form_primary is not None
            and layer.grain_form_primary.grain_form is not None
        ):
            grain_form = layer.grain_form_primary.grain_form
        else:
            grain_form = "!skip"

        density = compute_density(grain_form, hardness)
        if layer.thickness is not None:
            thickness, unit = layer.thickness
            thickness = thickness * convert_to_mm[unit]  # Convert to mm
        else:
            raise ValueError(
                "Thickness not found for layer: "
                + str(layer.depth_top)
                + " "
                + str(layer.thickness)
            )
        layers.append(Layer(rho=density, h=thickness))
    if len(layers) == 0:
        raise ValueError("No layers found for snowpit")
    return layers


def extract_scenarios(snowpit: SnowPit, layers: List[Layer]) -> List[ModelInput]:
    """Extract scenarios from snowpit stability tests."""
    scenarios: List[ModelInput] = []

    # Extract slope angle from snowpit
    slope_angle = snowpit.core_info.location.slope_angle
    if slope_angle is not None:
        slope_angle = slope_angle[0] * convert_to_deg[slope_angle[1]]
    else:
        raise ValueError("Slope angle not found for snowpit")

    # Add scenarios for PropSawTest
    psts: List[PropSawTest] = snowpit.stability_tests.PST
    if len(psts) > 0:
        # Implement logic that finds cut length based on PST
        for pst in psts:
            segments = []
            if (
                pst.cut_length is not None
                and pst.column_length is not None
                and pst.depth_top is not None
            ):
                cut_length = pst.cut_length[0] * convert_to_mm[pst.cut_length[1]]
                column_length = (
                    pst.column_length[0] * convert_to_mm[pst.column_length[1]]
                )
                segments.append(Segment(length=cut_length, has_foundation=False, m=0))
                segments.append(
                    Segment(length=column_length - cut_length, has_foundation=True, m=0)
                )
                scenario_config = ScenarioConfig(
                    system_type="-pst",
                    phi=slope_angle,
                    crack_length=cut_length,
                )
                weak_layer, layers_above = extract_weak_layer_and_layers_above(
                    snowpit, pst.depth_top[0] * convert_to_mm[pst.depth_top[1]], layers
                )
                if weak_layer is not None:
                    logger.info(
                        "Adding PST scenario with cut_length %s and column_length %s and weak_layer depth %s",
                        cut_length,
                        column_length,
                        sum([layer.h for layer in layers_above]),
                    )
                    scenarios.append(
                        ModelInput(
                            layers=layers_above,
                            weak_layer=weak_layer,
                            scenario_config=scenario_config,
                            segments=segments,
                        )
                    )
            else:
                continue

    # Add scenarios for ExtColumnTest, ComprTest, and RBlockTest
    standard_segments = [
        Segment(length=1000, has_foundation=True, m=0),
        Segment(length=1000, has_foundation=True, m=0),
    ]
    standard_scenario_config = ScenarioConfig(system_type="skier", phi=slope_angle)
    depth_tops = set()
    ects: List[ExtColumnTest] = snowpit.stability_tests.ECT
    if len(ects) > 0:
        for ect in ects:
            if ect.depth_top is not None:
                depth_tops.add(ect.depth_top[0] * convert_to_mm[ect.depth_top[1]])
    cts: List[ComprTest] = snowpit.stability_tests.CT
    if len(cts) > 0:
        for ct in cts:
            if ct.depth_top is not None:
                depth_tops.add(ct.depth_top[0] * convert_to_mm[ct.depth_top[1]])
    rblocks: List[RBlockTest] = snowpit.stability_tests.RBlock
    if len(rblocks) > 0:
        for rblock in rblocks:
            if rblock.depth_top is not None:
                depth_tops.add(rblock.depth_top[0] * convert_to_mm[rblock.depth_top[1]])

    for depth_top in sorted(depth_tops):
        weak_layer, layers_above = extract_weak_layer_and_layers_above(
            snowpit, depth_top, layers
        )
        scenarios.append(
            ModelInput(
                layers=layers_above,
                weak_layer=weak_layer,
                scenario_config=standard_scenario_config,
                segments=standard_segments,
            )
        )
        logger.info(
            "Adding scenario with depth_top %s and weak_layer depth %s",
            depth_top,
            sum([layer.h for layer in layers_above]),
        )

    # Add scenario for no stability tests
    if len(scenarios) == 0:
        scenarios.append(
            ModelInput(
                layers=layers,
                weak_layer=WeakLayer(rho=125, h=30),
                scenario_config=standard_scenario_config,
                segments=standard_segments,
            )
        )
    return scenarios


def extract_weak_layer_and_layers_above(
    snowpit: SnowPit, depth_top: float, layers: List[Layer]
) -> Tuple[WeakLayer, List[Layer]]:
    """Extract weak layer and layers above the weak layer for the given depth_top extracted from the stability test."""
    depth = 0
    layers_above = []
    for i, layer in enumerate(layers):
        if depth + layer.h < depth_top:
            layers_above.append(layer)
            depth += layer.h
        elif depth < depth_top and depth + layer.h > depth_top:
            layers_above.append(Layer(rho=layers[i].rho, h=depth_top - depth))
            weak_layer_rho = layers[i].rho
            break
        elif depth + layer.h == depth_top:
            layers_above.append(layer)
            if i + 1 < len(layers):
                weak_layer_rho = layers[i + 1].rho
            else:
                weak_layer_rho = layers[i].rho
            break
    weak_layer = WeakLayer(rho=weak_layer_rho, h=depth_top - depth)
    if len(layers_above) == 0:
        raise ValueError("No layers above weak layer found")
    return weak_layer, layers_above


def convert_snowpit_to_weac(file_path: str) -> List[ModelInput]:
    """Convert CAAML file to WEAC ModelInput."""
    snowpit = caaml_parser(file_path)
    layers = extract_layers(snowpit)
    model_inputs: List[ModelInput] = extract_scenarios(snowpit, layers)
    return model_inputs
