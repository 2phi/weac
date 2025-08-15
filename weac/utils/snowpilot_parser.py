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
from snowpylot.layer import Layer as SnowpylotLayer
from snowpylot.snow_pit import SnowPit
from snowpylot.snow_profile import DensityObs

# Import WEAC components
from weac.components import (
    Layer,
    WeakLayer,
)
from weac.utils.geldsetzer import compute_density

logger = logging.getLogger(__name__)

convert_to_mm = {"cm": 10, "mm": 1, "m": 1000, "dm": 100}
convert_to_deg = {"deg": 1, "rad": 180 / np.pi}


class SnowPilotParser:
    def __init__(self, file_path: str):
        self.snowpit: SnowPit = caaml_parser(file_path)

    def extract_layers(self) -> Tuple[List[Layer], List[str]]:
        """Extract layers from snowpit."""
        snowpit = self.snowpit
        # Extract layers from snowpit: List[SnowpylotLayer]
        sp_layers: List[SnowpylotLayer] = [
            layer
            for layer in snowpit.snow_profile.layers
            if layer.depth_top is not None
        ]
        sp_layers = sorted(sp_layers, key=lambda x: x.depth_top[0])  # type: ignore

        # Extract density layers from snowpit: List[DensityObs]
        sp_density_layers: List[DensityObs] = [
            layer
            for layer in snowpit.snow_profile.density_profile
            if layer.depth_top is not None
        ]
        sp_density_layers = sorted(sp_density_layers, key=lambda x: x.depth_top[0])  # type: ignore

        # Populate WEAC layers: List[Layer]
        layers: List[Layer] = []
        density_methods: List[str] = []
        for i, layer in enumerate(sp_layers):
            # Parameters
            grain_type = None
            grain_size = None
            hand_hardness = None
            density = None
            thickness = None

            # extract THICKNESS
            if layer.thickness is not None:
                thickness, unit = layer.thickness
                thickness = thickness * convert_to_mm[unit]  # Convert to mm
            else:
                raise ValueError("Thickness not found")

            # extract GRAIN TYPE and SIZE
            if layer.grain_form_primary:
                if layer.grain_form_primary.grain_form:
                    grain_type = layer.grain_form_primary.grain_form
                    if layer.grain_form_primary.grain_size_avg:
                        grain_size = (
                            layer.grain_form_primary.grain_size_avg[0]
                            * convert_to_mm[layer.grain_form_primary.grain_size_avg[1]]
                        )
                    elif layer.grain_form_primary.grain_size_max:
                        grain_size = (
                            layer.grain_form_primary.grain_size_max[0]
                            * convert_to_mm[layer.grain_form_primary.grain_size_max[1]]
                        )

            # extract DENSITY
            # Get layer depth range in mm for density matching
            layer_depth_top_mm = layer.depth_top[0] * convert_to_mm[layer.depth_top[1]]
            layer_depth_bottom_mm = layer_depth_top_mm + thickness
            # Try to find density measurement that overlaps with this layer
            measured_density = self.get_density_for_layer_range(
                layer_depth_top_mm, layer_depth_bottom_mm, sp_density_layers
            )

            # Handle hardness and create layers accordingly
            if layer.hardness_top is not None and layer.hardness_bottom is not None:
                hand_hardness_top = layer.hardness_top
                hand_hardness_bottom = layer.hardness_bottom

                # Two hardness values - split into two layers
                half_thickness = thickness / 2
                layer_mid_depth_mm = layer_depth_top_mm + half_thickness

                # Create top layer (first half)
                if measured_density is not None:
                    density_top = self.get_density_for_layer_range(
                        layer_depth_top_mm, layer_mid_depth_mm, sp_density_layers
                    )
                    if density_top is None:
                        density_methods.append("geldsetzer")
                        density_top = compute_density(grain_type, hand_hardness_top)
                    else:
                        density_methods.append("density_obs")
                else:
                    density_methods.append("geldsetzer")
                    density_top = compute_density(grain_type, hand_hardness_top)

                layers.append(
                    Layer(
                        rho=density_top,
                        h=half_thickness,
                        grain_type=grain_type,
                        grain_size=grain_size,
                        hand_hardness=hand_hardness_top,
                    )
                )

                # Create bottom layer (second half)
                if measured_density is not None:
                    density_bottom = self.get_density_for_layer_range(
                        layer_mid_depth_mm, layer_depth_bottom_mm, sp_density_layers
                    )
                    if density_bottom is None:
                        density_methods.append("geldsetzer")
                        density_bottom = compute_density(
                            grain_type, hand_hardness_bottom
                        )
                    else:
                        density_methods.append("density_obs")
                else:
                    try:
                        density_methods.append("geldsetzer")
                        density_bottom = compute_density(
                            grain_type, hand_hardness_bottom
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Error computing density for layer {layer.depth_top}: {e}"
                        )

                layers.append(
                    Layer(
                        rho=density_bottom,
                        h=half_thickness,
                        grain_type=grain_type,
                        grain_size=grain_size,
                        hand_hardness=hand_hardness_bottom,
                    )
                )
            else:
                # Single hardness value - create one layer
                hand_hardness = layer.hardness

                if measured_density is not None:
                    density = measured_density
                    density_methods.append("density_obs")
                else:
                    try:
                        density_methods.append("geldsetzer")
                        density = compute_density(grain_type, hand_hardness)
                    except Exception:
                        raise AttributeError(
                            "Layer is missing density information; density profile, hand hardness and grain type are all missing. Excluding SnowPit from calculations."
                        )

                layers.append(
                    Layer(
                        rho=density,
                        h=thickness,
                        grain_type=grain_type,
                        grain_size=grain_size,
                        hand_hardness=hand_hardness,
                    )
                )

        if len(layers) == 0:
            raise AttributeError(
                "No layers found for snowpit. Excluding SnowPit from calculations."
            )
        return layers, density_methods

    def get_density_for_layer_range(
        self,
        layer_top_mm: float,
        layer_bottom_mm: float,
        sp_density_layers: List[DensityObs],
    ) -> float | None:
        """Find density measurements that overlap with the given layer depth range.

        Args:
            layer_top_mm: Top depth of layer in mm
            layer_bottom_mm: Bottom depth of layer in mm
            sp_density_layers: List of density observations

        Returns:
            Average density from overlapping measurements, or None if no overlap
        """
        if not sp_density_layers:
            return None

        overlapping_densities = []
        overlapping_weights = []

        for density_obs in sp_density_layers:
            if density_obs.depth_top is None or density_obs.thickness is None:
                continue

            # Convert density observation depth range to mm
            density_top_mm = (
                density_obs.depth_top[0] * convert_to_mm[density_obs.depth_top[1]]
            )
            density_thickness_mm = (
                density_obs.thickness[0] * convert_to_mm[density_obs.thickness[1]]
            )
            density_bottom_mm = density_top_mm + density_thickness_mm

            # Check for overlap between layer and density measurement
            overlap_top = max(layer_top_mm, density_top_mm)
            overlap_bottom = min(layer_bottom_mm, density_bottom_mm)

            if overlap_top < overlap_bottom:  # There is overlap
                overlap_thickness = overlap_bottom - overlap_top

                # Extract density value
                if density_obs.density is not None:
                    density_value = density_obs.density[0]  # (value, unit)

                    overlapping_densities.append(density_value)
                    overlapping_weights.append(overlap_thickness)

        if overlapping_densities:
            # Calculate weighted average based on overlap thickness
            total_weight = sum(overlapping_weights)
            if total_weight > 0:
                weighted_density = (
                    sum(
                        d * w
                        for d, w in zip(overlapping_densities, overlapping_weights)
                    )
                    / total_weight
                )
                return float(weighted_density)
        return None

    def extract_weak_layer_and_layers_above(
        self, weak_layer_depth: float, layers: List[Layer]
    ) -> Tuple[WeakLayer, List[Layer]]:
        """Extract weak layer and layers above the weak layer for the given depth_top extracted from the stability test."""
        depth = 0
        layers_above = []
        weak_layer_rho = None
        weak_layer_hand_hardness = None
        weak_layer_grain_type = None
        weak_layer_grain_size = None
        if weak_layer_depth <= 0:
            raise ValueError(
                "The depth of the weak layer is not positive. Excluding SnowPit from calculations."
            )
        if weak_layer_depth > sum([layer.h for layer in layers]):
            raise ValueError(
                "The depth of the weak layer is below the recorded layers. Excluding SnowPit from calculations."
            )
        layers = [layer.model_copy(deep=True) for layer in layers]
        for i, layer in enumerate(layers):
            if depth + layer.h < weak_layer_depth:
                layers_above.append(layer)
                depth += layer.h
            elif depth < weak_layer_depth and depth + layer.h > weak_layer_depth:
                layer.h = weak_layer_depth - depth
                layers_above.append(layer)
                weak_layer_rho = layers[i].rho
                weak_layer_hand_hardness = layers[i].hand_hardness
                weak_layer_grain_type = layers[i].grain_type
                weak_layer_grain_size = layers[i].grain_size
                break
            elif depth + layer.h == weak_layer_depth:
                if i + 1 < len(layers):
                    layers_above.append(layer)
                    weak_layer_rho = layers[i + 1].rho
                    weak_layer_hand_hardness = layers[i + 1].hand_hardness
                    weak_layer_grain_type = layers[i + 1].grain_type
                    weak_layer_grain_size = layers[i + 1].grain_size
                else:
                    weak_layer_rho = layers[i].rho
                    weak_layer_hand_hardness = layers[i].hand_hardness
                    weak_layer_grain_type = layers[i].grain_type
                    weak_layer_grain_size = layers[i].grain_size
                break

        weak_layer = WeakLayer(
            rho=weak_layer_rho,
            h=20.0,
            hand_hardness=weak_layer_hand_hardness,
            grain_type=weak_layer_grain_type,
            grain_size=weak_layer_grain_size,
        )
        if len(layers_above) == 0:
            raise ValueError("No layers above weak layer found")
        return weak_layer, layers_above

    # def _assemble_model_inputs(
    #     self,
    #     snowpit: SnowPit,
    #     layers: List[Layer],
    #     psts: bool = True,
    #     ects: bool = True,
    #     cts: bool = True,
    #     rblocks: bool = True,
    # ) -> List[ModelInput]:
    #     """Extract scenarios from snowpit stability tests."""
    #     scenarios: List[ModelInput] = []

    #     # Extract slope angle from snowpit
    #     slope_angle = snowpit.core_info.location.slope_angle
    #     if slope_angle is not None:
    #         slope_angle = slope_angle[0] * convert_to_deg[slope_angle[1]]
    #     else:
    #         raise ValueError("Slope angle not found for snowpit")

    #     # Add scenarios for PropSawTest
    #     psts: List[PropSawTest] = snowpit.stability_tests.PST
    #     if len(psts) > 0 and psts:
    #         # Implement logic that finds cut length based on PST
    #         for pst in psts:
    #             if pst.failure:
    #                 continue
    #             segments = []
    #             if (
    #                 pst.cut_length is not None
    #                 and pst.column_length is not None
    #                 and pst.depth_top is not None
    #             ):
    #                 if pst.depth_top <= 0:
    #                     raise ValueError(
    #                         "The depth of the weak layer is not positive. Excluding SnowPit from calculations."
    #                     )
    #                 if pst.depth_top[0] * convert_to_mm[pst.depth_top[1]] > sum(
    #                     [layer.h for layer in layers]
    #                 ):
    #                     raise ValueError(
    #                         "The depth of the weak layer is below the recorded layers. Excluding SnowPit from calculations."
    #                     )
    #                 cut_length = pst.cut_length[0] * convert_to_mm[pst.cut_length[1]]
    #                 column_length = (
    #                     pst.column_length[0] * convert_to_mm[pst.column_length[1]]
    #                 )
    #                 segments.append(
    #                     Segment(length=cut_length, has_foundation=False, m=0)
    #                 )
    #                 segments.append(
    #                     Segment(
    #                         length=column_length - cut_length, has_foundation=True, m=0
    #                     )
    #                 )
    #                 scenario_config = ScenarioConfig(
    #                     system_type="-pst",
    #                     phi=slope_angle,
    #                     cut_length=cut_length,
    #                 )
    #                 weak_layer, layers_above = (
    #                     self._extract_weak_layer_and_layers_above(
    #                         pst.depth_top[0] * convert_to_mm[pst.depth_top[1]],
    #                         layers,
    #                     )
    #                 )
    #                 if weak_layer is not None:
    #                     logger.info(
    #                         "Adding PST scenario with cut_length %s and column_length %s and weak_layer depth %s",
    #                         cut_length,
    #                         column_length,
    #                         sum([layer.h for layer in layers_above]),
    #                     )
    #                     scenarios.append(
    #                         ModelInput(
    #                             layers=layers_above,
    #                             weak_layer=weak_layer,
    #                             scenario_config=scenario_config,
    #                             segments=segments,
    #                         )
    #                     )
    #             else:
    #                 continue

    #     # Add scenarios for ExtColumnTest, ComprTest, and RBlockTest
    #     standard_segments = [
    #         Segment(length=1000, has_foundation=True, m=0),
    #         Segment(length=1000, has_foundation=True, m=0),
    #     ]
    #     standard_scenario_config = ScenarioConfig(system_type="skier", phi=slope_angle)
    #     depth_tops = set()
    #     ects: List[ExtColumnTest] = snowpit.stability_tests.ECT
    #     if len(ects) > 0 and ects:
    #         for ect in ects:
    #             if ect.depth_top is not None:
    #                 depth_tops.add(ect.depth_top[0] * convert_to_mm[ect.depth_top[1]])
    #     cts: List[ComprTest] = snowpit.stability_tests.CT
    #     if len(cts) > 0 and cts:
    #         for ct in cts:
    #             if ct.depth_top is not None:
    #                 depth_tops.add(ct.depth_top[0] * convert_to_mm[ct.depth_top[1]])
    #     rblocks: List[RBlockTest] = snowpit.stability_tests.RBlock
    #     if len(rblocks) > 0 and rblocks:
    #         for rblock in rblocks:
    #             if rblock.depth_top is not None:
    #                 depth_tops.add(
    #                     rblock.depth_top[0] * convert_to_mm[rblock.depth_top[1]]
    #                 )

    #     for depth_top in sorted(depth_tops):
    #         weak_layer, layers_above = self._extract_weak_layer_and_layers_above(
    #             depth_top, layers
    #         )
    #         scenarios.append(
    #             ModelInput(
    #                 layers=layers_above,
    #                 weak_layer=weak_layer,
    #                 scenario_config=standard_scenario_config,
    #                 segments=standard_segments,
    #             )
    #         )
    #         logger.info(
    #             "Adding scenario with depth_top %s mm",
    #             sum([layer.h for layer in layers_above]),
    #         )

    #     # Add scenario for no stability tests
    #     if len(scenarios) == 0:
    #         scenarios.append(
    #             ModelInput(
    #                 layers=layers,
    #                 weak_layer=WeakLayer(rho=125, h=30),
    #                 scenario_config=standard_scenario_config,
    #                 segments=standard_segments,
    #             )
    #         )
    #     return scenarios
