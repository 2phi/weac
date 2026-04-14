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

import numpy as np
from snowpylot import caaml_parser
from snowpylot.layer import Layer as SnowpylotLayer
from snowpylot.snow_pit import SnowPit
from snowpylot.snow_profile import DensityObs

# Import WEAC components
from weac.components import (
    Layer,
)
from weac.utils.geldsetzer import compute_density

logger = logging.getLogger(__name__)

convert_to_mm = {"cm": 10, "mm": 1, "m": 1000, "dm": 100}
convert_to_deg = {"deg": 1, "rad": 180 / np.pi}


def vertical_to_slope_normal_depth_scale(phi_deg: float) -> float:
    """Scale vertical (plumb) depth/thickness to distance along slope normal.

    CAAML / SnowPilot report depths from the surface along the vertical. WEAC slab
    layer thicknesses are measured normal to the slope. Following the convention
    used for SnowPilot import here, the plumb-line depth ``d_v`` is converted to
    slope-normal depth ``d_n`` by ``d_n = d_v * cos(phi)``, where ``phi`` is the
    slope angle from horizontal.
    """
    phi = np.deg2rad(float(phi_deg))
    c = float(np.cos(phi))
    if c <= 1e-6:
        raise ValueError(
            f"Slope angle too close to ±90° ({phi_deg}°); cannot convert vertical "
            "depths to slope-normal."
        )
    return c


class SnowPilotParser:
    """Parser for SnowPilot files using the snowpylot library."""

    def __init__(self, file_path: str):
        self.snowpit: SnowPit = caaml_parser(file_path)

    def pit_slope_angle_deg(self) -> float | None:
        """Slope angle from CAAML ``validSlopeAngle`` if present, else ``None``."""
        loc = getattr(self.snowpit.core_info, "location", None)
        if loc is None or not getattr(loc, "slope_angle", None):
            return None
        value, unit = loc.slope_angle
        return float(value) * convert_to_deg[unit]

    def extract_layers(
        self, slope_angle_deg: float = 0.0
    ) -> tuple[list[Layer], list[str]]:
        """Extract layers from snowpit.

        Depths and thicknesses in CAAML are measured along the vertical. For WEAC,
        pass the same slope angle (degrees from horizontal) as ``ScenarioConfig.phi``;
        thicknesses and matching depths are scaled to slope-normal by ``cos(phi)``.
        The default ``0`` leaves depths unchanged (vertical equals slope-normal on flat
        terrain). To use the angle stored in the pit file, pass
        ``slope_angle_deg=parser.pit_slope_angle_deg() or 0.0``.
        """
        snowpit = self.snowpit
        phi_deg = float(slope_angle_deg)
        depth_scale = (
            vertical_to_slope_normal_depth_scale(phi_deg) if phi_deg != 0.0 else 1.0
        )
        # Extract layers from snowpit: list[SnowpylotLayer]
        sp_layers: list[SnowpylotLayer] = [
            layer
            for layer in snowpit.snow_profile.layers
            if layer.depth_top is not None
        ]
        sp_layers = sorted(sp_layers, key=lambda x: x.depth_top[0])  # type: ignore

        # Extract density layers from snowpit: list[DensityObs]
        sp_density_layers: list[DensityObs] = [
            layer
            for layer in snowpit.snow_profile.density_profile
            if layer.depth_top is not None
        ]
        sp_density_layers = sorted(sp_density_layers, key=lambda x: x.depth_top[0])  # type: ignore

        # Populate WEAC layers: list[Layer]
        layers: list[Layer] = []
        density_methods: list[str] = []
        for _i, layer in enumerate(sp_layers):
            # Parameters
            grain_type = None
            grain_size = None
            hand_hardness = None
            density = None
            thickness = None

            # extract THICKNESS (CAAML: vertical mm) -> slope-normal mm for WEAC
            if layer.thickness is not None:
                thickness, unit = layer.thickness
                thickness = thickness * convert_to_mm[unit] * depth_scale
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
            # Depth range in slope-normal mm (density profile uses same vertical depths)
            layer_depth_top_mm = (
                layer.depth_top[0] * convert_to_mm[layer.depth_top[1]] * depth_scale
            )
            layer_depth_bottom_mm = layer_depth_top_mm + thickness
            # Try to find density measurement that overlaps with this layer
            measured_density = self._get_density_for_layer_range(
                layer_depth_top_mm,
                layer_depth_bottom_mm,
                sp_density_layers,
                depth_scale=depth_scale,
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
                    density_top = self._get_density_for_layer_range(
                        layer_depth_top_mm,
                        layer_mid_depth_mm,
                        sp_density_layers,
                        depth_scale=depth_scale,
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
                    density_bottom = self._get_density_for_layer_range(
                        layer_mid_depth_mm,
                        layer_depth_bottom_mm,
                        sp_density_layers,
                        depth_scale=depth_scale,
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
                    except Exception as exc:
                        raise AttributeError(
                            "Layer is missing density information; density profile, "
                            "hand hardness and grain type are all missing. "
                            "Excluding SnowPit from calculations."
                        ) from exc

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
                    except Exception as exc:
                        raise AttributeError(
                            "Layer is missing density information; density profile, "
                            "hand hardness and grain type are all missing. "
                            "Excluding SnowPit from calculations."
                        ) from exc

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
        sp_density_layers: list[DensityObs],
        *,
        depth_scale: float = 1.0,
    ) -> float | None:
        """Public wrapper for :meth:`_get_density_for_layer_range`."""
        return self._get_density_for_layer_range(
            layer_top_mm,
            layer_bottom_mm,
            sp_density_layers,
            depth_scale=depth_scale,
        )

    def _get_density_for_layer_range(
        self,
        layer_top_mm: float,
        layer_bottom_mm: float,
        sp_density_layers: list[DensityObs],
        *,
        depth_scale: float = 1.0,
    ) -> float | None:
        """Find density measurements that overlap with the given layer depth range.

        Args:
            layer_top_mm: Top depth of layer in mm (slope-normal)
            layer_bottom_mm: Bottom depth of layer in mm (slope-normal)
            sp_density_layers: list of density observations
            depth_scale: vertical-to-slope-normal factor applied to CAAML density depths

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

            # CAAML density depths are vertical; convert to slope-normal mm
            density_top_mm = (
                density_obs.depth_top[0]
                * convert_to_mm[density_obs.depth_top[1]]
                * depth_scale
            )
            density_thickness_mm = (
                density_obs.thickness[0]
                * convert_to_mm[density_obs.thickness[1]]
                * depth_scale
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
