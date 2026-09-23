"""
Parse SnowMicroPen ``.PNT`` files into WEAC layers via snowmicropyn.

Density and SSA are derived with snowmicropyn's Löwe 2012 shot-noise
model plus one of the shipped parameterizations (see
https://snowmicropyn.readthedocs.io/en/latest/):

- Proksch, 2015 (``P2015``)
- Calonne & Richter, 2020 (``CR2020``)
- King, 2020a / 2020b (``K2020a``, ``K2020b``)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import snowmicropyn as smp
from snowmicropyn import loewe2012

from weac.components import Layer
from weac.parser.utils import (
    bin_profile_to_layers,
    detect_knappe_surface,
    gradient_profile_to_layers,
)

logger = logging.getLogger(__name__)

# Shipped shortnames; runtime lookup uses snowmicropyn's Parameterizations registry.
DensityMethod = Literal["P2015", "CR2020", "K2020a", "K2020b"]
# ``snowmicropyn`` is Profile.detect_surface(). ``knappe`` is Knappe surface detection.
# ``manual`` uses the ``surface_mm`` argument.
SurfaceMethod = Literal["snowmicropyn", "knappe", "manual"]
SMP_PARAMETERIZATIONS = smp.derivatives.parameterizations

# Standard SnowMicroPen tip diameter used to convert force (N) to kPa.
SMP_DIAMETER_MM = 5.0


def _force_n_to_kpa(
    force_N: np.ndarray,
    diameter_mm: float = SMP_DIAMETER_MM,
) -> np.ndarray:
    """Convert SMP force in newtons to penetration resistance in kPa."""
    radius_mm = diameter_mm / 2.0
    area_mm2 = np.pi * radius_mm**2
    return force_N / area_mm2 * 1000.0


@dataclass
class SMPProfile:
    """Windowed SMP penetration resistance, density, and SSA cropped to the snowpack."""

    coordinates: tuple[float, float] | None
    depth_mm: np.ndarray
    penetration_resistance_kPa: np.ndarray
    density_kg_m3: np.ndarray
    ssa_m2_kg: np.ndarray
    density_method: DensityMethod


class SMPParser:
    """Parser for SMP ``.PNT`` files using snowmicropyn."""

    file_path: str
    density_method: DensityMethod
    surface_method: SurfaceMethod
    loaded_profile: smp.Profile

    def __init__(
        self,
        file_path: str,
        *,
        density_method: DensityMethod = "P2015",
        surface_mm: float | None = None,
        surface_method: SurfaceMethod = "snowmicropyn",
        apply_drift_correction: bool = False,
    ):
        """Load a ``.PNT`` file and mark surface and ground.

        Args:
            file_path: Path to the SMP ``.PNT`` file.
            density_method: Default parameterization for :meth:`extract_profile`.
            surface_mm: Manual surface marker, mm along the probe. When set,
                the surface is placed here and ``surface_method`` is recorded
                as ``"manual"``. Required when ``surface_method`` is ``"manual"``.
            surface_method: ``"snowmicropyn"`` calls
                :meth:`snowmicropyn.Profile.detect_surface`. ``"knappe"`` uses
                :func:`~weac.parser.utils.detect_knappe_surface`. ``"manual"``
                places the surface at ``surface_mm``.
            apply_drift_correction: Subtract snowmicropyn's force offset after
                the surface marker is set.
        """
        self.file_path = file_path
        self.density_method = density_method
        self.surface_method: SurfaceMethod = (
            "manual" if surface_mm is not None else surface_method
        )

        loaded_profile = smp.Profile.load(file_path)

        if self.surface_method == "manual":
            if surface_mm is None:
                raise ValueError(
                    'surface_mm is required when surface_method is "manual".'
                )
            profile_length = loaded_profile.recording_length
            if not 0.0 <= surface_mm <= profile_length:
                raise ValueError(
                    f"surface_mm={surface_mm} is outside the profile range "
                    f"[0, {profile_length}]."
                )
            loaded_profile.set_marker("surface", float(surface_mm))
        elif self.surface_method == "knappe":
            loaded_profile.set_marker(
                "surface", detect_knappe_surface(loaded_profile.samples)
            )
        elif self.surface_method == "snowmicropyn":
            loaded_profile.detect_surface()
        else:
            raise ValueError(
                'surface_method must be "snowmicropyn", "knappe", or "manual", '
                f"got {surface_method!r}"
            )

        if apply_drift_correction:
            # ``subtract_force_offset`` mutates the force signal in place and
            # emits stray prints; rely on the once-only
            # construction flow to avoid double subtraction.
            loaded_profile.subtract_force_offset()

        loaded_profile.detect_ground()

        self.loaded_profile = loaded_profile
        logger.info(
            "Loaded SMP profile %s; density method %s; surface %.3f mm (%s)",
            Path(file_path).name,
            self.density_method,
            loaded_profile.surface,
            self.surface_method,
        )

    def extract_profile(
        self,
        *,
        density_method: DensityMethod | None = None,
    ) -> SMPProfile:
        """Return the windowed penetration resistance/density/SSA profile for one parameterization.

        Runs the Löwe 2012 shot-noise model plus the selected parameterization
        on the cropped snowpack and returns a fresh :class:`SMPProfile` each
        call (no caching). ``density_method=None`` uses the constructor default;
        surface and ground markers are set once at construction and reused here.
        """
        method = density_method if density_method is not None else self.density_method
        param = SMP_PARAMETERIZATIONS[method]

        samples = self.loaded_profile.samples_within_snowpack()
        loewe_df = loewe2012.calc(samples, param.window_size, param.overlap)
        derived = param.calc_from_loewe2012(loewe_df)
        smp_res = loewe_df.merge(derived)
        return SMPProfile(
            coordinates=self.loaded_profile.coordinates,
            depth_mm=smp_res["distance"].to_numpy(dtype=np.float64),
            penetration_resistance_kPa=_force_n_to_kpa(
                smp_res["force_median"].to_numpy(dtype=np.float64)
            ),
            density_kg_m3=smp_res[f"{param.shortname}_density"].to_numpy(
                dtype=np.float64
            ),
            ssa_m2_kg=smp_res[f"{param.shortname}_ssa"].to_numpy(dtype=np.float64),
            density_method=method,
        )

    def extract_layers(
        self,
        *,
        method: Literal["bin", "gradient"] = "bin",
        density_method: DensityMethod | None = None,
        layer_thickness_mm: float | None = None,
        gradient_threshold: float = 12.0,
    ) -> tuple[list[Layer], list[str]]:
        """Segment the SMP density profile into WEAC slab layers (top-down).

        The SnowMicroPen is driven normal to the slope, so ``depth_mm`` is
        already a slope-normal depth and is used directly as WEAC ``Layer.h``.
        Plumb profiles (SnowScope, SnowPilot) still scale by ``cos(phi)``; SMP
        does not.

        Args:
            method: Layering mode. ``"bin"`` groups at a fixed thickness (or the
                native spacing); ``"gradient"`` cuts where the density gradient
                over a fixed 2.5 mm span exceeds ``gradient_threshold``.
            density_method: Optional parameterization override for this call
                only (constructor default is unchanged).
            layer_thickness_mm: Bin mode only; ``None`` groups at native Loewe
                spacing with a 1 mm floor, a value groups samples into ~that
                thickness. Ignored in gradient mode.
            gradient_threshold: Gradient mode only; cut threshold ``T`` in
                kg/m^3/mm over the 2.5 mm span. Default ``12``.

        Returns:
            ``(layers, density_methods)`` with layers ordered surface -> ground
            and ``density_methods = [parameterization] * len(layers)``.
        """
        profile = self.extract_profile(density_method=density_method)
        dens = profile.density_kg_m3
        if not (np.all(np.isfinite(dens)) and np.all(dens > 0)):
            raise ValueError(
                f"SMP profile {Path(self.file_path).name}: "
                f"{profile.density_method} density not strictly positive "
                f"({float(np.nanmin(dens)):.4g}–{float(np.nanmax(dens)):.4g} kg/m³)"
            )
        if method == "bin":
            layers = bin_profile_to_layers(
                profile.depth_mm,
                profile.density_kg_m3,
                layer_thickness_mm=layer_thickness_mm,
            )
        elif method == "gradient":
            layers = gradient_profile_to_layers(
                profile.depth_mm,
                profile.density_kg_m3,
                threshold_kg_m3_per_mm=gradient_threshold,
            )
        else:
            raise ValueError(f'method must be "bin" or "gradient", got {method!r}')
        density_methods = [profile.density_method] * len(layers)
        return layers, density_methods
