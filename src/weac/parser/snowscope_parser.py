"""
Parse SnowScope CSV profiles into a force–density signal for WEAC.

Density is derived from the SnowScope hardness/force signal ``F`` (kPa)
with a semilog model ``D = a ln(F) + b`` (kg/m³):

- French Alps SP2/SMP, Hagenmüller 2018 (``HAGENMULLER2018``, default):
  ``a = 71.4``, ``b = 21.5`` (published as ``ρ = 21.5 + 71.4 log(σ)``)

Pass ``semilog_slope`` / ``semilog_intercept`` on :class:`SnowScopeParser` or
:meth:`SnowScopeParser.extract_profile` to use custom ``a`` and ``b`` instead
of a preset (label ``CUSTOM`` on the returned profile).
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from weac.components import Layer
from weac.parser.utils import (
    bin_profile_to_layers,
    gradient_profile_to_layers,
    plumb_to_slope_normal,
)

logger = logging.getLogger(__name__)

DensityMethod = Literal["HAGENMULLER2018"]
ProfileDensityLabel = DensityMethod | Literal["CUSTOM"]

# Semilog coefficients (slope a, intercept b) for D = a ln(F) + b.
_DENSITY_PARAMS: dict[DensityMethod, tuple[float, float]] = {
    "HAGENMULLER2018": (71.4, 21.5),
}

_MIN_DENSITY_KG_M3 = 1.0


def _validate_semilog_pair(
    semilog_slope: float | None,
    semilog_intercept: float | None,
    *,
    context: str,
) -> None:
    if (semilog_slope is None) ^ (semilog_intercept is None):
        raise ValueError(
            f"{context}: semilog_slope and semilog_intercept must both be "
            "set or both omitted"
        )


def _resolve_semilog_coefficients(
    *,
    density_method: DensityMethod | None,
    semilog_slope: float | None,
    semilog_intercept: float | None,
    default_density_method: DensityMethod,
    default_semilog_slope: float | None,
    default_semilog_intercept: float | None,
) -> tuple[float, float, ProfileDensityLabel]:
    """Resolve ``D = a ln(F) + b`` coefficients and a label for metadata."""
    _validate_semilog_pair(semilog_slope, semilog_intercept, context="extract_profile")
    if semilog_slope is not None:
        return semilog_slope, semilog_intercept, "CUSTOM"
    if density_method is not None:
        slope, intercept = _DENSITY_PARAMS[density_method]
        return slope, intercept, density_method
    if default_semilog_slope is not None:
        return default_semilog_slope, default_semilog_intercept, "CUSTOM"
    slope, intercept = _DENSITY_PARAMS[default_density_method]
    return slope, intercept, default_density_method


def _read_snowscope_csv(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load depth and hardness (kPa) from a SnowScope export CSV.

    Exports prefix ``GENERAL INFO`` / ``SCOPE PROFILE`` headers; the numeric
    block starts at the first line whose first column name contains ``depth``.
    Extra trailing columns (often ``null``) are ignored.
    """
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Exports prefix pit metadata; the numeric block starts at the ``depth …`` header row.
    data_start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lower().lstrip().startswith("depth")
        ),
        None,
    )
    if data_start is None:
        raise ValueError(f"Could not find depth/hardness table in {path}")

    reader = csv.reader(io.StringIO("\n".join(lines[data_start:])))
    header = [col.strip().lower() for col in next(reader, [])]
    if not header:
        raise ValueError(f"SnowScope file has no depth/hardness table: {path}")

    depth_idx = next((i for i, col in enumerate(header) if "depth" in col), 0)
    force_idx = next(
        (i for i, col in enumerate(header) if "hardness" in col or "force" in col),
        1,
    )
    need_cols = max(depth_idx, force_idx) + 1

    depths: list[float] = []
    forces: list[float] = []
    # Keep only rows with parseable depth and force; trailing ``null`` columns are ignored.
    for row in reader:
        if len(row) < need_cols:
            continue
        try:
            depths.append(float(row[depth_idx]))
            forces.append(float(row[force_idx]))
        except ValueError:
            continue

    if not depths:
        raise ValueError(f"No numeric depth/hardness samples in {path}")

    depth_mm = np.asarray(depths, dtype=np.float64)
    penetration_resistance_kPa = np.asarray(forces, dtype=np.float64)
    logger.info(
        "Read SnowScope profile %s: %d samples, %.0f–%.0f mm, "
        "penetration resistance %.2f–%.2f kPa",
        path.name,
        len(depth_mm),
        float(depth_mm[0]),
        float(depth_mm[-1]),
        float(np.min(penetration_resistance_kPa)),
        float(np.max(penetration_resistance_kPa)),
    )
    return depth_mm, penetration_resistance_kPa


@dataclass
class SnowScopeProfile:
    """Aligned SnowScope penetration resistance and derived density."""

    depth_mm: np.ndarray
    penetration_resistance_kPa: np.ndarray
    density_kg_m3: np.ndarray
    density_method: ProfileDensityLabel


class SnowScopeParser:
    """Parser for SnowScope CSV files."""

    def __init__(
        self,
        file_path: str,
        *,
        density_method: DensityMethod = "HAGENMULLER2018",
        semilog_slope: float | None = None,
        semilog_intercept: float | None = None,
    ):
        _validate_semilog_pair(
            semilog_slope, semilog_intercept, context="SnowScopeParser"
        )
        self.file_path = file_path
        self.density_method = density_method
        self._semilog_slope = semilog_slope
        self._semilog_intercept = semilog_intercept
        self.depth_mm, self.penetration_resistance_kPa = _read_snowscope_csv(file_path)
        # Semilog density needs F > 0; bad source rows must not become NaN in layers.
        invalid = np.isfinite(self.penetration_resistance_kPa) & (
            self.penetration_resistance_kPa <= 0
        )
        if np.any(invalid):
            n = int(np.count_nonzero(invalid))
            raise ValueError(
                f"SnowScope file {Path(file_path).name} has {n} sample(s) with "
                "non-positive hardness (kPa); hardness must be > 0"
            )
        if semilog_slope is not None:
            logger.info(
                "Loaded SnowScope profile %s; custom semilog a=%.4g b=%.4g",
                Path(file_path).name,
                semilog_slope,
                semilog_intercept,
            )
        else:
            logger.info(
                "Loaded SnowScope profile %s; density method %s",
                Path(file_path).name,
                self.density_method,
            )

    def extract_profile(
        self,
        *,
        density_method: DensityMethod | None = None,
        semilog_slope: float | None = None,
        semilog_intercept: float | None = None,
    ) -> SnowScopeProfile:
        """Return depth, penetration resistance, and density for one parameterization.

        Converts penetration resistance (kPa) to density (kg/m³) via the
        semilog model ``D = a ln(F) + b``, clipping sub-1 kg/m³ values.

        Coefficients are chosen in order: call-level ``semilog_slope`` /
        ``semilog_intercept``, then call-level ``density_method``, then
        constructor custom semilog (if set), else the constructor
        ``density_method`` preset.
        """
        slope, intercept, label = _resolve_semilog_coefficients(
            density_method=density_method,
            semilog_slope=semilog_slope,
            semilog_intercept=semilog_intercept,
            default_density_method=self.density_method,
            default_semilog_slope=self._semilog_slope,
            default_semilog_intercept=self._semilog_intercept,
        )
        resistance = self.penetration_resistance_kPa
        density = np.full(resistance.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(resistance) & (resistance > 0)
        density[valid] = slope * np.log(resistance[valid]) + intercept
        clip = valid & np.isfinite(density) & (density < _MIN_DENSITY_KG_M3)
        if np.any(clip):
            logger.warning(
                "Clipping %d non-positive %s densities to %.1f kg/m³",
                int(np.count_nonzero(clip)),
                label,
                _MIN_DENSITY_KG_M3,
            )
            density[clip] = _MIN_DENSITY_KG_M3
        return SnowScopeProfile(
            depth_mm=self.depth_mm,
            penetration_resistance_kPa=resistance,
            density_kg_m3=density,
            density_method=label,
        )

    def extract_layers(
        self,
        slope_angle_deg: float = 0.0,
        *,
        method: Literal["bin", "gradient"] = "bin",
        density_method: DensityMethod | None = None,
        semilog_slope: float | None = None,
        semilog_intercept: float | None = None,
        layer_thickness_mm: float | None = None,
        gradient_threshold: float = 12.0,
    ) -> tuple[list[Layer], list[str]]:
        """Segment the SnowScope density profile into WEAC slab layers (top-down).

        The SnowScope probe descends along the global vertical, so ``depth_mm``
        is a plumb depth while WEAC ``Layer.h`` is slope-normal. As in the SMP
        parser, thicknesses are converted plumb -> slope-normal by ``cos(phi)``
        via :func:`plumb_to_slope_normal`. The SnowScope file
        records no slope, so ``slope_angle_deg`` defaults to ``0`` (no scaling);
        scale only when the angle is known (pass it explicitly).

        Args:
            slope_angle_deg: Slope angle [deg from horizontal]. ``0`` leaves
                ``h`` as recorded; non-zero scales ``h`` by ``cos(phi)``.
            method: Layering mode. ``"bin"`` groups at a fixed thickness (or the
                native spacing); ``"gradient"`` cuts where the density gradient
                over a fixed 2.5 mm span exceeds ``gradient_threshold``.
            density_method: Optional preset override for this call only.
            semilog_slope: Optional semilog slope ``a`` in ``D = a ln(F) + b``
                for this call; must be paired with ``semilog_intercept``.
            semilog_intercept: Optional semilog intercept ``b``; must be paired
                with ``semilog_slope``. Takes precedence over ``density_method``.
            layer_thickness_mm: Bin mode only; ``None`` groups at native sample
                spacing with a 1 mm floor, a value groups samples into ~that
                thickness. Ignored in gradient mode.
            gradient_threshold: Gradient mode only; cut threshold ``T`` in
                kg/m^3/mm over the 2.5 mm span. Default ``12``.

        Returns:
            ``(layers, density_methods)`` with layers ordered surface -> ground
            and ``density_methods = [parameterization] * len(layers)``.
        """
        profile = self.extract_profile(
            density_method=density_method,
            semilog_slope=semilog_slope,
            semilog_intercept=semilog_intercept,
        )
        phi_deg = float(slope_angle_deg)
        depth_scale = plumb_to_slope_normal(phi_deg) if phi_deg != 0.0 else 1.0
        if method == "bin":
            layers = bin_profile_to_layers(
                profile.depth_mm,
                profile.density_kg_m3,
                layer_thickness_mm=layer_thickness_mm,
                depth_scale=depth_scale,
            )
        elif method == "gradient":
            layers = gradient_profile_to_layers(
                profile.depth_mm,
                profile.density_kg_m3,
                threshold_kg_m3_per_mm=gradient_threshold,
                depth_scale=depth_scale,
            )
        else:
            raise ValueError(f'method must be "bin" or "gradient", got {method!r}')
        density_methods = [profile.density_method] * len(layers)
        return layers, density_methods
