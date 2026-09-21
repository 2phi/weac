"""Extract-and-plot helper for :class:`~weac.parser.smp_parser.SMPParser`."""

from __future__ import annotations

from typing import Literal

from matplotlib.figure import Figure

from weac.parser.plotting.force_penetration_layers import plot_force_penetration_layers
from weac.parser.smp_parser import DensityMethod, SMPParser


def plot_smp_penetration_parser(
    parser: SMPParser,
    *,
    method: Literal["bin", "gradient"] = "bin",
    slope_angle_deg: float = 0.0,
    density_method: DensityMethod | None = None,
    layer_thickness_mm: float | None = None,
    gradient_threshold: float = 8.0,
    **plot_kwargs,
) -> Figure:
    """Extract from an SMP parser and plot the penetration-resistance debug view."""
    profile = parser.extract_profile(density_method=density_method)
    layers, _ = parser.extract_layers(
        slope_angle_deg,
        method=method,
        density_method=density_method,
        layer_thickness_mm=layer_thickness_mm,
        gradient_threshold=gradient_threshold,
    )
    plot_kwargs.setdefault("layer_label", method)
    plot_kwargs.setdefault("slope_angle_deg", slope_angle_deg)
    return plot_force_penetration_layers(profile, layers, **plot_kwargs)
