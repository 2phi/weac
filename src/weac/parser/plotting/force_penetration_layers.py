"""Shared four-panel penetration-resistance / density / layer plots."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from weac.components import Layer
from weac.parser.utils import plumb_to_slope_normal


class ForcePenetrationProfile(Protocol):
    """Profile from SMP or SnowScope (resistance–depth signal plus derived density)."""

    depth_mm: np.ndarray
    penetration_resistance_kPa: np.ndarray
    density_kg_m3: np.ndarray
    density_method: str


def _layer_staircase(
    layers: list[Layer], surface_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (density, depth) staircase points for a top-down layer list.

    Boundaries accumulate ``Layer.h`` from ``surface_mm`` (the profile's first
    sample depth). Consecutive vertical segments share a boundary depth, so a
    single ``plot`` call renders them as a density-vs-depth staircase. Because
    ``Layer.h`` is slope-normal, the staircase belongs on the slope-normal axis.
    """
    if not layers:
        return np.array([]), np.array([])
    thicknesses = np.array([layer.h for layer in layers], dtype=np.float64)
    densities = np.array([layer.rho for layer in layers], dtype=np.float64)
    tops = surface_mm + np.concatenate(([0.0], np.cumsum(thicknesses)[:-1]))
    bottoms = surface_mm + np.cumsum(thicknesses)
    rho = np.repeat(densities, 2)
    depth = np.empty(rho.size, dtype=np.float64)
    depth[0::2] = tops
    depth[1::2] = bottoms
    return rho, depth


def plot_force_penetration_layers(
    profile: ForcePenetrationProfile,
    layers: list[Layer],
    *,
    slope_angle_deg: float = 0.0,
    title: str | None = None,
    layer_label: str = "layers",
    save_path: str | Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (13.0, 6.5),
) -> Figure:
    """Plot penetration resistance, density, slope-normal density, and layers on a depth axis.

    Args:
        profile: :class:`~weac.parser.smp_parser.SMPProfile` or
            :class:`~weac.parser.snowscope_parser.SnowScopeProfile` (must expose
            ``depth_mm``, ``penetration_resistance_kPa``, ``density_kg_m3``,
            ``density_method``).
        layers: WEAC layers ordered surface -> ground (from ``extract_layers``).
            ``Layer.h`` is slope-normal, so the reconstructed staircase is drawn
            on the slope-normal axis (panels 3-4). Pass the same
            ``slope_angle_deg`` used to extract the layers so the slope-normal
            density panel and the staircase share one axis. The staircase bottom
            overshoots the last sample by one cell because the binning helpers
            repeat the final sample spacing (an intentional part of their cell
            model, see ``layer_binning``).
        slope_angle_deg: Slope angle [deg] used when the layers were extracted;
            sets the ``cos(phi)`` compression of the slope-normal depth axis.
        title: Figure suptitle; defaults to the density method name.
        layer_label: Legend/title label for the segmented panel.
        save_path: If given, save the figure (PNG) to this path.
        show: Call ``plt.show()`` before returning (default ``True``).
        figsize: Figure size in inches.

    Returns:
        The created :class:`~matplotlib.figure.Figure`.
    """
    depth = np.asarray(profile.depth_mm, dtype=np.float64)
    resistance = np.asarray(profile.penetration_resistance_kPa, dtype=np.float64)
    density = np.asarray(profile.density_kg_m3, dtype=np.float64)
    surface_mm = float(depth[0]) if depth.size else 0.0

    phi = float(slope_angle_deg)
    scale = plumb_to_slope_normal(phi) if phi != 0.0 else 1.0
    # Anchor at the surface; compress only the below-surface span (matches how
    # the layer staircase accumulates slope-normal thicknesses from surface_mm).
    depth_sn = surface_mm + (depth - surface_mm) * scale
    rho_step, depth_step = _layer_staircase(layers, surface_mm)

    fig, (ax_resistance, ax_density, ax_density_sn, ax_layers) = plt.subplots(
        1, 4, figsize=figsize, sharey=True
    )

    ax_resistance.plot(resistance, depth, color="#004E8A", lw=0.9)
    ax_resistance.set_xlabel("Penetration resistance [kPa]")
    ax_resistance.set_ylabel("Depth [mm]")
    ax_resistance.set_title("Penetration resistance\n(plumb)")

    ax_density.plot(density, depth, color="#00689D", lw=0.9)
    ax_density.set_xlabel("Density [kg m$^{-3}$]")
    ax_density.set_title("Sample density\n(plumb)")

    ax_density_sn.plot(density, depth_sn, color="#009D81", lw=0.9)
    ax_density_sn.set_xlabel("Density [kg m$^{-3}$]")
    ax_density_sn.set_title(f"Sample density\n(slope-normal, φ={phi:g}°)")

    # Segmented layers (slope-normal) with the slope-normal density overlaid.
    ax_layers.plot(
        density,
        depth_sn,
        color="#B5B5B5",
        lw=0.8,
        alpha=0.7,
        label="sample density",
    )
    if rho_step.size:
        ax_layers.plot(rho_step, depth_step, color="#EC6500", lw=1.6, label=layer_label)
        # Boundary ticks (every other point is a shared top/bottom depth).
        for boundary in depth_step[1::2]:
            ax_layers.axhline(boundary, color="#EC6500", lw=0.3, alpha=0.3)
    ax_layers.set_xlabel("Density [kg m$^{-3}$]")
    ax_layers.set_title(f"Segmented {layer_label}\n(slope-normal, n={len(layers)})")
    ax_layers.legend(loc="lower right", fontsize=8, frameon=False)

    # Depth increases downward: surface at the top. Plumb depth is the deepest.
    if depth.size:
        max_depth = float(np.nanmax(depth))
        if depth_step.size:
            max_depth = max(max_depth, float(np.nanmax(depth_step)))
        ax_resistance.set_ylim(max_depth, surface_mm)
    for ax in (ax_resistance, ax_density, ax_density_sn, ax_layers):
        ax.grid(True, alpha=0.25, lw=0.4)

    fig.suptitle(title or f"Penetration resistance profile ({profile.density_method})")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    return fig
