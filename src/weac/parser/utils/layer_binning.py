"""Group a windowed density profile into WEAC slab layers.

Shared binning helper for the parser package (used by the SMP ``bin`` mode and,
later, SnowScope / gradient modes). The input is a depth-ordered
(surface -> ground) density series; the output is :class:`~weac.components.Layer`
objects in that same top-down order.

Binning model
-------------
Each sample owns a "cell" whose thickness is the spacing to the next sample (the
native Loewe hop for SMP); the final sample repeats the last spacing so every
sample contributes a cell. Layers are contiguous runs of cells, so
``sum(layer.h)`` equals the summed cell thickness (times ``depth_scale``).

Thickness (``Layer.h``) is scaled by ``depth_scale`` to convert a plumb probe
depth to WEAC's slope-normal thickness; density (``Layer.rho``) is the
cell-thickness-weighted mean over the run.
"""

from __future__ import annotations

import numpy as np

from weac.components import Layer

# Tolerance for float thickness comparisons (mm).
_EPS = 1e-9


def _cell_thicknesses(depth_mm: np.ndarray, min_thickness_mm: float) -> np.ndarray:
    """Per-sample cell thickness: spacing to the next sample, last repeated."""
    n = depth_mm.size
    if n == 1:
        # A lone sample has no measurable spacing; fall back to the floor so it
        # still yields a valid (positive-thickness) layer.
        return np.array([float(min_thickness_mm)], dtype=np.float64)
    diff = np.diff(depth_mm)
    cell = np.empty(n, dtype=np.float64)
    cell[:-1] = diff
    cell[-1] = diff[-1]
    return cell


def bin_profile_to_layers(
    depth_mm: np.ndarray,
    density_kg_m3: np.ndarray,
    *,
    layer_thickness_mm: float | None = None,
    min_thickness_mm: float = 1.0,
    depth_scale: float = 1.0,
) -> list[Layer]:
    """Bin a density profile into WEAC slab layers (top-down).

    Args:
        depth_mm: Depth-ordered (surface -> ground) sample positions [mm].
        density_kg_m3: Density at each sample [kg m^-3]; same shape as
            ``depth_mm``.
        layer_thickness_mm: Target layer thickness. ``None`` groups at the
            native sample spacing (each cell is its own layer) subject to the
            ``min_thickness_mm`` floor; a value groups adjacent cells until each
            run reaches roughly that thickness.
        min_thickness_mm: Minimum layer thickness [mm]. Adjacent cells are
            merged until a run reaches this floor; a trailing remainder below
            the floor is folded into the previous layer.
        depth_scale: Plumb -> slope-normal factor applied to every ``h``
            (``cos(phi)``); ``1.0`` leaves thicknesses unscaled.

    Returns:
        Layers ordered surface -> ground. ``sum(layer.h)`` equals the summed
        cell thickness times ``depth_scale``.
    """
    depth = np.asarray(depth_mm, dtype=np.float64)
    density = np.asarray(density_kg_m3, dtype=np.float64)
    if depth.shape != density.shape:
        raise ValueError("depth_mm and density_kg_m3 must have the same shape")
    n = depth.size
    if n == 0:
        return []

    cell = _cell_thicknesses(depth, min_thickness_mm)

    # Run target: explicit bin size when given, else the 1 mm floor (native mode
    # where a native step >= 1 mm already closes each sample individually).
    target = (
        float(layer_thickness_mm)
        if layer_thickness_mm is not None
        else float(min_thickness_mm)
    )

    # Greedily accumulate cells until a run reaches `target`.
    runs: list[tuple[int, int, float]] = []  # (start, stop, thickness)
    start = 0
    acc = 0.0
    for i in range(n):
        acc += cell[i]
        if acc >= target - _EPS:
            runs.append((start, i + 1, acc))
            start = i + 1
            acc = 0.0

    # Trailing remainder: keep as its own layer if it clears the floor (or is the
    # only run), otherwise fold it into the previous layer.
    if start < n:
        if not runs or acc >= min_thickness_mm - _EPS:
            runs.append((start, n, acc))
        else:
            prev_start, _, prev_thickness = runs[-1]
            runs[-1] = (prev_start, n, prev_thickness + acc)

    layers: list[Layer] = []
    for run_start, run_stop, thickness in runs:
        weights = cell[run_start:run_stop]
        weight_sum = float(np.sum(weights))
        if weight_sum > 0:
            rho = float(np.sum(weights * density[run_start:run_stop]) / weight_sum)
        else:  # pragma: no cover - only if all cells are zero-thickness
            rho = float(np.mean(density[run_start:run_stop]))
        layers.append(Layer(rho=rho, h=float(thickness * depth_scale)))
    return layers
