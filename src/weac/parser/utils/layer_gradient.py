"""Segment a windowed density profile into WEAC slab layers by gradient.

Companion to :mod:`weac.parser.utils.layer_binning`. Instead of grouping at a
fixed thickness, this cuts the depth-ordered (surface -> ground) density series
wherever the density gradient over a **fixed 2.5 mm span** exceeds a threshold,
so ramps become a staircase of >= 1 mm layers while quiet stretches collapse
into one homogeneous layer.

Why a fixed span
----------------
The native Loewe hop differs between parameterizations (CR2020 / P2015 / King),
so an adjacent-sample slope is not comparable across methods. Measuring
``|drho/dz|`` over a constant 2.5 mm baseline (the P2015 window) makes a single
threshold ``T`` meaningful for every parameterization.

Segmentation model
-------------------
Each sample owns a "cell" (spacing to the next sample, last repeated), exactly
as in the binning helper, so ``sum(layer.h)`` equals the summed cell thickness
(times ``depth_scale``). A sample whose fixed-span gradient is ``> T`` is a cut:
it stands as its own layer (subject to the 1 mm floor). Consecutive ``<= T``
samples merge into one layer until the next cut. Thickness is ``depth_scale``
scaled; density is the cell-thickness-weighted mean over the run.
"""

from __future__ import annotations

import numpy as np

from weac.components import Layer
from weac.parser.utils.layer_binning import _EPS, _cell_thicknesses


def _fixed_span_gradient(
    depth_mm: np.ndarray,
    density_kg_m3: np.ndarray,
    span_mm: float,
) -> np.ndarray:
    """``|drho/dz|`` over a fixed ``span_mm`` look-ahead for each sample.

    For sample ``i`` the gradient is measured to the first sample at or beyond
    ``depth[i] + span_mm``. Near the ground the remaining pack can be thinner
    than ``span_mm``; there the last sample is used, so the span shrinks to the
    remaining depth (spec: "if remaining depth < span, use that remaining span").
    The final sample has no look-ahead span and is assigned ``0`` (never a cut).
    """
    n = depth_mm.size
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        target = depth_mm[i] + span_mm
        # First index at/after the target depth; clamp to the last sample so a
        # short pack tail falls back to the remaining span.
        j = int(np.searchsorted(depth_mm, target, side="left"))
        if j >= n:
            j = n - 1
        span = depth_mm[j] - depth_mm[i]
        if span > 0:
            grad[i] = abs(density_kg_m3[j] - density_kg_m3[i]) / span
    return grad


def gradient_profile_to_layers(
    depth_mm: np.ndarray,
    density_kg_m3: np.ndarray,
    *,
    threshold_kg_m3_per_mm: float = 12.0,
    span_mm: float = 2.5,
    min_thickness_mm: float = 1.0,
    depth_scale: float = 1.0,
) -> list[Layer]:
    """Segment a density profile into WEAC slab layers by gradient (top-down).

    Args:
        depth_mm: Depth-ordered (surface -> ground) sample positions [mm].
        density_kg_m3: Density at each sample [kg m^-3]; same shape as
            ``depth_mm``.
        threshold_kg_m3_per_mm: Cut threshold ``T``. Samples whose fixed-span
            ``|drho/dz|`` exceeds ``T`` start their own layer; ``<= T`` samples
            merge. Default ``12`` (== 30 kg/m^3 across the 2.5 mm span).
        span_mm: Fixed baseline for the gradient [mm]; default ``2.5``.
        min_thickness_mm: Minimum layer thickness [mm]. Sub-floor segments are
            merged forward; a trailing remainder below the floor is folded into
            the previous layer.
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
    grad = _fixed_span_gradient(depth, density, span_mm)
    is_cut = grad > threshold_kg_m3_per_mm  # strict: gradient > T stays separate

    # Ideal (pre-floor) layer boundaries: close after a cut sample (it owns its
    # layer) and just before the next cut (ending the preceding merged run). The
    # last sample always closes the pack.
    boundary_after = np.zeros(n, dtype=bool)
    boundary_after[is_cut] = True
    boundary_after[:-1] |= is_cut[1:]
    boundary_after[-1] = True

    # Greedily grow runs; a boundary only closes a layer once the 1 mm floor is
    # met, so sub-floor segments (e.g. a fine cut staircase) merge forward.
    runs: list[tuple[int, int, float]] = []  # (start, stop, thickness)
    start = 0
    acc = 0.0
    for i in range(n):
        acc += cell[i]
        if boundary_after[i] and acc >= min_thickness_mm - _EPS:
            runs.append((start, i + 1, acc))
            start = i + 1
            acc = 0.0

    # Trailing remainder: keep if it clears the floor (or is the only run), else
    # fold it into the previous layer (same rule as the binning helper).
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
