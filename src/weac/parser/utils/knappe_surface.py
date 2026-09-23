"""Knappe surface detection for SnowMicroPen force profiles.

Gradient detector: a 1 mm moving linear regression, Hanning-smoothed, then a
threshold of 5× the population standard deviation of the quietest 20 mm inside
the first 100 mm (air above the snow). The surface is the first sample after
the top 1000 where the gradient exceeds that threshold and the next 1 mm stays
above it. The fraction of that window still below the threshold must itself be
smaller than the threshold, which rejects isolated noise spikes.
"""

from __future__ import annotations

import numpy as np
from snowmicropyn.tools import smooth

# Fixed SMP sample spacing (mm).
_SMP_RESOLUTION_MM = 0.00413223123177886
_GRADIENT_WINDOW_MM = 1.0
_GRADIENT_SMOOTH_SAMPLES = int(_GRADIENT_WINDOW_MM / _SMP_RESOLUTION_MM)  # 242
_AIR_REGION_MM = 100.0
_AIR_STABLE_WINDOW_MM = 20.0
_AIR_STABLE_WINDOW_SAMPLES = int(_AIR_STABLE_WINDOW_MM / _SMP_RESOLUTION_MM)  # 4840
_SURFACE_SKIP_SAMPLES = 1000  # skip the top ~4 mm of noise
_SURFACE_THRESHOLD_FACTOR = 5.0


def _moving_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    window_mm: float = _GRADIENT_WINDOW_MM,
) -> np.ndarray:
    """Slope of ``y`` vs ``x`` in a centred moving window, NaN outside it."""
    window_size = int(window_mm / _SMP_RESOLUTION_MM)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ones = np.ones(window_size)
    sum_x = np.convolve(x, ones, mode="valid")
    sum_y = np.convolve(y, ones, mode="valid")
    sum_xy = np.convolve(x * y, ones, mode="valid")
    sum_x2 = np.convolve(x * x, ones, mode="valid")
    numerator = window_size * sum_xy - sum_x * sum_y
    denominator = window_size * sum_x2 - sum_x**2
    slope = numerator / denominator
    pad = (x.size - slope.size) // 2
    result = np.full(x.shape, np.nan, dtype=np.float64)
    result[pad : pad + slope.size] = slope
    return result


def _quietest_window_std(values: np.ndarray, window: int) -> float:
    """Population std of the fully finite window with the smallest std.

    Same selection as a sliding ``ndarray.std()`` (ddof 0). Windows that
    contain a NaN are ignored. Returns NaN when none are fully finite.
    """
    count = values.size
    if window <= 1 or count < window:
        return float("nan")
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    csum = np.empty(count + 1, dtype=np.float64)
    csum2 = np.empty(count + 1, dtype=np.float64)
    ccount = np.empty(count + 1, dtype=np.int64)
    csum[0] = 0.0
    csum2[0] = 0.0
    ccount[0] = 0
    csum[1:] = np.cumsum(filled)
    csum2[1:] = np.cumsum(filled * filled)
    ccount[1:] = np.cumsum(finite)
    total = csum[window:] - csum[:-window]
    total2 = csum2[window:] - csum2[:-window]
    n_finite = ccount[window:] - ccount[:-window]
    full = n_finite == window
    if not np.any(full):
        return float("nan")
    mean = np.full(total.shape, np.nan)
    var = np.full(total.shape, np.nan)
    mean[full] = total[full] / window
    var[full] = total2[full] / window - mean[full] ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    return float(std[int(np.nanargmin(std))])


def detect_knappe_surface(samples) -> float:
    """Return the Knappe surface distance in mm along the SMP profile.

    ``samples`` needs ``distance`` and ``force`` columns (snowmicropyn's
    ``Profile.samples``). Falls back to the first sample when the profile is
    shorter than the 1 mm window or no surface rises out of the air noise.
    """
    distance = np.asarray(samples["distance"], dtype=np.float64)
    force = np.asarray(samples["force"], dtype=np.float64)
    if distance.size == 0:
        raise ValueError("SMP profile has no samples to detect a surface in.")
    fallback = float(distance[0])
    window_len = _GRADIENT_SMOOTH_SAMPLES
    if distance.size < window_len:
        return fallback

    grad = _moving_linear_regression(distance, force, window_mm=_GRADIENT_WINDOW_MM)
    grad = smooth(grad, window_len)[: distance.size]

    air = grad[distance <= (distance[0] + _AIR_REGION_MM)]
    air_std = _quietest_window_std(air, _AIR_STABLE_WINDOW_SAMPLES)
    if not np.isfinite(air_std):
        air_std = float(np.nanstd(air))
    if not np.isfinite(air_std) or air_std == 0.0:
        return fallback

    threshold = _SURFACE_THRESHOLD_FACTOR * air_std
    for index in range(_SURFACE_SKIP_SAMPLES, grad.size):
        if not grad[index] > threshold:
            continue
        check_window = grad[index + 1 : index + 1 + window_len]
        if check_window.size == 0:
            continue
        fraction_below = float(np.sum(check_window < threshold) / check_window.size)
        if fraction_below >= threshold:
            continue
        return float(distance[index])
    return fallback
