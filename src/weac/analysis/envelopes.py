"""
Stress and fracture-toughness envelope evaluations.
"""

from __future__ import annotations

import numpy as np

from weac.components import CriteriaConfig, WeakLayer
from weac.constants import RHO_ICE

__all__ = [
    "fracture_toughness_envelope",
    "stress_envelope",
]


def fracture_toughness_envelope(
    criteria_config: CriteriaConfig,
    G_I: float | np.ndarray,
    G_II: float | np.ndarray,
    weak_layer: WeakLayer,
) -> float | np.ndarray:
    """
    Evaluate the fracture toughness criterion for Mode I / Mode II ERRs.

    The criterion is defined as:
        g_delta = (|G_I| / G_Ic)^gn + (|G_II| / G_IIc)^gm

    A value of 1 indicates the boundary of the fracture toughness envelope.
    """
    g_delta = (np.abs(G_I) / weak_layer.G_Ic) ** criteria_config.gn + (
        np.abs(G_II) / weak_layer.G_IIc
    ) ** criteria_config.gm
    return g_delta


def stress_envelope(
    criteria_config: CriteriaConfig,
    sigma: float | np.ndarray,
    tau: float | np.ndarray,
    weak_layer: WeakLayer,
    method: str | None = None,
) -> np.ndarray:
    """
    Evaluate the stress envelope for given stress components.

    Weak Layer failure is defined as the stress envelope crossing 1.
    """
    sigma = np.abs(np.asarray(sigma))
    tau = np.abs(np.asarray(tau))
    results = np.zeros_like(sigma)

    envelope_method = (
        method if method is not None else criteria_config.stress_envelope_method
    )
    density = weak_layer.rho
    sigma_c = weak_layer.sigma_c
    tau_c = weak_layer.tau_c
    fn = criteria_config.fn
    fm = criteria_config.fm
    order_of_magnitude = criteria_config.order_of_magnitude
    scaling_factor = criteria_config.scaling_factor

    def mede_common_calculations(sigma, tau, p0, tau_T, p_T):
        results_local = np.zeros_like(sigma)
        in_first_range = (sigma >= (p_T - p0)) & (sigma <= p_T)
        in_second_range = sigma > p_T
        results_local[in_first_range] = (
            -tau[in_first_range] * (p0 / (tau_T * p_T))
            + sigma[in_first_range] * (1 / p_T)
            + p0 / p_T
        )
        results_local[in_second_range] = (tau[in_second_range] ** 2) + (
            (tau_T / p0) ** 2
        ) * ((sigma[in_second_range] - p_T) ** 2)
        return results_local

    if envelope_method == "adam_unpublished":
        if scaling_factor > 1:
            order_of_magnitude = 0.7
        scaling_factor = max(scaling_factor, 0.55)
        scaled_sigma_c = sigma_c * (scaling_factor**order_of_magnitude)
        scaled_tau_c = tau_c * (scaling_factor**order_of_magnitude)
        return (sigma / scaled_sigma_c) ** fn + (tau / scaled_tau_c) ** fm

    if envelope_method == "schottner":
        sigma_y = 2000
        scaled_sigma_c = sigma_y * 13 * (density / RHO_ICE) ** order_of_magnitude
        scaled_tau_c = tau_c * (scaled_sigma_c / sigma_c)
        return (sigma / scaled_sigma_c) ** fn + (tau / scaled_tau_c) ** fm

    if envelope_method == "mede_s-RG1":
        p0, tau_T, p_T = 7.00, 3.53, 1.49
        results = mede_common_calculations(sigma, tau, p0, tau_T, p_T)
        return results

    if envelope_method == "mede_s-RG2":
        p0, tau_T, p_T = 2.33, 1.22, 0.19
        results = mede_common_calculations(sigma, tau, p0, tau_T, p_T)
        return results

    if envelope_method == "mede_s-FCDH":
        p0, tau_T, p_T = 1.45, 0.61, 0.17
        results = mede_common_calculations(sigma, tau, p0, tau_T, p_T)
        return results

    raise ValueError(f"Invalid envelope type: {envelope_method}")
