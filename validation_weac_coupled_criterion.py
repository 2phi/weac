"""
This script demonstrates the basic usage of the WEAC package to run a simulation.
"""

import sys

sys.path.append("examples")

from criterion_check import *

# Define thinner snow profile (standard snow profile A), with higher weak layer Young's Modulus
snow_profile = [
    [350, 120],  # (1) surface layer
    [270, 120],  # (2) 2nd layer
    [180, 120],
]  # (N) last slab layer above weak layer

phi = 30  # Slope angle in degrees
skier_weight = 75  # Skier weight in kg
envelope = "adam_unpublished"
scaling_factor = 1
E = 1  # Elastic modulus in MPa
order_of_magnitude = 1
density = 150  # Weak layer density in kg/m³
t = 30  # Weak layer thickness in mm

(
    result,
    crack_length,
    skier_weight,
    skier,
    C,
    segments,
    x_cm,
    sigma_kPa,
    tau_kPa,
    iteration_count,
    elapsed_times,
    skier_weights,
    crack_lengths,
    self_collapse,
    pure_stress_criteria,
    critical_skier_weight,
    g_delta_last,
    dist_max,
    g_delta_values,
    dist_max_values,
) = check_coupled_criterion_anticrack_nucleation(
    snow_profile=snow_profile,
    phi=phi,
    skier_weight=skier_weight,
    envelope=envelope,
    scaling_factor=scaling_factor,
    E=E,
    order_of_magnitude=order_of_magnitude,
    density=density,
    t=t,
)

# Print the results
print("Algorithm convergence:", result)
print("Anticrack nucleation governed by a pure stress criterion:", pure_stress_criteria)

print("Critical Skier Weight:", skier_weight, "kg")
print("Crack Length:", crack_length, "mm")
print("Fracture toughness envelope function:", g_delta_values[-1])
print("Stress failure envelope function:", dist_max_values[-1])
