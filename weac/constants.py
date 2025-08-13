"""
Constants for the WEAC simulation.
"""

from typing import Final

G_MM_S2: Final[float] = 9810.0  # gravitational acceleration (mm s⁻²)
NU: Final[float] = 0.25  # Global Poisson's ratio
SHEAR_CORRECTION_FACTOR: Final[float] = 5.0 / 6.0  # Shear-correction factor (slabs)
STIFFNESS_COLLAPSE_FACTOR: Final[float] = (
    1000.0  # Stiffness ratio between collapsed and uncollapsed weak layer.
)
ROMBERG_TOL: Final[float] = 1e-3  # Romberg integration tolerance
LSKI_MM: Final[float] = 1000.0  # Effective out-of-plane length of skis (mm)
EPS: Final[float] = 1e-9  # Global numeric tolerance for float comparisons

RHO_ICE: Final[float] = 916.7  # Density of ice (kg/m^3)
CB0: Final[float] = (
    6.5  # Multiplicative constant of Young modulus parametrization according to Bergfeld et al. (2023)
)
CB1: Final[float] = (
    4.4  # Exponent of Young modulus parameterization according to Bergfeld et al. (2023)
)
CG0: Final[float] = (
    6.0  # Multiplicative constant of Young modulus parametrization according to Gerling et al. (2017)
)
CG1: Final[float] = (
    4.5  # Exponent of Young modulus parameterization according to Gerling et al. (2017)
)
