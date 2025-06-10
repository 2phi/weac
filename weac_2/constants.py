"""
Constants for the WEAC simulation.
"""
from typing import Final

G_MM_S2: Final[float] = 9810.0        # gravitational acceleration (mm s⁻²)
NU: Final[float] = 0.25               # Global Poisson's ratio
K_SHEAR: Final[float] = 5.0 / 6.0     # Mindlin shear-correction factor (slabs)
ROMBERG_TOL:       float = 1e-3       # Romberg integration tolerance
LSKI_MM:           float = 1000.0     # Effective out-of-plane length of skis (mm)

RHO0: Final[float] = 917.0            # Density of ice (kg/m^3)
CB0: Final[float] = 6.5               # Multiplicative constant of Young modulus parametrization according to Bergfeld et al. (2023)
CB1: Final[float] = 4.4               # Exponent of Young modulus parameterization according to Bergfeld et al. (2023)
CG0: Final[float] = 6.0               # Multiplicative constant of Young modulus parametrization according to Gerling et al. (2017)
CG1: Final[float] = 4.5               # Exponent of Young modulus parameterization according to Gerling et al. (2017)

