"""
WEak Layer AntiCrack nucleation model.

Implementation of closed-form analytical models for the analysis of
dry-snow slab avalanche release.

Classes
-------
Eigensystem()
    Base class for a layered beam on an elastic foundation.
FieldQuantitiesMixin()
    Provides methods for the computation of displacements, stresses,
    strains, and energy release rates from the solution vector.
SolutionMixin()
    Provides methods for the assembly of the system of equations
    and for the computation of the free constants.
AnalysisMixin()
    Provides methods for the analysis of layered slabs on compliant
    elastic foundations.
Layered()
    Layered beam on elastic foundation model application interface.
Inverse()
    Fit the elastic properties of the layers of a snowpack.
"""

# Module imports
from weac.layered import Layered
from weac.inverse import Inverse
from weac.tools import time
from weac import plot

# Version
__version__ = '2.0.0'

# Public names
__all__ = [
    'Layered',
    'Inverse',
    'time',
    'plot'
]
