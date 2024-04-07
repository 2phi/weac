"""
WEak Layer AntiCrack nucleation model.

Implementation of closed-form analytical models for the analysis of
dry-snow slab avalanche release.
"""

# Module imports
from weac.layered import Layered
from weac.inverse import Inverse
from weac import plot

# Version
__version__ = '2.5.2'

# Public names
__all__ = [
    'Layered',
    'Inverse',
    'plot'
]
