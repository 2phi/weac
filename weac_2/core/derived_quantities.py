"""
This module defines the derived quantities for the WEAC simulation.
The derived quantities are calculated from the field quantities.
"""

import numpy as np
import logging

from weac_2.core.field_quantities import FieldQuantities
from weac_2.core.eigensystem import SystemProperties

logger = logging.getLogger(__name__)


class DerivedQuantities():
    """
    This class is used to define the derived quantities for the WEAC simulation.
    """
    unknown_constants: np.ndarray
    field_quantities: FieldQuantities
    
    # Derived Quantities
    tau: np.ndarray
    sigma: np.ndarray
    G_I: np.ndarray
    G_II: np.ndarray
    G_total: np.ndarray
    Txx: np.ndarray
    Txz: np.ndarray
    Sxx: np.ndarray
    # etc...
    
    def __init__(self, unknown_constants: np.ndarray, field_quantities: FieldQuantities):
        self.unknown_constants = unknown_constants
        self.field_quantities = field_quantities
        
    def compute_all_derived_quantities(self):
        pass
