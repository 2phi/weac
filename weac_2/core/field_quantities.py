"""
This module defines the field quantities for the WEAC simulation.
The field quantities are extracted from the system model and system properties.
"""

import numpy as np
import logging

from weac_2.core.eigensystem import SystemProperties

logger = logging.getLogger(__name__)

class FieldQuantities():
    """
    This class is used to define the field quantities for the WEAC simulation.
    """
    unknown_constants: np.ndarray
    system_properties: SystemProperties
    
    # Field quantities
    u: np.ndarray
    w: np.ndarray
    psi: np.ndarray
    du_dx: np.ndarray
    dw_dx: np.ndarray
    dpsi_dx: np.ndarray
    dz_dx: np.ndarray
    d2z_dx2: np.ndarray
    
    def __init__(self, unknown_constants: np.ndarray, system_properties: SystemProperties):
        self.unknown_constants = unknown_constants
        self.system_properties = system_properties
        
    def compute_all_field_quantities(self):
        pass
    
    def _calc_u(self):
        pass
    
    def _calc_w(self):
        pass
    
    def _calc_psi(self):
        pass
    