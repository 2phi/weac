"""Class for the elastic analysis of layered snow slabs."""
# pylint: disable=invalid-name

# Project imports
from weac.mixins import FieldQuantitiesMixin, SolutionMixin, AnalysisMixin
from weac.eigensystem import Eigensystem


class Inverse(FieldQuantitiesMixin, SolutionMixin, AnalysisMixin, Eigensystem):
    """
    Fit the elastic properties of the layers of a snowpack.

    Allows for the inverse identification of the elastic properties
    of the layers of a snowpack from full-field displacement
    measurements.

    Inherits methods for the eigensystem calculation from the base
    class Eigensystem(), methods for the calculation of field
    quantities from FieldQuantitiesMixin(), methods for the solution
    of the system from SolutionMixin() and methods for the output
    analysis from AnalysisMixin.
    """

    def __init__(
            self, system='pst-', layers=None, parameters=(6.0, 4.6, 0.25)):
        """
        Initialize model with user input.

        Arguments
        ---------
        system : str, optional
            Type of system to analyse(default is 'pst-')
        layers : list, optional
            List of layer densities and thicknesses(default is None)
        parameters : tuple, optional
            Fitting parameters C0, C1, and Eweak. Multiplicative constant
            of Young modulus parametrization, exponent constant of Young
            modulus parametrization, and weak-layer Young modulus,
            respectively(default is (6.0, 4.6, 0.25))
        """
        # Call parent __init__
        super().__init__(system=system)

        # Unpack fitting parameters
        C0, C1, Eweak = parameters

        # Set material properties
        self.set_beam_properties(layers=layers, C0=C0, C1=C1)
        self.set_foundation_properties(E=Eweak)

        # Set up model
        self.calc_foundation_stiffness()
        self.calc_laminate_stiffness_matrix()
        self.calc_system_matrix()
        self.calc_eigensystem()