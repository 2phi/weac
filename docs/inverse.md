Module weac.inverse
===================
Class for the elastic analysis of layered snow slabs.

Classes
-------

`Inverse(system='pst-', layers=None, parameters=(6.0, 4.6, 0.25))`
:   Fit the elastic properties of the layers of a snowpack.
    
    Allows for the inverse identification of the elastic properties
    of the layers of a snowpack from full-field displacement
    measurements.
    
    Inherits methods for the eigensystem calculation from the base
    class Eigensystem(), methods for the calculation of field
    quantities from FieldQuantitiesMixin(), methods for the solution
    of the system from SolutionMixin() and methods for the output
    analysis from AnalysisMixin().
    
    Initialize model with user input.
    
    Arguments
    ---------
    system : str, optional
        Type of system to analyse. Default is 'pst-'.
    layers : list, optional
        List of layer densities and thicknesses. Default is None.
    parameters : tuple, optional
        Fitting parameters C0, C1, and Eweak. Multiplicative constant
        of Young modulus parametrization, exponent constant of Young
        modulus parametrization, and weak-layer Young modulus,
        respectively. Default is (6.0, 4.6, 0.25).

    ### Ancestors (in MRO)

    * weac.mixins.FieldQuantitiesMixin
    * weac.mixins.SolutionMixin
    * weac.mixins.AnalysisMixin
    * weac.eigensystem.Eigensystem