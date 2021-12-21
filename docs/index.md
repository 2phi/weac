Module weac
===========
WEak Layer AntiCrack nucleation model.

Implementation of closed-form analytical models for the analysis of
dry-snow slab avalanche release.

Sub-modules
-----------
* weac.eigensystem
* weac.fracture
* weac.inverse
* weac.layered
* weac.mixins
* weac.plot
* weac.tools

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

`Layered(system='pst-', layers=None)`
:   Layered beam on elastic foundation model application interface.
    
    Inherits methods for the eigensystem calculation from the base
    class Eigensystem(), methods for the calculation of field
    quantities from FieldQuantitiesMixin(), methods for the solution
    of the system from SolutionMixin() and methods for the output
    analysis from AnalysisMixin().
    
    Initialize model with user input.
    
    Arguments
    ---------
    system : {'pst-', '-pst', 'skier', 'skiers'}, optional
        Type of system to analyse: PST cut from the right (pst-),
        PST cut form the left (-pst), one skier on infinite
        slab (skier) or multiple skiers on infinite slab (skiers).
        Default is 'pst-'.
    layers : list, optional
        2D list of layer densities and thicknesses. Columns are
        density(kg/m ^ 3) and thickness(mm). One row corresponds
        to one layer. Default is [[240, 200], ].

    ### Ancestors (in MRO)

    * weac.mixins.FieldQuantitiesMixin
    * weac.mixins.SolutionMixin
    * weac.mixins.AnalysisMixin
    * weac.eigensystem.Eigensystem