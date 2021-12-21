Module weac.tools
=================
Helper functions for the WEak Layer AntiCrack nucleation model.

Functions
---------

    
`calc_center_of_gravity(layers)`
:   Calculate z-coordinate of the center of gravity.
    
    Arguments
    ---------
    layers : list
        2D list of layer densities and thicknesses. Columns are
        density (kg/m^3) and thickness (mm). One row corresponds
        to one layer.
    
    Returns
    -------
    H : float
        Total slab thickness (mm).
    zs : float
        Z-coordinate of center of gravity (mm).

    
`gerling(rho, C0=6.0, C1=4.6)`
:   Compute Young's modulus density according to Gerling et al. 2017.
    
    Arguments
    ---------
    rho : float or ndarray
        Density (kg/m^3).
    C0 : float, optional
        Multiplicative constant of Young modulus parametrization
        according to Gerling et al. (2017). Default is 6.0.
    C1 : float, optional
        Exponent of Young modulus parameterization according to
        Gerling et al. (2017). Default is 4.6.
    
    Returns
    -------
    E : float or ndarray
        Young's modulus (MPa).

    
`isnotebook()`
:   Identify shell environment.

    
`load_dummy_profile(profile_id)`
:   Define standard layering types for comparison.

    
`scapozza(rho)`
:   Compute Young's modulus (MPa) from density (kg/m^3).
    
    Arguments
    ---------
    rho : float or ndarray
        Density (kg/m^3).
    
    Returns
    -------
    E : float or ndarray
        Young's modulus (MPa).

    
`time()`
:   Return current time in milliseconds.