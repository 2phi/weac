Module weac.plot
================
Plotting resources for the WEak Layer AntiCrack nucleation model.

Functions
---------

    
`contours(instance, xq, zq, window=1000000000000.0, scale=100)`
:   Plot 2D deformation contours.
    
    Arguments
    ---------
    instance : object
        Instance of layered class.
    xq : ndarray
        Discretized x-coordinates (mm).
    zq : ndarray
        Solution vectors at positions xq as columns of matrix zq.
    window : int
        Plot window (cm) around maximum vertical deflection.
    scale : int
        Scaling factor for the visualization of displacements.

    
`displacements(instance, x, z, **segments)`
:   Wrap for dispalcements plot.

    
`err_comp(da, Gdif, Ginc, mode=0)`
:   Wrap energy release rate plot.

    
`err_modes(da, G, kind='inc')`
:   Wrap energy release rate plot.

    
`fea_disp(instance, xq, zq, fea)`
:   Wrap dispalcements plot.

    
`fea_stress(instance, xb, zb, fea)`
:   Wrap stress plot.

    
`outline(grid)`
:   Extract outline values of a 2D array (matrix, grid).

    
`plot_data(name, ax1data, ax1label, ax2data=None, ax2label=None, labelpos=None, vlines=True, li=False, mi=False, ki=False, xlabel='Horizontal position $x$ (cm)')`
:   Plot data. Base function.

    
`section_forces(instance, x, z, **segments)`
:   Wrap section forces plot.

    
`set_plotstyles()`
:   Define styles plot markers, labels and colors.

    
`slab_profile(instance)`
:   Create bar chart of slab profile.

    
`stress_criteria(x, stress, **segments)`
:   Wrap plot of stress and energy criteria.

    
`stresses(instance, x, z, **segments)`
:   Wrap stress plot.

Classes
-------

`MidpointNormalize(vmin, vmax, midpoint=0, clip=False)`
:   Colormap normalization to a specified midpoint. Default is 0.
    
    Inizialize normalization.

    ### Ancestors (in MRO)

    * matplotlib.colors.Normalize