Module weac.mixins
==================
Mixins for the elastic analysis of layered snow slabs.

Classes
-------

`AnalysisMixin()`
:   Mixin for the analysis of model outputs.
    
    Provides methods for the analysis of layered slabs on compliant
    elastic foundations.

    ### Descendants

    * weac.inverse.Inverse
    * weac.layered.Layered

    ### Methods

    `gdif(self, C, phi, li, ki, **kwargs)`
    :   Compute differential energy release rate of all crack tips.
        
        Arguments
        ---------
        C : ndarray
            Free constants of the solution.
        phi : float
            Inclination (degress).
        li : ndarray
            List of segment lengths.
        ki : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not in the cracked configuration.
        
        Returns
        -------
        ndarray
            List of total, mode I, and mode II energy release rates.

    `ginc(self, C0, C1, phi, li, ki, k0, **kwargs)`
    :   Compute incremental energy relase rate of of all cracks.
        
        Arguments
        ---------
        C0 : ndarray
            Free constants of uncracked solution.
        C1 : ndarray
            Free constants of cracked solution.
        phi : float
            Inclination (degress).
        li : ndarray
            List of segment lengths.
        ki : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not in the cracked configuration.
        k0 : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not in the uncracked configuration.
        
        Returns
        -------
        ndarray
            List of total, mode I, and mode II energy release rates.

    `rasterize_solution(self, C, phi, li, ki, num=250, **kwargs)`
    :   Compute rasterized solution vector.
        
        Arguments
        ---------
        C : ndarray
            Vector of free constants.
        phi : float
            Inclination (degrees).
        li : ndarray
            List of segment lengths (mm).
        ki : ndarray
            List of booleans indicating whether segment lies on
            a foundation or not.
        num : int
            Number of grid points.
        
        Returns
        -------
        xq : ndarray
            Grid point x-coordinates at which solution vector
            is discretized.
        zq : ndarray
            Matrix with solution vectors as colums at grid
            points xq.
        xb : ndarray
            Grid point x-coordinates that lie on a foundation.

`FieldQuantitiesMixin()`
:   Mixin for field quantities.
    
    Provides methods for the computation of displacements, stresses,
    strains, and energy release rates from the solution vector.

    ### Descendants

    * weac.inverse.Inverse
    * weac.layered.Layered

    ### Methods

    `Gi(self, Ztip)`
    :   Get mode I differential energy release rate at crack tip.
        
        Arguments
        ---------
        Ztip : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T
            at the crack tip.
        
        Returns
        -------
        Gi : float
            Mode I differential energy release rate (N/mm) at the
            crack tip.

    `Gii(self, Ztip)`
    :   Get mode II differential energy release rate at crack tip.
        
        Arguments
        ---------
        Ztip : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T
            at the crack tip.
        
        Returns
        -------
        Gii : float
            Mode II differential energy release rate (N/mm) at the
            crack tip.

    `M(self, Z)`
    :   Get bending moment M = B11 u' + D11 psi' in the slab.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        M : float
            Bending moment M (Nmm) in the slab.

    `N(self, Z)`
    :   Get the axial normal force N = A11 u' + B11 psi' in the slab.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        N : float
            Axial normal force N (N) in the slab.

    `V(self, Z)`
    :   Get vertical shear force V = kA55(w' + psi).
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        V : float
            Vertical shear force V (N) in the slab.

    `eps(self, Z)`
    :   Get weak-layer normal strain.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        eps : float
            Weak-layer normal strain epsilon (MPa).

    `gamma(self, Z)`
    :   Get weak-layer shear strain.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        gamma : float
            Weak-layer shear strain gamma (MPa).

    `int1(self, x, z0, z1)`
    :   Get mode I crack opening integrand at integration points xi.
        
        Arguments
        ---------
        x : float, ndarray
            X-coordinate where integrand is to be evaluated (mm).
        z0 : callable
            Function that returns the solution vector of the uncracked
            configuration.
        z1 : callable
            Function that returns the solution vector of the cracked
            configuration.
        
        Returns
        -------
        int1 : float or ndarray
            Integrant of the mode I crack opening integral.

    `int2(self, x, z0, z1)`
    :   Get mode II crack opening integrand at integration points xi.
        
        Arguments
        ---------
        x: float, ndarray
            X-coordinate where integrand is to be evaluated(mm).
        z0: callable
            Function that returns the solution vector of the uncracked
            configuration.
        z1: callable
            Function that returns the solution vector of the cracked
            configuration.
        
        Returns
        -------
        int2 : float or ndarray
            Integrant of the mode II crack opening integral.

    `maxp(self, Z)`
    :   Get maximum principal stress in the weak layer.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        maxp : float
            Maximum principal stress (MPa) in the weak layer.

    `psi(self, Z)`
    :   Get midplane rotation psi.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        psi : float
            Midplane rotation psi (radians) of the slab.

    `psip(self, Z)`
    :   Get first derivative psi' of the midplane rotation.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        psip : float
            First derivative psi' of the midplane rotation (radians/mm)
             of the slab.

    `sig(self, Z)`
    :   Get weak-layer normal stress.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        sig : float
            Weak-layer normal stress sigma (MPa).

    `tau(self, Z)`
    :   Get weak-layer shear stress.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        tau : float
            Weak-layer shear stress tau (MPa).

    `u(self, Z, z0)`
    :   Get horizontal displacement u = u0 + z0 psi.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        z0 : float
            Z-coordinate (mm) where u is to be evaluated.
        
        Returns
        -------
        u : float
            Horizontal displacement u (mm) of the slab.

    `up(self, Z, z0)`
    :   Get first derivative of the horizontal displacement.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        z0 : float
            Z-coordinate (mm) where u is to be evaluated.
        
        Returns
        -------
        up : float
            First derivative u' = u0' + z0 psi' of the horizontal
            displacement of the slab.

    `w(self, Z)`
    :   Get centerline deflection w.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        w : float
            Deflection w (mm) of the slab.

    `wp(self, Z)`
    :   Get first derivative w' of the centerline deflection.
        
        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        
        Returns
        -------
        wp : float
            First derivative w' of the deflection of the slab.

`SolutionMixin()`
:   Mixin for the solution of boundary value problems.
    
    Provides methods for the assembly of the system of equations
    and for the computation of the free constants.

    ### Descendants

    * weac.inverse.Inverse
    * weac.layered.Layered

    ### Methods

    `assemble_and_solve(self, phi, li, mi, ki)`
    :   Compute free constants for arbitrary beam assembly.
        
        Assemble LHS from bedded and free segments in the form
        [][zh1  0   0  ...  0   0   0][][][]  left
        [] = [zh1 zh2  0  ...  0   0   0][] + [] = []  mid
        [][0  zh2 zh3 ...  0   0   0][][][]  mid
        [z0][... ... ... ... ... ... ...][C][zp][rhs]  mid
        [][0   0   0  ... zhL zhM  0][][][]  mid
        [][0   0   0  ...  0  zhM zhN][][][]  mid
        [][0   0   0  ...  0   0  zhN][][][]  right
        and solve for constants C.
        
        Arguments
        ---------
        phi: float
            Inclination(degrees).
        li: ndarray
            List of lengths of segements(mm).
        mi: ndarray
            List of skier weigths(kg) at segement boundaries.
        ki: ndarray
            List of one bool per segement indicating whether segement
            has foundation(True) or not (False).
        
        Returns
        -------
        C: ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.

    `bc(self, z)`
    :   Provide equations for free(pst) or infinite(skiers) ends.
        
        Arguments
        ---------
        z: ndarray
            Solution vector(6x1) at a certain position x.
        
        Returns
        -------
        bc: ndarray
            Boundary condition vector(lenght 3) at position x.

    `calc_segments(self, li=False, mi=False, ki=False, k0=False, L=10000.0, a=0, m=0, **kwargs)`
    :   Assemble lists defining the segments.
        
        This includes length(li), foundation(ki, k0), and skier weight(mi).
        
        Arguments
        ---------
        li: squence, optional
            List of lengths of segements(mm). Used for system 'skiers'.
        mi: squence, optional
            List of skier weigths(kg) at segement boundaries. Used for
            system 'skiers'.
        ki: squence, optional
            List of one bool per segement indicating whether segement
            has foundation(True) or not (False) in the cracked state.
            Used for system 'skiers'.
        k0: squence, optional
            List of one bool per segement indicating whether segement
            has foundation(True) or not (False) in the uncracked state.
            Used for system 'skiers'.
        L: float, optional
            Total length of model(mm). Used for systems 'pst-', '-pst',
            and 'skier'.
        a: float, optional
            Crack length(mm).  Used for systems 'pst-', '-pst', and
            'skier'.
        m: float, optional
            Weight of skier(kg) in the axial center of the model.
            Used for system 'skier'.
        
        Returns
        -------
        segments: dict
            Dictionary with lists of segement lengths(li), skier
            weights(mi), and foundation booleans in the cracked(ki)
            and ncracked(k0) configurations.

    `eqs(self, zl, zr, pos='mid')`
    :   Provide boundary or transmission conditions for beam segments.
        
        Arguments
        ---------
        zl: ndarray
            Solution vector(6x1) at left end of beam segement.
        zr: ndarray
            Solution vector(6x1) at right end of beam segement.
        pos: {'left', 'mid', 'right', 'l', 'm', 'r'}, optional
            Determines whether the segement under consideration
            is a left boundary segement(left, l), one of the
            center segement(mid, m), or a right boundary
            segement(right, r). Default is 'mid'.
        
        Returns
        -------
        eqs: ndarray
            Vector(of length 9) of boundary conditions(3) and
            transmission conditions(6) for boundary segements
            or vector of transmission conditions(of length 6+6)
            for center segments.