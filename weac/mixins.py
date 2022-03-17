"""Mixins for the elastic analysis of layered snow slabs."""
# pylint: disable=invalid-name,too-many-locals,too-many-arguments

# Standard library imports
from functools import partial

# Third party imports
import numpy as np
from scipy.integrate import romberg


class FieldQuantitiesMixin:
    """
    Mixin for field quantities.

    Provides methods for the computation of displacements, stresses,
    strains, and energy release rates from the solution vector.
    """

    # pylint: disable=no-self-use
    def w(self, Z, unit='mm'):
        """
        Get centerline deflection w.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Desired output unit. Default is mm.

        Returns
        -------
        float
            Deflection w (in specified unit) of the slab.
        """
        convert = {
            'm': 1e-3,   # meters
            'cm': 1e-1,  # centimeters
            'mm': 1,     # millimeters
            'um': 1e3    # micrometers
        }
        return convert[unit]*Z[2, :]

    def wp(self, Z):
        """
        Get first derivative w' of the centerline deflection.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            First derivative w' of the deflection of the slab.
        """
        return Z[3, :]

    def psi(self, Z, unit='rad'):
        """
        Get midplane rotation psi.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        unit : {'deg', 'degrees', 'rad', 'radians'}, optional
            Desired output unit. Default is radians.

        Returns
        -------
        psi : float
            Cross-section rotation psi (radians) of the slab.
        """
        if unit in ['deg', 'degree', 'degrees']:
            psi = np.rad2deg(Z[4, :])
        elif unit in ['rad', 'radian', 'radians']:
            psi = Z[4, :]
        return psi

    def psip(self, Z):
        """
        Get first derivative psi' of the midplane rotation.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            First derivative psi' of the midplane rotation (radians/mm)
             of the slab.
        """
        return Z[5, :]

    # pylint: enable=no-self-use
    def u(self, Z, z0, unit='mm'):
        """
        Get horizontal displacement u = u0 + z0 psi.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        z0 : float
            Z-coordinate (mm) where u is to be evaluated.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Desired output unit. Default is mm.

        Returns
        -------
        float
            Horizontal displacement u (mm) of the slab.
        """
        convert = {
            'm': 1e-3,   # meters
            'cm': 1e-1,  # centimeters
            'mm': 1,     # millimeters
            'um': 1e3    # micrometers
        }
        return convert[unit]*(Z[0, :] + z0*self.psi(Z))

    def up(self, Z, z0):
        """
        Get first derivative of the horizontal displacement.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        z0 : float
            Z-coordinate (mm) where u is to be evaluated.

        Returns
        -------
        float
            First derivative u' = u0' + z0 psi' of the horizontal
            displacement of the slab.
        """
        return Z[1, :] + z0*self.psip(Z)

    def N(self, Z):
        """
        Get the axial normal force N = A11 u' + B11 psi' in the slab.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            Axial normal force N (N) in the slab.
        """
        return self.A11*Z[1, :] + self.B11*Z[5, :]

    def M(self, Z):
        """
        Get bending moment M = B11 u' + D11 psi' in the slab.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            Bending moment M (Nmm) in the slab.
        """
        return self.B11*Z[1, :] + self.D11*Z[5, :]

    def V(self, Z):
        """
        Get vertical shear force V = kA55(w' + psi).

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            Vertical shear force V (N) in the slab.
        """
        return self.kA55*(Z[3, :] + Z[4, :])

    def sig(self, Z, unit='MPa'):
        """
        Get weak-layer normal stress.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        unit : {'MPa', 'kPa'}, optional
            Desired output unit. Default is MPa.

        Returns
        -------
        float
            Weak-layer normal stress sigma (in specified unit).
        """
        convert = {
            'kPa': 1e3,
            'MPa': 1
        }
        return -convert[unit]*self.kn*self.w(Z)

    def tau(self, Z, unit='MPa'):
        """
        Get weak-layer shear stress.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        unit : {'MPa', 'kPa'}, optional
            Desired output unit. Default is MPa.

        Returns
        -------
        float
            Weak-layer shear stress tau (in specified unit).
        """
        convert = {
            'kPa': 1e3,
            'MPa': 1
        }
        return -convert[unit]*self.kt*(
            self.wp(Z)*self.t/2 - self.u(Z, z0=self.h/2))

    def eps(self, Z):
        """
        Get weak-layer normal strain.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            Weak-layer normal strain epsilon.
        """
        return -self.w(Z)/self.t

    def gamma(self, Z):
        """
        Get weak-layer shear strain.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        float
            Weak-layer shear strain gamma.
        """
        return self.wp(Z)/2 - self.u(Z, z0=self.h/2)/self.t

    def maxp(self, Z):
        """
        Get maximum principal stress in the weak layer.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.

        Returns
        -------
        loat
            Maximum principal stress (MPa) in the weak layer.
        """
        sig = self.sig(Z)
        tau = self.tau(Z)
        return np.amax([[sig + np.sqrt(sig**2 + 4*tau**2),
                         sig - np.sqrt(sig**2 + 4*tau**2)]], axis=1)[0]/2

    def Gi(self, Ztip, unit='kJ/m^2'):
        """
        Get mode I differential energy release rate at crack tip.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T
            at the crack tip.
        unit : {'N/mm', 'kJ/m^2', 'J/m^2'}, optional
            Desired output unit. Default is kJ/m^2.

        Returns
        -------
        float
            Mode I differential energy release rate (N/mm = kJ/m^2
            or J/m^2) at the crack tip.
        """
        convert = {
            'J/m^2': 1e3,   # joule per square meter
            'kJ/m^2': 1,    # kilojoule per square meter
            'N/mm': 1       # newton per millimeter
        }
        return convert[unit]*self.sig(Ztip)**2/(2*self.kn)

    def Gii(self, Ztip, unit='kJ/m^2'):
        """
        Get mode II differential energy release rate at crack tip.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T
            at the crack tip.
        unit : {'N/mm', 'kJ/m^2', 'J/m^2'}, optional
            Desired output unit. Default is kJ/m^2 = N/mm.

        Returns
        -------
        float
            Mode II differential energy release rate (N/mm = kJ/m^2
            or J/m^2) at the crack tip.
        """
        convert = {
            'J/m^2': 1e3,   # joule per square meter
            'kJ/m^2': 1,    # kilojoule per square meter
            'N/mm': 1       # newton per millimeter
        }
        return convert[unit]*self.tau(Ztip)**2/(2*self.kt)

    def int1(self, x, z0, z1):
        """
        Get mode I crack opening integrand at integration points xi.

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
        float or ndarray
            Integrant of the mode I crack opening integral.
        """
        return self.sig(z0(x))*self.eps(z1(x))*self.t

    def int2(self, x, z0, z1):
        """
        Get mode II crack opening integrand at integration points xi.

        Arguments
        ---------
        x: float, ndarray
            X-coordinate where integrand is to be evaluated (mm).
        z0: callable
            Function that returns the solution vector of the uncracked
            configuration.
        z1: callable
            Function that returns the solution vector of the cracked
            configuration.

        Returns
        -------
        float or ndarray
            Integrant of the mode II crack opening integral.
        """
        return self.tau(z0(x))*self.gamma(z1(x))*self.t


class SolutionMixin:
    """
    Mixin for the solution of boundary value problems.

    Provides methods for the assembly of the system of equations
    and for the computation of the free constants.
    """

    def bc(self, z):
        """
        Provide equations for free (pst) or infinite (skiers) ends.

        Arguments
        ---------
        z: ndarray
            Solution vector (6x1) at a certain position x.

        Returns
        -------
        bc: ndarray
            Boundary condition vector (lenght 3) at position x.
        """
        if self.system in ['pst-', '-pst']:
            # Free ends
            bc = np.array([self.N(z), self.M(z), self.V(z)])
        elif self.system in ['skier', 'skiers']:
            # Infinite ends (vanishing complementary solution)
            bc = np.array([self.u(z, z0=0), self.w(z), self.psi(z)])
        else:
            raise ValueError(
                'Boundary conditions not defined for'
                f'system of type {self.system}.')

        return bc

    def eqs(self, zl, zr, pos='mid'):
        """
        Provide boundary or transmission conditions for beam segments.

        Arguments
        ---------
        zl: ndarray
            Solution vector (6x1) at left end of beam segement.
        zr: ndarray
            Solution vector (6x1) at right end of beam segement.
        pos: {'left', 'mid', 'right', 'l', 'm', 'r'}, optional
            Determines whether the segement under consideration
            is a left boundary segement (left, l), one of the
            center segement (mid, m), or a right boundary
            segemen t (right, r). Default is 'mid'.

        Returns
        -------
        eqs: ndarray
            Vector (of length 9) of boundary conditions (3) and
            transmission conditions (6) for boundary segements
            or vector of transmission conditions (of length 6+6)
            for center segments.
        """
        if pos in ('l', 'left'):
            eqs = np.array([
                self.bc(zl)[0],             # Left boundary condition
                self.bc(zl)[1],             # Left boundary condition
                self.bc(zl)[2],             # Left boundary condition
                self.u(zr, z0=0),           # ui(xi = li)
                self.w(zr),                 # wi(xi = li)
                self.psi(zr),               # psii(xi = li)
                self.N(zr),                 # Ni(xi = li)
                self.M(zr),                 # Mi(xi = li)
                self.V(zr)])                # Vi(xi = li)
        elif pos in ('m', 'mid'):
            eqs = np.array([
                -self.u(zl, z0=0),          # -ui(xi = 0)
                -self.w(zl),                # -wi(xi = 0)
                -self.psi(zl),              # -psii(xi = 0)
                -self.N(zl),                # -Ni(xi = 0)
                -self.M(zl),                # -Mi(xi = 0)
                -self.V(zl),                # -Vi(xi = 0)
                self.u(zr, z0=0),           # ui(xi = li)
                self.w(zr),                 # wi(xi = li)
                self.psi(zr),               # psii(xi = li)
                self.N(zr),                 # Ni(xi = li)
                self.M(zr),                 # Mi(xi = li)
                self.V(zr)])                # Vi(xi = li)
        elif pos in ('r', 'right'):
            eqs = np.array([
                -self.u(zl, z0=0),          # -ui(xi = 0)
                -self.w(zl),                # -wi(xi = 0)
                -self.psi(zl),              # -psii(xi = 0)
                -self.N(zl),                # -Ni(xi = 0)
                -self.M(zl),                # -Mi(xi = 0)
                -self.V(zl),                # -Vi(xi = 0)
                self.bc(zr)[0],             # Right boundary condition
                self.bc(zr)[1],             # Right boundary condition
                self.bc(zr)[2]])            # Right boundary condition
        else:
            raise ValueError(
                (f'Invalid position argument {pos} given. '
                 'Valid segment positions are l, m, and r, '
                 'or left, mid and right.'))

        return eqs

    def calc_segments(self, li=False, mi=False, ki=False, k0=False,
                      L=1e4, a=0, m=0, **kwargs):
        """
        Assemble lists defining the segments.

        This includes length (li), foundation (ki, k0), and skier
        weight (mi).

        Arguments
        ---------
        li: squence, optional
            List of lengths of segements(mm). Used for system 'skiers'.
        mi: squence, optional
            List of skier weigths (kg) at segement boundaries. Used for
            system 'skiers'.
        ki: squence, optional
            List of one bool per segement indicating whether segement
            has foundation (True) or not (False) in the cracked state.
            Used for system 'skiers'.
        k0: squence, optional
            List of one bool per segement indicating whether segement
            has foundation(True) or not (False) in the uncracked state.
            Used for system 'skiers'.
        L: float, optional
            Total length of model (mm). Used for systems 'pst-', '-pst',
            and 'skier'.
        a: float, optional
            Crack length (mm).  Used for systems 'pst-', '-pst', and
            'skier'.
        m: float, optional
            Weight of skier (kg) in the axial center of the model.
            Used for system 'skier'.

        Returns
        -------
        segments: dict
            Dictionary with lists of segement lengths (li), skier
            weights (mi), and foundation booleans in the cracked (ki)
            and uncracked (k0) configurations.
        """
        _ = kwargs                                      # Unused arguments
        if self.system == 'skiers':
            li = np.array(li)                           # Segment lengths
            mi = np.array(mi)                           # Skier weights
            ki = np.array(ki)                           # Crack
            k0 = np.array(k0)                           # No crack
        elif self.system == 'pst-':
            li = np.array([L - a, a])                   # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([True, False])                # Crack
            k0 = np.array([True, True])                 # No crack
        elif self.system == '-pst':
            li = np.array([a, L - a])                   # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([False, True])                # Crack
            k0 = np.array([True, True])                 # No crack
        elif self.system == 'skier':
            lb = (L - a)/2                              # Half bedded length
            lf = a/2                                    # Half free length
            li = np.array([lb, lf, lf, lb])             # Segment lengths
            mi = np.array([0, m, 0])                    # Skier weights
            ki = np.array([True, False, False, True])   # Crack
            k0 = np.array([True, True, True, True])     # No crack
        else:
            raise ValueError(f'System {self.system} is not implemented.')

        # Fill dictionary
        segments = {
            'nocrack': {'li': li, 'mi': mi, 'ki': k0},
            'crack': {'li': li, 'mi': mi, 'ki': ki},
            'both': {'li': li, 'mi': mi, 'ki': ki, 'k0': k0}}

        return segments

    def assemble_and_solve(self, phi, li, mi, ki):
        """
        Compute free constants for arbitrary beam assembly.

        Assemble LHS from supported and unsupported segments in the form
        [  ]   [ zh1  0   0  ...  0   0   0  ][   ]   [    ]   [     ]  left
        [  ]   [ zh1 zh2  0  ...  0   0   0  ][   ]   [    ]   [     ]  mid
        [  ]   [  0  zh2 zh3 ...  0   0   0  ][   ]   [    ]   [     ]  mid
        [z0] = [ ... ... ... ... ... ... ... ][ C ] + [ zp ] = [ rhs ]  mid
        [  ]   [  0   0   0  ... zhL zhM  0  ][   ]   [    ]   [     ]  mid
        [  ]   [  0   0   0  ...  0  zhM zhN ][   ]   [    ]   [     ]  mid
        [  ]   [  0   0   0  ...  0   0  zhN ][   ]   [    ]   [     ]  right
        and solve for constants C.

        Arguments
        ---------
        phi: float
            Inclination (degrees).
        li: ndarray
            List of lengths of segements (mm).
        mi: ndarray
            List of skier weigths (kg) at segement boundaries.
        ki: ndarray
            List of one bool per segement indicating whether segement
            has foundation (True) or not (False).

        Returns
        -------
        C: ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        """
        # --- CATCH ERRORS ----------------------------------------------------

        # No foundation
        if not any(ki):
            raise ValueError('Provide at least one bedded segment.')
        # Mismatch of number of segements and transisions
        if len(li) != len(ki) or len(li) - 1 != len(mi):
            raise ValueError('Make sure len(li)=N, len(ki)=N, and '
                             'len(mi)=N-1 for a system of N segments.')

        if self.system not in ['pst-', '-pst']:
            # Boundary segements must be on foundation for infinite BCs
            if not all([ki[0], ki[-1]]):
                raise ValueError('Provide bedded boundary segments in '
                                 'order to account for infinite extensions.')
            # Make sure infinity boundary conditions are far enough from skiers
            if li[0] < 5e3 or li[-1] < 5e3:
                print(('WARNING: Boundary segments are short. Make sure '
                       'the complementary solution has decayed to the '
                       'boundaries.'))

        # --- PREPROCESSING ---------------------------------------------------

        # Determine size of linear system of equations
        nS = len(li)            # Number of beam segments
        nDOF = 6                # Number of free constants per segment

        # Add dummy segment if only one segment provided
        if nS == 1:
            li.append(0)
            ki.append(True)
            mi.append(0)
            nS = 2

        # Assemble position vector
        pi = np.full(nS, 'm')
        pi[0], pi[-1] = 'l', 'r'

        # Initialize matrices
        zh0 = np.zeros([nS*6, nS*nDOF])
        zp0 = np.zeros([nS*6, 1])
        rhs = np.zeros([nS*6, 1])

        # --- ASSEMBLE LINEAR SYSTEM OF EQUATIONS -----------------------------

        # Loop through segments to assemble left-hand side
        for i in range(nS):
            # Length, foundation and position of segment i
            l, k, pos = li[i], ki[i], pi[i]
            # Transmission conditions at left and right segment ends
            zhi = self.eqs(
                zl=self.zh(x=0, l=l, bed=k),
                zr=self.zh(x=l, l=l, bed=k),
                pos=pos)
            zpi = self.eqs(
                zl=self.zp(x=0, phi=phi, bed=k),
                zr=self.zp(x=l, phi=phi, bed=k),
                pos=pos)
            # Rows for left-hand side assembly
            start = 0 if i == 0 else 3
            stop = 6 if i == nS - 1 else 9
            # Assemble left-hand side
            zh0[(6*i - start):(6*i + stop), i*nDOF:(i + 1)*nDOF] = zhi
            zp0[(6*i - start):(6*i + stop)] += zpi

        # Loop through loads to assemble right-hand side
        for i, m in enumerate(mi, start=1):
            # Get skier loads
            Fn, Ft = self.get_skier_load(m, phi)
            # Right-hand side for transmission from segment i-1 to segment i
            rhs[6*i:6*i + 3] = np.vstack([Ft, -Ft*self.h/2, Fn])

        # Set rhs so that complementary integral vanishes at boundaries
        if self.system not in ['pst-', '-pst']:
            rhs[:3] = self.bc(self.zp(x=0, phi=phi, bed=ki[0]))
            rhs[-3:] = self.bc(self.zp(x=li[-1], phi=phi, bed=ki[-1]))

        # --- SOLVE -----------------------------------------------------------

        # Solve z0 = zh0*C + zp0 = rhs for constants, i.e. zh0*C = rhs - zp0
        C = np.linalg.solve(zh0, rhs - zp0)

        # Sort (nDOF = 6) constants for each segment into columns of a matrix
        return C.reshape([-1, nDOF]).T


class AnalysisMixin:
    """
    Mixin for the analysis of model outputs.

    Provides methods for the analysis of layered slabs on compliant
    elastic foundations.
    """

    def rasterize_solution(self, C, phi, li, ki, num=250, **kwargs):
        """
        Compute rasterized solution vector.

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
        """
        # Unused arguments
        _ = kwargs

        # Drop zero-length segments
        isnonzero = li > 0
        C, ki, li = C[:, isnonzero], ki[isnonzero], li[isnonzero]

        # Compute number of plot points per segment (+1 for last segment)
        nq = np.ceil(li/li.sum()*num).astype('int')
        nq[-1] += 1

        # Provide cumulated length and plot point lists
        lic = np.insert(np.cumsum(li), 0, 0)
        nqc = np.insert(np.cumsum(nq), 0, 0)

        # Initialize arrays
        isbedded = np.full(nq.sum(), True)
        xq = np.full(nq.sum(), np.nan)
        zq = np.full([6, xq.size], np.nan)

        # Loop through segments
        for i, l in enumerate(li):
            # Get local x-coordinates of segment i
            xi = np.linspace(0, l, num=nq[i], endpoint=(i == li.size - 1))
            # Compute start and end coordinates of segment i
            x0 = lic[i]
            # Assemble global coordinate vector
            xq[nqc[i]:nqc[i + 1]] = x0 + xi
            # Mask coordinates not on foundation (including endpoints)
            if not ki[i]:
                isbedded[nqc[i]:nqc[i + 1]] = False
            # Compute segment solution
            zi = self.z(xi, C[:, [i]], l, phi, ki[i])
            # Assemble global solution matrix
            zq[:, nqc[i]:nqc[i + 1]] = zi

        # Make sure cracktips are included
        transmissionbool = [ki[j] or ki[j + 1] for j, _ in enumerate(ki[:-1])]
        for i, truefalse in enumerate(transmissionbool, start=1):
            isbedded[nqc[i]] = truefalse

        # Assemble vector of coordinates on foundation
        xb = np.full(nq.sum(), np.nan)
        xb[isbedded] = xq[isbedded]

        return xq, zq, xb

    def ginc(self, C0, C1, phi, li, ki, k0, **kwargs):
        """
        Compute incremental energy relase rate of of all cracks.

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
        """
        # Unused arguments
        _ = kwargs

        # Make sure inputs are np.arrays
        li, ki, k0 = np.array(li), np.array(ki), np.array(k0)

        # Reduce inputs to segments with crack advance
        iscrack = k0 & ~ki
        C0, C1, li = C0[:, iscrack], C1[:, iscrack], li[iscrack]

        # Compute total crack lenght and initialize outputs
        da = li.sum() if li.sum() > 0 else np.nan
        Ginc1, Ginc2 = 0, 0

        # Loop through segments with crack advance
        for j, l in enumerate(li):

            # Uncracked (0) and cracked (1) solutions at integration points
            z0 = partial(self.z, C=C0[:, [j]], l=l, phi=phi, bed=True)
            z1 = partial(self.z, C=C1[:, [j]], l=l, phi=phi, bed=False)

            # Mode I (1) and II (2) integrands at integration points
            int1 = partial(self.int1, z0=z0, z1=z1)
            int2 = partial(self.int2, z0=z0, z1=z1)

            # Segement contributions to total crack opening integral
            Ginc1 += romberg(int1, 0, l, rtol=self.tol,
                             vec_func=True)/(2*da)
            Ginc2 += romberg(int2, 0, l, rtol=self.tol,
                             vec_func=True)/(2*da)

        return np.array([Ginc1 + Ginc2, Ginc1, Ginc2]).flatten()

    def gdif(self, C, phi, li, ki, unit='kJ/m^2', **kwargs):
        """
        Compute differential energy release rate of all crack tips.

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
        """
        # Unused arguments
        _ = kwargs

        # Get number and indices of segment transitions
        ntr = len(li) - 1
        itr = np.arange(ntr)

        # Identify bedded-free and free-bedded transitions as crack tips
        iscracktip = [ki[j] != ki[j + 1] for j in range(ntr)]

        # Transition indices of crack tips and total number of crack tips
        ict = itr[iscracktip]
        nct = len(ict)

        # Initialize energy release rate array
        Gdif = np.zeros([3, nct])

        # Compute energy relase rate of all crack tips
        for j, idx in enumerate(ict):
            # Solution at crack tip
            z = self.z(li[idx], C[:, [idx]], li[idx], phi, bed=ki[idx])
            # Mode I and II differential energy release rates
            Gdif[1:, j] = self.Gi(z, unit=unit), self.Gii(z, unit=unit)

        # Sum mode I and II contributions
        Gdif[0, :] = Gdif[1, :] + Gdif[2, :]

        # Adjust contributions for center cracks
        if nct > 1:
            avgmask = np.full(nct, True)    # Initialize mask
            avgmask[[0, -1]] = ki[[0, -1]]  # Do not weight edge cracks
            Gdif[:, avgmask] *= 0.5         # Weigth with half crack length

        # Return total differential energy release rate of all crack tips
        return Gdif.sum(axis=1)


class OutputMixin:
    """
    Mixin for outputs.

    Provides convenience methods for the assembly of output lists
    such as rasterized displacements or rasterized stresses.
    """

    def get_weaklayer_shearstress(self, x, z, unit='MPa', removeNaNs=False):
        """
        Compute weak-layer shear stress.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of unsupported
            (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
        unit : {'MPa', 'kPa'}, optional
            Stress output unit. Default is MPa.
        keepNaNs : bool
            If set, do not remove

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        sig : ndarray
            Normal stress (stress unit input).
        """
        # Convert coordinates from mm to cm and stresses from MPa to unit
        x = x/10
        tau = self.tau(z, unit=unit)
        # Filter stresses in unspupported segments
        if removeNaNs:
            # Remove coordinate-stress pairs where no weak layer is present
            tau = tau[~np.isnan(x)]
            x = x[~np.isnan(x)]
        else:
            # Set stress NaN where no weak layer is present
            tau[np.isnan(x)] = np.nan

        return x, tau

    def get_weaklayer_normalstress(self, x, z, unit='MPa', removeNaNs=False):
        """
        Compute weak-layer normal stress.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of unsupported
            (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
        unit : {'MPa', 'kPa'}, optional
            Stress output unit. Default is MPa.
        keepNaNs : bool
            If set, do not remove

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        sig : ndarray
            Normal stress (stress unit input).
        """
        # Convert coordinates from mm to cm and stresses from MPa to unit
        x = x/10
        sig = self.sig(z, unit=unit)
        # Filter stresses in unspupported segments
        if removeNaNs:
            # Remove coordinate-stress pairs where no weak layer is present
            sig = sig[~np.isnan(x)]
            x = x[~np.isnan(x)]
        else:
            # Set stress NaN where no weak layer is present
            sig[np.isnan(x)] = np.nan

        return x, sig

    def get_slab_displacement(self, x, z, loc='mid', unit='mm'):
        """
        Compute horizontal slab displacement.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of
            unsupported (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
        loc : {'top', 'mid', 'bot'}
            Get displacements of top, midplane or bottom of slab.
            Default is mid.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Displacement output unit. Default is mm.

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        ndarray
            Horizontal displacements (unit input).
        """
        # Coordinates (cm)
        x = x/10
        # Locator
        z0 = {'top': -self.h/2, 'mid': 0, 'bot': self.h/2}
        # Displacement (unit)
        u = self.u(z, z0=z0[loc], unit=unit)
        # Output array
        return x, u

    def get_slab_deflection(self, x, z, unit='mm'):
        """
        Compute vertical slab displacement.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of
            unsupported (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
            Default is mid.
        unit : {'m', 'cm', 'mm', 'um'}, optional
            Displacement output unit. Default is mm.

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        ndarray
            Vertical deflections (unit input).
        """
        # Coordinates (cm)
        x = x/10
        # Deflection (unit)
        w = self.w(z, unit=unit)
        # Output array
        return x, w

    def get_slab_rotation(self, x, z, unit='degrees'):
        """
        Compute slab cross-section rotation angle.

        Arguments
        ---------
        x : ndarray
            Discretized x-coordinates (mm) where coordinates of
            unsupported (no foundation) segments are NaNs.
        z : ndarray
            Solution vectors at positions x as columns of matrix z.
            Default is mid.
        unit : {'deg', degrees', 'rad', 'radians'}, optional
            Rotation angle output unit. Default is degrees.

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        ndarray
            Cross section rotations (unit input).
        """
        # Coordinates (cm)
        x = x/10
        # Cross-section rotation angle (unit)
        psi = self.psi(z, unit=unit)
        # Output array
        return x, psi
