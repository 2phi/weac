"""Mixins for the elastic analysis of layered snow slabs."""
# pylint: disable=invalid-name,too-many-locals,too-many-arguments,too-many-lines

# Standard library imports
from functools import partial

# Third party imports
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import romberg, cumulative_trapezoid

# Module imports
from weac.tools import tensile_strength_slab, calc_vertical_bc_center_of_gravity


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

    def dw_dx(self, Z):
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

    def dpsi_dx(self, Z):
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
            Horizontal displacement u (unit) of the slab.
        """
        convert = {
            'm': 1e-3,   # meters
            'cm': 1e-1,  # centimeters
            'mm': 1,     # millimeters
            'um': 1e3    # micrometers
        }
        return convert[unit]*(Z[0, :] + z0*self.psi(Z))

    def du_dx(self, Z, z0):
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
        return Z[1, :] + z0*self.dpsi_dx(Z)

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
            self.dw_dx(Z)*self.t/2 - self.u(Z, z0=self.h/2))

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
        return self.dw_dx(Z)/2 - self.u(Z, z0=self.h/2)/self.t

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
            Integrant of the mode II crack opening integral.
        """
        return self.tau(z0(x))*self.gamma(z1(x))*self.t

    def dz_dx(self, z, phi):
        """
        Get first derivative z'(x) = K*z(x) + q of the solution vector.

        z'(x) = [u'(x) u''(x) w'(x) w''(x) psi'(x), psi''(x)]^T

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            First derivative z'(x) for the solution vector (6x1).
        """
        K = self.calc_system_matrix()
        q = self.get_load_vector(phi)
        return np.dot(K, z) + q

    def dz_dxdx(self, z, phi):
        """
        Get second derivative z''(x) = K*z'(x) of the solution vector.

        z''(x) = [u''(x) u'''(x) w''(x) w'''(x) psi''(x), psi'''(x)]^T

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            Second derivative z''(x) = (K*z(x) + q)' = K*z'(x) = K*(K*z(x) + q)
            of the solution vector (6x1).
        """
        K = self.calc_system_matrix()
        q = self.get_load_vector(phi)
        dz_dx = np.dot(K, z) + q
        return np.dot(K, dz_dx)

    def du0_dxdx(self, z, phi):
        """
        Get second derivative of the horiz. centerline displacement u0''(x).

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            Second derivative of the horizontal centerline displacement
            u0''(x) (1/mm).
        """
        return self.dz_dx(z, phi)[1, :]

    def dpsi_dxdx(self, z, phi):
        """
        Get second derivative of the cross-section rotation psi''(x).

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            Second derivative of the cross-section rotation psi''(x) (1/mm^2).
        """
        return self.dz_dx(z, phi)[5, :]

    def du0_dxdxdx(self, z, phi):
        """
        Get third derivative of the horiz. centerline displacement u0'''(x).

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            Third derivative of the horizontal centerline displacement
            u0'''(x) (1/mm^2).
        """
        return self.dz_dxdx(z, phi)[1, :]

    def dpsi_dxdxdx(self, z, phi):
        """
        Get third derivative of the cross-section rotation psi'''(x).

        Parameters
        ----------
        z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)]^T.
        phi : float
            Inclination (degrees). Counterclockwise positive.

        Returns
        -------
        ndarray, float
            Third derivative of the cross-section rotation psi'''(x) (1/mm^3).
        """
        return self.dz_dxdx(z, phi)[5, :]


class SlabContactMixin:
    """
    Mixin for handling the touchdown situation in a PST.

    Provides Methods for the calculation of substitute spring stiffnesses,
    cracklength-tresholds and element lengths.
    """
    # pylint: disable=too-many-instance-attributes

    def set_columnlength(self,L):
        """
        Set cracklength.

        Arguments
        ---------
        L : float
            Column length of a PST (mm).
        """
        self.L = L

    def set_cracklength(self,a):
        """
        Set cracklength.

        Arguments
        ---------
        a : float
            Cracklength in a PST (mm).
        """
        self.a = a

    def set_tc(self,cf):
        """
        Set height of the crack.

        Arguments
        ---------
        cf : float
            Collapse-factor. Ratio of the crack height to the
            uncollapsed weak-layer height.
        """
        # subtract displacement under constact load from collapsed wl height
        qn = self.calc_qn()
        self.tc = cf*self.t - qn/self.kn

    def set_phi(self,phi):
        """
        Set inclination of the slab.

        Arguments
        ---------
        phi : float
            Inclination of the slab (°).
        """
        self.phi = phi

    def set_stiffness_ratio(self, ratio=1000):
        """
        Set ratio between collapsed and uncollapsed weak-layer stiffness.

        Parameters
        ----------
        ratio : int, optional
            Stiffness ratio between collapsed and uncollapsed weak layer.
            Default is 1000.
        """
        self.ratio = ratio

    def calc_qn(self):
        """
        Calc total surface normal load.

        Returns
        -------
        float
            Total surface normal load (N/mm).
        """
        return self.get_weight_load(self.phi)[0] + self.get_surface_load(self.phi)[0]

    def calc_qt(self):
        """
        Calc total surface normal load.

        Returns
        -------
        float
            Total surface normal load (N/mm).
        """
        return self.get_weight_load(self.phi)[1] + self.get_surface_load(self.phi)[1]

    def substitute_stiffness(self, L, support='rested', dof='rot'):
        """
        Calc substitute stiffness for beam on elastic foundation. 

        Arguments
        ---------
        L : float
            Total length of the PST-column (mm).
        support : string
            Type of segment foundation. Defaults to 'rested'.
        dof : string 
            Type of substitute spring, either 'rot' or 'trans'. Defaults to 'rot'.

        Returns
        -------
        k : stiffness of substitute spring.
        """
        # adjust system to substitute system
        if dof in ['rot']:
            tempsys = self.system
            self.system = 'rot'
        if dof in ['trans']:
            tempsys = self.system
            self.system = 'trans'

        # Change eigensystem for rested segment
        if support in ['rested']:
            tempkn = self.kn
            tempkt = self.kt
            self.kn = self.ratio*self.kn
            self.kt = self.ratio*self.kt
            self.calc_system_matrix()
            self.calc_eigensystem()

        # prepare list of segment characteristics
        segments = {'li': np.array([L,  0.]),
                    'mi': np.array([0]),
                    'ki': np.array([True,  True])}
        # solve system of equations
        constants = self.assemble_and_solve(phi=0, **segments)
        # calculate stiffness
        _, z_pst, _ = self.rasterize_solution(
            C=constants, phi=0, num=1, **segments)
        if dof in ['rot']:
            k = abs(1/self.psi(z_pst)[0])
        if dof in ['trans']:
            k = abs(1/self.w(z_pst)[0])

        # Reset to previous system and eigensystem
        self.system = tempsys
        if support in ['rested']:
            self.kn = tempkn
            self.kt = tempkt
            self.calc_system_matrix()
            self.calc_eigensystem()

        return k

    def calc_a1(self):
        """
        Calc transition lengths a1 (aAB).

        Returns
        -------
        a1 : float
            Length of the crack for transition of stage A to stage B (mm).
        """
        # Unpack variables
        bs = -(self.B11**2/self.A11 - self.D11)
        ss = self.kA55
        L = self.L
        tc = self.tc
        qn = self.calc_qn()

        # Create polynomial expression
        def polynomial(x):
            # Spring stiffness supported segment
            kRl = self.substitute_stiffness(L-x, 'supported', 'rot')
            kNl = self.substitute_stiffness(L-x, 'supported', 'trans')
            c1 = 1/(8*bs)
            c2 = 1/(2*kRl)
            c3 = 1/(2*ss)
            c4 = 1/kNl
            c5 = -tc/qn
            return c1*x**4 + c2*x**3 + c3*x**2 + c4*x + c5

        # Find root
        a1 = brentq(polynomial, L/1000, 999/1000*L)

        return a1

    def calc_a2(self):
        """
        Calc transition lengths a2 (aBC).

        Returns
        -------
        a2 : float
            Length of the crack for transition of stage B to stage C (mm).
        """
        # Unpack variables
        bs = -(self.B11**2/self.A11 - self.D11)
        ss = self.kA55
        L = self.L
        tc = self.tc
        qn = self.calc_qn()

        # Create polynomial function
        def polynomial(x):
            # Spring stiffness supported segment
            kRl = self.substitute_stiffness(L-x, 'supported', 'rot')
            kNl = self.substitute_stiffness(L-x, 'supported', 'trans')
            c1 = ss**2*kRl*kNl*qn
            c2 = 6*ss**2*bs*kNl*qn
            c3 = 30*bs*ss*kRl*kNl*qn
            c4 = 24*bs*qn*(
                2*ss**2*kRl \
                + 3*bs*ss*kNl)
            c5 = 72*bs*(
                bs*qn*(
                    ss**2 \
                    + kRl*kNl) \
                - ss**2*kRl*kNl*tc)
            c6 = 144*bs*ss*(
                bs*kRl*qn \
                - bs*ss*kNl*tc)
            c7 = - 144*bs**2*ss*kRl*kNl*tc
            return c1*x**6 + c2*x**5 + c3*x**4 + c4*x**3 + c5*x**2 + c6*x + c7

        # Find root
        a2 = brentq(polynomial, L/1000, 999/1000*L)

        return a2

    def calc_lA(self):
        """
        Calculate the length of the touchdown element in mode A.
        """
        lA = self.a

        return lA

    def calc_lB(self):
        """
        Calculate the length of the touchdown element in mode B.
        """
        lB = self.a

        return lB

    def calc_lC(self):
        """
        Calculate the length of the touchdown element in mode C.
        """
        # Unpack variables
        bs = -(self.B11**2/self.A11 - self.D11)
        ss = self.kA55
        L = self.L
        a = self.a
        tc = self.tc
        qn = self.calc_qn()

        def polynomial(x):
            # Spring stiffness supported segment
            kRl = self.substitute_stiffness(L-a, 'supported', 'rot')
            kNl = self.substitute_stiffness(L-a, 'supported', 'trans')
            # Spring stiffness rested segment
            kRr = self.substitute_stiffness(a-x, 'rested', 'rot')
            # define constants
            c1 = ss**2*kRl*kNl*qn
            c2 = 6*ss*kNl*qn*(
                bs*ss \
                + kRl*kRr)
            c3 = 30*bs*ss*kNl*qn*(kRl + kRr)
            c4 = 24*bs*qn*(
                2*ss**2*kRl \
                + 3*bs*ss*kNl \
                + 3*kRl*kRr*kNl)
            c5 = 72*bs*(
                bs*qn*(
                    ss**2 \
                    + kNl*(kRl + kRr)) \
                + ss*kRl*(
                    2*kRr*qn \
                    - ss*kNl*tc))
            c6 = 144*bs*ss*(
                bs*qn*(kRl + kRr) \
                - kNl*tc*(
                    bs*ss \
                    + kRl*kRr))
            c7 = - 144*bs**2*ss*kNl*tc*(kRl + kRr)
            return c1*x**6 + c2*x**5 + c3*x**4 + c4*x**3 + c5*x**2 + c6*x + c7

        # Find root
        lC = brentq(polynomial, a/1000, 999/1000*a)

        return lC

    def set_touchdown_attributes(self, L, a, cf, phi, ratio):
        """Set class attributes for touchdown consideration"""
        self.set_columnlength(L)
        self.set_cracklength(a)
        self.set_tc(cf)
        self.set_phi(phi)
        self.set_stiffness_ratio(ratio)

    def calc_touchdown_mode(self):
        """Calculate touchdown-mode from thresholds"""
        if self.touchdown:
            # Calculate stage transitions
            a1 = self.calc_a1()
            a2 = self.calc_a2()
            # Assign stage
            if self.a <= a1:
                mode = 'A'
            elif a1 < self.a <= a2:
                mode = 'B'
            elif a2 < self.a:
                mode = 'C'
            self.mode = mode
        else:
            self.mode = 'A'

    def calc_touchdown_length(self):
        """Calculate touchdown length"""
        if self.mode in ['A']:
            self.td = self.calc_lA()
        elif self.mode in ['B']:
            self.td = self.calc_lB()
        elif self.mode in ['C']:
            self.td = self.calc_lC()

    def calc_touchdown_system(self, L, a, cf, phi, ratio=1000):
        """Calculate touchdown"""
        self.set_touchdown_attributes(L, a, cf, phi, ratio)
        self.calc_touchdown_mode()
        self.calc_touchdown_length()

class SolutionMixin:
    """
    Mixin for the solution of boundary value problems.

    Provides methods for the assembly of the system of equations
    and for the computation of the free constants.
    """

    def bc(self, z, k=False, pos='mid'):
        """
        Provide equations for free (pst) or infinite (skiers) ends.

        Arguments
        ---------
        z : ndarray
            Solution vector (6x1) at a certain position x.
        l : float, optional
            Length of the segment in consideration. Default is zero.
        k : boolean
            Indicates whether segment has foundation(True) or not (False).
            Default is False.
        pos : {'left', 'mid', 'right', 'l', 'm', 'r'}, optional
            Determines whether the segement under consideration
            is a left boundary segement (left, l), one of the
            center segement (mid, m), or a right boundary
            segement (right, r). Default is 'mid'.

        Returns
        -------
        bc : ndarray
            Boundary condition vector (lenght 3) at position x.
        """

        # Set boundary conditions for PST-systems
        if self.system in ['pst-', '-pst']:
            if not k:
                if self.mode in ['A']:
                    # Free end
                    bc = np.array([self.N(z),
                                   self.M(z),
                                   self.V(z)
                                   ])
                elif self.mode in ['B'] and pos in ['r', 'right']:
                    # Touchdown right
                    bc = np.array([self.N(z),
                                   self.M(z),
                                   self.w(z)
                                   ])
                elif self.mode in ['B'] and pos in ['l', 'left']:   # Kann dieser Block
                    # Touchdown left                                # verschwinden? Analog zu 'A'
                    bc = np.array([self.N(z),
                                   self.M(z),
                                   self.w(z)
                                   ])
                elif self.mode in ['C'] and pos in ['r', 'right']:
                    # Spring stiffness
                    kR = self.substitute_stiffness(self.a - self.td, 'rested', 'rot')
                    # Touchdown right
                    bc = np.array([self.N(z),
                                   self.M(z) + kR*self.psi(z),
                                   self.w(z)
                                   ])
                elif self.mode in ['C'] and pos in ['l', 'left']:
                    # Spring stiffness
                    kR = self.substitute_stiffness(self.a - self.td, 'rested', 'rot')
                    # Touchdown left
                    bc = np.array([self.N(z),
                                   self.M(z) - kR*self.psi(z),
                                   self.w(z)
                                   ])
            else:
                # Free end
                bc = np.array([
                    self.N(z),
                    self.M(z),
                    self.V(z)
                ])
        # Set boundary conditions for PST-systems with vertical faces
        elif self.system in ['-vpst', 'vpst-']:
            bc = np.array([
                self.N(z),
                self.M(z),
                self.V(z)
            ])
        # Set boundary conditions for SKIER-systems
        elif self.system in ['skier', 'skiers']:
            # Infinite end (vanishing complementary solution)
            bc = np.array([self.u(z, z0=0),
                           self.w(z),
                           self.psi(z)
                           ])
        # Set boundary conditions for substitute spring calculus
        elif self.system in ['rot', 'trans']:
            bc = np.array([self.N(z),
                            self.M(z),
                            self.V(z)
                            ])
        else:
            raise ValueError(
                'Boundary conditions not defined for'
                f'system of type {self.system}.')

        return bc

    def eqs(self, zl, zr, k=False, pos='mid'):
        """
        Provide boundary or transmission conditions for beam segments.

        Arguments
        ---------
        zl : ndarray
            Solution vector (6x1) at left end of beam segement.
        zr : ndarray
            Solution vector (6x1) at right end of beam segement.
        k : boolean
            Indicates whether segment has foundation(True) or not (False).
            Default is False.
        pos: {'left', 'mid', 'right', 'l', 'm', 'r'}, optional
            Determines whether the segement under consideration
            is a left boundary segement (left, l), one of the
            center segement (mid, m), or a right boundary
            segement (right, r). Default is 'mid'.

        Returns
        -------
        eqs : ndarray
            Vector (of length 9) of boundary conditions (3) and
            transmission conditions (6) for boundary segements
            or vector of transmission conditions (of length 6+6)
            for center segments.
        """
        if pos in ('l', 'left'):
            eqs = np.array([
                self.bc(zl, k, pos)[0],             # Left boundary condition
                self.bc(zl, k, pos)[1],             # Left boundary condition
                self.bc(zl, k, pos)[2],             # Left boundary condition
                self.u(zr, z0=0),           # ui(xi = li)
                self.w(zr),                 # wi(xi = li)
                self.psi(zr),               # psii(xi = li)
                self.N(zr),                 # Ni(xi = li)
                self.M(zr),                 # Mi(xi = li)
                self.V(zr)])                # Vi(xi = li)
        elif pos in ('m', 'mid'):
            eqs = np.array([
                -self.u(zl, z0=0),              # -ui(xi = 0)
                -self.w(zl),                    # -wi(xi = 0)
                -self.psi(zl),                  # -psii(xi = 0)
                -self.N(zl),                    # -Ni(xi = 0)
                -self.M(zl),                    # -Mi(xi = 0)
                -self.V(zl),                    # -Vi(xi = 0)
                self.u(zr, z0=0),               # ui(xi = li)
                self.w(zr),                     # wi(xi = li)
                self.psi(zr),                   # psii(xi = li)
                self.N(zr),                     # Ni(xi = li)
                self.M(zr),                     # Mi(xi = li)
                self.V(zr)])                    # Vi(xi = li)
        elif pos in ('r', 'right'):
            eqs = np.array([
                -self.u(zl, z0=0),          # -ui(xi = 0)
                -self.w(zl),                # -wi(xi = 0)
                -self.psi(zl),              # -psii(xi = 0)
                -self.N(zl),                # -Ni(xi = 0)
                -self.M(zl),                # -Mi(xi = 0)
                -self.V(zl),                # -Vi(xi = 0)
                self.bc(zr, k, pos)[0],             # Right boundary condition
                self.bc(zr, k, pos)[1],             # Right boundary condition
                self.bc(zr, k, pos)[2]])            # Right boundary condition
        else:
            raise ValueError(
                (f'Invalid position argument {pos} given. '
                 'Valid segment positions are l, m, and r, '
                 'or left, mid and right.'))
        return eqs

    def calc_segments(
            self,
            li: list[float] | list[int] |bool = False,
            mi: list[float] | list[int] |bool = False,
            ki: list[bool] | bool = False,
            k0: list[bool] | bool = False,
            L: float = 1e4,
            a: float = 0,
            m: float = 0,
            phi: float = 0,
            cf: float = 0.5,
            ratio: float = 1000,
            **kwargs):
        """
        Assemble lists defining the segments.

        This includes length (li), foundation (ki, k0), and skier
        weight (mi).

        Arguments
        ---------
        li : squence, optional
            List of lengths of segements(mm). Used for system 'skiers'.
        mi : squence, optional
            List of skier weigths (kg) at segement boundaries. Used for
            system 'skiers'.
        ki : squence, optional
            List of one bool per segement indicating whether segement
            has foundation (True) or not (False) in the cracked state.
            Used for system 'skiers'.
        k0 : squence, optional
            List of one bool per segement indicating whether segement
            has foundation(True) or not (False) in the uncracked state.
            Used for system 'skiers'.
        L : float, optional
            Total length of model (mm). Used for systems 'pst-', '-pst',
            'vpst-', '-vpst', and 'skier'.
        a : float, optional
            Crack length (mm). Used for systems 'pst-', '-pst',  'pst-',
            '-pst', and 'skier'.
        phi : float, optional
            Inclination (degree).
        m : float, optional
            Weight of skier (kg) in the axial center of the model.
            Used for system 'skier'.
        cf : float, optional
            Collapse factor. Ratio of the crack height to the uncollapsed
            weak-layer height. Used for systems 'pst-', '-pst'. Default is 0.5.
        ratio : float, optional
            Stiffness ratio between collapsed and uncollapsed weak layer.
            Default is 1000.

        Returns
        -------
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.
        """

        _ = kwargs                                      # Unused arguments

        # Precompute touchdown properties
        self.calc_touchdown_system(L=L, a=a, cf=cf, phi=phi, ratio=ratio)

        # Assemble list defining the segments
        if self.system == 'skiers':
            li = np.array(li)                           # Segment lengths
            mi = np.array(mi)                           # Skier weights
            ki = np.array(ki)                           # Crack
            k0 = np.array(k0)                           # No crack
        elif self.system == 'pst-':
            li = np.array([L - self.a, self.td])        # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([True, False])                # Crack
            k0 = np.array([True, True])                 # No crack
        elif self.system == '-pst':
            li = np.array([self.td, L - self.a])        # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([False, True])                # Crack
            k0 = np.array([True, True])                 # No crack
        elif self.system == 'vpst-':
            li = np.array([L - a, a])                   # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([True, False])                # Crack
            k0 = np.array([True, True])                 # No crack
        elif self.system == '-vpst':
            li = np.array([a, L - a])                   # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([False, True])                # Crack
            k0 = np.array([True, True])                 # No crack
        elif self.system == 'skier':
            lb = (L - self.a)/2                         # Half bedded length
            lf = self.a/2                               # Half free length
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
        phi : float
            Inclination (degrees).
        li : ndarray
            List of lengths of segements (mm).
        mi : ndarray
            List of skier weigths (kg) at segement boundaries.
        ki : ndarray
            List of one bool per segement indicating whether segement
            has foundation (True) or not (False).

        Returns
        -------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        """
        # --- CATCH ERRORS ----------------------------------------------------

        # No foundation
        if not any(ki):
            raise ValueError('Provide at least one supported segment.')
        # Mismatch of number of segements and transisions
        if len(li) != len(ki) or len(li) - 1 != len(mi):
            raise ValueError('Make sure len(li)=N, len(ki)=N, and '
                             'len(mi)=N-1 for a system of N segments.')

        if self.system not in ['pst-', '-pst', 'vpst-', '-vpst', 'rot', 'trans']:
            # Boundary segments must be on foundation for infinite BCs
            if not all([ki[0], ki[-1]]):
                raise ValueError('Provide supported boundary segments in '
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
                k=k, pos=pos)
            zpi = self.eqs(
                zl=self.zp(x=0, phi=phi, bed=k),
                zr=self.zp(x=l, phi=phi, bed=k),
                k=k, pos=pos)
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
        if self.system not in ['pst-', '-pst', 'rested']:
            rhs[:3] = self.bc(self.zp(x=0, phi=phi, bed=ki[0]))
            rhs[-3:] = self.bc(self.zp(x=li[-1], phi=phi, bed=ki[-1]))

        # Set rhs for vertical faces
        if self.system in ['vpst-', '-vpst']:
            # Calculate center of gravity and mass of
            # added or cut off slab segement
            xs, zs, m = calc_vertical_bc_center_of_gravity(self.slab, phi)
            # Convert slope angle to radians
            phi = np.deg2rad(phi)
            # Translate inbto section forces and moments
            N = -self.g*m*np.sin(phi)
            M = -self.g*m*(xs*np.cos(phi) + zs*np.sin(phi))
            V = self.g*m*np.cos(phi)
            # Add to right-hand side
            rhs[:3] = np.vstack([N, M, V])          # left end
            rhs[-3:] = np.vstack([N, M, V])         # right end

        # Loop through segments to set touchdown conditions at rhs
        for i in range(nS):
            # Length, foundation and position of segment i
            l, k, pos = li[i], ki[i], pi[i]
            # Set displacement BC in stage B
            if not k and bool(self.mode in ['B']):
                if i==0:
                    rhs[:3] = np.vstack([0,0,self.tc])
                if i == (nS - 1):
                    rhs[-3:] = np.vstack([0,0,self.tc])
            # Set normal force and displacement BC for stage C
            if not k and bool(self.mode in ['C']):
                N = self.calc_qt()*(self.a - self.td)
                if i==0:
                    rhs[:3] = np.vstack([-N,0,self.tc])
                if i == (nS - 1):
                    rhs[-3:] = np.vstack([N,0,self.tc])

        # Rhs for substitute spring stiffness
        if self.system in ['rot']:
            # apply arbitrary moment of 1 at left boundary
            rhs = rhs*0
            rhs[1] = 1
        if self.system in ['trans']:
            # apply arbitrary force of 1 at left boundary
            rhs = rhs*0
            rhs[2] = 1

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

    def rasterize_solution(
            self,
            C: np.ndarray,
            phi: float,
            li: list[float] | bool,
            ki: list[bool] | bool,
            num: int = 250,
            **kwargs):
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
        li = abs(li)
        isnonzero = li > 0
        C, ki, li = C[:, isnonzero], ki[isnonzero], li[isnonzero]

        # Compute number of plot points per segment (+1 for last segment)
        nq = np.ceil(li/li.sum()*num).astype('int')
        nq[-1] += 1

        # Provide cumulated length and plot point lists
        lic = np.insert(np.cumsum(li), 0, 0)
        nqc = np.insert(np.cumsum(nq), 0, 0)

        # Initialize arrays
        issupported = np.full(nq.sum(), True)
        xq = np.full(nq.sum(), np.nan)
        zq = np.full([6, xq.size], np.nan)

        # Loop through segments
        for i, l in enumerate(li):
            # Get local x-coordinates of segment i
            xi = np.linspace(0, l, num=nq[i], endpoint=(i == li.size - 1))  # pylint: disable=superfluous-parens
            # Compute start and end coordinates of segment i
            x0 = lic[i]
            # Assemble global coordinate vector
            xq[nqc[i]:nqc[i + 1]] = x0 + xi
            # Mask coordinates not on foundation (including endpoints)
            if not ki[i]:
                issupported[nqc[i]:nqc[i + 1]] = False
            # Compute segment solution
            zi = self.z(xi, C[:, [i]], l, phi, ki[i])
            # Assemble global solution matrix
            zq[:, nqc[i]:nqc[i + 1]] = zi

        # Make sure cracktips are included
        transmissionbool = [ki[j] or ki[j + 1] for j, _ in enumerate(ki[:-1])]
        for i, truefalse in enumerate(transmissionbool, start=1):
            issupported[nqc[i]] = truefalse

        # Assemble vector of coordinates on foundation
        xb = np.full(nq.sum(), np.nan)
        xb[issupported] = xq[issupported]

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

        # Identify supported-free and free-supported transitions as crack tips
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
            Gdif[1:, j] = np.concatenate((self.Gi(z, unit=unit), self.Gii(z, unit=unit)))

        # Sum mode I and II contributions
        Gdif[0, :] = Gdif[1, :] + Gdif[2, :]

        # Adjust contributions for center cracks
        if nct > 1:
            avgmask = np.full(nct, True)    # Initialize mask
            avgmask[[0, -1]] = ki[[0, -1]]  # Do not weight edge cracks
            Gdif[:, avgmask] *= 0.5         # Weigth with half crack length

        # Return total differential energy release rate of all crack tips
        return Gdif.sum(axis=1)

    def get_zmesh(self, dz=2):
        """
        Get z-coordinates of grid points and corresponding elastic properties.

        Arguments
        ---------
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.

        Returns
        -------
        mesh : ndarray
            Mesh along z-axis. Columns are a list of z-coordinates (mm) of
            grid points along z-axis with at least two grid points (top,
            bottom) per layer, Young's modulus of each grid point, shear
            modulus of each grid point, and Poisson's ratio of each grid
            point.
        """
        # Get ply (layer) coordinates
        z = self.get_ply_coordinates()
        # Compute number of grid points per layer
        nlayer = np.ceil((z[1:] - z[:-1])/dz).astype(np.int32) + 1
        # Calculate grid points as list of z-coordinates (mm)
        zi = np.hstack([
            np.linspace(z[i], z[i + 1], n, endpoint=True)
            for i, n in enumerate(nlayer)
        ])
        # Get lists of corresponding elastic properties (E, nu, rho)
        si = np.repeat(self.slab[:, [2, 4, 0]], nlayer, axis=0)
        # Assemble mesh with columns (z, E, G, nu)
        return np.column_stack([zi, si])

    def Sxx(self, Z, phi, dz=2, unit='kPa'):
        """
        Compute axial normal stress in slab layers.

        Arguments
        ----------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.

        Returns
        -------
        ndarray, float
            Axial slab normal stress in specified unit.
        """
        # Unit conversion dict
        convert = {
            'kPa': 1e3,
            'MPa': 1
        }

        # Get mesh along z-axis
        zmesh = self.get_zmesh(dz=dz)
        zi = zmesh[:, 0]
        rho = 1e-12*zmesh[:, 3]

        # Get dimensions of stress field (n rows, m columns)
        n = zmesh.shape[0]
        m = Z.shape[1]

        # Initialize axial normal stress Sxx
        Sxx = np.zeros(shape=[n, m])

        # Compute axial normal stress Sxx at grid points in MPa
        for i, (z, E, nu, _) in enumerate(zmesh):
            Sxx[i, :] = E/(1-nu**2)*self.du_dx(Z, z)

        # Calculate weight load at grid points and superimpose on stress field
        qt = -rho*self.g*np.sin(np.deg2rad(phi))
        for i, qi in enumerate(qt[:-1]):
            Sxx[i, :] += qi*(zi[i+1] - zi[i])
        Sxx[-1, :] += qt[-1]*(zi[-1] - zi[-2])

        # Return axial normal stress in specified unit
        return convert[unit]*Sxx

    def Txz(self, Z, phi, dz=2, unit='kPa'):
        """
        Compute shear stress in slab layers.

        Arguments
        ----------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.

        Returns
        -------
        ndarray
            Shear stress at grid points in the slab in specified unit.
        """
        # Unit conversion dict
        convert = {
            'kPa': 1e3,
            'MPa': 1
        }
        # Get mesh along z-axis
        zmesh = self.get_zmesh(dz=dz)
        zi = zmesh[:, 0]
        rho = 1e-12*zmesh[:, 3]

        # Get dimensions of stress field (n rows, m columns)
        n = zmesh.shape[0]
        m = Z.shape[1]

        # Get second derivatives of centerline displacement u0 and
        # cross-section rotaiton psi of all grid points along the x-axis
        du0_dxdx = self.du0_dxdx(Z, phi)
        dpsi_dxdx = self.dpsi_dxdx(Z, phi)

        # Initialize first derivative of axial normal stress sxx w.r.t. x
        dsxx_dx = np.zeros(shape=[n, m])

        # Calculate first derivative of sxx at z-grid points
        for i, (z, E, nu, _) in enumerate(zmesh):
            dsxx_dx[i, :] = E/(1-nu**2)*(du0_dxdx + z*dpsi_dxdx)

        # Calculate weight load at grid points
        qt = -rho*self.g*np.sin(np.deg2rad(phi))

        # Integrate -dsxx_dx along z and add cumulative weight load
        # to obtain shear stress Txz in MPa
        Txz = cumulative_trapezoid(dsxx_dx, zi, axis=0, initial=0)
        Txz += cumulative_trapezoid(qt, zi, initial=0)[:, None]

        # Return shear stress Txz in specified unit
        return convert[unit]*Txz

    def Szz(self, Z, phi, dz=2, unit='kPa'):
        """
        Compute transverse normal stress in slab layers.

        Arguments
        ----------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.

        Returns
        -------
        ndarray, float
            Transverse normal stress at grid points in the slab in
            specified unit.
        """
        # Unit conversion dict
        convert = {
            'kPa': 1e3,
            'MPa': 1
        }

        # Get mesh along z-axis
        zmesh = self.get_zmesh(dz=dz)
        zi = zmesh[:, 0]
        rho = 1e-12*zmesh[:, 3]

        # Get dimensions of stress field (n rows, m columns)
        n = zmesh.shape[0]
        m = Z.shape[1]

        # Get third derivatives of centerline displacement u0 and
        # cross-section rotaiton psi of all grid points along the x-axis
        du0_dxdxdx = self.du0_dxdxdx(Z, phi)
        dpsi_dxdxdx = self.dpsi_dxdxdx(Z, phi)

        # Initialize second derivative of axial normal stress sxx w.r.t. x
        dsxx_dxdx = np.zeros(shape=[n, m])

        # Calculate second derivative of sxx at z-grid points
        for i, (z, E, nu, _) in enumerate(zmesh):
            dsxx_dxdx[i, :] = E/(1-nu**2)*(du0_dxdxdx + z*dpsi_dxdxdx)

        # Calculate weight load at grid points
        qn = rho*self.g*np.cos(np.deg2rad(phi))

        # Integrate dsxx_dxdx twice along z to obtain transverse
        # normal stress Szz in MPa
        integrand = cumulative_trapezoid(dsxx_dxdx, zi, axis=0, initial=0)
        Szz = cumulative_trapezoid(integrand, zi, axis=0, initial=0)
        Szz += cumulative_trapezoid(-qn, zi, initial=0)[:, None]

        # Return shear stress txz in specified unit
        return convert[unit]*Szz

    def principal_stress_slab(self, Z, phi, dz=2, unit='kPa',
                              val='max', normalize=False):
        """
        Compute maxium or minimum principal stress in slab layers.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        phi : float
            Inclination (degrees). Counterclockwise positive.
        dz : float, optional
            Element size along z-axis (mm). Default is 2 mm.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.
        val : str, optional
            Maximum 'max' or minimum 'min' principal stress. Default is 'max'.
        normalize : bool
            Toggle layerwise normalization to strength.

        Returns
        -------
        ndarray
            Maximum or minimum principal stress in specified unit.

        Raises
        ------
        ValueError
            If specified principal stress component is neither 'max' nor
            'min', or if normalization of compressive principal stress
            is requested.
        """
        # Raise error if specified component is not available
        if val not in ['min', 'max']:
            raise ValueError(f'Component {val} not defined.')

        # Multiplier selection dict
        m = {'max': 1, 'min': -1}

        # Get axial normal stresses, shear stresses, transverse normal stresses
        Sxx = self.Sxx(Z=Z, phi=phi, dz=dz, unit=unit)
        Txz = self.Txz(Z=Z, phi=phi, dz=dz, unit=unit)
        Szz = self.Szz(Z=Z, phi=phi, dz=dz, unit=unit)

        # Calculate principal stress
        Ps = (Sxx + Szz)/2 + m[val]*np.sqrt((Sxx - Szz)**2 + 4*Txz**2)/2

        # Raise error if normalization of compressive stresses is attempted
        if normalize and val == 'min':
            raise ValueError('Can only normlize tensile stresses.')

        # Normalize tensile stresses to tensile strength
        if normalize and val == 'max':
            # Get layer densities
            rho = self.get_zmesh(dz=dz)[:, 3]
            # Normlize maximum principal stress to layers' tensile strength
            return Ps/tensile_strength_slab(rho, unit=unit)[:, None]

        # Return absolute principal stresses
        return Ps

    def principal_stress_weaklayer(self, Z, sc=2.6, unit='kPa', val='min',
                                   normalize=False):
        """
        Compute maxium or minimum principal stress in the weak layer.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x), psi'(x)]^T
        sc : float
            Weak-layer compressive strength. Default is 2.6 kPa.
        unit : {'kPa', 'MPa'}, optional
            Desired output unit. Default is 'kPa'.
        val : str, optional
            Maximum 'max' or minimum 'min' principal stress. Default is 'min'.
        normalize : bool
            Toggle layerwise normalization to strength.

        Returns
        -------
        ndarray
            Maximum or minimum principal stress in specified unit.

        Raises
        ------
        ValueError
            If specified principal stress component is neither 'max' nor
            'min', or if normalization of tensile principal stress
            is requested.
        """
        # Raise error if specified component is not available
        if val not in ['min', 'max']:
            raise ValueError(f'Component {val} not defined.')

        # Multiplier selection dict
        m = {'max': 1, 'min': -1}

        # Get weak-layer normal and shear stresses
        sig = self.sig(Z, unit=unit)
        tau = self.tau(Z, unit=unit)

        # Calculate principal stress
        ps = sig/2 + m[val]*np.sqrt(sig**2 + 4*tau**2)/2

        # Raise error if normalization of tensile stresses is attempted
        if normalize and val == 'max':
            raise ValueError('Can only normlize compressive stresses.')

        # Normalize compressive stresses to compressive strength
        if normalize and val == 'min':
            return ps/sc

        # Return absolute principal stresses
        return ps

class OutputMixin:
    """
    Mixin for outputs.

    Provides convenience methods for the assembly of output lists
    such as rasterized displacements or rasterized stresses.
    """
    def external_potential(self, C, phi, L, **segments):
        """
        Compute total external potential (pst only).

        Arguments
        ---------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        phi : float
            Inclination of the slab (°).
        L : float, optional
            Total length of model (mm).
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.

        Returns
        -------
        Pi_ext : float
            Total external potential (Nmm).
        """
        # Rasterize solution
        xq, zq, xb = self.rasterize_solution(C=C, phi=phi, **segments)
        _ = xq, xb
        # Compute displacements where weight loads are applied
        w0 = self.w(zq)
        us = self.u(zq, z0=self.zs)
        # Get weight loads
        qn = self.calc_qn()
        qt = self.calc_qt()
        # use +/- and us[0]/us[-1] according to system and phi
        # compute total external potential
        Pi_ext = - qn*(segments['li'][0] + segments['li'][1])*np.average(w0) \
            - qn*(L - (segments['li'][0] + segments['li'][1]))*self.tc
        # Ensure
        if self.system in ['pst-']:
            ub = us[-1]
        elif self.system in ['-pst']:
            ub = us[0]
        Pi_ext += - qt*(segments['li'][0] + segments['li'][1])*np.average(us) \
            - qt*(L - (segments['li'][0] + segments['li'][1]))*ub
        if self.system not in ['pst-', '-pst']:
            print('Input error: Only pst-setup implemented at the moment.')

        return Pi_ext

    def internal_potential(self, C, phi, L, **segments):
        """
        Compute total internal potential (pst only).

        Arguments
        ---------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        phi : float
            Inclination of the slab (°).
        L : float, optional
            Total length of model (mm).
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.

        Returns
        -------
        Pi_int : float
            Total internal potential (Nmm).
        """
        # Rasterize solution
        xq, zq, xb = self.rasterize_solution(C=C, phi=phi, **segments)

        # Compute section forces
        N, M, V = self.N(zq), self.M(zq), self.V(zq)

        # Drop parts of the solution that are not a foundation
        zweak = zq[:, ~np.isnan(xb)]
        xweak = xb[~np.isnan(xb)]

        # Compute weak layer displacements
        wweak = self.w(zweak)
        uweak = self.u(zweak, z0=self.h/2)

        # Compute stored energy of the slab (monte-carlo integration)
        n = len(xq)
        nweak = len(xweak)
        # energy share from moment, shear force, wl normal and tangential springs
        Pi_int = L/2/n/self.A11*np.sum([Ni**2 for Ni in N]) \
                + L/2/n/(self.D11-self.B11**2/self.A11)*np.sum([Mi**2 for Mi in M]) \
                + L/2/n/self.kA55*np.sum([Vi**2 for Vi in V]) \
                + L*self.kn/2/nweak*np.sum([wi**2 for wi in wweak]) \
                + L*self.kt/2/nweak*np.sum([ui**2 for ui in uweak])
        # energy share from substitute rotation spring
        if self.system in ['pst-']:
            Pi_int += 1/2*M[-1]*(self.psi(zq)[-1])**2
        elif self.system in ['-pst']:
            Pi_int += 1/2*M[0]*(self.psi(zq)[0])**2
        else:
            print('Input error: Only pst-setup implemented at the moment.')

        return Pi_int

    def total_potential(self, C, phi, L, **segments):
        """
        Returns total differential potential
    
        Arguments
        ---------
        C : ndarray
            Matrix(6xN) of solution constants for a system of N
            segements. Columns contain the 6 constants of each segement.
        phi : float
            Inclination of the slab (°).
        L : float, optional
            Total length of model (mm).
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.

        Returns
        -------
        Pi : float
            Total differential potential (Nmm).
        """
        Pi_int = self.internal_potential(C, phi, L, **segments)
        Pi_ext = self.external_potential(C, phi, L, **segments)

        return Pi_int + Pi_ext

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
