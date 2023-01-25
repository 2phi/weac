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


class SlabContactMixin:
    """
    Mixin for handling the touchdown situation in a PST.

    Provides Methods for the calculation of substitute spring stiffnesses,
    cracklength-tresholds and element lengths.
    """
    # pylint: disable=too-many-instance-attributes

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
            Collapse-factor. Ratio of the collapsed to the
            uncollapsed weak-layer height.
        """
        # subtract displacement under constact load from collapsed wl height
        qn = self.calc_qn()
        self.tc = cf*self.t - qn/self.kn

    def set_ratio(self,ratio):
        """
        Set the ratio of the stiffness of the collapsed
        to the uncollapsed weak-layer.

        Arguments
        ---------
        ratio : float
            Ratio of the stiffness of the collapsed to the
            uncollapsed weak-layer.
        """
        self.ratio = ratio

    def set_phi(self,phi):
        """
        Set inclination of the slab.

        Arguments
        ---------
        phi : float
            Inclination of the slab (°).
        """
        self.phi = phi

    def calc_beta(self):
        """
        Calc beta for collapsed (betaC) and uncollapsed (betaU) weak-layer.
        """
        # collapsed
        self.betaC = (self.ratio*self.kn/(4*self.D11))**(1/4)
        # uncollapsed
        self.betaU = (self.kn/(4*self.D11))**(1/4)

    def calc_qn(self):
        """
        Calc total surface normal load.

        Returns
        -------
        qn : float
            Total surface normal load (N/mm).
        """
        qn = self.get_weight_load(self.phi)[0] + self.get_surface_load(self.phi)[0]

        return qn

    def calc_qt(self):
        """
        Calc total surface normal load.

        Returns
        -------
        qn : float
            Total surface normal load (N/mm).
        """
        qt = self.get_weight_load(self.phi)[1] + self.get_surface_load(self.phi)[1]

        return qt

    def calc_trans_spring(self):
        """
        Calculate substitute translational spring stiffness from layer properties
        for uncollapsed and colapsed weak-layer.

        Returns
        -------
        list
            Translational spring stiffnesses (N/mm^2)
            for uncollapsed (zeroth entry) and uncollapsed weak-layer (first entry).
        """
        kNU = 4*self.D11*self.kA55*self.betaU**3/(2*self.kA55 + self.D11*self.betaU**2)
        kNC = 4*self.D11*self.kA55*self.betaC**3/(2*self.kA55 + self.D11*self.betaC**2)

        return [kNU,kNC]

    def calc_rot_spring(self):
        """
        Calculate substitute rotational spring stiffness from layer properties
        for uncollapsed and collapsed weak-layer.

        Returns
        -------
        list
            Rotational spring stiffnesses (Nmm/mm/rad)
            for uncollapsed (zeroth entry) and uncollapsed weak-layer (first entry).
        """
        kRU = 2*self.D11*self.kA55*self.betaU/(2*self.kA55 + self.D11*self.betaU**2)
        kRC = 2*self.D11*self.kA55*self.betaC/(2*self.kA55 + self.D11*self.betaC**2)

        return [kRU,kRC]

    def calc_a1(self):
        """
        Calculate cracklength where w(a) = tc.

        This is the longest crack, to which 'free end' conditions apply.
        It marks the threshold between Mode A and B.
        """
        def polynomial_4():
            """
            Calculate the coefficients of a fourth order polynomial equation.

            Returns
            -------
            list
                First coefficient for fourth order term,
                second coefficient for third order term and so on.
            """
            c1 = 1/(8*self.D11)*qn
            c2 = 1/(2*kRl)*qn
            c3 = 1/(2*self.kA55)*qn
            c4 = 1/kNl*qn
            c5 = -self.tc
            return [c1,c2,c3,c4,c5]

        # Get spring stiffnesses for adjacent segment with uncollapsed weak-layer
        kRl = self.calc_rot_spring()[0]
        kNl = self.calc_trans_spring()[0]
        # Get surface normal load components
        qn = self.calc_qn()
        # Calculate positive real roots
        pos = (np.roots(polynomial_4()).imag == 0) & (np.roots(polynomial_4()).real > 0)
        a1 = np.roots(polynomial_4())[pos].real[0]

        return a1

    def calc_a2(self):
        """
        Calculate cracklength where w(a) = tc and w'(a) = 0.

        This is the longest crack, to which M=0, N=0, w=tc boundary conditions apply.
        It marks the threshold between mode B and C.
        """
        def polynomial_6():
            """
            Calculate the coefficients of a sixth order polynomial equation.

            Returns
            -------
            list
                First coefficient for sixth order term,
                second coefficient for fith order term and so on.
            """
            c1 = self.kA55**2*kRl*kNl*qn
            c2 = 6*self.kA55**2*self.D11*kNl*qn
            c3 = 30*self.D11*self.kA55*kRl*kNl*qn
            c4 = 24*self.D11*qn*(
                2*self.kA55**2*kRl \
                + 3*self.D11*self.kA55*kNl)
            c5 = 72*self.D11*(
                self.D11*qn*(
                    self.kA55**2 \
                    + kRl*kNl) \
                - self.kA55**2*kRl*kNl*self.tc)
            c6 = 144*self.D11*self.kA55*(
                self.D11*kRl*qn \
                - self.D11*self.kA55*kNl*self.tc)
            c7 = - 144*self.D11**2*self.kA55*kRl*kNl*self.tc
            return [c1,c2,c3,c4,c5,c6,c7]

        # Get spring stiffnesses for adjacent segment with uncollapsed weak-layer
        kRl = self.calc_rot_spring()[0]
        kNl = self.calc_trans_spring()[0]
        # Get surface normal load components
        qn = self.calc_qn()
        # Calculate positive real roots
        pos = (np.roots(polynomial_6()).imag == 0) & (np.roots(polynomial_6()).real > 0)
        a2 = np.roots(polynomial_6())[pos].real[0]

        return a2

    def calc_a3(self):
        """
        Calculate cracklength w(a) = tc, w'(a) = 0 and kf = constant.

        This is the longest crack, to which M+-kf*kr*psi=0, N=0 and w=tc
        conditions apply. It marks the threshold between mode C and D.
        """
        def polynomial_6():
            """
            Calculate the coefficients of a sixth order polynomial equation.

            Returns
            -------
            list
                First coefficient for sixth order term,
                second coefficient for fith order term and so on.
            """
            c1 = self.kA55**2*kRl*kNl*qn
            c2 = 6*self.kA55*kNl*qn*(
                self.D11*self.kA55 \
                + kRl*kRr)
            c3 = 30*self.D11*self.kA55*kNl*qn*(kRl + kRr)
            c4 = 24*self.D11*qn*(
                2*self.kA55**2*kRl \
                + 3*self.D11*self.kA55*kNl \
                + 3*kRl*kRr*kNl)
            c5 = 72*self.D11*(
                self.D11*qn*(
                    self.kA55**2 \
                    + kNl*(kRl + kRr)) \
                + self.kA55*kRl*(
                    2*kRr*qn \
                    - self.kA55*kNl*self.tc))
            c6 = 144*self.D11*self.kA55*(
                self.D11*qn*(kRl + kRr) \
                - kNl*self.tc*(
                    self.D11*self.kA55 \
                    + kRl*kRr))
            c7 = - 144*self.D11**2*self.kA55*kNl*self.tc*(kRl + kRr)
            return [c1,c2,c3,c4,c5,c6,c7]

        # Get spring stiffnesses for adjacent segment with uncollapsed weak-layer
        kRl = self.calc_rot_spring()[0]
        kNl = self.calc_trans_spring()[0]
        # Get spring stiffnesses for adjacent segment with collapsed weak-layer
        kRr = self.calc_rot_spring()[1]
        # Get surface normal load components
        qn = self.calc_qn()
        # Calculate positive real roots
        pos = (np.roots(polynomial_6()).imag == 0) & (np.roots(polynomial_6()).real > 0)
        a3 = np.roots(polynomial_6())[pos].real[0] + np.pi/self.betaC

        return a3

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
        def polynomial_8():
            """
            Calculate the coefficients of a eighth order polynomial equation.

            Returns
            -------
            list
                First coefficient for eighth order term,
                second coefficient for seventh order term and so on.
            """
            c1 = -12/np.pi**3*self.kA55*kNl*kRl*kRr*qn*self.betaC**3
            c2 = -1/np.pi**3*self.kA55*kNl*kRr*qn*(
                    18*kRl*np.pi*self.betaC**2 \
                    + 12*self.betaC**3*(
                        5*self.D11 \
                        - 3*self.a*kRl))
            c3 = -1/np.pi**3*kNl*qn*(
                    self.kA55**2*kRl*np.pi**3 \
                    + 18*self.kA55*np.pi*kRr*self.betaC**2*(
                        5*self.D11 \
                        - 2*self.a*kRl) \
                    + 36*kRr*self.betaC**3*(
                        -5*self.a*self.D11*self.kA55 \
                        + 4*self.D11*kRl \
                        + self.a**2*self.kA55*kRl))
            c4 = 6/np.pi**3*qn*(
                    -self.D11*self.kA55**2*kNl*np.pi**3 \
                    - 3*kNl*kRr*np.pi*self.betaC**2*(
                        -10*self.a*self.D11*self.kA55 \
                        + 12*self.D11*kRl \
                        + self.a**2*self.kA55*kRl) \
                    + 2*kRr*self.betaC**3*(
                        -3*self.D11*kNl*(
                            4*self.D11 \
                            + 5*self.a**2*self.kA55) \
                        + kRl*(
                            -24*self.D11*self.kA55 \
                            + 36*self.a*self.D11*kNl \
                            + self.a**3*self.kA55*kNl)))
            c5 = 6/np.pi**3*self.D11*(
                    -5*self.kA55*kNl*kRl*np.pi**3*qn \
                    - 3*kRr*np.pi*qn*self.betaC**2*(
                        12*self.D11*kNl \
                        + 5*self.a**2*self.kA55*kNl \
                        + 24*kRl*(self.kA55 - self.a*kNl)) \
                    + 2*kRr*self.betaC**3*(
                        12*self.D11*qn*(
                            -2*self.kA55 \
                            + 3*self.a*kNl) \
                        + self.a*qn*(
                            5*self.a**2*self.kA55*kNl \
                            + 72*self.kA55*kRl \
                            - 36*self.a*kNl*kRl) \
                        + 24*self.kA55*kNl*kRl*self.tc))
            c6 = 24/np.pi**3*self.D11*(
                    -self.kA55*np.pi**3*qn*(
                        3*self.D11*kNl \
                        + 2*self.kA55*kRl) \
                    - 9*kRr*np.pi*self.betaC**2*(
                        2*self.D11*qn*(
                            self.kA55 \
                            - self.a*kNl) \
                        + self.a*kRl*qn*(
                            -4*self.kA55 \
                            + self.a*kNl) \
                        - 2*self.kA55*kNl*kRl*self.tc) \
                    + 6*kRr*self.betaC**3*(
                        self.a*qn*(
                            self.D11*(6*self.kA55 - 3*self.a*kNl) \
                            + self.a*kRl*(-6*self.kA55 + self.a*kNl)) \
                        + 2*self.kA55*kNl*self.tc*(
                            self.D11 \
                            - 3*self.a*kRl)))
            c7 = 72/np.pi**3*self.D11*(
                    np.pi**3*(
                        -self.D11*qn*(
                            self.kA55**2 \
                            + kNl*kRl) \
                        + self.kA55**2*kNl*kRl*self.tc) \
                    - 3*kRr*np.pi*self.betaC**2*(
                        self.a*qn*(
                            -4*self.D11*self.kA55 \
                            + self.a*self.D11*kNl \
                            + 2*self.a*self.kA55*kRl) \
                        - 2*self.kA55*kNl*self.tc*(
                            self.D11 \
                            - 2*self.a*kRl)) \
                    + 2*self.a*kRr*self.betaC**3*(
                        self.a*qn*(
                            -6*self.D11*self.kA55 \
                            + self.a*self.D11*kNl \
                            + 2*self.a*self.kA55*kRl) \
                        - 6*self.kA55*kNl*self.tc*(
                            self.D11 \
                            - self.a*kRl)))
            c8 = 144/np.pi**3*self.D11*self.kA55*(
                    self.D11*np.pi**3*(
                        -kRl*qn \
                        + self.kA55*kNl*self.tc) \
                    - 3*self.a*kRr*np.pi*self.betaC**2*(
                        self.a*self.D11*qn \
                        + 2*self.D11*kNl*self.tc \
                        - self.a*kNl*kRl*self.tc) \
                    + 2*self.a**2*kRr*self.betaC**3*(
                        self.a*self.D11*qn \
                        + 3*self.D11*kNl*self.tc \
                        - self.a*kNl*kRl*self.tc))
            c9 = 144/np.pi**3*self.D11**2*self.kA55*kNl*self.tc*(
                    kRl*np.pi**3 \
                    + self.a**2*kRr*self.betaC**2*(
                        3*np.pi \
                        - 2*self.a*self.betaC))
            return [c1,c2,c3,c4,c5,c6,c7,c8,c9]

        # Get spring stiffnesses for adjacent segment with uncollapsed weak-layer
        kRl = self.calc_rot_spring()[0]
        kNl = self.calc_trans_spring()[0]
        # Get spring stiffnesses for adjacent segment with collapsed weak-layer
        kRr = self.calc_rot_spring()[1]
        # Get surface normal load components
        qn = self.calc_qn()
        # Calculate positive real roots
        pos = (np.roots(polynomial_8()).imag == 0) & (np.roots(polynomial_8()).real > 0)
        lC = np.roots(polynomial_8())[pos].real[0]

        return lC

    def calc_lD(self):
        """
        Calculate the length of the touchdown element in mode D.
        """
        lD = self.calc_a3() - np.pi/self.betaC

        return lD

    def set_touchdown_attributes(self,a,cf,ratio,phi):
        """Set class attributes for touchdown consideration"""
        self.set_cracklength(a)
        self.set_tc(cf)
        self.set_ratio(ratio)
        self.set_phi(phi)
        self.calc_beta()

    def calc_touchdown_mode(self):
        """Calculate touchdown-mode from thresholds"""
        a1 = self.calc_a1()
        a2 = self.calc_a2()
        a3 = self.calc_a3()
        if self.a <= a1:
            mode = 'A'
        elif a1 < self.a <= a2:
            mode = 'B'
        elif a2 < self.a <= a3:
            mode = 'C'
        elif a3 < self.a:
            mode = 'D'
        self.mode = mode
        #print(a1, a2, a3)

    def calc_touchdown_length(self):
        """Calculate touchdown length"""
        if self.mode in ['A']:
            self.td = self.calc_lA()
        elif self.mode in ['B']:
            self.td = self.calc_lB()
        elif self.mode in ['C']:
            self.td = self.calc_lC()
        elif self.mode in ['D']:
            self.td = self.calc_lD()
        #print('td', self.td)

    def calc_touchdown_system(self,a,cf,ratio,phi):
        """Calculate touchdown"""
        self.set_touchdown_attributes(a,cf,ratio,phi)
        self.calc_touchdown_mode()
        self.calc_touchdown_length()


class SolutionMixin:
    """
    Mixin for the solution of boundary value problems.

    Provides methods for the assembly of the system of equations
    and for the computation of the free constants.
    """

    def reduce_stiffness(self):
        """
        Determines the reduction factor for a rotational spring.

        Arguments
        ---------
        Returns
        -------
        kf : float
            Reduction factor.
        """
        kf = - 2*(self.betaC*(self.a-self.td)/np.pi)**3 \
            + 3*(self.betaC*(self.a-self.td)/np.pi)**2

        return kf

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
                    kR = self.reduce_stiffness() * self.calc_rot_spring()[1]
                    # Touchdown right
                    bc = np.array([self.N(z),
                                   self.M(z) + kR*self.psi(z),
                                   self.w(z)
                                   ])
                elif self.mode in ['C'] and pos in ['l', 'left']:
                    # Spring stiffness
                    kR = self.reduce_stiffness() * self.calc_rot_spring()[1]
                    # Touchdown left
                    bc = np.array([self.N(z),
                                   self.M(z) - kR*self.psi(z),
                                   self.w(z)
                                   ])
                elif self.mode in ['D'] and pos in ['r', 'right']:
                    # Spring stiffness
                    kR = self.calc_rot_spring()[1]
                    # Touchdown right
                    bc = np.array([self.N(z),
                                   self.M(z) + kR*self.psi(z),
                                   self.w(z)
                                   ])
                elif self.mode in ['D'] and pos in ['l', 'left']:
                    # Spring stiffness
                    kR = self.calc_rot_spring()[1]
                    # Touchdown left
                    bc = np.array([self.N(z),
                                   self.M(z) - kR*self.psi(z),
                                   self.w(z)
                                   ])
            else:
                # Free end
                bc = np.array([self.N(z),
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
                self.bc(zr, k, pos)[0],             # Right boundary condition
                self.bc(zr, k, pos)[1],             # Right boundary condition
                self.bc(zr, k, pos)[2]])            # Right boundary condition
        else:
            raise ValueError(
                (f'Invalid position argument {pos} given. '
                 'Valid segment positions are l, m, and r, '
                 'or left, mid and right.'))
        return eqs

    def calc_segments(self, li=False, mi=False, ki=False, k0=False,
                      L=1e4, m=0, **kwargs):
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
            and 'skier'.
        a : float, optional
            Crack length (mm).  Used for systems 'pst-', '-pst', and
            'skier'.
        phi : float, optional
            Inclination (degree).
        m : float, optional
            Weight of skier (kg) in the axial center of the model.
            Used for system 'skier'.

        Returns
        -------
        segments : dict
            Dictionary with lists of touchdown booleans (tdi), segement
            lengths (li), skier weights (mi), and foundation booleans
            in the cracked (ki) and uncracked (k0) configurations.
        """

        _ = kwargs                                      # Unused arguments

        # Assemble list defining the segments
        if self.system == 'skiers':
            li = np.array(li)                           # Segment lengths
            mi = np.array(mi)                           # Skier weights
            ki = np.array(ki)                           # Crack
            k0 = np.array(k0)                           # No crack
        elif self.system == 'pst-':
            li = np.array([L - self.a, self.td])             # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([True, False])                # Crack
            k0 = np.array([True, True])                 # No crack
        elif self.system == '-pst':
            li = np.array([self.td, L - self.a])             # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([False, True])                # Crack
            k0 = np.array([True, True])                 # No crack
        elif self.system == 'skier':
            lb = (L - self.a)/2                              # Half bedded length
            lf = self.a/2                                    # Half free length
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
            'both': {'li': li, 'mi': mi, 'ki': ki}}
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
            raise ValueError('Provide at least one bedded segment.')
        # Mismatch of number of segements and transisions
        if len(li) != len(ki) or len(li) - 1 != len(mi):
            raise ValueError('Make sure len(li)=N, len(ki)=N, and '
                             'len(mi)=N-1 for a system of N segments.')

        if self.system not in ['pst-', '-pst']:
            # Boundary segments must be on foundation for infinite BCs
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
        if self.system not in ['pst-', '-pst']:
            rhs[:3] = self.bc(self.zp(x=0, phi=phi, bed=ki[0]))
            rhs[-3:] = self.bc(self.zp(x=li[-1], phi=phi, bed=ki[-1]))

        # Loop through segments to set touchdown conditions at rhs
        for i in range(nS):
            # Length, foundation and position of segment i
            l, k, pos = li[i], ki[i], pi[i]
            # Set displacement BC in stages B and C
            if not k and bool(self.mode in ['B', 'C']):
                if i==0:
                    rhs[:3] = np.vstack([0,0,self.tc])
                if i == (nS - 1):
                    rhs[-3:] = np.vstack([0,0,self.tc])
            # Set normal force and displacement BC for stage D
            if not k and bool(self.mode in ['D']):
                N = self.calc_qt()*(self.a-self.td)
                if i==0:
                    rhs[:3] = np.vstack([-N,0,self.tc])
                if i == (nS - 1):
                    rhs[-3:] = np.vstack([N,0,self.tc])

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
                + L/2/n/self.D11*np.sum([Mi**2 for Mi in M]) \
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
