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
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.
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
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.

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
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x)  phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.
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
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.

        Returns
        -------
        float
            First derivative psi' of the midplane rotation (radians/mm)
             of the slab.
        """
        return Z[5, :]

    # pylint: enable=no-self-use
    def u(self, Z, z0, bed = False,unit='mm' ):
        """
        Get horizontal displacement u = u0 + z0 psi for unbedded segments or
        u = u0 + (z0 + (h+t)/2) psi for bedded ones respectively.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.
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
        if not type(z0) == np.ndarray:
            if not bed:
                u =  convert[unit]*(Z[0,:] + z0 *self.psi(Z))
            else:
                u = convert[unit]*(Z[0, :] + (z0 + (self.t )/2) * Z[4,:])
        else:
            u =  convert[unit]*(Z[0,:] + z0 *self.psi(Z))
            u[~np.isnan(Z[-1,:])] = convert[unit]*(Z[0, ~np.isnan(Z[-1,:])] + (z0[~np.isnan(Z[-1,:])] + (self.t )/2) * Z[4,~np.isnan(Z[-1,:])])
        return u

    def up(self, Z, z0):
        """
        Get first derivative of the horizontal displacement.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.
        z0 : float
            Z-coordinate (mm) where u is to be evaluated.

        Returns
        -------
        float
            First derivative u' = u0' + z0 psi' of the horizontal
            displacement of the slab.
        """
        if np.reshape(Z[~np.isnan(Z)],Z.shape).shape[0]==10:
            return (Z[1, :] + (z0 + (self.t )/2) *self.psip(Z))
        else:
            return (Z[1, :] + z0 * self.psip(Z))


    def phi_u(self, Z):
        """
        Unpack amplitude of vertical cosine shaped displacements in the weak layer phi_u 

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phi_u(x) phi_u'(x) phi_w(x) phi_w'(x)]^T.
        

        Returns
        -------
        float
            Amplitude phi_u (mm) of the slab.
        """
        return Z[6,:]

    def phi_w(self, Z):
        """
        Unpack amplitude of horizontal cosine shaped displacements in the weak layer phi_w 

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.
        

        Returns
        -------
        float
            Amplitude phi_w (mm) of the slab.
        """
        return Z[8,:]


    def uweak(self, Z, z0, unit='mm'):
        """
        Get horizontal displacement uweak = u0 * (1/2 - z/t) + cos(pi * z/t) phi_u 

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.
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
        return convert[unit]* (Z[0, :] * (1/2 - z0/self.t) + np.cos( np.pi * z0 / self.t ) * Z[6, :] )
         
    def N(self, Z,bed = False):
        """
        Get the axial normal force N = A11 u' + B11 psi' in the unbedded slab and
        the axial force NSlab = A (u' + (t+h)/2 psi') + B psi' in bedded segments. 

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.

        Returns
        -------
        float
            Axial normal force N (N) in the slab.
        """
        if bed:
            Ew = self.weak['E']
            nuw = self.weak['nu']
            t = self.t
            Pi = np.pi

            return self.A * (Z[1,:] + (self.t )/2. * Z[5,:] ) + self.B * Z[5,:] + Ew * (-2*t*(Pi * Z[1,:] + 3 * Z[7,:]) + nuw * (3 * Pi * Z[2,:] - 12 * Z[8,:] + 2 * Pi * t * Z[1,:] + 6 * t * Z[7,:]))/(6 * Pi * (-1 + nuw + 2 * nuw**2))
        else:
            return self.A11*Z[1, :] + self.B11*Z[5, :]

    def M(self, Z):
        """
        Get bending moment M = B11 u' + D11 psi' in the slab, and
        the bending moment MSlab = B (u' + (t+h)/2 psi') + C psi' in bedded segments.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.

        Returns
        -------
        float
            Bending moment M (Nmm) in the slab.
        """
        if np.reshape(Z[~np.isnan(Z)],Z.shape).shape[0]==10:
            return self.B * (Z[1,:] + (self.t)/2. * Z[5,:] ) + self.C * Z[5,:]
        else:
            return self.B11*Z[1, :] + self.D11*Z[5, :]
        

    def V(self, Z,bed = False):
        """
        Get vertical shear force V = kA55(w' + psi), or the vertical shear force
        VSlab = D (psi + w') in bedded segments, respectively.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.

        Returns
        -------
        float
            Vertical shear force V (N) in the slab.
        """
        if bed:
            Ew = self.weak['E']
            nuw = self.weak['nu']
            t = self.t
            Pi = np.pi

            return self.D*(Z[3, :] + Z[4, :]) + Ew * (-3 * Pi * Z[0,:] + 12 * Z[6,:] + 2*Pi * t * Z[3,:] + 6 * t* Z[9,:])/(12 * Pi * (1 + nuw))
        else:
            return self.kA55*(Z[3, :] + Z[4, :])
        

    def NWeak(self, Z):
        """
        Get the work of the normal forces NWeak in the weak layer, given by
        Integrate[sig_xx(x,z) * cos(Pi * z/t) ,{z,-t/2,t/2}]

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.

        Returns
        -------
        float
            Normal force NWeak (N) in the weak layer.
        """    

        # Unpack weak layer material properties
        Ew = self.weak['E']
        nuw = self.weak['nu']
        t = self.t

        return Ew *(4 * nuw * Z[2,:] + 2 * t * ( -1 + nuw) * Z[1,:] + np.pi * t * (-1 + nuw) * Z[7,:])/(2 * np.pi * ( 1+ nuw) * (-1 + 2 * nuw))


    def VWeak(self, Z):
        """
        Get the work of the tangential forces VWeak in the weak layer, given by
        Integrate[tau_xz(x,z) * cos(Pi * z/t) ,{z,-t/2,t/2}]


        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.

        Returns
        -------
        float
            Vertical force VWeak (N) in the weak layer.
        """    

        # Unpack weak layer material properties
        Ew = self.weak['E']
        nuw = self.weak['nu']
        t = self.t

        return  Ew * (- 4 * Z[0,:] + 2 * t * Z[3,:] + np.pi *t* Z[9,:])/(4*np.pi * ( 1 + nuw))


    def sig(self, Z, unit='MPa', z = 0):
        """
        Get weak-layer normal stress.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T.
        unit : {'MPa', 'kPa'}, optional
            Desired output unit. Default is MPa.
        z : float, optional
            Z-coordinate (mm) where u is to be evaluated. Default is 0.

        Returns
        -------
        float
            Weak-layer normal stress sigma (in specified unit).
        """
        convert = {
            'kPa': 1e3,
            'MPa': 1
        }
        Ew = self.weak['E']
        nuw = self.weak['nu']

        t = self.t
        Pi = np.pi
        return -convert[unit]*Ew *(2 * (-1 + nuw) * Z[2,:] + 2 * Pi * np.sin(Pi * z/t)*(-1 + nuw)*Z[8,:] + nuw *((t-2*z)*Z[1,:]+ 2*t*np.cos(Pi * z/t) * Z[7,:]) )/(2*t*(1+nuw) * (-1 + 2*nuw))

    def tau(self, Z, unit='MPa',z0 = 0):
        """
        Get weak-layer shear stress.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phi_u phi_u' phi_w phi_w']^T.
        unit : {'MPa', 'kPa'}, optional
            Desired output unit. Default is MPa.
        z0 : float, optional
            Z-coordinate (mm) where u is to be evaluated. Default is 0.

        Returns
        -------
        float
            Weak-layer shear stress tau (in specified unit).
        """
        convert = {
            'kPa': 1e3,
            'MPa': 1
        }
        Ew = self.weak['E']
        nuw = self.weak['nu']

        t = self.t
        Pi = np.pi
        return convert[unit]*Ew *(-2 * Z[0,:] -2 * Pi * np.sin(Pi * z0/t) * Z[6,:] + t * Z[3,:] - 2*z0 *Z[3,:] + 2*t * np.cos(Pi * z0/t)*Z[9,:])/(4*t*(1+nuw))

    def eps(self, Z, z0 = 0):
        """
        Get weak-layer normal, vertical strain.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phi_u phi_u' phi_w phi_w']^T.
        z0 : float, optional
            Z-coordinate (mm) where u is to be evaluated. Default is 0.

        Returns
        -------
        float
            Weak-layer normal strain epsilon.
        """
        return -self.w(Z)/self.t - np.pi * np.sin(np.pi * z0 /self.t) * Z[8,:]/self.t

    def gamma(self, Z, z0 = 0):
        """
        Get weak-layer shear strain.

        Arguments
        ---------
        Z : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phi_u phi_u' phi_w phi_w']^T.
        z0 : float, optional
            Z-coordinate (mm) where u is to be evaluated. Default is 0.

        Returns
        -------
        float
            Weak-layer shear strain gamma.
        """
        return Z[0,:] / self.t - np.pi / self.t * np.sin(np.pi * z0 / self.t) * Z[6,:] + Z[3,:]/2 - z0 * Z[3,:] / self.t + np.cos(np.pi * z0 / self.t) * Z[9,:]

    # def maxp(self, Z, z0 = 0): ### Not necessary in WEAC 3.0
    #     """
    #     Get maximum principal stress in the weak layer.

    #     Arguments
    #     ---------
    #     Z : ndarray
    #         Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phi_u phi_u' phi_w phi_w']^T.
    #     z0 : float, optional
    #         Z-coordinate (mm) where u is to be evaluated. Default is 0.
    #     Returns
    #     -------
    #     loat
    #         Maximum principal stress (MPa) in the weak layer.
    #     """
    #     sig = self.sig(Z, z0)
    #     tau = self.tau(Z, z0)
    #     return np.amax([[sig + np.sqrt(sig**2 + 4*tau**2),
    #                      sig - np.sqrt(sig**2 + 4*tau**2)]], axis=1)[0]/2

    def Gi(self, Ztip, Zback, unit='kJ/m^2'):
        """
        Get mode I differential energy release rate at crack tip.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T
            at the crack tip.
        Zback : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T
            at the opposite side of the bedded segement.
        unit : {'N/mm', 'kJ/m^2', 'J/m^2'}, optional
            Desired output unit. Default is kJ/m^2.

        Returns
        -------
        float
            Mode I differential energy release rate (N/mm = kJ/m^2
            or J/m^2) at the crack tip.
            Derivation of ERR is based on the J-integral

        """

        Ew = self.weak['E']
        nuw = self.weak['nu']
        rhow = self.weak['rho']
        t = self.t
        Pi = np.pi

        convert = {
            'J/m^2': 1e3,   # joule per square meter
            'kJ/m^2': 1,    # kilojoule per square meter
            'N/mm': 1       # newton per millimeter
        }
        return convert[unit]*(Ew *(2*Pi * (-1 + nuw) * Ztip[2]**2 + Ztip[8]*(Pi**3*(-1 + nuw)*Ztip[8]-4*t*nuw*Ztip[1])+ t*nuw*Ztip[2]*(Pi*Ztip[1]+ 4 * Ztip[7])) \
            /(4*Pi*t*(-1 + nuw+2*nuw**2)) + self.g * rhow * t * np.cos(np.deg2rad(self.phi))/2*Pi *(Pi * Ztip[2] + 4 * Ztip[8]) - self.g * rhow * t * np.cos(np.deg2rad(self.phi))/2*Pi *(Pi * Zback[2] + 4 * Zback[8]))

    def Gii(self, Ztip, Zback, unit='kJ/m^2'):
        """
        Get mode II differential energy release rate at crack tip.

        Arguments
        ---------
        Ztip : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T
            at the crack tip.
        Zback : ndarray
            Solution vector [u(x) u'(x) w(x) w'(x) psi(x) psi'(x) phiu(x) phiu'(x) phiw(x) phiw'(x)]^T
            at the opposite side of the bedded segement.
        unit : {'N/mm', 'kJ/m^2', 'J/m^2'}, optional
            Desired output unit. Default is kJ/m^2 = N/mm.

        Returns
        -------
        float
            Mode II differential energy release rate (N/mm = kJ/m^2
            or J/m^2) at the crack tip.
        """

        Ew = self.weak['E']
        nuw = self.weak['nu']
        rhow= self.weak['rho']
        t = self.t
        Pi = np.pi


        convert = {
            'J/m^2': 1e3,   # joule per square meter
            'kJ/m^2': 1,    # kilojoule per square meter
            'N/mm': 1       # newton per millimeter
        }
        return convert[unit]*(Ew *(6 * Pi * Ztip[0]**2+ 3 * Pi**3 * Ztip[6]**2+ 24 * t * Ztip[6]*Ztip[3]- 6 * t * Ztip[0] *(Pi * Ztip[3] + 4 * Ztip[9]) + t**2 * ( 2 * Pi * Ztip[3]**2 + 12 * Ztip[3]*Ztip[9] + 3 * Pi * Ztip[9]**2))\
            /(24 * Pi * t * (1 + nuw)) - self.g * rhow * t * np.sin(np.deg2rad(self.phi))/2*Pi *(Pi * Ztip[0] + 4 * Ztip[6]) +self.g * rhow * t * np.sin(np.deg2rad(self.phi))/2*Pi *(Pi * Zback[0] + 4 * Zback[6]))


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


class SolutionMixin:
    """
    Mixin for the solution of boundary value problems.
    Still in need of transition to WEAC3.

    Provides methods for the assembly of the system of equations
    and for the computation of the free constants.
    """

    def mode_td(self, l=0):
        """
        Identify the mode of the pst-boundary.

        Arguments
        ---------
        l : float, optional
            Length of the segment in consideration. Default is zero.

        Returns
        -------
        mode : string
            Contains the mode for the boundary of the segment:
            A - free end, B - intermediate touchdown,
            C - full touchdown (maximum clamped end).
        """
        # Classify boundary type by element length
        if l <= self.lC:
            mode = 'A'
        elif self.lC < l <= self.lS:
            mode = 'B'
        elif self.lS < l:
            mode = 'C'

        return mode

    def reduce_stiffness(self, l=0, mode='A'):
        """
        Determines the reduction factor for a rotational spring.

        Arguments
        ---------
        l : float, optional
            Length of the segment in consideration. Default is zero.
        mode : string, optional
            Contains the mode for the boundary of the segment:
            A - free end, B - intermediate touchdown, C - full touchdown.
            Default is A.

        Returns
        -------
        kf : float
            Reduction factor.
        """
        # Reduction to zero for free end bc
        if mode in ['A']:
            kf = 0
        # Reduction factor for touchdown
        if mode in ['B', 'C']:
            l = l - self.lC
            # Beta needs to take into account different weak-layer spring stiffness
            beta = self.beta*self.ratio**(1/4)
            kf=(np.cos(2*beta*l)+np.cosh(2*beta*l)-2)/(np.sin(2*beta*l)+np.sinh(2*beta*l))

        return kf

    def bc(self, z, l=0, k=False, pos='mid'):
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
        # Check mode for free end
        mode = self.mode_td(l=l)
        # Get spring stiffness reduction factor
        kf = self.reduce_stiffness(l=l, mode=mode)
        # Get spring stiffness for collapsed weak-layer
        kR = self.calc_rot_spring(collapse=True)

        # Set boundary conditions for PST-systems
        if self.system in ['pst-', '-pst']:
            if not k:
                if mode in ['A']:
                    # Free end
                    bc = np.array([self.N(z),
                                   self.M(z),
                                   self.V(z)
                                   ])
                elif mode in ['B', 'C'] and pos in ['r', 'right']:
                    # Touchdown right
                    bc = np.array([self.N(z),
                                   self.M(z) + kf*kR*self.psi(z),
                                   self.w(z)
                                   ])
                elif mode in ['B', 'C'] and pos in ['l', 'left']:
                    # Touchdown left
                    bc = np.array([self.N(z),
                                   self.M(z) - kf*kR*self.psi(z),
                                   self.w(z)
                                   ])
            else:
                # Free end
                bc = np.array([self.N(z, bed = True),
                                self.M(z),
                                self.V(z, bed = True),
                                self.NWeak(z),
                                self.VWeak(z)
                                ])
        # Set boundary conditions for SKIER-systems
        elif self.system in ['skier', 'skiers']:
            # Infinite end (vanishing complementary solution)
            if not k:
                bc = np.array([self.u(z, z0=0, bed = True),
                           self.w(z),
                           self.psi(z)
                           ])
            else: 
                bc = np.array([self.u(z, z0 = -(self.t)/2, bed = True),
                                self.w(z),
                                self.psi(z),
                                self.phi_u(z),
                                self.phi_w(z)])
        else:
            raise ValueError(
                'Boundary conditions not defined for'
                f'system of type {self.system}.')

        return bc

    def eqs(self, zl, zr, l=0, k=False, pos='mid'):
        """
        Provide boundary or transmission conditions for beam segments.

        Arguments
        ---------
        zl : ndarray
            Solution vector at left end of beam segement. Size is (6x1) or (10x1) depending on foundation
        zr : ndarray
            Solution vector at right end of beam segement. Size is (6x1) or (10x1) depending on foundation
        l : float, optional
            Length of the segment in consideration. Default is zero.
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
        eqsSlab : ndarray
            Vector (of length 9) of boundary conditions for the slab (3) and
            transmission conditions for the slab (6) for boundary segements
            or vector of transmission conditions for the slab (of length 6+6)
            for center segments.
        eqsWeak: ndarray
            Vector (of length 6 ) of boundary conditions for the weak layer (2) and
            transmission conditions for the weak layer (4) for boundary segements
            or vector of transmission conditions for the weak layer (of length 8)
            for center segments.
        
        """

        if not k:
            if pos in ('l', 'left'):
                eqsSlab = np.array([
                    self.bc(zl, l, k, pos)[0],             # Left boundary condition
                    self.bc(zl, l, k, pos)[1],             # Left boundary condition
                    self.bc(zl, l, k, pos)[2],             # Left boundary condition
                    self.u(zr, z0=self.h/2, bed =k),                               # ui(xi = li)
                    self.w(zr),                                     # wi(xi = li)
                    self.psi(zr),                                   # psii(xi = li)
                    self.N(zr,bed = k),                                     # Ni(xi = li)
                    self.M(zr) - (self.h )/2 * self.N(zr, bed = k),  # Mi(xi = li)
                    self.V(zr, bed = k)])                                    # Vi(xi = li)
            elif pos in ('m', 'mid'):
                eqsSlab = np.array([
                    -self.u(zl, z0=self.h/2, bed = k),                              # -ui(xi = 0)
                    -self.w(zl),                                    # -wi(xi = 0)
                    -self.psi(zl),                                  # -psii(xi = 0)
                    -self.N(zl, bed = k),                                    # -Ni(xi = 0)
                    -self.M(zl) + (self.h )/2 * self.N(zl, bed = k), # -Mi(xi = 0)
                    -self.V(zl, bed = k),                                    # -Vi(xi = 0)
                    self.u(zr, z0=self.h/2, bed = k),                               # ui(xi = li)
                    self.w(zr),                                     # wi(xi = li)
                    self.psi(zr),                                   # psii(xi = li)
                    self.N(zr, bed = k),                                     # Ni(xi = li)
                    self.M(zr) - (self.h + self.t)/2 * self.N(zr, bed = k),  # Mi(xi = li)
                    self.V(zr, bed = k)])                                    # Vi(xi = li)
            elif pos in ('r', 'right'):
                eqsSlab = np.array([
                    -self.u(zl, z0=self.h/2, bed = k),                              # -ui(xi = 0)
                    -self.w(zl),                                    # -wi(xi = 0)
                    -self.psi(zl),                                  # -psii(xi = 0)
                    -self.N(zl, bed = k),                                    # -Ni(xi = 0)
                    -self.M(zl)+ (self.h )/2 * self.N(zl, bed = k), # -Mi(xi = 0)
                    -self.V(zl, bed = k),                                    # -Vi(xi = 0)
                    self.bc(zr, l, k, pos)[0],             # Right boundary condition
                    self.bc(zr, l, k, pos)[1],             # Right boundary condition
                    self.bc(zr, l, k, pos)[2]])            # Right boundary condition
            else:
                raise ValueError(
                    (f'Invalid position argument {pos} given. '
                    'Valid segment positions are l, m, and r, '
                    'or left, mid and right.'))
            eqsWeak = np.zeros((4,eqsSlab.shape[1]))            # Weak layer trabsmission conditions phi_u, phi_w, NWeak, VWeak 
        else:
            if pos in ('l', 'left'):
                eqsSlab = np.array([
                    self.bc(zl, l, k, pos)[0],             # Left boundary condition
                    self.bc(zl, l, k, pos)[1],             # Left boundary condition
                    self.bc(zl, l, k, pos)[2],             # Left boundary condition
                    self.u(zr, z0= - (self.t)/2, bed = k),           # ui(xi = li)
                    self.w(zr),                 # wi(xi = li)
                    self.psi(zr),               # psii(xi = li)
                    self.N(zr, bed = k),                 # Ni(xi = li)
                    self.M(zr),                 # Mi(xi = li)
                    self.V(zr, bed = k)])                 # Vi(xi = li)
    
                eqsWeak = np.array([
                    self.bc(zl, l, k, pos)[3],             # Left boundary condition in the weak layer
                    self.bc(zl, l, k, pos)[4],             # Left boundary condition in the weak layer
                    self.phi_u(zr),             # phi_ui(xi = li)
                    self.phi_w(zr),             # phi_wi(xi = li)
                    self.NWeak(zr),             # NWeaki(xi = li)
                    self.VWeak(zr)])            # VWeaki(xi = li)
                    
            elif pos in ('m', 'mid'):
                eqsSlab = np.array([
                    -self.u(zl, - (self.t)/2, bed = k),          # -ui(xi = 0)
                    -self.w(zl),                # -wi(xi = 0)
                    -self.psi(zl),              # -psii(xi = 0)
                    -self.N(zl, bed = k),                # -Ni(xi = 0)
                    -self.M(zl),                # -Mi(xi = 0)
                    -self.V(zl, bed = k),                # -Vi(xi = 0)
                    self.u(zr, - (self.t)/2, bed = k),           # ui(xi = li)
                    self.w(zr),                 # wi(xi = li)
                    self.psi(zr),               # psii(xi = li)
                    self.N(zr, bed = k),                 # Ni(xi = li)
                    self.M(zr),                 # Mi(xi = li)
                    self.V(zr, bed = k)])                 # Vi(xi = li)
                
                eqsWeak = np.array([
                    -self.phi_u(zl),            # -phi_ui(xi = 0)
                    -self.phi_w(zl),            # -phi_wi(xi = 0)
                    -self.NWeak(zl),            # -NWeaki(xi = 0)
                    -self.VWeak(zl),            # -VWeaki(xi = 0)
                    self.phi_u(zr),             # phi_ui(xi = li)
                    self.phi_w(zr),             # phi_wi(xi = li)
                    self.NWeak(zr),             # NWeaki(xi = li)
                    self.VWeak(zr)])            # VWeaki(xi = li)

            elif pos in ('r', 'right'):
                eqsSlab = np.array([
                    -self.u(zl, - (self.t)/2, bed = k),          # -ui(xi = 0)
                    -self.w(zl),                # -wi(xi = 0)
                    -self.psi(zl),              # -psii(xi = 0)
                    -self.N(zl, bed = k),       # -Ni(xi = 0)
                    -self.M(zl),                # -Mi(xi = 0)
                    -self.V(zl, bed = k),       # -Vi(xi = 0)
                    self.bc(zr, l, k, pos)[0],             # Right boundary condition
                    self.bc(zr, l, k, pos)[1],             # Right boundary condition
                    self.bc(zr, l, k, pos)[2]])            # Right boundary condition

                eqsWeak = np.array([
                    -self.phi_u(zl),            # -phi_ui(xi = 0)
                    -self.phi_w(zl),            # -phi_wi(xi = 0)
                    -self.NWeak(zl),            # -NWeaki(xi = 0)
                    -self.VWeak(zl),            # -VWeaki(xi = 0)
                    self.bc(zr, l, k, pos)[3],             # Right boundary condition
                    self.bc(zr, l, k, pos)[4]])            # Right boundary condition
            else:
                raise ValueError(
                    (f'Invalid position argument {pos} given. '
                    'Valid segment positions are l, m, and r, '
                    'or left, mid and right.'))
        return eqsSlab, eqsWeak

    def calc_segments(self, tdi=False, li=False, mi=False, ki=False, k0=False,
                      L=1e4, a=0, m=0, **kwargs):
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
        # Set unbedded segment length
        mode = self.mode_td(l=a)
        if mode in ['A', 'B']:
            lU = a
        if mode in ['C']:
            lU = self.lS

        # Assemble list defining the segments
        if self.system == 'skiers':
            li = np.array(li)                           # Segment lengths
            mi = np.array(mi)                           # Skier weights
            ki = np.array(ki)                           # Crack
            k0 = np.array(k0)                           # No crack
        elif self.system == 'pst-':
            li = np.array([L - a, lU])                   # Segment lengths
            mi = np.array([0])                          # Skier weights
            ki = np.array([True, False])                # Crack
            k0 = np.array([True, True])                 # No crack
        elif self.system == '-pst':
            li = np.array([lU, L - a])                   # Segment lengths
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

        Assemble LHS for slabs transition and boundary conditions from supported and unsupported segments in the form
        [       ]   [ zh1  0   0  ...  0   0   0  ][   ]   [    ]   [     ]  left
        [       ]   [ zh1 zh2  0  ...  0   0   0  ][   ]   [    ]   [     ]  mid
        [       ]   [  0  zh2 zh3 ...  0   0   0  ][   ]   [    ]   [     ]  mid
        [z0Slab ] = [ ... ... ... ... ... ... ... ][ C ] + [ zp ] = [ rhs ]  mid
        [       ]   [  0   0   0  ... zhL zhM  0  ][   ]   [    ]   [     ]  mid
        [       ]   [  0   0   0  ...  0  zhM zhN ][   ]   [    ]   [     ]  mid
        [       ]   [  0   0   0  ...  0   0  zhN ][   ]   [    ]   [     ]  right
        

        Assemble LHS for weak layers transition and boundary conditons zh0Weak from supported segements in separate matrix and 
        append to the matrix of the slab.
        Solve for constants.

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
            Matrix(10xNBedded + 6 x NUnbedded) of solution constants for a system of N
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
        nSBedded = ki.sum()         # Number of bedded segments
        nSFree = len(ki)- ki.sum()  # Number of free segments
        nS = nSBedded + nSFree      # Total number of segements
        nDOFfree = 6               # Number of free constants per free segment
        nDOFbedded = 10                # Number of free constants per bedded segments
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
        zh0Slab = np.zeros([nSBedded * nDOFfree + nSFree * nDOFfree, nSBedded * nDOFbedded + nSFree * nDOFfree])
        zp0Slab = np.zeros([nSBedded * nDOFfree + nSFree * nDOFfree, 1])
        rhsSlab = np.zeros([nSBedded * nDOFfree + nSFree * nDOFfree, 1])

        zh0Weak = np.zeros([nSBedded * (nDOFbedded-nDOFfree), nSBedded * nDOFbedded + nSFree * nDOFfree])
        zp0Weak = np.zeros([nSBedded * (nDOFbedded-nDOFfree), 1])
        rhsWeak = np.zeros([nSBedded * (nDOFbedded-nDOFfree), 1])

        # --- ASSEMBLE LINEAR SYSTEM OF EQUATIONS -----------------------------
        globalStartSlab = 0     # Gloabl counter on the start position in xxxSlab
        globalStartWeak = 0     # Global counter on the start position in xxxWeak
        # Loop through segments to assemble left-hand side
        for i in range(nS):
            # Length, foundation and position of segment i
            l, k, pos = li[i], ki[i], pi[i]
            # Transmission conditions at left and right segment ends
            zhiSlab,zhiWeak = self.eqs(
                zl=self.zh(x=0, l=l, bed=k),
                zr=self.zh(x=l, l=l, bed=k),
                l=l, k=k, pos=pos)
            zpiSlab,zpiWeak = self.eqs(
                zl=self.zp(x=0, phi=phi, bed=k),
                zr=self.zp(x=l, phi=phi, bed=k),
                l=l, k=k, pos=pos)

            # Rows for left-hand side assembly for the slab
            nConst = 10 if k else 6
            startSlab = 0 if i == 0 else 3
            stopSlab = 6 if i == nS-1 else 9
            # Assemble left-hand side for the slab
            zh0Slab[(6 * i - startSlab):(6 * i + stopSlab), globalStartSlab:globalStartSlab + nConst] = zhiSlab
            zp0Slab[(6 * i - startSlab):(6 * i + stopSlab)] += zpiSlab
        
            
            # Check if segment is bedded
            if (pos == 'l' or pos == 'left') and k:                                                                                                         # Case: Left-most segment is bedded
                zh0Weak[0:2, globalStartSlab:globalStartSlab + nConst] = zhiWeak[0:2, :]                                                                    #       Boundary Conditions for either pst or skier
                zp0Weak[0:2] += zpiWeak[0:2]
                if ki[i+1]:                                                                                                                                     # Case: Segment adjacent to the left-most on is bedded:
                    zh0Weak[2:6,globalStartSlab:globalStartSlab + nConst] = zhiWeak[2:,:]                                                                       #       Transmission conditions for phi_u, phi_w,
                    zp0Weak[2:6] += zpiWeak[2:]                                                                                                                 #       and NWeak and VWeak
                else:                                                                                                                                           # Case: Segment adjacent to the left-most on is free:
                    zh0Weak[2:4,globalStartSlab:globalStartSlab + nConst] = zhiWeak[4:,:]                                                                       #       Free (stress free) end for NWeak and VWeak
                    zp0Weak[2:4] += zpiWeak[4:]                                                                                                                 #       no transmission conditions for phi_u and phi_w
            elif (pos == 'm' or pos == 'mid') and k:                                                                                                        # Case: Middle segment is bedded
                if kprev and ki[i+1]:                                                                                                                           # Case: Both adjacent segments are bedded:
                    zh0Weak[globalStartWeak-2:globalStartWeak+6,globalStartSlab:globalStartSlab + nConst] = zhiWeak                                             #       Transmission conditions for phi_u, phi_w,    
                    zp0Weak[globalStartWeak-2:globalStartWeak+6] += zpiWeak                                            #       NWeak, VWeak left and right
                elif kprev and not ki[i+1]:                                                                                                                     # Case: left adjacent segment is bedded, right one not:
                    zh0Weak[globalStartWeak-2:globalStartWeak+4,globalStartSlab:globalStartSlab + nConst] = np.stack([zhiWeak[0:4,:],zhiWeak[6:,:]],axis = 0)   #       Transmission conditions to the previous and
                    zp0Weak[globalStartWeak-2:globalStartWeak+4] += np.stack([zpiWeak[0:4],zpiWeak[6:]],axis = 0)      #       stress-free conditions to the next segement
                elif not kprev and ki[i+1]:                                                                                                                     # Case: right adjacent segement is bedded, left one not:
                    zh0Weak[globalStartWeak:globalStartWeak+6,globalStartSlab:globalStartSlab + nConst] = zhiWeak[2:,:]                                         #       Stress-free condition to the previous segment,
                    zp0Weak[globalStartWeak:globalStartWeak+6] += zpiWeak[2:]                                          #       transmission condition to the next one
                elif not kprev and not ki[i+1]:                                                                                                                 # Case: Neither adjacent segement is bedded:
                    zh0Weak[globalStartWeak:globalStartWeak+4,globalStartSlab:globalStartSlab + nConst] = np.stack([zhiWeak[2:4,:],zhiWeak[6:,:]],axis = 0)     #       Stress-free conditions on both sides
                    zp0Weak[globalStartWeak:globalStartWeak+4] += np.stack([zpiWeak[2:4],zpiWeak[6:]],axis = 0)
            elif (pos ==  'r' or pos == 'right') and k:                                                                                                     # Case: Right-most segment is bedded:
                zh0Weak[-2:, globalStartSlab:globalStartSlab + nConst] = zhiWeak[-2:,:]                                                                     #       Boundary conditions for either pst or skier
                zp0Weak[-2:] +=zpiWeak[-2:]
                if kprev:                                                                                                                                       # Case: Previous segment is bedded:
                    zh0Weak[-6:-2, globalStartSlab:globalStartSlab + nConst] = zhiWeak[0:4,:]                                                                     #       Transmission conditions for phi_u, phi_w,
                    zp0Weak[-6:-2] += zpiWeak[0:4]                                                                                                              #       NWeak and VWeak
                else:                                                                                                                                           # Case: Previous segment is free:
                    zh0Weak[-4:-2, globalStartSlab:globalStartSlab + nConst] = zhiWeak[2:4,:]                                                                     #       Stress-free boundary conditions for the weak
                    zp0Weak[-4:-2] += zpiWeak[2:4]                                                                                                              #       layer for NWeak and VWeak.
            
            # Update the global weak counter
            globalStartWeak += 4 if k else 0    

            # Update the global slab counter
            globalStartSlab += nConst
            # Store information on previous segment
            kprev = k
        

        # Loop through loads to assemble right-hand side
        for i, m in enumerate(mi, start=1):
            # Get skier loads
            Fn, Ft = self.get_skier_load(m, phi)
            # Right-hand side for transmission from segment i-1 to segment i
            rhsSlab[6*i:6*i + 3] = np.vstack([Ft, -Ft*self.h/2 if not ki[i] else -Ft * self.h , Fn])
            
        # Set rhs so that complementary integral vanishes at boundaries
        if self.system not in ['pst-', '-pst']:
            rhsSlab[:3] = self.bc(self.zp(x=0, phi=phi, bed=ki[0]),l = li[0], k = ki[0])[0:3]
            print(ki[0])
            if ki[0]:
                rhsWeak[:2] = self.bc(self.zp(x=0, phi=phi, bed=ki[0]), l = li[0], k = ki[0])[3:]
            
            rhsSlab[-3:] = self.bc(self.zp(x=li[-1], phi=phi, bed=ki[-1]), l = li[-1], k = ki[-1])[0:3]
            if ki[nS-1]:
                rhsWeak[-2:] = self.bc(self.zp(x=li[-1], phi=phi, bed=ki[-1]), l = li[-1], k = ki[-1])[3:]
        
        # Loop through segments to set touchdown at rhs
        for i in range(nS):
            # Length, foundation and position of segment i
            l, k, pos = li[i], ki[i], pi[i]
            mode = self.mode_td(l=l)
            if not k and bool(mode in ['B', 'C']):
                if i==0:
                    rhs[:3] = np.vstack([0,0,self.tc])
                if i == (nS - 1):
                    rhs[-3:] = np.vstack([0,0,self.tc])
        
        # --- SOLVE -----------------------------------------------------------
        zh0 = np.vstack([zh0Slab, zh0Weak])
        zp0 = np.vstack([zp0Slab, zp0Weak])
        rhs = np.vstack([rhsSlab, rhsWeak])

        
        # Solve z0 = zh0*C + zp0 = rhs for constants, i.e. zh0*C = rhs - zp0
        C = np.linalg.solve(zh0, rhs - zp0)

        CReturn = np.full((nS, nDOFbedded),np.nan,dtype=float)
        pos = 0
        for i in range(nS):
            if ki[i]:
                CReturn[i,:] = np.reshape(C[pos:pos+nDOFbedded],C[pos:pos+nDOFbedded].shape[0])
            else:
                CReturn[i,:nDOFfree] = np.reshape(C[pos:pos+nDOFfree],C[pos:pos+nDOFfree].shape[0])
            pos = pos + 10 if ki[i] else 6
        
        # Sort (nDOF = 6) constants for each segment into columns of a matrix
        return CReturn.T


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
        zq = np.full([10, xq.size], np.nan)

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
            if ki[i]:
                zi = self.z(xi, C[:, [i]], l, phi, ki[i])
                zq[:, nqc[i]:nqc[i + 1]] = zi
            else:
                zi = self.z(xi, C[0:6, [i]], l, phi, ki[i])
                zq[0:6, nqc[i]:nqc[i + 1]] = zi
            # Assemble global solution matrix

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
        # Differ between bedded->free and free->bedded !!!!
        iscracktip = [ki[j] != ki[j + 1] for j in range(ntr)]
        

        # Transition indices of crack tips and total number of crack tips
        ict = itr[iscracktip]
        nct = len(ict)
        # Initialize energy release rate array
        Gdif = np.zeros([3, nct])
        # 
        # Compute energy relase rate of all crack tips
        for j, idx in enumerate(ict):
            # Solution at crack tip
            
            if ki[idx]:
                ztip = self.z(li[idx], C[:, [idx]], li[idx], phi, bed=ki[idx])
                zback= self.z(0*li[idx], C[:, [idx]], li[idx], phi, bed=ki[idx])
            else:
                ztip = self.z(0, C[:, [idx+1]], li[idx+1], phi, bed=ki[idx+1])
                zback= self.z(1*li[idx+1], C[:, [idx+1]], li[idx+1], phi, bed=ki[idx+1])
            # Mode I and II differential energy release rates
            Gdif[1:, j] = self.Gi(ztip,zback, unit=unit), self.Gii(ztip,zback, unit=unit)

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

    def get_slab_displacement(self, x, z, loc='mid', unit='mm' ):
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
        bed : boolean
            Boolean wether bedding exists i

        Returns
        -------
        x : ndarray
            Horizontal coordinates (cm).
        ndarray
            Horizontal displacements (unit input).
        """
        

        # Coordinates (cm)
        x = x/10
        z0top = np.full((z.shape[1]),-self.h/2)
        z0mid = np.full((z.shape[1]),0)
        z0bot = np.full((z.shape[1]),self.h/2)

        z0top[~np.isnan(z[-1,:])] = -self.h   - self.t/2
        z0mid[~np.isnan(z[-1,:])] = -self.h/2 - self.t/2
        z0bot[~np.isnan(z[-1,:])] =           - self.t/2

        bed = ~np.isnan(z[-1,:])
        print(bed)
        # Locator
        z0 = {'top': z0top, 'mid': z0mid, 'bot': z0bot}
        # Displacement (unit)
        u = self.u(z, z0=z0[loc], unit=unit,bed =bed)
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
