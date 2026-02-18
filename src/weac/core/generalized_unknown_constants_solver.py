"""
This module defines the system model for the WEAC simulation. The system
model is the heart of the WEAC simulation. All data sources are bundled into
the system model. The system model initializes and calculates all the
parameterizations and passes relevant data to the different components.

We utilize the pydantic library to define the system model.
"""

import logging
from typing import Literal, Optional

import numpy as np
from numpy.linalg import LinAlgError


from weac.components import SystemType
from weac.core.generalized_eigensystem import GeneralizedEigensystem
from weac.core.generalized_field_quantities import GeneralizedFieldQuantities
from weac.core.scenario import Scenario


logger = logging.getLogger(__name__)


class GeneralizedUnknownConstantsSolver:
    """
    This class solves the unknown constants for the WEAC simulation.
    """

    @classmethod
    def solve_for_unknown_constants(     # pylint: disable=unused-argument
        cls,
        scenario: Scenario,
        eigensystem: GeneralizedEigensystem,
        system_type: SystemType,
        touchdown_distance: Optional[float] = None,
        touchdown_mode: Optional[
            Literal["A_free_hanging", "B_point_contact", "C_in_contact"]
        ] = None,
        collapsed_weak_layer_kR: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute free constants *C* for system. \\
        Assemble LHS from supported and unsupported segments in the form::
        
            [  ]   [ zh1  0   0  ...  0   0   0  ][   ]   [    ]   [     ]  (left)
            [  ]   [ zh1 zh2  0  ...  0   0   0  ][   ]   [    ]   [     ]  (mid)
            [  ]   [  0  zh2 zh3 ...  0   0   0  ][   ]   [    ]   [     ]  (mid)
            [z0] = [ ... ... ... ... ... ... ... ][ C ] + [ zp ] = [ rhs ]  (mid)
            [  ]   [  0   0   0  ... zhL zhM  0  ][   ]   [    ]   [     ]  (mid)
            [  ]   [  0   0   0  ...  0  zhM zhN ][   ]   [    ]   [     ]  (mid)
            [  ]   [  0   0   0  ...  0   0  zhN ][   ]   [    ]   [     ]  (right)
        
        and solve for constants C.
        
        Returns
        -------
        C : ndarray
            Matrix(24xN) of solution constants for a system of N
            segements. Columns contain the 24 constants of each segement.
        """
        logger.debug("Starting solve unknown constants")
        phi = scenario.phi
        theta = scenario.theta
        qs = scenario.surface_load
        li = scenario.li
        ki = scenario.ki
        gi = scenario.gi
        fi = scenario.fi

        # Determine size of linear system of equations
        nS = len(li)  # Number of beam segments
        nS_supported = ki.sum() # Number of supported segments
        nS_unsupported = nS - nS_supported # Number of unsupported segments
        nDOF_supported = 24  # Number of free constants per segment
        nDOF_unsupported = 12
        logger.debug("Number of supported segments: %s, DOF per supported segment: %s", nS_supported, nDOF_supported)
        logger.debug("Number of unsupported segments: %s, DOF per unsupported segment: %s", nS_unsupported, nDOF_unsupported)

        # Assemble position vector
        pi = np.full(nS, "m")
        pi[0], pi[-1] = "l", "r"

        # Initialize matrices
        Zh0_slab = np.zeros((
            nS * nDOF_unsupported,
            nS_supported * nDOF_supported + nS_unsupported*nDOF_unsupported
        ))
        Zp0_slab = np.zeros((nS * nDOF_unsupported, 1))
        rhs_slab = np.zeros((nS * nDOF_unsupported, 1))

        Zh0_weak_layer = np.zeros((
            nS_supported * (nDOF_supported - nDOF_unsupported), nS_supported * nDOF_supported + nS_unsupported * nDOF_unsupported
        ))
        Zp0_weak_layer = np.zeros((
            nS_supported * (nDOF_supported-nDOF_unsupported),1))
        rhs_weak_layer = np.zeros((
            nS_supported * (nDOF_supported-nDOF_unsupported),1))

        logger.debug(
            "Initialized Zh0_Slab shape: %s, Zp0_Slab shape: %s, rhs_Slab shape: %s",
            Zh0_slab.shape,
            Zp0_slab.shape,
            rhs_slab.shape,
        )
        logger.debug(
            "Initialized Zh0_weak_layer shape: %s, Zp0_weak_layer shape: %s, rhs_weak_layer shape: %s",
            Zh0_weak_layer.shape,
            Zp0_weak_layer.shape,
            rhs_weak_layer.shape,
        )

        # LHS: Transmission & Boundary Conditions between segments
        global_start_slab =0
        global_start_weak_layer=0
        k_left = False
        for i in range(nS):
            # Length, foundation and position of segment i
            length, has_foundation, is_loaded, pos = li[i], ki[i], gi[i], pi[i]
            logger.debug(
                "Assembling segment %s: length=%s, has_foundation=%s,is_loaded=%s, pos=%s",
                i,
                length,
                has_foundation,
                is_loaded,
                pos,
            )
            # Matrix of Size one of: (l: [18,12], m: [24,12], r: [18,12])
            zhl = eigensystem.zh(
                x = 0,
                length = length,
                has_foundation = has_foundation)
            zhr = eigensystem.zh(
                x = length,
                length = length,
                has_foundation = has_foundation)
            zpl = eigensystem.zp(
                x = 0,
                phi = phi,
                theta = theta,
                has_foundation = has_foundation,
                qs = qs if is_loaded else 0.0)
            zpr=eigensystem.zp(
                x = length,
                phi = phi,
                theta = theta,
                has_foundation = has_foundation,
                qs = qs if is_loaded else 0.0)

            Zhi = cls._setup_conditions_slab(
                zl=zhl,
                zr=zhr,
                eigensystem=eigensystem,
                has_foundation=has_foundation,
                pos=pos,
                system_type=system_type,
            )
            # Vector of Size one of: (l: [18,1], m: [24,1], r: [18,1])
            zpi = cls._setup_conditions_slab(
                zl=zpl,
                zr=zpr,
                eigensystem=eigensystem,
                has_foundation=has_foundation,
                pos=pos,
                system_type=system_type,
            )
            # Rows for left-hand side assembly
            nDOF_segment = 24 if has_foundation else 12
            start = 0 if i == 0 else 6
            stop = 12 if i == nS - 1 else 18
            # Assemble left-hand side
            Zh0_slab[( 12 * i - start) : (12 * i + stop), global_start_slab : global_start_slab + nDOF_segment] = Zhi[:,:nDOF_segment]
            Zp0_slab[(12 * i - start) : (12 * i + stop)] += zpi

            logger.debug(
                "Segment %s: Zhi shape: %s, zpi shape: %s", i, Zhi.shape, zpi.shape
            )
            if has_foundation:
                # For supported segments, the Zh0_weak_layer and zp0_weak layer are assembled.
                Zhi_weak_layer = cls._setup_conditions_weak_layer(
                    zl=eigensystem.zh(
                        x=0,
                        length=length,
                        has_foundation=has_foundation),
                    zr = eigensystem.zh(
                        x=length,
                        length=length,
                        has_foundation=has_foundation),
                    eigensystem=eigensystem,pos=pos,
                    system_type=system_type)
                zpi_weak_layer = cls._setup_conditions_weak_layer(
                    zl=eigensystem.zp(
                        x=0,
                        phi=phi,
                        theta=theta,
                        has_foundation=has_foundation,
                        qs = qs if is_loaded else 0.0),
                    zr = eigensystem.zp(
                        x=length,
                        phi=phi,
                        theta=theta,
                        has_foundation=has_foundation,
                        qs = qs if is_loaded else 0.0),
                    eigensystem=eigensystem,
                    pos=pos,
                    system_type=system_type)
                if pos in ("l","left"):
                    Zh0_weak_layer[
                        0:6,
                        global_start_slab : global_start_slab + nDOF_segment
                        ] = Zhi_weak_layer[0:6, :]
                    Zp0_weak_layer[0:6] += zpi_weak_layer[0:6]
                    if ki[i + 1]: # Right neighboring segment is supported
                        Zh0_weak_layer[
                            6 : 18,
                            global_start_slab : global_start_slab + nDOF_segment
                            ] = Zhi_weak_layer[6:, :]
                        Zp0_weak_layer[6 : 18] +=  zpi_weak_layer[6:]
                        # Continuous displacements and section forces in the weak layer
                    else: # Right neighboring segments is unsupported
                        Zh0_weak_layer[
                            6 : 12,
                            global_start_slab : global_start_slab + nDOF_segment
                        ] = Zhi_weak_layer[12:,  :]
                        Zp0_weak_layer[6:12] += zpi_weak_layer[12:]
                        # Free edge at the weak layer with vanishing section forces
                elif pos in ("m", "mid"):
                    # Middle segment
                    local_start = -6 if k_left else 0

                    local_end = 18 if ki[i+1] else 12
                    # Ensures that possible free edges due to unsupported neighboring segments are correctly implemented.
                    Zh0_weak_layer[
                        global_start_weak_layer + local_start : global_start_weak_layer + local_end,
                        global_start_slab : global_start_slab + nDOF_segment ] = np.concatenate([
                            Zhi_weak_layer[6 + local_start : 12, :], Zhi_weak_layer[6-local_end:, :]],
                            axis = 0)
                    Zp0_weak_layer[
                        global_start_weak_layer + local_start : global_start_weak_layer + local_end] += np.concatenate([zpi_weak_layer[6 + local_start : 12], zpi_weak_layer[30-local_end:]], axis = 0)
                    # 30 = nDOF_supported + 6 boundary conditions
                elif pos in ("r", "right"):
                    local_start = -18 if k_left else -12
                    Zh0_weak_layer[
                        local_start : ,
                        global_start_slab : global_start_slab + nDOF_segment] = Zhi_weak_layer[local_start:, :]
                    Zp0_weak_layer[local_start :] += zpi_weak_layer[local_start:]

            global_start_slab += nDOF_segment
            global_start_weak_layer += (nDOF_supported - nDOF_unsupported) if has_foundation else 0
            k_left = has_foundation

        for i,f in enumerate(fi, start = 1):
            rhs_slab[12 * i : 12 * i + 6] = np.array(f).reshape(-1,1)
            logger.debug("RHS %s", rhs_slab[12 * i : 12 * i + 6])
        # Set RHS so that Complementary Integral vanishes at boundaries
        if system_type in ["pst-","-pst"]:

            rhs_slab[:6] = scenario.load_vector_left.reshape(-1,1)
            rhs_slab[-6:] = scenario.load_vector_right.reshape(-1,1)

        if system_type not in ["pst-", "-pst", "rested"]:
            logger.debug("Pre RHS %s", rhs_slab[:6])
            rhs_slab[:6] = cls._boundary_conditions_slab(
                eigensystem.zp(x=0, phi=phi, theta = theta, has_foundation=ki[0], qs=qs if gi[0] else 0.0),
                eigensystem,
                ki[0],
                "l",
                system_type,
            )
            if ki[0]:
                rhs_weak_layer[:6] = cls._boundary_conditions_weak_layer(
                    eigensystem.zp(x=0,phi=phi, theta = theta, has_foundation = ki[0],qs = qs if gi[0] else 0.0),
                    eigensystem,
                    pos = "l",
                    system_type=system_type,
                )
            logger.debug("Post RHS %s", rhs_slab[:6])
            rhs_slab[-6:] = cls._boundary_conditions_slab(
                eigensystem.zp(x=li[-1], phi=phi, theta = theta,has_foundation=ki[-1], qs=qs if gi[-1] else 0.0),
                eigensystem,
                ki[-1],
                "r",         
                system_type,
            )
            if ki[-1]:
                rhs_weak_layer[-6:] = cls._boundary_conditions_weak_layer(
                    eigensystem.zp(x=li[-1],phi=phi,theta = theta, has_foundation = ki[-1],qs = qs if gi[-1] else 0.0),
                    eigensystem,
                    pos = "r",
                    system_type=system_type,
                )
            logger.debug("Post RHS %s", rhs_slab[-6:])
            logger.debug("Set complementary integral vanishing at boundaries.")
        # Loop through segments to set touchdown conditions at rhs
        Zh0 = np.vstack([Zh0_slab,Zh0_weak_layer])
        Zp0 = np.vstack([Zp0_slab,Zp0_weak_layer])
        rhs = np.vstack([rhs_slab,rhs_weak_layer])
        # Solve z0 = Zh0*C + Zp0 = rhs for constants, i.e. Zh0*C = rhs - Zp0
        try:
            C = np.linalg.solve(Zh0, rhs - Zp0)
        except LinAlgError as e:
            zh_shape = Zh0.shape
            rhs_shape = rhs.shape
            zp_shape = Zp0.shape
            rank = int(np.linalg.matrix_rank(Zh0))
            min_dim = min(zh_shape)
            try:
                cond_val = float(np.linalg.cond(Zh0))
                cond_text = f"{cond_val:.3e}"
            except np.linalg.LinAlgError:  # Fallback if condition number fails
                cond_val = float("inf")
                cond_text = "inf"
            rank_status = "singular" if rank < min_dim else "full-rank"
            msg_format = (
                "Failed to solve linear system (np.linalg.solve) with diagnostics: "
                "Zh0.shape=%s, rhs.shape=%s, Zp0.shape=%s, "
                "rank(Zh0)=%s/%s (%s), cond(Zh0)=%s. "
                "Original error: %s"
            )
            msg_args = (
                zh_shape,
                rhs_shape,
                zp_shape,
                rank,
                min_dim,
                rank_status,
                cond_text,
                e,
            )
            logger.error(msg_format, *msg_args)
            raise LinAlgError(msg_format % msg_args) from e
        # Sort (nDOF = 6) constants for each segment into columns of a matrix
        C_return = np.zeros((nS, nDOF_supported), dtype = np.float64)
        pos = 0
        for i in range(nS):
            if ki[i]:
                C_return[i,:] = np.reshape(
                    C[pos : pos + nDOF_supported],
                    C[pos : pos + nDOF_supported].shape[0])
                pos += nDOF_supported
            else:
                C_return[i, :nDOF_unsupported] = np.reshape(
                    C[pos : pos + nDOF_unsupported],
                    C[pos : pos + nDOF_unsupported].shape[0]
                )
                pos += nDOF_unsupported
        return C_return.T

    @classmethod
    def _setup_conditions_slab(
        cls,
        zl: np.ndarray,
        zr: np.ndarray,
        eigensystem: GeneralizedEigensystem,
        has_foundation: bool,
        pos: Literal["l", "r", "m", "left", "right", "mid"],
        system_type: SystemType,
    ) -> np.ndarray:
        """
        Provide boundary or transmission conditions for beam segments.

        Arguments
        ---------
        zl : ndarray
            Solution vector (24x1), (12x1) or (24x24), (12x12) at left end of beam segement.
        zr : ndarray
            Solution vector (24x1), (12x1) or (24x24), (12x12) at right end of beam segement.
        has_foundation : boolean
            Indicates whether segment has foundation(True) or not (False).
            Default is False.
        pos: {'left', 'mid', 'right', 'l', 'm', 'r'}, optional
            Determines whether the segement under consideration
            is a left boundary segement (left, l), one of the
            center segement (mid, m), or a right boundary
            segement (right, r). Default is 'mid'.

        Returns
        -------
        conditions : ndarray
            `zh`: Matrix of Size one of: (`l: [18,12], m: [24,12], r: [18,12]`)

            `zp`: Vector of Size one of: (`l: [28,1], m: [24,1], r: [18,1]`)
        """
        fq = GeneralizedFieldQuantities(eigensystem=eigensystem)
        if pos in ("l", "left"):
            bcs = cls._boundary_conditions_slab(
                zl,
                eigensystem,
                has_foundation,
                pos,
                system_type,
            )  # Left boundary condition
            #has_foundation=False
            conditions = np.array(
                [
                    bcs[0],
                    bcs[1],
                    bcs[2],
                    bcs[3],
                    bcs[4],
                    bcs[5],
                    fq.u(zr, h0=0, b0=0),   # ui(xi = li)
                    fq.v(zr, h0=0),         # vi(xi = li)
                    fq.w(zr, b0=0),         # wi(xi = li)
                    fq.psix(zr),            # psixi(xi = li)
                    fq.psiy(zr),            # psiyi(xi = li)
                    fq.psiz(zr),            # psizi(xi = li)
                    fq.Nx(zr, has_foundation),              # Nxi(xi = li)
                    fq.Vy(zr, has_foundation),              # Vyi(xi = li)
                    fq.Vz(zr, has_foundation),              # Vzi(xi = li)
                    fq.Mx(zr, has_foundation),              # Mxi(xi = li)
                    fq.My(zr, has_foundation),              # Myi(xi = li)
                    fq.Mz(zr, has_foundation),              # Mzi(xi = li)
                ]
            )
        elif pos in ("m", "mid"):
            #has_foundation=False
            conditions = np.array(
                [
                    -fq.u(zl, h0=0, b0=0),   # -ui(xi = 0)
                    -fq.v(zl, h0=0),         # -vi(xi = 0)
                    -fq.w(zl, b0=0),         # -wi(xi = 0)
                    -fq.psix(zl),            # -psixi(xi = 0)
                    -fq.psiy(zl),            # -psiyi(xi = 0)
                    -fq.psiz(zl),            # -psizi(xi = 0)
                    -fq.Nx(zl, has_foundation),              # -Nxi(xi = 0)
                    -fq.Vy(zl, has_foundation),              # -Vyi(xi = 0)
                    -fq.Vz(zl, has_foundation),              # -Vzi(xi = 0)
                    -fq.Mx(zl, has_foundation),              # -Mxi(xi = 0)
                    -fq.My(zl, has_foundation),              # -Myi(xi = 0)
                    -fq.Mz(zl, has_foundation),              # -Mzi(xi = 0)
                    fq.u(zr, h0=0, b0=0),   # ui(xi = li)
                    fq.v(zr, h0=0),         # vi(xi = li)
                    fq.w(zr, b0=0),         # wi(xi = li)
                    fq.psix(zr),            # psixi(xi = li)
                    fq.psiy(zr),            # psiyi(xi = li)
                    fq.psiz(zr),            # psizi(xi = li)
                    fq.Nx(zr, has_foundation),              # Nxi(xi = li)
                    fq.Vy(zr, has_foundation),              # Vyi(xi = li)
                    fq.Vz(zr, has_foundation),              # Vzi(xi = li)
                    fq.Mx(zr, has_foundation),              # Mxi(xi = li)
                    fq.My(zr, has_foundation),              # Myi(xi = li)
                    fq.Mz(zr, has_foundation),              # Mzi(xi = li)
                ]
            )
        elif pos in ("r", "right"):
            bcs = cls._boundary_conditions_slab(
                zr,
                eigensystem,
                has_foundation,
                pos,
                system_type,
            )  # Right boundary condition
            #has_foundation=False
            conditions = np.array(
                [
                    -fq.u(zl, h0=0, b0=0),   # -ui(xi = 0)
                    -fq.v(zl, h0=0),         # -vi(xi = 0)
                    -fq.w(zl, b0=0),         # -wi(xi = 0)
                    -fq.psix(zl),            # -psixi(xi = 0)
                    -fq.psiy(zl),            # -psiyi(xi = 0)
                    -fq.psiz(zl),            # -psizi(xi = 0)
                    -fq.Nx(zl, has_foundation),              # -Nxi(xi = 0)
                    -fq.Vy(zl, has_foundation),              # -Vyi(xi = 0)
                    -fq.Vz(zl, has_foundation),              # -Vzi(xi = 0)
                    -fq.Mx(zl, has_foundation),              # -Mxi(xi = 0)
                    -fq.My(zl, has_foundation),              # -Myi(xi = 0)
                    -fq.Mz(zl, has_foundation),              # -Mzi(xi = 0)
                    bcs[0],
                    bcs[1],
                    bcs[2],
                    bcs[3],
                    bcs[4],
                    bcs[5],
                ]
            )
        logger.debug("Boundary Conditions at pos %s: %s", pos, conditions.shape)  # pylint: disable=E0606
        return conditions

    @classmethod
    def _setup_conditions_weak_layer(
        cls,
        zl: np.ndarray,
        zr: np.ndarray,
        eigensystem: GeneralizedEigensystem,
        pos: Literal["l", "r", "m", "left", "right", "mid"],
        system_type: SystemType,
    ) -> np.ndarray:
        """
            Provide boundary or transmission conditions for the weak layer of beam segments.

            Arguments
            ---------
            zl : ndarray
                Solution vector (24x1), (12x1) or (24x24), (12x12) at left end of beam segement.
            zr : ndarray
                Solution vector (24x1), (12x1) or (24x24), (12x12) at right end of beam segement.
            pos: {'left', 'mid', 'right', 'l', 'm', 'r'}, optional
                Determines whether the segement under consideration
                is a left boundary segement (left, l), one of the
                center segement (mid, m), or a right boundary
                segement (right, r). Default is 'mid'.

            Returns
            -------
            conditions : ndarray
                `zh`: Matrix of Size one of: (`l: [18,12], m: [24,12], r: [18,12]`)

                `zp`: Vector of Size one of: (`l: [28,1], m: [24,1], r: [18,1]`)
            """
        fq = GeneralizedFieldQuantities(eigensystem=eigensystem)
        if pos in ("l", "left"):
            bcs = cls._boundary_conditions_weak_layer(
                zl,
                eigensystem,
                pos,
                system_type,
            ) # Left boundary conditions
            conditions = np.array(
                [
                    bcs[0],
                    bcs[1],
                    bcs[2],
                    bcs[3],
                    bcs[4],
                    bcs[5],
                    fq.theta_uc(zr),        # theta_uci(xi = li)
                    fq.theta_ul(zr),        # theta_uli(xi = li)
                    fq.theta_vc(zr),        # theta_vci(xi = li)
                    fq.theta_vl(zr),        # theta_vli(xi = li)
                    fq.theta_wc(zr),        # theta_wci(xi = li)
                    fq.theta_wl(zr),        # theta_wli(xi = li)
                    fq.Nx_c_weakLayer(zr),  # Nx_c_weak_layeri(xi = li)
                    fq.Nx_l_weakLayer(zr),  # Nx_l_weak_layeri(xi = li)
                    fq.Vy_c_weakLayer(zr),  # Vy_c_weak_layeri(xi = li)
                    fq.Vy_l_weakLayer(zr),  # Vy_l_weak_layeri(xi = li)
                    fq.Vz_c_weakLayer(zr),  # Vz_c_weak_layeri(xi = li)
                    fq.Vz_l_weakLayer(zr),  # Vz_l_weak_layeri(xi = li)
                ]
            )
        elif pos in ("m", "mid"):
            conditions = np.array(
                [
                    -fq.theta_uc(zl),        # -theta_uci(xi = 0)
                    -fq.theta_ul(zl),        # -theta_uli(xi = 0)
                    -fq.theta_vc(zl),        # -theta_vci(xi = 0)
                    -fq.theta_vl(zl),        # -theta_vli(xi = 0)
                    -fq.theta_wc(zl),        # -theta_wci(xi = 0)
                    -fq.theta_wl(zl),        # -theta_wli(xi = 0)
                    -fq.Nx_c_weakLayer(zl),  # -Nx_c_weak_layeri(xi = 0)
                    -fq.Nx_l_weakLayer(zl),  # -Nx_l_weak_layeri(xi = 0)
                    -fq.Vy_c_weakLayer(zl),  # -Vy_c_weak_layeri(xi = 0)
                    -fq.Vy_l_weakLayer(zl),  # -Vy_l_weak_layeri(xi = 0)
                    -fq.Vz_c_weakLayer(zl),  # -Vz_c_weak_layeri(xi = 0)
                    -fq.Vz_l_weakLayer(zl),  # -Vz_l_weak_layeri(xi = 0)
                    fq.theta_uc(zr),        # theta_uci(xi = li)
                    fq.theta_ul(zr),        # theta_uli(xi = li)
                    fq.theta_vc(zr),        # theta_vci(xi = li)
                    fq.theta_vl(zr),        # theta_vli(xi = li)
                    fq.theta_wc(zr),        # theta_wci(xi = li)
                    fq.theta_wl(zr),        # theta_wli(xi = li)
                    fq.Nx_c_weakLayer(zr),  # Nx_c_weak_layeri(xi = li)
                    fq.Nx_l_weakLayer(zr),  # Nx_l_weak_layeri(xi = li)
                    fq.Vy_c_weakLayer(zr),  # Vy_c_weak_layeri(xi = li)
                    fq.Vy_l_weakLayer(zr),  # Vy_l_weak_layeri(xi = li)
                    fq.Vz_c_weakLayer(zr),  # Vz_c_weak_layeri(xi = li)
                    fq.Vz_l_weakLayer(zr),  # Vz_l_weak_layeri(xi = li)
                ]
            )  #
        elif pos in ("r", "right"):
            bcs = cls._boundary_conditions_weak_layer(
                zr,
                eigensystem,
                pos,
                system_type,
            ) # Right boundary conditions
            conditions = np.array(
                [
                    -fq.theta_uc(zl),        # -theta_uci(xi = 0)
                    -fq.theta_ul(zl),        # -theta_uli(xi = 0)
                    -fq.theta_vc(zl),        # -theta_vci(xi = 0)
                    -fq.theta_vl(zl),        # -theta_vli(xi = 0)
                    -fq.theta_wc(zl),        # -theta_wci(xi = 0)
                    -fq.theta_wl(zl),        # -theta_wli(xi = 0)
                    -fq.Nx_c_weakLayer(zl),  # -Nx_c_weak_layeri(xi = 0)
                    -fq.Nx_l_weakLayer(zl),  # -Nx_l_weak_layeri(xi = 0)
                    -fq.Vy_c_weakLayer(zl),  # -Vy_c_weak_layeri(xi = 0)
                    -fq.Vy_l_weakLayer(zl),  # -Vy_l_weak_layeri(xi = 0)
                    -fq.Vz_c_weakLayer(zl),  # -Vz_c_weak_layeri(xi = 0)
                    -fq.Vz_l_weakLayer(zl),  # -Vz_l_weak_layeri(xi = 0)
                    bcs[0],
                    bcs[1],
                    bcs[2],
                    bcs[3],
                    bcs[4],
                    bcs[5],
                ]
            )
        logger.debug("Boundary Conditions at pos %s: %s", pos, conditions.shape)  # pylint: disable=E0606
        return conditions


    @classmethod
    def _boundary_conditions_slab(
        cls,
        z,
        eigensystem: GeneralizedEigensystem,
        has_foundation: bool,
        pos: Literal["l", "r", "m", "left", "right", "mid"],
        system_type: SystemType,
    ):
        """
        Provide equations for free (pst) or infinite (skiers) ends.

        Arguments
        ---------
        z : ndarray
            Solution vector (12x1) or (24x1) at a certain position x.
        l : float, optional
            Length of the segment in consideration. Default is zero.
        Returns
        -------
        bc : ndarray
            Boundary condition vector (length 6) at position x.
        """
        fq = GeneralizedFieldQuantities(eigensystem=eigensystem)
        # Set boundary conditions for PST-systems
        # Reduced functionality as touchdown and vpst are not implemented for the generalized eigensystem
        factor = -1 if pos in ["l", "left"] else 1
        if system_type in ["pst-", "-pst"]:
            bc = np.array([
                fq.Nx(z, has_foundation),
                fq.Vy(z, has_foundation),
                fq.Vz(z, has_foundation),
                fq.Mx(z, has_foundation),
                fq.My(z, has_foundation),
                fq.Mz(z, has_foundation)])
        # Set boundary conditions for SKIER-systems
        elif system_type in ["skier", "skiers"]:
            # Infinite end (vanishing complementary solution)
            bc = np.array([
                fq.u(z, h0=0, b0=0),
                fq.v(z, h0=0),
                fq.w(z, b0=0),
                fq.psix(z),
                fq.psiy(z),
                fq.psiz(z)])
        else:
            raise ValueError(
                f"Boundary conditions not defined for system of type {system_type}."
            )

        return factor * bc


    @classmethod
    def _boundary_conditions_weak_layer(
        cls,
        z,
        eigensystem: GeneralizedEigensystem,
        pos: Literal["l", "r", "m", "left", "right", "mid"],
        system_type: SystemType,
    ):
        """
        Provide equations for the weak layer for free (pst) or infinite (skiers) ends.

        Arguments
        ---------
        z : ndarray
            Solution vector (24x1) at a certain position x.
        l : float, optional
            Length of the segment in consideration. Default is zero.
        has_foundation : boolean
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
            Boundary condition vector (length 6) at position x.
        """
        fq = GeneralizedFieldQuantities(eigensystem=eigensystem)
        # Set boundary conditions for PST-systems
        # Reduced functionality as touchdown and vpst are not implemented for the generalized eigensystem
        factor = -1 if pos in ["l", "left"] else 1
        if system_type in ["pst-", "-pst"]:
            bc = np.array([
                fq.Nx_c_weakLayer(z),
                fq.Nx_l_weakLayer(z),
                fq.Vy_c_weakLayer(z),
                fq.Vy_l_weakLayer(z),
                fq.Vz_c_weakLayer(z),
                fq.Vz_l_weakLayer(z)])
        # Set boundary conditions for SKIER-systems
        elif system_type in ["skier", "skiers"]:
            # Infinite end (vanishing complementary solution)
            bc = np.array([
                fq.theta_uc(z),
                fq.theta_ul(z),
                fq.theta_vc(z),
                fq.theta_vl(z),
                fq.theta_wc(z),
                fq.theta_wl(z)])
        else:
            raise ValueError(
                f"Boundary conditions not defined for system of type {system_type}."
            )

        return factor * bc
