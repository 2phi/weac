"""
Driver Code for Weac - PST
"""
# Third party imports
import numpy as np

# Project imports
import weac

# === DEFINE SLAB LAYERING ============================================

# Either use custom profile
myprofile = [[126, 570]]  # (N) last slab layer above weak layer

# Or select a predefined profile from database
# myprofile = 'medium'

# Example with a crack cut from the right-hand side.

# +-----------------------------+-----+
# |                             |     |
# |             1               |  2  |
# |                             |     |
# +-----------------------------+-----+
#  |||||||||||||||||||||||||||||
# --------------------------------------

# Input
totallength = 5000                      # Total length (mm)
cracklength = np.linspace(1610,2400,80)                       # Crack length (mm)
inclination = 0                      # Slope inclination (°)

for i in cracklength:
    print(i)
    # === CREATE MODEL INSTANCES ==========================================
    # Propagation saw test cut from the right side with custom layering
    pst_cut_right = weac.Layered(system='pst-', layers=myprofile, \
            a=i, cf=1.0/3, ratio=16, phi=inclination)

    # === INSPECT LAYERING ================================================
    #weac.plot.slab_profile(pst_cut_right)

    # Obtain lists of segment lengths, locations of foundations,
    # and position and magnitude of skier loads from inputs. We
    # can choose to analyze the situtation before a crack appears
    # even if a cracklength > 0 is set by replacing the 'crack'
    # key thorugh the 'nocrack' key.
    seg_pst = pst_cut_right.calc_segments(
        L=totallength)['crack']

    # Assemble system of linear equations and solve the
    # boundary-value problem for free constants.

    C_pst = pst_cut_right.assemble_and_solve(
        phi=inclination, **seg_pst)

    # Prepare the output by rasterizing the solution vector at all
    # horizontal positions xsl (slab). The result is returned in the
    # form of the ndarray z. Also provides xwl (weak layer) that only
    # contains x-coordinates that are supported by a foundation.
    xsl_pst, z_pst, xwl_pst = pst_cut_right.rasterize_solution(
        C=C_pst, phi=inclination, num=totallength/10, **seg_pst)

    # === VISUALIZE RESULTS =====================================
    weac.plot.contours(pst_cut_right, x=xsl_pst, z=z_pst, i=i, window=totallength, scale=10)
    weac.plot.displacements(pst_cut_right, x=xsl_pst, z=z_pst, i=i, **seg_pst)
    weac.plot.stresses(pst_cut_right, x=xwl_pst, z=z_pst, i=i, **seg_pst)
    weac.plot.section_forces(pst_cut_right, x=xsl_pst, z=z_pst, i=i, **seg_pst)

    # === WRITE BOUNDARY RESULTS =====================================
    file = open(file='scratch/out.txt', mode='a', encoding='UTF-8')
    
    # Fill left and right boundary values into list
    data = [i, pst_cut_right.M(z_pst)[-1], pst_cut_right.N(z_pst)[-1], pst_cut_right.V(z_pst)[-1],
            pst_cut_right.u(z_pst, z0=0)[-1], pst_cut_right.up(z_pst, z0=0)[-1], pst_cut_right.w(z_pst)[-1],
            pst_cut_right.wp(z_pst)[-1], pst_cut_right.psi(z_pst)[-1], pst_cut_right.psip(z_pst)[-1],
            pst_cut_right.sig(z_pst)[int((totallength-i)/10)], pst_cut_right.tau(z_pst)[int((totallength-i)/10)],
            pst_cut_right.Gi(z_pst)[int((totallength-i)/10)], pst_cut_right.Gii(z_pst)[int((totallength-i)/10)]]

    for x in data:
        file.write('{:10.6f}'.format(x))
        file.write('\t')
    file.write('\n')
