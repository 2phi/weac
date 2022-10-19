"""
Driver Code for Weac - PST
"""
# Third party imports
import numpy as np

# Project imports
import weac

# === VARIABLES ======================================================
# Input
rho = 250                                   # Slab density (kg/m3)
height = 400                                # Slab height (mm)
totallength = 7000                         # Total length (mm)
#cracklength =np.linspace(60.0,6000.0,100)   # Crack length (mm)
cracklength = np.array([2200.0,3500.0])
inclination = 0                             # Slope inclination (°); pos angle downslope cut in pst-

# Initiante Gdif
Gdif = np.array([0])

# Output location
out = '../../04_touchdown/04_results/data/long_pst/Gdif/data/'+str(rho)+'_'+str(height)+'.txt'

# === DEFINE SLAB LAYERING ============================================
# Either use custom profile
myprofile = [[rho,height]]  # (N) last slab layer above weak layer

# Or select a predefined profile from database
# myprofile = 'medium'

# === SOLVE SYSTEM ================================================
for i in cracklength:
    # === CREATE MODEL INSTANCES ==========================================
    # Propagation saw test cut from the right side with custom layering
    pst_cut_right = weac.Layered(system='-pst', layers=myprofile, \
            a=float(i), cf=1.0, ratio=16, phi=inclination)

    # === INSPECT LAYERING ========================================

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
    weac.plot.contours(pst_cut_right, x=xsl_pst, z=z_pst, i=i, window=totallength, scale=15)
    weac.plot.displacements(pst_cut_right, x=xsl_pst, z=z_pst, i=i, **seg_pst)
    weac.plot.stresses(pst_cut_right, x=xwl_pst, z=z_pst, i=i, **seg_pst)
    #weac.plot.section_forces(pst_cut_right, x=xsl_pst, z=z_pst, i=i, **seg_pst)

    # === COMPUTE ENERGY RELEASE RATE ===========================
    #Gdif = pst_cut_right.gdif(C_pst, inclination,**seg_pst)[0]

    #file = open(file=out, mode='a', encoding='UTF-8')
    #file.write('{:10.6f}'.format(Gdif))
    #file.write('\n')

    #sig = pst_cut_right.sig(z_pst,unit='kPa')
    tau = pst_cut_right.tau(z_pst,unit='kPa')
    #w = pst_cut_right.w(z_pst)
    
    write = 0
    if write:
        file = open(file='scratch/tau.txt', mode='a', encoding='UTF-8')
        for element in tau:
            file.write('{:10.6f}'.format(element))
            file.write('\n')
