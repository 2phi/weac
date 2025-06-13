'''
This script demonstrates the basic usage of the WEAC package to run a simulation.
'''
import weac

# 1. Define a snow profile
# Columns are density (kg/m^3) and layer thickness (mm)
# One row corresponds to one layer counted from top (below surface) to bottom (above weak layer).
my_profile = [
    [170, 100],  # (1) surface layer
    [190,  40],  # (2)
    [230, 130],  #  :
    [250,  20],  #  :
    [210,  70],  # (i)
    [380,  20],  #  :
    [280, 100]   # (N) last slab layer above weak layer
]

# 2. Create a model instance
# System can be 'skier', 'pst-' (Propagation Saw Test from left), etc.
skier_model = weac.Layered(system='skiers', layers=my_profile, touchdown=False)

# Optional: Set foundation properties if different from default
# skier_model.set_foundation_properties(E=0.25, t=30) # E in MPa, t in mm

# 3. Calculate segments for a more complex scenario
# We will define custom segment lengths (li), loads per segment (mi),
# and foundation support per segment (ki)

# li_custom: list of segment lengths in mm
li_custom = [500., 2000., 300., 800., 700.]  # Total length 1500mm (1.5m)

# mi_custom: list of skier masses (kg) for each segment. 0 means no point load.
# Represents two skiers on segments 1 and 3.
mi_custom = [80., 0., 0., 70.]

# ki_custom: list of booleans indicating foundation support for each segment.
# True = foundation present, False = no foundation (e.g., bridging a gap).
# Segment 2 has no foundation.
ki_custom = [True, True, False, True, True]

# Calculate total length from custom segments for consistency if needed by other parts,
# though 'li_custom' will primarily define the geometry.
L_total = sum(li_custom)

# 'a' (initial crack length) and 'm' (single skier mass) are set to 0
# as 'ki_custom' and 'mi_custom' now define these aspects.
# We still select the 'crack' configuration from the output dictionary,
# which will use our custom ki, mi, etc.
segments_data = skier_model.calc_segments(
    L=L_total, a=0, m=0,
    li=li_custom,
    mi=mi_custom,
    ki=ki_custom
)['crack']

# 4. Assemble the system of linear equations and solve
# Input: inclination phi (degrees, counterclockwise positive)
inclination_angle = 38  # degrees
unknown_constants = skier_model.assemble_and_solve(phi=inclination_angle, **segments_data)

# 5. Prepare the output by rasterizing the solution
# Input: Solution constants C, inclination phi, and segments data
xsl_slab, z_solution, xwl_weak_layer = skier_model.rasterize_solution(
    C=unknown_constants, phi=inclination_angle, **segments_data
)

print("Simulation completed. Solution constants C:", unknown_constants)
print("Slab x-coordinates (xsl_slab):", xsl_slab)
print("Solution vector (z_solution):", z_solution)
print("Weak layer x-coordinates (xwl_weak_layer):", xwl_weak_layer)

# 6. Visualize the results (optional, requires matplotlib)
# Ensure you have matplotlib installed: pip install matplotlib
try:
    # Visualize deformations as a contour plot
    weac.plot.deformed(skier_model, xsl=xsl_slab, xwl=xwl_weak_layer, z=z_solution,
                       phi=inclination_angle, window=L_total/2, scale=200,
                       field='u', filename='deformed_plot_u')
    weac.plot.deformed(skier_model, xsl=xsl_slab, xwl=xwl_weak_layer, z=z_solution,
                       phi=inclination_angle, window=L_total/2, scale=200,
                       field='w', filename='deformed_plot_w')
    weac.plot.deformed(skier_model, xsl=xsl_slab, xwl=xwl_weak_layer, z=z_solution,
                       phi=inclination_angle, window=L_total/2, scale=200,
                       field='Sxx', filename='deformed_plot_Sxx')
    weac.plot.deformed(skier_model, xsl=xsl_slab, xwl=xwl_weak_layer, z=z_solution,
                       phi=inclination_angle, window=L_total/2, scale=200,
                       field='Szz', filename='deformed_plot_Szz')
    weac.plot.deformed(skier_model, xsl=xsl_slab, xwl=xwl_weak_layer, z=z_solution,
                       phi=inclination_angle, window=L_total/2, scale=200,
                       field='Txz', filename='deformed_plot_Txz')
    weac.plot.deformed(skier_model, xsl=xsl_slab, xwl=xwl_weak_layer, z=z_solution,
                       phi=inclination_angle, window=L_total/2, scale=200,
                       field='principal', filename='deformed_plot_principal')

    # Plot slab displacements
    weac.plot.displacements(skier_model, x=xsl_slab, z=z_solution, **segments_data)

    # Plot weak-layer stresses
    weac.plot.stresses(skier_model, x=xwl_weak_layer, z=z_solution, **segments_data)
    
    # Plot shear/normal stress criteria
    weac.plot.stress_envelope(skier_model, x=xwl_weak_layer, z=z_solution, **segments_data)

except ImportError:
    print("Matplotlib not found. Skipping plot generation. Install with: pip install matplotlib")
except Exception as e:
    print(f"An error occurred during plotting: {e}")

# 7. Compute output quantities (optional)
# Slab deflections
x_cm_deflection, w_um_deflection = skier_model.get_slab_deflection(x=xsl_slab, z=z_solution, unit='um')
print("Slab deflection (x_cm, w_um):", list(zip(x_cm_deflection, w_um_deflection))[:5]) # Print first 5 for brevity

# Weak-layer shear stress
x_cm_shear, tau_kPa_shear = skier_model.get_weaklayer_shearstress(x=xwl_weak_layer, z=z_solution, unit='kPa')
print("Weak-layer shear stress (x_cm, tau_kPa):", list(zip(x_cm_shear, tau_kPa_shear))[:5]) # Print first 5

print("\nSuccessfully ran a basic WEAC simulation.") 