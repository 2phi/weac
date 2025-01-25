import numpy as np
import weac
import time
from scipy.optimize import root_scalar

def check_crack_propagation_criterion(snow_profile, phi, segments, skier_weight=0, E=0.25, t=30):
    """
    Evaluate the crack propagation criterion.

    Parameters
    ----------
    snow_profile : object
        Layered representation of snowpack.
    phi : float
        Slope angle (degrees).
    segments : dict
        Segment-specific data required for the calculation, containing:
        - 'li' : ndarray
            List of segment lengths.
        - 'ki' : ndarray
            List of booleans indicating whether a segment lies on 
            a foundation or not in the cracked configuration.
    skier_weight : float, optional
        Weight of the skier (kg). Default is 0, indicating no skier weight.
    E : float, optional
        Elastic modulus (MPa) of the snow layers. Default is 0.25 MPa.
    t : float, optional
        Weak layer thickness (mm). Default is 30 mm.

    Returns
    -------
    g_delta_diff : float
        Evaluation of fracture toughness envelope for differential energy release rates at crack tips of system.
    crack_propagation_criterion_check : bool
        True if the crack propagation criterion is met (g_delta_diff >= 1),
        otherwise False.

    Notes
    -----
    - gdif function returns differential ERR in kJ, while fracture toughness criterion is evaluated in J.
    - Crack propagation is by default evaluated


    """
    
    li = segments['li']
    ki = segments['ki']
    
    skier_no_weight, C_no_weight, segments_no_weight, _, _, _ = create_skier_object(
        snow_profile, skier_weight, phi, li, ki, 
        crack_case='crack', E=E, t=t
    )
    
    diff_energy = skier_no_weight.gdif(C=C_no_weight, phi=phi, **segments_no_weight)
    g_delta_diff = fracture_toughness_criterion(1000 * diff_energy[1], 1000 * diff_energy[2])
    crack_propagation_criterion_check = g_delta_diff >= 1
    
    return g_delta_diff, crack_propagation_criterion_check



def check_coupled_criterion_anticrack_nucleation(snow_profile, phi, skier_weight, envelope='adam_unpublished', scaling_factor=1, E = 0.25, order_of_magnitude = 1, density = 250, t=30):
    
    """
    Evaluate coupled criterion for anticrack nucleation.

    Parameters
    ----------
    snow_profile : object
        Layered representation of the snowpack containing density and layer-specific properties.
    phi : float
        Slope angle (degrees).
    skier_weight : float
        Weight of the skier (kg).
    envelope : str, optional
        Type of stress failure envelope. Default is 'adam_unpublished'.
    scaling_factor : float, optional
        Scaling factor applied to the stress envelope. Default is 1. 
    E : float, optional
        Elastic modulus (MPa) of the snow layers. Default is 0.25 MPa.
    order_of_magnitude : int, optional
        Order of magnitude for scaling law used for 'adam_unpublished'. Default is 1.
    density : float, optional
        Weak layer density (kg/m³). Default is 250 kg/m³.
    t : float, optional
        Weak layer thickness (mm). Default is 30 mm.

    Returns
    -------
    result : bool
        True if the criteria for coupled criterion for anticrack nucleation are met, otherwise False.
    crack_length : float
        Length of the anticrack (mm) at the found minimum critical solution.
    skier_weight : float
        Skier weight (kg) at the found minimum critical solution.
    skier : object
        Skier object representing the state of the system.
    C : ndarray
        Free constants of the solution for the skier's loading state.
    segments : dict
        Segment-specific data for the cracked solution:
        - 'li': ndarray of segment lengths (mm).
        - 'ki': ndarray of booleans indicating whether a segment lies on 
          a foundation (True) or not (False) in the cracked configuration.
    x_cm : ndarray
        Discretized horizontal positions (cm) of the snowpack.
    sigma_kPa : ndarray
        Weak-layer normal stresses (kPa) at discretized horizontal positions.
    tau_kPa : ndarray
        Weak-layer shear stresses (kPa) at discretized horizontal positions.
    iteration_count : int
        Number of iterations performed in the optimization algorithm.
    elapsed_times : list of float
        Elapsed times for each iteration (seconds).
    skier_weights : list of float
        Skier weights for each iteration (kg).
    crack_lengths : list of float
        Crack lengths for each iteration (mm).
    self_collapse : bool
        True if the system is fully cracked without any additional load, otherwise False.
    pure_stress_criteria : bool
        True if the fracture toughness criteria is met at the found minimum critical skier weight, otherwise False.
    critical_skier_weight : float
        Minimum skier weight (kg) required to surpass stress failure envelope in one point.
    g_delta_last : float
        Fracture toughness envelope evaluation of incremental ERR at solution.
    dist_max : float
        Maximum distance to the stress envelope (non-dimensional).
    g_delta_values : list of float
        Fracture toughness envelope evaluations of incremental ERR for each iteration.
    dist_max_values : list of float
        History of maximum distances to the stress envelope over iterations.

    Notes
    -----
    - This algorithm finds the minimum critical soltuion for which both the stress failure, and fracture toughness envelope boundary conditions. are fulfilled.
    - The algorithm begins by finding the minimum critical skier weight for which the stress failure envelope is suprassed in at least one point.
        It then sets a maximum skier weight of five times the initalised weight, and employs a binary search algorithm to narrow down intervals and find the
        solution of critical skier weight and associated anticrack nucleation length.
    - The setup is robust and well functioning in most cases, but will fail to handle critical skier weights which are very low, or which are higher than the initialised maximum, 
        or cetrain special cases where highly localized stresses results in multiple cracked segments (separated by an uncracked segment).
        In these instances, the dampened version of this method is called.
    - The fracture toughness criterion is evaluated in J, while ERR differentials
      are calculated in kJ.
    

    """
    
    
    start_time = time.time()
    elapsed_times = []

    # Trackers for algorithm
    skier_weights = []
    crack_lengths = []
    dist_max_values = []  
    dist_min_values = []
    g_delta_values = []   
    iteration_count = 0
    max_iterations = 25

    # Initialize parameters
    length = 1000 * sum(layer[1] for layer in snow_profile)  # Total length (mm)
    k0 = [True, True, True, True]  # Support boolean for uncracked solution
    li = [length / 2, 0, 0, length / 2]  # Length segments
    ki = [True, False, False, True]  # Length of segments with foundations
    
    # Find minimum critical force to initialize algorithm 
    critical_skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min = find_minimum_force(snow_profile, phi, li, k0, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude, density = density, t=t)
    
    
    # Exception: the entire solution is cracked
    if (dist_min > 1): 
        crack_length = length
        skier_weight = 0
        
        # Create a longer profile to enable a derivation of the incremental ERR of the completely cracked solution
        li_complete_crack = [50000] + li + [50000]
        ki_complete_crack = [False] * len(ki) 
        ki_complete_crack = [True] + ki_complete_crack + [True]
        k0 = [True] * len(ki_complete_crack)
        
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
            snow_profile, skier_weight, phi, li_complete_crack, k0, crack_case='nocrack', E = E, t=t
        )

        # Solving a cracked solution, to calculate incremental ERR
        c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object(
            snow_profile, skier_weight, phi, li_complete_crack, ki_complete_crack, crack_case='crack', E = E, t=t
        )

        # Calculate incremental energy released compared to uncracked solution
        incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=phi, **c_segments, k0=k0)
        g_delta = fracture_toughness_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
        
        self_collapse = True
        return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, 0, elapsed_times, skier_weights, crack_lengths, self_collapse, False, critical_skier_weight, g_delta, dist_min, g_delta_values, dist_min_values
        

    elif (dist_min <= 1) and (critical_skier_weight >= 1) :
  
        # Set max skier weight as 5x, and minimum weight slightly above the found minimum to ensure being outside the stress envelope      
        skier_weight = critical_skier_weight * 1.005
        max_skier_weight = 5 * skier_weight
        min_skier_weight = critical_skier_weight
        
        # Set initial crack length and error margin
        crack_length = 1 
        err = 1000  
        li = [length / 2 - crack_length / 2, crack_length / 2, crack_length / 2, length / 2 - crack_length / 2]
        ki = [True, False, False, True]
        
        while np.abs(err) > 0.002 and iteration_count < max_iterations and any(ki):
            # Track skier weight, crack length, dist_max, g_delta, and time for each iteration
            iteration_count += 1
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)
            dist_max_values.append(dist_max) 
            dist_min_values.append(dist_min)
            elapsed_times.append(time.time() - start_time)

            # Create base_case with the correct number of segments
            k0 = [True] * len(ki)
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
                snow_profile, skier_weight, phi, li, k0, crack_case='nocrack', E = E, t=t
            )

            # Check distance to failure for uncracked solution
            distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density)
            dist_max = np.max(distance_to_failure)
            dist_min = np.min(distance_to_failure)

            # Solving a cracked solution, to calculate incremental ERR
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object(
                snow_profile, skier_weight, phi, li, ki, crack_case='crack', E = E, t=t
            )

            # Calculate incremental energy released compared to uncracked solution
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=phi, **c_segments, k0=k0)
            g_delta = fracture_toughness_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
            g_delta_values.append(g_delta) 
            
            # Update error margin
            err = np.abs(g_delta - 1)

            if iteration_count == 1 and (g_delta > 1 or err < 0.02):
                # Exception: the fracture is governed by a pure stress criterion as the fracture toughess envelope is superseded for minmum critical skier weight
                pure_stress_criteria = True
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, pure_stress_criteria, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values

            # Update of skier weight boundaries
            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            new_skier_weight = (min_skier_weight + max_skier_weight) / 2

            if np.abs(err) > 0.002:
                skier_weight = new_skier_weight
                # g_delta_last = g_delta
                new_crack_length, li, ki = find_new_anticrack_length(snow_profile, skier_weight, phi, li, ki, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude, density = density, t=t)
                crack_length = new_crack_length

        # End of loop: convergence or max iterations reached
        if iteration_count < max_iterations and any(ki):
            if crack_length > 0:
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values
            else:
                # Call dampened version to attempt to solve certain convergence issues
                return check_coupled_criterion_anticrack_nucleation_dampened(snow_profile, phi, skier_weight, dampening = 1, envelope=envelope, scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude, t=t)

        elif not any(ki):
            # Exception: Entire solution is cracked - should in general not happen and is indication of poor assumptions
            return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_min, g_delta_values, dist_min_values

        else:
            return check_coupled_criterion_anticrack_nucleation_dampened(snow_profile, phi, skier_weight, dampening = 1, envelope=envelope, scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude, density = density)
        
    else:
        # Rarely occurs - often caused by a skier weight below one kilo
        return False, 0, critical_skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, 0, dist_max, g_delta_values, dist_max_values

    
def check_coupled_criterion_anticrack_nucleation_dampened(snow_profile, phi, skier_weight, dampening=1, envelope='adam_unpublished', scaling_factor=1, E=0.25, order_of_magnitude=1, density = 250, t=30):
    """
    Evaluate coupled criterion for anticrack nucleation using dampened algorithm.

    Parameters
    ----------
    snow_profile : object
        Layered representation of the snowpack containing density and layer-specific properties.
    phi : float
        Slope angle (degrees).
    skier_weight : float
        Weight of the skier (kg).
    dampening : float, optional
        Dampening factor applied to adjust convergence. Default is 1.
    envelope : str, optional
        Type of stress failure envelope. Default is 'adam_unpublished'.
    scaling_factor : float, optional
        Scaling factor applied to the stress envelope. Default is 1.
    E : float, optional
        Elastic modulus (MPa) of the snow layers. Default is 0.25 MPa.
    order_of_magnitude : int, optional
        Order of magnitude for scaling law used for 'adam_unpublished'. Default is 1.
    density : float, optional
        Weak layer density (kg/m³). Default is 250 kg/m³.
    t : float, optional
        Weak layer thickness (mm). Default is 30 mm.

    Returns
    -------
    result : bool
        True if the criteria for coupled criterion for anticrack nucleation are met, otherwise False.
    crack_length : float
        Length of the anticrack (mm) at the found minimum critical solution.
    skier_weight : float
        Skier weight (kg) at the found minimum critical solution.
    skier : object
        Skier object representing the state of the system.
    C : ndarray
        Free constants of the solution for the skier's loading state.
    segments : dict
        Segment-specific data for the cracked solution:
        - 'li': ndarray of segment lengths (mm).
        - 'ki': ndarray of booleans indicating whether a segment lies on 
          a foundation (True) or not (False) in the cracked configuration.
    x_cm : ndarray
        Discretized horizontal positions (cm) of the snowpack.
    sigma_kPa : ndarray
        Weak-layer normal stresses (kPa) at discretized horizontal positions.
    tau_kPa : ndarray
        Weak-layer shear stresses (kPa) at discretized horizontal positions.
    iteration_count : int
        Number of iterations performed in the optimization algorithm.
    elapsed_times : list of float
        Elapsed times for each iteration (seconds).
    skier_weights : list of float
        Skier weights for each iteration (kg).
    crack_lengths : list of float
        Crack lengths for each iteration (mm).
    self_collapse : bool
        True if the system is fully cracked without any additional load, otherwise False.
    pure_stress_criteria : bool
        True if the fracture toughness criteria is met at the found minimum critical skier weight, otherwise False.
    critical_skier_weight : float
        Minimum skier weight (kg) required to surpass stress failure envelope in one point.
    g_delta_last : float
        Fracture toughness envelope evaluation of incremental ERR at solution.
    dist_max : float
        Maximum distance to the stress envelope (non-dimensional).
    g_delta_values : list of float
        Fracture toughness envelope evaluations of incremental ERR for each iteration.
    dist_max_values : list of float
        History of maximum distances to the stress envelope over iterations.

    Notes
    -----
    - This algorithm is a dampened version of the coupled criterion algorithm, intended to improve convergence for challenging cases.
    - It begins by finding the minimum critical skier weight and incrementally adjusts the crack length and skier weight while ensuring stability through dampened scaling.
    - The method is designed to handle instances where rapid oscillations or multiple cracked segments hinder convergence.
    - The fracture toughness criterion is evaluated in J, while ERR differentials are calculated in kJ.

    """
    
    
    # Trackers
    start_time = time.time()
    elapsed_times = []
    skier_weights = []
    crack_lengths = []
    dist_max_values = []
    dist_min_values = []
    g_delta_values = []

    # Initialize parameters
    length = 100 * sum(layer[1] for layer in snow_profile)  # Total length (mm)
    li = [length / 2, 0, 0, length / 2]  # Length segments
    ki = [True, False, False, True]  # Initial crack configuration
    k0 = [True] * len(ki)

    # Find minimum critical force to initialize
    critical_skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min = find_minimum_force(
        snow_profile, phi, li, k0, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude=order_of_magnitude, density = density, t=t
    )

    if (dist_min > 1):
        self_collapse = True
        crack_length = length
        skier_weight = 0
        
        # Add 1000 to the start and end of `li`
        li_complete_crack = [50000] + li + [50000]

        # Create `ki_complete_crack` with False and add True at start and end
        ki_complete_crack = [False] * len(ki)  # Matches length of `ki`
        ki_complete_crack = [True] + ki_complete_crack + [True]

        # Create `k0` with all True
        k0 = [True] * len(ki_complete_crack)
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
            snow_profile, skier_weight, phi, li_complete_crack, k0, crack_case='nocrack', E = E, t=t
        )

        # Solving a cracked solution, to calculate incremental ERR
        c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object(
            snow_profile, skier_weight, phi, li_complete_crack, ki_complete_crack, crack_case='crack', E = E, t=t
        )

        # Calculate incremental energy released compared to uncracked solution
        incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=phi, **c_segments, k0=k0)
        g_delta = fracture_toughness_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
        
        return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, 0, elapsed_times, skier_weights, crack_lengths, self_collapse, False, critical_skier_weight, g_delta, dist_min, g_delta_values, dist_min_values



    elif (dist_min <= 1) and (critical_skier_weight >= 1):
        crack_length = 1 
        err = 1000
        li = [length / 2 - crack_length / 2, crack_length / 2, crack_length / 2, length / 2 - crack_length / 2]
        ki = [True, False, False, True]

        # Allow 50 iterations in the dampened version
        iteration_count = 0
        max_iterations = 50
        
        # Need to initialise 
        skier_weight = critical_skier_weight * 1.005 
        min_skier_weight = critical_skier_weight
        max_skier_weight = 3 * critical_skier_weight
        g_delta_max_weight = 0
        
        # New method to ensure that the set max weight will surpass the fracture toughness criterion
        while g_delta_max_weight < 1:
            max_skier_weight = max_skier_weight * 2
            
            # Create base_case with the correct number of segments
            k0 = [True] * len(ki)
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
                snow_profile, max_skier_weight, phi, li, k0, crack_case='nocrack', E=E, t=t
            )

            # Solving a cracked solution, to calculate incremental ERR
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object(
                snow_profile, max_skier_weight, phi, li, ki, crack_case='crack', E=E, t=t
            )

            # Calculate incremental energy released compared to uncracked solution
            k0 = [True] * len(ki)
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=phi, **c_segments, k0=k0)
            g_delta_max_weight = fracture_toughness_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
             

        while abs(err) > 0.002 and iteration_count < max_iterations and any(ki):
            iteration_count += 1
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)
            elapsed_times.append(time.time() - start_time)

            # Create skier object for uncracked case
            k0 = [True] * len(ki)
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
                snow_profile, skier_weight, phi, li, ki, crack_case='nocrack', E=E, t=t
            )

            # Check distance to failure
            distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude=order_of_magnitude, density = density)
            dist_max = np.max(distance_to_failure)
            dist_min = np.min(distance_to_failure)
            dist_max_values.append(dist_max)
            dist_min_values.append(dist_min)

            # Cracked solution for energy release
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object(
                snow_profile, skier_weight, phi, li, ki, crack_case='crack', E=E, t=t
            )

            # Incremental energy
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=phi, **c_segments, k0=k0)
            g_delta = fracture_toughness_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
            g_delta_values.append(g_delta)

            err = abs(g_delta - 1)

            if iteration_count == 1 and g_delta > 1:
                pure_stress_criteria = True
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, pure_stress_criteria, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values


            # Adjust skier boundary weights
            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            new_skier_weight = (min_skier_weight + max_skier_weight) / 2
            
            
            # Apply dampening of algorithm if we are sufficiently close to the goal, to avoid non convergence due to oscillation, but ensure we do close in on the target
            if np.abs(err) < 0.5: 
                scaling = (dampening + 1 + (new_skier_weight / skier_weight) ) / (dampening + 1 + 1)  # Dampened scaling
            else:
                scaling = 1

            if np.abs(err) > 0.002:
                # old_skier_weight = skier_weight
                skier_weight = scaling * new_skier_weight
                # g_delta_last = g_delta
                new_crack_length, li, ki = find_new_anticrack_length(snow_profile, skier_weight, phi, li, ki, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude, density = density, t=t)
                crack_length = new_crack_length

        # Check final convergence
        if iteration_count < max_iterations and any(ki):
            return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values
        else:
            return False, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values
    else:
        return False, 0, critical_skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, 0, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, 0, dist_max, g_delta_values, dist_max_values
        
    


def stress_envelope(sigma, tau, envelope='adam_unpublished', scaling_factor=1, order_of_magnitude = 1, density = 250):
    
    """
        Evaluate the stress envelope for given stress components.

        Parameters
        ----------
        sigma : array-like
            Normal stress components (kPa). Must be non-negative.
        tau : array-like
            Shear stress components (kPa). Must be non-negative.
        envelope : str, optional
            Type of stress envelope to evaluate. Options include:
            - 'adam_unpublished' (default): Adam unpublished results .
            - 'schottner': Schottner's envelope.
            - 'mede_s-RG1', 'mede_s-RG2', 'mede_s-FCDH': Mede's criterion with
            different parameterizations for specific snow types.
        scaling_factor : float, optional
            Scaling factor applied to the envelope equations. Default is 1.
        order_of_magnitude : float, optional
            Exponent used for scaling in certain envelopes. Default is 1.
        density : float, optional
            Snow density (kg/m³). Used in certain envelope calculations.
            Default is 250 kg/m³.

        Returns
        -------
        results : ndarray
            Non-dimensional stress evaluation values. For most envelopes,
            values greater than 1 indicate failure, while values less than 1
            indicate stability.

        Notes
        -----
        - Mede's envelopes ('mede_s-RG1', 'mede_s-RG2', 'mede_s-FCDH') are derived from the work of Mede et al. (2018), "Snow Failure Modes Under Mixed Loading," published in Geophysical Research Letters.
        - Schöttner's envelope ('schottner') is based on the preprint by Schöttner et al. (2025), "On the Compressive Strength of Weak Snow Layers of Depth Hoar".
        - The 'adam_unpublished' envelope scales with weak layer density linearly (compared to density baseline) by a 'scaling_factor' (weak layer density / density baseline),
        unless modified by 'order_of_magnitude'.
        - Mede's criteria ('mede_s-RG1', 'mede_s-RG2', 'mede_s-FCDH') define
        failure based on a piecewise function of stress ranges.

        Raises
        ------
        ValueError
            If an invalid `envelope` type is provided.
        
    """
   
    sigma = np.abs(np.asarray(sigma))
    tau = np.abs(np.asarray(tau))
    results = np.zeros_like(sigma)

    if envelope == 'adam_unpublished':
        # Case for 'adam_unpublished'
        # Rescaling emulates previous literature best using a density baseline of 250 kg/m^3 and order of magnitude 3
        
        # Ensuring sublinear scaling for weak layer densities above 250 kg/m^3
        if scaling_factor > 1:
            order_of_magnitude = 0.7
        
        if scaling_factor < 0.55:
            scaling_factor = 0.55
    
        sigma_c = 6.16 * (scaling_factor**order_of_magnitude)  # (kPa) 6.16 / 2.6
        tau_c = 5.09 * (scaling_factor**order_of_magnitude)      # (kPa) 5.09 / 0.7
        
        return (sigma / sigma_c) ** 2 + (tau / tau_c) ** 2
    
    
    elif envelope == 'schottner': 
        
        rho_ice = 916.7
        sigma_y = 2000
        sigma_c_adam = 6.16
        tau_c_adam = 5.09
        
        
        sigma_c = sigma_y * 13 * (density / rho_ice)**order_of_magnitude
        tau_c = tau_c_adam * (sigma_c / sigma_c_adam)
        
        return (sigma / sigma_c) ** 2 + (tau / tau_c) ** 2 
    
    # Case for 'mede_s-RG1'
    elif envelope == 'mede_s-RG1':
        p0 = 7.00
        tau_T = 3.53
        p_T = 1.49
        
        # Condition for sigma within range of p_T-p0 to p_T
        in_first_range = (sigma >= (p_T - p0)) & (sigma <= p_T)
        
        # Condition for sigma in second range: p_T to p_T + p0
        in_second_range = (sigma > p_T) 
        
        # Apply the calculation for values in the first range
        results[in_first_range] = -tau[in_first_range] * (p0 / (tau_T * p_T)) + sigma[in_first_range] * (1 / p_T) + p0 / p_T
        
        # Apply the calculation for values in the second range
        results[in_second_range] = (tau[in_second_range] ** 2) + ((tau_T / p0) ** 2) * ((sigma[in_second_range] - p_T) ** 2)
        return results
        
        
    elif envelope == 'mede_s-RG2':
        p0 = 2.33
        tau_T = 1.22
        p_T = 0.19
        
        # Condition for sigma within range of p_T-p0 to p_T
        in_first_range = (sigma >= (p_T - p0)) & (sigma <= p_T)
        
        # Condition for sigma in second range: p_T to p_T + p0
        in_second_range = (sigma > p_T) 
        
        # Apply the calculation for values in the first range
        results[in_first_range] = -tau[in_first_range] * (p0 / (tau_T * p_T)) + sigma[in_first_range] * (1 / p_T) + p0 / p_T
        
        # Apply the calculation for values in the second range
        results[in_second_range] = (tau[in_second_range] ** 2) + ((tau_T / p0) ** 2) * ((sigma[in_second_range] - p_T) ** 2)
        return results
   
    elif envelope == 'mede_s-FCDH':
        p0 = 1.45
        tau_T = 0.61
        p_T = 0.17
        
        # Condition for sigma within range of p_T-p0 to p_T
        in_first_range = (sigma >= (p_T - p0)) & (sigma <= p_T)
        
        # Condition for sigma in second range: p_T to p_T + p0
        in_second_range = (sigma > p_T) 
        
        # Apply the calculation for values in the first range
        results[in_first_range] = -tau[in_first_range] * (p0 / (tau_T * p_T)) + sigma[in_first_range] * (1 / p_T) + p0 / p_T
        
        # Apply the calculation for values in the second range
        results[in_second_range] = (tau[in_second_range] ** 2) + ((tau_T / p0) ** 2) * ((sigma[in_second_range] - p_T) ** 2)
        return results
        
    else:
        raise ValueError("Invalid envelope type. Choose 'adam_unpublished' ")



# Kill x_value?
def find_roots_around_x(skier, C, li, phi, sigma_kPa, tau_kPa, x_cm, envelope='adam_unpublished', scaling_factor=1, order_of_magnitude = 1, density = 250):
    
    """
    Exact solution of position where stresses surpass failure envelope boundary.

    Parameters
    ----------
    x_value : float
        The initial x-value to search for roots around (mm).
    skier : object
        Skier object representing the state of the system.
    C : ndarray
        Free constants of the solution for the skier's loading state.
    li : ndarray
        Segment lengths (mm).
    phi : float
        Slope angle (degrees).
    sigma_kPa : ndarray
        Weak-layer normal stresses (kPa) at discretized horizontal positions.
    tau_kPa : ndarray
        Weak-layer shear stresses (kPa) at discretized horizontal positions.
    x_cm : ndarray
        Discretized horizontal positions (cm) of the snowpack.
    envelope : str, optional
        Type of stress failure envelope. Default is 'adam_unpublished'.
    scaling_factor : float, optional
        Scaling factor applied to the stress envelope. Default is 1.
    order_of_magnitude : float, optional
        Exponent used for scaling in certain envelopes. Default is 1.
    density : float, optional
        Weak layer density (kg/m³). Default is 250 kg/m³.

    Returns
    -------
    roots : list of float
        The x-coordinates (mm) of the roots found around the given x-value.

    Notes
    -----
    - The function finds the root search intervals based on stress evaluations of the discretized positions, and then finds the exact solution.


    Raises
    ------
    ValueError
        If no root can be found within the identified bracket.
    """
    
    
    
    
    # Define the lambda function for the root function
    func = lambda x: root_function(x, skier, C, li, phi, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density)

    # Calculate the discrete distance to failure using the envelope function
    discrete_dist_to_fail = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density) - 1
    
    # Find indices where the envelope function transitions from positive to negative
    transition_indices = np.where(np.diff(np.sign(discrete_dist_to_fail)))[0]
    
    # Lists to store indices and values of local minima and maxima
    local_minima_indices = []
    local_maxima_indices = []
    local_minima_values = []
    local_maxima_values = []

    # Loop through the list (ignoring the first and last elements)
    for i in range(1, len(discrete_dist_to_fail) - 1):
        # Check for local maximum
        if discrete_dist_to_fail[i] > discrete_dist_to_fail[i - 1] and discrete_dist_to_fail[i] > discrete_dist_to_fail[i + 1]:
            local_maxima_indices.append(i)
            local_maxima_values.append(discrete_dist_to_fail[i])
        
        # Check for local minimum
        elif discrete_dist_to_fail[i] < discrete_dist_to_fail[i - 1] and discrete_dist_to_fail[i] < discrete_dist_to_fail[i + 1]:
            local_minima_indices.append(i)
            local_minima_values.append(discrete_dist_to_fail[i])
    
    # Extract the corresponding x_cm values at those transition indices
    root_candidates = []
    for idx in transition_indices:
        # Get the x_cm values surrounding the transition
        x_left = x_cm[idx]
        x_right = x_cm[idx+1]
        root_candidates.append((10*x_left, 10*x_right))
        # Adding one millimetre on each side

    # Search for roots within the identified candidates
    roots = []
    for x_left, x_right in root_candidates:
        try:
            root_result = root_scalar(func, bracket=[x_left, x_right], method='brentq')
            if root_result.converged:
                roots.append(root_result.root)
        except ValueError:
            print(f"No root found between x = {x_left} and x = {x_right}.")

    return roots

# The root function we seek to minimize
def root_function(x_value, skier, C, li, phi, envelope='adam_unpublished', scaling_factor=1, order_of_magnitude = 1, density = 250):
    """
    Compute the root function value at a given x-coordinate.

    Returns
    -------
    float
        The result of the stress envelope evaluation minus 1. A value of 0
        indicates the system is on the stability boundary, values < 0 indicate
        stability, and values > 0 indicate failure.

    """

    sigma, tau = calculate_sigma_tau(x_value, skier, C, li, phi)
    return stress_envelope(sigma, tau, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density) - 1

def calculate_sigma_tau(x_value, skier, C, li, phi):
    """
    Calculate normal and shear stresses at a given horizontal x-coordinate.

    Parameters
    ----------
    x_value : float
        The x-coordinate (mm) where stresses are calculated.
    skier : object
        Skier object representing the state of the system.
    C : ndarray
        Free constants of the solution for the skier's loading state.
    li : list or ndarray
        Segment lengths (mm).
    phi : float
        Slope angle (degrees).

    Returns
    -------
    sigma : float
        Normal stress (kPa) at the given x-coordinate.
    tau : float
        Shear stress (kPa) at the given x-coordinate.

    Notes
    -----
    - Shear stress ('tau') is returned with a switched sign to match
      the system's convention.


    """
    segment_index, coordinate_in_segment = find_segment_index(li, x_value)
    Z = skier.z(coordinate_in_segment, C, li[segment_index], phi, bed=True)
    t = skier.tau(Z, unit='kPa')
    s = skier.sig(Z, unit='kPa')

    tau = -t[segment_index]  # Remember to switch sign
    sigma = s[segment_index]
    return sigma, tau



# segment_lengths should be li
def find_segment_index(segment_lengths, coordinate):

    """
    Determine the index of the segment containing a given coordinate. Help method to place skier point mass in centered position. 

    Parameters
    ----------
    segment_lengths : list, ndarray, or float
        Lengths of the segments (mm).
    coordinate : float
        The coordinate (mm) to locate within the segments.

    Returns
    -------
    index : int
        Index of the segment containing the coordinate. Returns -1 if the
        coordinate exceeds all segments.
    relative_value : float or None
        Coordinate value relative to the start of the identified segment.
        Returns None if the coordinate exceeds all segments.

    """


    # Handle the case where segment_lengths is a single integer
    if isinstance(segment_lengths, (int, float)):
        return 0, coordinate  # Return index 0 and the coordinate as the relative value

    # Convert segment_lengths to an array if it's a list
    segment_lengths = np.asarray(segment_lengths)

    # Check for singular segment
    if len(segment_lengths) == 1:
        return 0, coordinate  # Return index 0 and the coordinate as the relative value

    cumulative_length = 0

    for index, length in enumerate(segment_lengths):
        cumulative_length += length
        if coordinate <= cumulative_length:
            # Calculate the relative value within the segment
            relative_value = coordinate - (cumulative_length - length)
            return index, relative_value

    return -1, None  # Return -1 if coordinate exceeds all segments







def find_new_anticrack_length(snow_profile, skier_weight, phi, li, ki, envelope='adam_unpublished', scaling_factor=1, E = 0.25, order_of_magnitude = 1, density = 250, t=30):

    """
    Find the resulting anticrack length and updated segment configurations, for a given skier weight.

    Parameters
    ----------
    snow_profile : object
        Layered representation of the snowpack.
    skier_weight : float
        Weight of the skier (kg).
    phi : float
        Slope angle (degrees).
    li : list or ndarray
        Current segment lengths (mm).
    ki : list of bool
        Boolean flags indicating whether each segment lies on a foundation
        (True) or is cracked (False).
    envelope : str, optional
        Type of stress failure envelope. Default is 'adam_unpublished'.
    scaling_factor : float, optional
        Scaling factor applied to the stress envelope. Default is 1.
    E : float, optional
        Elastic modulus (MPa) of the snow layers. Default is 0.25 MPa.
    order_of_magnitude : float, optional
        Exponent used for scaling in certain envelopes. Default is 1.
    density : float, optional
        Snow density (kg/m³). Default is 250 kg/m³.
    t : float, optional
        Weak layer thickness (mm). Default is 30 mm.

    Returns
    -------
    new_crack_length : float
        Length of the skier weight implied anticrack (mm).
    li : list of float
        Updated segment lengths (mm).
    ki : list of bool
        Updated boolean flags indicating the foundation state of segments.

    Notes
    -----
    - The segment lengths and foundations are split at the center, assuming point load mass from the skier is centered.

    """

    # Initialize object
    total_length = np.sum(li)
    midpoint = total_length / 2
    li = [midpoint, midpoint]
    ki = [True, True]
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
        snow_profile, skier_weight, phi, li, ki, crack_case='nocrack', E=E, t=t
        ) 

    all_points_are_outside = np.min(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density)) > 1

    # Finding all horizontal positions (roots) where the stress envelope function crosses the boundary 
    roots_x = find_roots_around_x(skier, C, li, phi, sigma_kPa, tau_kPa, x_cm, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density)

    if len(roots_x) > 0:
        # Method to reconstruct li and ki
        segment_boundaries = [0] + roots_x + [total_length]
        li_temp = np.diff(segment_boundaries).tolist()  # Convert to a list
        ki_temp = [True] * (len(segment_boundaries) - 1) 

        # Create a boolean list indicating root positions
        is_root = [False] * len(segment_boundaries)
        for root in roots_x:
            is_root[segment_boundaries.index(root)] = True

        # Iterate over the roots to determine cracked segments
        cracked_segment = True
        
        for i in range(1, len(is_root)):  # Start from the second root
            # Check if the current and previous boundaries are both roots
            if is_root[i] and (is_root[i - 1]) and cracked_segment:
                ki_temp[i - 1] = False  # Mark the segment as cracked
                cracked_segment = not cracked_segment
                # A cracked segment, if there exists more than one, will always switch between cracked and uncracked
                
            elif is_root[i] and (is_root[i - 1]) and (not cracked_segment):
                # These are uncracked segments, i.e. they have support
                ki_temp[i - 1] = True
                cracked_segment = not cracked_segment           

        # Proceed to split li and ki at the midpoint
        li, ki = split_segments_at_midpoint(li_temp, ki_temp)

    elif all_points_are_outside:
        ki = [False] * len(ki)
    else:
        # No changes to li and ki
        li = li
        ki = [True]*len(ki)

    # Calculate new crack length
    new_crack_length = sum(length for length, foundation in zip(li, ki) if not foundation)

    return new_crack_length, li, ki


def split_segments_at_midpoint(segment_lengths, segment_support):
    """
    Split segments at the midpoint of the total length.

    Parameters
    ----------
    segment_lengths : list of float
        Lengths of the segments (mm).
    segment_support : list of bool
        Boolean flags indicating whether each segment is supported (True)
        or not (False).

    Returns
    -------
    new_segments : list of float
        Updated segment lengths after splitting at the midpoint.
    new_support : list of bool
        Updated support flags for the new segments.

    """

    # Calculate the cumulative lengths of segments to find the midpoint
    cumulative_lengths = np.cumsum(segment_lengths)
    total_length = cumulative_lengths[-1]
    midpoint = total_length / 2

    # Find the segment that contains the midpoint
    for i, length in enumerate(segment_lengths):
        if cumulative_lengths[i] >= midpoint:
            # Split the segment at the exact midpoint
            if i == 0:
                # If the midpoint is in the first segment
                new_segments = [midpoint] + segment_lengths[i:]  # split before the first segment
                new_support = [segment_support[0]] + segment_support[i:]  # retain support value
            else:
                # Split the found segment at the midpoint
                segment_start = cumulative_lengths[i - 1] if i > 0 else 0
                new_segments = (
                    segment_lengths[:i] +
                    [midpoint - segment_start] + 
                    [cumulative_lengths[i] - midpoint] + 
                    segment_lengths[i + 1:]
                )
                # Split support for the two new segments
                new_support = (
                    segment_support[:i] + 
                    [segment_support[i]] + 
                    [segment_support[i]] + 
                    segment_support[i + 1:]
                )
            break
    else:
        # If no segment contains the midpoint, return the original segments and support
        return segment_lengths, segment_support

    return new_segments, new_support





def find_minimum_force(snow_profile, phi, li, ki, envelope='adam_unpublished', scaling_factor=1, E = 0.25, order_of_magnitude = 1, density = 250, t=30): 
    """
    Find the minimum skier weight at which the stress failure envelope is surpassed in one point.

    Parameters
    ----------
    snow_profile : object
        Layered representation of the snowpack.
    phi : float
        Slope angle (degrees).
    li : list or ndarray
        Segment lengths (mm).
    ki : list of bool
        Boolean flags indicating whether each segment lies on a foundation (True)
        or is cracked (False).
    envelope : str, optional
        Type of stress failure envelope. Default is 'adam_unpublished'.
    scaling_factor : float, optional
        Scaling factor applied to the stress envelope. Default is 1.
    E : float, optional
        Elastic modulus (MPa) of the snow layers. Default is 0.25 MPa.
    order_of_magnitude : float, optional
        Exponent used for scaling in certain envelopes. Default is 1.
    density : float, optional
        Weak layer density (kg/m³). Default is 250 kg/m³.
    t : float, optional
        Weak layer thickness (mm). Default is 30 mm.

    Returns
    -------
    skier_weight : float
        Critical skier weight (kg) required to surpass the stress failure envelope.
    skier : object
        Skier object representing the system at the critical state.
    C : ndarray
        Free constants of the solution for the skier's loading state.
    segments : dict
        Segment-specific data of the cracked configuration.
    x_cm : ndarray
        Discretized horizontal positions (cm) of the snowpack.
    sigma_kPa : ndarray
        Weak-layer normal stresses (kPa) at discretized horizontal positions.
    tau_kPa : ndarray
        Weak-layer shear stresses (kPa) at discretized horizontal positions.
    dist_max : float
        Maximum distance to the stress envelope (non-dimensional).
    dist_min : float
        Minimum distance to the stress envelope (non-dimensional).

    Notes
    -----
    - The algorithm iteratively adjusts the skier weight until the maximum
      distance to the stress envelope converges to 1 (indicating critical state).
    - If convergence is not achieved within 50 iterations, the dampened version
      of the method ('find_minimum_force_dampened') is called.

    """
    
    
    
    
    # Initial parameters
    skier_weight = 1  # Starting weight of skier
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
        snow_profile, skier_weight, phi, li, ki, crack_case='nocrack', E = E, t=t
    )

    # Calculate the distance to failure
    dist_max = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
    dist_min = np.min(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
    
    if dist_min >= 1:
        # We are outside the stress envelope without any additional skier weight
        return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min

    iteration_count = 0

    # While the stress envelope boundary is not superseeded in any point
    while np.abs(dist_max - 1) > 0.005 and iteration_count < 50:   
        # Scale with the inverse of the distance to stress failure envelope
        skier_weight = skier_weight / dist_max

        # Recreate the skier object with the updated weight
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
            snow_profile, skier_weight, phi, li, ki, crack_case='nocrack', E = E, t=t
        )

        # Recalculate the distance to failure (stress envelope)
        dist_max = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
        dist_min = np.min(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
        iteration_count = iteration_count + 1

    if iteration_count == 50:
        skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min = find_minimum_force_dampened(snow_profile, phi, li, ki, envelope=envelope, scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude, dampening = 1, density = density, t=t)
        
    # Once the loop exits, the critical skier weight has been found
    return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min




def find_minimum_force_dampened(snow_profile, phi, li, ki, envelope='adam_unpublished', scaling_factor=1, E = 0.25, order_of_magnitude = 1, dampening = 1, density = 250, t=30): 
    """
    Dampened version of algorithm to find the minimum skier weight at which the stress failure envelope is surpassed in one point.

    Parameters
    ----------
    snow_profile : object
        Layered representation of the snowpack.
    phi : float
        Slope angle (degrees).
    li : list or ndarray
        Segment lengths (mm).
    ki : list of bool
        Boolean flags indicating whether each segment lies on a foundation (True)
        or is cracked (False).
    envelope : str, optional
        Type of stress failure envelope. Default is 'adam_unpublished'.
    scaling_factor : float, optional
        Scaling factor applied to the stress envelope. Default is 1.
    E : float, optional
        Elastic modulus (MPa) of the snow layers. Default is 0.25 MPa.
    order_of_magnitude : float, optional
        Exponent used for scaling in certain envelopes. Default is 1.
    dampening : float, optional
        Dampening factor for the adjustment of skier weight. Default is 1.
    density : float, optional
        Weak layer density (kg/m³). Default is 250 kg/m³.
    t : float, optional
        Weak layer thickness (mm). Default is 30 mm.

    Returns
    -------
    skier_weight : float
        Critical skier weight (kg) required to surpass the stress failure envelope.
    skier : object
        Skier object representing the system at the critical state.
    C : ndarray
        Free constants of the solution for the skier's loading state.
    segments : dict
        Segment-specific data of the cracked configuration.
    x_cm : ndarray
        Discretized horizontal positions (cm) of the snowpack.
    sigma_kPa : ndarray
        Weak-layer normal stresses (kPa) at discretized horizontal positions.
    tau_kPa : ndarray
        Weak-layer shear stresses (kPa) at discretized horizontal positions.
    dist_max : float
        Maximum distance to the stress envelope (non-dimensional).
    dist_min : float
        Minimum distance to the stress envelope (non-dimensional).

    Notes
    -----
    - If convergence is not achieved within 50 iterations, the dampening factor
      is incremented recursively up to a limit of 5.
   
    """
    

    skier_weight = 1  # Starting weight of skier
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
        snow_profile, skier_weight, phi, li, ki, crack_case='nocrack', E = E, t=t
    )

    # Calculate the distance to failure
    dist_max = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
    dist_min = np.min(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
    
    if dist_min >= 1:
        # We are outside the stress envelope without any additional skier weight
        return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min

    iteration_count = 0

    # If the regular version did not work, it might be because error margin was too small
    while np.abs(dist_max - 1) > 0.01 and iteration_count < 50:  
        # Weighted scaling factor to reduce large oscillations
        skier_weight = (dampening + 1) * skier_weight / (dampening + dist_max)

        # Recreate the skier object with the updated weight
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
            snow_profile, skier_weight, phi, li, ki, crack_case='nocrack', E = E, t=t
        )

        # Recalculate the distance to failure
        dist_max = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
        dist_min = np.min(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
        iteration_count = iteration_count + 1

    if iteration_count == 50:

        if dampening < 5:
            skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min = find_minimum_force_dampened(snow_profile, phi, li, ki, envelope=envelope, scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude, dampening = dampening + 1, density = density, t=t)
            
        else:
            return 0, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min

    return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min






def find_min_crack_length_self_propagation(snow_profile, phi, E, t, initial_interval=(1, 3000)):
    """
    Find the minimum crack length required for self-propagation.

    Parameters
    ----------
    snow_profile : object
        Layered representation of the snowpack.
    phi : float
        Slope angle (degrees).
    E : float
        Elastic modulus (MPa) of the snow layers.
    t : float
        Weak layer thickness (mm).
    initial_interval : tuple of float, optional
        Interval (in mm) within which to search for the minimum crack length.
        Default is (1, 3000).

    Returns
    -------
    crack_length : float or None
        The minimum crack length (mm) required for self-propagation if found,
        or None if the search did not converge.

    Notes
    -----
    - The crack propagation criterion evaluates the fracture toughness of the differential ERR of an existing crack,
        without any additional skier weight (self propagation).
    """
   
   
    # Define the interval for crack_length search
    a, b = initial_interval
    
    # Use root_scalar to find the root
    result = root_scalar(
        g_delta_diff_objective,
        args=(snow_profile, phi, E, t),
        bracket=[a, b],  # Interval where the root is expected
        method='brentq'  # Brent's method
    )
    
    if result.converged:
        return result.root
    else:
        print("Root search did not converge.")
        return None



def g_delta_diff_objective(crack_length, snow_profile, phi, E, t, target=1):

    """
    Objective function to evaluate the fracture toughness function.

    Parameters
    ----------
    crack_length : float
        Length of the crack (mm).
    snow_profile : object
        Layered representation of the snowpack.
    phi : float
        Slope angle (degrees).
    E : float
        Elastic modulus (MPa) of the snow layers.
    t : float
        Weak layer thickness (mm).
    target : float, optional
        Target value for the fracture toughness function. Default is 1.

    Returns
    -------
    difference : float
        Difference between fracture toughness envelope function and the boundary (value equal to one).
        Positive values indicate the energy release rate exceeds the target.

    """
    # Initialize parameters
    length = 1000 * sum(layer[1] for layer in snow_profile)  # Total length (mm)
    li = [(length / 2 - crack_length / 2), (crack_length / 2), (crack_length / 2), (length / 2 - crack_length / 2)]  # Length segments
    ki = [True, False, False, True]  # Length of segments with foundations

    # Create skier object
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(
        snow_profile, 0, phi, li, ki, crack_case='crack', E = E, t=t
    )
    
    # Calculate differential ERR
    diff_energy = skier.gdif(C=C, phi=phi, **segments)
    
    # Evaluate the fracture toughness function (boundary is equal to 1)
    g_delta_diff = fracture_toughness_criterion(1000 * diff_energy[1], 1000 * diff_energy[2])
    
    # Return the difference from the target
    return g_delta_diff - target


def failure_envelope_mede(sigma, sample_type='s-RG1'):
    """
    Compute the shear stress (τ) for a given compression strength (σ) based on the failure envelope parameters. Used for plots.

    Parameters
    ----------
    sigma : array-like
        Array of compression strengths (σ) (kPa).
    sample_type : str, optional
        Type of snow sample for failure envelope calculation. Options are:
        - 's-RG1': Represents rounded grains (type 1).
        - 's-RG2': Represents rounded grains (type 2).
        - 's-FCDH': Represents facets with depth hoar.
        Default is 's-RG1'.

    Returns
    -------
    tau : np.ndarray
        Shear stresses (τ) (kPa) calculated for the given compression strengths (σ).

    Raises
    ------
    ValueError
        If an invalid `sample_type` is provided.

    Notes
    -----
    - The failure envelope is defined by two intervals of σ:
      1. For σ in [p_T - p_0, p_T], τ is calculated linearly.
      2. For σ in (p_T, p_T + p_0], τ is calculated using a parabolic relationship.
    - The parameters (p_0, τ_T, p_T) are specific to each sample type and
      are derived from the study by Mede et al. (2018).

    References
    ----------
    Mede, T., Chambon, G., Hagenmuller, P., & Nicot, F. (2018). "Snow Failure Modes Under Mixed Loading." Geophysical Research Letters, 45(24), 13351-13358. https://doi.org/10.1029/2018GL080637

    """

    # Failure envelope parameters for different sample types
    if sample_type == 's-RG1':
        p0 = 7.00
        tau_T = 3.53
        p_T = 1.49
    elif sample_type == 's-RG2':
        p0 = 2.33
        tau_T = 1.22
        p_T = 0.19
    elif sample_type == 's-FCDH':
        p0 = 1.45
        tau_T = 0.61
        p_T = 0.17
    else:
        raise ValueError("Invalid sample type. Choose 's-RG1', 's-RG2', or 's-FCDH'.")

    # Ensure sigma is a numpy array for element-wise operations
    sigma = np.asarray(sigma)

    # Initialize tau array to store the shear stresses
    tau = np.zeros_like(sigma)

    # First interval: pT - p0 <= p <= pT
    condition_1 = (sigma >= p_T-p0) & (sigma <= p_T)
    tau[condition_1] = (tau_T / p0) *sigma[condition_1]  + (tau_T - (tau_T * p_T / p0))
    
    # Second interval: pT < p <= pT + p0
    condition_2 = (sigma > p_T) & (sigma <= p_T + p0)
    tau[condition_2] = np.sqrt(tau_T**2 - (tau_T **2)/(p0**2) * ((sigma[condition_2] - p_T)**2))

    return tau


def failure_envelope_adam_unpublished(x, scaling_factor=1, order_of_magnitude = 1):
    """
    Compute the shear stress (τ) for a given normal stress (σ) based on the unpublished failure envelope model by Adam. Used for plots. 

    Parameters
    ----------
    x : array-like or float
        Normal stress values (σ) (kPa).
    scaling_factor : float, optional
        Scaling factor applied to the failure envelope. Default is 1.
    order_of_magnitude : float, optional
        Exponent used for scaling the critical parameters. Default is 1.

    Returns
    -------
    tau : np.ndarray
        Shear stress (τ) (kPa) calculated based on the failure envelope model.
        Values are zero outside the bounds of ±σ_c.

    """ 

    # Ensure x is a numpy array for element-wise operations
    x = np.asarray(x)

    # Define critical parameters for failure envelope calculation
    sigma_c = 6.16 * (scaling_factor**order_of_magnitude)  # (kPa)
    tau_c = 5.09 * (scaling_factor**order_of_magnitude)    # (kPa)

    # Calculate shear stress based on the failure envelope equation
    return np.where(
        (x >= -sigma_c) & (x <= sigma_c),  # condition: sigma_c bounds
        np.sqrt(1 - (x**2 / sigma_c**2)) * tau_c,  # equation for valid range
        0  # otherwise, return 0
    )


def failure_envelope_schottner(x, order_of_magnitude = 1, density = 250):
    """
    Compute the shear stress (τ) for a given normal stress (σ) based on the failure envelope model by Schöttner et al.

    Parameters
    ----------
    x : array-like or float
        Normal stress values (σ) (kPa).
    order_of_magnitude : float, optional
        Exponent used for scaling the critical parameters. Default is 1.
    density : float, optional
        Snow density (kg/m³). Default is 250 kg/m³.

    Returns
    -------
    tau : np.ndarray
        Shear stress (τ) (kPa) calculated based on the failure envelope model.
        Values are zero outside the bounds of ±σ_c.

    References
    ----------
    Schöttner, J., Walet, M., Rosendahl, P., Weißgraeber, P., Adam, V., Walter, B., 
    Rheinschmidt, F., Löwe, H., Schweizer, J., & van Herwijnen, A. (2025). "On the 
    Compressive Strength of Weak Snow Layers of Depth Hoar." Preprint, WSL Institute 
    for Snow and Avalanche Research SLF, TU Darmstadt, University of Rostock.

    """
    # Ensure x is a numpy array for element-wise operations
    x = np.asarray(x)

    rho_ice = 916.7
    sigma_y = 2000
    sigma_c_adam = 6.16
    tau_c_adam = 5.09
    
    sigma_c = sigma_y * 13 * (density / rho_ice)**order_of_magnitude
    tau_c = tau_c_adam * (sigma_c / sigma_c_adam)

    # Calculate shear stress based on the failure envelope equation
    return np.where(
        (x >= -sigma_c) & (x <= sigma_c),  # condition: sigma_c bounds
        np.sqrt(1 - (x**2 / sigma_c**2)) * tau_c,  # equation for valid range
        0  # otherwise, return 0
    )
      


def failure_envelope_chandel(sigma, sample_type='FCsf'):
    """
    Compute the shear stress (τ) for a given normal stress (σ) based on the Chandel failure envelope model. Used for plots.

    Parameters
    ----------
    sigma : array-like
        Normal stress values (σ) (kPa).
    sample_type : str, optional
        Type of snow sample for failure envelope calculation. Options are:
        - 'FCsf': Represents near-surface faceted particles.
        - 'FCso': Represents faceted snow.
        Default is 'FCsf'.

    Returns
    -------
    tau : np.ndarray
        Shear stress (τ) (kPa) calculated for the given normal stress (σ).

    Raises
    ------
    ValueError
        If an invalid `sample_type` is provided.

    References
    ----------
    Chandel, C., Srivastava, P., Mahajan, P., & Kumar, V. (2014). "The behaviour of snow under the effect of combined compressive and shear loading." Current Science, 107(5), 888-894.

    """
    # Ensure sigma is an array
    sigma = np.asarray(sigma)
    tau = np.zeros_like(sigma)

    # Define parameters based on sample type
    if sample_type == 'FCso':  # FCsf model
        
        sigma_C = 7.5     # Compressive strength (kPa)
        sigma_Tmax = 2.5  # Threshold stress (kPa)
        c = 7.3            # Cohesion (kPa)
        phi = 22           # Friction angle (degrees)
        
        tau_max = c  + sigma_Tmax * np.tan(np.radians(phi))   # Maximum shear stress (kPa)
        
        
        condition_1 = (sigma <= sigma_Tmax) & (sigma >= 0)
        tau[condition_1] = c  + sigma[condition_1] * np.tan(np.radians(phi)) 
        
        condition_2 = (sigma > sigma_Tmax) & (sigma <= sigma_C)
        tau[condition_2] = tau_max*np.sqrt( 1 - ((sigma[condition_2] - sigma_Tmax )/(sigma_C - sigma_Tmax))**2)
        
        

    elif sample_type == 'FCsf':  # FCso model
        tau0 = 4.1     # Maximum shear stress (kPa)
        sigma0 = 6.05
        
        condition_1 = (sigma <= sigma0) & (sigma >= 0)
        tau[condition_1] = tau0 * np.sqrt( 1 - ( (sigma[condition_1]/sigma0)**2 ) )
        

    else:
        raise ValueError("Unknown sample type. Choose from ['FCsf', 'FCso']")

    return tau


def fracture_toughness_envelope(G_I):
    """
    Compute the Mode II energy release rate (G_II) as a function of the Mode I energy release rate (G_I), given Adam fracture toughness envelope. Used for plots.

    Parameters
    ----------
    G_I : array-like or float
        Mode I energy release rate (ERR) values (J/m²).

    Returns
    -------
    G_II : np.ndarray
        Corresponding Mode II energy release rate (ERR) values (J/m²).
        Values are zero for G_I outside the range [0, G_Ic].

    """
    # Ensure G_I is a numpy array
    G_I_values = np.array(G_I)
    
    # Define the critical values and parameters
    G_Ic = 0.56  # Critical value of G_I in J/m^2
    G_IIc = 0.79  # Critical value of G_II in J/m^2
    n = 5.0  # Exponent for G_I
    m = 2.2  # Exponent for G_II
    
    # Mask for valid G_I values (between 0 and G_Ic)
    valid_mask = (G_I_values >= 0) & (G_I_values <= G_Ic)
    
    # Initialize G_II_values with zeros
    G_II_values = np.zeros_like(G_I_values)
    
    # Calculate G_II for valid G_I values
    G_II_values[valid_mask] = G_IIc * (1 - (G_I_values[valid_mask] / G_Ic)**n)**(1/m)
    
    return G_II_values



# This is latest: keep
def create_skier_object(snow_profile, skier_weight_x, phi, li_x, ki_x, crack_case='nocrack', E = 0.25, t=30):
    """
    Create and configure a skier object to represent the layered snowpack system.

    Parameters
    ----------
    snow_profile : object
        Layered representation of the snowpack.
    skier_weight_x : float
        Weight of the skier (kg) applied to the snowpack.
    phi : float
        Slope angle (degrees).
    li_x : list of float
        Segment lengths (mm).
    ki_x : list of bool
        Boolean flags indicating whether each segment lies on a foundation (True)
        or is cracked (False).
    crack_case : str, optional
        Configuration of the snowpack. Options are:
        - 'nocrack': Represents an uncracked snowpack (default).
        - 'crack': Represents a cracked snowpack.
    E : float, optional
        Elastic modulus (MPa) of the snow layers. Default is 0.25 MPa.
    t : float, optional
        Weak layer thickness (mm). Default is 30 mm.

    Returns
    -------
    skier : object
        Configured skier object representing the snowpack.
    C : ndarray
        Solution constants for the skier's loading state.
    segments : dict
        Segment-specific data based on the crack configuration:
        - 'li': Segment lengths (mm).
        - 'ki': Foundation flags.
        - 'mi': Distributed skier weight (kg).
        - 'k0': Uncracked solution flags.
    x_cm : np.ndarray
        Discretized horizontal positions (cm) of the snowpack.
    sigma_kPa : np.ndarray
        Weak-layer normal stresses (kPa) at discretized horizontal positions.
    tau_kPa : np.ndarray
        Weak-layer shear stresses (kPa) at discretized horizontal positions.

    """

    # Define a skier object - skiers is used to allow for multiple cracked segments
    skier = weac.Layered(system='skiers', layers=snow_profile)
    skier.set_foundation_properties(E = E, t= t, update = True)

    n = len(ki_x)-1

    # Calculate the total sum of the array
    mi_x = np.zeros(n)

    # Initialize cumulative sum and find median index of where to apply skier force
    cumulative_sum = 0
    median_index = -1  # Initialize median_index

    total_length = sum(li_x)
    half_sum = total_length / 2  # Half of the total sum (median point)

    for i, value in enumerate(li_x):
        cumulative_sum += value
        
        if cumulative_sum >= half_sum:
            if li_x[i+1]==0:
                median_index = i+1
            else:
                median_index = i
            break

    mi_x[median_index] = skier_weight_x  # Assign skier_weight to the median index
    k0 = np.full(len(ki_x), True)

    # Calculate segments based on crack case: 'nocrack' or 'crack'
    segments = skier.calc_segments(
                            li=li_x,           # Use the lengths of the segments
                            ki=ki_x,
                            mi=mi_x,
                            k0=k0                 # Use the boolean flags
                            )[crack_case]     # Switch between 'crack' or 'nocrack'

    # Solve and rasterize the solution
    C = skier.assemble_and_solve(phi=phi, **segments)
    xsl_skier, z_skier, xwl_skier = skier.rasterize_solution(C=C, phi=(phi), num=800, **segments)

    # Calculate compressions and shear stress
    x_cm, tau_kPa = skier.get_weaklayer_shearstress(x=xwl_skier, z=z_skier, unit='kPa')
    x_cm, sigma_kPa = skier.get_weaklayer_normalstress(x=xwl_skier, z=z_skier, unit='kPa')

    return skier, C, segments, x_cm, sigma_kPa, tau_kPa


def fracture_toughness_criterion(G_sigma, G_tau):
    
    """
    Evaluate the fracture toughness criterion for a given combination of
    compression (G_sigma) and shear (G_tau) energy release rates (ERR).

    Parameters
    ----------
    G_sigma : float or np.ndarray
        Mode I energy release rate (ERR) (J/m²).
    G_tau : float or np.ndarray
        Mode II energy release rate (ERR) (J/m²).

    Returns
    -------
    g_delta : float or np.ndarray
        Non-dimensional evaluation of the fracture toughness envelope function. A value
        of 1 indicates that the boundary of the fracture toughness envelope is reached.

    Notes
    -----
    - The fracture toughness criterion is defined as:
        g_delta = (|G_sigma| / G_Ic)^n + (|G_tau| / G_IIc)^m
      where:
        G_Ic = 0.56 J/m² (critical Mode I ERR)
        G_IIc = 0.79 J/m² (critical Mode II ERR)
        n = 1 / 0.2 = 5.0 (exponent for G_sigma)
        m = 1 / 0.45 ≈ 2.22 (exponent for G_tau)
    - The criterion is based on the parametrization from Valentin Adam et al. (2024).

    References
    ----------
    Adam, V., Bergfeld, B., & Weißgraeber, P. (2024). "Fracture toughness of mixed-mode
    anticracks in highly porous materials." Nature Communications.

    """

    compression_toughness = 0.56
    n = 1/0.2 
    shear_toughness = 0.79
    m=1/0.45

    g_delta = ( np.abs(G_sigma) / compression_toughness )**n + ( np.abs(G_tau) / shear_toughness )**m 
    
    return g_delta