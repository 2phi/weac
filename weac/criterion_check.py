import pandas as pd
import numpy as np
import os
import ast
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import weac
import time
from scipy.optimize import root_scalar

def apply_check_first_criterion(row, envelope='no_cap', scaling_factor=1, manually_overwritten_wl = True, scale_envelope_with_density = True, density_baseline = 250, order_of_magnitude = 1, scale_YM = False, density = 250, wl_thickness = 30):
    
    
    if row['snow_profiles'] in [None, '[]', []]:
        print(f"\033[91m SNOW PROFILE IS EMPTY \033[0m")
        return False, 0, 0, 0, 0, 0, 0, False, False, 0, 0, 0, 0, False, 0, 0, 0, 0, False, 0
    
    
    if manually_overwritten_wl: 
        # The weak layer should be overwritten
        
        if scale_envelope_with_density:
            # The stress envelope should be scaled with density
            
            if row['manually_overwritten_wl'] and row['non_empty_profile_above_manual_wl']:
                # We should only update rows which can be manually overwritten, and which have a non-empty snow profile above the new weak layer
                
                snow_profile = ast.literal_eval(row['snow_profiles_new'])
                scaling_factor = float(row['wl_density_new']) / density_baseline
                
                # We rescale YoungModulus with respect to the new manually overwritten wl
                if scale_YM:
                    E = row['new_YM_new_wl']
                else:
                    E = 0.25 
            
            else:
                # The envelope is not manually overwritten either because the identified weak layer already is weak, because there exists no alternative weak layer, or because any identified weak layer does not have a well defined snow profile above it
                # In any case, we must now use the regular snow profiles
                snow_profile = ast.literal_eval(row['snow_profiles']) 
                scaling_factor = float(row['wl_density']) / density_baseline # Convert wl_density_new to float if valid
                
                # In this case we should use the regular YoungModulus applied to original weak layer
                if scale_YM:
                    E = row['new_YM']
                else:
                    E = 0.25 
                # Might catch Null cases here
        else:
            # We are not scaling stress envelope with density, but are still overwriting the weak layer
            
            if row['manually_overwritten_wl'] and row['non_empty_profile_above_manual_wl']:
                # If we are to manually overwrite the weak layer, then we should use new snow profiles, and YM updated with regards to this new weak layer
                snow_profile = ast.literal_eval(row['snow_profiles_new'])
                if scale_YM:
                    E = row['new_YM_new_wl']
                else:
                    E = 0.25 
            else:
                # 
                snow_profile = ast.literal_eval(row['snow_profiles'])
                if scale_YM:
                    E = row['new_YM']
                else:
                    E = 0.25 
            
            scaling_factor = scaling_factor  # Retain existing scaling factor if wl_density_new is empty or invalid

    else:
        # We are not manually overwriting the weak layer
        snow_profile = ast.literal_eval(row['snow_profiles'])
        
        if scale_YM:
            E = row['new_YM']
        else:
            E = 0.25      
        
        if scale_envelope_with_density and row['wl_density'] not in [None, '[]', []]:  
            if float(row['wl_density'])>-1:
                scaling_factor = float(row['wl_density']) / density_baseline  # Convert wl_density to float if valid
            else: 
                scaling_factor = scaling_factor 
        else:
            scaling_factor = scaling_factor # Retain existing scaling factor if wl_density is empty or invalid



    inclination = row['slopeangle']

    print(f"Called scaling_factor={scaling_factor}")
    print(f"Order of magnitude={order_of_magnitude}")

    # REMEMBER THAT WE NEED TO REVERSE
    snow_profile_reversed = snow_profile[::-1]

    print(snow_profile_reversed)
    
    convergence_check, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iterations, elapsed_times, skier_weights, crack_lengths, starting_outside_stress_envelope, starting_outside_energy_envelope, critical_skier_weight, distance_to_energy_envelope, distance_to_stress_envelope, g_delta_values, dist_max_values = check_first_criterion_v5(
        snow_profile=snow_profile_reversed, inclination=inclination, skier_weight=80, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude, density = density, wl_thickness=wl_thickness
    )
    
    
    # Here we implement check of second criterion


    print(f" THIS IS THE SETMENT WE GET BACK: {c_segments}")    

    g_delta_diff, second_criterion_check = check_second_criterion(snow_profile_reversed, inclination, c_segments, E = E, wl_thickness=wl_thickness)
    print(f"\033[91m G_DELTA_DIFF = {g_delta_diff} / {second_criterion_check} \033[0m")
    
    min_crack_length_self_propagation = find_min_crack_length_self_propagation(snow_profile_reversed, inclination, E, initial_interval=(0, 5000), wl_thickness=wl_thickness)
    
    kappa = crack_length / min_crack_length_self_propagation
    
    g_delta_diff_with_weight, second_criterion_check_with_weight = check_second_criterion(snow_profile_reversed, inclination, c_segments, skier_weight = skier_weight, E = E, wl_thickness=wl_thickness)
    print(f"\033[91m G_DELTA_DIFF_with_skier_weight = {g_delta_diff_with_weight} / {second_criterion_check_with_weight} \033[0m")

    if convergence_check:
        print(f"\033[91m CONVERGENCE: WE FOUND A SOLUTION AND WILL NOW LOOK AT THE NEXT DATA POINT \033[0m")
        print(f"\033[91m CONVERGENCE: profID: {row['profID']} \033[0m")
    else:
        print(f"\033[91m ALGORITHM DID NOT CONVERGE \033[0m")
        print(f"\033[91m CONVERGENCE: profID: {row['profID']} \033[0m")
    
    return convergence_check, skier_weight, crack_length, iterations, elapsed_times, skier_weights, crack_lengths, starting_outside_stress_envelope, starting_outside_energy_envelope, critical_skier_weight, distance_to_energy_envelope, distance_to_stress_envelope, g_delta_diff, second_criterion_check, min_crack_length_self_propagation, g_delta_values, dist_max_values, kappa, g_delta_diff_with_weight, second_criterion_check_with_weight


def check_second_criterion(snow_profile, inclination, segments, skier_weight = 0, E = 0.25, wl_thickness=30):
    # We have results, but would like to fetch g_delta without the skier weight 
    
    print("INSIDE CHECK SECOND CRITERION")
    
    li = segments['li']
    ki = segments['ki']
    print(li)
    print(ki)
    
    
    print(" NOW WE CREATE THE NEW SKIER OBJECT")
    skier_no_weight, C_no_weight, segments_no_weight, _, _, _ = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='crack', E = E, wl_thickness=wl_thickness)
    print(f" SEGMENTS AFTER CREATING THE SKIER OBJECT: {segments_no_weight}")
    
    energy_second_criterion = skier_no_weight.gdif(C=C_no_weight, phi=inclination, **segments_no_weight)
    
    # We get it back in kJ
    g_delta_diff = energy_criterion(1000 * energy_second_criterion[1], 1000 * energy_second_criterion[2])

    second_criterion_check = g_delta_diff >= 1
    
    return g_delta_diff, second_criterion_check, 


def check_first_criterion_v5(snow_profile, inclination, skier_weight, envelope='no_cap', scaling_factor=1, E = 0.25, order_of_magnitude = 1, density = 250, wl_thickness=30):
    print(f"Called scaling_factor={scaling_factor}")
    # Time tracker
    start_time = time.time()
    elapsed_times = []

    # Trackers for skier weights, crack lengths, dist_max, and g_delta
    skier_weights = []
    crack_lengths = []
    dist_max_values = []  # Tracker for dist_max
    dist_min_values = []
    g_delta_values = []   # Tracker for g_delta
    
    # Initialize iteration variables
    iteration_count = 0
    max_iterations = 25

    # Initialize parameters
    length = 1000 * sum(layer[1] for layer in snow_profile)  # Total length (mm)
    k0 = [True, True, True, True]  # Support boolean for uncracked solution
    li = [length / 2, 0, 0, length / 2]  # Length segments
    ki = [True, False, False, True]  # Length of segments with foundations
    
    # Find minimum critical force to initialize our algorithm 
    critical_skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min = find_minimum_force_v4(snow_profile, inclination, li, k0, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude, density = density, wl_thickness=wl_thickness)
    
    
    if (dist_min > 1): 
        print("Entire solution is cracked without any additional load - all points are outside the envelope")
        crack_length = length
        skier_weight = 0
        
        # Add 1000 to the start and end of `li`
        li_complete_crack = [50000] + li + [50000]

        # Create `ki_complete_crack` with False and add True at start and end
        ki_complete_crack = [False] * len(ki)  # Matches length of `ki`
        ki_complete_crack = [True] + ki_complete_crack + [True]

        # Create `k0` with all True
        k0 = [True] * len(ki_complete_crack)
        
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
            snow_profile, skier_weight, inclination, li_complete_crack, k0, crack_case='nocrack', E = E, wl_thickness=wl_thickness
        )

        # Solving a cracked solution, to calculate incremental ERR
        c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
            snow_profile, skier_weight, inclination, li_complete_crack, ki_complete_crack, crack_case='crack', E = E, wl_thickness=wl_thickness
        )

        # Calculate incremental energy released compared to uncracked solution
        incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
        g_delta = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
        
        return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, 0, elapsed_times, skier_weights, crack_lengths, True, False, critical_skier_weight, g_delta, dist_min, g_delta_values, dist_min_values
        

    elif (dist_min <= 1) and (critical_skier_weight >= 1) :
        # We have a well defined skier weight from which we will initialise our algorithm
        
        skier_weight = critical_skier_weight * 1.005
        max_skier_weight = 5 * skier_weight
        min_skier_weight = critical_skier_weight
        
        # Set initial crack length and error margin
        crack_length = 1  # Initial crack length
        err = 1000  # Error margin
        li = [length / 2 - crack_length / 2, crack_length / 2, crack_length / 2, length / 2 - crack_length / 2]
        ki = [True, False, False, True]
        
        while np.abs(err) > 0.002 and iteration_count < max_iterations and any(ki):
            # Track skier weight, crack length, dist_max, g_delta, and time for each iteration
            iteration_count += 1
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)
            dist_max_values.append(dist_max)  # Add dist_max value to the tracker
            dist_min_values.append(dist_min)
            elapsed_times.append(time.time() - start_time)

            # Create base_case with the correct number of segments
            k0 = [True] * len(ki)
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, k0, crack_case='nocrack', E = E, wl_thickness=wl_thickness
            )

            # Check distance to failure for uncracked solution
            distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density)
            dist_max = np.max(distance_to_failure)
            dist_min = np.min(distance_to_failure)

            # Solving a cracked solution, to calculate incremental ERR
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case='crack', E = E, wl_thickness=wl_thickness
            )

            # Calculate incremental energy released compared to uncracked solution
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
            g_delta = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
            g_delta_values.append(g_delta) 
            
            # Update error margin
            err = np.abs(g_delta - 1)

            if iteration_count == 1 and (g_delta > 1 or err < 0.02):
                print(f"EXCEPTION: Energy criterion is fulfilled at minimum critical skier load: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, True, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values

            print(f"START OF ITERATION {iteration_count}: crack length: {crack_length} mm, Skier Weight: {skier_weight} kg, Max Distance to Failure: {dist_max}, Distance to energy envelope: {g_delta}")

            # 
            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            new_skier_weight = (min_skier_weight + max_skier_weight) / 2
            scaling = new_skier_weight / skier_weight

            if np.abs(err) > 0.002:
                skier_weight = new_skier_weight
                g_delta_last = g_delta
                new_crack_length, li, ki = find_new_crack_length_v3(snow_profile, skier_weight, inclination, li, ki, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude, density = density, wl_thickness=wl_thickness)
                crack_length = new_crack_length

                print(f"END OF ITERATION: g_delta: {g_delta} J/m^2, Old skier weight: {skier_weight} kg, Scaling: {scaling}, New skier weight: {skier_weight} kg, crack length: {crack_length} mm, li = {li}, ki = {ki}")

        # End of loop: convergence or max iterations reached
        if iteration_count < max_iterations and any(ki):
            if crack_length > 0:
                print(f"CONVERGENCE: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}, Segments: {c_segments}")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta_last, dist_max, g_delta_values, dist_max_values
            else:
                print("Crack length is zero and our skier load is lower than minimum critical load; redoing the algorithm with the dampened version.")
                #skier_weight_average = np.average(skier_weights)
                #variance = np.var(np.asarray(skier_weights), ddof=1)
                return check_first_criterion_dampened_v5(snow_profile, inclination, skier_weight, dampening = 1, envelope=envelope, scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude, wl_thickness=wl_thickness)

        elif not any(ki):
            print("We are outside the envelope at all points.")
            return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta_last, dist_min, g_delta_values, dist_min_values

        else:
            print("Maximum iterations reached without convergence.")
            
            return check_first_criterion_dampened_v5(snow_profile, inclination, skier_weight, dampening = 1, envelope=envelope, scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude, density = density)
        
    else:
        print(f"Entire solution is uncracked - all points are inside the envelope with max distance to envelope: {dist_max}")
        
        #if dist_max > 0.9 and envelope=='no_cap':
        #   print(f"Since the distance to failure: {dist_max} is close to the boundary, we proceed to evaluate the stress_criterion for scaled version of the stress-envelope")
        #  return check_first_criterion_v5(snow_profile, inclination, skier_weight, envelope=envelope, scaling_factor=scaling_factor / dist_max, E=E, order_of_magnitude = order_of_magnitude)
        #else:
        print(f"Since the distance to failure: {dist_max} is NOT close to the boundary, causing an avalanche is unlikely")
        return False, 0, critical_skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, 0, dist_max, g_delta_values, dist_max_values

    
def check_first_criterion_dampened_v5(snow_profile, inclination, skier_weight, dampening=1, envelope='no_cap', scaling_factor=1, E=0.25, order_of_magnitude=1, density = 250, wl_thickness=30):
    print(f"Called with scaling_factor={scaling_factor}, dampening={dampening}")
    
    
    
    print(f"\033[91m DAMPENED VERSION 000000000000000000000000000000000000000000000000000000000000000000000 \033[0m")
    
    
    
    # Time tracker
    start_time = time.time()
    elapsed_times = []

    # Trackers for skier weights, crack lengths, dist_max, and g_delta
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
    critical_skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min = find_minimum_force_v4(
        snow_profile, inclination, li, k0, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude=order_of_magnitude, density = density, wl_thickness=wl_thickness
    )

    if (dist_min > 1):
        print("Entire solution is cracked without any additional load - all points are outside the envelope")
        crack_length = length
        skier_weight = 0
        
        # Add 1000 to the start and end of `li`
        li_complete_crack = [50000] + li + [50000]

        # Create `ki_complete_crack` with False and add True at start and end
        ki_complete_crack = [False] * len(ki)  # Matches length of `ki`
        ki_complete_crack = [True] + ki_complete_crack + [True]

        # Create `k0` with all True
        k0 = [True] * len(ki_complete_crack)
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
            snow_profile, skier_weight, inclination, li_complete_crack, k0, crack_case='nocrack', E = E, wl_thickness=wl_thickness
        )

        # Solving a cracked solution, to calculate incremental ERR
        c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
            snow_profile, skier_weight, inclination, li_complete_crack, ki_complete_crack, crack_case='crack', E = E, wl_thickness=wl_thickness
        )

        # Calculate incremental energy released compared to uncracked solution
        incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
        g_delta = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
        
        return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, 0, elapsed_times, skier_weights, crack_lengths, True, False, critical_skier_weight, g_delta, dist_min, g_delta_values, dist_min_values



    elif (dist_min <= 1) and (critical_skier_weight >= 1):
        crack_length = 1  # Initial crack length
        err = 1000
        li = [length / 2 - crack_length / 2, crack_length / 2, crack_length / 2, length / 2 - crack_length / 2]
        ki = [True, False, False, True]

        iteration_count = 0
        max_iterations = 50
        
        # Need to initialise 
        # skier_weight = critical_skier_weight * 1.005 (might destroy one out of 124 of the inital ones)
        skier_weight = critical_skier_weight * 1.005 
        min_skier_weight = critical_skier_weight
        
        max_skier_weight = 3 * critical_skier_weight
        g_delta_max_weight = 0
        
        while g_delta_max_weight < 1:
            max_skier_weight = max_skier_weight * 2
            
            # Create base_case with the correct number of segments
            k0 = [True] * len(ki)
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
                snow_profile, max_skier_weight, inclination, li, k0, crack_case='nocrack', E=E, wl_thickness=wl_thickness
            )

            # Solving a cracked solution, to calculate incremental ERR
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
                snow_profile, max_skier_weight, inclination, li, ki, crack_case='crack', E=E, wl_thickness=wl_thickness
            )

            # Calculate incremental energy released compared to uncracked solution
            k0 = [True] * len(ki)
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
            g_delta_max_weight = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
            
            
        print(f"\033[91m MAX WEIGHT: {max_skier_weight} \033[0m")
        print(f"\033[91m MAX WEIGHT: {max_skier_weight} \033[0m")
        print(f"\033[91m MAX WEIGHT: {max_skier_weight} \033[0m")
        print(f"\033[91m MAX WEIGHT: {max_skier_weight} \033[0m")
        print(f"\033[91m MAX WEIGHT: {max_skier_weight} \033[0m")
        

        while abs(err) > 0.002 and iteration_count < max_iterations and any(ki):
            iteration_count += 1
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)
            elapsed_times.append(time.time() - start_time)

            # Create skier object for uncracked case
            k0 = [True] * len(ki)
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E=E, wl_thickness=wl_thickness
            )

            # Check distance to failure
            distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude=order_of_magnitude, density = density)
            dist_max = np.max(distance_to_failure)
            dist_min = np.min(distance_to_failure)
            dist_max_values.append(dist_max)
            dist_min_values.append(dist_min)

            # Cracked solution for energy release
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case='crack', E=E, wl_thickness=wl_thickness
            )

            # Incremental energy
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
            g_delta = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
            g_delta_values.append(g_delta)

            err = abs(g_delta - 1)

            if iteration_count == 1 and g_delta > 1:
                print(f"EXCEPTION: Energy criterion met immediately. Critical skier weight: {skier_weight} kg, crack length: {crack_length} mm.")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, True, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values

            print(f"Iteration {iteration_count}: skier weight: {skier_weight:.2f} kg, crack length: {crack_length} mm, g_delta_NEW: {g_delta:.4f}, error: {err:.4f}, SEGMENTS {c_segments}")

            # Adjust skier weight using dampened scaling
            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            new_skier_weight = (min_skier_weight + max_skier_weight) / 2
            
            
            # Apply dampening of algorithm if we are sufficiently close to the gaol, to avoid non convergence due to oscillation, but ensure we do close in on the target
            if np.abs(err) < 0.5: 
                scaling = (dampening + 1 + (new_skier_weight / skier_weight) ) / (dampening + 1 + 1)  # Dampened scaling
            else:
                scaling = 1
                # skier_weight = new_skier_weight * scaling
        

            if np.abs(err) > 0.002:
                old_skier_weight = skier_weight
                skier_weight = scaling * new_skier_weight
                g_delta_last = g_delta
                new_crack_length, li, ki = find_new_crack_length_v3(snow_profile, skier_weight, inclination, li, ki, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude, density = density, wl_thickness=wl_thickness)
                crack_length = new_crack_length

                print(f"END OF ITERATION: g_delta: {g_delta} J/m^2, Old skier weight: {old_skier_weight} kg, Scaling: {scaling}, New skier weight: {skier_weight} kg, crack length: {crack_length} mm, li = {li}, ki = {ki}")


        # Check final convergence
        if iteration_count < max_iterations and any(ki):
            print(f"CONVERGENCE: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}, Segments: {c_segments}")
            return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values

        # elif (iteration_count < max_iterations) and (not any(ki)):
        #   print(f"CONVERGENCE ENTIRE : crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}, Segments: {c_segments}")

        else:
            print("Maximum iterations reached without convergence.")
            return False, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values

    else:
        print("We are inside the envelope for all points")
        return False, 0, critical_skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, 0, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, 0, dist_max, g_delta_values, dist_max_values
        
        #print(f"Entire solution is uncracked - max distance to envelope: {dist_max}")
        #if dist_max > 0.9 and envelope == 'no_cap':
        #   print(f"Proceeding with scaled stress envelope: scaling_factor={scaling_factor / dist_max:.4f}")
        #  dampening = 1
        # return check_first_criterion_dampened_v5(
        #    snow_profile, inclination, skier_weight, dampening, envelope, scaling_factor / dist_max, E, order_of_magnitude
        #)
        #else:
    
    


def stress_envelope(sigma, tau, envelope='no_cap', scaling_factor=1, order_of_magnitude = 1, density = 250):
    
    print(f"WITHIN STRESS ENVELOPLE FUNCTION")
    print(f"Called scaling_factor={scaling_factor}")
    sigma = np.abs(np.asarray(sigma))
    tau = np.abs(np.asarray(tau))
    results = np.zeros_like(sigma)

    if envelope == 'no_cap':
        # Case for 'no_cap'
        # Use the provided scaling_factor or default to 1
        
        # Ensuring that we scale linearly above the density baseline, and not using whichever order_of_magntidue we have defined
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
    
    elif envelope == 'no_cap_OLD':
        
        sigma_c = 2.6  # (kPa) 6.16 / 2.6
        tau_c = 0.7     # (kPa) 5.09 / 0.7
        
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
        raise ValueError("Invalid envelope type. Choose 'no_cap' ")


def find_roots_around_x(x_value, skier, C, li, inclination, sigma_kPa, tau_kPa, x_cm, envelope='no_cap', scaling_factor=1, order_of_magnitude = 1, density = 250):
    # Define the lambda function for the root function
    func = lambda x: root_function(x, skier, C, li, inclination, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density)

    # Calculate the discrete distance to failure using the envelope function
    discrete_dist_to_fail = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density) - 1
    
    # Find indices where the envelope function transitions from positive to negative
    transition_indices = np.where(np.diff(np.sign(discrete_dist_to_fail)))[0]
    print("Transition indices:")
    print(transition_indices)
    print(discrete_dist_to_fail[transition_indices])
    
    
    print(discrete_dist_to_fail)
    
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

    # Print the results
    print(f"\033[91m Local Minima Indices: {local_minima_indices} \033[0m")
    print("Local Minima Values:", local_minima_values)
    print("Local Maxima Indices:", local_maxima_indices)
    print("Local Maxima Values:", local_maxima_values)


    
    # Extract the corresponding x_cm values at those transition indices
    root_candidates = []
    for idx in transition_indices:
        # Get the x_cm values surrounding the transition
        x_left = x_cm[idx]
        x_right = x_cm[idx+1]
        root_candidates.append((10*x_left, 10*x_right))
        # Adding one millimetre on each side

    # Print the root candidates
    print("Root candidates based on envelope function transitions:")
    for x_left, x_right in root_candidates:
        print(f"From x = {x_left} to x = {x_right}")

    # Search for roots within the identified candidates
    roots = []
    for x_left, x_right in root_candidates:
        try:
            root_result = root_scalar(func, bracket=[x_left, x_right], method='brentq')
            if root_result.converged:
                roots.append(root_result.root)
                print(f"Root found at x = {root_result.root}")
        except ValueError:
            print(f"No root found between x = {x_left} and x = {x_right}.")

    return roots

# The root function we seek to minimize
def root_function(x_value, skier, C, li, inclination, envelope='no_cap', scaling_factor=1, order_of_magnitude = 1, density = 250):
    sigma, tau = calculate_sigma_tau(x_value, skier, C, li, inclination)
    return stress_envelope(sigma, tau, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density) - 1


def find_segment_index(segment_lengths, coordinate):
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



def calculate_sigma_tau(x_value, skier, C, li, inclination):
    segment_index, coordinate_in_segment = find_segment_index(li, x_value)
    Z = skier.z(coordinate_in_segment, C, li[segment_index], inclination, bed=True)
    t = skier.tau(Z, unit='kPa')
    s = skier.sig(Z, unit='kPa')

    tau = -t[segment_index]  # Remember to switch sign
    sigma = s[segment_index]
    return sigma, tau



def find_new_crack_length_v3(snow_profile, skier_weight, inclination, li, ki, envelope='no_cap', scaling_factor=1, E = 0.25, order_of_magnitude = 1, density = 250, wl_thickness=30):


    total_length = np.sum(li)
    midpoint = total_length / 2
    
    # Here is new to see if this might be an issue?
    #n =  len(li)
    # Create the list with zeros
    # li = [0] * n
    #li[0] = total_length / 2
    #li[-1] = total_length / 2   
    #ki = n * [True]
    
    li = [midpoint, midpoint]
    ki = [True, True]
    
    # Kill if needed

    print("INSIDE FIND NEW CRACK LENGTH METHOD")
    print(f"Total Length: {total_length}")
    print(f"Midpoint: {midpoint}")
    
    

    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E=E, wl_thickness=wl_thickness
        ) 

    all_points_are_outside = np.min(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density)) > 1
    print(f"All Points Are Outside Envelope: {all_points_are_outside}")

    roots_x = find_roots_around_x(midpoint, skier, C, li, inclination, sigma_kPa, tau_kPa, x_cm, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density)
    print(f"Roots Found: {roots_x}")

    if len(roots_x) > 0:
        segment_boundaries = [0] + roots_x + [total_length]
        li_temp = np.diff(segment_boundaries).tolist()  # Convert to a list
        ki_temp = [True] * (len(segment_boundaries) - 1) 

        print(f"Segment Boundaries: {segment_boundaries}")
        print(f"li_temp: {li_temp}")
        print(f"Initial ki_temp: {ki_temp}")

        # Create a boolean list indicating root positions
        is_root = [False] * len(segment_boundaries)
        for root in roots_x:
            is_root[segment_boundaries.index(root)] = True

        print(f"Root Positions (is_root): {is_root}")

        # Iterate over the roots to determine cracked segments
        nbr_roots = len(roots_x)
        cracked_segment = True
        
        for i in range(1, len(is_root)):  # Start from the second root
            # Check if the current and previous boundaries are both roots
            if is_root[i] and (is_root[i - 1]) and cracked_segment:
                ki_temp[i - 1] = False  # Mark the segment as cracked
                cracked_segment = not cracked_segment
                
            elif is_root[i] and (is_root[i - 1]) and (not cracked_segment):
                # These are uncracked segments, i.e. they have support
                ki_temp[i - 1] = True
                cracked_segment = not cracked_segment
                
                # A cracked segment, if there exists more than one, will always switch between cracked and uncracked

        print(f"Updated ki_temp After Checking Roots: {ki_temp}")

        # Proceed to split li and ki at the midpoint
        li, ki = split_segments_at_midpoint(li_temp, ki_temp)
        print(f"Split li: {li}")
        print(f"Split ki: {ki}")

    elif all_points_are_outside:
        print("All points are outside the envelope. No cracks.")
        ki = [False] * len(ki)
    else:
        print("No roots found. Returning original segment lengths but now uncracked")
        # No changes to li and ki
        li = li
        ki = [True]*len(ki)

    # Calculate new crack length
    new_crack_length = sum(length for length, foundation in zip(li, ki) if not foundation)
    print(f"New Crack Length: {new_crack_length}")

    return new_crack_length, li, ki



def determine_segment_cracking(roots_x, total_length):
    # Check if we have any roots
    if len(roots_x) > 0:
        # Create segment boundaries including total_length
        segment_boundaries = [0] + roots_x + [total_length]
        li_temp = np.diff(segment_boundaries)

        # Initialize the helper vector to track cracked segments
        ki_temp = [True] * (len(li_temp) - 1)  # There will be one less segment than boundaries

        # Create a boolean list indicating root positions
        is_root = [False] * len(segment_boundaries)

        # Set True for each root in the middle
        for root in roots_x:
            is_root[segment_boundaries.index(root)] = True

        # We should have at least two roots here to determine segments
        nbr_roots = len(roots_x)

        # Iterate over the roots to determine cracked segments
        for i in range(1, nbr_roots):  # Start from the second root
            # Check if the current and previous boundaries are both roots
            if is_root[i] and is_root[i - 1]:
                ki_temp[i - 1] = False  # Mark the segment as cracked

        return ki_temp, is_root
    else:
        return [], []  # Return empty lists if no roots are found


def split_segments_at_midpoint(segment_lengths, segment_support):
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





def find_minimum_force_v4(snow_profile, inclination, li, ki, envelope='no_cap', scaling_factor=1, E = 0.25, order_of_magnitude = 1, density = 250, wl_thickness=30): 
    # Initial parameters
    crack_length = 0
    crack_case = 'nocrack'
    skier_weight = 1  # Starting weight of skier

    print("Starting the minimum force calculation...")
    print(f"Initial Skier Weight: {skier_weight}")
    print(f"Initial Crack Case: {crack_case}")

    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E, wl_thickness=wl_thickness
    )

    # Calculate the distance to failure
    dist_max = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
    dist_min = np.min(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
    print(f"Initial Distance to Failure: {dist_max}")
    
    if dist_min >= 1:
        print("Exception: we are outside the stress envelope in all points without any additional load")
        # We are outside the stress envelope without any additional skier weight
        return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min

    iteration_count = 0

    # Thursday changed 0.015 to 0.005

    while np.abs(dist_max - 1) > 0.005 and iteration_count < 50:   # While no point is outside the envelope
        print("Distance to failure is below threshold; increasing skier weight...")
        skier_weight = skier_weight / (dist_max)
        print(f"Updated Skier Weight: {skier_weight}")

        # Recreate the skier object with the updated weight
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
            snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E, wl_thickness=wl_thickness
        )

        # Recalculate the distance to failure
        dist_max = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
        dist_min = np.min(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
        print(f"New Distance to Failure: {dist_max}")
        iteration_count = iteration_count + 1

    if iteration_count == 50:
        print("Did not find critical skier weight. Calling dampened version.")
        print(f"\033[91m CALLING DAMPENED find_min_force_dampened_v3 \033[0m")
        skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min = find_minimum_force_dampened_v4(snow_profile, inclination, li, ki, envelope=envelope, scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude, dampening = 1, density = density, wl_thickness=wl_thickness)
        

    # Once the loop exits, it means we have found the critical skier weight
    print("Critical skier weight found. Exiting the calculation.")
    print(f"Final Skier Weight: {skier_weight}")
    print(f"Final Distance to Failure: {dist_max}")

    return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min




def find_minimum_force_dampened_v4(snow_profile, inclination, li, ki, envelope='no_cap', scaling_factor=1, E = 0.25, order_of_magnitude = 1, dampening = 1, density = 250, wl_thickness=30): 
    # Initial parameters
    crack_length = 0
    crack_case = 'nocrack'
    skier_weight = 1  # Starting weight of skier

    print("Starting the minimum force calculation...")
    print(f"Initial Skier Weight: {skier_weight}")
    print(f"Initial Crack Case: {crack_case}")

    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E, wl_thickness=wl_thickness
    )

    # Calculate the distance to failure
    dist_max = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
    dist_min = np.min(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
    print(f"Initial Distance to Failure: {dist_max}")
    
    if dist_min >= 1:
        print("Exception: we are outside the stress envelope in all points without any additional load")
        # We are outside the stress envelope without any additional skier weight
        return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min

    iteration_count = 0

    # Changed to 0.01 Thursday

    while np.abs(dist_max - 1) > 0.01 and iteration_count < 50:   # While no point is outside the envelope
        print("Distance to failure is below threshold; increasing skier weight...")
        skier_weight = (dampening + 1) * skier_weight / (dampening + dist_max)
        print(f"Updated Skier Weight: {skier_weight}")

        # Recreate the skier object with the updated weight
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
            snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E, wl_thickness=wl_thickness
        )

        # Recalculate the distance to failure
        dist_max = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
        dist_min = np.min(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude, density = density))
        print(f"New Distance to Failure: {dist_max}")
        iteration_count = iteration_count + 1

    if iteration_count == 50:
        print(f"Did not find critical skier weight. Calling dampened version with dampening {dampening}.")
        print(f"\033[91m CALLING DAMPENED AGAIN find_min_force_dampened_v3 with dampening {dampening+1} \033[0m")
        
        if dampening < 5:
            skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min = find_minimum_force_dampened_v4(snow_profile, inclination, li, ki, envelope=envelope, scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude, dampening = dampening + 1, density = density, wl_thickness=wl_thickness)
            
        else:
            print(f"\033[91m COULD NOT FIND CRITICAL SKIER WEIGHT \033[0m")
            return 0, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min

    # Once the loop exits, it means we have found the critical skier weight
    print("Critical skier weight found. Exiting the calculation.")
    print(f"Final Skier Weight: {skier_weight}")
    print(f"Final Distance to Failure: {dist_max}")

    return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_max, dist_min










def g_delta_diff_objective(crack_length, snow_profile, inclination, E, wl_thickness, target=1):
    # Initialize parameters
    length = 1000 * sum(layer[1] for layer in snow_profile)  # Total length (mm)
    li = [(length / 2 - crack_length / 2), (crack_length / 2), (crack_length / 2), (length / 2 - crack_length / 2)]  # Length segments
    ki = [True, False, False, True]  # Length of segments with foundations

    # Create skier object
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, 0, inclination, li, ki, crack_case='crack', E = E, wl_thickness=wl_thickness
    )
    
    # Calculate energy criteria
    energy_second_criterion = skier.gdif(C=C, phi=inclination, **segments)
    
    # We get it back in kJ actually
    g_delta_diff = energy_criterion(1000 * energy_second_criterion[1], 1000 * energy_second_criterion[2])
    
    # Return the difference from the target
    return g_delta_diff - target


def find_min_crack_length_self_propagation(snow_profile, inclination, E, wl_thickness, initial_interval=(1, 1000)):
    # Define the interval for crack_length search
    a, b = initial_interval
    
    # Use root_scalar to find the root
    result = root_scalar(
        g_delta_diff_objective,
        args=(snow_profile, inclination, E, wl_thickness),
        bracket=[a, b],  # Interval where the root is expected
        method='brentq'  # Brent's method
    )
    
    if result.converged:
        print(f"Root found: crack_length = {result.root}")
        return result.root
    else:
        print("Root search did not converge.")
        return None



def failure_envelope_mede(sigma, sample_type='s-RG1'):
    """
    Compute shear stress (τ) for a given compression strength (σ) based on the failure envelope parameters.

    Parameters:
    sigma (array-like): Array of compression strengths (σ)
    sample_type (str): Type of sample ('s-RG1', 's-RG2', or 's-FCDH')

    Returns:
    tau (np.array): Shear stresses calculated for the given σ values
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


def failure_envelope_present_no_cap(x, scaling_factor=1, order_of_magnitude = 1):
    """
    Computes the failure envelope (shear stress) based on the given input and scaling factor.

    Parameters:
    x (array) or single float: Normal stress values (sigma).
    scaling_factor (float): Scaling factor for the calculation (default is 1).

    Returns:
    np.ndarray: Shear stress values based on the failure envelope model.
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


def failure_envelope_present_no_cap_schottner(x, order_of_magnitude = 1, density = 250):
    """
    Computes the failure envelope (shear stress) based on the given input and scaling factor.

    Parameters:
    x (array) or single float: Normal stress values (sigma).
    scaling_factor (float): Scaling factor for the calculation (default is 1).

    Returns:
    np.ndarray: Shear stress values based on the failure envelope model.
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
    Function to calculate shear stress based on the Chandel failure envelope.
    Args:
    sigma (array): Normal stress values (in kPa)
    sample_type (str): The sample type, either 'FCsf' or 'FCso'
    
    Returns:
    tau (array): Shear stress values (in kPa)
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
    Compute G_II as a function of G_I for a vector of G_I values, based on the parametrization.
    Values of G_I outside the valid range (0 to G_Ic) are set to 0.
    
    Parameters:
    G_I: Mode I ERR
    
    Returns:
    np.array: A vector of corresponding G_II values.
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

















### Entirely new methods are above









def find_minimum_force_v3(snow_profile, inclination, li, ki, envelope='no_cap', scaling_factor=1, E = 0.25, order_of_magnitude = 1): 
    # Initial parameters
    crack_length = 0
    crack_case = 'nocrack'
    skier_weight = 1  # Starting weight of skier

    print("Starting the minimum force calculation...")
    print(f"Initial Skier Weight: {skier_weight}")
    print(f"Initial Crack Case: {crack_case}")

    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E
    )

    # Calculate the distance to failure
    distance_to_failure = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude))
    print(f"Initial Distance to Failure: {distance_to_failure}")

    iteration_count = 0

    while np.abs(distance_to_failure - 1) > 0.015 and iteration_count<50:   # While no point is outside the envelope
        print("Distance to failure is below threshold; increasing skier weight...")
        skier_weight = skier_weight / distance_to_failure
        print(f"Updated Skier Weight: {skier_weight}")

        # Recreate the skier object with the updated weight
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
            snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E
        )

        # Recalculate the distance to failure
        distance_to_failure = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude))
        print(f"New Distance to Failure: {distance_to_failure}")
        iteration_count = iteration_count + 1

    if iteration_count == 50:
        print("Did not find critical skier weight. Calling dampened version.")
        print(f"\033[91m CALLING DAMPENED find_min_force_dampened_v3 \033[0m")
        skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, distance_to_failure = find_minimum_force_dampened_v3(snow_profile, inclination, li, ki, envelope='no_cap', scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude, dampening = 1)
        

    # Once the loop exits, it means we have found the critical skier weight
    print("Critical skier weight found. Exiting the calculation.")
    print(f"Final Skier Weight: {skier_weight}")
    print(f"Final Distance to Failure: {distance_to_failure}")

    return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, distance_to_failure


def find_minimum_force_dampened_v3(snow_profile, inclination, li, ki, envelope='no_cap', scaling_factor=1, E = 0.25, order_of_magnitude = 1, dampening = 1): 
    # Initial parameters
    crack_length = 0
    crack_case = 'nocrack'
    skier_weight = 1  # Starting weight of skier

    print("Starting the minimum force calculation...")
    print(f"Initial Skier Weight: {skier_weight}")
    print(f"Initial Crack Case: {crack_case}")

    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E
    )

    # Calculate the distance to failure
    distance_to_failure = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude))
    print(f"Initial Distance to Failure: {distance_to_failure}")

    iteration_count = 0;

    while np.abs(distance_to_failure - 1) > 0.015 and iteration_count<50:   # While no point is outside the envelope
        print("Distance to failure is below threshold; increasing skier weight...")
        skier_weight = (dampening + 1) * skier_weight / (dampening + distance_to_failure)
        print(f"Updated Skier Weight: {skier_weight}")

        # Recreate the skier object with the updated weight
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
            snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E
        )

        # Recalculate the distance to failure
        distance_to_failure = np.max(stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude))
        print(f"New Distance to Failure: {distance_to_failure}")
        iteration_count = iteration_count + 1

    if iteration_count == 50:
        print(f"Did not find critical skier weight. Calling dampened version with dampening {dampening}.")
        print(f"\033[91m CALLING DAMPENED AGAIN find_min_force_dampened_v3 with dampening {dampening} \033[0m")
        
        if dampening < 5:
            skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, distance_to_failure = find_minimum_force_dampened_v3(snow_profile, inclination, li, ki, envelope='no_cap', scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude, dampening = dampening+1)
            
        else:
            
            print(f"\033[91m COULD NOT FIND CRITICAL SKIER WEIGHT \033[0m")

    # Once the loop exits, it means we have found the critical skier weight
    print("Critical skier weight found. Exiting the calculation.")
    print(f"Final Skier Weight: {skier_weight}")
    print(f"Final Distance to Failure: {distance_to_failure}")

    return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, distance_to_failure









def check_first_criterion_v4(snow_profile, inclination, skier_weight, envelope='no_cap', scaling_factor=1, E = 0.25, order_of_magnitude = 1):
    print(f"Called scaling_factor={scaling_factor}")
    # Time tracker
    start_time = time.time()
    elapsed_times = []

    # Trackers for skier weights, crack lengths, dist_max, and g_delta
    skier_weights = []
    crack_lengths = []
    dist_max_values = []  # Tracker for dist_max
    dist_min_values = []
    g_delta_values = []   # Tracker for g_delta

    # Initialize parameters
    length = 1000 * sum(layer[1] for layer in snow_profile)  # Total length (mm)
    k0 = [True, True, True, True]  # Support boolean for uncracked solution
    li = [length / 2, 0, 0, length / 2]  # Length segments
    ki = [True, True, True, True]  # Length of segments with foundations
    
    
    # Create initial skier object
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E
    )


    #### Unsure about how we handle this

    # This is needed to handle exception cases when we start outside the envelope
    c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case='crack', E=E
            )

    ### Unsure about how we handle this

    k0 = np.full(len(ki), True)
    incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
    g_delta = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])

    # Check if we are outside the stress envelope at any point
    distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude)
    dist_max = np.max(distance_to_failure)
    dist_min = np.min(distance_to_failure)

    # Initialize iteration variables
    iteration_count = 0
    max_iterations = 25

    if dist_max > 1 and not dist_min > 1:
        # Find minimum critical force to initialize our algorithm (BASE CASE)
        critical_skier_weight, *_ = find_minimum_force_v3(snow_profile, inclination, li, ki, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude)
        
        skier_weight = critical_skier_weight * 1.005
        max_skier_weight = 5 * skier_weight
        min_skier_weight = critical_skier_weight

        # Set initial crack length and error margin
        crack_length = 1  # Initial crack length
        err = 1000  # Error margin
        li = [length / 2 - crack_length / 2, crack_length / 2, crack_length / 2, length / 2 - crack_length / 2]
        ki = [True, False, False, True]
        
        while np.abs(err) > 0.002 and iteration_count < max_iterations and any(ki):
            # Track skier weight, crack length, dist_max, g_delta, and time for each iteration
            iteration_count += 1
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)
            dist_max_values.append(dist_max)  # Add dist_max value to the tracker
            dist_min_values.append(dist_min)
            elapsed_times.append(time.time() - start_time)

            # Create base_case with the correct number of segments
            k0 = [True] * len(ki)
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, k0, crack_case='nocrack', E = E
            )

            # Check distance to failure for uncracked solution
            distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude)
            dist_max = np.max(distance_to_failure)
            dist_min = np.min(distance_to_failure)

            # Solving a cracked solution, to calculate incremental ERR
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case='crack', E = E
            )

            # Calculate incremental energy released compared to uncracked solution
            k0 = np.full(len(ki), True)
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
            g_delta = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
            g_delta_values.append(g_delta) 
            
            # Update error margin
            err = np.abs(g_delta - 1)

            if iteration_count == 1 and (g_delta > 1 or err < 0.02):
                print(f"EXCEPTION: Energy criterion is fulfilled at minimum critical skier load: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, True, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values

            print(f"START OF ITERATION {iteration_count}: crack length: {crack_length} mm, Skier Weight: {skier_weight} kg, Max Distance to Failure: {dist_max}, Distance to energy envelope: {g_delta}")

            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            new_skier_weight = (min_skier_weight + max_skier_weight) / 2

            scaling = new_skier_weight / skier_weight

            if np.abs(err) > 0.002:
                skier_weight = new_skier_weight
                g_delta_last = g_delta
                new_crack_length, li, ki = find_new_crack_length_v3(snow_profile, skier_weight, inclination, li, ki, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude)
                crack_length = new_crack_length

                print(f"END OF ITERATION: g_delta: {g_delta} J/m^2, Old skier weight: {skier_weight} kg, Scaling: {scaling}, New skier weight: {skier_weight} kg, crack length: {crack_length} mm")

        # End of loop: convergence or max iterations reached
        if iteration_count < max_iterations and any(ki):
            if crack_length > 0:
                print(f"CONVERGENCE: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta_last, dist_max, g_delta_values, dist_max_values
            else:
                print("Crack length is zero and our skier load is lower than minimum critical load; redoing the algorithm with the dampened version.")
                skier_weight_average = np.average(skier_weights)
                variance = np.var(np.asarray(skier_weights), ddof=1)
                return check_first_criterion_dampened_v3(snow_profile, inclination, skier_weight, variance, envelope=envelope, scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude)

        elif not any(ki):
            print("We are outside the envelope at all points.")
            return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta_last, dist_min, g_delta_values, dist_min_values

        else:
            print("Maximum iterations reached without convergence.")
            skier_weight_average = np.average(skier_weights)
            variance = np.var(np.asarray(skier_weights), ddof=1)
            return check_first_criterion_dampened_v4(snow_profile, inclination, skier_weight, variance, envelope=envelope, scaling_factor=scaling_factor, E = E, order_of_magnitude = order_of_magnitude)

    elif dist_min > 1:
        print("Entire solution is cracked without any additional load - all points are outside the envelope")
        crack_length = length
        skier_weight = 0
        return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, True, False, skier_weight, g_delta, dist_min, g_delta_values, dist_min_values

    else:
        print(f"Entire solution is uncracked - all points are inside the envelope with max distance to envelope: {dist_max}")
        
        if dist_max > 0.9 and envelope=='no_cap':
            print(f"Since the distance to failure: {dist_max} is close to the boundary, we proceed to evaluate the stress_criterion for scaled version of the stress-envelope")
            return check_first_criterion_v4(snow_profile, inclination, skier_weight, envelope=envelope, scaling_factor=scaling_factor / dist_max, E=E, order_of_magnitude = order_of_magnitude)
        else:
            print(f"Since the distance to failure: {dist_max} is NOT close to the boundary, causing an avalanche is unlikely")
            return False, 0, skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, skier_weight, 0, dist_max, g_delta_values, dist_max_values










def check_first_criterion_dampened_v4(snow_profile, inclination, skier_weight, variance, envelope = 'no_cap', scaling_factor = 1, E = 0.25, order_of_magnitude = 1):
    
    standardized_std = np.sqrt(variance) / skier_weight * 10
    
    print(f" STANDARDIZED STD: {standardized_std}")

    print(f"Called scaling_factor={scaling_factor}")
    # Time tracker
    start_time = time.time()
    elapsed_times = []

    # Trackers for skier weights, crack lengths, dist_max, and g_delta
    skier_weights = []
    crack_lengths = []
    dist_max_values = []  # Tracker for dist_max
    dist_min_values = []
    g_delta_values = []   # Tracker for g_delta

    # Initialize parameters
    length = 1000 * sum(layer[1] for layer in snow_profile)  # Total length (mm)
    k0 = [True, True, True, True]  # Support boolean for uncracked solution
    li = [length / 2, 0, 0, length / 2]  # Length segments
    ki = [True, True, True, True]  # Length of segments with foundations

    # Create initial skier object
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E
    )
    
    # This is needed to handle exception cases when we start outside the envelope
    c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case='crack', E=E
            )

    k0 = np.full(len(ki), True)
    incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
    g_delta = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])

    # Check if we are outside the stress envelope at any point
    distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude)
    dist_max = np.max(distance_to_failure)
    dist_min = np.min(distance_to_failure)

    # Initialize iteration variables
    iteration_count = 0
    max_iterations = 50

    if dist_max > 1 and not dist_min > 1:
        # Find minimum critical force to initialize our algorithm (BASE CASE)
        critical_skier_weight, *_ = find_minimum_force_v3(snow_profile, inclination, li, ki, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude)

        skier_weight = critical_skier_weight * 1.005
        min_skier_weight = critical_skier_weight
        
        max_skier_weight = 2 * critical_skier_weight
        g_delta_max_weight = 0
        
        while g_delta_max_weight < 1:
            max_skier_weight = max_skier_weight * 2
            # Create base_case with the correct number of segments
            ki = [True, False, False, True]
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
                snow_profile, max_skier_weight, inclination, li, ki, crack_case='nocrack', E=E
            )

            # Solving a cracked solution, to calculate incremental ERR
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
                snow_profile, max_skier_weight, inclination, li, ki, crack_case='crack', E=E
            )

            # Calculate incremental energy released compared to uncracked solution
            k0 = np.full(len(ki), True)
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
            g_delta_max_weight = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
        
        

        # Set initial crack length and error margin
        crack_length = 1  # Initial crack length
        err = 1000  # Error margin
        li = [length / 2 - crack_length / 2, crack_length / 2, crack_length / 2, length / 2 - crack_length / 2]
        ki = [True, False, False, True]
        
        while np.abs(err) > 0.002 and iteration_count < max_iterations and any(ki):
            # Track skier weight, crack length, dist_max, g_delta, and time for each iteration
            iteration_count += 1
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)
            dist_max_values.append(dist_max)  # Add dist_max value to the tracker
            dist_min_values.append(dist_min)
            
            elapsed_times.append(time.time() - start_time)

            # Create base_case with the correct number of segments
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E=E
            )

            # Check distance to failure for uncracked solution
            distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor, order_of_magnitude = order_of_magnitude)
            dist_max = np.max(distance_to_failure)
            dist_min = np.min(distance_to_failure)

            # Solving a cracked solution, to calculate incremental ERR
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case='crack', E=E
            )

            # Calculate incremental energy released compared to uncracked solution
            k0 = np.full(len(ki), True)
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
            g_delta = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
            g_delta_values.append(g_delta)    # Add g_delta value to the tracker

            # If the energy criterion is satisfied directly, handle the exception
            if g_delta > 1 and iteration_count == 1:
                print(f"EXCEPTION: Energy criterion is fulfilled at minimum critical skier load: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, True, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values

            print(f"START OF ITERATION {iteration_count}: crack length: {crack_length} mm, Skier Weight: {skier_weight} kg, Max Distance to Failure: {dist_max}, Distance to energy envelope: {g_delta}")
            print(f" STANDARDIZED STD: {standardized_std}")
            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            new_skier_weight = (min_skier_weight + max_skier_weight) / 2

            # Dampened version of scaling, with more dampening for cases with higher variance
            scaling = new_skier_weight / skier_weight
            # scaling = (standardized_std + scaling) / (standardized_std + 1)
            scaling = (1 + scaling) / (2)
            
            # Update error margin
            err = np.abs(g_delta - 1)

            # Adjust skier weight based on error margin
            if np.abs(err) > 0.002:
                skier_weight = scaling * new_skier_weight

                # Find new crack length for the next iteration
                new_crack_length, li, ki = find_new_crack_length_v3(snow_profile, skier_weight, inclination, li, ki, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude)
                crack_length = new_crack_length

                print(f"END OF ITERATION: g_delta: {g_delta} J/m^2, Old skier weight: {skier_weight} kg, Scaling: {scaling}, New skier weight: {skier_weight} kg, crack length: {crack_length} mm")

        # End of loop: convergence or max iterations reached
        if iteration_count < max_iterations and any(ki):
            if crack_length > 0:
                print(f"CONVERGENCE: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values
            else:
                print("Crack length is zero and our skier load is lower than minimum critical load; redoing the algorithm with the dampened version.")
                skier_weight_average = np.average(skier_weights)
                variance = np.var(np.asarray(skier_weights), ddof=1)
                return check_first_criterion_dampened_v4(snow_profile, inclination, skier_weight, variance, envelope=envelope, scaling_factor=scaling_factor, E=E, order_of_magnitude = order_of_magnitude)

        elif not any(ki):
            print("We are outside the envelope at all points.")
            return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_min, g_delta_values, dist_min_values

        else:
            print("Maximum iterations reached without convergence - algorithm is NOT re-run.")
            return False, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_max, g_delta_values, dist_max_values

    elif dist_min > 1:
        print("Entire solution is cracked without any additional load - all points are outside the envelope")
        crack_length = length
        skier_weight = 0
        g_delta_last = 1
        return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, True, False, skier_weight, g_delta, dist_min, g_delta_values, dist_min_values

    else:
        print(f"Entire solution is uncracked - all points are inside the envelope with max distance to envelope: {dist_max}")
        
        if dist_max > 0.9 and envelope=='no_cap':
            print(f"Since the distance to failure: {dist_max} is close to the boundary, we proceed to evaluate the stress_criterion for scaled version of the stress-envelope")
            
            # NO VARIANCE HERE
            variance=1
            return check_first_criterion_dampened_v4(snow_profile, inclination, skier_weight, variance, envelope=envelope, scaling_factor=scaling_factor * dist_max, E=E, order_of_magnitude = order_of_magnitude)
        
        else:
            print(f"Since the distance to failure: {dist_max} is NOT close to the boundary, causing an avalanche is unlikely")
            return False, 0, skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, skier_weight, 0, dist_max, g_delta_values, dist_max_values









def check_first_criterion_v3(snow_profile, inclination, skier_weight, envelope='no_cap', scaling_factor=1, E = 0.25):

    print(f"Called scaling_factor={scaling_factor}")
    # Time tracker
    start_time = time.time()
    elapsed_times = []

    # Trackers for skier weights and crack lengths
    skier_weights = []
    crack_lengths = []

    # Initialize parameters
    length = 1000 * sum(layer[1] for layer in snow_profile)  # Total length (mm)
    # length = 10000
    k0 = [True, True, True, True]  # Support boolean for uncracked solution
    li = [length / 2, 0, 0, length / 2]  # Length segments
    ki = [True, True, True, True]  # Length of segments with foundations

    # Create initial skier object
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E
    )

    # Check if we are outside the stress envelope at any point
    distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor)
    dist_max = np.max(distance_to_failure)
    dist_min = np.min(distance_to_failure)


    # Initialize iteration variables
    iteration_count = 0
    max_iterations = 25

    if dist_max > 1 and not dist_min > 1:
        # Find minimum critical force to initialize our algorithm (BASE CASE)
        critical_skier_weight, *_ = find_minimum_force_v3(snow_profile, inclination, li, ki, envelope=envelope, scaling_factor=scaling_factor, E=E)
        
        # 
        skier_weight = critical_skier_weight*1.005 
        max_skier_weight = 4*skier_weight
        min_skier_weight = critical_skier_weight


        # Set initial crack length and error margin
        crack_length = 1  # Initial crack length
        err = 1000  # Error margin
        li = [length / 2 - crack_length / 2, crack_length / 2, crack_length / 2, length / 2 - crack_length / 2]
        ki = [True, False, False, True]
        
        
        while np.abs(err) > 0.002 and iteration_count < max_iterations and any(ki):

            # Track skier weight, crack length and time for each iteration
            iteration_count += 1
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)
            elapsed_times.append(time.time() - start_time)


            # Create base_case with the correct number of segments
            
            # Does it matter if we call k0 or k1 here?
            k0 = [True] * len(ki)
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, k0, crack_case = 'nocrack', E = E
            )

            # Check distance to failure for uncracked solution
            distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor)
            dist_max = np.max(distance_to_failure)
            dist_min = np.min(distance_to_failure)

            # Solving a cracked solution, to calculate incremental ERR
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case = 'crack', E = E
            )

            # Calculate incremental energy released compared to uncracked solution
            k0 = np.full(len(ki), True)
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
            g_delta = energy_criterion(1000*incr_energy[1], 1000*incr_energy[2])

            # Update error margin
            err = np.abs(g_delta-1)
            
            if iteration_count==1 and (g_delta>1 or err < 0.02) :
                # We have an exception where the energy criterion is fulfilled directly for our lowest possible weight, and we have found a solution
                print(f"EXCEPTION: Energy criterion is fulfilled at minimum critical skier load: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, True, critical_skier_weight, g_delta, dist_max

            print(f"START OF ITERATION {iteration_count}: crack length: {crack_length} mm, Skier Weight: {skier_weight} kg, Max Distance to Failure: {dist_max}, Distance to energy envelope: {g_delta}")

            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            new_skier_weight = (min_skier_weight + max_skier_weight) / 2 

            # scaling = g_delta ** (-1 / 12)
            scaling = new_skier_weight / skier_weight
            

            # Adjust skier weight based on error margin
            if np.abs(err) > 0.002:
                skier_weight = new_skier_weight
                g_delta_last = g_delta
                # Find new crack length for the next iteration
                new_crack_length, li, ki = find_new_crack_length_v3(snow_profile, skier_weight, inclination, li, ki, envelope=envelope, scaling_factor = scaling_factor, E=E)
                crack_length = new_crack_length

                print(f"END OF ITERATION: g_delta: {g_delta} J/m^2, Old skier weight: {skier_weight} kg, Scaling: {scaling}, New skier weight: {skier_weight} kg, crack length: {crack_length} mm")

        # End of loop: convergence or max iterations reached
        if iteration_count < max_iterations and any(ki):
            
            if crack_length > 0:
                print(f"CONVERGENCE: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta_last, dist_max
            
            else:
                print("Crack length is zero and our skier load is lower than minimum critical load; redoing the algorithm with the dampened version.")
                skier_weight_average = np.average(skier_weights)
                variance = np.var(np.asarray(skier_weights), ddof=1)
                return check_first_criterion_dampened_v3(snow_profile, inclination, skier_weight, variance, envelope=envelope, scaling_factor=scaling_factor, E = E)

        elif not any(ki):
            print("We are outside the envelope at all points.")
            return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta_last, dist_min

        else:
            print("Maximum iterations reached without convergence.")
            skier_weight_average = np.average(skier_weights)
            variance = np.var(np.asarray(skier_weights), ddof=1)
            return check_first_criterion_dampened_v3(snow_profile, inclination, skier_weight, variance, envelope=envelope, scaling_factor=scaling_factor, E=E)

    elif dist_min>1:
        print("Entire solution is cracked without any additional load - all points are outside the envelope")
        crack_length = length
        skier_weight = 0
        # How to best handle this?
        g_delta_last = 1
        return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths, True, False, skier_weight, g_delta_last, dist_max

    else:
        print(f"Entire solution is uncracked - all points are inside the envelope with max distance to envelope: {dist_max} ")
        
        if dist_max > 0.9: 
            print(f"Since the distance to failure: {dist_max} is close to the boundary, we proceed to evaluate the stress_criterion for scaled version of the stress-envelope")
            
            return check_first_criterion_v3(snow_profile, inclination, skier_weight, envelope=envelope, scaling_factor=scaling_factor/dist_max, E=E)
        
        else:
            print(f"Since the distance to failure: {dist_max} is NOT close to the boundary, causing an avalanche is unlikely") 
            return False, 0, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, skier_weight, 0, dist_max
            


def check_first_criterion_dampened_v3(snow_profile, inclination, skier_weight, variance, envelope='no_cap', scaling_factor=1, E = 0.25):

    standardized_std = np.sqrt(variance)/skier_weight*100
    print(f" STANDARDIZED STD: {standardized_std}")

    print(f"Called scaling_factor={scaling_factor}")
    # Time tracker
    start_time = time.time()
    elapsed_times = []

    # Trackers for skier weights and crack lengths
    skier_weights = []
    crack_lengths = []

    # Initialize parameters
    length = 1000 * sum(layer[1] for layer in snow_profile)  # Total length (mm)
    # length = 10000
    k0 = [True, True, True, True]  # Support boolean for uncracked solution
    li = [length / 2, 0, 0, length / 2]  # Length segments
    ki = [True, True, True, True]  # Length of segments with foundations

    # Create initial skier object
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, skier_weight, inclination, li, ki, crack_case = 'nocrack', E = E
    )

    # Check if we are outside the stress envelope at any point
    distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor)
    dist_max = np.max(distance_to_failure)
    dist_min = np.min(distance_to_failure)


    # Initialize iteration variables
    iteration_count = 0
    max_iterations = 50

    if dist_max > 1 and not dist_min > 1:
        # Find minimum critical force to initialize our algorithm (BASE CASE)
        critical_skier_weight, *_ = find_minimum_force_v3(snow_profile, inclination, li, ki, envelope=envelope, scaling_factor=scaling_factor, E=E)
        

        skier_weight = critical_skier_weight*1.005 
        max_skier_weight = 4*skier_weight
        min_skier_weight = critical_skier_weight


        # Set initial crack length and error margin
        crack_length = 1  # Initial crack length
        err = 1000  # Error margin
        li = [length / 2 - crack_length / 2, crack_length / 2, crack_length / 2, length / 2 - crack_length / 2]
        ki = [True, False, False, True]
        
        
        while np.abs(err) > 0.002 and iteration_count < max_iterations and any(ki):

            # Track skier weight, crack length and time for each iteration
            iteration_count += 1
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)
            elapsed_times.append(time.time() - start_time)


            # Create base_case with the correct number of segments
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack', E = E
            )

            # Check distance to failure for uncracked solution
            distance_to_failure = stress_envelope(sigma_kPa, tau_kPa, envelope=envelope, scaling_factor=scaling_factor)
            dist_max = np.max(distance_to_failure)
            dist_min = np.min(distance_to_failure)

            # Solving a cracked solution, to calculate incremental ERR
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(
                snow_profile, skier_weight, inclination, li, ki, crack_case='crack', E = E
            )

            # Calculate incremental energy released compared to uncracked solution
            k0 = np.full(len(ki), True)
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)
            g_delta = energy_criterion(1000*incr_energy[1], 1000*incr_energy[2])


            if g_delta > 1 and iteration_count==1:
                # We have an exception where the energy criterion is fulfilled directly for our lowest possible weight, and we have found a solution
                print(f"EXCEPTION: Energy criterion is fulfilled at minimum critical skier load: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, True, critical_skier_weight, g_delta, dist_max

            # scaling_in_theory = g_delta ** (-1 / 12)
            print(f"START OF ITERATION {iteration_count}: crack length: {crack_length} mm, Skier Weight: {skier_weight} kg, Max Distance to Failure: {dist_max}, Distance to energy envelope: {g_delta}")

            if g_delta < 1:
                min_skier_weight = skier_weight
            else:
                max_skier_weight = skier_weight

            new_skier_weight = (min_skier_weight + max_skier_weight) / 2 

            # Dampened version of scaling, with more dampening for cases with higher variance
            # scaling = g_delta ** (-1 / 12)
            scaling = new_skier_weight / skier_weight
            scaling = (standardized_std+scaling)/(standardized_std+1)

            # Update error margin
            err = np.abs(g_delta-1)

            # Adjust skier weight based on error margin
            if np.abs(err) > 0.002:
                skier_weight = new_skier_weight

                # Find new crack length for the next iteration
                new_crack_length, li, ki = find_new_crack_length_v3(snow_profile, skier_weight, inclination, li, ki, envelope=envelope, scaling_factor = scaling_factor, E=E)
                crack_length = new_crack_length

                print(f"END OF ITERATION: g_delta: {g_delta} J/m^2, Old skier weight: {skier_weight} kg, Scaling: {scaling}, New skier weight: {skier_weight} kg, crack length: {crack_length} mm")

        # End of loop: convergence or max iterations reached
        if iteration_count < max_iterations and any(ki):
            
            if crack_length > 0:
                print(f"CONVERGENCE: crack length: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {dist_max}")
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_max
            
            else:
                print("Crack length is zero and our skier load is lower than minimum critical load; redoing the algorithm with the dampened version.")
                skier_weight_average = np.average(skier_weights)
                variance = np.var(np.asarray(skier_weights), ddof=1)
                return check_first_criterion_dampened_v3(snow_profile, inclination, skier_weight, variance, envelope=envelope, scaling_factor=scaling_factor, E=E)

        elif not any(ki):
            print("We are outside the envelope at all points.")
            return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_min

        else:
            print("Maximum iterations reached without convergence - algorithm is NOT re-run.")
            return False, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, critical_skier_weight, g_delta, dist_max

    elif dist_min>1:
        print("Entire solution is cracked without any additional load - all points are outside the envelope")
        crack_length = length
        skier_weight = 0
        g_delta_last = 1
        return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths, True, False, skier_weight, g_delta, dist_min

    else:
        print(f"Entire solution is uncracked - all points are inside the envelope with max distance to envelope: {dist_max} ")
        
        if dist_max > 0.9: 
            
            # This will continue looping, but perhaps smart given that we are close to the envelope
            
            print(f"Since the distance to failure: {dist_max} is close to the boundary, we proceed to evaluate the stress_criterion for scaled version of the stress-envelope")
            
            return check_first_criterion_v3(snow_profile, inclination, skier_weight, envelope=envelope, scaling_factor=scaling_factor*dist_max, E=E)
        
        else:
            print(f"Since the distance to failure: {dist_max} is NOT close to the boundary, causing an avalanche is unlikely") 
            return False, 0, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths, False, False, skier_weight, 0, dist_max









def check_first_criterion_v2(snow_profile, inclination, skier_weight, envelope='reiweger'):
    # Assuming nocrack case to begin with and create a skier-object
    crack_length = 0
    length = 100 * (sum(layer[1] for layer in snow_profile))  # Total length (mm)
    
    # length = 100 * 100 * 10  # Total length (mm) (one hundred meters in mm)
    # length = 100 * (sum(layer[1] for layer in snow_profile)) 
    # length = sum(li)

    k0 = [True, True, True, True]  # Support boolean for uncracked solution
    li = [length / 2, 0, 0, length / 2]  # Support boolean for uncracked solution
    ki = [True, True, True, True]  # Length of segments with foundations as specified by ki

    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack')

    # For the initial skier object, check if we are outside the stress envelope at any point
    checker, dist_to_failure = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope=envelope)

    # Time tracker
    start_time = time.time()
    elapsed_times = []

    # Tracker lists for skier weights and crack lengths
    skier_weights = []
    crack_lengths = []

    # Initialize iteration variables
    iteration_count = 0
    max_iterations = 25

    if checker.any() and not checker.all():
        # Find the weight associated with minimum critical force to initialize our algorithm (BASE CASE)
        skier_weight, *_ = find_minimum_force_v2(snow_profile, inclination, li, ki, envelope=envelope) 

        initial_skier_weight = skier_weight

        # Initialize variables for algorithm
        init_crack_length = 1  # Initial crack length
        err = 1000  # Error margin
        li = [length / 2 - init_crack_length / 2, init_crack_length / 2, init_crack_length / 2, length / 2 - init_crack_length / 2]
        ki = [True, False, False, True]


        while np.abs(err) > 0.002 and iteration_count < max_iterations and any(ki):
            # ki.any() ensures we move out in case we are outside the envelope in all points

            # Increment iteration counter
            iteration_count += 1


            # Track skier weight and crack length for each iteration
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)

            # Time tracking for each iteration
            elapsed_times.append(time.time() - start_time)

            # Create the base_case with correct number of segments
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack')

            # Solving a cracked solution
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='crack')

            # print(f" Solution: {c_segments}")

            # Section to keep track of distance to failure
            checker_2, dist_to_failure_2 = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope=envelope)
            # WHY IS THIS NOT THE SAME DISTANCE TO FAILURE AS THE LAST ONE IN find_min_force

            print(f" START OF ITERATION {iteration_count}: cracklength: {crack_length} mm, Skier Weight: {skier_weight} kg, Max Distance to Failure: {np.max(dist_to_failure_2)}")


            # The uncracked solution will have True for all values
            k0 = np.full(len(ki), True)

            # Calculate incremental energy released compared to (BASE CASE) solution above
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)

            # Evaluate energy envelope (scaling by 1000 to convert kJ to J)
            # g_delta = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
            g_delta = energy_criterion(1000*incr_energy[1], 1000*incr_energy[2])
            
            scaling = (g_delta**(-1/10))

            # For any scaling below 1, we are not outside the energy envelope and must increase the force
            current_normal_force, current_tangential_force = c_skier.get_skier_load(skier_weight, inclination)
            current_gravitational_force = np.sqrt(current_normal_force ** 2 + current_tangential_force ** 2)
            updated_gravitational_force = scaling * (current_gravitational_force)



            # Updating error margin
            err = np.abs(updated_gravitational_force - current_gravitational_force) / updated_gravitational_force

            # For errors > margin we scale skier weight according to g_delta calculated above, and find new crack length
            if np.abs(err) > 0.002:
                new_skier_weight = skier_weight * scaling

                # For the updated skier force, we find all points where the weak layer is overloaded and use this as 

                # the crack length in the next iteration (with li and ki specifying how the crack is positioned in the weak layer)

                new_crack_length, li, ki = find_new_crack_length_v2(snow_profile, skier_weight, inclination, li, ki, envelope=envelope)
                crack_length = new_crack_length
                print(f" END OF ITERATION: g_delta: {g_delta} J/m^2, Old skier weight: {skier_weight} kg, Scaling: {scaling} kg, New skier weight: {new_skier_weight} kg, cracklength: {crack_length} mm,")

                skier_weight = new_skier_weight

        # End of loop, i.e., convergence or max iteration reached
        if iteration_count < max_iterations and any(ki):
            print(f" CONVERGENCE: cracklength: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {np.max(dist_to_failure_2)}")

            if crack_length>0:
                return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths
            else:
                print(f" Cracklength is zero and we redo the algorithm with the dampened version")
                skier_weight_average = np.average(skier_weights)
                variance = np.var(np.asarray(skier_weights), ddof=1)

                return check_first_criterion_damped_outside_envelope(snow_profile, inclination, skier_weight_average, variance, envelope=envelope)


        elif not any(ki):
            print("We are outside the envelope in all points")
            return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths
        else:
            # The issue of non-convergence is often bouncing too far above and too far below the solution. If we take the average of the approxima

            print("Maximum iterations reached without convergence.")
            skier_weight_average = np.average(skier_weights)

            variance = np.var(np.asarray(skier_weights), ddof=1)


            return check_first_criterion_damped(snow_profile, inclination, skier_weight_average, variance, envelope=envelope)

    elif checker.all():
        # Extreme case (i) we are outside the boundary at all points: the entire solution is cracked
        crack_length = length
        skier_weight = 0

        return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths

    else:
        # Extreme case (ii) We do not fulfill the stress criterion in any point, and will therefore not be able to trigger an avalanche
        return False, 0, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths




def check_first_criterion_damped_outside_envelope(snow_profile, inclination, skier_weight, variance, envelope='reiweger'):

    # We use the variance to determine a suitable coefficient to scale
    standardized_std = np.sqrt(variance)/skier_weight*100

    print(f" STANDARDIZED STD: {standardized_std}")
    # Assuming nocrack case to begin with and create a skier-object
    crack_length = 0
    length = 100 * (sum(layer[1] for layer in snow_profile))  # Total length (mm)
    # length = 100 * 100 * 10  # Total length (mm) (one hundred meters in mm)
    # length = 100 * (sum(layer[1] for layer in snow_profile)) 
    # length = sum(li)

    k0 = [True, True, True, True]  # Support boolean for uncracked solution
    li = [length / 2, 0, 0, length / 2]  # Support boolean for uncracked solution
    ki = [True, True, True, True]  # Length of segments with foundations as specified by ki

    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack')

    # For the initial skier object, check if we are outside the stress envelope at any point
    checker, dist_to_failure = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope=envelope)

    # Time tracker
    start_time = time.time()
    elapsed_times = []

    # Tracker lists for skier weights and crack lengths
    skier_weights = []
    crack_lengths = []

    # Initialize iteration variables
    iteration_count = 0
    max_iterations = 40

    if checker.any() and not checker.all():
        # Find the weight associated with minimum critical force to initialize our algorithm (BASE CASE)
        skier_weight, *_ = find_minimum_force_v2(snow_profile, inclination, li, ki, envelope=envelope) 

        # Initialize variables for algorithm
        init_crack_length = 1  # Initial crack length
        err = 1000  # Error margin
        li = [length / 2 - init_crack_length / 2, init_crack_length / 2, init_crack_length / 2, length / 2 - init_crack_length / 2]
        ki = [True, False, False, True]


        while np.abs(err) > 0.001 and iteration_count < max_iterations and any(ki):
            # ki.any() ensures we move out in case we are outside the envelope in all points

            # Increment iteration counter
            iteration_count += 1


            # Track skier weight and crack length for each iteration
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)

            # Time tracking for each iteration
            elapsed_times.append(time.time() - start_time)

            # Create the base_case with correct number of segments
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack')

            # Solving a cracked solution
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='crack')

            # print(f" Solution: {c_segments}")

            # Section to keep track of distance to failure
            checker_2, dist_to_failure_2 = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope=envelope)
            # WHY IS THIS NOT THE SAME DISTANCE TO FAILURE AS THE LAST ONE IN find_min_force

            print(f" START OF ITERATION {iteration_count}: cracklength: {crack_length} mm, Skier Weight: {skier_weight} kg, Max Distance to Failure: {np.max(dist_to_failure_2)}")

            # The uncracked solution will have True for all values
            k0 = np.full(len(ki), True)

            # Calculate incremental energy released compared to (BASE CASE) solution above
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)

            # Evaluate energy envelope (scaling by 1000 to convert kJ to J)
            # g_delta = energy_criterion(1000 * incr_energy[1], 1000 * incr_energy[2])
            g_delta = energy_criterion(incr_energy[1]*1000,  incr_energy[2]*1000)

            scaling = (g_delta**(-1/10))

            # We weight scaling to dampen the behavior
            # Greater standard deviations put more emphasis on dampening

            scaling = (standardized_std+scaling)/(standardized_std+1)


            # For any g_delta below 1, we are not outside the energy envelope and must increase the force
            current_normal_force, current_tangential_force = c_skier.get_skier_load(skier_weight, inclination)
            current_gravitational_force = np.sqrt(current_normal_force ** 2 + current_tangential_force ** 2)
            updated_gravitational_force = scaling * (current_gravitational_force)





            # We have now dampened the behavior

            # Updating error margin
            err = np.abs(updated_gravitational_force - current_gravitational_force) / updated_gravitational_force

            # For errors > margin we scale skier weight according to g_delta calculated above, and find new crack length
            if np.abs(err) > 0.001:
                new_skier_weight = skier_weight * scaling

                # For the updated skier force, we find all points where the weak layer is overloaded and use this as 

                # the crack length in the next iteration (with li and ki specifying how the crack is positioned in the weak layer)

                new_crack_length, li, ki = find_new_crack_length_v2(snow_profile, skier_weight, inclination, li, ki, envelope=envelope)
                crack_length = new_crack_length
                print(f" END OF ITERATION: g_delta: {g_delta} J/m^2, Old skier weight: {skier_weight} kg, Scaling: {scaling} kg, New skier weight: {new_skier_weight} kg, cracklength: {crack_length} mm,")

                skier_weight = new_skier_weight

        # End of loop, i.e., convergence or max iteration reached
        if iteration_count < max_iterations and any(ki):
            print(f" CONVERGENCE: cracklength: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {np.max(dist_to_failure_2)}")
            return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths
        
        elif not any(ki):
        
            print("We are outside the envelope in all points")
            return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths
        else:
            # The issue of non-convergence is often bouncing too far above and too far below the solution. If we take the average of the approxima

            print("Maximum iterations reached without convergence.")
            skier_weight = np.average(skier_weights)

            return False, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths

    elif checker.all():
        # Extreme case (i) we are outside the boundary at all points
        crack_length = length
        skier_weight = 0

        # Elapsed time is zero

        return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths

    else:
        # Extreme case (ii) We do not fulfill the stress criterion in any point, and will therefore not be able to trigger an avalanche
        
        return False, 0, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths









# This method is specifically used in instances when the original algorithm will not converge, as we bounce between solutions 

def check_first_criterion_damped(snow_profile, inclination, skier_weight, variance, envelope='reiweger'):
    # We use the variance to determine a suitable coefficient to scale
    standardized_std = np.sqrt(variance)/skier_weight*100

    print(f" STANDARDIZED STD: {standardized_std}")
    # Assuming nocrack case to begin with and create a skier-object
    crack_length = 0
    length = 100 * (sum(layer[1] for layer in snow_profile))  # Total length (mm)
    # length = 100 * 100 * 10  # Total length (mm) (one hundred meters in mm)
    # length = 100 * (sum(layer[1] for layer in snow_profile)) 
    # length = sum(li)

    k0 = [True, True, True, True]  # Support boolean for uncracked solution
    li = [length / 2, 0, 0, length / 2]  # Support boolean for uncracked solution
    ki = [True, True, True, True]  # Length of segments with foundations as specified by ki

    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack')

    # For the initial skier object, check if we are outside the stress envelope at any point
    checker, dist_to_failure = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope=envelope)

    # Time tracker
    start_time = time.time()
    elapsed_times = []

    # Tracker lists for skier weights and crack lengths
    skier_weights = []
    crack_lengths = []

    # Initialize iteration variables
    iteration_count = 0
    max_iterations = 60

    if checker.any() and not checker.all():
        # Find the weight associated with minimum critical force to initialize our algorithm (BASE CASE)
        skier_weight, *_ = find_minimum_force_v2(snow_profile, inclination, li, ki, envelope=envelope) 

        # Initialize variables for algorithm
        init_crack_length = 1  # Initial crack length
        err = 1000  # Error margin
        li = [length / 2 - init_crack_length / 2, init_crack_length / 2, init_crack_length / 2, length / 2 - init_crack_length / 2]
        ki = [True, False, False, True]

        while np.abs(err) > 0.01 and iteration_count < max_iterations and any(ki):
            # ki.any() ensures we move out in case we are outside the envelope in all points

            # Increment iteration counter
            iteration_count += 1


            # Track skier weight and crack length for each iteration
            skier_weights.append(skier_weight)
            crack_lengths.append(crack_length)

            # Time tracking for each iteration
            elapsed_times.append(time.time() - start_time)

            # Create the base_case with correct number of segments
            skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack')

            # Solving a cracked solution
            c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='crack')

            # print(f" Solution: {c_segments}")

            # Section to keep track of distance to failure
            checker_2, dist_to_failure_2 = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope=envelope)
            # WHY IS THIS NOT THE SAME DISTANCE TO FAILURE AS THE LAST ONE IN find_min_force

            print(f" START OF ITERATION {iteration_count}: cracklength: {crack_length} mm, Skier Weight: {skier_weight} kg, Max Distance to Failure: {np.max(dist_to_failure_2)}")


            # The uncracked solution will have True for all values
            k0 = np.full(len(ki), True)

            # Calculate incremental energy released compared to (BASE CASE) solution above
            incr_energy = c_skier.ginc(C0=C, C1=c_C, phi=inclination, **c_segments, k0=k0)

            # Evaluate energy envelope (scaling by 1000 to convert kJ to J)
            g_delta = energy_criterion(1000*incr_energy[1], 1000*incr_energy[2])


            scaling = (g_delta**(-1/10))


            # We weight scaling to dampen the behavior
            # Greater standard deviations put more emphasis on dampening

            scaling = (standardized_std+scaling)/(standardized_std+1)


            # For any g_delta below 1, we are not outside the energy envelope and must increase the force
            current_normal_force, current_tangential_force = c_skier.get_skier_load(skier_weight, inclination)
            current_gravitational_force = np.sqrt(current_normal_force ** 2 + current_tangential_force ** 2)
            updated_gravitational_force = scaling * (current_gravitational_force)

            # We have now dampened the behavior

            # Updating error margin
            err = np.abs(updated_gravitational_force - current_gravitational_force) / updated_gravitational_force

            # For errors > margin we scale skier weight according to g_delta calculated above, and find new crack length
            if np.abs(err) > 0.001:
                new_skier_weight = skier_weight * scaling

                # For the updated skier force, we find all points where the weak layer is overloaded and use this as 

                # the crack length in the next iteration (with li and ki specifying how the crack is positioned in the weak layer)

                new_crack_length, li, ki = find_new_crack_length_v2(snow_profile, skier_weight, inclination, li, ki, envelope=envelope)
                crack_length = new_crack_length
                print(f" END OF ITERATION: g_delta: {g_delta} J/m^2, Old skier weight: {skier_weight} kg, Scaling: {scaling} kg, New skier weight: {new_skier_weight} kg, cracklength: {crack_length} mm,")

                skier_weight = new_skier_weight

        # End of loop, i.e., convergence or max iteration reached
        if iteration_count < max_iterations and any(ki):
            print(f" CONVERGENCE: cracklength: {crack_length} mm, Critical Skier Weight: {skier_weight} kg, Distance to energy envelope: {g_delta} J/m^2, Max Distance to Stress Envelope: {np.max(dist_to_failure_2)}")
            return True, crack_length, skier_weight, c_skier, c_C, c_segments, c_x_cm, c_sigma_kPa, c_tau_kPa, iteration_count, elapsed_times, skier_weights, crack_lengths
        elif not any(ki):
            print("We are outside the envelope in all points")
            return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths
        else:
            # The issue of non-convergence is often bouncing too far above and too far below the solution. If we take the average of the approxima

            print("Maximum iterations reached without convergence.")
            skier_weight = np.average(skier_weights)

            return False, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths

    elif checker.all():
        # Extreme case (i) we are outside the boundary at all points
        crack_length = length
        skier_weight = 0

        # Elapsed time is zero

        return True, crack_length, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths

    else:
        # Extreme case (ii) We do not fulfill the stress criterion in any point, and will therefore not be able to trigger an avalanche
        return False, 0, skier_weight, None, None, None, None, None, None, iteration_count, elapsed_times, skier_weights, crack_lengths



def find_new_crack_length_v2(snow_profile, skier_weight_z, inclination, li, ki, envelope='reiweger'):

    # This version assumes a full-crack from first to last, without the uncracked segment in the middle

    crack_length = 0

    print("INSIDE FIND NEW CRACK LENGTH METHOD")
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(
        snow_profile, 
        
        
        skier_weight_z, inclination, li, ki, crack_case='nocrack') 

    # Check if we are outside the stress envelope

    checker, dist_to_failure = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope=envelope) 

    # Flattening arrays
    x_cm_array = np.array(10*x_cm).flatten()
    dist_to_fail_array = np.array(dist_to_failure).flatten()
    checker_array = np.array(checker).flatten()

    print("Distance to Failure Array:", dist_to_fail_array)
    #print("X cm Array:", x_cm_array)

    # Create a foundation object
    foundations = ~checker_array

    # Finding switch indices
    switches = np.where(np.diff(np.sign(dist_to_fail_array - 0.999)))[0]

    print("Switches Indices:", switches, x_cm[switches+1], dist_to_fail_array[switches+1])
    #print("Switches Indices:", switches, x_cm[switches+1], dist_to_fail_array[switches+1])

    # This new version just takes the first and last instances



    # Ensure switches + 1 are within bounds
    if switches.size > 0:

        start_crack = switches[0]
        end_crack = switches[-1]

        switches_short = np.array([start_crack, end_crack])
        print(type(switches_short)) 
        print(switches_short) 



        # OLD
        # ki = [foundations[0]]
        # ki.extend(foundations[switches + 1])

        # NEW
        ki = [foundations[0]]

        ki.extend(foundations[switches_short+1])
        # ki.extend(foundations[switches_short])

        # Constructing segment boundaries

        # OLD
        # segment_boundaries = [x_cm_array[0]]  # Start with the first point
        #segment_boundaries.extend(x_cm_array[switches+1])  # Append the points where switches occur
        #segment_boundaries.append(x_cm_array[-1])  # End with the last point

        # NEW - only two points of interest
        segment_boundaries = [x_cm_array[0]]  # Start with the first point
        segment_boundaries.extend(x_cm_array[switches_short+1])
        #segment_boundaries.extend(x_cm_array[switches_short])  # Append the 
        segment_boundaries.append(x_cm_array[-1]) # Last point


        # Calculate lengths between segment boundaries
        li = np.diff(segment_boundaries)

        # print("Lengths between segment boundaries (li):", li)

        # Calculate total length of li
        total_length = np.sum(li)
        midpoint = total_length / 2
        #print("Total Length of li:", total_length)
        #print("Midpoint of Total Length:", midpoint)

        # Locate the segment that brings the cumulative sum to the midpoint
        cumulative_length = np.cumsum(li)
        #print("Cumulative Lengths of li:", cumulative_length)

        # Find the index of the segment containing the midpoint
        segment_index = np.where(cumulative_length >= midpoint)[0][0]  # First segment to exceed midpoint
        #print("Segment Index at Midpoint:", segment_index)

        # If the segment index is valid, split that segment
        if segment_index < len(li):
            segment_start = cumulative_length[segment_index - 1] if segment_index > 0 else 0
            segment_to_split_length = li[segment_index]

            # Calculate the length to split
            split_length = midpoint - segment_start

            # Adjust the lengths
            li = np.insert(li, segment_index, [split_length, segment_to_split_length - split_length])
            li = np.delete(li, segment_index + 2)  # Remove the original segment that was split

            # Split the ki array accordingly
            # Duplicate the foundation state of the original segment
            ki = np.insert(ki, segment_index, ki[segment_index])  # Insert the existing boolean value twice
            print("New ki after splitting:", ki)

            print("New Lengths after Splitting Segment:", li)

    else:
        ki = [foundations[0],foundations[0], foundations[-1], foundations[-1]]  # If there are no switches, we are on the same side of the envelope in all points

        # This should mean that we are outside the envelope in all places
        total_length = np.sum(li)
        midpoint = total_length / 2

        li = [total_length/2,0,0, total_length/2] # We just make sure to split the segment in half



    # Calculate new crack length
    new_crack_length = sum(length for length, foundation in zip(li, ki) if not foundation)

    return new_crack_length, li, ki



def create_skier_object_v2(snow_profile, skier_weight_x, inclination, li_x, ki_x, crack_case='nocrack', E = 0.25, wl_thickness=30):

    # Define a skier object

    # Changing to 'skiers'
    skier = weac.Layered(system='skiers', layers=snow_profile)
    skier.set_foundation_properties(E = E, t= wl_thickness, update = True)

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

    # We also need to feed k0 for uncracked solution= which is 
    k0 = np.full(len(ki_x), True)


    # Calculate segments based on crack case: 'nocrack' or 'crack'
    segments = skier.calc_segments(
                            li=li_x,           # Use the lengths of the segments
                            ki=ki_x,
                            mi=mi_x,
                            k0=k0                 # Use the boolean flags
                            )[crack_case]     # Switch between 'crack' or 'nocrack'

    print(f"SEGMENT SOLUTIONS?: {segments}")

    # Solve and rasterize the solution
    C = skier.assemble_and_solve(phi=inclination, **segments)
    xsl_skier, z_skier, xwl_skier = skier.rasterize_solution(C=C, phi=(inclination), num=800, **segments)

    # Calculate compressions and shear stress
    x_cm, tau_kPa = skier.get_weaklayer_shearstress(x=xwl_skier, z=z_skier, unit='kPa')
    x_cm, sigma_kPa = skier.get_weaklayer_normalstress(x=xwl_skier, z=z_skier, unit='kPa')

    return skier, C, segments, x_cm, sigma_kPa, tau_kPa



def energy_criterion(G_sigma, G_tau):
    
    # Valle Nature Comms

    compression_toughness = 0.56
    n = 1/0.2 
    shear_toughness = 0.79
    m=1/0.45

    g_delta = ( np.abs(G_sigma) / compression_toughness )**n + ( np.abs(G_tau) / shear_toughness )**m 
    
    return g_delta


## These are transitioning state

def find_minimum_force_v2(snow_profile, inclination, li, ki, envelope='reiweger'):
    # Initial parameters
    crack_length = 0
    crack_case = 'nocrack'
    skier_weight = 1  # Starting weight of skier

    #ToDo: make this use v2

    # Create a skier object with no weight and check if it is already outside the envelope
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object_v2(snow_profile, skier_weight, inclination, li, ki, crack_case='nocrack') 

    # Check if we are already outside the stress envelope
    checker, dist_to_failure = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope='new')

    # Increment skier weight by 1kg until at least one point is outside the envelope
    while not checker.any():  # While no point is outside the envelope
        # skier_weight += 1  # Increase skier weight by 1kg

        # Recreate the skier object with the updated weight
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(snow_profile, crack_length, skier_weight, inclination, crack_case='nocrack')

        # Check again if we are outside the envelope with the new weight
        checker, dist_to_failure = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope=envelope)

        print(f"Skier Weight: {skier_weight} kg, Max Distance to Failure: {np.max(dist_to_failure)}")

        if not checker.any():
            skier_weight = skier_weight * 1/np.max(dist_to_failure)

    # Once the loop exits, it means we have found the critical skier weight
    return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_to_failure






















## None of the below are used currently



# Used for initialization of algorithm to find critical force
def create_skier_object(snow_profile, crack_length, skier_weight, inclination, crack_case='nocrack'):
    # Define a skier object
    skier = weac.Layered(system='skier', layers=snow_profile)

    # Assuming 100x snow profile thickness
    total_length = 100 * (sum(layer[1] for layer in snow_profile))  # Total length (mm)

    # Calculate segments based on crack case: 'nocrack' or 'crack'
    segments = skier.calc_segments(
                            L=total_length, 
                            a=crack_length, 
                            m=skier_weight  # Set current skier weight
                            )[crack_case]  # Switch between 'crack' or 'nocrack'

    # Solve and rasterize solution
    C = skier.assemble_and_solve(phi=inclination, **segments)
    xsl_skier, z_skier, xwl_skier = skier.rasterize_solution(C=C, phi=inclination, **segments)

    # Calculate compressions and shear stress
    x_cm, tau_kPa = skier.get_weaklayer_shearstress(x=xwl_skier, z=z_skier, unit='kPa')
    x_cm, sigma_kPa = skier.get_weaklayer_normalstress(x=xwl_skier, z=z_skier, unit='kPa')

    return skier, C, segments, x_cm, sigma_kPa, tau_kPa




def find_intersect_2(sigma, tau, envelope="reiweger"):
        # Ensure sigma and tau are arrays to handle vectors
        sigma = np.asarray(sigma)
        tau = np.asarray(tau)

        # Initialize a list to store the intersections for each pair of sigma and tau
        all_intersects = []

        # Loop over each sigma and tau pair
        for s, t in zip(sigma, tau):
            # Generate sigma values for the current sigma
            sigma_values = np.linspace(s, 3, 100)
            # Calculate the vectorized point for the current tau and sigma
            vector = vectorized_point_new(sigma_values, t / s)

            # Different cases for different envelopes
            if envelope == "reiweger":
                stress_envelope = failure_envelope_reiweger(sigma_values)
            elif envelope == "new":
                stress_envelope = failure_envelope_new(sigma_values)
            elif envelope == "smooth":
                stress_envelope = failure_envelope_smooth(sigma_values)
            elif envelope == "no_cap":
                stress_envelope = failure_envelope_present_no_cap(sigma_values)
            else:
                raise ValueError("Unsupported type of envelope")

            # Compute intersections where the vector crosses the envelope
            idx = np.argwhere(
                np.diff(np.sign(vector - stress_envelope))
            ).flatten()

            # If intersections are found, extract the corresponding sigma values
            if idx.size > 0:
                intersect_sigma = sigma_values[
                    idx
                ]  # Extract sigma values at intersection indices
                all_intersects.append(
                    intersect_sigma
                )  # Add found intersections to the list
            else:
                all_intersects.append(
                    np.array([])
                )  # Append an empty array for no intersections

        # Return the list of intersections, ensuring each input pair has a corresponding output
        return all_intersects




def distance_to_failure_2(sigma, tau, envelope='reiweger'):
    # Ensure sigma and tau are arrays to handle vectors
    sigma = np.asarray(sigma)
    tau = np.asarray(tau)
    
    # Handle single-point (scalar) inputs by converting them to 1-element arrays
    if sigma.ndim == 0:
        sigma = np.array([sigma])
    if tau.ndim == 0:
        tau = np.array([tau])

    # Calculate the slope and the magnitude (vector_abs)
    slope = tau / sigma
    vector_abs = np.sqrt(sigma**2 + tau**2)

    # Find intersections with the envelope - first column is the actual point, second column is where the intersection happened
    intersect_sigma_list = find_intersect_2(sigma, tau, envelope=envelope)

    distance_factors = []

    # Iterate through each intersection for each (sigma, tau) pair
    for i, (intersect_sigma, current_slope) in enumerate(zip(intersect_sigma_list, slope)):
        if intersect_sigma.size > 0:  # We have found an intersect with sigma_value specified at second column
            # Compute total distance to the envelope for each intersection
            total_distance_to_envelope = np.sqrt(intersect_sigma[1]**2 + vectorized_point_new(intersect_sigma[1], slope[i])**2)
            # Calculate distance factor for this intersection
            distance_factor = vector_abs[i] / total_distance_to_envelope
            # Append the distance factor to the list
            distance_factors.append(distance_factor)
        else:  # We have not found an intersect and are inside the envelope
            x_values = np.linspace(-8, 1, 100)  # We must extrapolate the slope outside the envelope
            intersect = find_intersect_2(x_values, vectorized_point_new(x_values, current_slope), envelope=envelope)
            intersect_tau = vectorized_point_new(intersect[0][1], current_slope)
            total_distance_to_envelope = np.sqrt(intersect[0][1]**2 + intersect_tau**2)
            distance_factor = vector_abs[i] / total_distance_to_envelope
            # Append the distance factor to the list
            distance_factors.append(distance_factor)

    # If we only had single-point inputs, return a single distance factor instead of a list
    if len(distance_factors) == 1:
        return distance_factors[0]
    return distance_factors  # Return all distance factors for each pair




def distance_to_failure(intersect_sigma, point_sigma, point_tau):

    # Would like to update this to take in the envelope we are comparing against, instead of intersect_sigma

    vector_abs = np.sqrt(point_sigma**2 + point_tau**2)
    slope = point_tau/point_sigma

    if np.any(intersect_sigma):
        #We are outside the envelope as intersect is not empty
        total_distance_to_envelope = np.sqrt(intersect_sigma**2 + vectorized_point(intersect_sigma,slope)**2)
        distance_factor = vector_abs / total_distance_to_envelope

        return distance_factor
    else:
        # We are inside the envelope: need to find the intersect of the extrapolated vector
        x_values = np.linspace(-8,0,100)

        intersect = find_intersect(x_values, vectorized_point(x_values,slope), failure_envelope(x_values))
        intersect_tau = vectorized_point(intersect, slope)

        total_distance_to_envelope = np.sqrt(intersect**2 + intersect_tau**2)

        distance_factor = vector_abs/total_distance_to_envelope

        return distance_factor


def find_intersect(x_values, vectorized_point, envelope):
    idx = np.argwhere(np.diff(np.sign(vectorized_point - envelope))).flatten()
    intersect_x = x_values[idx]

    return intersect_x


def failure_envelope(x):
    a = -2.75
    return np.where((x <= 0) & (x > a), np.sqrt(1 - (x**2 / a**2)),0)



def failure_envelope_new(x):
    # Ensure x is treated as a NumPy array for vectorized operations
    x = np.asarray(x)

    a = -2.75
    # Apply conditions using np.where: If (x <= 0) & (x > a), compute the square root, otherwise return 0
    return np.where((x <= 0) & (x >= a), np.sqrt(1 - (x**2 / a**2)), 0)





def failure_envelope_smooth(x):
    x = np.asarray(x)

    sigma_c_plus = 0.4        # (kPa)
    sigma_c_minus = 2.6      # (kPa)
    w = 5                     # Shaprness of transition into capped region
    theta = np.pi/6                # Internal friction angle of the material

    tau_c = 0.7               # (kPa) - NOT USED

    return np.where( ( x < tau_c), (sigma_c_plus-x)*np.tan(theta)*np.tanh(w*(x-sigma_c_minus)), 0)


def failure_envelope_reiweger(x):
    # Approximate linear interpolation for two line segments defined by a,b,c below
    a = -3
    b = -2.75
    c = 0.25

    # y-values at the specified points
    y_a = 0
    y_b = 1
    y_c = 0

    slope1 = (y_b - y_a) / (b - a)
    slope2 = (y_c - y_b) / (c - b)

    # Convert scalar inputs to an array for consistent processing
    x = np.asarray(x)

    # Initialize result array with zeros
    result = np.zeros_like(x)

    # Apply conditions
    mask1 = (x <= a)
    mask2 = (a < x) & (x < b)
    mask3 = (b <= x) & (x < c)

    result[mask1] = 0
    result[mask2] = y_a + x[mask2] * slope1
    result[mask3] = y_b + x[mask3] * slope2
    result[x >= c] = 0

    return result



def vectorized_point(x, slope):
    # Ensure x and slope are arrays to handle vectors
    x = np.asarray(x)
    slope = np.asarray(slope)

    # Return the element-wise product of x and slope
    return x * slope



def find_minimum_force(snow_profile, inclination, envelope='reiweger'):
    # Initial parameters
    crack_length = 0
    crack_case = 'nocrack'
    skier_weight = 1  # Starting weight of skier

    #ToDo: make this use v2

    # Create a skier object with no weight and check if it is already outside the envelope
    skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(snow_profile, crack_length, skier_weight, inclination, crack_case='nocrack') 

    # Check if we are already outside the stress envelope
    checker, dist_to_failure = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope='new')

    # Increment skier weight by 1kg until at least one point is outside the envelope
    while not checker.any():  # While no point is outside the envelope
        # skier_weight += 1  # Increase skier weight by 1kg

        # Recreate the skier object with the updated weight
        skier, C, segments, x_cm, sigma_kPa, tau_kPa = create_skier_object(snow_profile, crack_length, skier_weight, inclination, crack_case='nocrack')

        # Check again if we are outside the envelope with the new weight
        checker, dist_to_failure = is_outside_stress_envelope(sigma_kPa, -tau_kPa, envelope=envelope)

        print(f"Skier Weight: {skier_weight} kg, Max Distance to Failure: {np.max(dist_to_failure)}")

        if not checker.any():
            skier_weight = skier_weight * 1/np.max(dist_to_failure)

    # Once the loop exits, it means we have found the critical skier weight
    return skier_weight, skier, C, segments, x_cm, sigma_kPa, tau_kPa, dist_to_failure



def vectorized_point_new(x, slope):
    # Ensure x and slope are arrays to handle vectors
    x = np.asarray(x)
    slope = np.asarray(slope)

    # Return the element-wise product of x and slope
    return np.outer(slope, x)



def is_outside_stress_envelope(sigma, tau, envelope='reiweger'):
    # Ensure sigma and tau are arrays to handle vectors
    sigma = np.asarray(sigma)
    tau = np.asarray(tau)

    # Calculate the distance to failure for each point
    dist_to_fail = distance_to_failure_2(sigma, tau, envelope=envelope)

    # Convert dist_to_fail to a NumPy array if it's a list
    dist_to_fail = np.asarray(dist_to_fail)

    # Check if each distance is greater than 1 and mark as outside (True) or inside (False)
    outside_envelope = dist_to_fail >= 1  # This will now work if dist_to_fail is a NumPy array

    return outside_envelope, dist_to_fail  # Return two separate arrays


