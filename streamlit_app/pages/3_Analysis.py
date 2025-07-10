import sys
from typing import List, Literal, cast, Tuple, Optional, Dict, Any, Union
import streamlit as st
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.optimize import brentq
from matplotlib.patches import Rectangle, Patch
from matplotlib.figure import Figure

sys.path.append("/home/pillowbeast/Documents/weac")

from weac_2.analysis.analyzer import Analyzer
from weac_2.analysis.criteria_evaluator import CriteriaEvaluator, FindMinimumForceResult, CoupledCriterionResult
from weac_2.analysis.plotter import Plotter
from weac_2.components import (
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac_2.core.system_model import SystemModel

# Core functions from notebook
def _evaluate_system(system: SystemModel, criteria_evaluator: CriteriaEvaluator):
    """Evaluate a system and return stress/energy results"""
    analyzer = Analyzer(system)
    xsl, z, xwl = analyzer.rasterize_solution(mode="cracked", num=2000)
    fq = analyzer.sm.fq

    sigma_kPa = fq.sig(z, unit="kPa")
    tau_kPa = fq.tau(z, unit="kPa")
    stress_envelope = criteria_evaluator.stress_envelope(sigma_kPa, tau_kPa, system.weak_layer)

    DERR = analyzer.differential_ERR(unit="J/m^2")
    IERR = analyzer.incremental_ERR(unit="J/m^2")
    DERR_tot = DERR[0]
    DERR_I = DERR[1]
    DERR_II = DERR[2]
    IERR_tot = IERR[0]
    IERR_I = IERR[1]
    IERR_II = IERR[2]
    
    DERR_crit = criteria_evaluator.fracture_toughness_envelope(DERR_I, DERR_II, system.weak_layer)
    IERR_crit = criteria_evaluator.fracture_toughness_envelope(IERR_I, IERR_II, system.weak_layer)
    
    return xsl, z, xwl, stress_envelope, DERR_crit, DERR_tot, DERR_I, DERR_II, IERR_crit, IERR_tot, IERR_I, IERR_II

def update_segments(segments: List[Segment], crack_mid_point: float, crack_length: float) -> List[Segment]:
    """Update segments based on crack parameters"""
    new_segments = []
    covered_length = 0
    for segment in segments:
        start_point = covered_length
        end_point = covered_length + segment.length
        
        # segment to the left of the crack
        if end_point < crack_mid_point - crack_length/2:
            new_segments.append(segment)
            covered_length += segment.length
        # segment to the right of the crack
        elif start_point > crack_mid_point + crack_length/2:
            new_segments.append(segment)
            covered_length += segment.length
        # crack in the middle of the segment
        elif start_point < crack_mid_point - crack_length/2 and end_point > crack_mid_point + crack_length/2:
            new_segments.append(Segment(length=crack_mid_point - crack_length/2 - covered_length, has_foundation=segment.has_foundation, m=0))
            new_segments.append(Segment(length=crack_length, has_foundation=False, m=0))
            new_segments.append(Segment(length=segment.length - (crack_mid_point + crack_length/2 - covered_length), has_foundation=segment.has_foundation, m=segment.m))
            covered_length += segment.length
        # crack touches the right side of the segment
        elif end_point < crack_mid_point + crack_length/2:
            new_segments.append(Segment(length=crack_mid_point - crack_length/2 - covered_length, has_foundation=segment.has_foundation, m=0))
            new_segments.append(Segment(length=segment.length - (crack_mid_point - crack_length/2 - covered_length), has_foundation=False, m=segment.m))
            covered_length += segment.length
        # crack touches the left side of the segment
        elif start_point < crack_mid_point + crack_length / 2:
            new_segments.append(Segment(length=crack_mid_point + crack_length/2 - covered_length, has_foundation=False, m=0))
            new_segments.append(Segment(length=segment.length - (crack_mid_point + crack_length/2 - covered_length), has_foundation=segment.has_foundation, m=segment.m))
            covered_length += segment.length
    return new_segments

def plot_system_evaluation_with_params(system: SystemModel, criteria_evaluator: CriteriaEvaluator, window_size: int):
    """Plot system evaluation with adjustable parameters showing all four cases"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)

    # Get all computed results
    computed_results = st.session_state.computed_results
    
    # Define colors and labels for each case
    cases = {
        "current": {"system": system, "color": "blue", "label": "Current Segments", "linestyle": "-"},
        "coupled_criterion": {"color": "red", "label": "Coupled Criterion", "linestyle": "-"},
        "minimum_force": {"color": "green", "label": "Minimum Force", "linestyle": "-"},
        "minimum_crack_length": {"color": "orange", "label": "Minimum Crack Length", "linestyle": "-"}
    }
    
    # Store all stress envelopes and positions
    all_data = {}
    
    # Calculate stress envelope for each case
    for case_name, case_info in cases.items():
        try:
            if case_name == "current":
                current_system = case_info["system"]
            elif computed_results[case_name] is not None:
                current_system = computed_results[case_name]["system"]
            else:
                continue

            # Evaluate this system
            xsl, z, xwl, stress_envelope, DERR_crit, DERR_tot, DERR_I, DERR_II, IERR_crit, IERR_tot, IERR_I, IERR_II = _evaluate_system(current_system, criteria_evaluator)

            # Store the data
            all_data[case_name] = {
                "xsl": xsl,
                "xwl": xwl,
                "stress_envelope": stress_envelope,
                "DERR_crit": DERR_crit,
                "IERR_crit": IERR_crit,
                "system": current_system
            }
            
        except Exception as e:
            print(f"Error processing {case_name}: {e}")
            continue
    
    # Use window from basic case for consistency
    if "current" in all_data:
        xsl_ref = all_data["current"]["xsl"]
        x_mid = (xsl_ref[0] + xsl_ref[-1]) / 2
        window_start = x_mid - window_size/2
        window_end = x_mid + window_size/2
    else:
        # Fallback if basic case not available
        window_start = -window_size/2
        window_end = window_size/2
    
    # Plot critical threshold line
    ax.hlines(1, window_start, window_end, color="black", linestyle="--", alpha=0.7, label="Critical threshold")
    
    # Plot stress envelopes for each case
    for case_name, case_info in cases.items():
        if case_name not in all_data:
            continue
            
        data = all_data[case_name]
        xsl = data["xsl"]
        xwl = data["xwl"]
        stress_envelope = data["stress_envelope"]
        
        # Filter data to window
        mask = (xsl > window_start) & (xsl < window_end)
        x_orig = xsl[mask]
        xwl_orig = xwl[mask]
        stress_orig = stress_envelope[mask]
        
        # Plot stress envelope
        ax.plot(xwl_orig, stress_orig, 
                color=case_info["color"], 
                linewidth=2, 
                linestyle=case_info["linestyle"],
                label=f"{case_info['label']} Stress Envelope")
    
    # Plot all DERR and IERR
    for case_name, case_info in cases.items():
        if case_name not in all_data:
            continue
        data = all_data[case_name]
        xsl = data["xsl"]
        xwl = data["xwl"]
        stress_envelope = data["stress_envelope"]
        DERR_crit = data["DERR_crit"]
        IERR_crit = data["IERR_crit"]
        
        # Filter data to window
        mask = (xsl > window_start) & (xsl < window_end)
        x_orig = xsl[mask]
        xwl_orig = xwl[mask]
        stress_orig = stress_envelope[mask]
        
        derr = np.full_like(x_orig, DERR_crit)
        ierr = np.full_like(x_orig, IERR_crit)
        
        # Plot DERR and IERR where xwl is NaN (no crack in weak layer)
        mask_no_crack = np.isnan(xwl_orig)
        if np.any(mask_no_crack):
            ax.plot(x_orig[mask_no_crack], derr[mask_no_crack], 
                    color=case_info["color"], linewidth=2, linestyle="-", label=f"{case_info['label']} DERR Critical")
            ax.plot(x_orig[mask_no_crack], ierr[mask_no_crack], 
                    color=case_info["color"], linewidth=2, linestyle="--", label=f"{case_info['label']} IERR Critical")

    # Formatting
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Stress/Energy Release Rate")
    ax.set_title(f"Stress Analysis Comparison - All Cases (Window: {window_size}mm)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Set reasonable y-limits based on all data
    all_derrs = []
    all_ierrs = []
    for data in all_data.values():
        all_derrs.append(data["DERR_crit"])
        all_ierrs.append(data["IERR_crit"])
    y_max = max(all_derrs + all_ierrs)
    ax.set_ylim(0, y_max * 1.1)

    plt.tight_layout()
    return fig

def plot_stress_envelope_comparison(selected_cases: List[str], criteria_evaluator: CriteriaEvaluator):
    """Plot stress envelope in τ-σ space for selected cases"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    computed_results = st.session_state.computed_results
    colors = {"current": "blue", "coupled_criterion": "red", "minimum_force": "green", "minimum_crack_length": "orange"}
    
    for case_name in selected_cases:
        if computed_results[case_name] is not None:
            system_model = computed_results[case_name]["system"]
        else:
            continue
            
        analyzer = Analyzer(system_model)
        _, z, _ = analyzer.rasterize_solution(num=10000)
        fq = system_model.fq

        # Calculate stresses
        sigma = np.abs(fq.sig(z, unit="kPa"))
        tau = fq.tau(z, unit="kPa")

        # Plot stress path
        ax.plot(sigma, tau, "-", linewidth=2, color=colors[case_name], label=f"{case_name.replace('_', ' ').title()} Stress Path")
        ax.scatter(sigma[0], tau[0], color=colors[case_name], s=10, marker="o", alpha=0.7)
        ax.scatter(sigma[-1], tau[-1], color=colors[case_name], s=10, marker="s", alpha=0.7)

    # Plot envelope for reference case
    if selected_cases:
        reference_case = selected_cases[0]
        if reference_case == "current" and "current_system" in st.session_state:
            reference_system = st.session_state.current_system
        elif computed_results[reference_case] is not None:
            reference_system = computed_results[reference_case]["system"]
        else:
            reference_system = None
            
        if reference_system is not None:
            weak_layer = reference_system.weak_layer
            
            def find_sigma_for_tau(tau_val, sigma_c, method: Optional[str] = None):
                def envelope_root_func(sigma_val):
                    return criteria_evaluator.stress_envelope(sigma_val, tau_val, weak_layer, method=method) - 1
                try:
                    search_upper_bound = sigma_c * 1.1
                    sigma_root = brentq(envelope_root_func, a=0, b=search_upper_bound, 
                                       xtol=1e-6, rtol=1e-6)
                    return sigma_root
                except ValueError:
                    return np.nan

            method = criteria_evaluator.criteria_config.stress_envelope_method
            config = criteria_evaluator.criteria_config
            density = weak_layer.rho
            
            # Calculate tau_c and sigma_c based on method
            if method == "adam_unpublished":
                scaling_factor = config.scaling_factor
                order_of_magnitude = config.order_of_magnitude
                if scaling_factor > 1:
                    order_of_magnitude = 0.7
                if scaling_factor < 0.55:
                    scaling_factor = 0.55
                tau_c = 5.09 * (scaling_factor**order_of_magnitude)
                sigma_c = 6.16 * (scaling_factor**order_of_magnitude)
            elif method == "schottner":
                rho_ice = 916.7
                sigma_y = 2000
                sigma_c_adam = 6.16
                tau_c_adam = 5.09
                order_of_magnitude = config.order_of_magnitude
                sigma_c = sigma_y * 13 * (density / rho_ice) ** order_of_magnitude
                tau_c = tau_c_adam * (sigma_c / sigma_c_adam)
            elif method == "mede_s-RG1":
                tau_c = 3.53
                sigma_c = 7.00
            elif method == "mede_s-RG2":
                tau_c = 1.22
                sigma_c = 2.33
            elif method == "mede_s-FCDH":
                tau_c = 0.61
                sigma_c = 1.49
            else:
                tau_c = 5.09
                sigma_c = 6.16

            tau_range = np.linspace(0, tau_c, 100)
            sigma_envelope = np.array([find_sigma_for_tau(t, sigma_c, method) for t in tau_range])

            # Remove nan values
            valid_points = ~np.isnan(sigma_envelope)
            valid_tau_range = tau_range[valid_points]
            sigma_envelope = sigma_envelope[valid_points]

            ax.plot(sigma_envelope, valid_tau_range, "--", linewidth=2, label=f"{method} Envelope", color="black")
            ax.plot(-sigma_envelope, valid_tau_range, "--", linewidth=2, color="black", alpha=0.5)
            ax.plot(-sigma_envelope, -valid_tau_range, "--", linewidth=2, color="black", alpha=0.5)
            ax.plot(sigma_envelope, -valid_tau_range, "--", linewidth=2, color="black", alpha=0.5)

    # Formatting
    ax.set_xlabel("Compressive Strength σ (kPa)")
    ax.set_ylabel("Shear Strength τ (kPa)")
    ax.set_title("Weak Layer Stress Envelope Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)

    plt.tight_layout()
    return fig

def plot_err_envelope_comparison(selected_cases: List[str], criteria_evaluator: CriteriaEvaluator):
    """Plot ERR envelope for selected cases"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    computed_results = st.session_state.computed_results
    colors = {"current": "blue", "coupled_criterion": "red", "minimum_force": "green", "minimum_crack_length": "orange"}
    
    for case_name in selected_cases:
        if computed_results[case_name] is not None:
            system_model = computed_results[case_name]["system"]
        else:
            continue
            
        analyzer = Analyzer(system_model)
        incr_energy = analyzer.incremental_ERR(unit="J/m^2")
        G_I = incr_energy[1]
        G_II = incr_energy[2]
        
        diff_energy = analyzer.differential_ERR(unit="J/m^2")
        DERR_I = diff_energy[1]
        DERR_II = diff_energy[2]

        # Plot ERR path
        ax.scatter(np.abs(G_I), np.abs(G_II), color=colors[case_name], s=50, marker="o", 
                  label=f"{case_name.replace('_', ' ').title()} Incremental ERR", alpha=0.7)
        ax.scatter(np.abs(DERR_I), np.abs(DERR_II), color=colors[case_name], s=50, marker="s", 
                  label=f"{case_name.replace('_', ' ').title()} Differential ERR", alpha=0.7)

    # Plot envelope for reference case
    if selected_cases:
        reference_case = selected_cases[0]
        if computed_results[reference_case] is not None:
            reference_system = computed_results[reference_case]["system"]
        else:
            reference_system = None
            
        if reference_system is not None:
            weak_layer = reference_system.weak_layer
            G_Ic = weak_layer.G_Ic
            G_IIc = weak_layer.G_IIc
            
            ax.scatter(0, G_IIc, color="black", s=100, marker="o", zorder=5)
            ax.text(0.01, G_IIc + 0.02, r"$G_{IIc}$", color="black", ha="left", va="center")
            ax.scatter(G_Ic, 0, color="black", s=100, marker="o", zorder=5)
            ax.text(G_Ic + 0.01, 0.01, r"$G_{Ic}$", color="black")

            def find_GI_for_GII(GII_val):
                def envelope_root_func(GI_val):
                    return criteria_evaluator.fracture_toughness_envelope(GI_val, GII_val, weak_layer) - 1
                try:
                    GI_root = brentq(envelope_root_func, a=0, b=50, xtol=1e-6, rtol=1e-6)
                    return GI_root
                except ValueError:
                    return np.nan

            GII_max = G_IIc * 1.1
            GII_range = np.linspace(0, GII_max, 100)
            GI_envelope = np.array([find_GI_for_GII(t) for t in GII_range])

            valid_points = ~np.isnan(GI_envelope)
            valid_GII_range = GII_range[valid_points]
            GI_envelope = GI_envelope[valid_points]

            ax.plot(GI_envelope, valid_GII_range, "--", linewidth=2, label="Fracture Toughness Envelope", color="black")

    # Formatting
    ax.set_xlabel("GI (J/m²)")
    ax.set_ylabel("GII (J/m²)")
    ax.set_title("Fracture Toughness Envelope Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)

    plt.tight_layout()
    return fig

st.set_page_config(page_title="Analysis", layout="wide")

# Initialize plotter in session state if not already present
if "plotter" not in st.session_state:
    st.session_state.plotter = Plotter()

st.markdown("# Interactive Analysis")
st.sidebar.header("Analysis")

if "system" not in st.session_state:
    st.warning("Please assemble the slab and the scenario on the 'Slab Definition & Scenario Definition' page first.")
    st.stop()

# Initialize session state for parameters if not present
if "params" not in st.session_state:
    st.session_state.params = {
        "weight": 100,
        "window_size": 3000,
    }

if "computed_results" not in st.session_state:
    st.session_state.computed_results = {
        "coupled_criterion": None,
        "minimum_force": None,
        "minimum_crack_length": None,
        "current": None
    }

# Get system components
system: SystemModel = st.session_state["system"]
weak_layer: WeakLayer = system.weak_layer
layers: List[Layer] = system.slab.layers
scenario_config: ScenarioConfig = system.scenario.scenario_config
original_segments: List[Segment] = system.scenario.segments

# SIDEBAR
st.sidebar.subheader("Analysis Configuration")
stress_envelope_method = cast(
    Literal["adam_unpublished", "schottner", "mede_s-RG1", "mede_s-RG2", "mede_s-FCDH"],
    st.sidebar.selectbox(
        "Stress Envelope Method",
        ["adam_unpublished", "schottner", "mede_s-RG1", "mede_s-RG2", "mede_s-FCDH"],
        index=0,
        help="Method to use for stress envelope evaluation",
    )
)

scaling_factor = st.sidebar.slider(
    "Scaling Factor",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Scaling factor for adam_unpublished method",
)

order_of_magnitude = st.sidebar.slider(
    "Order of Magnitude",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Order of magnitude parameter",
)

criteria_config = CriteriaConfig(
    stress_envelope_method=stress_envelope_method,
    scaling_factor=scaling_factor,
    order_of_magnitude=order_of_magnitude,
)
# Initialize Analysis Tools
criteria_evaluator = CriteriaEvaluator(criteria_config=criteria_config)

# PARAMETER SLIDERS
col1, col2 = st.columns(2)
with col2:
    weight = st.slider(
        "Skier Weight",
        min_value=0,
        max_value=400,
        value=st.session_state.params["weight"],
        step=10,
        help="Skier weight in kg"
    )
    window_size = st.slider(
        "Window Size",
        min_value=500,
        max_value=4000,
        value=st.session_state.params["window_size"],
        step=500,
        help="Plotting window size in mm"
    )

# Detect parameter changes
current_params = {
    "weight": weight,
    "window_size": window_size,
}

# Determine what needs to be recomputed
params_changed = any(current_params[k] != st.session_state.params[k] for k in ["weight"])
window_changed = any(current_params[k] != st.session_state.params[k] for k in ["window_size"])

# UPDATE SESSION STATE
st.session_state.params = current_params

# RECOMPUTATION LOGIC
needs_full_recompute = st.session_state.computed_results["coupled_criterion"] is None
needs_current_recompute = params_changed and not needs_full_recompute

# SETUP BASE SYSTEM
model_input = ModelInput(
    scenario_config=scenario_config,
    weak_layer=weak_layer,
    layers=layers,
    segments=original_segments,
    criteria_config=criteria_config,
)
base_system = SystemModel(model_input)
if needs_full_recompute:
    with st.spinner("Computing all analysis cases..."):        
        # Compute minimum force
        mf_system = deepcopy(base_system)
        min_force_result = criteria_evaluator.find_minimum_force(mf_system)
        mf_system.update_scenario(segments=min_force_result.new_segments)

        # Compute coupled criterion
        cc_system = deepcopy(base_system)
        coupled_criterion_result = criteria_evaluator.evaluate_coupled_criterion(cc_system)
        cc_system.update_scenario(segments=coupled_criterion_result.final_system.scenario.segments)        
        
        # Compute minimum crack length
        mc_system = deepcopy(base_system)
        min_crack_length, new_segments = criteria_evaluator.find_minimum_crack_length(mc_system)
        mc_system.update_scenario(segments=new_segments)
        
        # Store results
        st.session_state.computed_results = {
            "coupled_criterion": {
                "system": cc_system,
                "result": coupled_criterion_result,
                "segments": cc_system.scenario.segments
            },
            "minimum_force": {
                "system": mf_system,
                "result": min_force_result,
                "segments": min_force_result.new_segments
            },
            "minimum_crack_length": {
                "system": mc_system,
                "crack_length": min_crack_length,
                "segments": new_segments
            },
        }

if original_segments is not None:
    # Create current segments by applying weight to basic segments
    current_segments = deepcopy(original_segments)
    for seg in current_segments:
        if seg.m != 0:
            seg.m = weight
    
    current_system = deepcopy(base_system)
    current_system.update_scenario(segments=current_segments)
else:
    st.error("Basic segments not available. Please wait for computation to complete.")
    st.stop()

if needs_current_recompute or needs_full_recompute or st.session_state.computed_results["current"] is None:
    with st.spinner("Computing current case..."):
        # Update current system with new weight
        new_crack_length, new_segments = (
            criteria_evaluator.find_crack_length_for_weight(
                current_system, weight
            )
        )
        current_system.update_scenario(segments=new_segments)
        
        # Store the updated current case in computed_results        
        st.session_state.computed_results["current"] = cast(Any, {
            "system": current_system,
            "crack_length": new_crack_length,
            "segments": new_segments
        })

# --- Display Results ---
st.subheader("Results")

# Display current system
if not window_changed or "analysis_plot" not in st.session_state:
    with st.spinner("Generating analysis plot..."):
        plotter = st.session_state.plotter
        min_force_result = st.session_state.computed_results["minimum_force"]["result"]
        min_crack_length = st.session_state.computed_results["minimum_crack_length"]["crack_length"]
        coupled_criterion_result = st.session_state.computed_results["coupled_criterion"]["result"]
        fig = plotter.plot_analysis(current_system, criteria_evaluator, min_force_result, min_crack_length, coupled_criterion_result, window=window_size)
        st.session_state.analysis_plot = fig

st.pyplot(st.session_state.analysis_plot)

# Generate and display plot
if not window_changed or "current_plot" not in st.session_state:
    col1, col2, col3 = st.columns((1,3,1))
    with col2:
        with st.spinner("Generating plot..."):
            fig = plot_system_evaluation_with_params(current_system, criteria_evaluator, window_size)
            st.session_state.current_plot = fig

st.pyplot(st.session_state.current_plot)

# Additional plotting options
st.subheader("Additional Analysis Plots")

# Case selection for additional plots
case_options = {
    "current": "Current Segments",
    "coupled_criterion": "Coupled Criterion", 
    "minimum_force": "Minimum Force",
    "minimum_crack_length": "Minimum Crack Length"
}

# Case selection for additional plots
st.write("**Select cases to compare:**")
selected_cases = []
for case_key, case_label in case_options.items():
    if case_key == "current" or st.session_state.computed_results[case_key] is not None:
        if st.checkbox(case_label, value=True, key=f"check_{case_key}"):
            selected_cases.append(case_key)

if selected_cases:
    # Create tabs for different plot types
    tab1, tab2 = st.tabs(["Stress Envelope", "ERR Envelope"])
    
    with tab1:
        with st.spinner("Generating stress envelope plot..."):
            fig_stress = plot_stress_envelope_comparison(selected_cases, criteria_evaluator)
            st.pyplot(fig_stress)
    
    with tab2:
        with st.spinner("Generating ERR envelope plot..."):
            fig_err = plot_err_envelope_comparison(selected_cases, criteria_evaluator)
            st.pyplot(fig_err)
else:
    st.info("Please select at least one case to display additional plots.")


# Show case-specific information
if st.session_state.computed_results["coupled_criterion"] is not None:
    cc_data = st.session_state.computed_results["coupled_criterion"]
    if cc_data is not None and "result" in cc_data:
        cc_result = cc_data["result"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Coupled Criterion - Critical Weight", f"{cc_result.critical_skier_weight:.1f} kg")
        with col2:
            st.metric("Coupled Criterion - Crack Length", f"{cc_result.crack_length:.1f} mm")
        with col3:
            st.metric("Converged", str(cc_result.converged))

if st.session_state.computed_results["minimum_force"] is not None:
    mf_data = st.session_state.computed_results["minimum_force"]
    if mf_data is not None and "result" in mf_data:
        mf_result = mf_data["result"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Stress Criterion - Critical Weight", f"{mf_result.critical_skier_weight:.1f} kg")
        with col2:
            pass

if st.session_state.computed_results["minimum_crack_length"] is not None:
    mc_result = st.session_state.computed_results["minimum_crack_length"]
    if mc_result is not None and "crack_length" in mc_result and "segments" in mc_result:
        crack_length_val = mc_result["crack_length"]
        segments_val = mc_result["segments"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Self Propagation - Crack Length", f"{crack_length_val:.1f} mm")
        with col2:
            pass

# --- System Information ---
st.subheader("System Information")
with st.expander("Show System Details"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Parameters")
        st.write(f"Weight: {weight} kg")
        st.write(f"Window Size: {window_size} mm")

    with col2:
        st.subheader("Weak Layer")
        st.write(f"Density: {weak_layer.rho} kg/m³")
        st.write(f"Thickness: {weak_layer.h} mm")
        st.write(f"Elastic Modulus: {weak_layer.E} MPa")
        st.write(f"G_Ic: {weak_layer.G_Ic} J/m²")
        st.write(f"G_IIc: {weak_layer.G_IIc} J/m²")

# Show current segments
with st.expander("Show Current Segments"):
    segments_df = []
    for i, seg in enumerate(current_system.scenario.segments):
        segments_df.append({
            "Segment": i+1,
            "Length (mm)": seg.length,
            "Has Foundation": seg.has_foundation,
            "Load (kg)": seg.m
        })
    st.dataframe(segments_df)
