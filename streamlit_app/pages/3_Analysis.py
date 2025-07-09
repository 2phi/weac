import sys
from typing import List
import streamlit as st

sys.path.append("/home/pillowbeast/Documents/weac")

from weac_2.analysis.analyzer import Analyzer
from weac_2.analysis.criteria_evaluator import CriteriaEvaluator
from weac_2.analysis.plotter import Plotter

# Initialize plotter in session state if not already present
if "plotter" not in st.session_state:
    st.session_state.plotter = Plotter()
from weac_2.components import (
    CriteriaConfig,
    Layer,
    ModelInput,
    ScenarioConfig,
    Segment,
    WeakLayer,
)
from weac_2.core.system_model import SystemModel

st.set_page_config(page_title="Scenario and Analysis", layout="wide")

st.markdown("# Scenario and Analysis")
st.sidebar.header("Scenario and Analysis")

# Existence checks for weak layer and layers
if "system" not in st.session_state:
    st.warning("Please assemble the system on the 'Slab Definition' page first.")
    st.stop()

system: SystemModel = st.session_state["system"]
weak_layer: WeakLayer = system.weak_layer
layers: List[Layer] = system.slab.layers
scenario_config: ScenarioConfig = system.scenario.scenario_config
segments: List[Segment] = system.scenario.segments

# --- Criteria Configuration ---
st.sidebar.subheader("Analysis Configuration")
stress_envelope_method = st.sidebar.selectbox(
    "Stress Envelope Method",
    ["adam_unpublished", "schottner", "mede_s-RG1", "mede_s-RG2", "mede_s-FCDH"],
    index=0,
    help="Method to use for stress envelope evaluation",
)

scaling_factor = st.sidebar.slider(
    "Scaling Factor",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
    help="Scaling factor for adam_unpublished method",
)

order_of_magnitude = st.sidebar.slider(
    "Order of Magnitude",
    min_value=0.1,
    max_value=5.0,
    value=3.0,
    step=0.1,
    help="Order of magnitude parameter",
)

criteria_config = CriteriaConfig(
    stress_envelope_method=stress_envelope_method,
    scaling_factor=scaling_factor,
    order_of_magnitude=order_of_magnitude,
)

# --- System Model ---
model_input = ModelInput(
    scenario_config=scenario_config,
    weak_layer=weak_layer,
    layers=layers,
    segments=segments,
    criteria_config=criteria_config,
)

system_model = SystemModel(model_input)

# --- Initialize Analysis Tools ---
analyzer = Analyzer(system_model)
plotter = st.session_state.plotter  # Use plotter from session state
criteria_evaluator = CriteriaEvaluator(criteria_config=criteria_config)


st.header("Comprehensive Analysis")

# --- Analysis Options ---
st.subheader("Analysis Options")
col1, col2 = st.columns(2)

with col1:
    run_full_analysis = st.button("🔬 Run Full Analysis", type="primary")

with col2:
    show_individual_plots = st.checkbox("Show Individual Analysis Steps", value=False)

# --- Full Analysis ---
if run_full_analysis:
    st.subheader("Analysis Results")

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Coupled Criterion Evaluation
    status_text.text("Evaluating coupled criterion...")
    progress_bar.progress(10)

    with st.spinner("Evaluating coupled criterion..."):
        coupled_criterion_result = criteria_evaluator.evaluate_coupled_criterion(
            system_model
        )
        analyzer = Analyzer(coupled_criterion_result.final_system)
        # Calculate fracture toughness criterion
        diff_energy = analyzer.differential_ERR(unit="J/m^2")
        diff_err = criteria_evaluator.fracture_toughness_envelope(
            diff_energy[1], diff_energy[2], weak_layer
        )

    progress_bar.progress(30)

    # Display coupled criterion results
    st.success("✅ Coupled Criterion Analysis Complete")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Converged", "Yes" if coupled_criterion_result.converged else "No")
        st.metric(
            "Critical Skier Weight",
            f"{coupled_criterion_result.critical_skier_weight:.1f} kg",
        )

    with col2:
        st.metric("Crack Length", f"{coupled_criterion_result.crack_length:.1f} mm")
        st.metric("IERR Envelope", f"{coupled_criterion_result.g_delta:.3f}") # TODO: change to G_delta
        st.metric("DERR Envelope", f"{diff_err:.3f}")

    with col3:
        st.metric("Iterations", f"{coupled_criterion_result.iterations}")
        st.metric("Max Dist Stress", f"{coupled_criterion_result.max_dist_stress:.3f}")

    st.info(f"**Message:** {coupled_criterion_result.message}")

    # Step 2: Crack Propagation Analysis
    status_text.text("Analyzing crack propagation...")
    progress_bar.progress(50)

    with st.spinner("Analyzing crack propagation..."):
        final_system = coupled_criterion_result.final_system
        g_delta_with_weight, propagation_with_weight = (
            criteria_evaluator.check_crack_self_propagation(
                final_system, rm_skier_weight=False
            )
        )
        g_delta_without_weight, propagation_without_weight = (
            criteria_evaluator.check_crack_self_propagation(
                final_system, rm_skier_weight=True
            )
        )

    progress_bar.progress(60)

    # Display crack propagation results
    st.success("✅ Crack Propagation Analysis Complete")
    col1, col2 = st.columns(2)
    st.header("Propagation of Crack")
    with col1:
        st.subheader("With Critical Skier Weight")
        st.metric("Differential ERR", f"{g_delta_with_weight:.3f}")
        st.metric("Can Propagate", "Yes" if propagation_with_weight else "No")

    with col2:
        st.subheader("Without Any Skier Weight")
        st.metric("Differential ERR", f"{g_delta_without_weight:.3f}")
        st.metric("Can Propagate", "Yes" if propagation_without_weight else "No")

    # Step 3: Minimum Force Analysis
    status_text.text("Finding minimum force...")
    progress_bar.progress(70)

    with st.spinner("Finding minimum force..."):
        min_force_result = criteria_evaluator.find_minimum_force(final_system)
        # Reset system to old segments for next analysis
        final_system.update_scenario(segments=min_force_result.old_segments)

    progress_bar.progress(80)

    # Display minimum force results
    st.success("✅ Minimum Force Analysis Complete")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Success", "Yes" if min_force_result.success else "No")
        st.metric(
            "Critical Skier Weight", f"{min_force_result.critical_skier_weight:.1f} kg"
        )

    with col2:
        st.metric("Iterations", f"{min_force_result.iterations}")
        st.metric("Max Dist Stress", f"{min_force_result.max_dist_stress:.3f}")

    # Step 4: Minimum Crack Length Analysis
    status_text.text("Finding minimum crack length...")
    progress_bar.progress(85)

    with st.spinner("Finding minimum crack length..."):
        print(final_system.scenario.segments)
        min_crack_length, new_segments = criteria_evaluator.find_minimum_crack_length(final_system)

    progress_bar.progress(90)

    # Display minimum crack length results
    st.success("✅ Minimum Crack Length Analysis Complete")
    st.metric("Minimum Crack Length", f"{min_crack_length:.1f} mm")

    # # Step 5: Find crack length for increased weight
    # status_text.text("Analyzing crack length for increased weight...")
    # with st.spinner("Analyzing crack length for increased weight..."):
    #     increased_weight = min_force_result.critical_skier_weight + 20
    #     new_crack_length, new_segments = (
    #         criteria_evaluator.find_crack_length_for_weight(
    #             final_system, increased_weight
    #         )
    #     )

    # progress_bar.progress(95)

    # # Display increased weight results
    # st.success("✅ Crack Length for Increased Weight Analysis Complete")
    # col1, col2 = st.columns(2)

    # with col1:
    #     st.metric("Test Weight", f"{increased_weight:.1f} kg")

    # with col2:
    #     st.metric("Resulting Crack Length", f"{new_crack_length:.1f} mm")

    # Step 6: Generate Plots
    status_text.text("Generating plots...")
    progress_bar.progress(100)

    with st.spinner("Generating comprehensive plots..."):
        # Generate all plots
        fig_stress_envelope = plotter.plot_stress_envelope(
            system_model=final_system,
            criteria_evaluator=criteria_evaluator,
            all_envelopes=False,
            filename="stress_envelope",
        )

        fig_err_envelope = plotter.plot_err_envelope(
            system_model=final_system,
            criteria_evaluator=criteria_evaluator,
            filename="err_envelope",
        )

        # Reset system to original segments for comprehensive analysis plot
        final_system.update_scenario(segments=segments)

        fig_analysis = plotter.plot_analysis(
            system=final_system,
            criteria_evaluator=criteria_evaluator,
            min_force_result=min_force_result,
            min_crack_length=min_crack_length,
            coupled_criterion_result=coupled_criterion_result,
            new_crack_length=0.0,
            filename="analysis",
            deformation_scale=500.0,
        )

    status_text.text("Analysis complete!")
    st.success("🎉 **Full Analysis Complete!**")

    # --- Display Plots ---
    st.subheader("Analysis Plots")

    # Comprehensive Analysis Plot
    st.subheader("Comprehensive Analysis")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.pyplot(fig_analysis)

    # Individual plots in tabs
    if show_individual_plots:
        tab1, tab2 = st.tabs(["Stress Envelope", "ERR Envelope"])

        with tab1:
            st.subheader("Stress Envelope")
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.pyplot(fig_stress_envelope)

        with tab2:
            st.subheader("Energy Release Rate Envelope")
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.pyplot(fig_err_envelope)

# --- Individual Analysis Options ---
else:
    st.subheader("Individual Analysis Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Slab Profile"):
            with st.spinner("Generating slab profile..."):
                fig_profile = plotter.plot_slab_profile(
                    weak_layers=weak_layer,
                    slabs=system_model.slab,
                    filename="slab_profile",
                )
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(fig_profile)

    with col2:
        if st.button("📊 Section Forces"):
            with st.spinner("Generating section forces plot..."):
                fig_forces = plotter.plot_section_forces(
                    system_model=system_model, filename="section_forces"
                )
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(fig_forces)

    col3, col4 = st.columns(2)

    with col3:
        if st.button("⚡ Energy Release Rates"):
            with st.spinner("Generating energy release rates plot..."):
                fig_err = plotter.plot_energy_release_rates(
                    system_model=system_model, filename="energy_release_rates"
                )
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(fig_err)

    with col4:
        if st.button("🎯 Stress Envelope Only"):
            with st.spinner("Generating stress envelope plot..."):
                fig_stress = plotter.plot_stress_envelope(
                    system_model=system_model,
                    criteria_evaluator=criteria_evaluator,
                    filename="stress_envelope_only",
                )
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.pyplot(fig_stress)

# --- Additional Information ---
st.subheader("System Information")
with st.expander("Show System Details"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weak Layer")
        st.write(f"Density: {weak_layer.rho} kg/m³")
        st.write(f"Thickness: {weak_layer.h} mm")
        st.write(f"Elastic Modulus: {weak_layer.E} MPa")
        st.write(f"G_Ic: {weak_layer.G_Ic} J/m²")
        st.write(f"G_IIc: {weak_layer.G_IIc} J/m²")

    with col2:
        st.subheader("Scenario")
        st.write(f"System Type: {scenario_config.system_type}")
        st.write(f"Slope Angle: {scenario_config.phi}°")
        st.write(f"Total Length: {sum(seg.length for seg in segments) / 1000:.1f} m")

        st.subheader("Layers")
        for i, layer in enumerate(layers):
            st.write(f"Layer {i + 1}: {layer.rho} kg/m³, {layer.h} mm")
