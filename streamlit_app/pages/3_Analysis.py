from typing import List
import streamlit as st

from weac_2.analysis.analyzer import Analyzer
from weac_2.analysis.plotter import Plotter
from weac_2.components import Layer, WeakLayer, Segment, ScenarioConfig, ModelInput
from weac_2.core.system_model import SystemModel

st.set_page_config(page_title="Scenario and Analysis", layout="wide")

st.markdown("# Scenario and Analysis")
st.sidebar.header("Scenario and Analysis")

# Existence checks for weak layer and layers
if "weak_layer" not in st.session_state or "layers" not in st.session_state:
    st.warning("Please assemble the system on the 'Slab Definition' page first.")
    st.stop()

# Existence checks for scenario
if "scenario" not in st.session_state:
    st.warning("Please define the scenario on the 'Scenario Definition' page first.")
    st.stop()

weak_layer: WeakLayer = st.session_state["weak_layer"]
layers: List[Layer] = st.session_state["layers"]
scenario_config: ScenarioConfig = st.session_state["scenario_config"]
segments: List[Segment] = st.session_state["segments"]

# --- System Model ---
model_input = ModelInput(
    scenario_config=scenario_config,
    weak_layer=weak_layer,
    layers=layers,
    segments=segments,
)

system_model = SystemModel(model_input)

st.header("Analysis")
analyzer = Analyzer(system_model)
plotter = Plotter(system_model)

# --- Initial Plots ---
st.subheader("Slab Profile")
with st.spinner("Generating slab profile plot..."):
    fig_profile = plotter.plot_slab_profile()
    st.pyplot(fig_profile)

# --- Deformations Analysis ---
st.subheader("Slab Deformations")
if st.button("Analyze Deformations"):
    with st.spinner("Analyzing deformations and generating plots..."):
        xsl_skier, z_skier, xwl_skier = analyzer.rasterize_solution(mode="cracked")

        fig_deformed = plotter.plot_deformed(
            xsl_skier,
            xwl_skier,
            z_skier,
            analyzer,
            scale=200,
            window=200,
            aspect=2,
            field="principal",
        )
        st.pyplot(fig_deformed)

        fig_displacement = plotter.plot_displacement_profile(xsl_skier, z_skier)
        st.pyplot(fig_displacement)

        st.success("Deformation analysis complete.")

# --- Crack Propagation Analysis ---
st.subheader("Crack Propagation Analysis")

# Add inputs for crack propagation if needed, e.g., crack length
# For now, using defaults from the notebook.

if st.button("Analyze Crack Propagation"):
    with st.spinner("Analyzing crack propagation..."):
        crit_force, crit_length = analyzer.analyze_crack_propagation()
        st.write(f"Critical Force: {crit_force:.2f} N")
        st.write(f"Critical Length: {crit_length:.2f} m")

        fig_crack = plotter.plot_critical_crack_length(crit_force, crit_length)
        st.pyplot(fig_crack)
        st.success("Crack propagation analysis complete.")
