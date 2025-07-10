import sys
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st

sys.path.append("/home/pillowbeast/Documents/weac")

from weac_2.components.model_input import ModelInput
from weac_2.components.scenario_config import ScenarioConfig
from weac_2.components.segment import Segment
from weac_2.core.scenario import Scenario
from weac_2.core.system_model import SystemModel
from weac_2.analysis.analyzer import Analyzer
from weac_2.analysis.plotter import Plotter

# Initialize plotter in session state if not already present
if "plotter" not in st.session_state:
    st.session_state.plotter = Plotter()

st.set_page_config(page_title="Scenario and Analysis", layout="wide")

st.markdown("# Scenario Definition")
st.sidebar.header("Scenario Definition")

st.write("""This page allows you to define the scenario.""")

# --- Slab Existence Check ---
if "system" not in st.session_state:
    st.warning("Please define the slab on the 'Slab Definition' page first.")
    st.stop()
system: SystemModel = st.session_state["system"]
weak_layer = system.weak_layer

# --- Scenario Config ---
st.header("Scenario Config")
configs = st.columns(4)

system_type = configs[0].radio(
    "System Type",
    ("skier", "skiers", "pst-", "-pst", "vpst-", "-vpst"),
    index=0,
    horizontal=True,
)
slope_angle = st.slider(
    "Slope Angle [deg]", min_value=-45, max_value=45, value=22, step=1
)
crack_length = configs[1].number_input(
    "Crack Length [mm]", min_value=0.0, value=0.0, step=1.0
)
surface_load = configs[2].number_input(
    "Surface Load (N/mm)", min_value=0.0, value=0.0, step=0.1
)
touchdown = configs[3].radio("Touchdown", (True, False), index=1, horizontal=True)

# --- Scenario ---
col1, col2, col3 = st.columns([2, 1, 7], vertical_alignment="bottom")
with col1:
    st.header("Segments")
with col2:
    num_segments = st.number_input(
        "Number of segments", min_value=2, value=2, step=1, label_visibility="collapsed"
    )

segments: list[Segment] = []

# Create column headers
col_headers = st.columns(num_segments)
for i in range(num_segments):
    if i == 0:
        col_headers[i].markdown("**Left Boundary Segment**")
    elif i == num_segments - 1:
        col_headers[i].markdown("**Right Boundary Segment**")
    else:
        col_headers[i].markdown(f"**Segment {i + 1}**")

# Create rows for each attribute
cols = st.columns(num_segments)
weight_cols = st.columns(2 * num_segments - 1)
lengths = []
foundations = []
skier_weights = []

# Length row
for i in range(num_segments):
    if i == 0 or i == num_segments - 1:
        length = cols[i].number_input(
            "Length [mm]", key=f"length_{i}", value=10000.0, step=100.0
        )
    else:
        length = cols[i].number_input(
            "Length [mm]", key=f"length_{i}", value=5000.0, step=100.0
        )
    lengths.append(length)

# Foundation row
for i in range(num_segments):
    has_foundation = cols[i].checkbox(
        "Has foundation", key=f"has_foundation_{i}", value=True
    )
    foundations.append(has_foundation)

# Skier weight row
for i in range(2 * num_segments - 1):
    if i % 2 == 1:
        skier_weight = weight_cols[i].number_input(
            "Skier weight [kg]",
            key=f"skier_weight_{i}",
            min_value=0.0,
            value=50.0,
            step=1.0,
        )
        skier_weights.append(skier_weight)
    if i == 2 * num_segments - 2:
        skier_weights.append(0.0)

# Create segments from collected values
for i in range(num_segments):
    segments.append(
        Segment(length=lengths[i], has_foundation=foundations[i], m=skier_weights[i])
    )

scenario_config = ScenarioConfig(
    phi=slope_angle,
    system_type=system_type,
    crack_length=crack_length,
    surface_load=surface_load,
)

system.update_scenario(segments=segments, scenario_config=scenario_config)
system.toggle_touchdown(touchdown=touchdown)
# Plot the deformed slab
analyzer = Analyzer(system_model=system)
xs, zs, xwls = analyzer.rasterize_solution(mode="cracked", num=2000)

col1, col2 = st.columns([2, 14])
with col1:
    st.markdown("**Field Quantity**")
with col2:
    st.markdown("**Deformed Slab**")

# Provide radio choice for field quantity
field = col1.radio(
    "Field Quantity",
    ("w", "u", "principal", "Sxx", "Txz", "Szz"),
    index=2,
    horizontal=False,
)
fig = st.session_state.plotter.plot_deformed(
    xsl=xs,
    xwl=xwls,
    z=zs,
    analyzer=analyzer,
    dz=2,
    scale=100,
    window=int(1e6),  # Using large int instead of np.inf
    pad=2,
    levels=300,
    aspect=2,
    field=field,
    normalize=True,
)
col2.pyplot(fig)
plt.close(fig)

st.header("Next Step")
if st.button("To Analysis"):
    st.session_state["system"] = system
    st.switch_page("pages/3_Analysis.py")
