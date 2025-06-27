import streamlit as st

from weac_2.components.scenario_config import ScenarioConfig
from weac_2.components.segment import Segment

st.set_page_config(page_title="Scenario and Analysis", layout="wide")

st.markdown("# Scenario Definition")
st.sidebar.header("Scenario Definition")

st.write("""This page allows you to define the scenario.""")

# --- Slab Existence Check ---
if "weak_layer" not in st.session_state or "layers" not in st.session_state:
    st.warning("Please define the slab on the 'Slab Definition' page first.")
    st.stop()

weak_layer = st.session_state["weak_layer"]
layers = st.session_state["layers"]

# --- Scenario Config ---
st.header("Scenario Config")
configs = st.columns(3)

system_type = configs[0].radio(
    "System Type",
    ("skier", "skiers", "pst-", "-pst", "vpst-", "-vpst"),
    index=0,
    horizontal=True,
)
slope_angle = st.slider(
    "Slope Angle [deg]", min_value=-45, max_value=45, value=0, step=1
)
crack_length = configs[1].number_input(
    "Crack Length [mm]", min_value=0.0, value=0.0, step=1.0
)
surface_load = configs[2].number_input(
    "Surface Load (N/mm)", min_value=0.0, value=0.0, step=1.0
)

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
    length = cols[i].number_input("Length (m)", key=f"length_{i}", value=1.0, step=0.1)
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
            "Skier weight (kg)",
            key=f"skier_weight_{i}",
            min_value=0.0,
            value=0.0,
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

st.header("Next Step")

if st.button("To Analysis"):
    with st.spinner("Assembling system..."):
        st.session_state["segments"] = segments
        st.session_state["scenario_config"] = scenario_config

        st.success("Scenario defined successfully!")
        st.write("You can now proceed to the 'Analysis' page.")

if "scenario" in st.session_state:
    st.success("You can proceed to the next page.")
