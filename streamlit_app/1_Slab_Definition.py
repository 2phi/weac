import sys
import random
import matplotlib.pyplot as plt
import streamlit as st

sys.path.append("/home/pillowbeast/Documents/weac")

from weac_2.components import Layer
from weac_2.components.layer import WeakLayer
from weac_2.components.model_input import ModelInput
from weac_2.components.scenario_config import ScenarioConfig
from weac_2.core.slab import Slab
from weac_2.core.system_model import SystemModel
from weac_2.utils.misc import load_dummy_profile
from weac_2.analysis.plotter import Plotter

if "plotter" not in st.session_state:
    st.session_state.plotter = Plotter()

st.set_page_config(layout="wide")

st.markdown("# Slab Definition")
st.sidebar.header("Slab Definition")

# --- Page Layout ---
col1, col2 = st.columns([1, 1])
plot_placeholder = col2.empty()

# --- Weak Layer Properties ---
with col1:
    st.header("Weak Layer Properties")
    col1, col2 = st.columns(2)
    rho = col1.number_input(
        "Density (kg/m^3)",
        key="rho_weak",
        value=125.0,
        min_value=80.0,
        step=10.0,
    )
    h = col2.number_input(
        "Thickness (mm)",
        key="h_weak",
        value=30.0,
        min_value=10.0,
        step=5.0,
    )

    # Create a default weak layer instance
    default_wl = WeakLayer(rho=rho, h=h)

    with st.expander("Advanced Properties"):
        edit_wl = st.checkbox("Overwrite properties", value=False)
        # --- Elastic Properties ---
        elastic_cols = st.columns(3)
        nu = elastic_cols[0].number_input(
            "Poisson's ratio",
            key="nu_weak",
            value=default_wl.nu,
            step=0.01,
            disabled=not edit_wl,
        )
        G = elastic_cols[1].number_input(
            "Shear modulus (MPa)",
            key="G_weak",
            value=default_wl.G,
            step=0.01,
            disabled=not edit_wl,
        )
        E = elastic_cols[2].number_input(
            "Young's modulus (MPa)",
            key="E_weak",
            value=1.0,  # TODO: this is not default right now 'default_wl.E'
            step=0.01,
            disabled=not edit_wl,
        )

        # --- Stiffness Properties ---
        stiffness_cols = st.columns(3)
        kn = stiffness_cols[0].number_input(
            "Normal Spring stiffness (N/mm)",
            key="kn_weak",
            value=default_wl.kn,
            step=0.001,
            disabled=not edit_wl,
        )
        kt = stiffness_cols[1].number_input(
            "Shear Spring stiffness (N/mm)",
            key="kt_weak",
            value=default_wl.kt,
            step=0.001,
            disabled=not edit_wl,
        )
        with stiffness_cols[2]:
            st.write("")
            st.write("")
            e_method_options = ("bergfeld", "scapazzo", "gerling")
            e_method_default_index = e_method_options.index(default_wl.E_method)
            E_method = st.radio(
                "Young's modulus method",
                e_method_options,
                index=e_method_default_index,
                horizontal=True,
                label_visibility="collapsed",
                disabled=not edit_wl,
                key="e_method_weak",
            )

        # --- Fracture Properties ---
        fracture_cols = st.columns(3)
        G_c = fracture_cols[0].number_input(
            "Total Fracture Energy Release Rate (N/mm)",
            key="G_c_weak",
            value=default_wl.G_c,
            step=0.01,
            disabled=not edit_wl,
        )
        G_Ic = fracture_cols[1].number_input(
            "Mode I Fracture Energy Release Rate (N/mm)",
            key="G_Ic_weak",
            value=default_wl.G_Ic,
            step=0.01,
            disabled=not edit_wl,
        )
        G_IIc = fracture_cols[2].number_input(
            "Mode II Fracture Energy Release Rate (N/mm)",
            key="G_IIc_weak",
            value=default_wl.G_IIc,
            step=0.01,
            disabled=not edit_wl,
        )

        if edit_wl:
            weak_layer = WeakLayer(
                rho=rho,
                h=h,
                nu=nu,
                E=E,
                E_method=E_method,
                G=G,
                kn=kn,
                kt=kt,
                G_c=G_c,
                G_Ic=G_Ic,
                G_IIc=G_IIc,
            )
        else:
            weak_layer = default_wl
    # --- Slab Properties ---
    col1, col2 = st.columns([2, 2], vertical_alignment="bottom")
    with col1:
        st.header("Slab Properties")
    with col2:
        profile_type = st.radio(
            "Slab Profile Type",
            ("From Database", "Custom"),
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )
    if profile_type == "Custom":
        col1, col2 = st.columns([2, 1], vertical_alignment="bottom")
        with col1:
            st.subheader("Custom Slab Profile")
        with col2:
            num_layers = st.number_input(
                "Number of slab layers", min_value=1, value=1, step=1
            )

        if "custom_layer_defaults" not in st.session_state:
            st.session_state.custom_layer_defaults = []

        # Adjust the number of defaults to match the number of layers
        current_defaults_count = len(st.session_state.custom_layer_defaults)
        if num_layers > current_defaults_count:
            for _ in range(num_layers - current_defaults_count):
                density = random.randint(100, 300)
                thickness = random.randint(10, 200)
                st.session_state.custom_layer_defaults.append(
                    {"density": density, "thickness": thickness}
                )
        elif num_layers < current_defaults_count:
            st.session_state.custom_layer_defaults = (
                st.session_state.custom_layer_defaults[:num_layers]
            )

        layers = []
        for i in range(num_layers):
            defaults = st.session_state.custom_layer_defaults[i]
            cols = st.columns([1, 2, 2])
            with cols[0]:
                st.write("")
                st.write("")
                st.markdown(f"**Layer {i + 1}**")
            rho_layer = cols[1].number_input(
                "Density (kg/m^3)",
                key=f"rho_{i}",
                value=float(defaults["density"]),
                min_value=10.0,
                step=10.0,
            )
            h_layer = cols[2].number_input(
                "Thickness (mm)",
                key=f"h_{i}",
                value=float(defaults["thickness"]),
                min_value=10.0,
                step=10.0,
            )
            layers.append(Layer(rho=rho_layer, h=h_layer))
    elif profile_type == "From Database":
        st.subheader("Database Slab Profile")
        col1, col2 = st.columns([1, 3], vertical_alignment="bottom")
        profile_options = ["a", "b", "c", "d", "e", "f", "tested"]
        col1.write("Select Profile:")
        profile_name = col2.radio(
            "Select a profile",
            profile_options,
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )
        layers = load_dummy_profile(profile_name)


if "weak_layer" not in locals():
    weak_layer = default_wl

# --- Plot Slab Profile ---
with plot_placeholder.container():
    st.header("Slab Profile")
    slab = Slab(layers=layers)
    fig = st.session_state.plotter.plot_slab_profile(weak_layers=weak_layer, slabs=slab)
    st.pyplot(fig)
    plt.close(fig)

# --- Next Step ---
st.header("Next Step")

if st.button("To Scenario Definition"):
    model_input = ModelInput(
        layers=layers,
        weak_layer=weak_layer,
    )

    system = SystemModel(model_input=model_input)
    st.session_state["system"] = system
    st.switch_page("pages/2_Scenario_Definition.py")
