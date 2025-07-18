import sys

sys.path.append("/home/pillowbeast/Documents/weac")

from copy import deepcopy

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.plotting import plot_traffic_light

from weac_2.components import (
    Layer,
    WeakLayer,
    Segment,
    CriteriaConfig,
    ModelInput,
    ScenarioConfig,
    Config,
)
from weac_2.core import SystemModel, Scenario, Slab
from weac_2.analysis import (
    CriteriaEvaluator,
    Plotter,
    CoupledCriterionResult,
    CoupledCriterionHistory,
    FindMinimumForceResult,
)
from weac_2.analysis.analyzer import Analyzer
from weac_2.utils.misc import load_dummy_profile

NORMAL_SKIER_WEIGHT = 100

# Initialize session state
if "plotter" not in st.session_state:
    st.session_state.plotter = Plotter()

if "current_stage" not in st.session_state:
    st.session_state.current_stage = 1

if "slab_layers" not in st.session_state:
    st.session_state.slab_layers = []

if "selected_weak_layer" not in st.session_state:
    st.session_state.selected_weak_layer = None

# Predefined slab types
SLAB_TYPES = {
    "leicht gebundener Neuschnee": {"density": 150, "default_thickness": 200},
    "frischer weicher Treibschnee": {"density": 180, "default_thickness": 200},
    "alter harter Treibschnee": {"density": 270, "default_thickness": 200},
    "Schmelzhartkruste": {"density": 350, "default_thickness": 200},
}

# Predefined weak layer types
WEAK_LAYER_TYPES = {
    "Very Weak": {
        "density": 125,
        "thickness": 10,
        "sigma_c": 5.16,
        "tau_c": 4.09,
        "E": 2.0,
    },
    "Weak": {"density": 125, "thickness": 10, "sigma_c": 6.16, "tau_c": 5.09, "E": 2.0},
    "Less Weak": {
        "density": 125,
        "thickness": 10,
        "sigma_c": 7.16,
        "tau_c": 6.09,
        "E": 2.0,
    },
}

st.set_page_config(page_title="Avalanche Risk Assessment", layout="wide")

# Create centered layout (80% width)
_, main_col, _ = st.columns([1, 8, 1])

with main_col:
    # Main title
    st.title("🏔️ Avalanche Risk Assessment Tool")

    # STAGE 1: Slab Assembly
    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("Build Your Slab")

        # Slab layers section
        st.write("**Add Slab Layers:**")
        for i, (slab_type, properties) in enumerate(SLAB_TYPES.items()):
            cols = st.columns([4, 2, 2])
            with cols[0]:
                st.write(f"{slab_type} (ρ={properties['density']} kg/m³)")
            with cols[2]:
                if st.button("Add", key=f"add_slab_{i}"):
                    new_layer = Layer(
                        rho=properties["density"],
                        h=properties["default_thickness"],
                    )
                    st.session_state.slab_layers.insert(
                        0,
                        {
                            "type": slab_type,
                            "layer": new_layer,
                            "thickness": properties["default_thickness"],
                        },
                    )
                    st.rerun()

        # Display current slab layers
        if st.session_state.slab_layers:
            st.write("**Current Slab Layers:**")
            for i, layer_info in enumerate(st.session_state.slab_layers):
                cols = st.columns([4, 3, 2])
                with cols[0]:
                    st.write(f"{layer_info['type']}")
                with cols[1]:
                    # Allow thickness adjustment - height text and input side by side
                    input_col, unit_col = st.columns([2, 1])
                    with input_col:
                        new_thickness = st.number_input(
                            "Layer thickness",
                            min_value=10.0,
                            max_value=500.0,
                            value=float(layer_info["thickness"]),
                            step=10.0,
                            key=f"thickness_{i}",
                            label_visibility="collapsed",
                        )
                    with unit_col:
                        st.write("mm")
                    if new_thickness != layer_info["thickness"]:
                        # Create a new layer instance since Layer is frozen/immutable
                        old_layer = layer_info["layer"]
                        new_layer = Layer(
                            rho=old_layer.rho,
                            h=new_thickness,
                            nu=old_layer.nu,
                            E=old_layer.E,
                            G=old_layer.G,
                            E_method=old_layer.E_method,
                        )
                        st.session_state.slab_layers[i]["thickness"] = new_thickness
                        st.session_state.slab_layers[i]["layer"] = new_layer
                        st.rerun()
                with cols[2]:
                    if st.button("Remove", key=f"remove_slab_{i}"):
                        st.session_state.slab_layers.pop(i)
                        st.rerun()

        st.divider()

        # Weak layer section
        st.write("**Select Weak Layer:**")
        wl_col1, wl_col2 = st.columns([1, 1])
        with wl_col1:
            weak_layer_choice = st.radio(
                "Choose weak layer type:",
                index=0,
                options=list(WEAK_LAYER_TYPES.keys()),
                key="weak_layer_radio",
            )

            weak_props = WEAK_LAYER_TYPES[weak_layer_choice]
            st.session_state.selected_weak_layer = WeakLayer(
                rho=weak_props["density"],
                h=weak_props["thickness"],
                sigma_c=weak_props["sigma_c"],
                tau_c=weak_props["tau_c"],
                E=weak_props["E"],
            )

            st.write(f"ρ={weak_props['density']} kg/m³")
            st.write(f"h={weak_props['thickness']}mm")
        with wl_col2:
            st.write(f"σ_c={weak_props['sigma_c']} kPa")
            st.write(f"τ_c={weak_props['tau_c']} kPa")
            st.write(f"E={weak_props['E']}")

    with col2:
        st.subheader("Slab Profile")

        # Create and display slab profile
        if st.session_state.slab_layers and st.session_state.selected_weak_layer:
            layers = [
                layer_info["layer"] for layer_info in st.session_state.slab_layers
            ]
            slab = Slab(layers=layers)
            weak_layer = st.session_state.selected_weak_layer

            fig = st.session_state.plotter.plot_slab_profile(
                weak_layers=weak_layer, slabs=slab
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Add slab layers and select a weak layer to see the profile")

    # STAGE 2: Scenario Setup
    col1, col2 = st.columns([1, 1])
    # Vertically center the content in col1 using st.markdown with custom CSS
    with col1:
        st.subheader("Scenario Parameters")

        # Add vertical centering using st.markdown and CSS
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; justify-content: center; height: 100%; min-height: 300px;">
            """,
            unsafe_allow_html=True,
        )

        # Slope angle slider
        slope_angle = st.slider(
            "Slope Angle (degrees)",
            min_value=0,
            max_value=45,
            value=st.session_state.get("slope_angle", 30),
            step=1,
            help="Angle of the slope in degrees",
            key="slope_angle_slider",
        )
        st.session_state.slope_angle = slope_angle

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Slab Visualization")

        # Create rotated slab visualization
        if st.session_state.slab_layers and st.session_state.selected_weak_layer:
            layers = [
                layer_info["layer"] for layer_info in st.session_state.slab_layers
            ]
            slab = Slab(layers=layers)
            weak_layer = st.session_state.selected_weak_layer

            fig = st.session_state.plotter.plot_rotated_slab_profile(
                weak_layer=weak_layer,
                slab=slab,
                angle=slope_angle,
                weight=NORMAL_SKIER_WEIGHT,
                title="Slab Visualization",
            )
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Risk Level")

    # Calculate actual risk using system analysis
    if st.session_state.slab_layers and st.session_state.selected_weak_layer:
        # Get current parameters from session state or defaults
        slope_angle = st.session_state.get("slope_angle", 30)

        # Build the system model
        layers = [layer_info["layer"] for layer_info in st.session_state.slab_layers]
        weak_layer = st.session_state.selected_weak_layer
        print("weak_layer", weak_layer)

        # Create a simple scenario with one skier
        segments = [
            Segment(length=18000, has_foundation=True, m=0),
            Segment(length=0, has_foundation=False, m=NORMAL_SKIER_WEIGHT),
            Segment(length=0, has_foundation=False, m=0),
            Segment(length=18000, has_foundation=True, m=0),
        ]
        scenario_config = ScenarioConfig(
            phi=slope_angle,
            system_type="skier",
            crack_length=0.0,
            surface_load=0.0,
        )
        model_input = ModelInput(
            scenario_config=scenario_config,
            weak_layer=weak_layer,
            layers=layers,
            segments=segments,
        )

        system = SystemModel(model_input, config=Config(touchdown=True))
        criteria_evaluator = CriteriaEvaluator(CriteriaConfig())
        analyzer = Analyzer(system)

        # Debug: Check if the system actually has the correct weak layer
        print("=== SYSTEM DEBUG ===")
        print("System weak layer kn:", system.eigensystem.weak_layer.kn)
        print("System weak layer kt:", system.eigensystem.weak_layer.kt)
        print("System weak layer rho:", system.eigensystem.weak_layer.rho)
        print("Field quantities weak layer kn:", system.fq.es.weak_layer.kn)
        print("Field quantities weak layer kt:", system.fq.es.weak_layer.kt)

        # Evaluate stress envelope for the slab without skier
        xs, zs, x_founded = analyzer.rasterize_solution(mode="uncracked", num=4000)
        sigma_kPa = system.fq.sig(zs, unit="kPa")
        tau_kPa = system.fq.tau(zs, unit="kPa")
        print("sigma_kPa", sigma_kPa)
        print("tau_kPa", tau_kPa)
        print("Max Sigma", np.max(np.abs(sigma_kPa)))
        print("Max Tau", np.max(np.abs(tau_kPa)))
        print("kn", weak_layer.kn)
        print("kt", weak_layer.kt)

        stress_envelope = criteria_evaluator.stress_envelope(
            sigma=sigma_kPa,
            tau=tau_kPa,
            weak_layer=weak_layer,
        )

        max_stress = np.max(np.abs(stress_envelope))
        print("max_stress", max_stress)

        st.session_state.max_stress = max_stress

        coupled_result = criteria_evaluator.evaluate_coupled_criterion(deepcopy(system))

        # Determine risk level based on analysis
        coupled_critical = coupled_result.critical_skier_weight
        min_force_critical = coupled_result.initial_critical_skier_weight

        # Extract touchdown distance
        if system.slab_touchdown is not None:
            l_BC = system.slab_touchdown.l_BC
            # l_AB = system.slab_touchdown.l_AB
            segments = [
                Segment(length=18000, has_foundation=True, m=0),
                Segment(length=2 * l_BC, has_foundation=False, m=0),
                # Segment(length=18000, has_foundation=True, m=0),
            ]
            scenario_config = ScenarioConfig(
                phi=slope_angle,
                system_type="pst-",
                crack_length=2 * l_BC,
                surface_load=0.0,
            )
            model_input = ModelInput(
                scenario_config=scenario_config,
                weak_layer=weak_layer,
                layers=layers,
                segments=segments,
            )

            system = SystemModel(model_input, config=Config(touchdown=True))
            print("Touchdown distance", system.slab_touchdown.touchdown_distance)
            touchdown_distance = system.slab_touchdown.touchdown_distance
            analyzer = Analyzer(system)
            diff_energy = analyzer.differential_ERR(unit="J/m^2")
            DERR_I = diff_energy[1]
            DERR_II = diff_energy[2]
            g_delta = criteria_evaluator.fracture_toughness_envelope(
                G_I=DERR_I, G_II=DERR_II, weak_layer=weak_layer
            )
            print("GDELTA", g_delta)
        else:
            touchdown_distance = 0.0
            g_delta = 0.0

        # Store g_delta in session state for later use
        st.session_state.g_delta = g_delta
        st.session_state.touchdown_distance = touchdown_distance

        # Store results for display
        st.session_state.min_force_critical = min_force_critical
        st.session_state.coupled_critical = coupled_critical

    # Impact Resistance -> Distance to stress envelope
    if (
        hasattr(st.session_state, "max_stress")
        and st.session_state.max_stress is not None
    ):
        max_stress = st.session_state.max_stress
        min_stress = 0.0
        max_stress_val = 1.0
        min_bar = 0.0
        max_bar = 1.0
        clamped_stress = min(max(max_stress, min_stress), max_stress_val)
        bar_position = min_bar + (clamped_stress - min_stress) * (max_bar - min_bar) / (
            max_stress_val - min_stress
        )
        print("Bar position", bar_position)

        # Create theme for the plot
        theme = {
            "backgroundColor": "#FFFFFF",
            "textColor": "#000000",
            "base": "light",
        }

        with st.expander("Impact Resistance", expanded=False):
            st.write("""
                Impact resistance measures the ability of the slab to resist the impact of a skier.
                It's based on the differential energy release rate (ERR) - the amount of energy available to drive crack growth.
                
                **Interpretation:**
                - **High bar position (red zone)**: High impact resistance - skier likely to bounce off
                - **Medium bar position (yellow zone)**: Moderate impact resistance - skier may bounce off under certain conditions  
                - **Low bar position (green zone)**: Low impact resistance - skier likely to bounce off
                
                This is calculated from the mechanical properties of the slab and weak layer, considering the energy balance during impact.
                """)

        impact_resistance_fig = plot_traffic_light(bar_position, theme)
        st.plotly_chart(
            impact_resistance_fig,
            use_container_width=True,
            key="impact_resistance_fig",
        )

    # Fracture resistance visualization
    if hasattr(st.session_state, "coupled_critical"):
        ratio_weights = st.session_state.coupled_critical / NORMAL_SKIER_WEIGHT

        min_ratio_weights = 1.0
        max_ratio_weights_val = 5.0
        min_bar = 0.0
        max_bar = 1.0
        clamped_ratio_weights = min(
            max(ratio_weights, min_ratio_weights), max_ratio_weights_val
        )
        bar_position = max_bar - (clamped_ratio_weights - min_ratio_weights) * (
            max_bar - min_bar
        ) / (max_ratio_weights_val - min_ratio_weights)

        # Create theme for the plot
        theme = {
            "backgroundColor": "#FFFFFF",
            "textColor": "#000000",
            "base": "light",
        }

        with st.expander("Fracture Resistance", expanded=False):
            st.write("""
                Fracture resistance measures the ability of the slab to resist crack propagation.
                It's based on the differential energy release rate (ERR) - the amount of energy available to drive crack growth.
                
                **Interpretation:**
                - **High bar position (red zone)**: High fracture resistance - crack likely to spread rapidly
                - **Medium bar position (yellow zone)**: Moderate fracture resistance - crack may propagate under certain conditions  
                - **Low bar position (green zone)**: Low fracture resistance - crack growth is unlikely
                
                This is calculated from the mechanical properties of the slab and weak layer, considering the energy balance during crack propagation.
                """)

        fracture_resistance_fig = plot_traffic_light(bar_position, theme)
        st.plotly_chart(
            fracture_resistance_fig,
            use_container_width=True,
            key="fracture_resistance_fig",
        )

    # Propagation potential visualization
    if hasattr(st.session_state, "g_delta") and st.session_state.g_delta is not None:
        # g_delta = st.session_state.g_delta
        # min_g_delta = 0.3
        # max_g_delta_val = 1.0
        # min_bar = 0.0
        # max_bar = 1.0
        # clamped_g_delta = min(max(g_delta, min_g_delta), max_g_delta_val)
        # bar_position = min_bar + (clamped_g_delta - min_g_delta) * (
        #     max_bar - min_bar
        # ) / (max_g_delta_val - min_g_delta)
        touchdown_distance = st.session_state.touchdown_distance
        min_touchdown_distance = 1500
        max_touchdown_distance_val = 4000
        min_bar = 0.0
        max_bar = 1.0
        clamped_touchdown_distance = min(
            max(touchdown_distance, min_touchdown_distance), max_touchdown_distance_val
        )
        bar_position = min_bar + (
            clamped_touchdown_distance - min_touchdown_distance
        ) * (max_bar - min_bar) / (max_touchdown_distance_val - min_touchdown_distance)

        # Create theme for the plot
        theme = {
            "backgroundColor": "#FFFFFF",
            "textColor": "#000000",
            "base": "light",
        }

        with st.expander("Propagation Potential", expanded=False):
            st.write("""
            Propagation potential measures how likely a crack is to propagate through the weak layer once initiated. 
            It's based on the differential energy release rate (ERR) - the amount of energy available to drive crack growth.
            
            **Interpretation:**
            - **High bar position (red zone)**: High propagation potential - crack likely to spread rapidly
            - **Medium bar position (yellow zone)**: Moderate propagation potential - crack may propagate under certain conditions  
            - **Low bar position (green zone)**: Low propagation potential - crack growth is unlikely
            
            This is calculated from the mechanical properties of the slab and weak layer, considering the energy balance during crack propagation.
            """)

        propagation_potential_fig = plot_traffic_light(bar_position, theme)
        st.plotly_chart(
            propagation_potential_fig,
            use_container_width=True,
            key="propagation_potential_fig",
        )

    if hasattr(st.session_state, "coupled_critical"):
        ratio_weights = st.session_state.coupled_critical / NORMAL_SKIER_WEIGHT
        if ratio_weights >= 3.0:  # 1/0.7
            st.success("✅ Well below critical threshold")
        elif ratio_weights >= 2.0:  # 1/0.9
            st.warning("⚠️ Approaching critical threshold")
        else:
            st.error("❌ Above critical threshold")

    col1, col2 = st.columns([1, 1])
    with col1:
        # Additional risk information
        st.write("**Assessment Summary:**")
        st.write(f"- Slope Angle: {slope_angle}°")
        st.write(f"- Slab Layers: {len(st.session_state.slab_layers)}")
        st.write(
            f"- Weak Layer: {st.session_state.get('weak_layer_radio', 'Not selected')}"
        )

    with col2:
        # Show critical weights if calculated
        if hasattr(st.session_state, "min_force_critical") and hasattr(
            st.session_state, "coupled_critical"
        ):
            st.write("**Analysis Results:**")
            st.write(
                f"- Min Force Critical Weight: {st.session_state.min_force_critical:.1f} kg"
            )
            st.write(
                f"- Coupled Criterion Critical Weight: {st.session_state.coupled_critical:.1f} kg"
            )
            st.write(
                f"- Overall Critical Weight: {st.session_state.coupled_critical:.1f} kg"
            )
            st.write(f"Steady State ERR: {st.session_state.g_delta:.2f}")
            st.write(
                f"Touchdown Distance: {system.slab_touchdown.touchdown_distance:.2f} m"
            )

    # Footer
    st.divider()
    st.markdown("*Avalanche Risk Assessment Tool - For Educational Purposes*")
