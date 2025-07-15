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
from weac_2.utils import load_dummy_profile

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
    "Very Weak": {"density": 50, "thickness": 30},
    "Weak": {"density": 75, "thickness": 30},
    "Less Weak": {"density": 150, "thickness": 30},
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
        weak_layer_choice = st.radio(
            "Choose weak layer type:",
            index=0,
            options=list(WEAK_LAYER_TYPES.keys()),
            key="weak_layer_radio",
        )

        weak_props = WEAK_LAYER_TYPES[weak_layer_choice]
        st.session_state.selected_weak_layer = WeakLayer(
            rho=weak_props["density"], h=weak_props["thickness"]
        )
        st.write(
            f"Selected: {weak_layer_choice} (ρ={weak_props['density']} kg/m³, h={weak_props['thickness']}mm)"
        )

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

    with col1:
        st.subheader("Scenario Parameters")

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

        # Skier weight slider
        skier_weight = st.slider(
            "Skier Weight (kg)",
            min_value=0,
            max_value=300,
            value=st.session_state.get("skier_weight", 80),
            step=5,
            help="Weight of the skier in kilograms",
            key="skier_weight_slider",
        )
        st.session_state.skier_weight = skier_weight

        st.write("**Current Settings:**")
        st.write(f"- Slope Angle: {slope_angle}°")
        st.write(f"- Skier Weight: {skier_weight} kg")

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
                weight=skier_weight,
                title="Slab Visualization",
            )
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Risk Level")

    # Calculate actual risk using system analysis
    if st.session_state.slab_layers and st.session_state.selected_weak_layer:
        # Get current parameters from session state or defaults
        slope_angle = st.session_state.get("slope_angle", 30)
        skier_weight = st.session_state.get("skier_weight", 80)

        try:
            # Build the system model
            layers = [
                layer_info["layer"] for layer_info in st.session_state.slab_layers
            ]
            weak_layer = st.session_state.selected_weak_layer
            print("weak_layer", weak_layer)

            # Create a simple scenario with one skier
            segments = [
                Segment(length=18000, has_foundation=True, m=0),
                Segment(length=0, has_foundation=False, m=skier_weight),
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

            coupled_result = criteria_evaluator.evaluate_coupled_criterion(
                deepcopy(system)
            )

            # Determine risk level based on analysis
            coupled_critical = coupled_result.critical_skier_weight
            min_force_critical = coupled_result.initial_critical_skier_weight

            # Use the lower of the two critical weights as the threshold
            critical_weight = min(min_force_critical, coupled_critical)

            # Extract touchdown distance
            if system.slab_touchdown is not None:
                l_BC = system.slab_touchdown.l_BC
                l_AB = system.slab_touchdown.l_AB
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

            # Store results for display
            st.session_state.min_force_critical = min_force_critical
            st.session_state.coupled_critical = coupled_critical
            st.session_state.critical_weight = critical_weight

            if skier_weight < critical_weight * 0.7:
                risk_level = "LOW"
                color = "🟢"
            elif skier_weight < critical_weight * 0.9:
                risk_level = "MODERATE"
                color = "🟡"
            else:
                risk_level = "HIGH"
                color = "🔴"

        except Exception as e:
            # Fallback to dummy logic if calculation fails
            st.error(f"Calculation error: {str(e)}")
            if slope_angle < 15 and skier_weight < 60:
                risk_level = "LOW"
                color = "🟢"
            elif slope_angle < 30 and skier_weight < 100:
                risk_level = "MODERATE"
                color = "🟡"
            else:
                risk_level = "HIGH"
                color = "🔴"
    # else:
    #     # Fallback logic
    #     slope_angle = st.session_state.get("slope_angle", 30)
    #     skier_weight = st.session_state.get("skier_weight", 80)

    #     if slope_angle < 15 and skier_weight < 60:
    #         risk_level = "LOW"
    #         color = "🟢"
    #     elif slope_angle < 30 and skier_weight < 100:
    #         risk_level = "MODERATE"
    #         color = "🟡"
    #     else:
    #         risk_level = "HIGH"
    #         color = "🔴"

    # # Display traffic light
    # st.markdown(
    #     f"<div style='text-align: center; font-size: 100px;'>{color}</div>",
    #     unsafe_allow_html=True,
    # )
    # st.markdown(
    #     f"<div style='text-align: center; font-size: 30px; font-weight: bold;'>{risk_level} RISK</div>",
    #     unsafe_allow_html=True,
    # )

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

        st.subheader("Impact Resistance")
        impact_resistance_fig = plot_traffic_light(bar_position, theme)
        st.plotly_chart(
            impact_resistance_fig,
            use_container_width=True,
            key="impact_resistance_fig",
        )

    # Fracture resistance visualization
    if (
        hasattr(st.session_state, "critical_weight")
        and st.session_state.critical_weight is not None
    ):
        safety_factor = st.session_state.critical_weight / 100

        min_safety_factor = 0.1
        max_safety_factor_val = 5.0
        min_bar = 0.0
        max_bar = 1.0
        clamped_safety_factor = min(
            max(safety_factor, min_safety_factor), max_safety_factor_val
        )
        bar_position = max_bar - (clamped_safety_factor - min_safety_factor) * (
            max_bar - min_bar
        ) / (max_safety_factor_val - min_safety_factor)

        # Create theme for the plot
        theme = {
            "backgroundColor": "#FFFFFF",
            "textColor": "#000000",
            "base": "light",
        }

        st.subheader("Fracture Resistance")
        fracture_resistance_fig = plot_traffic_light(bar_position, theme)
        st.plotly_chart(
            fracture_resistance_fig,
            use_container_width=True,
            key="fracture_resistance_fig",
        )

    # Propagation potential visualization
    if hasattr(st.session_state, "g_delta") and st.session_state.g_delta is not None:
        g_delta = st.session_state.g_delta
        min_g_delta = 0.3
        max_g_delta_val = 1.0
        min_bar = 0.0
        max_bar = 1.0
        clamped_g_delta = min(max(g_delta, min_g_delta), max_g_delta_val)
        bar_position = min_bar + (clamped_g_delta - min_g_delta) * (
            max_bar - min_bar
        ) / (max_g_delta_val - min_g_delta)

        # Create theme for the plot
        theme = {
            "backgroundColor": "#FFFFFF",
            "textColor": "#000000",
            "base": "light",
        }

        st.subheader("Propagation Potential")
        propagation_potential_fig = plot_traffic_light(bar_position, theme)
        st.plotly_chart(
            propagation_potential_fig,
            use_container_width=True,
            key="propagation_potential_fig",
        )

    # Additional risk information
    st.write("**Assessment Summary:**")
    st.write(f"- Slope Angle: {slope_angle}°")
    st.write(f"- Skier Weight: {skier_weight} kg")
    st.write(f"- Slab Layers: {len(st.session_state.slab_layers)}")
    st.write(
        f"- Weak Layer: {st.session_state.get('weak_layer_radio', 'Not selected')}"
    )

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
            f"- Overall Critical Weight: {st.session_state.critical_weight:.1f} kg"
        )

        safety_factor = (
            st.session_state.critical_weight / skier_weight
            if skier_weight > 0
            else float("inf")
        )
        st.write(f"- Safety Factor: {safety_factor:.2f}")

        if safety_factor >= 1.43:  # 1/0.7
            st.success("✅ Well below critical threshold")
        elif safety_factor >= 1.11:  # 1/0.9
            st.warning("⚠️ Approaching critical threshold")
        else:
            st.error("❌ Above critical threshold")

    # Footer
    st.divider()
    st.markdown("*Avalanche Risk Assessment Tool - For Educational Purposes*")
