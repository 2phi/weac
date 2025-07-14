import sys
sys.path.append("/home/pillowbeast/Documents/weac")

from typing import List, Literal, cast, Tuple, Optional, Dict, Any, Union
import random
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.figure import Figure
import scipy.interpolate
from scipy.optimize import brentq
from copy import deepcopy

from weac_2.components import (
    Layer,
    WeakLayer,
    Segment,
    CriteriaConfig,
    ModelInput,
    ScenarioConfig,
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
    "Snow Type 1": {"density": 150, "default_thickness": 100},
    "Snow Type 2": {"density": 200, "default_thickness": 100},
    "Snow Type 3": {"density": 250, "default_thickness": 100},
    "Snow Type 4": {"density": 300, "default_thickness": 100},
}

# Predefined weak layer types
WEAK_LAYER_TYPES = {
    "Very Weak": {"density": 50, "thickness": 30},
    "Weak": {"density": 75, "thickness": 30},
    "Less Weak": {"density": 100, "thickness": 30},
}

st.set_page_config(page_title="Avalanche Risk Assessment", layout="wide")

# Create centered layout (80% width)
_, main_col, _ = st.columns([1, 8, 1])

with main_col:
    # Main title
    st.title("🏔️ Avalanche Risk Assessment Tool")

    # STAGE 1: Slab Assembly
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Build Your Slab")
        
        # Slab layers section
        st.write("**Add Slab Layers:**")
        slab_cols = st.columns([3, 1])
        
        with slab_cols[0]:
            for i, (slab_type, properties) in enumerate(SLAB_TYPES.items()):
                cols = st.columns([2, 1])
                with cols[0]:
                    st.write(f"{slab_type} (ρ={properties['density']} kg/m³)")
                with cols[1]:
                    if st.button("Add", key=f"add_slab_{i}"):
                        new_layer = Layer(
                            rho=properties["density"], 
                            h=properties["default_thickness"]
                        )
                        st.session_state.slab_layers.append({
                            "type": slab_type,
                            "layer": new_layer,
                            "thickness": properties["default_thickness"]
                        })
                        st.rerun()
        
        # Display current slab layers
        if st.session_state.slab_layers:
            st.write("**Current Slab Layers:**")
            for i, layer_info in enumerate(st.session_state.slab_layers):
                cols = st.columns([2, 1, 1])
                with cols[0]:
                    st.write(f"{layer_info['type']}")
                with cols[1]:
                    # Allow thickness adjustment
                    new_thickness = st.number_input(
                        "Height (mm)",
                        min_value=10.0,
                        max_value=500.0,
                        value=float(layer_info['thickness']),
                        step=10.0,
                        key=f"thickness_{i}"
                    )
                    if new_thickness != layer_info['thickness']:
                        st.session_state.slab_layers[i]['thickness'] = new_thickness
                        st.session_state.slab_layers[i]['layer'].h = new_thickness
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
            options=list(WEAK_LAYER_TYPES.keys()),
            key="weak_layer_radio"
        )
        
        if weak_layer_choice:
            weak_props = WEAK_LAYER_TYPES[weak_layer_choice]
            st.session_state.selected_weak_layer = WeakLayer(
                rho=weak_props["density"],
                h=weak_props["thickness"]
            )
            st.write(f"Selected: {weak_layer_choice} (ρ={weak_props['density']} kg/m³, h={weak_props['thickness']}mm)")

    with col2:
        st.subheader("Slab Profile")
        
        # Create and display slab profile
        if st.session_state.slab_layers and st.session_state.selected_weak_layer:
            layers = [layer_info['layer'] for layer_info in st.session_state.slab_layers]
            slab = Slab(layers=layers)
            weak_layer = st.session_state.selected_weak_layer
            
            fig = st.session_state.plotter.plot_slab_profile(
                weak_layers=weak_layer, 
                slabs=slab
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
            key="slope_angle_slider"
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
            key="skier_weight_slider"
        )
        st.session_state.skier_weight = skier_weight
        
        st.write(f"**Current Settings:**")
        st.write(f"- Slope Angle: {slope_angle}°")
        st.write(f"- Skier Weight: {skier_weight} kg")

    with col2:
        st.subheader("Slab Visualization")
        
        # Create rotated slab visualization
        if st.session_state.slab_layers and st.session_state.selected_weak_layer:
            # For now, show the same slab profile plot
            # TODO: Implement rotation visualization
            layers = [layer_info['layer'] for layer_info in st.session_state.slab_layers]
            slab = Slab(layers=layers)
            weak_layer = st.session_state.selected_weak_layer
            
            fig = st.session_state.plotter.plot_slab_profile(
                weak_layers=weak_layer, 
                slabs=slab
            )
            st.pyplot(fig)
            plt.close(fig)
            
            st.write(f"Slope angle: {slope_angle}° (rotation visualization coming soon)")

    # STAGE 3: Risk Assessment
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Assessment Details")
        
        # Information panels with question marks
        with st.expander("ℹ️ What does this assessment mean?"):
            st.write("""This is dummy explanatory text about the assessment methodology. 
                        The traffic light system indicates the risk level based on various factors 
                        including slope angle, skier weight, and slab properties.""")
        
        with st.expander("ℹ️ How is the risk calculated?"):
            st.write("""This is dummy text explaining the calculation methodology. 
                        The system analyzes stress distribution, energy release rates, 
                        and other mechanical properties to determine avalanche risk.""")
        
        with st.expander("ℹ️ What should I do with these results?"):
            st.write("""This is dummy text providing recommendations based on the risk level. 
                        Green means low risk, yellow means caution advised, 
                        red means high risk - avoid the slope.""")

    with col2:
        st.subheader("Risk Level")
        
        # Calculate actual risk using system analysis
        if st.session_state.slab_layers and st.session_state.selected_weak_layer:
            # Get current parameters from session state or defaults
            slope_angle = st.session_state.get("slope_angle", 30)
            skier_weight = st.session_state.get("skier_weight", 80)
            
            try:
                # Build the system model
                layers = [layer_info['layer'] for layer_info in st.session_state.slab_layers]
                weak_layer = st.session_state.selected_weak_layer
                
                # Create a simple scenario with one skier
                segments = [
                    Segment(length=10000.0, has_foundation=True, m=0),  # Left boundary
                    Segment(length=1000.0, has_foundation=True, m=skier_weight),  # Middle with skier
                    Segment(length=10000.0, has_foundation=True, m=0),  # Right boundary
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
                    criteria_config=CriteriaConfig(),
                )
                
                system = SystemModel(model_input)
                criteria_evaluator = CriteriaEvaluator(CriteriaConfig())
                
                # Calculate minimum force and coupled criterion
                min_force_result = criteria_evaluator.find_minimum_force(deepcopy(system))
                coupled_result = criteria_evaluator.evaluate_coupled_criterion(deepcopy(system))
                
                # Determine risk level based on analysis
                min_force_critical = min_force_result.critical_skier_weight
                coupled_critical = coupled_result.critical_skier_weight
                
                # Use the lower of the two critical weights as the threshold
                critical_weight = min(min_force_critical, coupled_critical)
                
                if skier_weight < critical_weight * 0.7:
                    risk_level = "LOW"
                    color = "🟢"
                elif skier_weight < critical_weight * 0.9:
                    risk_level = "MODERATE"
                    color = "🟡"
                else:
                    risk_level = "HIGH"
                    color = "🔴"
                
                # Store results for display
                st.session_state.min_force_critical = min_force_critical
                st.session_state.coupled_critical = coupled_critical
                st.session_state.critical_weight = critical_weight
                
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
        else:
            # Fallback logic
            slope_angle = st.session_state.get("slope_angle", 30)
            skier_weight = st.session_state.get("skier_weight", 80)
            
            if slope_angle < 15 and skier_weight < 60:
                risk_level = "LOW"
                color = "🟢"
            elif slope_angle < 30 and skier_weight < 100:
                risk_level = "MODERATE"
                color = "🟡"
            else:
                risk_level = "HIGH"
                color = "🔴"
        
        # Display traffic light
        st.markdown(f"<div style='text-align: center; font-size: 100px;'>{color}</div>", 
                   unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 30px; font-weight: bold;'>{risk_level} RISK</div>", 
                   unsafe_allow_html=True)
        
        # Additional risk information
        st.write(f"**Assessment Summary:**")
        st.write(f"- Slope Angle: {slope_angle}°")
        st.write(f"- Skier Weight: {skier_weight} kg")
        st.write(f"- Slab Layers: {len(st.session_state.slab_layers)}")
        st.write(f"- Weak Layer: {st.session_state.get('weak_layer_radio', 'Not selected')}")
        
        # Show critical weights if calculated
        if hasattr(st.session_state, 'min_force_critical') and hasattr(st.session_state, 'coupled_critical'):
            st.write(f"**Analysis Results:**")
            st.write(f"- Min Force Critical Weight: {st.session_state.min_force_critical:.1f} kg")
            st.write(f"- Coupled Criterion Critical Weight: {st.session_state.coupled_critical:.1f} kg")
            st.write(f"- Overall Critical Weight: {st.session_state.critical_weight:.1f} kg")
            
            safety_factor = st.session_state.critical_weight / skier_weight if skier_weight > 0 else float('inf')
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
