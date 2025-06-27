import sys
import streamlit as st

sys.path.append("/home/ubuntu/Documents/weac")

from weac_2.analysis.plotter import Plotter


st.set_page_config(
    page_title="WEAC Streamlit App",
    page_icon="👋",
)

if "plotter" not in st.session_state:
    st.session_state.plotter = Plotter()

st.title("Welcome to the WEAC Streamlit App! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This app allows you to perform snow slab analysis using the WEAC codebase.
    
    **👈 Select a page from the sidebar** to get started.
    
    ### Pages:
    - **Slab Definition**: Define the properties of the slab and weak layer.
    - **Scenario and Analysis**: Define a scenario (e.g., skier load) and run the analysis.
    """
)
