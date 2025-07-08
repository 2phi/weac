import sys

sys.path.append("/home/pillowbeast/Documents/weac")

import streamlit as st

st.set_page_config(
    page_title="WEAC",
    page_icon="☃️",
)
pg = st.navigation(
    [
        st.Page("pages/1_Slab_Definition.py", title="Slab Definition"),
        st.Page("pages/2_Scenario_Definition.py", title="Scenario Definition"),
        st.Page("pages/3_Analysis.py", title="Analysis"),
    ],
    # position="hidden",
)

pg.run()