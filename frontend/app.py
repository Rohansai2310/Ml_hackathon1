
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="PathSnap",
    page_icon="🗺️",
    layout="wide"
)

# Main heading
st.title("🗺️ PathSnap")
st.subheader("GPS Trajectory Reconstruction System")

st.markdown("""
Welcome to PathSnap!

Select a trip to visualise its GPS trajectory
and the reconstructed road route.
""")

# Trip selection
st.header("Trip Selection")

trip_id = st.selectbox(
    "Choose a trip",
    ["Select a trip"]
)

st.info("Trip data and reconstructed routes will appear here once the backend API is integrated.")
