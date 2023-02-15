import streamlit as st

st.title("Restart - Page")

for k in st.session_state.keys():
    del st.session_state[k]

st.markdown("ALL Data Wiped Clean. Press Home to go back.")