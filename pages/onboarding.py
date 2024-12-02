
import streamlit as st

def run():
    st.title("Onboarding")
    st.markdown("Welcome to the onboarding process!")
    if st.button("Go to Home Page"):
        st.session_state["page"] = "Home Page"
