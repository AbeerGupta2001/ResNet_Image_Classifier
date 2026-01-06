


import streamlit as st

st.set_page_config("Image Classifier",layout="centered",initial_sidebar_state="expanded")

welcome = st.Page("introduction.py",title="Introduction")
app = st.Page("app.py",title="Model")

py = st.navigation([welcome,app])
py.run()