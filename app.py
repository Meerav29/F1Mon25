import streamlit as st
from model import train_model, predict_upcoming

st.title("F1 Race Predictor")

if "model" not in st.session_state:
    st.session_state["model"] = train_model()

if st.button("Run Prediction"):
    results = predict_upcoming(st.session_state["model"])
    st.dataframe(results)
