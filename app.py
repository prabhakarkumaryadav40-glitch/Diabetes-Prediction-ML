import streamlit as st
import pandas as pd
from datetime import datetime

from streamlit_option_menu import option_menu

from config import *
from utils import *

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Home", "Predict", "Batch", "History", "About"],
    icons=["house", "activity", "upload", "clock-history", "info-circle"],
    orientation="horizontal"
)

if selected == "Home":

    st.title("🩺 Diabetes Prediction Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Model", "Random Forest")
    c2.metric("Features", "8")
    c3.metric("Status", "Ready")
    c4.metric("Framework", "Streamlit")

    st.info("Select **Predict** from the menu to make a diabetes prediction.")
    
    st.markdown("---")

st.subheader("About the Project")

st.write("""
This application predicts whether a patient is diabetic using a
Random Forest Machine Learning model trained on the
PIMA Indians Diabetes Dataset.
""")

c1, c2, c3 = st.columns(3)

c1.metric("Algorithm", "Random Forest")
c2.metric("Dataset", "PIMA")
c3.metric("Inputs", "8 Features")
    
    elif selected == "Predict":

    st.title("📝 Manual Prediction")

    with st.form("predict_form"):

        c1, c2 = st.columns(2)

        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 0)
            glucose = st.number_input("Glucose", 50, 300, 120)
            blood_pressure = st.number_input("Blood Pressure", 30, 180, 80)
            skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

        with c2:
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
            age = st.number_input("Age", 1, 100, 30)

        submit = st.form_submit_button("🔍 Predict")
        
            if submit:

        patient = pd.DataFrame([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]], columns=FEATURES)

        prediction, probability = predict(patient)

        label = prediction_label(prediction)
        risk = risk_level(probability)

        if prediction == 1:
            st.error(f"⚠️ {label}")
        else:
            st.success(f"✅ {label}")

        if probability is not None:
            st.metric("Confidence", f"{probability*100:.2f}%")
            st.progress(float(probability))

        st.write(f"**Risk Level:** {risk}")

        history = patient.copy()
        history["Prediction"] = label
        history["Confidence"] = (
            round(probability * 100, 2)
            if probability is not None else None
        )
        history["Timestamp"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        if "history" not in st.session_state:
            st.session_state.history = pd.DataFrame()

        st.session_state.history = pd.concat(
            [st.session_state.history, history],
            ignore_index=True
        )
        
        elif selected == "History":

    st.title("📊 Prediction History")

    if "history" in st.session_state and not st.session_state.history.empty:

        st.dataframe(st.session_state.history, use_container_width=True)

        st.subheader("Prediction Distribution")

        st.bar_chart(
            st.session_state.history["Prediction"].value_counts()
        )

    else:

        st.info("No prediction history available.")
        
        elif selected == "About":

    st.title("ℹ️ About")

    st.markdown("""
### Diabetes Prediction Dashboard

This project predicts whether a patient is likely to have diabetes
using a Machine Learning model trained on the PIMA Indians Diabetes Dataset.

### Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- Joblib

### Model

Random Forest Classifier

### Developer

Prabhakar Kumar Yadav
""")