import streamlit as st
import pandas as pd
from datetime import datetime

from streamlit_option_menu import option_menu

from config import *
from utils import *

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT
)

# ---------------- LOAD CSS ----------------

try:
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except:
    pass

# ---------------- SESSION ----------------

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

# ---------------- MENU ----------------

selected = option_menu(
    menu_title=None,
    options=[
        "Home",
        "Predict",
        "History",
        "About"
    ],
    icons=[
        "house",
        "activity",
        "clock-history",
        "info-circle"
    ],
    orientation="horizontal",
)

# ==========================================================
# HOME
# ==========================================================

if selected == "Home":

    st.title("🩺 Diabetes Prediction Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Model", "Random Forest")
    c2.metric("Features", "8")
    c3.metric("Status", "Ready")
    c4.metric("Framework", "Streamlit")

    st.markdown("---")

    st.subheader("About Project")

    st.write("""
This application predicts whether a patient is diabetic
using a Machine Learning Random Forest model trained
on the PIMA Indians Diabetes Dataset.
""")

    a, b, c = st.columns(3)

    a.metric("Algorithm", "Random Forest")
    b.metric("Dataset", "PIMA")
    c.metric("Inputs", "8")

# ==========================================================
# PREDICT
# ==========================================================

elif selected == "Predict":

    st.title("📝 Manual Prediction")

    with st.form("predict_form"):

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Pregnancies",
                0,
                20,
                0
            )

            glucose = st.number_input(
                "Glucose",
                50,
                300,
                120
            )

            blood_pressure = st.number_input(
                "Blood Pressure",
                30,
                180,
                80
            )

            skin = st.number_input(
                "Skin Thickness",
                0,
                100,
                20
            )

        with col2:

            insulin = st.number_input(
                "Insulin",
                0,
                900,
                80
            )

            bmi = st.number_input(
                "BMI",
                10.0,
                70.0,
                25.0
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                0.0,
                2.5,
                0.47
            )

            age = st.number_input(
                "Age",
                1,
                100,
                30
            )

        submit = st.form_submit_button("Predict")

    if submit:

        patient = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin,
                insulin,
                bmi,
                dpf,
                age
            ]],
            columns=FEATURES
        )

        prediction, probability = predict(patient)

        label = prediction_label(prediction)

        risk = risk_level(probability)

        st.markdown("---")

        if prediction == 1:
            st.error("⚠ Patient is likely Diabetic")
        else:
            st.success("✅ Patient is Not Diabetic")

        if probability is not None:

            st.metric(
                "Confidence",
                f"{probability*100:.2f}%"
            )

            st.progress(float(probability))

        st.write(f"### Risk Level : {risk}")

        history = patient.copy()

        history["Prediction"] = label

        history["Confidence"] = (
            round(probability * 100, 2)
            if probability is not None
            else None
        )

        history["Timestamp"] = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        st.session_state.history = pd.concat(
            [
                st.session_state.history,
                history
            ],
            ignore_index=True
        )

# ==========================================================
# HISTORY
# ==========================================================

elif selected == "History":

    st.title("📊 Prediction History")

    if st.session_state.history.empty:

        st.info("No predictions available.")

    else:

        st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

        st.subheader("Prediction Count")

        st.bar_chart(
            st.session_state.history[
                "Prediction"
            ].value_counts()
        )

# ==========================================================
# ABOUT
# ==========================================================

elif selected == "About":

    st.title("ℹ About")

    st.markdown("""

### Diabetes Prediction Dashboard

Predict whether a patient is diabetic using Machine Learning.

### Technology Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

### Machine Learning Model

Random Forest Classifier

### Dataset

PIMA Indians Diabetes Dataset

### Developed By

Prabhakar Kumar Yadav

""")