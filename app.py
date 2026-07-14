import plotly.express as px
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
    "Batch",
    "History",
    "About"
],
   icons=[
    "house",
    "activity",
    "upload",
    "clock-history",
    "info-circle"
],
    orientation="horizontal",
)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.title("🩺 Diabetes Prediction")

    st.caption("ML Dashboard")

    st.markdown("---")

    st.success("🟢 System Ready")

    st.markdown("### Quick Tips")

    st.info("""
- Use **Predict** for one patient.
- Use **Batch** for CSV files.
- View previous results in **History**.
""")

    st.markdown("---")

    st.caption("Version 1.0")

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

    st.subheader("📌 Project Overview")

    st.write("""
This web application predicts whether a patient is likely to have diabetes
using a Machine Learning model trained on the PIMA Indians Diabetes Dataset.

### Features

✅ Manual Prediction

✅ Batch CSV Prediction

✅ Confidence Score

✅ Prediction History

✅ CSV Download

✅ Streamlit Cloud Deployment
""")

    st.info(
        "👈 Use the navigation menu above to explore all features."
    )

    st.markdown("---")

    st.subheader("📈 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", "78%")
    c2.metric("Precision", "74%")
    c3.metric("Recall", "68%")

    st.caption(
        "Performance measured on the test dataset."
    )

    st.markdown("---")

    st.subheader("📊 Feature Importance")

    importance = get_feature_importance()

    if importance is not None:

        st.bar_chart(
            importance.set_index("Feature")
        )
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

        submit = st.form_submit_button("🔍 Predict")

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
        st.subheader("🧾 Prediction Summary")

        if prediction == 1:
            st.error("⚠️ Patient is likely Diabetic")
        else:
            st.success("✅ Patient is Not Diabetic")

        col1, col2 = st.columns(2)

        col1.metric(
            "Risk Level",
            risk
        )

        if probability is not None:
            col2.metric(
                "Confidence",
                f"{probability*100:.2f}%"
            )

            st.progress(float(probability))

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
# BATCH
# ==========================================================

elif selected == "Batch":

    st.title("📂 Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.dataframe(df, use_container_width=True)

        # Validate required columns
        missing = [col for col in FEATURES if col not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")

        else:

            if st.button("Predict All"):

                # Keep only model input columns
                input_df = df[FEATURES]

                # Make predictions
                predictions, probabilities = predict(input_df)

                # Create output dataframe
                result = df.copy()

                result["Prediction"] = [
                    prediction_label(p)
                    for p in predictions
                ]

                if probabilities is not None:
                    result["Confidence (%)"] = (
                        probabilities * 100
                    ).round(2)

                st.success("✅ Prediction Completed")

                st.dataframe(
                    result,
                    use_container_width=True
                )

                st.download_button(
                    label="⬇ Download Results",
                    data=result.to_csv(index=False),
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )

                # Save history
                if "history" not in st.session_state:
                    st.session_state.history = pd.DataFrame()

                st.session_state.history = pd.concat(
                    [st.session_state.history, result],
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

        st.subheader("Prediction Distribution")

        counts = (
            st.session_state.history["Prediction"]
            .value_counts()
            .reset_index()
        )

        counts.columns = ["Prediction", "Count"]

        fig = px.pie(
            counts,
            names="Prediction",
            values="Count",
            title="Prediction Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
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