import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load("best_diabetes_model.pkl")

# Page configuration
st.set_page_config(page_title="Diabetic Analysis App", layout="centered")
st.title("🩺 Diabetic Analysis: Predict if a Person is Diabetic")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Prediction"
    ])

# CSV Upload
st.sidebar.header("📂 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload Patient Data (CSV)", type=["csv"])

required_cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if all(col in df.columns for col in required_cols):
        st.write("✅ Uploaded Data:")
        st.dataframe(df)

        if st.sidebar.button("🔍 Predict from File"):
            preds = model.predict(df[required_cols])
            df["Prediction"] = ["Diabetic" if p == 1 else "Not Diabetic" for p in preds]
            st.dataframe(df)
            st.session_state.history = pd.concat([st.session_state.history, df], ignore_index=True)
    else:
        st.error("CSV must contain all 8 required columns.")

# Manual Input
st.subheader("📝 Enter Patient Details")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Glucose", min_value=50, max_value=300)
        bp = st.number_input("Blood Pressure", min_value=30, max_value=180)
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100)
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=100)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                                columns=required_cols)
        prediction = model.predict(input_df)[0]
        label = "Diabetic" if prediction == 1 else "Not Diabetic"
        st.success(f"🎯 Prediction: **{label}**")

        input_df["Prediction"] = label
        st.session_state.history = pd.concat([st.session_state.history, input_df], ignore_index=True)

# Prediction History
st.subheader("📄 Patient Prediction History")
st.dataframe(st.session_state.history)

# Chart
if not st.session_state.history.empty:
    st.subheader("📊 Diabetes Prediction Count")
    chart_data = st.session_state.history["Prediction"].value_counts().reset_index()
    chart_data.columns = ["Diabetes Status", "Count"]
    st.bar_chart(data=chart_data, x="Diabetes Status", y="Count")
