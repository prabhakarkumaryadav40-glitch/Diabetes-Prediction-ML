import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("model/best_diabetes_model.pkl")

model = load_model()

required_cols = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

st.sidebar.title("🩺 Diabetes Prediction")
st.sidebar.markdown("---")
st.sidebar.info(
    """
    Enter patient information or upload a CSV file.

    **Model:** Random Forest

    **Framework:** Streamlit
    """
)

st.title("🩺 Diabetes Prediction using Machine Learning")

st.write(
    "Predict whether a patient is likely to have diabetes based on medical parameters."
)

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if all(col in df.columns for col in required_cols):

        st.subheader("Uploaded Dataset")
        st.dataframe(df)

        if st.button("Predict CSV"):

            pred = model.predict(df)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(df)[:,1]
                df["Probability"] = (prob*100).round(2)

            df["Prediction"] = [
                "Diabetic" if x==1 else "Not Diabetic"
                for x in pred
            ]

            st.success("Prediction Complete")

            st.dataframe(df)

            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "prediction.csv"
            )

            st.session_state.history = pd.concat(
                [st.session_state.history,df],
                ignore_index=True
            )

    else:
        st.error("CSV format is incorrect.")

st.divider()

st.subheader("Manual Prediction")

with st.form("prediction"):

    c1,c2,c3,c4 = st.columns(4)

    pregnancies = c1.number_input("Pregnancies",0,20)
    glucose = c2.number_input("Glucose",50,300)
    bp = c3.number_input("Blood Pressure",30,180)
    skin = c4.number_input("Skin Thickness",0,100)

    c5,c6,c7,c8 = st.columns(4)

    insulin = c5.number_input("Insulin",0,900)
    bmi = c6.number_input("BMI",10.0,70.0)
    dpf = c7.number_input("DPF",0.0,2.5)
    age = c8.number_input("Age",1,100)

    submit = st.form_submit_button("Predict")

if submit:

    sample = pd.DataFrame([[
        pregnancies,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]],columns=required_cols)

    prediction = model.predict(sample)[0]

    probability = None

    if hasattr(model,"predict_proba"):
        probability = model.predict_proba(sample)[0][1]

    if prediction==1:
        st.error("⚠️ Patient is likely Diabetic")
    else:
        st.success("✅ Patient is Not Diabetic")

    if probability is not None:
        st.metric(
            "Prediction Confidence",
            f"{probability*100:.2f}%"
        )

    sample["Prediction"] = (
        "Diabetic"
        if prediction==1
        else
        "Not Diabetic"
    )

    st.session_state.history = pd.concat(
        [st.session_state.history,sample],
        ignore_index=True
    )

if not st.session_state.history.empty:

    st.divider()

    st.subheader("Prediction History")

    st.dataframe(st.session_state.history)

    st.subheader("Prediction Distribution")

    st.bar_chart(
        st.session_state.history["Prediction"].value_counts()
    )

st.markdown("---")
st.caption("Developed using Streamlit and Scikit-learn")