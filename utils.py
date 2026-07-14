import joblib
import pandas as pd
from config import MODEL_PATH

model = joblib.load(MODEL_PATH)

def predict(df: pd.DataFrame):
    prediction = model.predict(df)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df)[0][1]

    return prediction, probability


def prediction_label(prediction):
    return "Diabetic" if prediction == 1 else "Not Diabetic"


def risk_level(probability):
    if probability is None:
        return "Unknown"
    elif probability >= 0.75:
        return "High"
    elif probability >= 0.40:
        return "Moderate"
    else:
        return "Low"