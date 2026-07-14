import joblib
import pandas as pd
from config import MODEL_PATH, FEATURES

model = joblib.load(MODEL_PATH)


def predict(df: pd.DataFrame):

    predictions = model.predict(df)

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df)[:, 1]

    if len(predictions) == 1:
        return int(predictions[0]), float(probability[0])

    return predictions, probability


def prediction_label(prediction):
    return "Diabetic" if prediction == 1 else "Not Diabetic"


def risk_level(probability):
    if probability >= 0.75:
        return "High"
    elif probability >= 0.40:
        return "Moderate"
    return "Low"

def get_feature_importance():

    clf = model

    # If the model is a Pipeline, get the last step automatically
    if hasattr(model, "named_steps"):
        clf = list(model.named_steps.values())[-1]

    # Return feature importance if available
    if hasattr(clf, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": clf.feature_importances_
        })

        return importance.sort_values(
            by="Importance",
            ascending=False
        )

    return None

