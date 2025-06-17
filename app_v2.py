from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load CatBoost model
with open("modelv2.joblib", "rb") as f:
    model = joblib.load(f)

# Load threshold
with open("modelv2_threshold.txt", "r") as f:
    best_threshold = float(f.read())

# Load feature order
with open("modelv2_features.joblib", "rb") as f:
    feature_order = joblib.load(f)

# Define input schema based on feature_order
class InputData(BaseModel):
    age: int
    education: int
    sex: int
    is_smoking: int
    BPMeds: int
    prevalentStroke: int
    prevalentHyp: int
    diabetes: int
    totChol: int
    sysBP: int
    BMI: int
    heartRate: int
    glucose: int
    smoking_intensity: int
    pulse_pressure: int
    chol_age_ratio: int
    bmi_sysbp_interaction: int
    high_glucose: int
    high_chol: int
    obese: int

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[feature_order]  # Ensure correct column order

    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= best_threshold)

    return {
        "prediction": prediction,
        "probability": round(prob, 3),
        "threshold": best_threshold
    }

@app.get("/")
def read_root():
    return {"message": "Cardio Risk Prediction API - Model V2 (CatBoost)"}
