from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle

app = FastAPI()

# Load AdaBoost model
with open("modelv1.pkl", "rb") as f:
    model = pickle.load(f)

# Load threshold
with open("modelv1_threshold.txt", "r") as f:
    best_threshold = float(f.read())

# Load feature order
with open("modelv1_features.pkl", "rb") as f:
    feature_order = pickle.load(f)

# Update this to match modelv1 feature schema
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
    BMI: float
    heartRate: int
    glucose: int
    smoking_intensity: int
    pulse_pressure: int
    chol_age_ratio: float
    bmi_sysbp_interaction: float
    high_glucose: int
    high_chol: int
    obese: int
    
@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[feature_order]  # Ensure correct order

    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= best_threshold)

    return {
        "prediction": prediction,
        "probability": round(prob, 3),
        "threshold": best_threshold
    }

@app.get("/")
def read_root():
    return {"message": "Cardio Risk Prediction API - Model V1 (AdaBoost)"}
