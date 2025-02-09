# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

# Define FastAPI app
app = FastAPI()

# Define the input data structure
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float

# Prediction endpoint
@app.post("/predict/")
def predict(request: PredictionRequest):
    features = np.array([[request.feature1, request.feature2]])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
