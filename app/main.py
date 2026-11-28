import logging
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# JSON logging configuration
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_data)


logger = logging.getLogger("fare_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


# Load the champion model
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)


# Create FastAPI application
app = FastAPI(title="MP165 Auto-Rickshaw Fare Prediction API")


# Request schema
class FareRequest(BaseModel):
    distance_km: float = Field(ge=0)
    wait_min: float
    passengers: int
    hour: int
    luggage_bags: int
    fuel_price_rs: float = Field(le=150)
    booking: str
    zone: str


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Fare prediction
@app.post("/predict")
def predict_fare(request: FareRequest):

    input_data = pd.DataFrame([request.model_dump()])

    prediction = model.predict(input_data)[0]

    logger.info("Fare prediction generated successfully")

    return {
        "predicted_fare_rs": round(float(prediction), 2)
    }