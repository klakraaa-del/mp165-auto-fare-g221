import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
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
app = FastAPI(
    title="MP165 Auto-Rickshaw Fare Prediction API",
    description="Production-grade API for MP165 Jaipur Auto-Rickshaw Fare Prediction (Group G221)",
    version="1.0.0"
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(process_time_ms)
    logger.info(f"{request.method} {request.url.path} completed in {process_time_ms}ms with status {response.status_code}")
    return response


# Root service metadata
@app.get("/")
def root():
    return {
        "service": "MP165 Auto-Rickshaw Fare Prediction API",
        "group": "G221",
        "stage": "Stage 3 - Deployment & Serving",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


# Model metadata and architecture information
@app.get("/model-info")
def model_info():
    model_type = (
        type(model.named_steps["model"]).__name__
        if hasattr(model, "named_steps") and "model" in model.named_steps
        else type(model).__name__
    )
    return {
        "model_name": "mp165-fare-model",
        "model_type": model_type,
        "features": {
            "numeric": ["distance_km", "wait_min", "passengers", "hour", "luggage_bags", "fuel_price_rs"],
            "categorical": ["booking", "zone"]
        },
        "target": "fare_rs",
        "status": "loaded"
    }


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