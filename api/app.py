import os
import time
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Config: Load model from Production stage by default
MODEL_URI = os.getenv("MODEL_URI", "models:/ticket-router/Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.ticket-mlops.svc.cluster.local:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

# Attempt to get the model version from MLmodel metadata
def get_model_version_from_flavor(model_uri: str) -> Optional[int]:
    try:
        import mlflow.models
        m = mlflow.models.get_model_info(model_uri)
        # When loading from registry, model version may appear in m.model_uri or tags
        # Not guaranteed; in demo we’ll fallback to None if unavailable
        return None
    except Exception:
        return None

loaded_version = get_model_version_from_flavor(MODEL_URI)

app = FastAPI(title="Ticket Router", version="1.0.0")

class Ticket(BaseModel):
    title: str
    description: str = ""

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
def predict(ticket: Ticket):
    t0 = time.time()
    text = f"{ticket.title}. {ticket.description}".strip()
    pred = model.predict([text])[0]
    # For demo: MultinomialNB doesn't give calibrated proba; use max log-proba if available
    try:
        import numpy as np
        # If underlying model supports predict_proba
        # This works if pipeline has clf with predict_proba
        proba = getattr(model._model_impl, "predict_proba", None)
        if proba:
            conf = float(proba([text]).max())
        else:
            conf = 0.8  # fallback
    except Exception:
        conf = 0.8

    latency_ms = int((time.time() - t0) * 1000)
    return {
        "predicted_queue": str(pred),
        "confidence": round(conf, 2),
        "model_version": loaded_version,  # may be None in this minimal demo
        "latency_ms": latency_ms
    }
